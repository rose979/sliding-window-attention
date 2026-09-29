import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class Attention(nn.Module):
    """Causal multi-head self-attention with two computation paths.

    is_global=True:  every token sees all previous tokens (standard causal attention).
                     Scores have shape (b, h, L, L) -> memory O(L^2).
    is_global=False: every token sees the last `window_size` tokens, itself included.
                     Computed chunk-wise: each chunk of W queries only looks at its own
                     and the previous chunk of keys. Scores have shape (b, h, L/W, W, 2W)
                     -> memory O(L * W).

    Notation in the comments below: b = batch size, h = num_heads, L = num_tokens,
    d = head_dim, D = hidden_size = h * d, W = window_size, n = number of chunks.

    Memory: everything outside the score/weight tensors (x, queries, keys, values,
    context vectors) is O(L * D) in both paths. The paths only differ in the size of
    the score-shaped tensors, reduced by a factor of L / 2W in the chunked path.
    In training these matter most: the softmax weights of every layer are kept
    until the backward pass.
    """

    def __init__(self, config: Config, is_global: bool = False, qkv_bias: bool = False):
        super().__init__()
        assert (config.hidden_size % config.num_heads == 0), \
            "hidden_size must be divisible by num_heads"

        self.d_out = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.window_size = config.window_size
        self.is_global = is_global

        # Parameters: 4 * D^2 (+ biases) per layer, independent of L and of the path
        self.W_query = nn.Linear(config.hidden_size, config.hidden_size, bias=qkv_bias)
        self.W_key = nn.Linear(config.hidden_size, config.hidden_size, bias=qkv_bias)
        self.W_value = nn.Linear(config.hidden_size, config.hidden_size, bias=qkv_bias)
        # Linear layer that mixes the outputs of the individual heads
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        b, num_tokens, _ = x.shape
        # Project the input into keys, queries and values: (b, L, D) each.
        # Compute: ~3 * L * D^2 multiply-adds per sequence, linear in L.
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Split the last dimension into the heads:
        # (b, L, D) -> (b, L, h, d)
        # .view() only reinterprets the shape, no data is copied.
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Swap the token and head dimensions:
        #    (b, L, h, d)
        # -> (b, h, L, d)
        # Convention: for tensors with more than two dimensions, matrix multiplication
        # (@) acts on the last two dimensions; the leading ones (b, h) are batch dims.
        # .transpose() is also a view, no copy.
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        if self.is_global:
            context_vec = self._full_attention(queries, keys, values)
        else:
            context_vec = self._chunked_sliding_attention(queries, keys, values)

        # Combine the context vectors of all heads, with D = h * d:
        # (b, h, L, d) -> (b, L, D)
        # .contiguous() copies the data into the new order (O(L * D)), .view() needs that.
        context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)

    # ------------------------------------------------------------------
    # Global path: full (L, L) score matrix
    # ------------------------------------------------------------------
    @staticmethod
    def _full_mask(num_tokens: int, device, window_size: int | None = None) -> torch.Tensor:
        """(L, L) bool mask, True = masked out.

        Causal only if window_size is None, otherwise causal + sliding window.
        Memory: L^2 bytes (1 byte per bool), e.g. 4 MB for L = 2048.
        """
        ones = torch.ones(num_tokens, num_tokens, dtype=torch.bool, device=device)
        mask = torch.triu(ones, diagonal=1)                           # key after the query (future)
        if window_size is not None:
            # Element-wise OR: masked if the key is in the future (mask) or too old (tril),
            # i.e. W or more tokens back
            mask = mask | torch.tril(ones, diagonal=-window_size)
        return mask

    def _full_attention(self, queries, keys, values, window_size: int | None = None):
        """Steps 1-5 of standard attention on (b, h, L, d) tensors.

        window_size=None is the global path. Passing a window_size gives the
        sliding window via masking only; it is used in tests as the reference
        for the chunked path (same result, but still O(L^2) memory).
        """
        num_tokens = queries.shape[2]
        mask = self._full_mask(num_tokens, queries.device, window_size)

        #   1. Attention scores
        #      Matrix multiplication of queries and keys: (b, h, L, L)
        #      Memory: b * h * L^2 floats (4 bytes each) -> the O(L^2) part.
        #      Compute: b * h * L^2 * d multiply-adds.
        attn_scores = queries @ keys.transpose(2, 3)

        #   2. Fill the masked positions of the scores with -inf
        #      (in place, so no second (L, L) tensor is created)
        attn_scores.masked_fill_(mask, -torch.inf)

        #   3. Scale the scores and normalize them with softmax.
        #      The division creates a temporary (L, L) tensor; the softmax output
        #      is another one and is kept in memory until the backward pass.
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        #   4. Dropout on the attention weights (another (L, L) tensor if dropout > 0)
        attn_weights = self.dropout(attn_weights)

        #   5. Context vectors: (b, h, L, d)
        #      Compute: b * h * L^2 * d multiply-adds, same as step 1.
        return attn_weights @ values

    # ------------------------------------------------------------------
    # Sliding-window path: chunked, (L/W, W, 2W) scores
    # ------------------------------------------------------------------
    @staticmethod
    def _chunk_mask(num_chunks: int, window_size: int, device) -> torch.Tensor:
        """(n, W, 2W) bool mask, True = masked out.

        Within a chunk, query i sits at position W + i of the 2W keys
        [previous chunk | own chunk]. It may see keys i+1 ... W+i, which are
        exactly the last W tokens including itself.
        Memory: n * W * 2W = 2 * L * W bytes, e.g. 128 KB for L = 2048, W = 32.
        """
        W = window_size
        ones = torch.ones(W, 2 * W, dtype=torch.bool, device=device)
        future = torch.triu(ones, diagonal=W + 1)       # key after the query
        too_old = torch.tril(ones, diagonal=0)          # key W or more tokens back
        # .expand() repeats the (W, 2W) mask n times without copying;
        # .clone() makes real copies so chunk 0 can be changed on its own below
        mask = (future | too_old).expand(num_chunks, W, 2 * W).clone()
        # Chunk 0 has no previous chunk; its "previous" keys are zero padding
        mask[0, :, :W] = True
        return mask

    def _chunked_sliding_attention(self, queries, keys, values):
        b, h, num_tokens, d = queries.shape
        W = self.window_size

        # 1. Pad L up to a multiple of W. Padded keys lie in the future of every
        #    real query (masked), padded queries are cut off at the end.
        #    Cost: at most W - 1 extra tokens; skipped if L is a multiple of W.
        pad = (-num_tokens) % W
        if pad:
            queries = F.pad(queries, (0, 0, 0, pad))
            keys = F.pad(keys, (0, 0, 0, pad))
            values = F.pad(values, (0, 0, 0, pad))
        num_chunks = (num_tokens + pad) // W

        # 2. Split the sequence into chunks: (b, h, n, W, d)
        #    .view() only reinterprets the shape, no data is copied.
        queries = queries.view(b, h, num_chunks, W, d)
        keys = keys.view(b, h, num_chunks, W, d)
        values = values.view(b, h, num_chunks, W, d)

        # 3. Keys/values of the previous chunk (a zero chunk in front of chunk 0),
        #    concatenated with the own chunk: (b, h, n, 2W, d)
        #    Cost: keys and values are now stored twice (2 * L * d per head),
        #    linear in L and small compared to the saving in step 4.
        keys_prev = F.pad(keys, (0, 0, 0, 0, 1, 0))[:, :, :-1]
        values_prev = F.pad(values, (0, 0, 0, 0, 1, 0))[:, :, :-1]
        keys = torch.cat([keys_prev, keys], dim=3)
        values = torch.cat([values_prev, values], dim=3)

        # 4. Attention scores per chunk: (b, h, n, W, 2W)
        #    THIS is where the memory is saved:
        #    n * W * 2W = 2 * L * W entries instead of L * L, i.e. a factor L / 2W
        #    (2x for L = 128, 8x for L = 512, 32x for L = 2048 with W = 32).
        #    Compute: b * h * 2 * L * W * d multiply-adds instead of b * h * L^2 * d.
        attn_scores = queries @ keys.transpose(-2, -1)

        # 5. Same mask for every chunk (except chunk 0, see _chunk_mask).
        #    In place, and the (n, W, 2W) mask is broadcast over b and h.
        attn_scores.masked_fill_(self._chunk_mask(num_chunks, W, queries.device), -torch.inf)

        # 6. Softmax, dropout, weighted sum of values: (b, h, n, W, d)
        #    The temporary scaled scores and the softmax weights have the same
        #    reduced size as the scores (2 * L * W per head). The weights are what
        #    stays in memory until the backward pass.
        attn_weights = torch.softmax(attn_scores / d**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values

        # 7. Merge the chunks and drop the padded positions: (b, h, L, d)
        #    Both are views on context_vec, no copy.
        return context_vec.reshape(b, h, num_chunks * W, d)[:, :, :num_tokens]
