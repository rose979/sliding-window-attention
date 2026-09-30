import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import TransformerBlock
from .config import Config


class Model(nn.Module):
    """Decoder-only language model: embedding -> N transformer blocks -> final norm -> lm_head.

    Every `global_every_n`-th block uses global attention, all others local
    sliding-window attention (see block.py / attention.py).

    Notation in the comments below: b = batch size, L = num_tokens, D = hidden_size,
    V = vocab_size, N = num_layers.

    No positional encoding (NoPE), deliberately:
      - The experiment measures RAM and time per training step, not output quality.
        A positional encoding would add the same small, linear-in-L cost to every
        configuration (learned table: max_seq_len * D parameters + one (b, L, D)
        addition; RoPE: a rotation of Q and K per layer). It does not change the
        O(L^2) vs. O(L * W) difference in attention.
      - The model still works without it: with a causal mask, decoder models learn
        position implicitly (Haviv et al. 2022; Kazemnejad et al. 2023).
      - Caveat: in local (sliding-window) layers this implicit signal is weak, since
        every token sees the same number of keys. If losses of the 4:1 model and the
        all-global baseline are ever compared, part of a gap may come from the
        missing positional encoding, not from sliding-window attention itself.
        A learned position embedding (lecture / GPT-2 style) could then be added
        behind a config flag (~5 lines).

    Memory: apart from the attention scores inside the blocks, every activation is
    linear in L. The largest one outside the blocks is the (b, L, V) logits tensor.
    """

    def __init__(self, config: Config):
        super().__init__()
        # Stored on the model; scripts/generate.py reads model.config.max_seq_len
        self.config = config
        # Token embedding: V x D table, one row per vocabulary entry
        # (24.6M parameters with the defaults V = 32000, D = 768).
        # Output (b, L, D), linear in L.
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        # N transformer blocks; each decides itself whether it is global
        # (block.is_global). With global_every_n=5: layers 5, 10, 15, ... are global.
        self.blocks = nn.ModuleList(
            TransformerBlock(config, layer_idx) for layer_idx in range(config.num_layers)
        )
        # Final LayerNorm: needed with pre-norm blocks, since the residual stream x
        # itself is never normalized inside the blocks.
        self.final_norm = nn.LayerNorm(config.hidden_size)
        # Language modeling head: D -> V logits (one next-token score per vocabulary
        # entry). Memory: the (b, L, V) logits are large, e.g. ~262 MB for b=1,
        # L=2048, V=32000 (about as much as one global layer's attention weights with
        # 12 heads), plus same-sized tensors in cross-entropy. Identical in all
        # configurations, but it can hide the attention differences in total RAM.
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying (as in GPT-2): lm_head uses the embedding matrix as its weight.
        # Both are (vocab_size, D) with row v belonging to token v, so the logit for
        # token v becomes the dot product of the hidden state with v's embedding.
        # Saves vocab_size * D parameters (24.6M with the defaults), plus their
        # gradients and optimizer states. Same in all configurations, so it does not
        # affect the comparison.
        self.lm_head.weight = self.embed.weight

        # GPT-2 style weight initialization, replacing the PyTorch defaults set by
        # the layer constructors above. Affects how stably training starts, not the
        # RAM/time measurements (tensor shapes are unchanged).
        # 1. All Linear/Embedding weights ~ N(0, 0.02), biases = 0
        #    (LayerNorm keeps its defaults: gamma = 1, beta = 0).
        self.apply(self._init_weights)
        # 2. Residual projections (the layers whose output is added to x) get a
        #    smaller std, 0.02 / sqrt(2 * N): every block adds two contributions to
        #    the residual stream, and this keeps the variance of x roughly constant
        #    over all 2 * N additions at the start of training.
        residual_std = 0.02 / (2 * config.num_layers) ** 0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp[2].weight, mean=0.0, std=residual_std)   # 2nd MLP Linear

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Rule 1 of the GPT-2 initialization, applied to every submodule."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def num_parameters(self) -> int:
        """Number of trainable parameters. The tied embedding/lm_head matrix counts once,
        since self.parameters() yields a shared tensor only once."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns (logits, loss).

        input_ids: (b, L) token IDs.
        targets:   optional (b, L) token IDs the model should predict, usually
                   input_ids shifted by one position. If given, the mean
                   cross-entropy loss is computed; otherwise loss is None.
        logits:    (b, L, V) next-token scores.
        """
        # (b, L) token IDs -> (b, L, D) start vectors (embedding rows, no position info)
        x = self.embed(input_ids)
        # N blocks, each (b, L, D) -> (b, L, D)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        # (b, L, D) -> (b, L, V)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # cross_entropy expects (num_predictions, V) and (num_predictions,), so the
            # batch and token dimensions are flattened into one: b * L predictions.
            # .reshape instead of .view: targets are usually a slice (tokens[:, 1:]),
            # which is not contiguous, so .view would fail.
            # Memory: softmax probabilities and their gradient are two more
            # (b * L, V) tensors, same size as the logits.
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
