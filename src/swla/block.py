import torch
import torch.nn as nn

from .attention import Attention
from .config import Config


class TransformerBlock(nn.Module):
    """One pre-norm transformer block: attention sublayer + MLP sublayer.

    Each sublayer is normalized first (pre-norm) and its result is added back to
    the input through a shortcut (residual) connection:

        x = x + dropout(attention(norm(x)))     tokens exchange information
        x = x + dropout(mlp(norm(x)))           each token is processed on its own

    Every `global_every_n`-th block uses global attention, all others use local
    sliding-window attention (see attention.py).

    Notation in the comments below: b = batch size, L = num_tokens, D = hidden_size.

    Memory: every tensor in this block except the attention scores inside
    `self.attn` has shape (b, L, D) or (b, L, mlp_ratio * D), so it grows linearly
    with L and is identical in local and global blocks. The difference between the
    two block types comes only from the attention path.
    """

    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        # Every global_every_n-th layer is global. layer_idx counts from 0, so with
        # global_every_n=5 the global layers are layer_idx 4, 9, 14, ... (layers 5, 10, 15).
        is_global = (layer_idx + 1) % config.global_every_n == 0
        # Stored on the block so evaluation/profiling code can label layers directly
        self.is_global = is_global

        # Attention sublayer with its pre-norm.
        # LayerNorm standardizes each token vector over its D values, then applies a
        # learned scale and shift (2 * D parameters).
        self.attn = Attention(config, is_global=is_global)
        self.attn_norm = nn.LayerNorm(config.hidden_size)

        # MLP sublayer with its pre-norm: D -> mlp_ratio * D -> D, applied to each
        # token separately.
        # Parameters: ~2 * mlp_ratio * D^2, the largest share of the block.
        # Memory: the (b, L, mlp_ratio * D) hidden activation is the largest
        # per-token tensor in the block (linear in L).
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.mlp_ratio * config.hidden_size),
            nn.GELU(),
            nn.Linear(config.mlp_ratio * config.hidden_size, config.hidden_size),
        )
        self.mlp_norm = nn.LayerNorm(config.hidden_size)

        # Dropout on each sublayer output before the residual addition ("residual
        # dropout"). One module can serve both sublayers: it has no parameters or
        # state, and each call draws a new random mask. Only active in train mode.
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, L, D), from the embedding or the previous block.
        # Both lines read from the inside out: norm -> sublayer -> dropout -> add.
        # The `x + ...` is the shortcut connection: the sublayer only adds a
        # correction on top of x, and in backpropagation the gradient flows through
        # the `+` directly to earlier blocks (against vanishing gradients).

        # 1. Attention sublayer: each token gathers information from other tokens
        #    (last W tokens in local blocks, all previous tokens in global blocks).
        x = x + self.dropout(self.attn(self.attn_norm(x)))

        # 2. MLP sublayer: each token is transformed on its own, using the context
        #    gathered in step 1.
        x = x + self.dropout(self.mlp(self.mlp_norm(x)))

        # Same shape as the input, (b, L, D), so blocks can be stacked
        return x
