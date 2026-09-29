import torch
import torch.nn as nn

from .attention import Attention
from .config import Config

class TransformerBlock(nn.Module):
    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        is_global = (layer_idx + 1) % config.global_every_n == 0
        self.attn = Attention(config, is_global=is_global)
        self.attn_norm = nn.LayerNorm(config.hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.GELU(),
            nn.Linear(4 * config.hidden_size, config.hidden_size),
        )
        self.mlp_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, kv_cache=None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x), kv_cache=kv_cache))
        x = x + self.dropout(self.mlp(self.mlp_norm(x)))
        return x
