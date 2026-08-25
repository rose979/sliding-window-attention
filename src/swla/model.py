import torch
import torch.nn as nn

from .block import TransformerBlock
from .cache import KVCache
from .config import Config


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            TransformerBlock(config, layer_idx) for layer_idx in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, kv_caches: list[KVCache] | None = None) -> torch.Tensor:
        x = self.embed(input_ids)
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x = block(x, kv_cache=cache)
        x = self.final_norm(x)
        return self.lm_head(x)

    def init_cache(self, batch_size: int, device=None) -> list[KVCache]:
        return [
            KVCache(
                batch_size=batch_size,
                num_heads=self.config.num_heads,
                head_dim=self.config.hidden_size // self.config.num_heads,
                max_seq_len=self.config.max_seq_len,
                window_size=self.config.window_size,
                is_global=block.attn.is_global,
                device=device,
            )
            for block in self.blocks
        ]
