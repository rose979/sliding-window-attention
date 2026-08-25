import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class Attention(nn.Module):
    def __init__(self, config: Config, is_global: bool = False):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.window_size = config.window_size
        self.is_global = is_global

        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x: torch.Tensor, kv_cache=None) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        attn_mask = None if self.is_global else self._sliding_window_mask(seq_len, k.shape[2], x.device)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=attn_mask is None)
        out = out.transpose(1, 2).reshape(batch, seq_len, hidden)
        return self.out_proj(out)

    def _sliding_window_mask(self, q_len: int, k_len: int, device) -> torch.Tensor:
        q_idx = torch.arange(q_len, device=device).unsqueeze(1) + (k_len - q_len)
        k_idx = torch.arange(k_len, device=device).unsqueeze(0)
        causal = k_idx <= q_idx
        within_window = k_idx > (q_idx - self.window_size)
        allowed = causal & within_window
        return torch.zeros_like(allowed, dtype=torch.float32).masked_fill(~allowed, float("-inf"))
