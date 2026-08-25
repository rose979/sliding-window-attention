import torch


class KVCache:
    """Per-layer KV cache. Sliding-window layers evict tokens beyond window_size;
    global layers keep the full history."""

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        window_size: int,
        is_global: bool,
        device=None,
    ):
        self.is_global = is_global
        capacity = max_seq_len if is_global else window_size
        self.capacity = capacity
        self.k = torch.zeros(batch_size, num_heads, capacity, head_dim, device=device)
        self.v = torch.zeros(batch_size, num_heads, capacity, head_dim, device=device)
        self.length = 0

    def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        new_len = k.shape[2]
        total = self.length + new_len

        if total <= self.capacity:
            self.k[:, :, self.length:total] = k
            self.v[:, :, self.length:total] = v
            self.length = total
            return self.k[:, :, :self.length], self.v[:, :, :self.length]

        overflow = total - self.capacity
        self.k = torch.cat([self.k[:, :, overflow:self.length], k], dim=2)
        self.v = torch.cat([self.v[:, :, overflow:self.length], v], dim=2)
        self.length = self.capacity
        return self.k, self.v
