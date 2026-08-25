from dataclasses import dataclass


@dataclass
class Config:
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    max_seq_len: int = 4096
    window_size: int = 512
    global_every_n: int = 5
    dropout: float = 0.0
