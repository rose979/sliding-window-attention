from dataclasses import replace

from swla.config import Config

# Standard attention baseline: every layer global (window_size is not used by global layers).
# Model size ("medium", chosen in the feasibility run): must match ratio_5to1.py,
# so the two configs differ only in the attention setting (checked in tests/test_configs.py).
config = replace(
    Config(),
    hidden_size=512,
    num_layers=10,
    num_heads=8,
    vocab_size=256,
    window_size=Config().max_seq_len,
    global_every_n=1,
)
