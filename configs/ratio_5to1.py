from dataclasses import replace

from swla.config import Config

# Sliding-window model: 4 local layers (window 32) + every 5th layer global.
# Model size ("medium", chosen in the feasibility run): must match global_baseline.py,
# so the two configs differ only in the attention setting (checked in tests/test_configs.py).
config = replace(
    Config(),
    hidden_size=512,
    num_layers=10,
    num_heads=8,
    vocab_size=256,
    window_size=32,
    global_every_n=5,
)
