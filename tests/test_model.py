import torch

from swla.config import Config
from swla.model import Model


def test_forward_output_shape():
    config = Config(vocab_size=100, hidden_size=32, num_heads=4, num_layers=2, max_seq_len=64, window_size=8)
    model = Model(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    logits = model(input_ids)
    assert logits.shape == (2, 16, config.vocab_size)
