import torch

from swla.block import TransformerBlock
from swla.config import Config

def test_output_shape():
    config = Config(hidden_size=64, num_heads=4)
    block = TransformerBlock(config, layer_idx=0)
    x = torch.randn(2, 16, config.hidden_size)
    out = block(x)
    assert out.shape == x.shape


def test_global_layer_placement():
    config = Config(hidden_size=64, num_heads=4, global_every_n=5)
    blocks = [TransformerBlock(config, layer_idx=i) for i in range(10)]
    global_layers = [b.layer_idx for b in blocks if b.is_global]
    assert global_layers == [4, 9]
    # The block's label must match what its attention module actually does
    assert all(b.is_global == b.attn.is_global for b in blocks)
