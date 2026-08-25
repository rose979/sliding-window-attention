import torch

from swla.attention import Attention
from swla.config import Config


def test_output_shape():
    config = Config(hidden_size=64, num_heads=4)
    attn = Attention(config)
    x = torch.randn(2, 16, config.hidden_size)
    out = attn(x)
    assert out.shape == x.shape


def test_sliding_window_mask_respects_window_size():
    config = Config(hidden_size=64, num_heads=4, window_size=4)
    attn = Attention(config)
    mask = attn._sliding_window_mask(q_len=8, k_len=8, device="cpu")
    allowed = mask == 0
    assert allowed[7, 3] == False
    assert allowed[7, 4] == True
    assert allowed[7, 8 - 1] == True


def test_global_attention_has_no_mask():
    config = Config(hidden_size=64, num_heads=4)
    attn = Attention(config, is_global=True)
    x = torch.randn(2, 16, config.hidden_size)
    out = attn(x)
    assert out.shape == x.shape
