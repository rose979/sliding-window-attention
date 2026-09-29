import torch

from swla.attention import Attention
from swla.config import Config


def test_output_shape_sliding():
    config = Config(hidden_size=64, num_heads=4, window_size=8)
    attn = Attention(config)
    x = torch.randn(2, 20, config.hidden_size)
    out = attn(x)
    assert out.shape == x.shape


def test_output_shape_global():
    config = Config(hidden_size=64, num_heads=4)
    attn = Attention(config, is_global=True)
    x = torch.randn(2, 16, config.hidden_size)
    out = attn(x)
    assert out.shape == x.shape


def test_full_mask_respects_window_size():
    mask = Attention._full_mask(num_tokens=8, device="cpu", window_size=4)
    allowed = ~mask
    assert allowed[7, 3] == False
    assert allowed[7, 4] == True
    assert allowed[7, 7] == True
    assert allowed[3, 4] == False       # causal
    assert allowed.sum(dim=1).tolist() == [1, 2, 3, 4, 4, 4, 4, 4]


def test_chunk_mask_sees_exactly_window_size_keys():
    W = 4
    mask = Attention._chunk_mask(num_chunks=2, window_size=W, device="cpu")
    allowed = ~mask
    # Chunk 1: every query sees W keys, the last one being itself (position W + i)
    assert allowed[1].sum(dim=1).tolist() == [W] * W
    for i in range(W):
        assert allowed[1, i, W + i]
        assert not allowed[1, i, W + i + 1:].any()
    # Chunk 0: nothing in the (padded) previous chunk is visible
    assert not allowed[0, :, :W].any()
    assert allowed[0].sum(dim=1).tolist() == [1, 2, 3, 4]


def test_chunked_matches_full_reference():
    torch.manual_seed(0)
    b, h, d = 2, 4, 16
    for W in (4, 32):
        # L a multiple of W, not a multiple, shorter than W, equal to W
        for L in (2 * W, 2 * W + 6, W - 1, W, 70):
            config = Config(hidden_size=h * d, num_heads=h, window_size=W)
            attn = Attention(config)
            q = torch.randn(b, h, L, d, requires_grad=True)
            k = torch.randn(b, h, L, d, requires_grad=True)
            v = torch.randn(b, h, L, d, requires_grad=True)

            out_chunked = attn._chunked_sliding_attention(q, k, v)
            out_full = attn._full_attention(q, k, v, window_size=W)
            assert out_chunked.shape == (b, h, L, d)
            assert torch.allclose(out_chunked, out_full, atol=1e-5), (W, L)

            # Gradients must match too, since the chunked path is used for training
            grad_out = torch.randn_like(out_full)
            grads_chunked = torch.autograd.grad(out_chunked, (q, k, v), grad_out)
            grads_full = torch.autograd.grad(out_full, (q, k, v), grad_out)
            for g_c, g_f in zip(grads_chunked, grads_full):
                assert torch.allclose(g_c, g_f, atol=1e-5), (W, L)


def test_sliding_with_large_window_equals_global():
    torch.manual_seed(0)
    config = Config(hidden_size=64, num_heads=4, window_size=32)
    sliding = Attention(config, is_global=False)
    global_ = Attention(config, is_global=True)
    global_.load_state_dict(sliding.state_dict())
    x = torch.randn(2, 20, config.hidden_size)
    # window_size >= L: every token sees all previous tokens in both paths
    assert torch.allclose(sliding(x), global_(x), atol=1e-5)
