import torch

from swla.config import Config
from swla.model import Model


def test_forward_output_shape():
    config = Config(vocab_size=100, hidden_size=32, num_heads=4, num_layers=2, max_seq_len=64, window_size=8)
    model = Model(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(input_ids)
    assert logits.shape == (2, 16, config.vocab_size)
    assert loss is None


def test_forward_with_targets_returns_loss():
    torch.manual_seed(0)
    config = Config(vocab_size=100, hidden_size=32, num_heads=4, num_layers=2, window_size=8)
    model = Model(config)
    tokens = torch.randint(0, config.vocab_size, (2, 17))
    input_ids, targets = tokens[:, :-1], tokens[:, 1:]    # predict the next token
    logits, loss = model(input_ids, targets)
    assert loss.dim() == 0
    # Freshly initialized model guesses roughly uniformly: loss ~ ln(V)
    assert abs(loss.item() - torch.log(torch.tensor(100.0)).item()) < 0.5
    loss.backward()                                       # gradients flow
    assert model.embed.weight.grad is not None


def test_num_parameters_counts_tied_matrix_once():
    config = Config(vocab_size=100, hidden_size=32, num_heads=4, num_layers=2)
    model = Model(config)
    assert model.num_parameters() == sum(p.numel() for p in set(model.parameters()))


def test_weight_tying_shares_one_matrix():
    config = Config(vocab_size=100, hidden_size=32, num_heads=4, num_layers=2)
    model = Model(config)
    assert model.lm_head.weight is model.embed.weight
    # model.parameters() yields a shared tensor only once
    total = sum(p.numel() for p in model.parameters())
    untied = total + config.vocab_size * config.hidden_size
    assert sum(p.numel() for p in set(model.parameters())) == total < untied


def test_gpt2_style_initialization():
    torch.manual_seed(0)
    config = Config(vocab_size=1000, hidden_size=128, num_heads=4, num_layers=8)
    model = Model(config)
    residual_std = 0.02 / (2 * config.num_layers) ** 0.5

    def close(std, expected):
        return abs(std - expected) / expected < 0.1

    assert close(model.embed.weight.std().item(), 0.02)
    for block in model.blocks:
        assert close(block.attn.W_query.weight.std().item(), 0.02)
        assert close(block.mlp[0].weight.std().item(), 0.02)
        assert close(block.attn.out_proj.weight.std().item(), residual_std)
        assert close(block.mlp[2].weight.std().item(), residual_std)
        assert torch.all(block.attn.out_proj.bias == 0)
        assert torch.all(block.mlp[0].bias == 0)
        assert torch.all(block.attn_norm.weight == 1)
