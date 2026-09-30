import torch

from swla.config import Config
from swla.model import Model
from swla.training import random_batch, train_step


def small_config(**overrides) -> Config:
    base = dict(vocab_size=50, hidden_size=64, num_heads=4, num_layers=5, window_size=8)
    base.update(overrides)
    return Config(**base)


def test_random_batch_targets_are_shifted_inputs():
    gen = torch.Generator().manual_seed(0)
    input_ids, targets = random_batch(batch_size=2, seq_len=10, vocab_size=50, generator=gen)
    assert input_ids.shape == targets.shape == (2, 10)
    assert torch.equal(input_ids[:, 1:], targets[:, :-1])


def test_train_step_updates_weights():
    torch.manual_seed(0)
    model = Model(small_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids, targets = random_batch(2, 16, 50)
    before = [p.detach().clone() for p in model.parameters()]

    loss = train_step(model, optimizer, input_ids, targets)

    assert isinstance(loss, float)
    changed = [not torch.equal(b, p) for b, p in zip(before, model.parameters())]
    assert all(changed), "every parameter should receive a gradient and be updated"


def test_overfit_single_batch():
    """Sanity check of the whole chain (model, loss, backward, optimizer): training
    on one fixed batch over and over must drive the loss close to 0. Run for the
    mixed local/global model (every 5th layer global) and the all-global baseline."""
    for global_every_n in (5, 1):
        torch.manual_seed(0)
        model = Model(small_config(global_every_n=global_every_n))
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        input_ids, targets = random_batch(4, 32, 50)

        first = train_step(model, optimizer, input_ids, targets)
        for _ in range(150):
            last = train_step(model, optimizer, input_ids, targets)

        # Start: about ln(50) = 3.9 (uniform guessing); after memorizing: close to 0
        assert first > 3.0, (global_every_n, first)
        assert last < 0.1, (global_every_n, last)
