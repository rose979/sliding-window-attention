import torch
import torch.nn as nn


def random_batch(
    batch_size: int, seq_len: int, vocab_size: int, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random token batch for measurements: (input_ids, targets), each (b, L).

    Memory and time of a training step depend only on the tensor shapes, not on
    the token values, so random tokens cost exactly as much as real text.
    seq_len + 1 tokens are drawn so that targets can be the inputs shifted by one.
    """
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len + 1), generator=generator)
    # Position i has to predict the token at position i + 1
    return tokens[:, :-1], tokens[:, 1:]


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """One training step: forward -> loss -> backward -> optimizer update.

    This is the unit the experiment measures (RAM and time per training step).
    Returns the loss as a Python float.
    """
    model.train()
    # set_to_none=True frees the gradient tensors instead of filling them with
    # zeros, so they don't occupy memory during the next forward pass
    optimizer.zero_grad(set_to_none=True)
    # Forward pass: the activations (including the attention weights) are kept
    # for the backward pass -> the peak memory of the step is reached here or
    # during backward
    _, loss = model(input_ids, targets)
    # Backward pass: gradients for all parameters
    loss.backward()
    # Parameter update (AdamW keeps two extra tensors per parameter: its moments)
    optimizer.step()
    return loss.item()
