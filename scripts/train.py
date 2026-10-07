"""Run training steps on random tokens and print loss and time per step.

The experiment measures RAM and time per training step, which depend only on the
tensor shapes, so no dataset is needed. Random tokens can't be learned, though:
with a new batch every step the loss stays around ln(vocab_size). Use
--fixed-batch to train on one batch repeatedly (overfit sanity check); the loss
should then drop close to 0.

Examples:
    python scripts/train.py --config configs.global_baseline --seq-len 512 --steps 20
    python scripts/train.py --fixed-batch --steps 200
"""
import argparse
import importlib
import time

import torch

from swla.model import Model
from swla.training import random_batch, train_step


def load_config(config_path: str):
    module = importlib.import_module(config_path)
    return module.config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs.ratio_5to1", help="module path of the experiment config")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--fixed-batch", action="store_true", help="reuse one batch every step (overfit check)")
    args = parser.parse_args()

    config = load_config(args.config)
    # Seed before creating the model, so initial weights and data are reproducible
    torch.manual_seed(args.seed)
    model = Model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    pattern = "".join("G" if block.is_global else "L" for block in model.blocks)
    print(f"config {args.config}: {model.num_parameters():,} parameters, layers {pattern}")
    print(f"batch {args.batch_size} x {args.seq_len} tokens, {args.steps} steps"
          f"{', fixed batch' if args.fixed_batch else ''}")

    batch = random_batch(args.batch_size, args.seq_len, config.vocab_size)
    for step in range(1, args.steps + 1):
        if not args.fixed_batch:
            batch = random_batch(args.batch_size, args.seq_len, config.vocab_size)
        start = time.perf_counter()
        loss = train_step(model, optimizer, *batch)
        step_time = time.perf_counter() - start
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"step {step:5d}  loss {loss:.4f}  {step_time * 1000:.0f} ms/step")


if __name__ == "__main__":
    main()
