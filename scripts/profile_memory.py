import argparse
import importlib

import torch

from swla.model import Model


def load_config(config_path: str):
    module = importlib.import_module(config_path)
    return module.config


def profile(config_path: str, batch_size: int, seq_len: int, out_path: str):
    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(config).to(device)

    if device == "cuda":
        torch.cuda.memory._record_memory_history()

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    model(input_ids)

    if device == "cuda":
        torch.cuda.memory._dump_snapshot(out_path)
        print(f"peak allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"snapshot written to {out_path}")
    else:
        print("CUDA not available; no memory snapshot taken")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs.ratio_5to1")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--out", default="memory_snapshot.pickle")
    args = parser.parse_args()

    profile(args.config, args.batch_size, args.seq_len, args.out)


if __name__ == "__main__":
    main()
