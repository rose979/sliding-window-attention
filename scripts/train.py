import argparse
import importlib

import torch

from swla.model import Model


def load_config(config_path: str):
    module = importlib.import_module(config_path)
    return module.config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs.ratio_5to1")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    config = load_config(args.config)
    model = Model(config)
    optimizer = torch.optim.AdamW(model.parameters())

    for step in range(args.steps):
        raise NotImplementedError("wire up your dataloader here")


if __name__ == "__main__":
    main()
