import argparse
import importlib

import torch

from swla.model import Model


def load_config(config_path: str):
    module = importlib.import_module(config_path)
    return module.config


@torch.no_grad()
def generate(model: Model, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    # No KV cache: the full sequence (cropped to max_seq_len) is recomputed each step
    tokens = input_ids

    for _ in range(max_new_tokens):
        logits, _ = model(tokens[:, -model.config.max_seq_len:])
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs.ratio_5to1")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    model = Model(config)
    model.eval()

    prompt = torch.zeros(1, 1, dtype=torch.long)
    output = generate(model, prompt, args.max_new_tokens)
    print(output)


if __name__ == "__main__":
    main()
