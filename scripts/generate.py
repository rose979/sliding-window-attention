import argparse
import importlib

import torch

from swla.model import Model


def load_config(config_path: str):
    module = importlib.import_module(config_path)
    return module.config


@torch.no_grad()
def generate(model: Model, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    kv_caches = model.init_cache(batch_size=input_ids.shape[0], device=input_ids.device)
    tokens = input_ids

    for _ in range(max_new_tokens):
        logits = model(tokens[:, -1:] if kv_caches[0].length > 0 else tokens, kv_caches=kv_caches)
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
