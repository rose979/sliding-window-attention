from dataclasses import replace

from configs.base import Config

config = replace(Config(), window_size=32, global_every_n=5)
