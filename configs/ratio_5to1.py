from dataclasses import replace

from configs.base import Config

config = replace(Config(), window_size=512, global_every_n=5)
