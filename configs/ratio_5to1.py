from dataclasses import replace

from swla.config import Config

config = replace(Config(), window_size=32, global_every_n=5)
