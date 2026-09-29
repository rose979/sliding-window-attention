from dataclasses import replace

from swla.config import Config

config = replace(Config(), window_size=Config().max_seq_len, global_every_n=1)
