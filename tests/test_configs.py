from dataclasses import asdict

from configs.global_baseline import config as baseline
from configs.ratio_5to1 import config as ratio


def test_experiment_configs_differ_only_in_attention():
    """The comparison is only fair if both models are identical apart from attention."""
    attention_fields = {"window_size", "global_every_n"}
    differing = {k for k, v in asdict(baseline).items() if asdict(ratio)[k] != v}
    assert differing == attention_fields


def test_ratio_config_matches_pi_requirements():
    assert ratio.window_size == 32
    assert ratio.global_every_n == 5
    assert ratio.num_layers % ratio.global_every_n == 0    # clean 4:1 pattern
    assert baseline.global_every_n == 1                    # all layers global
