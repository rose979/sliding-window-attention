"""Tests for the repetition check and the pairing in scripts/plot_results.py."""
import csv
import importlib.util
import math
from pathlib import Path

import pytest

# scripts/ is not a package, so the script is loaded from its file path
_spec = importlib.util.spec_from_file_location(
    "plot_results", Path(__file__).resolve().parents[1] / "scripts" / "plot_results.py"
)
plot_results = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plot_results)

BASELINE, SLIDING = plot_results.BASELINE, plot_results.SLIDING
FIELDS = ["config", "seq_len", "batch_size", "params_m", "layers", "threads", "warmup_steps",
          "steps", "peak_ram_gb", "base_ram_gb", "step_ram_gb", "step_time_s",
          "step_time_median_s", "step_time_std_s", "status"]


def row(config, seq_len, peak=1.0, **overrides):
    r = {"config": config, "seq_len": seq_len, "batch_size": 1, "params_m": 31.64,
         "layers": "GGGGGGGGGG" if config == BASELINE else "LLLLGLLLLG", "threads": 12,
         "warmup_steps": 2, "steps": 20, "peak_ram_gb": peak, "base_ram_gb": 0.8,
         "step_ram_gb": peak - 0.8, "step_time_s": 1.0, "step_time_median_s": 1.0,
         "step_time_std_s": 0.1, "status": "ok"}
    r.update(overrides)
    return r


def complete_rows(seq_lens=(128, 2048), **overrides):
    return [row(c, L, **overrides) for L in seq_lens for c in (BASELINE, SLIDING)]


def write(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def check(files):
    plot_results.check_repetitions(files, plot_results.load(files))


def test_complete_repetitions_pass(tmp_path):
    files = [write(tmp_path / f"rep{i}.csv", complete_rows()) for i in (1, 2, 3)]
    check(files)                                             # no exception


def test_missing_or_aborted_row_is_reported(tmp_path):
    rows = complete_rows()
    rows[-2]["status"] = "aborted: RSS > 14.0 GB"            # baseline at L=2048
    files = [write(tmp_path / "rep1.csv", complete_rows()), write(tmp_path / "rep2.csv", rows)]
    with pytest.raises(ValueError, match=r"rep2\.csv: 0 valid rows for configs\.global_baseline at L=2048"):
        check(files)


def test_duplicate_row_is_reported(tmp_path):
    files = [write(tmp_path / "rep1.csv", complete_rows() + [row(SLIDING, 128)])]
    with pytest.raises(ValueError, match=r"2 valid rows for configs\.ratio_5to1 at L=128 \(duplicate rows\)"):
        check(files)


def test_different_lengths_are_reported(tmp_path):
    files = [write(tmp_path / "rep1.csv", complete_rows((128, 2048))),
             write(tmp_path / "rep2.csv", complete_rows((128, 512)))]
    with pytest.raises(ValueError, match="different sequence lengths"):
        check(files)


def test_different_settings_are_reported(tmp_path):
    files = [write(tmp_path / "rep1.csv", complete_rows()),
             write(tmp_path / "rep2.csv", complete_rows(steps=10))]
    with pytest.raises(ValueError, match="different steps"):
        check(files)


def test_different_model_is_reported(tmp_path):
    files = [write(tmp_path / "rep1.csv", complete_rows()),
             write(tmp_path / "rep2.csv", complete_rows(params_m=8.0))]
    with pytest.raises(ValueError, match="different params_m"):
        check(files)


def test_same_file_twice_is_reported(tmp_path):
    path = write(tmp_path / "rep1.csv", complete_rows())
    with pytest.raises(ValueError, match="given more than once"):
        check([path, path])


def test_all_problems_are_reported_together(tmp_path):
    rows = complete_rows(steps=10)
    rows.pop()                                               # missing row as well
    files = [write(tmp_path / "rep1.csv", complete_rows()), write(tmp_path / "rep2.csv", rows)]
    with pytest.raises(ValueError) as error:
        check(files)
    assert "missing or aborted" in str(error.value) and "different steps" in str(error.value)


def test_repetition_files_are_found_by_number(tmp_path):
    for name in ("benchmark_rep10.csv", "benchmark_rep2.csv", "benchmark_rep1.csv",
                 "benchmark_rep_old.csv", "benchmark.csv"):
        (tmp_path / name).write_text("")
    names = [Path(p).name for p in plot_results.find_repetition_files(tmp_path)]
    assert names == ["benchmark_rep1.csv", "benchmark_rep2.csv", "benchmark_rep10.csv"]


def test_pairs_come_from_the_same_file(tmp_path):
    """Run 1 is slow, run 2 fast; in both the 4:1 model needs exactly half. Pairing
    within each file gives -50 % with no spread, whatever order the files are in."""
    run1 = [row(BASELINE, 2048, peak=4.0), row(SLIDING, 2048, peak=2.0)]
    run2 = [row(BASELINE, 2048, peak=2.0), row(SLIDING, 2048, peak=1.0)]
    for order in ((run1, run2), (run2, run1)):
        files = [write(tmp_path / f"rep{i}.csv", rows) for i, rows in enumerate(order, 1)]
        summary = plot_results.aggregate(plot_results.load(files))
        assert math.isclose(summary[0]["peak_ram_gb_diff_pct_mean"], -50.0)
        assert math.isclose(summary[0]["peak_ram_gb_diff_pct_std"], 0.0, abs_tol=1e-9)
