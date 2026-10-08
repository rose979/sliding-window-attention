"""Aggregate benchmark repetitions and plot RAM / time per training step vs sequence length.

Reads the CSV files written by scripts/profile_memory.py (one file per repetition),
writes a summary table and two figures:

    results/summary.csv                     mean / std over repetitions per (config, L),
                                            plus the relative difference 4:1 vs. baseline
    results/figures/absolute.png|pdf        peak RAM, step RAM and time vs. L, both models
    results/figures/relative.png|pdf        relative difference 4:1 vs. baseline per metric,
                                            with the theoretical attention-score saving

Relative differences are computed within each repetition (both models measured in
the same run) and then averaged. This matters for time: the CPU speed varies between
runs, so absolute seconds differ between repetitions, but the ratio within a run is stable.
Before aggregating, check_repetitions() makes sure every file is complete and all files
come from the same settings, so the baseline/4:1 pairs are valid.

Example:
    python scripts/plot_results.py results/benchmark_rep1.csv results/benchmark_rep2.csv results/benchmark_rep3.csv
"""
import argparse
import csv
import importlib
import math
import re
import statistics
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                       # render to files, no window
import matplotlib.pyplot as plt  # noqa: E402

BASELINE = "configs.global_baseline"
SLIDING = "configs.ratio_5to1"
METRICS = [                                 # (CSV column, axis label, short name)
    ("peak_ram_gb", "Peak RAM [GB]", "Peak RAM"),
    ("step_ram_gb", "Step RAM [GB]", "Step RAM"),
    ("step_time_median_s", "Time per step [s]", "Time per step"),
]

# Colors: reference palette of the dataviz guide, categorical slots 1 and 2
# (documented to pass the colorblind-safety and contrast checks as a pair).
# Text and axes use neutral text tokens, never the series colors.
STYLE = {
    BASELINE: {"color": "#2a78d6", "label": "Baseline (all layers global)"},
    SLIDING: {"color": "#eb6834", "label": "4:1 model (sliding window W=32, every 5th layer global)"},
}
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e2dc"
# Below this baseline value a relative difference is mostly measurement noise
# (e.g. step RAM at L=128: 0.010 vs. 0.013 GB) and is left out of the relative plot
MIN_FOR_RELATIVE = 0.05
# Measurement settings that must be identical in every row of every repetition
SETTINGS = ["batch_size", "warmup_steps", "steps", "threads"]
# Model description that must be identical for each config across repetitions
MODEL = ["params_m", "layers"]


# ----------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------
def load(files: list[str]) -> list[list[dict]]:
    """One list of rows per repetition (file); only successful measurements."""
    reps = []
    for path in files:
        with open(path, newline="") as f:
            reps.append([r for r in csv.DictReader(f) if r["status"] == "ok"])
    return reps


def check_repetitions(files: list[str], reps: list[list[dict]]) -> None:
    """Make sure the repetitions can be compared pairwise; raise ValueError otherwise.

    aggregate() pairs the baseline and the 4:1 value of the same file by position, so
    every file must contain exactly one valid row per (config, L) for both models, all
    files must cover the same lengths, and all must come from the same measurement
    settings and model. Collects every problem and reports them together.
    """
    problems = []

    # The same file twice would count one run twice and make the spread look smaller
    resolved = [Path(f).resolve() for f in files]
    for path in sorted({p for p in resolved if resolved.count(p) > 1}):
        problems.append(f"{path}: given more than once")

    lengths_per_file = []
    for path, rep in zip(files, reps):
        if not rep:
            problems.append(f"{path}: no valid measurements")
            continue
        # Exactly one valid row per (config, L): a missing row means an aborted or
        # failed measurement (filtered out in load), two rows are ambiguous
        counts = Counter((r["config"], int(r["seq_len"])) for r in rep)
        lengths = sorted({seq_len for _, seq_len in counts})
        for seq_len in lengths:
            for config in (BASELINE, SLIDING):
                n_rows = counts[(config, seq_len)]
                if n_rows != 1:
                    reason = "missing or aborted" if n_rows == 0 else "duplicate rows"
                    problems.append(f"{path}: {n_rows} valid rows for {config} at L={seq_len} ({reason})")
        unknown = sorted({config for config, _ in counts} - {BASELINE, SLIDING})
        if unknown:
            problems.append(f"{path}: unexpected configs {unknown}")
        lengths_per_file.append((path, lengths))

    # All files must cover the same sequence lengths
    if len({tuple(lengths) for _, lengths in lengths_per_file}) > 1:
        detail = "; ".join(f"{path}: {lengths}" for path, lengths in lengths_per_file)
        problems.append(f"files cover different sequence lengths ({detail})")

    # Same measurement settings everywhere, same model per config across files
    rows = [(path, r) for path, rep in zip(files, reps) for r in rep]
    for key in SETTINGS:
        values = {r[key] for _, r in rows}
        if len(values) > 1:
            problems.append(f"different {key} across measurements: {sorted(values)}")
    for config in (BASELINE, SLIDING):
        for key in MODEL:
            values = {r[key] for _, r in rows if r["config"] == config}
            if len(values) > 1:
                problems.append(f"different {key} for {config} across files: {sorted(values)}")

    if problems:
        raise ValueError("repetitions can't be compared:\n  - " + "\n  - ".join(problems))


def find_repetition_files(directory: Path) -> list[str]:
    """benchmark_rep<number>.csv files, sorted by number (rep2 before rep10).
    Other names like benchmark_rep_old.csv are not picked up."""
    pattern = re.compile(r"benchmark_rep(\d+)\.csv")
    matches = [(int(m.group(1)), p) for p in directory.iterdir() if (m := pattern.fullmatch(p.name))]
    return [str(p) for _, p in sorted(matches)]


def value(rep: list[dict], config: str, seq_len: int, key: str) -> float:
    return next(float(r[key]) for r in rep if r["config"] == config and int(r["seq_len"]) == seq_len)


def mean_std(xs: list[float]) -> tuple[float, float]:
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def aggregate(reps: list[list[dict]]) -> list[dict]:
    """Per sequence length: mean/std of each metric for both models, and the mean/std
    of the per-repetition relative difference (sliding / baseline - 1) in percent."""
    seq_lens = sorted({int(r["seq_len"]) for r in reps[0]})
    summary = []
    for L in seq_lens:
        row = {"seq_len": L, "repetitions": len(reps)}
        for key, _, _ in METRICS:
            base = [value(rep, BASELINE, L, key) for rep in reps]
            slide = [value(rep, SLIDING, L, key) for rep in reps]
            row[f"{key}_baseline_mean"], row[f"{key}_baseline_std"] = mean_std(base)
            row[f"{key}_sliding_mean"], row[f"{key}_sliding_std"] = mean_std(slide)
            if statistics.mean(base) >= MIN_FOR_RELATIVE:
                rel = [100 * (s / b - 1) for b, s in zip(base, slide)]
                row[f"{key}_diff_pct_mean"], row[f"{key}_diff_pct_std"] = mean_std(rel)
            else:
                row[f"{key}_diff_pct_mean"] = row[f"{key}_diff_pct_std"] = math.nan
        summary.append(row)
    return summary


def attention_score_ratio(seq_len: int) -> float:
    """Theoretical size of all attention-score tensors, 4:1 model / baseline.

    Per head and layer, a global layer holds L * L scores, a local (chunked) layer
    n * W * 2W = 2 * L_pad * W, with L_pad = L rounded up to a multiple of W.
    Uses the layer pattern and window of the two experiment configs.
    """
    sliding = importlib.import_module(SLIDING).config
    baseline = importlib.import_module(BASELINE).config
    W = sliding.window_size
    L_pad = math.ceil(seq_len / W) * W

    def total(cfg):
        n_global = sum((i + 1) % cfg.global_every_n == 0 for i in range(cfg.num_layers))
        n_local = cfg.num_layers - n_global
        return n_global * seq_len * seq_len + n_local * 2 * L_pad * W

    return total(sliding) / total(baseline)


def write_summary(summary: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]) + ["attention_scores_theory_diff_pct"])
        writer.writeheader()
        for row in summary:
            theory = 100 * (attention_score_ratio(row["seq_len"]) - 1)
            writer.writerow({k: fmt(v) for k, v in row.items()}
                            | {"attention_scores_theory_diff_pct": round(theory, 2)})


def fmt(v):
    """Round floats for the CSV; an empty cell where no value exists (nan)."""
    if isinstance(v, float):
        return "" if math.isnan(v) else round(v, 4)
    return v


def pct(v: float) -> str:
    """Signed percentage with a typographic minus sign, e.g. '−34 %'."""
    return f"{v:+.0f} %".replace("-", "−")


def end_label(ax, x: float, y: float, text: str) -> None:
    """Direct label to the right of a line's last point (selective: one per panel)."""
    ax.annotate(text, xy=(x, y), xytext=(8, 0), textcoords="offset points",
                ha="left", va="center", color=TEXT, fontsize=10)


# ----------------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------------
def style_axis(ax, seq_lens: list[int], ylabel: str) -> None:
    """Recessive axes: hairline horizontal grid, no top/right spines, log2 x-axis
    with ticks exactly at the measured lengths."""
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log", base=2)
    ax.set_xticks(seq_lens, [str(L) for L in seq_lens])
    ax.minorticks_off()
    ax.set_xlabel("Sequence length L [tokens]", color=TEXT_2)
    ax.set_ylabel(ylabel, color=TEXT_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, length=0)
    # Room on the right for the end labels
    ax.set_xlim(seq_lens[0] / 1.15, seq_lens[-1] * 1.6)


def plot_line(ax, xs, means, stds, color, label=None):
    """2px line, >= 8px markers with a surface-colored ring, error bars = std over repetitions."""
    ax.errorbar(xs, means, yerr=stds, color=color, linewidth=1.8, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, elinewidth=1.2, capsize=3,
                label=label, zorder=3)


def plot_absolute(summary: list[dict], out: Path) -> None:
    seq_lens = [r["seq_len"] for r in summary]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=SURFACE)
    for ax, (key, ylabel, name) in zip(axes, METRICS):
        for config, which in ((BASELINE, "baseline"), (SLIDING, "sliding")):
            plot_line(ax, seq_lens, [r[f"{key}_{which}_mean"] for r in summary],
                      [r[f"{key}_{which}_std"] for r in summary], STYLE[config]["color"],
                      STYLE[config]["label"] if ax is axes[0] else None)
        style_axis(ax, seq_lens, ylabel)
        ax.set_ylim(bottom=0)
        ax.set_title(name, color=TEXT, loc="left", fontweight="bold")
        # Selective direct label: only the headline difference at the longest length
        last = summary[-1]
        end_label(ax, seq_lens[-1], last[f"{key}_sliding_mean"], pct(last[f"{key}_diff_pct_mean"]))
    fig.legend(loc="upper center", ncol=2, frameon=False, labelcolor=TEXT, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.5, -0.03, f"Medium model (32M parameters, 10 layers), batch 1, CPU. Mean ± std over "
             f"{summary[0]['repetitions']} repetitions; the label at L={seq_lens[-1]} is the 4:1 model "
             "relative to the baseline.", ha="center", color=TEXT_2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out)


def plot_relative(summary: list[dict], out: Path) -> None:
    seq_lens = [r["seq_len"] for r in summary]
    color = STYLE[SLIDING]["color"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=SURFACE, sharey=True)
    theory = [100 * (attention_score_ratio(L) - 1) for L in seq_lens]
    for ax, (key, _, name) in zip(axes, METRICS):
        ax.axhline(0, color=TEXT_2, linewidth=0.8, zorder=1)
        ax.plot(seq_lens, theory, color=TEXT_2, linewidth=1.5, linestyle=(0, (4, 3)), zorder=2,
                label="Theory: attention scores only" if ax is axes[0] else None)
        plot_line(ax, seq_lens, [r[f"{key}_diff_pct_mean"] for r in summary],
                  [r[f"{key}_diff_pct_std"] for r in summary], color,
                  "4:1 model vs. baseline (measured)" if ax is axes[0] else None)
        style_axis(ax, seq_lens, "Difference to baseline [%]" if ax is axes[0] else "")
        ax.set_title(name, color=TEXT, loc="left", fontweight="bold")
        last = summary[-1]
        end_label(ax, seq_lens[-1], last[f"{key}_diff_pct_mean"], pct(last[f"{key}_diff_pct_mean"]))
    axes[0].set_ylim(-100, 20)
    fig.legend(loc="upper center", ncol=2, frameon=False, labelcolor=TEXT, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.5, -0.06, "Negative = the 4:1 model needs less. Measured: mean ± std of the per-repetition "
             "difference. Theory: total size of all attention-score tensors (the part sliding-window\n"
             f"attention changes). Values whose baseline is below {MIN_FOR_RELATIVE} (GB or s) are left "
             "out as too small to compare (step RAM at L=128).", ha="center", color=TEXT_2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out)


def save(fig, out: Path) -> None:
    """PNG for viewing, PDF (vector) for the report."""
    for suffix in (".png", ".pdf"):
        fig.savefig(out.with_suffix(suffix), dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="benchmark CSVs, one per repetition "
                        "(default: results/benchmark_rep<number>.csv)")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    files = args.files or find_repetition_files(Path("results"))
    if not files:
        parser.error("no benchmark CSV files given or found in results/")
    reps = load(files)
    try:
        check_repetitions(files, reps)
    except ValueError as error:
        parser.error(str(error))
    out_dir = Path(args.out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    summary = aggregate(reps)
    write_summary(summary, out_dir / "summary.csv")
    plot_absolute(summary, out_dir / "figures" / "absolute")
    plot_relative(summary, out_dir / "figures" / "relative")
    print(f"{len(files)} repetitions -> {out_dir / 'summary.csv'}, {out_dir / 'figures'}/absolute|relative .png/.pdf")


if __name__ == "__main__":
    main()
