"""Benchmark RAM and time per training step (CPU) for several configs and sequence lengths.

Every (config, seq_len) combination runs in its own fresh Python process, so memory
from one measurement can't carry over into the next. In each process:

    build model + AdamW  ->  warm-up steps (not measured)  ->  measured steps

and per measured step a background thread samples the process RAM (RSS) every ~1 ms.
Reported per combination:

    peak_ram_gb    highest RSS during the measured steps: total RAM use of training
                   (weights, gradients, AdamW states, activations, Python/torch overhead)
    base_ram_gb    RSS at the start of the measured steps: everything that stays in
                   memory between steps (weights, gradients, AdamW states, overhead)
    step_ram_gb    peak_ram_gb - base_ram_gb: memory the step itself adds, mostly
                   activations; this is where the attention types differ
    step_time_s    mean, median and std of the wall time of the measured steps
                   (the median is robust against single slow outliers, e.g. OS activity)

Example:
    python scripts/profile_memory.py --seq-lens 128 512 2048 --out results/benchmark.csv
"""
import argparse
import csv
import importlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import torch

from swla.model import Model
from swla.training import random_batch, train_step

DEFAULT_CONFIGS = ["configs.global_baseline", "configs.ratio_5to1"]
FIELDS = [
    "config", "seq_len", "batch_size", "params_m", "layers", "threads",
    "warmup_steps", "steps", "peak_ram_gb", "base_ram_gb", "step_ram_gb",
    "step_time_s", "step_time_median_s", "step_time_std_s", "status",
]


def load_config(config_path: str):
    module = importlib.import_module(config_path)
    return module.config


class PeakRSSSampler:
    """Samples the RSS of this process in a background thread.

    PyTorch releases the Python GIL inside its operations, so the thread keeps
    sampling while a training step runs. Call reset() before a step and read
    .peak after it. Aborts the process if RSS exceeds max_bytes.
    """

    def __init__(self, interval_s: float = 0.001, max_bytes: float | None = None):
        self.process = psutil.Process()
        self.interval_s = interval_s
        self.max_bytes = max_bytes
        self.peak = 0
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self._running:
            rss = self.process.memory_info().rss
            self.peak = max(self.peak, rss)
            if self.max_bytes is not None and rss > self.max_bytes:
                print(json.dumps({"status": f"aborted: RSS > {self.max_bytes / 1e9:.1f} GB"}), flush=True)
                os._exit(1)
            time.sleep(self.interval_s)

    def current(self) -> int:
        return self.process.memory_info().rss

    def reset(self) -> int:
        """Start a new measurement window; returns the RSS at this moment."""
        rss = self.current()
        self.peak = rss
        return rss

    def stop(self):
        self._running = False


def measure(config_path: str, seq_len: int, batch_size: int, warmup: int, steps: int,
            threads: int, max_ram_gb: float) -> dict:
    """One measurement inside the current process (called in worker mode)."""
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    sampler = PeakRSSSampler(max_bytes=max_ram_gb * 1e9)

    config = load_config(config_path)
    model = Model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batch = random_batch(batch_size, seq_len, config.vocab_size)

    # Warm-up: the first step creates the AdamW states and one-time buffers;
    # these steps are not part of any reported value
    for _ in range(warmup):
        train_step(model, optimizer, *batch)

    base_rss, peaks, times = None, [], []
    for _ in range(steps):
        start_rss = sampler.reset()
        base_rss = start_rss if base_rss is None else min(base_rss, start_rss)
        t0 = time.perf_counter()
        train_step(model, optimizer, *batch)
        times.append(time.perf_counter() - t0)
        time.sleep(0.005)                    # let the sampler catch the end of the step
        peaks.append(sampler.peak)
    sampler.stop()

    peak = max(peaks)
    return {
        "config": config_path, "seq_len": seq_len, "batch_size": batch_size,
        "params_m": round(model.num_parameters() / 1e6, 2),
        "layers": "".join("G" if b.is_global else "L" for b in model.blocks),
        "threads": threads, "warmup_steps": warmup, "steps": steps,
        "peak_ram_gb": round(peak / 1e9, 3),
        "base_ram_gb": round(base_rss / 1e9, 3),
        "step_ram_gb": round((peak - base_rss) / 1e9, 3),
        "step_time_s": round(statistics.mean(times), 4),
        "step_time_median_s": round(statistics.median(times), 4),
        "step_time_std_s": round(statistics.stdev(times), 4) if len(times) > 1 else 0.0,
        "status": "ok",
    }


def run_in_subprocess(args, config_path: str, seq_len: int) -> dict:
    """Run one measurement in a fresh Python process and parse its JSON line."""
    cmd = [
        sys.executable, __file__, "--worker",
        "--configs", config_path, "--seq-lens", str(seq_len),
        "--batch-size", str(args.batch_size), "--warmup", str(args.warmup),
        "--steps", str(args.steps), "--threads", str(args.threads),
        "--max-ram-gb", str(args.max_ram_gb),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    lines = out.stdout.strip().splitlines()
    if not lines:
        return {"config": config_path, "seq_len": seq_len, "status": f"failed: {out.stderr.strip()[-200:]}"}
    result = json.loads(lines[-1])
    result.setdefault("config", config_path)
    result.setdefault("seq_len", seq_len)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[128, 512, 2048])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--threads", type=int, default=psutil.cpu_count(logical=False),
                        help="CPU threads for torch (fixed so all runs are comparable)")
    parser.add_argument("--max-ram-gb", type=float, default=14.0, help="abort a run above this RSS")
    parser.add_argument("--out", default="results/benchmark.csv")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        result = measure(args.configs[0], args.seq_lens[0], args.batch_size, args.warmup,
                         args.steps, args.threads, args.max_ram_gb)
        print(json.dumps(result), flush=True)
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{args.steps} measured steps after {args.warmup} warm-up steps, "
          f"batch {args.batch_size}, {args.threads} threads -> {out_path}")
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        # Configs alternate within each length, so slow drift (e.g. CPU heat)
        # affects both configs similarly
        for seq_len in args.seq_lens:
            for config_path in args.configs:
                r = run_in_subprocess(args, config_path, seq_len)
                writer.writerow(r)
                f.flush()
                if r["status"] == "ok":
                    print(f"{config_path:25s} L={seq_len:5d}  peak {r['peak_ram_gb']:6.2f} GB  "
                          f"step {r['step_ram_gb']:6.2f} GB  time {r['step_time_s']:7.3f} (median {r['step_time_median_s']:.3f}) "
                          f"± {r['step_time_std_s']:.3f} s", flush=True)
                else:
                    print(f"{config_path:25s} L={seq_len:5d}  {r['status']}", flush=True)


if __name__ == "__main__":
    main()
