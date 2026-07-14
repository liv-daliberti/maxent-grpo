#!/usr/bin/env python3
"""Render the exploration side-car figure (paper Figure 1) from training logs.

Reads the per-step train_metrics.jsonl that every learner writes under its run
directory (rank 0, see ZeroMathRunMixin._append_train_metrics_jsonl) and plots,
per arm, the two candidate-level aggregation diagnostics logged identically for
every quartet arm:

  * effective number of active rollouts  exp(H(a_x))   (train/agg_eff_rollouts)
  * share of aggregation mass on incorrect rollouts    (train/agg_incorrect_mass)

Legacy xdr-arm-only keys (train/xdr_eff_rollouts, train/xdr_incorrect_mass)
are accepted as fallbacks so older xDr.GRPO runs still plot.

Usage (system python3 with matplotlib; the oat env does not ship it):

  python3 ops/plot_exploration_sidecar.py --stamp-prefix gccomp1_stable \
      [--arms grpo,xdr_tau0p5,seed,grpo_entropy] \
      [--output paper/figures/exploration_sidecar.pdf]

Run dirs are discovered from the submit manifest
var/artifacts/<stamp>_comparative_jobs.tsv (glob fallback otherwise), matching
run_countdown_comparative_eval.sh. Lines are seed means; bands are seed
min-max. Missing metrics files are reported and skipped, never silently.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical colors: validated reference palette, fixed slot order
# (blue, aqua, yellow, violet, red); the baseline wears neutral ink so the
# treatment arms carry the hue identity.
BASELINE_COLOR = "#6b6a63"
SLOT_COLORS = ["#2a78d6", "#eda100", "#1baf7a", "#4a3aa7", "#e34948"]
TEXT_COLOR = "#3a3935"

METRICS = (
    ("agg_eff_rollouts", "xdr_eff_rollouts", "Effective active rollouts"),
    ("agg_incorrect_mass", "xdr_incorrect_mass", "Incorrect-mass share"),
)

RUN_STAMP_RE = re.compile(r"^(?P<arm>.+)_s(?P<seed>\d+)$")


def arm_label(arm: str) -> str:
    if arm == "grpo":
        return "Dr.GRPO"
    if arm == "grpo_entropy":
        return "Token-MaxEnt Dr.GRPO"
    if arm == "seed":
        return "SEED-Dr.GRPO"
    match = re.fullmatch(r"xdr_tau(?P<tau>.+)", arm)
    if match:
        return f"xDr.GRPO ($\\tau={match.group('tau').replace('p', '.')}$)"
    return arm


def discover_runs(root: Path, artifacts: Path, run_data_root: Path, stamp: str):
    """Yield (arm, seed, run_dir) for each training run of the stamp."""
    manifest = artifacts / f"{stamp}_comparative_jobs.tsv"
    stamps: list[str] = []
    if manifest.is_file():
        for line in manifest.read_text().splitlines()[1:]:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 4 and fields[3]:
                stamps.append(fields[3])
    else:
        print(f"[sidecar] WARNING: no manifest {manifest}; globbing", file=sys.stderr)
        stamps = [
            p.name.split("oat_zero_tiny_", 1)[-1]
            for p in run_data_root.glob(f"oat_zero_tiny_*_{stamp}_*")
        ]
    for run_stamp in stamps:
        matches = sorted(run_data_root.glob(f"oat_zero_tiny_*_{run_stamp}"))
        if not matches:
            print(f"[sidecar] missing run dir for {run_stamp}", file=sys.stderr)
            continue
        arm_seed = run_stamp[len(stamp) + 1 :] if run_stamp.startswith(stamp) else run_stamp
        match = RUN_STAMP_RE.fullmatch(arm_seed)
        if match is None:
            print(f"[sidecar] cannot parse arm/seed from {run_stamp}", file=sys.stderr)
            continue
        yield match.group("arm"), int(match.group("seed")), matches[-1]


def load_series(run_dir: Path) -> dict[str, dict[int, float]]:
    """Map metric key -> {step: value} from a run's train_metrics.jsonl."""
    path = run_dir / "train_metrics.jsonl"
    if not path.is_file():
        print(f"[sidecar] no train_metrics.jsonl in {run_dir}", file=sys.stderr)
        return {}
    series: dict[str, dict[int, float]] = defaultdict(dict)
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = record.get("trainer/step")
            if step is None:
                continue
            for primary, legacy, _title in METRICS:
                value = record.get(f"train/{primary}")
                if value is None:
                    value = record.get(f"train/{legacy}")
                if value is not None:
                    series[primary][int(step)] = float(value)
    return series


def seed_band(per_seed: list[dict[int, float]]):
    """Mean line plus min-max band over the union of logged steps."""
    steps = sorted({s for one in per_seed for s in one})
    mean, low, high = [], [], []
    for step in steps:
        values = [one[step] for one in per_seed if step in one]
        mean.append(sum(values) / len(values))
        low.append(min(values))
        high.append(max(values))
    return steps, mean, low, high


def rolling(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo : i + 1]) / (i + 1 - lo))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp-prefix", required=True)
    parser.add_argument("--run-data-root", type=Path, default=Path("var/data"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("var/artifacts"))
    parser.add_argument(
        "--arms",
        default="grpo,xdr_tau0p5,seed,grpo_entropy",
        help="comma-separated arm names in fixed color order",
    )
    parser.add_argument("--smooth", type=int, default=16, help="rolling-mean window (steps)")
    parser.add_argument(
        "--output", type=Path, default=Path("paper/figures/exploration_sidecar.pdf")
    )
    args = parser.parse_args()

    wanted = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    runs: dict[str, list[dict[str, dict[int, float]]]] = defaultdict(list)
    for arm, _seed, run_dir in discover_runs(
        Path.cwd(), args.artifacts_root, args.run_data_root, args.stamp_prefix
    ):
        if arm not in wanted:
            continue
        series = load_series(run_dir)
        if series:
            runs[arm].append(series)

    missing = [arm for arm in wanted if not runs.get(arm)]
    if missing:
        print(
            f"[sidecar] WARNING: no metric data for arms: {', '.join(missing)} "
            "(runs may predate the train_metrics.jsonl sink)",
            file=sys.stderr,
        )
    if not runs:
        raise SystemExit("no plottable data found; nothing written")

    colors: dict[str, str] = {}
    slot = 0
    for arm in wanted:
        if arm == "grpo":
            colors[arm] = BASELINE_COLOR
        else:
            colors[arm] = SLOT_COLORS[slot % len(SLOT_COLORS)]
            slot += 1

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "axes.edgecolor": "#b9b7ac",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.3), constrained_layout=True)
    for axis, (primary, _legacy, title) in zip(axes, METRICS):
        for arm in wanted:
            if not runs.get(arm):
                continue
            per_seed = [series[primary] for series in runs[arm] if series.get(primary)]
            if not per_seed:
                continue
            steps, mean, low, high = seed_band(per_seed)
            mean = rolling(mean, args.smooth)
            low = rolling(low, args.smooth)
            high = rolling(high, args.smooth)
            axis.fill_between(steps, low, high, color=colors[arm], alpha=0.14, linewidth=0)
            axis.plot(
                steps,
                mean,
                color=colors[arm],
                linewidth=1.6,
                label=arm_label(arm),
                solid_capstyle="round",
            )
        axis.set_title(title, loc="left")
        axis.set_xlabel("Learner step")
        axis.grid(True, linewidth=0.4, alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.margins(x=0.01)
    axes[1].set_ylim(bottom=0)
    axes[0].legend(frameon=False, handlelength=1.6, borderaxespad=0.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    fig.savefig(args.output.with_suffix(".png"), dpi=200)
    print(f"[sidecar] wrote {args.output} (+ .png preview)")


if __name__ == "__main__":
    main()
