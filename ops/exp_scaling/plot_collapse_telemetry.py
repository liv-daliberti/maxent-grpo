#!/usr/bin/env python3
"""Plot and summarize exploratory telemetry inside 3B coverage collapse.

The figure deliberately distinguishes token-level policy entropy from xDr's
candidate aggregation entropy. It reads the longest metrics log for each run,
so it can be regenerated as the ongoing campaign advances without joining
restarted attempts or double-counting steps.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ARMS = {
    "grpo": ("Dr.GRPO", "#767676", (0, (4, 2))),
    "xdr_tau0p05": (
        r"xDr.GRPO $\tau_{\rm agg}{=}0.05$",
        "#31688e",
        "-",
    ),
}
INK = "#1a1a1a"


def _read_rows(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _max_step(rows: list[dict[str, float]]) -> int:
    return max((int(row.get("trainer/global_step", -1)) for row in rows), default=-1)


def load_longest_attempt(run_dir: Path) -> tuple[list[dict[str, float]], Path]:
    """Load one coherent attempt, preferring the log with the largest step."""

    candidates: list[tuple[int, Path, list[dict[str, float]]]] = []
    for path in sorted(run_dir.glob("debug_*/train_metrics.jsonl")):
        rows = _read_rows(path)
        candidates.append((_max_step(rows), path, rows))
    if not candidates:
        raise FileNotFoundError(f"no train_metrics.jsonl under {run_dir}")
    _step, path, rows = max(candidates, key=lambda item: (item[0], str(item[1])))
    return rows, path


def discover_runs(
    *, stamp_prefix: str
) -> dict[str, dict[int, tuple[list[dict[str, float]], Path]]]:
    pattern = re.compile(rf"{re.escape(stamp_prefix)}_(.+)_s(\d+)$")
    runs: dict[str, dict[int, tuple[list[dict[str, float]], Path]]] = defaultdict(dict)
    for run_dir in sorted((ROOT / "var/data").glob(f"*{stamp_prefix}*")):
        match = pattern.search(run_dir.name)
        if match is None or match.group(1) not in ARMS:
            continue
        arm, seed = match.group(1), int(match.group(2))
        runs[arm][seed] = load_longest_attempt(run_dir)
    missing = sorted(set(ARMS) - set(runs))
    if missing:
        raise FileNotFoundError(f"missing telemetry arms: {', '.join(missing)}")
    return runs


def trailing_smooth(
    points: list[tuple[int, float]], *, window: int
) -> list[tuple[int, float]]:
    ordered = sorted(points)
    result: list[tuple[int, float]] = []
    for index, (step, _value) in enumerate(ordered):
        values = [value for _, value in ordered[max(0, index - window) : index + 1]]
        result.append((step, sum(values) / len(values)))
    return result


def metric_points(
    rows: list[dict[str, float]],
    value_fn: Callable[[dict[str, float]], float | None],
) -> list[tuple[int, float]]:
    points = []
    for row in rows:
        value = value_fn(row)
        if value is not None:
            points.append((int(row["trainer/global_step"]), float(value)))
    return points


def seed_mean(
    per_seed: dict[int, list[tuple[int, float]]], *, min_seeds: int = 2
) -> tuple[list[int], list[float], list[int]]:
    by_step: dict[int, list[float]] = defaultdict(list)
    for points in per_seed.values():
        for step, value in points:
            by_step[step].append(value)
    steps = sorted(step for step, values in by_step.items() if len(values) >= min_seeds)
    return (
        steps,
        [sum(by_step[step]) / len(by_step[step]) for step in steps],
        [len(by_step[step]) for step in steps],
    )


def first_persistent_crossing(
    steps: list[int], values: list[float], *, threshold: float, hold: int = 25
) -> int | None:
    for index in range(0, len(values) - hold + 1):
        if all(value < threshold for value in values[index : index + hold]):
            return steps[index]
    return None


def build_series(
    runs: dict[str, dict[int, tuple[list[dict[str, float]], Path]]],
    *,
    smooth_window: int,
):
    entropy: dict[str, dict[int, list[tuple[int, float]]]] = defaultdict(dict)
    informative: dict[str, dict[int, list[tuple[int, float]]]] = defaultdict(dict)
    all_correct: dict[str, dict[int, list[tuple[int, float]]]] = defaultdict(dict)
    all_zero: dict[str, dict[int, list[tuple[int, float]]]] = defaultdict(dict)
    reward: dict[str, dict[int, list[tuple[int, float]]]] = defaultdict(dict)
    effective: dict[str, dict[int, list[tuple[int, float]]]] = defaultdict(dict)
    phase: dict[str, dict[int, list[tuple[int, float, float]]]] = defaultdict(dict)

    for arm, seed_runs in runs.items():
        for seed, (rows, _path) in seed_runs.items():
            entropy[arm][seed] = trailing_smooth(
                metric_points(rows, lambda row: row.get("train/entropy")),
                window=smooth_window,
            )
            all_correct[arm][seed] = trailing_smooth(
                metric_points(rows, lambda row: row.get("train/all_one_rewards_count")),
                window=smooth_window,
            )
            all_zero[arm][seed] = trailing_smooth(
                metric_points(rows, lambda row: row.get("train/all_zero_rewards_count")),
                window=smooth_window,
            )
            informative[arm][seed] = trailing_smooth(
                metric_points(
                    rows,
                    lambda row: (
                        1.0
                        - row["train/all_one_rewards_count"]
                        - row["train/all_zero_rewards_count"]
                        if row.get("train/all_one_rewards_count") is not None
                        and row.get("train/all_zero_rewards_count") is not None
                        else None
                    ),
                ),
                window=smooth_window,
            )
            reward[arm][seed] = trailing_smooth(
                metric_points(rows, lambda row: row.get("actor/rewards")),
                window=smooth_window,
            )
            effective[arm][seed] = trailing_smooth(
                metric_points(rows, lambda row: row.get("train/agg_eff_rollouts")),
                window=smooth_window,
            )

            entropy_at_step = dict(entropy[arm][seed])
            observations = []
            for row in rows:
                step = int(row.get("trainer/global_step", -1))
                coverage = row.get("eval/multi_answer/sampled_mode_coverage_at_8")
                if step > 0 and coverage is not None and step in entropy_at_step:
                    observations.append((step, entropy_at_step[step], float(coverage)))
            phase[arm][seed] = observations

    return entropy, informative, all_correct, all_zero, reward, effective, phase


def _mean_map(per_seed: dict[int, list[tuple[int, float]]]):
    steps, values, counts = seed_mean(per_seed)
    return {step: (value, count) for step, value, count in zip(steps, values, counts)}


def summarize(
    runs,
    entropy,
    informative,
    all_correct,
    all_zero,
    reward,
    effective,
    phase,
    *,
    steps_per_pass: int,
) -> dict[str, object]:
    mean_maps = {
        name: {arm: _mean_map(series[arm]) for arm in ARMS}
        for name, series in {
            "token_entropy": entropy,
            "informative_fraction": informative,
            "all_correct_fraction": all_correct,
            "all_zero_fraction": all_zero,
            "rollout_reward": reward,
            "effective_rollouts": effective,
        }.items()
    }
    common_steps = set.intersection(
        *(set(mean_maps["token_entropy"][arm]) for arm in ARMS)
    )
    snapshot_step = max(common_steps)

    threshold_crossings = {}
    for arm in ARMS:
        steps = sorted(mean_maps["token_entropy"][arm])
        values = [mean_maps["token_entropy"][arm][step][0] for step in steps]
        threshold_crossings[arm] = first_persistent_crossing(
            steps, values, threshold=0.20
        )

    phase_rows = {
        arm: [point for points in phase[arm].values() for point in points]
        for arm in ARMS
    }
    overlap_low = max(min(point[1] for point in phase_rows[arm]) for arm in ARMS)
    overlap_high = min(max(point[1] for point in phase_rows[arm]) for arm in ARMS)
    overlap = {}
    for arm in ARMS:
        selected = [
            point for point in phase_rows[arm] if overlap_low <= point[1] <= overlap_high
        ]
        overlap[arm] = {
            "observations": len(selected),
            "mean_coverage_at_8": sum(point[2] for point in selected) / len(selected),
        }

    return {
        "stamp_prefix": "gce1_3b",
        "steps_per_pass": steps_per_pass,
        "runs": {
            arm: {
                str(seed): {"max_step": _max_step(rows), "log": str(path.relative_to(ROOT))}
                for seed, (rows, path) in seed_runs.items()
            }
            for arm, seed_runs in runs.items()
        },
        "latest_common_step": snapshot_step,
        "latest_common_pass": snapshot_step / steps_per_pass,
        "latest_common_seed_count": {
            arm: mean_maps["token_entropy"][arm][snapshot_step][1] for arm in ARMS
        },
        "token_entropy_below_0p20_persistent_step": threshold_crossings,
        "latest_common_metrics": {
            name: {
                arm: mean_maps[name][arm][snapshot_step][0] for arm in ARMS
            }
            for name in mean_maps
        },
        "phase_overlap": {
            "token_entropy_low": overlap_low,
            "token_entropy_high": overlap_high,
            **overlap,
            "coverage_delta_xdr_minus_grpo": (
                overlap["xdr_tau0p05"]["mean_coverage_at_8"]
                - overlap["grpo"]["mean_coverage_at_8"]
            ),
            "interpretation": "descriptive; observations occur at different steps",
        },
    }


def plot_telemetry(
    entropy,
    informative,
    phase,
    *,
    steps_per_pass: int,
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.7,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.35))

    for ax, series, ylabel, title in [
        (axes[0], entropy, "token entropy", "a  Policy sharpening"),
        (
            axes[1],
            informative,
            "informative-group fraction",
            "b  Group signal remains",
        ),
    ]:
        ax.axvspan(0, 1, color="#f4f4f4", zorder=0)
        for arm, (_label, color, linestyle) in ARMS.items():
            for points in series[arm].values():
                ax.plot(
                    [step / steps_per_pass for step, _ in points],
                    [value for _, value in points],
                    color=color,
                    alpha=0.20,
                    lw=0.55,
                    zorder=1,
                )
            steps, values, _counts = seed_mean(series[arm])
            ax.plot(
                [step / steps_per_pass for step in steps],
                values,
                color=color,
                linestyle=linestyle,
                lw=1.7,
                zorder=3,
            )
        ax.axvline(1, color="#aaaaaa", lw=0.6, ls=(0, (1.5, 2)), zorder=2)
        ax.set_xlabel("training passes")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=8.3)

    axes[1].set_ylim(0.45, 1.01)

    ax = axes[2]
    all_phase = {}
    for arm, (label, color, linestyle) in ARMS.items():
        for points in phase[arm].values():
            ax.plot(
                [entropy_value for _step, entropy_value, _coverage in points],
                [coverage for _step, _entropy, coverage in points],
                color=color,
                alpha=0.25,
                lw=0.65,
                marker="o",
                ms=2.0,
            )
        by_step: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for points in phase[arm].values():
            for step, entropy_value, coverage in points:
                by_step[step].append((entropy_value, coverage))
        mean_points = []
        for step in sorted(by_step):
            values = by_step[step]
            if len(values) >= 2:
                mean_points.append(
                    (
                        sum(value[0] for value in values) / len(values),
                        sum(value[1] for value in values) / len(values),
                    )
                )
        all_phase[arm] = [point for points in phase[arm].values() for point in points]
        ax.plot(
            [point[0] for point in mean_points],
            [point[1] for point in mean_points],
            color=color,
            linestyle=linestyle,
            lw=1.7,
            marker="o",
            ms=2.8,
            label=label,
        )

    overlap_low = max(min(point[1] for point in all_phase[arm]) for arm in ARMS)
    overlap_high = min(max(point[1] for point in all_phase[arm]) for arm in ARMS)
    ax.axvspan(overlap_low, overlap_high, color="#eeeeee", zorder=0)
    ax.invert_xaxis()
    ax.set_xlabel("token entropy")
    ax.set_ylabel("eval coverage@8")
    ax.set_title("c  Coverage--entropy frontier", loc="left", fontsize=8.3)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")

    for ax in axes:
        ax.grid(axis="y", color="#dddddd", lw=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.5, width=0.7)

    fig.tight_layout(pad=0.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=220)
    print(f"wrote {output}.pdf/.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp-prefix", default="gce1_3b")
    parser.add_argument("--steps-per-pass", type=int, default=1024)
    parser.add_argument("--smooth-window", type=int, default=25)
    parser.add_argument(
        "--output", type=Path, default=Path("paper/figures/collapse_telemetry")
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("paper/results/collapse_telemetry_3b_summary.json"),
    )
    args = parser.parse_args()

    runs = discover_runs(stamp_prefix=args.stamp_prefix)
    series = build_series(runs, smooth_window=args.smooth_window)
    summary = summarize(
        runs,
        *series,
        steps_per_pass=args.steps_per_pass,
    )
    summary["stamp_prefix"] = args.stamp_prefix
    summary_path = ROOT / args.summary_output
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {summary_path}")
    plot_telemetry(
        series[0],
        series[1],
        series[-1],
        steps_per_pass=args.steps_per_pass,
        output=ROOT / args.output,
    )


if __name__ == "__main__":
    main()
