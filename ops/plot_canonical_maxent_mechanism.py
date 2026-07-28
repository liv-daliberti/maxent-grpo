#!/usr/bin/env python3
"""Plot the canonical-MaxEnt dose response and 0.5B controller dynamics.

The left column reads the exact, single-seed held-out graph-coloring
calibrations.  The right block reads 0.5B optimizer records and deliberately plots
only the exact *current training-prompt* entropy sensor and the coefficient
used by each update.  Those curves are mechanism telemetry, not the pending
exact held-out endpoint audits.

Run from anywhere with:

    python ops/plot_canonical_maxent_mechanism.py

This writes ``paper/figures/canonical_maxent_mechanism.{pdf,png}``.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_INITIAL = ROOT / "paper/results/e14_canonical_graph_actions.json"
CALIBRATION_EXTENSION = ROOT / "paper/results/e15_canonical_dose_calibration.json"
DATA_ROOT = ROOT / "var/data"
OUT = ROOT / "paper/figures/canonical_maxent_mechanism"

SMOOTH_PASSES = 0.125
TARGET_RATIO = 0.8487901916960259
INK = "#1b1b1b"
GRID = "#dedede"
GUIDE = "#98a4ad"

METHODS = {
    "maxent": ("fixed", "#666666", (0, (4, 2))),
    "maxent_control": ("proportional", "#0072B2", "-"),
    "maxent_dual": ("Haarnoja dual", "#D55E00", "-"),
}

DOMAINS = {
    "graph": {
        "title": "Graph coloring",
        "stamp": "gce16_canonical_maxent_05b_v2",
        "pool_size": 192,
    },
    "countdown": {
        "title": "Countdown",
        "stamp": "cde16_canonical_maxent_05b_v2",
        "pool_size": 384,
    },
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 7.6,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def max_step(rows: list[dict]) -> int:
    return max(
        (
            int(row.get("trainer/global_step", row.get("misc/global_step", -1)))
            for row in rows
        ),
        default=-1,
    )


def load_longest_attempt(run_dir: Path) -> list[dict]:
    """Select the complete attempt; break max-step ties by later timestamp."""

    attempts = []
    for path in sorted(run_dir.glob("debug_*/train_metrics.jsonl")):
        rows = read_jsonl(path)
        attempts.append((max_step(rows), str(path), rows))
    if not attempts:
        raise FileNotFoundError(f"no train_metrics.jsonl below {run_dir}")
    return max(attempts, key=lambda item: (item[0], item[1]))[2]


def discover_e16(domain: dict) -> dict[str, dict[int, list[dict]]]:
    stamp = str(domain["stamp"])
    pattern = re.compile(rf"{re.escape(stamp)}_(.+)_s(\d+)$")
    runs: dict[str, dict[int, list[dict]]] = defaultdict(dict)
    for run_dir in sorted(DATA_ROOT.glob(f"*{stamp}*")):
        match = pattern.search(run_dir.name)
        if match is None:
            continue
        arm, seed = match.group(1), int(match.group(2))
        if arm not in METHODS or seed not in {43, 44, 45}:
            continue
        runs[arm][seed] = load_longest_attempt(run_dir)

    missing = [
        f"{arm}/s{seed}"
        for arm in METHODS
        for seed in (43, 44, 45)
        if seed not in runs.get(arm, {})
    ]
    if missing:
        raise FileNotFoundError(
            f"incomplete 0.5B telemetry for {domain['title']}: {', '.join(missing)}"
        )
    return runs


def centered_smooth(values: list[float], window: int) -> list[float]:
    """Centered moving average with truncated windows at both endpoints."""

    if not values:
        return []
    window = max(1, int(window))
    before = (window - 1) // 2
    after = window // 2
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)
    smoothed = []
    for index in range(len(values)):
        lo = max(0, index - before)
        hi = min(len(values), index + after + 1)
        smoothed.append((prefix[hi] - prefix[lo]) / (hi - lo))
    return smoothed


def metric_series(
    rows: list[dict], *, key: str, pool_size: int
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        step = int(row.get("trainer/global_step", row.get("misc/global_step", -1)))
        if step <= 0:
            continue
        prompt_consumed = row.get("misc/prompt_consumed")
        passes = (
            float(prompt_consumed) / (pool_size * 16)
            if prompt_consumed is not None
            else step / pool_size
        )
        points.append((passes, float(value)))
    points.sort()
    window = round(pool_size * SMOOTH_PASSES)
    smoothed = centered_smooth([value for _, value in points], window)
    return [(points[index][0], smoothed[index]) for index in range(len(points))]


def seed_mean(
    series: dict[int, list[tuple[float, float]]]
) -> list[tuple[float, float]]:
    by_x: dict[float, list[float]] = defaultdict(list)
    for points in series.values():
        for x_value, y_value in points:
            by_x[round(x_value, 8)].append(y_value)
    return [
        (x_value, sum(values) / len(values))
        for x_value, values in sorted(by_x.items())
        if len(values) == 3
    ]


def dose_points() -> list[dict[str, float | str]]:
    initial = read_json(CALIBRATION_INITIAL)
    extension = read_json(CALIBRATION_EXTENSION)
    records: dict[float, dict[str, float | str]] = {}
    for protocol, document in (("initial", initial), ("extension", extension)):
        for arm in document["arms"].values():
            alpha = float(arm["alpha"])
            # The control is repeated verbatim in the extension; keep one copy.
            if alpha in records:
                continue
            records[alpha] = {
                "protocol": protocol if alpha else "control",
                "alpha": alpha,
                "entropy": float(arm["exact_action_entropy_mean_nats"]),
                "support": float(arm["n_eff_valid_mean"]),
                "valid": float(arm["p_valid_mean"]),
            }
    return [records[alpha] for alpha in sorted(records)]


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID, lw=0.45, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.3, width=0.65, pad=2)


def plot_dose(ax: plt.Axes, points: list[dict], metric: str, ylabel: str) -> None:
    xs = [float(point["alpha"]) for point in points]
    ys = [float(point[metric]) for point in points]
    ax.plot(xs, ys, color=GUIDE, lw=1.0, zorder=2)
    for point in points:
        protocol = str(point["protocol"])
        marker = {"control": "D", "initial": "o", "extension": "s"}[protocol]
        face = INK if protocol == "control" else ("#3B78A5" if protocol == "extension" else "white")
        ax.plot(
            float(point["alpha"]),
            float(point[metric]),
            marker=marker,
            ms=4.4,
            mec="#3B78A5" if protocol != "control" else INK,
            mfc=face,
            mew=0.85,
            ls="",
            zorder=3,
        )
    ax.set_xlim(-0.006, 0.106)
    ax.set_xticks([0, 0.05, 0.10])
    ax.set_ylabel(ylabel)
    style_axis(ax)


def plot_controller_panel(
    ax: plt.Axes,
    runs: dict[str, dict[int, list[dict]]],
    *,
    pool_size: int,
    metric_key: str,
) -> None:
    for arm, (_label, color, linestyle) in METHODS.items():
        per_seed = {
            seed: metric_series(rows, key=metric_key, pool_size=pool_size)
            for seed, rows in runs[arm].items()
        }
        for points in per_seed.values():
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                color=color,
                ls=linestyle,
                lw=0.45,
                alpha=0.22,
                zorder=2,
            )
        mean = seed_mean(per_seed)
        ax.plot(
            [point[0] for point in mean],
            [point[1] for point in mean],
            color=color,
            ls=linestyle,
            lw=1.65,
            zorder=3,
        )
    ax.set_xlim(0, 5)
    ax.set_xticks(range(0, 6))
    style_axis(ax)


def main() -> None:
    doses = dose_points()
    telemetry = {name: discover_e16(domain) for name, domain in DOMAINS.items()}

    fig = plt.figure(figsize=(7.25, 4.15))
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=(0.82, 2.18),
        left=0.075,
        right=0.992,
        bottom=0.245,
        top=0.865,
        wspace=0.32,
    )
    dose_grid = outer[0].subgridspec(3, 1, hspace=0.25)
    controller_grid = outer[1].subgridspec(2, 2, hspace=0.16, wspace=0.19)

    dose_axes = [fig.add_subplot(dose_grid[row, 0]) for row in range(3)]
    dose_metrics = [
        ("entropy", r"$H(A)$ (nats)"),
        ("support", r"$N_{\rm eff,valid}$"),
        ("valid", r"$P(\mathrm{valid})$"),
    ]
    for ax, (metric, ylabel) in zip(dose_axes, dose_metrics):
        plot_dose(ax, doses, metric, ylabel)
    for ax in dose_axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)
    dose_axes[-1].set_xlabel(r"entropy coefficient $\alpha$")
    dose_axes[-1].set_ylim(0.25, 0.32)
    dose_axes[-1].set_yticks([0.26, 0.29, 0.32])

    metric_rows = [
        (
            "train/canonical_exact_sequence_entropy_ratio",
            r"Exact entropy / $\log|\mathcal{A}|$",
        ),
        ("train/maxent_alpha_used", r"Coefficient $\alpha$ used"),
    ]
    controller_axes: list[list[plt.Axes]] = [[], []]
    for row, (metric_key, ylabel) in enumerate(metric_rows):
        for col, (domain_name, domain) in enumerate(DOMAINS.items()):
            ax = fig.add_subplot(controller_grid[row, col])
            controller_axes[row].append(ax)
            plot_controller_panel(
                ax,
                telemetry[domain_name],
                pool_size=int(domain["pool_size"]),
                metric_key=metric_key,
            )
            if row == 0:
                ax.set_title(str(domain["title"]), fontsize=8.3, pad=3)
                ax.axhline(TARGET_RATIO, color=INK, lw=0.8, ls=(0, (1.5, 2.2)), zorder=1)
                ax.annotate(
                    "target",
                    (4.97, TARGET_RATIO),
                    xytext=(-2, 3),
                    textcoords="offset points",
                    ha="right",
                    va="bottom",
                    fontsize=6.4,
                    color="#444444",
                )
                ax.set_ylim(0.27, 0.94)
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.set_ylim(0.047, 0.103)
                ax.set_yticks([0.05, 0.075, 0.10])
                ax.set_xlabel("training passes")
            if col == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.tick_params(axis="y", labelleft=False)

    dose_box = outer[0].get_position(fig)
    control_box = outer[1].get_position(fig)
    fig.text(
        (dose_box.x0 + dose_box.x1) / 2,
        0.94,
        "(a) Exact dose calibration\nGraph coloring, 0.5B — one seed",
        ha="center",
        va="top",
        fontsize=8.3,
        weight="semibold",
    )
    fig.text(
        (control_box.x0 + control_box.x1) / 2,
        0.94,
        "(b) 0.5B dynamics — exact training-prompt sensor (not held-out)",
        ha="center",
        va="top",
        fontsize=8.0,
        weight="semibold",
    )

    dose_handles = [
        Line2D([], [], marker="D", ls="", color=INK, ms=4.1, label=r"$\alpha=0$ control"),
        Line2D([], [], marker="o", ls="", mec="#3B78A5", mfc="white", color="#3B78A5", ms=4.2, label="initial sweep"),
        Line2D([], [], marker="s", ls="", color="#3B78A5", ms=4.2, label="dose extension"),
    ]
    method_handles = [
        Line2D([], [], color=color, ls=linestyle, lw=1.6, label=label)
        for label, color, linestyle in METHODS.values()
    ]
    fig.legend(
        handles=dose_handles + method_handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.082),
        ncol=6,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.25,
        fontsize=7.0,
    )
    fig.text(
        0.995,
        0.012,
        "Controller curves: centered 0.125-pass mean; thin lines are seeds 43–45, heavy lines are their mean.",
        ha="right",
        va="bottom",
        fontsize=6.3,
        color="#4a4a4a",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(
        OUT.with_suffix(".png"),
        dpi=240,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
    print(f"wrote {OUT.with_suffix('.pdf')}")
    print(f"wrote {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
