#!/usr/bin/env python3
"""Plot and freeze the paired common-horizon ModeBench interim snapshot."""

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "Graph coloring": ROOT
    / "var/artifacts/gce50_uncapped_normalized_canonical_haarnoja_05b_50ep_v1_scaling_curve.json",
    "Countdown": ROOT
    / "var/artifacts/cde50_uncapped_normalized_canonical_haarnoja_05b_50ep_v1_scaling_curve.json",
}
FROZEN_HORIZONS = {
    "Graph coloring": 8.25,
    "Countdown": 2.75,
}
MATHIR_DOMAIN = "Executable MathIR (held-out action menu; not MATH-500)"
EXTENSION_SOURCES = {
    MATHIR_DOMAIN: ROOT
    / "var/artifacts/mie59_mathir_global_verified_replay_05b_50ep_scaling_curve.json",
}
ARMS = {
    "grpo": ("Dr.GRPO", "#333333"),
    "online_canonical_haarnoja": ("Verified MaxEnt", "#d65f9e"),
}
TREATMENT_LABELS = {
    MATHIR_DOMAIN: "E59 global verified replay",
}
ROW_LABELS = {
    "Graph coloring": "Graph coloring",
    "Countdown": "Countdown",
    MATHIR_DOMAIN: "Executable MathIR\n(not MATH-500)",
}
METRICS = (
    ("pass8", r"pass@8"),
    ("coverage8", r"valid coverage@8"),
    ("distinct8", r"distinct correct@8"),
    ("verified_discovery_cumulative_outcomes", "verified discoveries"),
    ("verified_discovery_mean_support_per_prompt", "verified support / prompt"),
    (
        "online_canonical_entropy_alpha_used",
        r"adaptive coefficient $\alpha/\beta$",
    ),
)


def _load(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_mathir(path: Path) -> list[dict[str, Any]]:
    rows = []
    for raw in _load(path):
        row = dict(raw)
        row["native_arm"] = row.get("arm")
        if row.get("arm") == "verified_first_global_replay_canonical":
            row["arm"] = "online_canonical_haarnoja"
        rows.append(row)
    return rows


def _is_evaluation(row: dict[str, Any]) -> bool:
    return row.get("pass8") is not None


def _paired_seed_horizon(
    rows: list[dict[str, Any]],
) -> tuple[float | None, tuple[int, ...]]:
    maxima: dict[tuple[str, int], float] = {}
    for row in rows:
        if not _is_evaluation(row) or row.get("arm") not in ARMS:
            continue
        key = (str(row["arm"]), int(row["seed"]))
        maxima[key] = max(maxima.get(key, 0.0), float(row["training_passes"]))
    paired = tuple(
        seed
        for seed in (43, 44, 45)
        if all((arm, seed) in maxima for arm in ARMS)
    )
    if not paired:
        return None, ()
    return (
        min(maxima[(arm, seed)] for arm in ARMS for seed in paired),
        paired,
    )


def _common_evaluation_horizon(rows: list[dict[str, Any]]) -> float:
    horizon, paired = _paired_seed_horizon(rows)
    if set(paired) != {43, 44, 45} or horizon is None:
        missing = sorted({43, 44, 45} - set(paired))
        raise RuntimeError(f"missing ModeBench paired seeds: {missing}")
    return horizon


def _series(
    rows: list[dict[str, Any]],
    *,
    arm: str,
    seed: int,
    metric: str,
    horizon: float,
) -> list[tuple[float, float]]:
    points: dict[float, float] = {}
    for row in rows:
        if row.get("arm") != arm or int(row.get("seed", -1)) != int(seed):
            continue
        training_passes = float(row.get("training_passes", 0.0))
        value = row.get(metric)
        if training_passes <= horizon + 1e-9 and value is not None:
            points[training_passes] = float(value)
    return sorted(points.items())


def _mean_series(
    rows: list[dict[str, Any]],
    *,
    arm: str,
    metric: str,
    horizon: float,
    seeds: tuple[int, ...] = (43, 44, 45),
) -> list[tuple[float, float]]:
    by_pass: dict[float, dict[int, float]] = defaultdict(dict)
    for seed in seeds:
        for training_passes, value in _series(
            rows,
            arm=arm,
            seed=seed,
            metric=metric,
            horizon=horizon,
        ):
            by_pass[training_passes][seed] = value
    return [
        (training_passes, float(fmean(seed_values.values())))
        for training_passes, seed_values in sorted(by_pass.items())
        if set(seed_values) == set(seeds)
    ]


def _endpoint(
    rows: list[dict[str, Any]],
    *,
    arm: str,
    seed: int,
    metric: str,
    horizon: float,
) -> tuple[float, float]:
    points = _series(
        rows,
        arm=arm,
        seed=seed,
        metric=metric,
        horizon=horizon,
    )
    if not points:
        raise RuntimeError(f"missing endpoint for {arm=} {seed=} {metric=}")
    return points[-1]


def _summary(source_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    domains: dict[str, Any] = {}
    for domain, rows in source_rows.items():
        horizon = FROZEN_HORIZONS[domain]
        available_horizon = _common_evaluation_horizon(rows)
        if available_horizon + 1e-9 < horizon:
            raise RuntimeError(
                f"{domain} has only {available_horizon} paired passes; "
                f"the frozen horizon requires {horizon}"
            )
        endpoints: dict[str, Any] = {}
        for arm in ARMS:
            by_seed: dict[str, Any] = {}
            for seed in (43, 44, 45):
                values = {}
                endpoint_passes = None
                for metric in ("pass8", "mean8", "coverage8", "distinct8", "greedy"):
                    metric_passes, value = _endpoint(
                        rows,
                        arm=arm,
                        seed=seed,
                        metric=metric,
                        horizon=horizon,
                    )
                    endpoint_passes = (
                        metric_passes if endpoint_passes is None else endpoint_passes
                    )
                    if abs(endpoint_passes - metric_passes) > 1e-9:
                        raise RuntimeError("evaluation metrics have mismatched endpoints")
                    values[metric] = value
                by_seed[str(seed)] = {
                    "training_passes": endpoint_passes,
                    **values,
                }
            means = {
                metric: float(
                    fmean(by_seed[str(seed)][metric] for seed in (43, 44, 45))
                )
                for metric in ("pass8", "mean8", "coverage8", "distinct8", "greedy")
            }
            endpoints[arm] = {"by_seed": by_seed, "mean": means}
        treatment = endpoints["online_canonical_haarnoja"]["mean"]
        control = endpoints["grpo"]["mean"]
        domains[domain] = {
            "paired_common_horizon_passes": horizon,
            "endpoints": endpoints,
            "treatment_minus_control_mean": {
                metric: treatment[metric] - control[metric]
                for metric in treatment
            },
        }
    return {
        "schema": "modebench_long_horizon_interim_v1",
        "status": "FROZEN_INTERIM_PAIRED_COMMON_HORIZON",
        "interpretation": (
            "Exploratory snapshot from an incomplete prospectively fixed "
            "50-pass cohort; not a terminal comparison."
        ),
        "arms": {
            "grpo": "matched Dr.GRPO with passive verified-discovery tracking",
            "online_canonical_haarnoja": (
                "uncapped support-normalized online-canonical Haarnoja"
            ),
        },
        "seeds": [43, 44, 45],
        "sources": {
            domain: {
                "path": str(path.relative_to(ROOT)),
                "sha256_current_append_only_curve": hashlib.sha256(
                    path.read_bytes()
                ).hexdigest(),
            }
            for domain, path in SOURCES.items()
        },
        "domains": domains,
    }


def _mathir_extension_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    horizon, paired_seeds = _paired_seed_horizon(rows)
    if horizon is None:
        return {
            "status": "WAITING_FOR_FIRST_PAIRED_EVALUATION",
            "evaluation_dataset": "mathir_action_menu_v1/multi_answer",
            "is_math500": False,
            "paired_seeds": [],
        }
    endpoints: dict[str, Any] = {}
    for arm in ARMS:
        by_seed: dict[str, Any] = {}
        for seed in paired_seeds:
            values = {}
            endpoint_passes = None
            for metric in ("pass8", "mean8", "coverage8", "distinct8", "greedy"):
                metric_passes, value = _endpoint(
                    rows,
                    arm=arm,
                    seed=seed,
                    metric=metric,
                    horizon=horizon,
                )
                endpoint_passes = (
                    metric_passes if endpoint_passes is None else endpoint_passes
                )
                if abs(endpoint_passes - metric_passes) > 1e-9:
                    raise RuntimeError("MathIR evaluation endpoint mismatch")
                values[metric] = value
            by_seed[str(seed)] = {
                "training_passes": endpoint_passes,
                **values,
            }
        means = {
            metric: float(
                fmean(by_seed[str(seed)][metric] for seed in paired_seeds)
            )
            for metric in ("pass8", "mean8", "coverage8", "distinct8", "greedy")
        }
        endpoints[arm] = {"by_seed": by_seed, "mean": means}
    treatment = endpoints["online_canonical_haarnoja"]["mean"]
    control = endpoints["grpo"]["mean"]
    return {
        "status": (
            "LIVE_COMPLETE_THREE_SEED_PAIRED_HORIZON"
            if set(paired_seeds) == {43, 44, 45}
            else "LIVE_PARTIAL_PAIRED_SEEDS"
        ),
        "evaluation_dataset": "mathir_action_menu_v1/multi_answer",
        "is_math500": False,
        "native_treatment_arm": "verified_first_global_replay_canonical",
        "paired_seeds": list(paired_seeds),
        "paired_common_horizon_passes": horizon,
        "endpoints": endpoints,
        "treatment_minus_control_mean": {
            metric: treatment[metric] - control[metric]
            for metric in treatment
        },
    }


def _plot(
    source_rows: dict[str, list[dict[str, Any]]],
    *,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        nrows=len(source_rows),
        ncols=len(METRICS),
        figsize=(13.2, 7.0),
        squeeze=False,
    )
    for row_index, (domain, rows) in enumerate(source_rows.items()):
        if domain in FROZEN_HORIZONS:
            horizon = FROZEN_HORIZONS[domain]
            paired_seeds = (43, 44, 45)
        else:
            horizon, paired_seeds = _paired_seed_horizon(rows)
        if horizon is None:
            for axis in axes[row_index]:
                axis.text(
                    0.5,
                    0.5,
                    "AWAITING FIRST PAIRED EVALUATION",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#777777",
                )
                axis.set_xticks([])
                axis.set_yticks([])
            axes[row_index, 0].set_ylabel(ROW_LABELS[domain], fontsize=9)
            continue
        for column_index, (metric, title) in enumerate(METRICS):
            axis = axes[row_index, column_index]
            plotted_metric = (
                "semantic_shannon_success_conditioned_signed_open_set_next_coefficient"
                if (
                    domain == MATHIR_DOMAIN
                    and metric == "online_canonical_entropy_alpha_used"
                )
                else metric
            )
            for arm, (label, color) in ARMS.items():
                display_label = (
                    TREATMENT_LABELS.get(domain, label)
                    if arm != "grpo"
                    else label
                )
                line_style = (
                    "--" if domain == MATHIR_DOMAIN and arm != "grpo" else "-"
                )
                if plotted_metric in {
                    "online_canonical_entropy_alpha_used",
                    "semantic_shannon_success_conditioned_signed_open_set_next_coefficient",
                } and arm == "grpo":
                    continue
                for seed in paired_seeds:
                    points = _series(
                        rows,
                        arm=arm,
                        seed=seed,
                        metric=plotted_metric,
                        horizon=horizon,
                    )
                    if points:
                        axis.plot(
                            [point[0] for point in points],
                            [point[1] for point in points],
                            color=color,
                            alpha=0.22,
                            linewidth=0.8,
                            linestyle=line_style,
                        )
                mean_points = _mean_series(
                    rows,
                    arm=arm,
                    metric=plotted_metric,
                    horizon=horizon,
                    seeds=paired_seeds,
                )
                if mean_points:
                    axis.plot(
                        [point[0] for point in mean_points],
                        [point[1] for point in mean_points],
                        color=color,
                        linewidth=2.1,
                        linestyle=line_style,
                        label=display_label,
                    )
            axis.axvline(horizon, color="#888888", linestyle=":", linewidth=0.8)
            axis.set_xlim(0, max(horizon, 0.25))
            axis.grid(alpha=0.18, linewidth=0.5)
            if row_index == 0:
                axis.set_title(title, fontsize=9)
            if row_index == len(source_rows) - 1:
                axis.set_xlabel("training-pool passes", fontsize=8)
            axis.tick_params(labelsize=7)
            if column_index == 0:
                axis.set_ylabel(
                    ROW_LABELS[domain],
                    fontsize=9,
                )
                if set(paired_seeds) != {43, 44, 45}:
                    axis.text(
                        0.02,
                        0.96,
                        "LIVE PARTIAL — paired seeds "
                        + ",".join(str(seed) for seed in paired_seeds),
                        transform=axis.transAxes,
                        ha="left",
                        va="top",
                        fontsize=6.5,
                        color="#9a4f00",
                    )
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=ARMS["grpo"][1], linewidth=2.1, label="Dr.GRPO"),
        Line2D(
            [],
            [],
            color=ARMS["online_canonical_haarnoja"][1],
            linewidth=2.1,
            label="E50 Verified MaxEnt (Graph/Countdown)",
        ),
        Line2D(
            [],
            [],
            color=ARMS["online_canonical_haarnoja"][1],
            linewidth=2.1,
            linestyle="--",
            label="E59 global verified replay (MathIR)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.015),
        fontsize=8.5,
    )
    fig.suptitle(
        "ModeBench separate paired cohorts — MathIR is not MATH-500",
        y=0.97,
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.025, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "paper/figures/modebench_long_horizon_interim.png",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "paper/results/modebench_long_horizon_interim.json",
    )
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    source_rows = {domain: _load(path) for domain, path in SOURCES.items()}
    extension_rows = {
        domain: _load_mathir(path)
        for domain, path in EXTENSION_SOURCES.items()
    }
    summary = _summary(source_rows)
    summary["extensions"] = {
        MATHIR_DOMAIN: _mathir_extension_summary(extension_rows[MATHIR_DOMAIN])
    }
    summary["extension_sources"] = {
        domain: {
            "path": str(path.relative_to(ROOT)),
            "sha256_live": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for domain, path in EXTENSION_SOURCES.items()
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not args.summary_only:
        _plot({**source_rows, **extension_rows}, output=args.output)
    print(
        json.dumps(
            {
                "domains": summary["domains"],
                "extensions": summary["extensions"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
