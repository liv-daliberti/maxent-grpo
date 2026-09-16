#!/usr/bin/env python3
"""Render paired, training-wide Re:Dr effects for Qwen2.5-0.5B.

The terminal E78 record already contains the preregistered trapezoidal AUC
through every half-pass checkpoint from zero through eight passes.  This view
turns those stored estimands into a compact effect forest: exact paired seeds,
the paired mean, and a two-sided Student-t interval are shown separately for
sampling correctness, raw verified breadth, and correctness-adjusted breadth.
No domain is pooled and no checkpoint is selected post hoc.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
import paper_style as style  # noqa: E402


SOURCE = ROOT / "paper/results/e78_terminal_05b.json"
DEFAULT_OUTPUT = ROOT / "paper/figures/sustained_auc_effects_qwen05b"
DOMAIN_ORDER = (
    "graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan",
)
DOMAIN_LABEL = {
    "graph_coloring": "Graph coloring",
    "countdown": "Countdown",
    "python_factors": "Python factors",
    "mathir": "MathIR",
    "pantry_plan": "PantryPlan",
}
SEEDS = (43, 44, 45, 46, 47)
METRICS = (
    "normalized_auc_pass8",
    "normalized_auc_distinct8",
    "normalized_auc_adjusted_breadth8",
)
METRIC_LABEL = {
    "normalized_auc_pass8": "correctness AUC",
    "normalized_auc_distinct8": "distinct-mode AUC",
    "normalized_auc_adjusted_breadth8": r"adjusted AUC ($D-P$)",
}
METRIC_TICK = {
    "normalized_auc_pass8": "correctness",
    "normalized_auc_distinct8": "distinct modes",
    "normalized_auc_adjusted_breadth8": r"adjusted $D-P$",
}
METRIC_VISUAL = {
    "normalized_auc_pass8": {"color": style.METRIC, "marker": "o"},
    "normalized_auc_distinct8": {"color": style.METHOD, "marker": "s"},
    "normalized_auc_adjusted_breadth8": {
        "color": style.ADD_ON,
        "marker": "D",
    },
}
T_CRIT_DF4 = 2.7764451051977987


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _interval(per_seed: dict[str, float]) -> dict[str, Any]:
    if list(per_seed) != [str(seed) for seed in SEEDS]:
        raise RuntimeError(f"unexpected paired seed set: {list(per_seed)}")
    values = list(per_seed.values())
    mean = statistics.fmean(values)
    half_width = T_CRIT_DF4 * statistics.stdev(values) / math.sqrt(len(values))
    return {
        "mean": mean,
        "student_t_95": [mean - half_width, mean + half_width],
        "range": [min(values), max(values)],
        "per_seed": per_seed,
    }


def _same(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def _validate_source(source: dict[str, Any]) -> None:
    design = source.get("design", {})
    if (
        source.get("schema") != "e78_terminal_paper_results_v1"
        or design.get("model") != "Qwen2.5-0.5B-Instruct"
        or design.get("domains") != list(DOMAIN_ORDER)
        or design.get("arms") != ["control", "replay"]
        or design.get("paired_seeds") != list(SEEDS)
        or design.get("evaluation_draws") != 4
        or design.get("checkpoint_interval_steps") != 192
        or design.get("target_steps") != 3072
        or design.get("passes") != 8
    ):
        raise RuntimeError("E78 terminal/AUC source contract drifted")
    if list(source.get("domains", {})) != list(DOMAIN_ORDER):
        raise RuntimeError("E78 domain order drifted")


def build() -> dict[str, Any]:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    _validate_source(source)
    cells: list[dict[str, Any]] = []
    for domain in DOMAIN_ORDER:
        paired = source["domains"][domain].get("paired_effects", {})
        pass_auc = paired.get("normalized_auc_pass8", {})
        distinct_auc = paired.get("normalized_auc_distinct8", {})
        expected_keys = [str(seed) for seed in SEEDS]
        if (
            list(pass_auc.get("per_seed", {})) != expected_keys
            or list(distinct_auc.get("per_seed", {})) != expected_keys
        ):
            raise RuntimeError(f"{domain}: incomplete normalized-AUC seed block")
        pass_record = _interval(
            {key: float(pass_auc["per_seed"][key]) for key in expected_keys}
        )
        distinct_record = _interval(
            {key: float(distinct_auc["per_seed"][key]) for key in expected_keys}
        )
        for generated, frozen, metric in (
            (pass_record, pass_auc, "pass@8"),
            (distinct_record, distinct_auc, "distinct@8"),
        ):
            if (
                not _same(generated["mean"], frozen.get("mean"))
                or any(
                    not _same(left, right)
                    for left, right in zip(
                        generated["student_t_95"],
                        frozen.get("student_t_95", []),
                    )
                )
            ):
                raise RuntimeError(f"{domain}: frozen {metric} AUC summary drifted")
        adjusted_record = _interval(
            {
                key: distinct_record["per_seed"][key] - pass_record["per_seed"][key]
                for key in expected_keys
            }
        )
        cells.append(
            {
                "domain": domain,
                "n": len(SEEDS),
                "seeds": list(SEEDS),
                "evidence": "balanced_five_seed_full_trajectory",
                "effects": {
                    "normalized_auc_pass8": pass_record,
                    "normalized_auc_distinct8": distinct_record,
                    "normalized_auc_adjusted_breadth8": adjusted_record,
                },
            }
        )
    return {
        "schema": "paper-sustained-auc-effects-qwen05b-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "balanced five-seed full-trajectory effect forest",
        "model": "Qwen2.5-0.5B-Instruct",
        "comparison": "Re:Dr minus matched Dr.GRPO",
        "domain_order": list(DOMAIN_ORDER),
        "paired_seeds": list(SEEDS),
        "metrics": {
            "normalized_auc_pass8": (
                "paired effect in pass@8 trajectory AUC normalized by eight passes"
            ),
            "normalized_auc_distinct8": (
                "paired effect in distinct@8 trajectory AUC normalized by eight passes"
            ),
            "normalized_auc_adjusted_breadth8": (
                "per-seed normalized_auc_distinct8 minus normalized_auc_pass8"
            ),
        },
        "auc_definition": source["auc"],
        "uncertainty": source["uncertainty"],
        "selection_rule": (
            "all 17 registered half-pass checkpoints from step 0 through 3072, "
            "all four sampled-evaluation draws, and all five paired seeds"
        ),
        "cells": cells,
        "source_json": str(SOURCE.relative_to(ROOT)),
        "source_sha256": _sha256(SOURCE),
    }


def render(payload: dict[str, Any], output: Path) -> None:
    style.apply_rcparams()
    figure, axes = plt.subplots(
        1,
        len(DOMAIN_ORDER),
        figsize=(style.WIDTH, 2.9),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    cells = {cell["domain"]: cell for cell in payload["cells"]}
    y_positions = {
        "normalized_auc_pass8": 2.0,
        "normalized_auc_distinct8": 1.0,
        "normalized_auc_adjusted_breadth8": 0.0,
    }
    jitter = (-0.08, -0.04, 0.0, 0.04, 0.08)
    plotted = [0.0]
    for cell in payload["cells"]:
        for record in cell["effects"].values():
            plotted.extend(record["student_t_95"])
            plotted.extend(record["per_seed"].values())
    x_low = min(-0.25, math.floor((min(plotted) - 0.08) * 4) / 4)
    x_high = max(1.0, math.ceil((max(plotted) + 0.08) * 4) / 4)

    for column, domain in enumerate(DOMAIN_ORDER):
        axis = axes[0][column]
        style.style_axis(axis, grid="both", title=DOMAIN_LABEL[domain])
        axis.axvline(0.0, color=style.MUTED, linewidth=0.75, linestyle=(0, (2, 2)))
        axis.set_xlim(x_low, x_high)
        axis.set_ylim(-0.55, 2.55)
        cell = cells[domain]
        for metric in METRICS:
            visual = METRIC_VISUAL[metric]
            record = cell["effects"][metric]
            values = [record["per_seed"][str(seed)] for seed in SEEDS]
            y = y_positions[metric]
            axis.scatter(
                values,
                [y + offset for offset in jitter],
                s=12,
                marker=visual["marker"],
                facecolors="none",
                edgecolors=visual["color"],
                linewidths=0.68,
                zorder=3,
            )
            low, high = record["student_t_95"]
            axis.plot(
                [low, high],
                [y, y],
                color=visual["color"],
                linewidth=1.2,
                zorder=4,
            )
            axis.scatter(
                [record["mean"]],
                [y],
                s=25,
                marker="D",
                facecolors=visual["color"],
                edgecolors=style.WHITE,
                linewidths=0.45,
                zorder=5,
            )
        axis.set_xlabel(r"$\Delta$ normalized AUC", fontsize=style.LABEL_FONT)
        axis.set_yticks(
            [y_positions[metric] for metric in METRICS],
            [METRIC_TICK[metric] for metric in METRICS],
        )
        if column != 0:
            axis.tick_params(labelleft=False)

    metric_handles = [
        Line2D(
            [0],
            [0],
            marker=METRIC_VISUAL[metric]["marker"],
            linestyle="none",
            markersize=4,
            markerfacecolor="none",
            markeredgecolor=METRIC_VISUAL[metric]["color"],
            label=METRIC_LABEL[metric],
        )
        for metric in METRICS
    ]
    evidence_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=3.5,
            markerfacecolor="none", markeredgecolor=style.MUTED,
            label="paired seed",
        ),
        Line2D(
            [0, 1], [0, 0], color=style.MUTED, marker="D",
            linewidth=1.0, markersize=3.5, label="mean + paired 95% interval",
        ),
    ]
    handles = metric_handles + evidence_handles
    style.bottom_legend(
        figure,
        handles,
        [handle.get_label() for handle in handles],
        y=0.003,
        ncol=5,
    )
    figure.suptitle(
        "Sustained Re:Dr (ours) effects through training",
        fontsize=style.TITLE_FONT,
        color=style.INK,
        y=0.995,
    )
    figure.text(
        0.5,
        0.925,
        (
            "Re:Dr (ours) − matched Dr.GRPO · trapezoidal AUC over all 17 "
            "checkpoints from 0–8 passes; paired n=5; no domain pooling."
        ),
        ha="center",
        va="top",
        fontsize=style.SMALL_FONT,
        color=style.MUTED,
    )
    figure.subplots_adjust(
        left=0.105,
        right=0.995,
        top=0.82,
        bottom=0.25,
        wspace=0.22,
    )
    style.save(figure, output, png=True, dpi=260)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    payload = build()
    render(payload, output)
    output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {output.with_suffix('.pdf')}, {output.with_suffix('.png')}, "
        f"and {output.with_suffix('.json')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
