#!/usr/bin/env python3
"""Plot E105 versus matched Re:Dr.GRPO in the baseline forest grammar."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
import paper_method_style as method_visuals  # noqa: E402
import paper_style as style  # noqa: E402


DEFAULT_INPUT = ROOT / (
    "paper/results/e105_group_centered_semantic_repair_three_scale.json"
)
DEFAULT_OUTPUT = ROOT / (
    "paper/figures/e105_group_centered_semantic_endpoint_effects"
)
DEFAULT_AUC_OUTPUT = ROOT / (
    "paper/figures/e105_group_centered_semantic_auc_effects"
)
SCALES = ("qwen05b", "falcon1b", "qwen3b")
MODEL_LABELS = {
    "qwen05b": "Qwen 0.5B",
    "falcon1b": "Falcon 1B",
    "qwen3b": "Qwen 3B",
}
DOMAINS = (
    "graph_coloring",
    "countdown",
    "python_factors",
    "mathir",
    "pantry_plan",
)
DOMAIN_LABELS = {
    "graph_coloring": "Graph coloring",
    "countdown": "Countdown",
    "python_factors": "Python factors",
    "mathir": "MathIR",
    "pantry_plan": "PantryPlan",
}
EFFECT_SPECS = {
    "terminal": {
        "schema": "paper-e105-group-centered-semantic-endpoint-effects-v1",
        "metrics": (
            ("terminal_sampled_pass8", r"$\Delta P$"),
            ("terminal_sampled_excess8", r"$\Delta(D-P)$"),
        ),
        "title": (
            "Repaired Semantic MaxEnt on Re:Dr.GRPO: paired terminal effects"
        ),
        "descriptions": {
            "terminal_sampled_pass8": (
                "treatment terminal pass@8 minus matched Re:Dr.GRPO "
                "terminal pass@8"
            ),
            "terminal_sampled_excess8": (
                "treatment terminal (distinct@8-pass@8) minus matched "
                "Re:Dr.GRPO terminal (distinct@8-pass@8)"
            ),
        },
    },
    "auc": {
        "schema": "paper-e105-group-centered-semantic-auc-effects-v1",
        "metrics": (
            ("auc_sampled_pass8", r"$\Delta\operatorname{AUC}(P)$"),
            ("auc_sampled_excess8", r"$\Delta\operatorname{AUC}(D-P)$"),
        ),
        "title": (
            "Repaired Semantic MaxEnt on Re:Dr.GRPO: paired trajectory AUC"
        ),
        "descriptions": {
            "auc_sampled_pass8": (
                "treatment normalized pass@8 trajectory AUC minus matched "
                "Re:Dr.GRPO normalized pass@8 trajectory AUC"
            ),
            "auc_sampled_excess8": (
                "treatment normalized (distinct@8-pass@8) trajectory AUC "
                "minus matched Re:Dr.GRPO normalized "
                "(distinct@8-pass@8) trajectory AUC"
            ),
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def plot_payload(
    result: dict[str, Any],
    *,
    input_path: Path,
    effect_kind: str = "terminal",
) -> dict[str, Any]:
    if effect_kind not in EFFECT_SPECS:
        raise RuntimeError(f"unknown E105 effect kind: {effect_kind}")
    spec = EFFECT_SPECS[effect_kind]
    metrics = spec["metrics"]
    if result.get("schema") != "e105_group_centered_semantic_repair_results_v1":
        raise RuntimeError("unexpected E105 result schema")
    if result.get("pointmaze") != "excluded":
        raise RuntimeError("E105 paired forest requires PointMaze exclusion")
    if result.get("analysis_plotter") != str(Path(__file__).resolve()):
        raise RuntimeError("E105 result names another paired-effect plotter")
    if result.get("analysis_plotter_sha256") != sha256(Path(__file__)):
        raise RuntimeError("E105 result paired-effect plotter digest mismatch")
    design = result.get("design", {})
    if design.get("scales") != list(SCALES) or design.get("domains") != list(DOMAINS):
        raise RuntimeError("E105 endpoint forest design drifted")
    cells: list[dict[str, Any]] = []
    for scale in SCALES:
        for domain in DOMAINS:
            family = result["families"][scale][domain]
            seeds = [int(seed) for seed in family["paired_seeds"]]
            if len(seeds) != 5:
                raise RuntimeError(f"{scale}/{domain}: expected exact n=5")
            per_seed = {
                str(seed): {
                    metric: float(
                        family["paired_effects"][metric]["per_seed"][str(seed)]
                    )
                    for metric, _label in metrics
                }
                for seed in seeds
            }
            summaries = {
                metric: family["paired_effects"][metric]
                for metric, _label in metrics
            }
            if any(summary.get("n") != 5 for summary in summaries.values()):
                raise RuntimeError(f"{scale}/{domain}: summary n drifted")
            cells.append(
                {
                    "scale": scale,
                    "model": MODEL_LABELS[scale],
                    "domain": domain,
                    "n": 5,
                    "seeds": seeds,
                    "per_seed_effects": per_seed,
                    "summaries": summaries,
                }
            )
    if len(cells) != 15:
        raise RuntimeError(f"expected 15 E105 paired-effect cells, got {len(cells)}")
    return {
        "schema": spec["schema"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pointmaze": "excluded",
        "effect_kind": effect_kind,
        "input": str(input_path.resolve()),
        "input_sha256": sha256(input_path),
        "plotter": str(Path(__file__).resolve()),
        "plotter_sha256": sha256(Path(__file__)),
        "baseline": "matched Re:Dr.GRPO",
        "treatment": "Re:Dr.GRPO + repaired Semantic MaxEnt",
        "model_rows": list(SCALES),
        "domain_order": list(DOMAINS),
        "metric_order": [metric for metric, _label in metrics],
        "metric_labels": {metric: label for metric, label in metrics},
        "metrics": spec["descriptions"],
        "title": spec["title"],
        "evidence_encoding": {
            "paired_seed": "open circle",
            "five_seed_mean": "filled diamond",
            "interval": "paired two-sided 95% Student-t, df=4",
            "sample_size": "exact n printed in every panel",
        },
        "cells": cells,
        "registered_general_extension_criterion": result[
            "registered_general_extension_criterion"
        ],
    }


def render(payload: dict[str, Any], output: Path) -> None:
    style.apply_rcparams()
    metrics = tuple(
        (metric, payload["metric_labels"][metric])
        for metric in payload["metric_order"]
    )
    figure, axes = plt.subplots(
        3,
        5,
        figsize=(style.WIDTH, 5.35),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    visual = method_visuals.method_style("replay_semantic_maxent")
    cells = {
        (cell["scale"], cell["domain"]): cell for cell in payload["cells"]
    }
    bounds = [
        float(bound)
        for cell in payload["cells"]
        for metric, _label in metrics
        for bound in cell["summaries"][metric]["student_t_95"]
    ]
    low = math.floor((min(bounds) - 0.10) * 10.0) / 10.0
    high = math.ceil((max(bounds) + 0.10) * 10.0) / 10.0
    if math.isclose(low, high):
        low, high = low - 0.1, high + 0.1
    x_positions = (0.0, 1.0)
    jitter = (-0.045, -0.0225, 0.0, 0.0225, 0.045)
    for row_index, scale in enumerate(SCALES):
        for column_index, domain in enumerate(DOMAINS):
            axis = axes[row_index][column_index]
            style.style_axis(
                axis,
                grid="both",
                title=DOMAIN_LABELS[domain] if row_index == 0 else None,
            )
            axis.axhline(0.0, color=style.MUTED, lw=0.8, linestyle=(0, (2, 2)))
            cell = cells[(scale, domain)]
            for x, (metric, _label) in zip(x_positions, metrics):
                values = [
                    cell["per_seed_effects"][str(seed)][metric]
                    for seed in cell["seeds"]
                ]
                for offset, value in zip(jitter, values):
                    axis.scatter(
                        x + offset,
                        value,
                        s=15,
                        marker="o",
                        facecolors="white",
                        edgecolors=visual["color"],
                        linewidths=0.8,
                        zorder=3,
                    )
                summary = cell["summaries"][metric]
                interval = summary["student_t_95"]
                axis.vlines(
                    x,
                    interval[0],
                    interval[1],
                    color=visual["color"],
                    linewidth=1.25,
                    zorder=2,
                )
                axis.scatter(
                    x,
                    summary["mean"],
                    s=25,
                    marker="D",
                    color=visual["color"],
                    edgecolors="white",
                    linewidths=0.45,
                    zorder=4,
                )
            axis.text(
                0.97,
                0.96,
                "n=5",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=6.2,
                color=style.MUTED,
            )
            axis.set_xlim(-0.35, 1.35)
            axis.set_ylim(low, high)
            axis.set_xticks(x_positions)
            axis.set_xticklabels(
                [label for _metric, label in metrics]
                if row_index == len(SCALES) - 1
                else []
            )
            if column_index == 0:
                axis.set_ylabel(MODEL_LABELS[scale])

    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=visual["color"],
            markersize=4.5,
            label="paired training seed",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="-",
            color=visual["color"],
            markerfacecolor=visual["color"],
            markersize=4.5,
            label="five-seed mean and paired 95% t interval",
        ),
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.002),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(payload["title"], y=0.995)
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.965))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--auc-output", type=Path, default=DEFAULT_AUC_OUTPUT)
    args = parser.parse_args()
    result = json.loads(args.input.read_text(encoding="utf-8"))
    endpoint = plot_payload(
        result,
        input_path=args.input,
        effect_kind="terminal",
    )
    auc = plot_payload(result, input_path=args.input, effect_kind="auc")
    render(endpoint, args.output)
    render(auc, args.auc_output)
    print(
        f"wrote {args.output.with_suffix('.pdf')}, "
        f"{args.output.with_suffix('.png')}, and "
        f"{args.output.with_suffix('.json')}; "
        f"{args.auc_output.with_suffix('.pdf')}, "
        f"{args.auc_output.with_suffix('.png')}, and "
        f"{args.auc_output.with_suffix('.json')}"
    )


if __name__ == "__main__":
    main()
