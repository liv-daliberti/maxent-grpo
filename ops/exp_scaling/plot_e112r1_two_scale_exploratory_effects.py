#!/usr/bin/env python3
"""Render the verified-support two-scale endpoint forest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
sys.path.insert(0, str(ROOT / "ops" / "exp_scaling"))
import paper_method_style as method_visuals  # noqa: E402
import paper_style as style  # noqa: E402
from plot_e105_group_centered_semantic_endpoint_effects import sha256  # noqa: E402


DEFAULT_INPUT = ROOT / "paper/results/e112r1_two_scale_exploratory_results.json"
DEFAULT_OUTPUT = ROOT / "paper/figures/verified_support_discovery_two_scale_effects"
SCALES = ("qwen05b", "falcon1b")
MODEL_LABELS = {"qwen05b": "Qwen 0.5B", "falcon1b": "Falcon 1B"}
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
        "schema": "paper-verified-support-two-scale-endpoint-effects-v1",
        "metrics": (
            ("terminal_sampled_pass8", r"$\Delta P$"),
            ("terminal_sampled_excess8", r"$\Delta(D-P)$"),
        ),
        "title": "Verified-support Semantic-MaxEnt + Re:Dr.GRPO",
    },
}


def plot_payload(
    result: dict[str, Any], *, input_path: Path, effect_kind: str
) -> dict[str, Any]:
    if effect_kind not in EFFECT_SPECS:
        raise RuntimeError(f"unknown effect kind: {effect_kind}")
    if result.get("schema") != "e112r1_two_scale_exploratory_results_v1":
        raise RuntimeError("unexpected two-scale verified-support result schema")
    if result.get("pointmaze") != "excluded":
        raise RuntimeError("PointMaze exclusion drifted")
    disclosure = result.get("disclosure", {})
    if (
        disclosure.get("confirmatory") is not False
        or disclosure.get("continuous_outcome_blindness") is not False
        or disclosure.get("campaign_mutation_allowed_from_outcomes") is not False
    ):
        raise RuntimeError("two-scale comparison integrity metadata drifted")
    estimand = result.get("estimand_scope", {})
    if (
        estimand.get("kind") != "bundled historical-comparator contrast"
        or estimand.get("isolated_semantic_v7_effect") is not False
        or estimand.get("confirmatory_blind") is not False
    ):
        raise RuntimeError("bundled historical estimand disclosure drifted")
    design = result.get("design", {})
    exclusions = design.get("integrity_exclusions", [])
    if (
        design.get("scales") != list(SCALES)
        or design.get("domains") != list(DOMAINS)
        or design.get("frozen_cells") != 50
        or design.get("cells") != 49
        or design.get("families") != 10
        or design.get("qwen3b_cells_included") != 0
        or len(exclusions) != 1
        or (
            exclusions[0].get("scale"),
            exclusions[0].get("domain"),
            exclusions[0].get("seed"),
        )
        != ("falcon1b", "countdown", 59)
        or exclusions[0].get("outcome_value_selected") is not False
    ):
        raise RuntimeError("two-scale integrity-valid design drifted")
    excluded = {
        (str(row["scale"]), str(row["domain"]), int(row["seed"]))
        for row in exclusions
    }
    provenance = result.get("analysis_provenance", {})
    if provenance.get("analysis_plotter") != str(Path(__file__).resolve()):
        raise RuntimeError("result names another plotter")
    if provenance.get("analysis_plotter_sha256") != sha256(Path(__file__)):
        raise RuntimeError("plotter digest drifted")

    spec = EFFECT_SPECS[effect_kind]
    cells: list[dict[str, Any]] = []
    for scale in SCALES:
        for domain in DOMAINS:
            family = result["families"][scale][domain]
            seeds = [int(seed) for seed in family["paired_seeds"]]
            expected_seeds = [
                int(seed)
                for seed in design["paired_seeds"][scale]
                if (scale, domain, int(seed)) not in excluded
            ]
            if seeds != expected_seeds or len(seeds) not in (4, 5):
                raise RuntimeError(f"{scale}/{domain}: paired seeds drifted")
            summaries = {
                metric: family["paired_effects"][metric]
                for metric, _label in spec["metrics"]
            }
            if any(summary.get("n") != len(seeds) for summary in summaries.values()):
                raise RuntimeError(f"{scale}/{domain}: summary n drifted")
            cells.append(
                {
                    "scale": scale,
                    "model": MODEL_LABELS[scale],
                    "domain": domain,
                    "n": len(seeds),
                    "seeds": seeds,
                    "per_seed_effects": {
                        str(seed): {
                            metric: float(summary["per_seed"][str(seed)])
                            for metric, summary in summaries.items()
                        }
                        for seed in seeds
                    },
                    "summaries": summaries,
                }
            )
    return {
        "schema": spec["schema"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "effect_kind": effect_kind,
        "title": spec["title"],
        "subtitle": (
            "Paired effects vs Re:Dr.GRPO; Qwen 0.5B n=25, Falcon 1B n=24; "
            "Qwen 3B not analyzed"
        ),
        "input": str(input_path.resolve()),
        "input_sha256": sha256(input_path),
        "plotter": str(Path(__file__).resolve()),
        "plotter_sha256": sha256(Path(__file__)),
        "pointmaze": "excluded",
        "treatment": "verified-support Semantic-MaxEnt + Re:Dr.GRPO",
        "baseline": "Re:Dr.GRPO",
        "model_rows": list(SCALES),
        "domain_order": list(DOMAINS),
        "metric_order": [metric for metric, _label in spec["metrics"]],
        "metric_labels": {metric: label for metric, label in spec["metrics"]},
        "cells": cells,
        "estimand_scope": {
            "kind": "bundled verified-support proposal-pressure-replay contrast",
            "component_isolated": False,
            "models": ["Qwen2.5-0.5B", "Falcon3-1B"],
            "pair_count": 49,
            "qwen3b_analyzed": False,
        },
        "evidence_encoding": {
            "paired_seed": "open circle",
            "family_mean": "filled diamond",
            "interval": "paired two-sided 95% Student-t, df=n-1",
            "sample_size": "exact n printed in every panel",
        },
    }


def render(payload: dict[str, Any], output: Path) -> None:
    style.apply_rcparams()
    metrics = tuple(
        (metric, payload["metric_labels"][metric])
        for metric in payload["metric_order"]
    )
    figure, axes = plt.subplots(
        len(SCALES),
        len(DOMAINS),
        figsize=(style.WIDTH, 4.35),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    visual = method_visuals.method_style("replay_semantic_maxent")
    cells = {(row["scale"], row["domain"]): row for row in payload["cells"]}
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
            jitter = tuple(
                (index - (cell["n"] - 1) / 2.0) * 0.0225
                for index in range(cell["n"])
            )
            for x, (metric, _label) in zip(x_positions, metrics):
                values = [
                    cell["per_seed_effects"][str(seed)][metric]
                    for seed in cell["seeds"]
                ]
                axis.scatter(
                    [x + offset for offset in jitter],
                    values,
                    s=15,
                    marker="o",
                    facecolors="white",
                    edgecolors=visual["color"],
                    linewidths=0.8,
                    zorder=3,
                )
                summary = cell["summaries"][metric]
                axis.vlines(
                    x,
                    summary["student_t_95"][0],
                    summary["student_t_95"][1],
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
                f"n={cell['n']}",
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

    figure.legend(
        handles=[
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
                label="family mean and paired 95% t interval",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.002),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(payload["title"], y=0.995)
    figure.text(
        0.5,
        0.955,
        payload["subtitle"],
        ha="center",
        va="top",
        fontsize=style.SMALL_FONT,
        color=style.MUTED,
    )
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = json.loads(args.input.read_text(encoding="utf-8"))
    endpoint = plot_payload(result, input_path=args.input, effect_kind="terminal")
    render(endpoint, args.output)
    print(
        f"wrote {args.output.with_suffix('.pdf')}, "
        f"{args.output.with_suffix('.png')}, and "
        f"{args.output.with_suffix('.json')}"
    )


if __name__ == "__main__":
    main()
