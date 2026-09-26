#!/usr/bin/env python3
"""Render every exact terminal Qwen2.5-3B endpoint without aggregation.

The panel exposes all currently terminal core replay, ordinary-GRPO, fixed
semantic-on-replay, and adaptive-semantic-on-replay seeds.  It intentionally
draws no means, ranges, or intervals: completion is unequal across methods and
domains, so every marker is an exact registered terminal seed.
"""

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
import paper_method_style as method_visuals  # noqa: E402
import paper_style as style  # noqa: E402
from plot_paper_direct_comparator_endpoint_effects import (  # noqa: E402
    _completion_marker,
    _relative,
    _runs,
    _sha256,
    _terminal_metrics,
)


DEFAULT_OUTPUT = ROOT / "paper/figures/qwen3b_exact_endpoint_progress"
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
SEEDS = (70, 71, 72, 73, 74)
TARGET = 3072
METHODS = (
    "drgrpo",
    "grpo",
    "replay_grpo",
    "replay_semantic_maxent",
    "adaptive_semantic_replay",
)
METHOD_SHORT = {
    "drgrpo": "D",
    "grpo": "G",
    "replay_grpo": "R",
    "replay_semantic_maxent": "F",
    "adaptive_semantic_replay": "A",
}
METHOD_LABEL = {
    "drgrpo": "Dr.GRPO",
    "grpo": "GRPO",
    "replay_grpo": "Re:Dr",
    "replay_semantic_maxent": "Fixed Semantic MaxEnt + Re:Dr",
    "adaptive_semantic_replay": "Adaptive Semantic MaxEnt + Re:Dr",
}
LEDGERS = {
    "core": ROOT / "var/artifacts/e80r1_qwen3b_aligned_verified_replay_jobs.json",
    "grpo": ROOT / "var/artifacts/e95_plain_grpo_Qwen25-3B_jobs.json",
    "grpo_extension": ROOT / "var/artifacts/e114_plain_grpo_qwen3b_extension_jobs.json",
    "fixed": ROOT / "var/artifacts/e87_qwen3b_semantic_maxent_seed70_jobs.json",
    "adaptive": ROOT / "var/artifacts/e92_qwen3b_adaptive_semantic_maxent_jobs.json",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build() -> dict[str, Any]:
    ledgers = {name: _read(path) for name, path in LEDGERS.items()}
    if {int(ledger["target_steps"]) for ledger in ledgers.values()} != {TARGET}:
        raise RuntimeError("Qwen2.5-3B target-step mismatch")
    grpo_runs = _runs(ledgers["grpo"])
    grpo_extension_runs = _runs(ledgers["grpo_extension"])
    overlap = grpo_runs.keys() & grpo_extension_runs.keys()
    if overlap:
        raise RuntimeError(f"duplicate Qwen2.5-3B GRPO cells: {sorted(overlap)}")
    grpo_runs.update(grpo_extension_runs)
    run_maps = {
        "drgrpo": _runs(ledgers["core"], arm="control"),
        "replay_grpo": _runs(ledgers["core"], arm="replay"),
        "grpo": grpo_runs,
        "replay_semantic_maxent": _runs(ledgers["fixed"], arm="semantic"),
        "adaptive_semantic_replay": _runs(
            ledgers["adaptive"], arm="adaptive_semantic_reachable"
        ),
    }
    source_paths: set[Path] = set(LEDGERS.values())
    endpoint_cache: dict[Path, dict[str, float]] = {}
    cells: list[dict[str, Any]] = []
    for domain in DOMAIN_ORDER:
        methods: dict[str, Any] = {}
        for method in METHODS:
            per_seed: dict[str, Any] = {}
            for seed in SEEDS:
                run = run_maps[method].get((domain, seed))
                if run is None:
                    continue
                run_dir = Path(str(run["run_dir"]))
                marker = _completion_marker(run_dir, target=TARGET)
                if marker is None:
                    continue
                source_paths.add(marker)
                if run_dir not in endpoint_cache:
                    endpoint, paths = _terminal_metrics(run_dir, target=TARGET)
                    endpoint_cache[run_dir] = endpoint
                    source_paths.update(paths)
                per_seed[str(seed)] = {
                    **endpoint_cache[run_dir],
                    "job_id": int(run["job_id"]),
                }
            if not per_seed:
                continue
            seeds = sorted(int(seed) for seed in per_seed)
            methods[method] = {
                "label": METHOD_LABEL[method],
                "n": len(seeds),
                "seeds": seeds,
                "evidence": "exact_terminal_seeds_no_aggregate",
                "per_seed": per_seed,
            }
        cells.append({"domain": domain, "methods": methods})
    return {
        "schema": "paper-qwen3b-exact-endpoint-progress-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "all exact terminal Qwen2.5-3B endpoints at the freeze",
        "model": "Qwen2.5-3B",
        "domain_order": list(DOMAIN_ORDER),
        "methods": list(METHODS),
        "selection_rule": (
            "every registered seed with a terminal completion marker at step "
            "3072 and all four registered sampled-evaluation draws"
        ),
        "aggregation": "none; no means, ranges, intervals, or pooling",
        "metrics": {"x": "pass@8", "y": "distinct@8"},
        "cells": cells,
        "input_sha256": {
            _relative(path): {
                "byte_length": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(source_paths)
        },
    }


def render(payload: dict[str, Any], output: Path) -> None:
    style.apply_rcparams()
    figure, axes = plt.subplots(
        1,
        len(DOMAIN_ORDER),
        figsize=(style.WIDTH, 2.85),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    cells = {cell["domain"]: cell for cell in payload["cells"]}
    panel_counts: list[tuple[Any, str]] = []
    maximum = max(
        point["distinct8"]
        for cell in payload["cells"]
        for method in cell["methods"].values()
        for point in method["per_seed"].values()
    )
    high = max(2.25, math.ceil((maximum + 0.15) * 2) / 2)
    for column, domain in enumerate(DOMAIN_ORDER):
        axis = axes[0][column]
        style.style_axis(axis, grid="both", title=DOMAIN_LABEL[domain])
        axis.set_xlim(-0.03, 1.03)
        axis.set_ylim(-0.06, high)
        cell = cells[domain]
        for method in METHODS:
            record = cell["methods"].get(method)
            if record is None:
                continue
            visual = method_visuals.method_style(method)
            axis.scatter(
                [point["pass8"] for point in record["per_seed"].values()],
                [point["distinct8"] for point in record["per_seed"].values()],
                s=26,
                marker=visual["marker"],
                facecolors="none",
                edgecolors=visual["color"],
                linewidths=0.85,
                zorder=3,
            )
        counts = " · ".join(
            f"{METHOD_SHORT[method]} {cell['methods'][method]['n']}"
            for method in METHODS
            if method in cell["methods"]
        )
        panel_counts.append((axis, counts))
        axis.set_xlabel("pass@8", fontsize=style.LABEL_FONT)
        if column == 0:
            axis.set_ylabel("distinct@8", fontsize=style.LABEL_FONT)
        else:
            axis.tick_params(labelleft=False)

    # The per-method n is the same string in every panel here, so printing it
    # five times spends the top of each panel restating a constant. Hoist it
    # into the subtitle and keep a per-panel badge only where a panel differs.
    shared_counts = style.dominant_note(counts for _axis, counts in panel_counts)
    for axis, counts in panel_counts:
        if counts == shared_counts:
            continue
        axis.text(
            0.03,
            0.96,
            f"n: {counts}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=style.SMALL_FONT,
            color=style.MUTED,
        )

    handles = []
    for method in METHODS:
        visual = method_visuals.method_style(method)
        handles.append(
            Line2D(
                [0],
                [0],
                marker=visual["marker"],
                linestyle="none",
                markersize=4.5,
                markerfacecolor="none",
                markeredgecolor=visual["color"],
                label=METHOD_LABEL[method],
            )
        )
    style.bottom_legend(
        figure,
        handles,
        [handle.get_label() for handle in handles],
        y=0.003,
        ncol=5,
    )
    figure.suptitle(
        "Qwen2.5-3B exact terminal progress",
        fontsize=style.TITLE_FONT,
        color=style.INK,
        y=0.995,
    )
    figure.text(
        0.5,
        0.925,
        (
            "Every marker is one registered terminal seed; no mean or interval "
            "is drawn."
            + (f"  All panels n: {shared_counts} unless marked." if shared_counts else "")
        ),
        ha="center",
        va="top",
        fontsize=style.SMALL_FONT,
        color=style.MUTED,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.995,
        top=0.72,
        bottom=0.31,
        wspace=0.24,
    )
    style.save(figure, output, png=True, dpi=260)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    output = args.output.resolve()
    render(payload, output)
    output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {output.with_suffix('.pdf')}, {output.with_suffix('.png')}, "
        f"and {output.with_suffix('.json')}"
    )


if __name__ == "__main__":
    main()
