#!/usr/bin/env python3
"""Render the pass@K companion to the frozen full-x-Mode breadth figure.

The source figure's JSON already contains paired pass@K summaries for every
displayed K.  This renderer reads that frozen record rather than reopening live
logs, preserves its invariant 3x5 model/domain grid and checkpoint labels, and
writes a metric-specific evidence record beside the PDF and PNG.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
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


DEFAULT_SOURCE = (
    ROOT / "paper/figures/xmode_adaptive_cross_scale_distinct_at_k.json"
)
DEFAULT_OUTPUT = ROOT / "paper/figures/xmode_adaptive_cross_scale_pass_at_k"
ROW_ORDER = ("qwen05b", "falcon1b", "qwen3b")
ROW_LABEL = {
    "qwen05b": "Qwen2.5-0.5B",
    "falcon1b": "Falcon3-1B",
    "qwen3b": "Qwen2.5-3B",
}
DOMAIN_ORDER = (
    "graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan",
)
DOMAIN_LABEL = {
    "graph_coloring": "Graph",
    "countdown": "Countdown",
    "python_factors": "Python",
    "mathir": "MathIR",
    "pantry_plan": "Pantry",
}
KS = (1, 2, 4, 8, 16, 32)
SOURCE_METHOD = {
    "control": "drgrpo",
    "xmode": "adaptive_semantic_replay",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_source(source: dict[str, Any]) -> None:
    if source.get("schema") != "xmode-adaptive-cross-scale-frontier-v1":
        raise RuntimeError("unexpected x-Mode frontier schema")
    if source.get("row_order") != list(ROW_ORDER):
        raise RuntimeError("x-Mode frontier model rows drifted")
    if source.get("domain_order") != list(DOMAIN_ORDER):
        raise RuntimeError("x-Mode frontier domain columns drifted")
    if source.get("ks") != list(KS):
        raise RuntimeError("x-Mode frontier sample budgets drifted")
    if source.get("methods") != ["drgrpo", "adaptive_semantic_replay"]:
        raise RuntimeError("x-Mode frontier methods drifted")
    cells = source.get("cells")
    if not isinstance(cells, dict) or set(cells) != set(ROW_ORDER):
        raise RuntimeError("x-Mode frontier cell rows are incomplete")
    for row in ROW_ORDER:
        row_cells = cells[row]
        if not isinstance(row_cells, dict):
            raise RuntimeError(f"{row}: malformed cell mapping")
        if set(row_cells) != set(DOMAIN_ORDER):
            raise RuntimeError(f"{row}: incomplete static-domain grid")
        for domain, cell in row_cells.items():
            seeds = cell.get("seeds", [])
            if cell.get("n") != len(seeds) or not seeds:
                raise RuntimeError(f"{row}/{domain}: invalid constant seed set")
            for source_method in SOURCE_METHOD:
                summaries = cell.get("summaries", {}).get(source_method, {})
                if set(summaries) != {str(k) for k in KS}:
                    raise RuntimeError(
                        f"{row}/{domain}/{source_method}: incomplete K grid"
                    )
                for k in KS:
                    record = summaries[str(k)].get("pass8", {})
                    per_seed = record.get("per_seed", {})
                    if set(per_seed) != {str(seed) for seed in seeds}:
                        raise RuntimeError(
                            f"{row}/{domain}/{source_method}/K={k}: "
                            "seed set drifted"
                        )
                    values = [float(per_seed[str(seed)]) for seed in seeds]
                    if not all(0.0 <= value <= 1.0 for value in values):
                        raise RuntimeError(
                            f"{row}/{domain}/{source_method}/K={k}: "
                            "pass@K is outside [0,1]"
                        )


def build_record(source_path: Path) -> dict[str, Any]:
    source_path = source_path.resolve()
    source = _read(source_path)
    _validate_source(source)
    cells: dict[str, dict[str, Any]] = {}
    for row in ROW_ORDER:
        cells[row] = {}
        for domain, cell in source["cells"][row].items():
            methods: dict[str, Any] = {}
            for source_method, method in SOURCE_METHOD.items():
                methods[method] = {
                    "label": (
                        "matched Dr.GRPO"
                        if method == "drgrpo"
                        else "x-Mode Dr.GRPO (replay + adaptive semantic MaxEnt)"
                    ),
                    "by_k": {
                        str(k): cell["summaries"][source_method][str(k)]["pass8"]
                        for k in KS
                    },
                }
            cells[row][domain] = {
                "scale": row,
                "scale_label": cell["scale_label"],
                "domain": domain,
                "seeds": cell["seeds"],
                "n": cell["n"],
                "checkpoint_step": cell["checkpoint_step"],
                "checkpoint_training_pass": cell["checkpoint_training_pass"],
                "terminal_checkpoint": cell["terminal_checkpoint"],
                "methods": methods,
            }
    return {
        "schema": "paper-xmode-pass-at-k-companion-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": source["status"],
        "metric": "pass@K",
        "treatment": source["treatment"],
        "methods": ["drgrpo", "adaptive_semantic_replay"],
        "row_order": list(ROW_ORDER),
        "domain_order": list(DOMAIN_ORDER),
        "ks": list(KS),
        "blank_rule": source["blank_rule"],
        "sample_reuse": source["sample_reuse"],
        "selection_rule": source["selection_rule"],
        "source_json": str(source_path.relative_to(ROOT)),
        "source_sha256": _sha256(source_path),
        "cells": cells,
    }


def render(record: dict[str, Any], output: Path) -> None:
    style.apply_rcparams()
    figure, axes = plt.subplots(
        len(ROW_ORDER), len(DOMAIN_ORDER),
        figsize=(style.WIDTH, 6.8), sharex=True, sharey=True, squeeze=False,
    )
    visuals = {
        method: method_visuals.method_style(method)
        for method in record["methods"]
    }
    def cell_note(cell):
        status = "terminal" if cell["terminal_checkpoint"] else "progress"
        return (
            f"n={cell['n']} \u00b7 {cell['checkpoint_training_pass']:g}p"
            f" \u00b7 {status}"
        )

    shared_note = style.dominant_note(
        cell_note(cell)
        for row_cells in record["cells"].values()
        for cell in row_cells.values()
    )
    for row_index, row in enumerate(ROW_ORDER):
        for column, domain in enumerate(DOMAIN_ORDER):
            axis = axes[row_index][column]
            style.style_axis(
                axis,
                grid="both",
                title=DOMAIN_LABEL[domain] if row_index == 0 else None,
            )
            axis.set_xscale("log", base=2)
            axis.set_xlim(1, 32)
            axis.set_xticks(KS, [str(k) for k in KS])
            axis.set_ylim(-0.03, 1.03)
            axis.set_yticks((0.0, 0.25, 0.5, 0.75, 1.0))
            cell = record["cells"][row].get(domain)
            if cell is not None:
                for method in record["methods"]:
                    visual = visuals[method]
                    by_k = cell["methods"][method]["by_k"]
                    means = [by_k[str(k)]["mean"] for k in KS]
                    lows = [by_k[str(k)]["range"][0] for k in KS]
                    highs = [by_k[str(k)]["range"][1] for k in KS]
                    axis.fill_between(
                        KS,
                        lows,
                        highs,
                        color=visual["color"],
                        alpha=style.BAND_ALPHA,
                        linewidth=0,
                        zorder=1,
                    )
                    axis.plot(
                        KS,
                        means,
                        color=visual["color"],
                        linestyle=visual["linestyle"],
                        linewidth=style.MEAN_LW,
                        marker=visual["marker"],
                        markersize=2.7,
                        markeredgewidth=0.45,
                        zorder=3,
                    )
                note = cell_note(cell)
                if note != shared_note:
                    axis.text(
                        0.03,
                        0.96,
                        note,
                        transform=axis.transAxes,
                        ha="left",
                        va="top",
                        fontsize=style.SMALL_FONT,
                        color=style.MUTED,
                    )
            if column == 0:
                axis.set_ylabel(
                    f"{ROW_LABEL[row]}\nmean pass@K",
                    fontsize=style.LABEL_FONT,
                )
            else:
                axis.tick_params(labelleft=False)
            if row_index == len(ROW_ORDER) - 1:
                axis.set_xlabel("samples K", fontsize=style.LABEL_FONT)

    handles = [
        Line2D(
            [0],
            [0],
            color=visuals[method]["color"],
            linestyle=visuals[method]["linestyle"],
            linewidth=style.MEAN_LW,
            marker=visuals[method]["marker"],
            markersize=3,
        )
        for method in record["methods"]
    ]
    style.bottom_legend(
        figure,
        handles,
        [
            "matched Dr.GRPO",
            "x-Mode Dr.GRPO (replay + adaptive semantic MaxEnt)",
        ],
        y=0.003,
        ncol=2,
    )
    figure.suptitle(
        "Sampling correctness for full x-Mode Dr.GRPO",
        fontsize=style.TITLE_FONT,
        color=style.INK,
        y=0.995,
    )
    figure.text(
        0.5,
        0.96,
        (
            "Pass@K companion to verified-mode breadth; rows: Qwen2.5-0.5B · "
            "Falcon3-1B · Qwen2.5-3B; unavailable cells are blank."
            + (f"  All panels {shared_note} unless marked." if shared_note else "")
        ),
        ha="center",
        va="top",
        fontsize=style.SMALL_FONT,
        color=style.MUTED,
    )
    figure.subplots_adjust(
        left=0.105,
        right=0.995,
        top=0.90,
        bottom=0.105,
        hspace=0.30,
        wspace=0.24,
    )
    style.save(figure, output, png=True, dpi=260)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    record = build_record(args.source)
    output = args.output.resolve()
    render(record, output)
    output.with_suffix(".json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {output.with_suffix('.pdf')}, {output.with_suffix('.png')}, "
        f"and {output.with_suffix('.json')}"
    )


if __name__ == "__main__":
    main()
