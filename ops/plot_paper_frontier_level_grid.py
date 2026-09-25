#!/usr/bin/env python3
"""Fig. 11's plate, one row per hosted deployment instead of per open scale.

The scale grid answers where a single open checkpoint sits without a darker one
drawn over it. The same question is worth asking of the deployments, and the
hosted measurements support it: seven deployments at Levels 1--3, and GPT-5.6
Sol further up its own ladder. Overlaying them, as the family figures do, hides
exactly the per-deployment detail this is for.

Rows carry only the levels a deployment was actually measured at, so a short row
is a statement about coverage rather than a gap in the plate. The encoding is
Fig. 11's: horizontal is pass@8, vertical is PCMD, marker shape is the level, an
open mark is an estimate below the support bar, and a tick on the floor strip is
a cell too rarely solved to place.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))
import paper_style as style  # noqa: E402
import plot_paper_mode_diversity_levels as levels  # noqa: E402

SOURCE = ROOT / "paper/results/mode_diversity_frontier_points.json"
OUT = ROOT / "paper/figures/frontier_level_grid"
# Lowest measured hosted pass@8 is .898; start below it so the
# window is a zoom rather than a crop.
X_WINDOW = (0.888, 1.004)


def load(path: Path = SOURCE):
    payload = json.loads(path.read_text())
    if payload.get("schema") != "mode-diversity-frontier-points-v1":
        raise ValueError("unexpected frontier point schema")
    cells = []
    for cell in payload["cells"]:
        # _panel reads the open-grid shape; a deployment name stands in for the
        # scale label, and a cell with no defined-prompt count cannot be drawn
        # as provisional, so it falls to the floor strip rather than inventing
        # support it never reported.
        cells.append({**cell,
                      "model_label": cell["model"],
                      "defined_prompts": cell.get("defined_prompts", 0)})
    return payload, cells


def balanced_average(cells, name, ladder):
    """Domain-averaged PCMD per level, on a domain set fixed for the deployment.

    A mean over whichever domains happen to clear the support bar at each level
    would let the domain mix move with the level, which is the thing the plate
    is trying to read. So the average uses only the domains this deployment
    reports at every level it covers.
    """
    mine = [c for c in cells if c["model_label"] == name and c.get("reportable")]
    covered = [l for l in ladder if any(c["level"] == l for c in mine)]
    keep = [d for d in levels.DOMAINS
            if all(any(c["level"] == l and c["domain"] == d for c in mine)
                   for l in covered)]
    out = {}
    for level in covered:
        vals = [c["pmd"] for c in mine
                if c["level"] == level and c["domain"] in keep]
        if vals:
            out[level] = sum(vals) / len(vals)
    return out, keep


def build(cells):
    """PCMD against level: the domain average first, then each domain.

    The ladder is the x axis because the claim is about it. Correctness is not
    drawn -- every hosted cell sits above pass@8 .898, so plotting it would
    spend the width on a constant and hide the slope that matters.
    """
    order = {}
    for name in {c["model_label"] for c in cells}:
        mine = [c["pmd"] for c in cells
                if c["model_label"] == name and c.get("reportable")]
        order[name] = sum(mine) / len(mine) if mine else 0.0
    # Darkest is most collapsed, so colour runs the same way as height.
    deployments = sorted(order, key=lambda n: -order[n])
    ramp = style.sequential_cmap()
    colors = {name: ramp(0.92 - 0.55 * i / max(len(deployments) - 1, 1))
              for i, name in enumerate(deployments)}

    style.apply_rcparams()
    # Only the rungs every deployment covers. Levels 4 and 5 exist for one
    # deployment alone, and a line that runs further than its neighbours reads
    # as a deployment holding up where the others fall, when it is really the
    # only one measured there. A rung joins when the cohort covers it.
    ladder = [l for l in levels.levels_in(cells)
              if all(any(c["model_label"] == name and c["level"] == l
                         for c in cells) for name in deployments)]
    positions = {level: i for i, level in enumerate(ladder)}
    panels = ["average"] + list(levels.DOMAINS)
    figure, axes = plt.subplots(1, len(panels), figsize=(style.WIDTH, 1.92),
                                sharey=True, squeeze=False)
    # Default margins spend a fifth of the canvas on white; the panels are the
    # point, so they take the width back.
    figure.subplots_adjust(left=.072, right=.996, bottom=.20, top=.88,
                           wspace=.10)
    # Everything drawn sets the ceiling, not just the reportable cells: the
    # open marks below the support bar are plotted too, and a ceiling taken
    # from the reportable subset would put any high provisional estimate
    # outside the axis -- silently, and only for the arms the support bar
    # happens to thin.
    ceiling = max(c["pmd"] for c in cells
                  if c.get("pmd") is not None
                  and (c.get("reportable") or c.get("defined_prompts")))
    kept = {}

    for column, panel in enumerate(panels):
        axis = axes[0][column]
        average = panel == "average"
        axis.set_facecolor(style.PANEL if average
                           else levels.DOMAIN_BACKGROUNDS[panel])
        for name in deployments:
            if average:
                series, keep = balanced_average(cells, name, ladder)
                kept[name] = keep
                xs = [positions[l] for l in ladder if l in series]
                ys = [series[l] for l in ladder if l in series]
                dashed = []
            else:
                mine = {c["level"]: c for c in cells if c["model_label"] == name
                        and c["domain"] == panel}
                xs = [positions[l] for l in ladder
                      if l in mine and mine[l].get("reportable")]
                ys = [mine[l]["pmd"] for l in ladder
                      if l in mine and mine[l].get("reportable")]
                dashed = [(positions[l], c["pmd"]) for l, c in mine.items()
                          if not c.get("reportable")]
            if xs:
                axis.plot(xs, ys, color=colors[name],
                          linewidth=1.6 if average else 1.05,
                          marker="o", markersize=3.2 if average else 2.6,
                          markeredgecolor="white", markeredgewidth=.35,
                          alpha=.95, zorder=3)
            # A level measured but below the support bar keeps its place and
            # stays off the line, so a gap reads as a gap.
            for x, y in dashed:
                axis.scatter(x, y, s=10, facecolors="none",
                             edgecolors=colors[name], linewidths=.65,
                             alpha=.9, zorder=2)
        axis.set_title("Average" if average else levels.TITLES[panel],
                       fontsize=8.5, pad=4)
        axis.set_xticks(list(positions.values()))
        axis.set_xticklabels([l[len("level"):] for l in ladder], fontsize=7.0)
        # Three rungs in a narrow column: keep the padding small so the
        # ladder uses the width it has.
        axis.set_xlim(-0.22, len(ladder) - 0.78)
        axis.set_ylim(0, ceiling * 1.06)
        style.style_axis(axis)
        axis.tick_params(labelleft=column == 0, labelsize=7.0)

    axes[0][0].set_ylabel("Pairwise correct-mode\ndiversity", fontsize=8.0,
                          linespacing=.95)
    figure.supxlabel("ModeBench level", fontsize=8.5, color=style.INK, y=0.005)
    style.bottom_legend(
        figure,
        [Line2D([], [], color=colors[n], linewidth=1.6, marker="o",
                markersize=3.2, markeredgecolor="white", markeredgewidth=.35)
         for n in deployments],
        deployments, y=-0.30, ncol=len(deployments))
    return figure, deployments, kept, ladder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()

    payload, cells = load(args.source)
    figure, deployments, kept, drawn_levels = build(cells)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    style.save(figure, args.output)
    plt.close(figure)

    coverage = {}
    for name in deployments:
        mine = [c for c in cells if c["model_label"] == name]
        coverage[name] = {
            "levels": sorted({c["level"] for c in mine}),
            "cells": len(mine),
            "reportable": sum(1 for c in mine if c.get("reportable")),
        }
    args.output.with_suffix(".json").write_text(json.dumps({
        "schema": "paper-frontier-level-grid-v1",
        "source": {"path": str(args.source.relative_to(ROOT)),
                   "sha256": hashlib.sha256(args.source.read_bytes()).hexdigest()},
        "builder": {"path": str(Path(__file__).resolve().relative_to(ROOT)),
                    "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "deployments": deployments,
        "coverage": coverage,
        "levels_drawn": drawn_levels,
        "levels_present_but_not_drawn": [l for l in levels.levels_in(cells)
                                         if l not in drawn_levels],
        "levels_excluded_note": "a rung is drawn only where every deployment "
                                "is measured; Levels 4 and 5 exist for one "
                                "deployment alone and are reported in their "
                                "own paragraphs instead.",
        "average_domain_sets": kept,
        "encoding_note": "PCMD against level; correctness is not drawn because "
                         "every hosted cell sits above pass@8 .898.",
        "encoding": "pass@8 horizontal, PCMD vertical, marker shape is the "
                    "level; open marks sit below the support bar and floor "
                    "ticks are cells too rarely solved to place.",
        "outputs": {
            extension: {
                "path": args.output.with_suffix("." + extension)
                        .relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(
                    args.output.with_suffix("." + extension).read_bytes()).hexdigest(),
            }
            for extension in ("pdf", "png")
        },
    }, indent=1) + "\n")
    print(json.dumps({"event": "built", "deployments": len(deployments),
                      "cells": len(cells)}))


if __name__ == "__main__":
    main()
