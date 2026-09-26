#!/usr/bin/env python3
"""Render the E72 decoding frontier: breadth against accuracy, parameterized by temperature.

Form. The question is a trade-off, not a time series: for each arm, sweeping
decoding temperature traces a curve of (accuracy, breadth) pairs, and the claim
is that one curve dominates the other. A connected scatter parameterized by
temperature shows exactly that, and makes "just raise the temperature" a
visible movement along a curve rather than an untested objection.

Color. Arm identity uses the manuscript's established pair -- control orange
and method teal -- because color must follow the entity across every figure in
the paper. That teal sits just under the chroma floor and its tritan separation
from the control orange is 5.4 (validator: chroma FAIL, CVD PASS on protan/
deutan), so identity here is never carried by color alone: each arm additionally
gets its own marker shape, its own line style, and a direct label on the curve.
Raising the method hue to #0E9CB0 would clear every check, but that is a
manuscript-wide decision rather than something to change in one new figure.

Rendering aborts unless the reproduction gate passed, so a frontier can never be
displayed on top of cells that failed to reproduce their published endpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys  # noqa: E402

_OPS = Path(__file__).resolve().parents[1]
if str(_OPS) not in sys.path:
    sys.path.insert(0, str(_OPS))
import paper_style as style  # noqa: E402

INK = style.INK
MUTED = style.MUTED
GRID = style.GRID
WHITE = style.WHITE
CONTROL = style.CONTROL
METHOD = style.METHOD

ARM_STYLE: dict[str, dict[str, Any]] = {
    "drgrpo": {
        "label": "matched Dr.GRPO",
        "color": CONTROL,
        "marker": "o",
        "linestyle": style.ARM_DASH[CONTROL],
    },
    "xgrpo": {
        "label": "historical treatment",
        "color": METHOD,
        "marker": "s",
        "linestyle": style.ARM_DASH[METHOD],
    },
}

DOMAIN_TITLES = {
    "graph_coloring": "Graph coloring",
    "countdown": "Countdown",
    "python_factors": "Python factors",
    "mathir": "MathIR action menu",
    "pantry_plan": "PantryPlan",
}
DOMAIN_ORDER = tuple(DOMAIN_TITLES)

# Labelling every point turns the panel into a wall of numbers. The two ends
# orient the sweep; the reported T=1 operating point is identified by its ring.
LABELLED = (0.5, 2.0)

style.apply_rcparams()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_points(summary: dict[str, Any]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for point in summary["frontier"]["points"]:
        if point["k"] != 8 or point["mean_at_k"] is None:
            continue
        grouped.setdefault((point["domain"], point["arm"]), []).append(point)
    for series in grouped.values():
        series.sort(key=lambda item: item["temperature"])
    return grouped


def render(summary: dict[str, Any], output: Path) -> dict[str, Any]:
    grouped = load_points(summary)
    domains = [
        domain
        for domain in DOMAIN_ORDER
        if any(key[0] == domain for key in grouped)
    ]
    if not domains:
        raise SystemExit("no frontier points to render")

    fig, axes = plt.subplots(
        1, len(domains), figsize=(style.WIDTH, 2.25), constrained_layout=True
    )
    if len(domains) == 1:
        axes = [axes]

    drawn: list[dict[str, Any]] = []
    for axis, domain in zip(axes, domains):
        style.style_axis(axis, title=DOMAIN_TITLES[domain])

        # The panel's own data extent, used below to decide whether an arm's
        # sweep is long enough for its endpoint labels to sit apart.
        panel_points = [
            point
            for arm in ARM_STYLE
            for point in (grouped.get((domain, arm)) or ())
        ]
        domain_x_span = (
            max(point["mean_at_k"] for point in panel_points)
            - min(point["mean_at_k"] for point in panel_points)
            if panel_points
            else 0.0
        )
        domain_y_span = (
            max(point["distinct_at_k"] for point in panel_points)
            - min(point["distinct_at_k"] for point in panel_points)
            if panel_points
            else 0.0
        )

        for arm, arm_style in ARM_STYLE.items():
            series = grouped.get((domain, arm))
            if not series:
                continue
            xs = [point["mean_at_k"] for point in series]
            ys = [point["distinct_at_k"] for point in series]
            axis.plot(
                xs,
                ys,
                color=arm_style["color"],
                linewidth=style.MEAN_LW,
                linestyle=arm_style["linestyle"],
                marker=arm_style["marker"],
                markersize=3.4,
                markeredgecolor=WHITE,
                markeredgewidth=0.7,
                zorder=3,
                label=arm_style["label"],
            )
            # Where an arm barely moves across the whole sweep --- the
            # PantryPlan control collapses to almost a single point --- its
            # "T=2" and "T=0.5" labels land on top of each other and say
            # nothing the collapsed curve has not already said. Drop them and
            # let the ringed T=1 marker stand for the arm.
            labelled_points = [
                point for point in series if point["temperature"] in LABELLED
            ]
            sweep_span = 0.0
            if len(labelled_points) > 1:
                spread_x = max(p["mean_at_k"] for p in labelled_points) - min(
                    p["mean_at_k"] for p in labelled_points
                )
                spread_y = max(p["distinct_at_k"] for p in labelled_points) - min(
                    p["distinct_at_k"] for p in labelled_points
                )
                width = max(domain_x_span, 1e-9)
                height = max(domain_y_span, 1e-9)
                sweep_span = max(spread_x / width, spread_y / height)
            label_endpoints = sweep_span >= 0.12
            for point in series:
                temperature = point["temperature"]
                is_published = temperature == 1.0
                if is_published:
                    # The temperature-one point is the operating point every
                    # published number was measured at; ring it. Sized in area
                    # units (s = 235 pt^2) so the ring reads as a deliberate
                    # marker rather than a slightly fat data point.
                    axis.plot(
                        [point["mean_at_k"]],
                        [point["distinct_at_k"]],
                        marker=arm_style["marker"],
                        markersize=9.2,
                        markerfacecolor="none",
                        markeredgecolor=arm_style["color"],
                        markeredgewidth=1.1,
                        zorder=4,
                    )
                if temperature not in LABELLED or not label_endpoints:
                    continue
                # Endpoint labels are enough to orient the sweep. Offsets keep
                # the two arm labels apart where a nearly vertical baseline
                # collapses several temperatures onto the same point.
                # Both arms put T=2 at the left end of their sweep, so a
                # leftward label runs into the y tick labels (it overprinted
                # "0.7" in the MathIR panel) and the two arms' labels landed on
                # each other wherever their left endpoints were close in y.
                # Label the left end vertically instead --- treatment above,
                # control below --- and only the right end horizontally, where
                # there is open panel to grow into.
                if temperature == 0.5:
                    offset = (4.0, -4.0) if arm == "drgrpo" else (4.0, -7.0)
                    horizontal_alignment = "left"
                else:
                    offset = (0.0, -9.5) if arm == "drgrpo" else (0.0, 5.0)
                    horizontal_alignment = "center"
                axis.annotate(
                    f"T={temperature:g}",
                    (point["mean_at_k"], point["distinct_at_k"]),
                    textcoords="offset points",
                    xytext=offset,
                    ha=horizontal_alignment,
                    fontsize=style.SMALL_FONT - 0.5,
                    color=MUTED,
                    zorder=5,
                )
            drawn.append(
                {
                    "domain": domain,
                    "arm": arm,
                    "points": [
                        {
                            "temperature": point["temperature"],
                            "n": point["n_seeds"],
                            "seeds": point["seeds"],
                            "mean_at_8": point["mean_at_k"],
                            "distinct_at_8": point["distinct_at_k"],
                            "per_seed": point["per_seed"],
                        }
                        for point in series
                    ],
                }
            )

    for axis in axes:
        # Room for the temperature annotations, which otherwise clip against
        # the panel edge at the extreme points of each curve. The x margin also
        # keeps the left-hand "T=2" label off the y tick labels, which it used
        # to overprint in the MathIR panel.
        axis.margins(x=0.20, y=0.16)
    axes[0].set_ylabel(
        "breadth  (# distinct@8)", fontsize=style.LABEL_FONT, labelpad=2
    )
    # One shared x label rather than five copies of the same string. Let the
    # layout engine place it: a hand-set y lands inside the panel row.
    fig.supxlabel("accuracy  (mean@8)", fontsize=style.FONT)

    handles, labels = axes[0].get_legend_handles_labels()
    style.bottom_legend(fig, handles, labels, y=-0.13)

    style.save(fig, output)
    plt.close(fig)
    return {"series": drawn, "domains": domains}


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "byte_length": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(handle, "w", encoding="utf-8") as sink:
        json.dump(payload, sink, indent=2, sort_keys=True)
        sink.write("\n")
    os.replace(temporary, path)


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=root / "var" / "artifacts" / "e72_decoding_frontier_summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "paper" / "figures" / "e72_decoding_frontier.pdf",
    )
    parser.add_argument(
        "--allow-ungated",
        action="store_true",
        help="render even though the reproduction gate has not passed (drafts only)",
    )
    args = parser.parse_args()

    if args.output.suffix.lower() != ".pdf":
        raise SystemExit("--output must name the PDF artifact")

    summary = json.loads(args.summary.read_text())
    gate = summary["reproduction_gate"]
    if not gate["passed"] and not args.allow_ungated:
        raise SystemExit(
            "refusing to render: reproduction gate has not passed "
            f"({gate['failed']}/{gate['checked']} checks failed)"
        )

    drawn = render(summary, args.output)
    source_paths = [
        root / "docs" / "e72_baseline_and_decoding_control_suite.md",
        root / "var" / "artifacts" / "e72_frontier_source_runs.json",
        root / "var" / "artifacts" / "e72_frontier_stage_a_jobs.json",
        root / "var" / "artifacts" / "e72_frontier_stage_b_jobs.json",
        root / "var" / "artifacts" / "e72_frontier_stage_c_jobs.json",
        root / "var" / "artifacts" / "e72_decoding_frontier_cells.jsonl",
        args.summary,
    ]
    missing = [path for path in source_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"missing frozen decoding source: {missing[0]}")

    provenance = {
        "schema": "paper-historical-decoding-frontier-v2",
        "status": "complete five-seed historical control; not x-Mode evidence",
        "scope": (
            "terminal checkpoints from a superseded 12-pass multi-component "
            "treatment; the sweep tests decoding of frozen policies only"
        ),
        "figure": {
            "pdf": str(args.output.relative_to(root)),
            "png": str(args.output.with_suffix(".png").relative_to(root)),
        },
        "summary": str(args.summary.relative_to(root)),
        "input_sha256": {
            str(path.relative_to(root)): file_record(path, root)
            for path in source_paths
        },
        "reproduction_gate_passed": bool(gate["passed"]),
        "reproduction_gate_checked": int(gate["checked"]),
        "reproduction_gate_failed": int(gate["failed"]),
        "cells_measured": int(summary["cells_measured"]),
        "model": "Qwen2.5-0.5B-Instruct",
        "checkpoint": "terminal pass 12",
        "paired_seeds": [43, 44, 45, 46, 47],
        "metrics": {"x": "mean@8", "y": "distinct@8"},
        "temperature_sweep": [0.5, 0.7, 1.0, 1.3, 1.6, 2.0],
        "arms": {
            "drgrpo": "matched Dr.GRPO",
            "xgrpo": "historical multi-component treatment",
        },
        "temperature_repair": [
            row
            for row in summary["frontier"]["temperature_repair"]
            if row["arm"] == "drgrpo"
        ],
        "sample_budget": [
            row
            for row in summary["frontier"]["budget"]
            if row["arm"] in {"drgrpo", "xgrpo"}
        ],
        "nucleus": [
            row
            for row in summary["frontier"]["nucleus"]
            if row["arm"] in {"drgrpo", "xgrpo"}
        ],
        "displayed": drawn,
    }
    adjacent = args.output.with_suffix(".json")
    legacy = root / "var" / "artifacts" / "e72_decoding_frontier_figure_provenance.json"
    write_json(adjacent, provenance)
    write_json(legacy, provenance)
    print(
        f"[e72-frontier] wrote {args.output}, {args.output.with_suffix('.png')}, "
        f"{adjacent}, and {legacy}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
