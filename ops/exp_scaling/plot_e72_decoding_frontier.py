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
        "label": "xGRPO",
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

# Temperatures that carry a printed label. Labelling every point turns the
# panel into a wall of numbers; the ends and the published operating point are
# what a reader needs to orient the curve.
LABELLED = (0.5, 1.0, 2.0)

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
        1, len(domains), figsize=(style.WIDTH, 2.15), constrained_layout=True
    )
    if len(domains) == 1:
        axes = [axes]

    drawn: list[dict[str, Any]] = []
    for axis, domain in zip(axes, domains):
        style.style_axis(axis, title=DOMAIN_TITLES[domain])

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
            for point in series:
                temperature = point["temperature"]
                if temperature not in LABELLED:
                    continue
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
                        markersize=235 ** 0.5,
                        markerfacecolor="none",
                        markeredgecolor=arm_style["color"],
                        markeredgewidth=1.3,
                        zorder=4,
                    )
                # The enlarged ring would swallow a label placed above-right, and
                # at the top of a panel that label also collided with the axis;
                # the temperature-one label therefore sits below its ring.
                offset = (7.0, -8.5) if is_published else (3.2, 3.2)
                axis.annotate(
                    f"{temperature:g}",
                    (point["mean_at_k"], point["distinct_at_k"]),
                    textcoords="offset points",
                    xytext=offset,
                    fontsize=style.SMALL_FONT - 0.5,
                    color=MUTED,
                    zorder=5,
                )
            drawn.append(
                {
                    "domain": domain,
                    "arm": arm,
                    "temperatures": [point["temperature"] for point in series],
                    "mean_at_8": xs,
                    "distinct_at_8": ys,
                }
            )

    for axis in axes:
        # Room for the temperature annotations, which otherwise clip against
        # the panel edge at the extreme points of each curve.
        axis.margins(x=0.14, y=0.16)
    axes[0].set_ylabel(
        "breadth  (# distinct@8)", fontsize=style.LABEL_FONT, labelpad=2
    )
    # One shared x label rather than five copies of the same string. Let the
    # layout engine place it: a hand-set y lands inside the panel row.
    fig.supxlabel("accuracy  (mean@8)", fontsize=style.FONT)

    handles, labels = axes[0].get_legend_handles_labels()
    style.bottom_legend(fig, handles, labels, y=-0.14)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return {"series": drawn, "domains": domains}


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

    summary = json.loads(args.summary.read_text())
    gate = summary["reproduction_gate"]
    if not gate["passed"] and not args.allow_ungated:
        raise SystemExit(
            "refusing to render: reproduction gate has not passed "
            f"({gate['failed']}/{gate['checked']} checks failed)"
        )

    drawn = render(summary, args.output)
    provenance = {
        "schema": "e72_decoding_frontier_figure_v1",
        "figure": str(args.output),
        "summary": str(args.summary),
        "summary_sha256": hashlib.sha256(args.summary.read_bytes()).hexdigest(),
        "reproduction_gate_passed": bool(gate["passed"]),
        "reproduction_gate_checked": int(gate["checked"]),
        "cells_measured": int(summary["cells_measured"]),
        "displayed": drawn,
    }
    path = root / "var" / "artifacts" / "e72_decoding_frontier_figure_provenance.json"
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(handle, "w", encoding="utf-8") as sink:
        json.dump(provenance, sink, indent=2, sort_keys=True)
        sink.write("\n")
    os.replace(temporary, path)
    print(f"[e72-frontier] wrote {args.output} and {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
