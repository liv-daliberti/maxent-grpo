#!/usr/bin/env python3
"""Plot the registered E88 adaptive semantic-dose mechanism-gate failure."""

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
import paper_style as style  # noqa: E402


DEFAULT_INPUT = ROOT / "var/artifacts/e88_closure_record.json"
DEFAULT_OUTPUT = ROOT / "paper/figures/adaptive_semantic_gate_e88"
DOMAIN_ORDER = {
    "graph_coloring": 0,
    "countdown": 1,
    "python_factors": 2,
    "mathir": 3,
    "pantry_plan": 4,
}
DOMAIN_LABELS = {
    "graph_coloring": "Graph",
    "countdown": "Countdown",
    "python_factors": "Python",
    "mathir": "MathIR",
    "pantry_plan": "Pantry",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = sorted(
        payload["evidence"],
        key=lambda row: (DOMAIN_ORDER[row["domain"]], int(row["seed"])),
    )
    if len(records) != int(payload["cells_with_controller_evidence"]):
        raise RuntimeError("E88 closure count disagrees with its evidence rows")
    return records


def render(payload: dict[str, Any], records: list[dict[str, Any]], output: Path) -> None:
    style.apply_rcparams()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(style.WIDTH, 3.05),
        sharey=True,
    )
    positions = list(range(len(records) - 1, -1, -1))
    labels = [
        f"{DOMAIN_LABELS[row['domain']]} · s{row['seed']}" for row in records
    ]
    colors = [style.ADAPTIVE if row["at_max"] else style.METHOD for row in records]

    panels = (
        (
            "final_realized_ratio",
            "realized semantic/task RMS ratio",
            (0.0, 0.062),
            float(payload["target_ratio"]),
            r"registered target $\rho=.05$",
        ),
        (
            "bound_hit_fraction",
            "fraction of controller updates at a bound",
            (0.0, 1.04),
            0.20,
            "registered maximum .20",
        ),
    )
    for axis, (field, xlabel, limits, threshold, threshold_label) in zip(
        axes, panels
    ):
        style.style_axis(axis)
        axis.axvline(
            threshold,
            color=style.CONTROL,
            lw=1.0,
            linestyle=style.ARM_DASH[style.CONTROL],
            zorder=2,
        )
        for y, row, color in zip(positions, records, colors):
            value = float(row[field])
            axis.hlines(y, 0.0, value, color=color, lw=0.8, alpha=0.7, zorder=2)
            axis.scatter(
                [value],
                [y],
                s=25,
                marker="D" if row["at_max"] else "o",
                color=color,
                edgecolor=style.WHITE,
                linewidth=0.5,
                zorder=3,
            )
        axis.set_xlim(*limits)
        axis.set_xlabel(xlabel, fontsize=style.LABEL_FONT)
        axis.set_yticks(positions, labels)
        axis.text(
            threshold,
            len(records) - 0.2,
            threshold_label,
            fontsize=style.SMALL_FONT,
            color=style.CONTROL,
            ha="right",
            va="bottom",
        )

    axes[0].set_ylim(-0.65, len(records) - 0.1)
    figure.suptitle(
        "Why the original adaptive semantic-dose target failed",
        fontsize=style.TITLE_FONT,
        color=style.INK,
        y=0.995,
    )
    figure.text(
        0.5,
        0.955,
        (
            "Controller target and bound criteria; task outcomes do not "
            "determine the mechanism verdict"
        ),
        ha="center",
        va="top",
        fontsize=style.SMALL_FONT,
        color=style.MUTED,
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=style.METHOD,
            lw=0,
            markersize=4,
            label=r"$\eta<.40$ at closure",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color=style.ADAPTIVE,
            lw=0,
            markersize=4,
            label=r"$\eta=.40$ ceiling",
        ),
        Line2D(
            [0],
            [0],
            color=style.CONTROL,
            linestyle=style.ARM_DASH[style.CONTROL],
            lw=1.0,
            label="registered gate",
        ),
    ]
    style.bottom_legend(
        figure,
        handles,
        [handle.get_label() for handle in handles],
        y=-0.015,
        ncol=3,
    )
    figure.subplots_adjust(
        left=0.16,
        right=0.995,
        top=0.86,
        bottom=0.18,
        wspace=0.14,
    )
    style.save(figure, output, png=True, dpi=240)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.input.resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != "e88_closure_v1":
        raise RuntimeError("unexpected E88 closure schema")
    records = _records(payload)
    output = args.output.resolve()
    render(payload, records, output)
    provenance = {
        "schema": "paper-adaptive-dose-gate-figure-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "closed registered mechanism-gate evidence",
        "source": str(source),
        "source_sha256": _sha256(source),
        "target_ratio": payload["target_ratio"],
        "maximum_bound_hit_fraction": 0.20,
        "records": records,
        "outcome": payload["outcome"],
        "gate_failure": payload["gate_failure"],
        "superseded_by": payload["superseded_by"],
    }
    output.with_suffix(".json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output.with_suffix('.pdf')}")
    print(f"wrote {output.with_suffix('.png')}")
    print(f"wrote {output.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
