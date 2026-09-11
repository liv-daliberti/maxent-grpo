#!/usr/bin/env python3
"""Render the refreshed wide graph with eight compact, near-square rows."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt

import plot_e70_clean_05b_wide_repairs_20260730 as repair


ROOT = Path(__file__).resolve().parents[2]


def render() -> None:
    # Keep the five still-valid clean-campaign rows, then replace the three
    # diagnosed environment rows with their explicit secondary repairs.
    repair.base.DOMAIN_SPECS = tuple(repair.base.DOMAIN_SPECS[:5])

    # V16 is a recorded failed antecedent; the current live gate is v17.
    repair.REPAIR_INPUTS["ant_protocol"] = (
        ROOT
        / "paper/preregistration/"
        "ant_sequential_waypoint_controller_v17_20260730.md"
    )
    repair.REPAIR_INPUTS["ant_identity"] = (
        ROOT
        / "var/artifacts/"
        "ant_sequential_waypoint_controller_v17_identity.json"
    )
    repair.REPAIR_INPUTS["ant_controller_gate"] = (
        ROOT
        / "var/maze_runtime/controllers/"
        "ant_sequential_waypoint_v17.evaluation.json"
    )
    repair.REPAIR_INPUTS["ant_v16_failed_gate"] = (
        ROOT
        / "var/maze_runtime/controllers/"
        "ant_sequential_waypoint_v16.evaluation.json"
    )

    real_subplots = plt.subplots
    real_suptitle = matplotlib.figure.Figure.suptitle
    real_text = matplotlib.figure.Figure.text
    real_tight_layout = matplotlib.figure.Figure.tight_layout
    generated = datetime.now(timezone.utc)

    def square_subplots(*args, **kwargs):
        # This reproduces the requested historical canvas ratio while giving
        # each of 8x13 panels approximately square plotting area.
        kwargs["figsize"] = (31.5, 18.0)
        return real_subplots(*args, **kwargs)

    def compact_tight_layout(figure, *args, **kwargs):
        kwargs["pad"] = 0.25
        kwargs["w_pad"] = 0.35
        kwargs["h_pad"] = 0.32
        return real_tight_layout(figure, *args, **kwargs)

    def square_suptitle(figure, _text, *args, **kwargs):
        summary = repair._repair_summary()
        text = (
            "Clean 0.5B campaign + secondary environment repairs "
            "— historical E68 diagnostic format\n"
            "13 outcome/mechanism panels; 8 compact rows; no carry-forward\n"
            f"Original estimator: {repair._count_original_terminal()}/80 "
            "audited terminal; geometry-shift replacement passed\n"
            "Repaired rows: "
            f"PantryPlan {summary['pantry_model_jobs_launched']}/10 launched; "
            f"PointMaze qualification {summary['point_model_jobs_launched']}/2 "
            f"launched; Ant v17 controller job "
            f"{summary['ant_controller_job_id']} "
            f"({summary['ant_controller_gate_status']}) "
            f"({generated.strftime('%Y-%m-%d %H:%M UTC')})"
        )
        kwargs["fontsize"] = 10.5
        kwargs["y"] = 1.008
        return real_suptitle(figure, text, *args, **kwargs)

    def square_text(figure, x, y, text, *args, **kwargs):
        if isinstance(text, str) and (
            text.startswith("Rows 1–8 are the original")
            or text.startswith("Thin lines are exact E70")
        ):
            text = (
                "Rows 1–5 retain eligible clean-campaign trajectories. "
                "Rows 6–8 replace the diagnosed PantryPlan, PointMaze, and "
                "AntMaze configurations with secondary repairs.\n"
                "PointMaze plots only its preregistered live qualification "
                "pair; PantryPlan and AntMaze remain status-only until their "
                "fresh audits pass. Repairs are excluded from the original "
                "80-cell estimator."
            )
            kwargs.update(ha="center", fontsize=7.2, color="#444444")
        return real_text(figure, x, y, text, *args, **kwargs)

    plt.subplots = square_subplots
    matplotlib.figure.Figure.suptitle = square_suptitle
    matplotlib.figure.Figure.text = square_text
    matplotlib.figure.Figure.tight_layout = compact_tight_layout
    try:
        repair.render()
    finally:
        plt.subplots = real_subplots
        matplotlib.figure.Figure.suptitle = real_suptitle
        matplotlib.figure.Figure.text = real_text
        matplotlib.figure.Figure.tight_layout = real_tight_layout

    provenance = repair._read(repair.SIDECAR)
    provenance.update(
        schema="e70-clean-05b-wide-eight-row-repair-replacement-v2",
        layout={
            "rows": 8,
            "columns": 13,
            "figsize_inches": [31.5, 18.0],
            "tight_layout_pad": 0.25,
            "tight_layout_w_pad": 0.35,
            "tight_layout_h_pad": 0.32,
            "mistuned_original_rows_foregrounded": False,
            "repair_rows_replace_diagnosed_rows": True,
        },
        hashes={
            **provenance.get("hashes", {}),
            "square_plot_source": repair.base._sha256(Path(__file__).resolve()),
            "figure": repair.base._sha256(repair.OUTPUT),
        },
    )
    temporary = repair.SIDECAR.with_suffix(repair.SIDECAR.suffix + ".tmp")
    temporary.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(repair.SIDECAR)
    print(
        "[e70-wide-repairs-square] "
        f"output={repair.OUTPUT.relative_to(ROOT)} rows=8 columns=13"
    )


if __name__ == "__main__":
    render()
