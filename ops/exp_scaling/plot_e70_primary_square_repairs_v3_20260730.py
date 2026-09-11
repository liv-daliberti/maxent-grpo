#!/usr/bin/env python3
"""Zero-line and interpretation correction for the square primary surface."""

from __future__ import annotations

import plot_e70_primary_square_repairs_v2_20260730 as labels


primary = labels.primary
primary.repair.REPAIR_INPUTS.update(
    {
        "pantry_audit": primary.ROOT
        / "var/artifacts/"
        "pantry_support_retention_final_v1_audit_amendment_v3.json",
        "point_protocol": primary.ROOT
        / "paper/preregistration/"
        "point_maze_balanced_warmstart_v5_20260730.md",
        "point_identity": primary.ROOT
        / "var/artifacts/"
        "point_maze_balanced_warmstart_v5_identity.json",
        "point_audit": primary.ROOT
        / "var/artifacts/"
        "point_maze_balanced_warmstart_v5_qualification.json",
        "ant_protocol": primary.ROOT
        / "paper/preregistration/"
        "ant_stable_handoff_controller_v18_20260730.md",
        "ant_identity": primary.ROOT
        / "var/artifacts/"
        "ant_stable_handoff_controller_v18_identity.json",
        "ant_controller_gate": primary.ROOT
        / "var/maze_runtime/controllers/"
        "ant_stable_handoff_v18.evaluation.json",
    }
)
primary.ROWS = tuple(
    (
        label,
        (
            "Historical replacement · 10/10 provenance audit; "
            "efficacy null/weak"
            if domain == "point_maze_geometry_shift"
            else (
                "Secondary balanced-warmstart v5 qualification; "
                "no online pair promoted"
                if domain == "point_maze_algorithm_repair"
                else note
            )
        ),
        domain,
        prefix,
        steps,
        seeds,
    )
    for label, note, domain, prefix, steps, seeds in primary.ROWS
)
_stage_a = primary.repair._read(primary.base.AUDIT).get("summary", {})
_stage_a_remaining = max(
    0, 40 - int(_stage_a.get("terminal_runs", 0) or 0)
)
primary.ROWS = tuple(
    (
        label,
        (
            f"Original cohort · {_stage_a_remaining} jobs still active"
            if domain == "mathir"
            else note
        ),
        domain,
        prefix,
        steps,
        seeds,
    )
    for label, note, domain, prefix, steps, seeds in primary.ROWS
)

_plot = primary._plot
_points = primary._points
_status = primary._status
_repair_summary = primary.repair._repair_summary


def repair_summary():
    summary = _repair_summary()
    summary["point_metric_rows"] = 0
    return summary


def points(domain, prefix, steps):
    # V2 stays in its frozen receipt, but it is not the current repair cohort.
    # V5 stopped at its prospectively frozen qualification gate.
    if domain == "point_maze_algorithm_repair":
        return []
    return _points(domain, prefix, steps)


def status(domain: str):
    if domain == "point_maze_algorithm_repair":
        qualification = primary.repair._read(
            primary.ROOT
            / "var/artifacts/"
            "point_maze_balanced_warmstart_v5_qualification.json"
        )
        if qualification.get("status") == "fail":
            rate = qualification.get("summary", {}).get("verified_rate")
            rate_text = (
                f"{100.0 * float(rate):.1f}%"
                if isinstance(rate, (int, float))
                else "unknown"
            )
            return (
                "V5 QUALIFICATION STOPPED FAIL-CLOSED",
                f"verified completion rate {rate_text}, above the frozen "
                "2–50% trainability band; zero online jobs launched",
                "#B91C1C",
            )
        return (
            "V5 QUALIFICATION PENDING",
            "balanced warmstart gate controls all ten online jobs",
            "#B36B00",
        )
    if domain == "ant_maze_harder_repair":
        identity = primary.repair._read(
            primary.ROOT
            / "var/artifacts/"
            "ant_stable_handoff_controller_v18_identity.json"
        )
        receipt = primary.repair._read(
            primary.ROOT
            / "var/maze_runtime/controllers/"
            "ant_stable_handoff_v18.evaluation.json"
        )
        if receipt.get("status") == "pass":
            return (
                "V18 CONTROLLER GATE PASSED",
                "the unchanged harder v15 route slate is eligible for "
                "fresh executable admission",
                "#008A5A",
            )
        if receipt.get("status") == "fail":
            return (
                "V18 CONTROLLER GATE FAILED",
                "the harder AntMaze repair remains fail-closed",
                "#B91C1C",
            )
        return (
            "STABLE-HANDOFF V18 CONTROLLER TRAINING",
            f"identity-bound job {identity.get('job_id', 'unknown')}; "
            "a fresh 96-episode gate controls route admission",
            "#B36B00",
        )
    return _status(domain)


def plot(axis, points, seeds, metric, integer):
    _plot(axis, points, seeds, metric, integer)
    values = [
        value
        for arm in (primary.CONTROL, primary.TREATMENT)
        for seed in seeds
        for _pass_index, value in primary._series(
            points, arm, seed, metric
        )
    ]
    if values and max(abs(value) for value in values) <= 1e-12:
        upper = 1.02 if metric in {"greedy", "pass8"} else 1.0
        axis.set_ylim(-0.035, upper)
        if metric in {"greedy", "pass8"}:
            axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.axhline(0.0, color="#64748B", linewidth=0.7, zorder=1)
        axis.text(
            0.5,
            0.055,
            "all checkpoints = 0",
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#64748B",
        )


primary._plot = plot
primary._points = points
primary._status = status
primary.repair._repair_summary = repair_summary


if __name__ == "__main__":
    primary.render()
