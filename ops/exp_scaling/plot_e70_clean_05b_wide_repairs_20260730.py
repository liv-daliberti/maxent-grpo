#!/usr/bin/env python3
"""Refresh the historical wide E70 graph with a separate repair band.

The first eight rows retain the original clean-campaign estimator and its
registered seeds.  Three appended rows are explicitly secondary post-outcome
repairs; they never enter the original 80-cell terminal count.  Incomplete
repair gates are displayed as status cards, while any already-materialized
qualification telemetry is plotted without carrying values forward.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.figure
import matplotlib.pyplot as plt

import plot_e70_clean_05b_wide_live as base


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = (
    ROOT
    / "paper/figures/"
    "e68_e58_vs_grpo_05b_12ep_terminal_provenance_historical_20260730.png"
)
SIDECAR = (
    ROOT
    / "var/artifacts/"
    "e68_e58_vs_grpo_05b_12ep_terminal_provenance_"
    "historical_20260730_provenance.json"
)

OPEN_SET_ENTROPY = (
    "semantic_shannon_success_conditioned_signed_open_set_entropy_ema"
)
OPEN_SET_COEFFICIENT = (
    "semantic_shannon_success_conditioned_signed_open_set_next_coefficient"
)
MASS_COEFFICIENT = "canonical_replay_mass_next_alpha"
BALANCE_COEFFICIENT = "canonical_replay_next_alpha"

WIDE_PANELS = (
    ("greedy", "neutral pass@1", False),
    ("mean8", "neutral mean@8", False),
    ("pass8", "neutral pass@8", False),
    ("distinct8", "mean # distinct correct@8", True),
    (OPEN_SET_ENTROPY, "open-set predictive entropy EMA", False),
    (OPEN_SET_COEFFICIENT, r"next semantic coefficient $\beta$", False),
    (MASS_COEFFICIENT, r"next verified-mass coefficient $\mu$", False),
    (BALANCE_COEFFICIENT, r"next known-mode coefficient $\alpha$", False),
    ("canonical_replay_balance_loss", "verified replay KL", False),
    ("canonical_replay_available_modes", "replayed verified modes", True),
    (
        "online_canonical_new_outcome_row_fraction",
        "new verified outcome fraction",
        False,
    ),
    (
        "online_canonical_mean_support_per_prompt",
        "mean verified support per prompt",
        False,
    ),
    (
        "online_canonical_tracked_outcomes",
        "cumulative verified discoveries",
        True,
    ),
)

REPAIR_DOMAINS = (
    (
        "PantryPlan support-retention repair\n"
        "(secondary; five fresh paired seeds)",
        "pantry_support_retention_repair",
        None,
        None,
    ),
    (
        "PointMaze algorithm repair\n"
        "(secondary qualification pair)",
        "point_maze_algorithm_repair",
        "__point_maze_algorithm_repair_v2r5__",
        1,
    ),
    (
        "AntMaze harder-task repair\n"
        "(secondary controller/admission gate)",
        "ant_maze_harder_repair",
        None,
        None,
    ),
)

REPAIR_INPUTS = {
    "diagnosis": (
        ROOT
        / "paper/preregistration/"
        "ant_point_pantry_repair_diagnosis_20260730.md"
    ),
    "pantry_protocol": (
        ROOT
        / "paper/preregistration/"
        "pantry_support_retention_final_v1_20260730.md"
    ),
    "pantry_identity": (
        ROOT / "var/artifacts/pantry_support_retention_final_v1_identity.json"
    ),
    "pantry_audit": (
        ROOT / "var/artifacts/pantry_support_retention_final_v1_audit.json"
    ),
    "point_protocol": (
        ROOT
        / "paper/preregistration/"
        "point_maze_algorithm_repair_pair_v2_20260730.md"
    ),
    "point_qualification": (
        ROOT
        / "var/artifacts/point_maze_algorithm_repair_v2_qualification.json"
    ),
    "point_identity": (
        ROOT
        / "var/artifacts/point_maze_algorithm_repair_pair_v2r5_identity.json"
    ),
    "point_audit": (
        ROOT
        / "var/artifacts/point_maze_algorithm_repair_pair_v2r5_audit.json"
    ),
    "point_grpo_metrics": (
        ROOT
        / "var/artifacts/"
        "point_maze_algorithm_repair_v2r5_grpo_s76521.metrics.jsonl"
    ),
    "point_maxent_metrics": (
        ROOT
        / "var/artifacts/"
        "point_maze_algorithm_repair_v2r5_"
        "verified_first_global_replay_canonical_s76521.metrics.jsonl"
    ),
    "ant_protocol": (
        ROOT
        / "paper/preregistration/"
        "ant_sequential_waypoint_controller_v16_r1_20260730.md"
    ),
    "ant_identity": (
        ROOT
        / "var/artifacts/ant_sequential_waypoint_controller_v16_identity.json"
    ),
    "ant_controller_gate": (
        ROOT
        / "var/maze_runtime/controllers/"
        "ant_sequential_waypoint_v16.evaluation.json"
    ),
    "geometry_shift_audit": (
        ROOT
        / "var/artifacts/"
        "point_maze_geometry_shift_stage_b_05b_12pass_audit.json"
    ),
}


def _read(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_point_repair() -> list[dict[str, Any]]:
    merged: dict[tuple[str, int, float], dict[str, Any]] = {}
    for arm, path in (
        ("grpo", REPAIR_INPUTS["point_grpo_metrics"]),
        (
            "verified_first_global_replay_canonical",
            REPAIR_INPUTS["point_maxent_metrics"],
        ),
    ):
        if not path.is_file():
            continue
        for raw in path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if (
                not isinstance(row, Mapping)
                or not base._finite(row.get("training_passes"))
            ):
                continue
            seed = int(row.get("seed", 76521))
            passes = round(float(row["training_passes"]), 9)
            point = merged.setdefault(
                (arm, seed, passes),
                {"arm": arm, "seed": seed, "passes": passes},
            )
            for metric, value in row.items():
                if base._finite(value):
                    point[str(metric)] = float(value)
            if base._finite(row.get("replay_active_modes")):
                point["canonical_replay_available_modes"] = float(
                    row["replay_active_modes"]
                )
            if base._finite(row.get("canonical_tracked_outcomes")):
                point["online_canonical_tracked_outcomes"] = float(
                    row["canonical_tracked_outcomes"]
                )
            tracked_prompts = row.get("canonical_tracked_prompts")
            tracked_outcomes = row.get("canonical_tracked_outcomes")
            if (
                base._finite(tracked_prompts)
                and float(tracked_prompts) > 0
                and base._finite(tracked_outcomes)
            ):
                point["online_canonical_mean_support_per_prompt"] = (
                    float(tracked_outcomes) / float(tracked_prompts)
                )
    return [merged[key] for key in sorted(merged)]


def _repair_status(domain: str) -> tuple[str, str, str]:
    if domain == "pantry_support_retention_repair":
        audit = _read(REPAIR_INPUTS["pantry_audit"])
        if audit.get("status") == "pass":
            return (
                "TEN REPAIR RUNS PASSED AUDIT",
                "five paired seeds × 12 passes on the untouched 64-row split",
                "#008A5A",
            )
        if audit.get("status") == "fail":
            return (
                "REPAIR COHORT FAILED AUDIT",
                "secondary result stopped fail-closed",
                "#B91C1C",
            )
        identity = _read(REPAIR_INPUTS["pantry_identity"])
        jobs = identity.get("jobs", {})
        return (
            "TEN REPAIR RUNS ACTIVE",
            f"{len(jobs) if isinstance(jobs, Mapping) else 0}/10 launched; "
            "five fresh paired seeds; untouched evaluation split; audit pending",
            "#B36B00",
        )
    if domain == "ant_maze_harder_repair":
        gate = _read(REPAIR_INPUTS["ant_controller_gate"])
        if gate.get("status") == "pass":
            return (
                "FRESH CONTROLLER GATE PASSED",
                "harder frozen route admission is eligible next; no model result yet",
                "#008A5A",
            )
        if gate.get("status") == "fail":
            return (
                "FRESH CONTROLLER GATE FAILED",
                "harder AntMaze repair remains fail-closed",
                "#B91C1C",
            )
        identity = _read(REPAIR_INPUTS["ant_identity"])
        return (
            "GENERIC TURN CONTROLLER TRAINING",
            f"identity-bound job {identity.get('job_id', 'unknown')}; "
            "64 fresh held-out controller episodes gate route admission",
            "#B36B00",
        )
    return base._status_for_pending_row(domain)


def _count_original_terminal() -> int:
    stage_a = _read(base.AUDIT).get("summary", {})
    total = int(stage_a.get("terminal_runs", 0))
    for path in (
        ROOT / "var/artifacts/pantry_stage_b_05b_12pass_audit.json",
        ROOT / "var/artifacts/point_maze_stage_b_05b_12pass_audit.json",
        ROOT / "var/artifacts/ant_maze_stage_b_05b_12pass_audit.json",
        REPAIR_INPUTS["geometry_shift_audit"],
    ):
        if _read(path).get("status") == "pass":
            total += 10
    return total


def _repair_summary() -> dict[str, Any]:
    pantry_identity = _read(REPAIR_INPUTS["pantry_identity"])
    point_identity = _read(REPAIR_INPUTS["point_identity"])
    ant_identity = _read(REPAIR_INPUTS["ant_identity"])
    pantry_jobs = pantry_identity.get("jobs", {})
    point_jobs = point_identity.get("jobs", {})
    return {
        "pantry_model_jobs_launched": (
            len(pantry_jobs) if isinstance(pantry_jobs, Mapping) else 0
        ),
        "point_model_jobs_launched": (
            len(point_jobs) if isinstance(point_jobs, Mapping) else 0
        ),
        "ant_controller_job_id": ant_identity.get("job_id"),
        "pantry_audit_status": _read(REPAIR_INPUTS["pantry_audit"]).get(
            "status", "pending"
        ),
        "point_audit_status": _read(REPAIR_INPUTS["point_audit"]).get(
            "status", "pending"
        ),
        "ant_controller_gate_status": _read(
            REPAIR_INPUTS["ant_controller_gate"]
        ).get("status", "running"),
        "point_metric_rows": len(_load_point_repair()),
    }


def render() -> None:
    original_load_points = base._load_points
    original_status = base._status_for_pending_row
    original_subplots = plt.subplots
    original_suptitle = matplotlib.figure.Figure.suptitle
    original_figure_text = matplotlib.figure.Figure.text

    def load_points(prefix: str | None, steps: int | None):
        if prefix == "__point_maze_algorithm_repair_v2r5__":
            return _load_point_repair()
        return original_load_points(prefix, steps)

    def wide_subplots(*args, **kwargs):
        kwargs["figsize"] = (31.5, 15.4)
        return original_subplots(*args, **kwargs)

    generated = datetime.now(timezone.utc)
    original_terminal = _count_original_terminal()
    repairs = _repair_summary()

    def repaired_suptitle(figure, text, *args, **kwargs):
        replacement = (
            "Clean 0.5B replacement campaign + post-outcome repair band "
            "— historical E68 diagnostic format\n"
            "13 outcome/mechanism panels; original cohort remains "
            "5 seeds × 12 passes and estimator-locked\n"
            f"Original campaign: {original_terminal}/80 audited terminal; "
            "PointMaze geometry-shift replacement audit passed\n"
            "Secondary repairs (excluded from 80): "
            f"PantryPlan {repairs['pantry_model_jobs_launched']}/10 launched; "
            f"PointMaze qualification "
            f"{repairs['point_model_jobs_launched']}/2 launched; "
            f"Ant controller job {repairs['ant_controller_job_id']} "
            f"({repairs['ant_controller_gate_status']})\n"
            "No repair curve is promoted before its registered audit "
            f"({generated.strftime('%Y-%m-%d %H:%M UTC')})"
        )
        kwargs["fontsize"] = 9.2
        kwargs["y"] = 1.015
        return original_suptitle(figure, replacement, *args, **kwargs)

    def repaired_text(figure, x, y, text, *args, **kwargs):
        if isinstance(text, str) and text.startswith(
            "Thin lines are exact E70 seed trajectories."
        ):
            text = (
                "Rows 1–8 are the original estimator-locked campaign; "
                "thin lines are exact seed trajectories and thick lines "
                "average only available registered seeds.\n"
                "Rows 9–11 are secondary post-outcome repairs and are "
                "excluded from the original 80-cell count. PantryPlan uses "
                "five fresh paired seeds and a previously untouched split;\n"
                "PointMaze shows only the live preregistered qualification "
                "pair (seed 76521); AntMaze remains at its fresh controller "
                "gate. Pending states are not terminal outcomes."
            )
            kwargs.update(
                ha="center",
                fontsize=6.7,
                color="#444444",
            )
        return original_figure_text(figure, x, y, text, *args, **kwargs)

    base.PANELS = WIDE_PANELS
    base.QUALITY_METRICS = {"greedy", "mean8", "pass8", "distinct8"}
    base.DOMAIN_SPECS = tuple(base.DOMAIN_SPECS) + REPAIR_DOMAINS
    base.SEEDS = (43, 44, 45, 46, 47, 76521)
    base.SEED_STYLES[76521] = (0, (3, 1, 1, 1))
    base._load_points = load_points
    base._status_for_pending_row = _repair_status
    plt.subplots = wide_subplots
    matplotlib.figure.Figure.suptitle = repaired_suptitle
    matplotlib.figure.Figure.text = repaired_text
    try:
        base.render(OUTPUT, SIDECAR)
    finally:
        base._load_points = original_load_points
        base._status_for_pending_row = original_status
        plt.subplots = original_subplots
        matplotlib.figure.Figure.suptitle = original_suptitle
        matplotlib.figure.Figure.text = original_figure_text

    provenance = _read(SIDECAR)
    provenance.update(
        schema="e70-clean-05b-wide-live-with-secondary-repairs-v1",
        generated_at=generated.isoformat(),
        original_campaign_terminal_cells=original_terminal,
        original_campaign_expected_cells=80,
        repair_band=repairs,
        repair_rows=[
            "pantry_support_retention_repair",
            "point_maze_algorithm_repair",
            "ant_maze_harder_repair",
        ],
        repair_excluded_from_original_estimator=True,
        hashes={
            **(
                provenance.get("hashes", {})
                if isinstance(provenance.get("hashes"), Mapping)
                else {}
            ),
            "repair_plot_source": base._sha256(Path(__file__).resolve()),
            **{
                key: base._sha256(path)
                for key, path in REPAIR_INPUTS.items()
            },
            "figure": base._sha256(OUTPUT),
        },
    )
    temporary = SIDECAR.with_suffix(SIDECAR.suffix + ".tmp")
    temporary.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(SIDECAR)
    print(
        "[e70-wide-repairs] "
        f"output={OUTPUT.relative_to(ROOT)} "
        f"original_terminal={original_terminal}/80 "
        f"repair_metric_rows={repairs['point_metric_rows']}"
    )


if __name__ == "__main__":
    render()
