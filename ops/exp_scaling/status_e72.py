#!/usr/bin/env python3
"""Print E72 campaign progress: where each cohort is, and how far it moved.

Every figure is measured, not estimated from wall clock: training progress is
the sum of realized optimizer steps, and frontier progress counts completion
markers. Each run appends a snapshot to a history file, so the ``was`` column
is the previous invocation rather than a remembered number, and the rate is
derived from the gap between them.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

# --- cohort sizes, from the registered protocols -----------------------------
B3A_RUNS = 25
STEPS_PER_RUN = 4608
STEPS_PER_PASS = 384

# Progress is measured against the depth the cohort is *wanted* to reach, not
# the depth its runs happen to be configured for. E74 is capped at four passes
# by decision; three of its runs were launched on the original twelve-pass
# config and stopped just past four. Charging those the twelve-pass budget made
# a cohort that has met its target read as barely started.
PASS_RE = re.compile(r"_(\d+)pass_")


def configured_passes(name: str, default_passes: int) -> int:
    """What a run was launched to do, from its stamp. Reported, not targeted."""
    match = PASS_RE.search(name)
    return int(match.group(1)) if match else default_passes


B3A_TOTAL_STEPS = B3A_RUNS * STEPS_PER_RUN

# 50 trained checkpoints plus 10 base-model references (one per domain and GPU
# model) = 60 measurable runs, times the temperatures each stage sweeps.
FRONTIER_RUNS = 60
FRONTIER_STAGES = {
    "a": FRONTIER_RUNS * 6,
    "b": FRONTIER_RUNS * 3,
    "c": FRONTIER_RUNS * 2,
}
FRONTIER_TOTAL_CELLS = sum(FRONTIER_STAGES.values())

# --- cost model, for the compute-share rows ----------------------------------
# Observed on this cohort: ~640 optimizer steps/hour/run, and ~3 minutes per
# eval-only cell including model load.
HOURS_PER_TRAINING_RUN = STEPS_PER_RUN / 640
HOURS_PER_CELL = 0.05
B3A_GPU_HOURS = B3A_RUNS * HOURS_PER_TRAINING_RUN
FRONTIER_GPU_HOURS = FRONTIER_TOTAL_CELLS * HOURS_PER_CELL
# Wave A of the baseline suite is five training arms of this size, plus Tier 0.
WAVE_A_ARMS = 5
WAVE_A_GPU_HOURS = WAVE_A_ARMS * B3A_GPU_HOURS + FRONTIER_GPU_HOURS

# Every training cohort in the campaign, newest last. A cohort is 25 runs
# unless it pairs two arms on a fresh seed set, which the confirmation does.
COHORTS: tuple[tuple[str, str, int, int], ...] = (
    (
        "B3a  replay gradient removed",
        "xdr_qwen25_0p5b_instruct_verified_first_replay_gradient_ablation_*_b3a_s*",
        25,
        12,
    ),
    (
        "B1a  semantic MaxEnt removed",
        "xdr_qwen25_0p5b_instruct_verified_first_replay_only_ablation_*_b1a_s*",
        25,
        12,
    ),
    (
        "B1b  rehearsal only, no balance",
        "xdr_qwen25_0p5b_instruct_verified_first_replay_rehearsal_only_*_b1b_s*",
        25,
        12,
    ),
    # `s4?` and not `s4*`: the controller calibration screen writes runs named
    # ..._b2b_s43_cal_<setting>, and a trailing wildcard would count those short

    # two-pass cells as cohort progress.
    (
        "B2b  matched token entropy",
        "xdr_qwen25_0p5b_instruct_matched_token_entropy_ablation_*_b2b_s4?",
        25,
        12,
    ),
    # Both arms of the confirmation, matched pairwise on fresh seeds.
    ("B1a confirmation  seeds 48-52", "xdr_qwen25_0p5b_instruct_*conf_s*", 50, 12),
    # E74: the headline design at 3B on the idle A100 node. Expected count is
    # the wave in flight, not the eventual cohort, so the line reads as
    # progress rather than as a fraction of an unfunded plan.
    ("E74  Qwen2.5-3B scale", "xdr_qwen25_3b_instruct_*e74_qwen3b_*pass_*_s4?", 10, 4),
)
# E73, the cross-family replication on Falcon3-1B: five domains x two arms x
# five seeds. Python factors and MathIR were relaunched as "_r1_" cohorts after
# the originals OOM-looped, so the replacement supersedes the original wherever
# it exists --- the same supersede-don't-pool rule the manuscript applies.
FALCON_DOMAINS: tuple[tuple[str, str], ...] = (
    ("Graph coloring", "gc"),
    ("Countdown", "cd"),
    ("Python factors", "py"),
    ("MathIR", "mi"),
    ("PantryPlan", "pp"),
)
FALCON_RUNS_PER_DOMAIN = 10

# E76 is deliberately separate from the fixed-recipe E73/E74 transfer rows.
# Its later stages do not exist until validation-only selectors release them,
# so a missing ledger means "gated", not "never started".
E76_STAGES: tuple[tuple[str, str, int], ...] = (
    ("E76A optimizer + stopping", "var/artifacts/e76_tuned_scale_stage_a_jobs.json", 48),
    ("E76B replay dose + form", "var/artifacts/e76_tuned_scale_stage_b_jobs.json", 28),
    ("E76C untouched-test confirm", "var/artifacts/e76_tuned_scale_stage_c_jobs.json", 36),
)
E76_SELECTIONS = {
    "E76A optimizer + stopping": "var/artifacts/e76_tuned_scale_stage_a_selection.json",
    "E76B replay dose + form": "var/artifacts/e76_tuned_scale_stage_b_selection.json",
}

# E77 is a deliberately small validation-only component screen. Its run count
# and optimizer horizons are frozen in the submission ledger, so the monitor
# reads those registered targets instead of inferring them from directory names.
E77_LEDGER = "var/artifacts/e77_fixed_component_screen_jobs.json"
E77_EXPECTED_RUNS = 12


HISTORY = "var/artifacts/e72_status_history.jsonl"
WAYPOINT_ARMS: tuple[tuple[str, str], ...] = (
    ("grpo", "grpo"),
    ("current", "verified_first_global_replay_canonical"),
    ("delayed", "verified_first_delayed_singleton_replay_canonical"),
)
WAYPOINT_EXPECTED_UPDATES = 64
ANT_STAGE_B_ARMS: tuple[str, str] = (
    "grpo",
    "verified_first_global_replay_canonical",
)
ANT_STAGE_B_SEEDS = range(43, 48)
ANT_STAGE_B_EXPECTED_UPDATES = 48
ANT_V19_EXPECTED_TIMESTEPS = 6_000_000
ANT_V19_TIMESTEP_RE = re.compile(
    rb"\|\s+total_timesteps\s+\|\s+([0-9]+)\s+\|"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def max_global_step(metrics_path: Path) -> int:
    """Deepest optimizer step this attempt reached, from a bounded tail read.

    The *last* logged step is the wrong measure: when a job is requeued it
    re-emits a step-0 evaluation before resuming from its checkpoint, so a run
    that reached step 3,000 and restarted five minutes ago reads as zero
    progress. Progress is monotone across attempts because training resumes
    from checkpointed state, so the maximum observed step is the honest one.
    """
    try:
        with metrics_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 200_000))
            tail = handle.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return 0
    best = 0
    for line in tail:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        step = record.get("misc/global_step")
        if step is not None:
            best = max(best, int(step))
    return best


def cohort_progress(
    root: Path, pattern: str, expected_runs: int, target_passes: int = 12
) -> dict[str, Any]:
    steps = 0
    progressing = 0
    complete = 0
    seen = 0
    over_configured = 0
    target = target_passes * STEPS_PER_PASS
    # The pattern is the whole run-directory glob, model tag included. It used
    # to be a suffix appended to a hardcoded 0.5B prefix, which silently matched
    # nothing the moment a second model size joined the campaign.
    for run_dir in sorted(root.glob(f"var/data/{pattern}")):
        best = max(
            (
                max_global_step(Path(path))
                for path in glob.glob(
                    str(run_dir / "debug_job*" / "train_metrics.jsonl")
                )
            ),
            default=0,
        )
        seen += 1
        over_configured += (
            configured_passes(run_dir.name, target_passes) > target_passes
        )
        # Credit is capped at the target: a run that trained deeper than the
        # cohort needs is done, not 130% done.
        steps += min(best, target)
        progressing += best > 0
        # Step-based, because a run stopped at the target never writes the
        # completion marker its longer configuration would have written.
        complete += best >= target or (run_dir / "TRAINING_COMPLETE.json").is_file()
    return {
        "steps": steps,
        "total_steps": expected_runs * target,
        "target_passes": target_passes,
        "runs_over_configured": over_configured,
        "runs_progressing": progressing,
        "runs_complete": complete,
        "runs_total": expected_runs,
    }


def falcon_progress(root: Path) -> dict[str, Any]:
    """Per-domain E73 progress, reported on the manuscript's own endpoint rule.

    The headline quantity is the *common* endpoint: the deepest step every run
    of a domain has reached, since a domain is reportable only when all ten
    cells reach the terminal pass. A mean over runs would look healthier than
    the domain actually is whenever one cell is stuck.
    """
    domains: list[dict[str, Any]] = []
    for title, prefix in FALCON_DOMAINS:
        replacement = sorted(
            root.glob(
                f"var/data/xdr_falcon3_1b_instruct_*_{prefix}e73_falcon3_1b_12pass_r1_*_s4[3-7]"
            )
        )
        original = [
            path
            for path in sorted(
                root.glob(
                    f"var/data/xdr_falcon3_1b_instruct_*_{prefix}e73_falcon3_1b_12pass_*_s4[3-7]"
                )
            )
            if "_r1_" not in path.name
        ]
        live = replacement if len(replacement) >= len(original) else original
        cohort = "r1" if live is replacement and replacement else "original"
        steps = []
        terminal = 0
        for run_dir in live:
            best = max(
                (
                    max_global_step(Path(path))
                    for path in glob.glob(
                        str(run_dir / "debug_job*" / "train_metrics.jsonl")
                    )
                ),
                default=0,
            )
            steps.append(best)
            terminal += best >= STEPS_PER_RUN
        # Absent runs count as zero so a short cohort cannot inflate the floor.
        while len(steps) < FALCON_RUNS_PER_DOMAIN:
            steps.append(0)
        domains.append(
            {
                "domain": title,
                "cohort": cohort,
                "runs_found": len(live),
                "terminal": terminal,
                "steps": sum(min(step, STEPS_PER_RUN) for step in steps),
                "total_steps": FALCON_RUNS_PER_DOMAIN * STEPS_PER_RUN,
                "common_endpoint": min(steps) if steps else 0,
                "deepest": max(steps) if steps else 0,
            }
        )
    return {
        "domains": domains,
        "runs_per_domain": FALCON_RUNS_PER_DOMAIN,
        "runs_complete": sum(entry["terminal"] for entry in domains),
        "runs_total": len(FALCON_DOMAINS) * FALCON_RUNS_PER_DOMAIN,
        "steps": sum(entry["steps"] for entry in domains),
        "total_steps": sum(entry["total_steps"] for entry in domains),
    }


def frontier_progress(root: Path) -> dict[str, Any]:
    per_stage = {}
    for stage, expected in FRONTIER_STAGES.items():
        stage_root = root / "var" / "data" / "e72_frontier" / stage
        # Count *cells*, not completion markers. A cell that was resubmitted
        # before its first attempt finished ends up with a marker under each
        # attempt directory, which counted twice and reported 121/120. The
        # aggregator already reads one result per cell, so this only affected
        # the progress line --- but a progress line that can exceed 100% is a
        # progress line nobody trusts.
        done = (
            len(
                {
                    marker.parent.parent
                    for marker in stage_root.glob("*/*/*/*/EVAL_ONLY_COMPLETE.json")
                }
            )
            if stage_root.is_dir()
            else 0
        )
        per_stage[stage] = {"done": done, "expected": expected}
    return {
        "per_stage": per_stage,
        "done": sum(entry["done"] for entry in per_stage.values()),
        "total": FRONTIER_TOTAL_CELLS,
    }



def e76_progress(root: Path) -> dict[str, Any]:
    """Progress for the validation-gated tuned-scale campaign.

    Targets come from the submission ledger because Stage B/C horizons are
    selected rather than fixed globally. This also prevents the monitor from
    inventing downstream work before its validation gate has opened.
    """
    stages: list[dict[str, Any]] = []
    for label, relative, expected in E76_STAGES:
        ledger_path = root / relative
        if not ledger_path.is_file():
            stages.append(
                {
                    "label": label,
                    "ledger": relative,
                    "released": False,
                    "runs_registered": 0,
                    "runs_expected": expected,
                    "runs_progressing": 0,
                    "runs_complete": 0,
                    "steps": 0,
                    "total_steps": 0,
                    "selected": False,
                    "controller_job_id": None,
                }
            )
            continue
        try:
            ledger = json.loads(ledger_path.read_text())
        except (OSError, ValueError):
            ledger = {"runs": [], "malformed": True}
        runs = ledger.get("runs", [])
        steps = 0
        total_steps = 0
        progressing = 0
        complete = 0
        for run in runs:
            run_dir = Path(run.get("run_dir", ""))
            target = int(run.get("target_steps", 0))
            best = max(
                (
                    max_global_step(Path(path))
                    for path in glob.glob(
                        str(run_dir / "debug_job*" / "train_metrics.jsonl")
                    )
                ),
                default=0,
            )
            total_steps += target
            steps += min(best, target)
            progressing += best > 0
            complete += bool(target and best >= target) or (
                run_dir / "TRAINING_COMPLETE.json"
            ).is_file()
        selection_path = E76_SELECTIONS.get(label)
        stages.append(
            {
                "label": label,
                "ledger": relative,
                "released": True,
                "malformed": bool(ledger.get("malformed")),
                "runs_registered": len(runs),
                "runs_expected": expected,
                "runs_progressing": progressing,
                "runs_complete": int(complete),
                "steps": steps,
                "total_steps": total_steps,
                "selected": bool(
                    selection_path and (root / selection_path).is_file()
                ),
                "controller_job_id": ledger.get("controller_job_id"),
            }
        )
    return {"stages": stages}


def e77_progress(root: Path) -> dict[str, Any]:
    """Measured progress for the fixed-component necessity screen."""
    ledger_path = root / E77_LEDGER
    if not ledger_path.is_file():
        return {
            "released": False,
            "malformed": False,
            "runs_registered": 0,
            "runs_expected": E77_EXPECTED_RUNS,
            "runs_progressing": 0,
            "runs_complete": 0,
            "steps": 0,
            "total_steps": 0,
        }
    try:
        ledger = json.loads(ledger_path.read_text())
    except (OSError, ValueError):
        ledger = {"runs": [], "malformed": True}
    runs = ledger.get("runs", [])
    if not isinstance(runs, list):
        runs = []
        ledger["malformed"] = True

    steps = 0
    total_steps = 0
    progressing = 0
    complete = 0
    domains: dict[str, dict[str, int]] = {}
    for run in runs:
        if not isinstance(run, dict):
            ledger["malformed"] = True
            continue
        run_dir = Path(str(run.get("run_dir", "")))
        target = int(run.get("target_steps", 0))
        best = max(
            (
                max_global_step(Path(path))
                for path in glob.glob(
                    str(run_dir / "debug_job*" / "train_metrics.jsonl")
                )
            ),
            default=0,
        )
        is_complete = bool(target and best >= target) or (
            run_dir / "TRAINING_COMPLETE.json"
        ).is_file()
        total_steps += target
        steps += min(best, target)
        progressing += best > 0
        complete += is_complete

        domain = str(run.get("domain", "unknown"))
        domain_entry = domains.setdefault(
            domain,
            {"runs_registered": 0, "runs_progressing": 0, "runs_complete": 0},
        )
        domain_entry["runs_registered"] += 1
        domain_entry["runs_progressing"] += best > 0
        domain_entry["runs_complete"] += is_complete

    return {
        "released": True,
        "malformed": bool(ledger.get("malformed")),
        "runs_registered": len(runs),
        "runs_expected": E77_EXPECTED_RUNS,
        "runs_progressing": progressing,
        "runs_complete": int(complete),
        "steps": steps,
        "total_steps": total_steps,
        "domains": domains,
    }



def max_waypoint_update(metrics_path: Path) -> int:
    """Deepest realized E75 online update in one arm's append-only metrics."""
    try:
        with metrics_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 1_000_000))
            rows = handle.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return 0
    best = 0
    for line in rows:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("schema") != "point-maze-waypoint-pilot-training-v1":
            continue
        best = max(best, int(record.get("learning_round", 0)))
    return best


def max_ant_stage_b_update(metrics_path: Path) -> int:
    """Deepest realized update in one frozen AntMaze Stage-B cell."""
    try:
        with metrics_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 1_000_000))
            rows = handle.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return 0
    best = 0
    for line in rows:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("schema") != "ant-maze-stage-b-training-metric-v1":
            continue
        best = max(best, int(record.get("learning_round", 0)))
    return best


def read_json_object(path: Path) -> dict[str, Any]:
    """Read one monitor artifact without letting a partial write break status."""
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return {"status": "invalid", "decision": "invalid_artifact"}
    return dict(payload) if isinstance(payload, dict) else {}


def max_ant_v19_timestep(log_path: Path) -> int:
    """Latest measured SB3 transition count from the bounded controller log."""
    try:
        with log_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 2_000_000))
            tail = handle.read()
    except OSError:
        return 0
    return max(
        (int(match) for match in ANT_V19_TIMESTEP_RE.findall(tail)),
        default=0,
    )


def scheduler_job_state(job_id: Any) -> str:
    """Current or terminal Slurm state for one frozen pipeline job."""
    if not isinstance(job_id, int):
        return "UNKNOWN"
    commands = (
        ["squeue", "-h", "-j", str(job_id), "-o", "%T"],
        [
            "sacct",
            "-X",
            "-n",
            "-j",
            str(job_id),
            "--format=State",
            "--parsable2",
        ],
    )
    for command in commands:
        try:
            output = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        for line in output.splitlines():
            state = line.strip().split("|", 1)[0].split("+", 1)[0]
            if state:
                return state.upper()
    return "UNKNOWN"


def ant_v19_progress(root: Path) -> dict[str, Any]:
    """Measured controller -> route -> 0.5B viability repair chain."""
    artifacts = root / "var/artifacts"
    controllers = root / "var/maze_runtime/controllers"
    controller_submission = read_json_object(
        artifacts / "ant_continuing_waypoint_controller_v19_submission.json"
    )
    if not controller_submission:
        return {}
    controller_identity = read_json_object(
        artifacts / "ant_continuing_waypoint_controller_v19_identity.json"
    )
    controller_evaluation = read_json_object(
        controllers / "ant_continuing_waypoint_v19.evaluation.json"
    )
    controller_job = controller_submission.get("job_id")
    controller_log = artifacts / f"logs/ant-cont-v19-{controller_job}.out"

    route_launcher = read_json_object(
        artifacts
        / "ant_maze_v15_controller_v19_dependent_launcher_identity.json"
    )
    route_submission = read_json_object(
        artifacts / "ant_maze_v15_controller_v19_admission_submission.json"
    )
    route_audit = read_json_object(
        artifacts
        / "ant_maze_modebench_v15_controller_v19_admission_audit.json"
    )
    route_job = route_submission.get("job_id")
    route_launcher_job = route_launcher.get("dependent_launcher_job_id")

    viability_preparer = read_json_object(
        artifacts / "ant_maze_v15_v19_viability_preparer_identity.json"
    )
    viability_launcher = read_json_object(
        artifacts
        / "ant_maze_v15_v19_viability_dependent_launcher_identity.json"
    )
    viability_submission = read_json_object(
        artifacts / "ant_maze_v15_controller_v19_viability_submission.json"
    )
    viability_result = read_json_object(
        artifacts / "ant_maze_v15_controller_v19_05b_viability.json"
    )
    viability_qualification = read_json_object(
        artifacts / "ant_maze_v15_controller_v19_qualification.json"
    )
    viability_job = viability_submission.get("job_id")
    viability_launcher_job = viability_launcher.get("dependent_launcher_job_id")
    viability_preparer_job = viability_preparer.get("preparer_job_id")

    return {
        "launched": True,
        "controller": {
            "job_id": controller_job,
            "job_state": scheduler_job_state(controller_job),
            "timesteps": min(
                max_ant_v19_timestep(controller_log),
                int(
                    controller_identity.get(
                        "timesteps", ANT_V19_EXPECTED_TIMESTEPS
                    )
                ),
            ),
            "expected_timesteps": int(
                controller_identity.get(
                    "timesteps", ANT_V19_EXPECTED_TIMESTEPS
                )
            ),
            "evaluation": controller_evaluation,
        },
        "route": {
            "launcher_job_id": route_launcher_job,
            "job_id": route_job,
            "job_state": scheduler_job_state(route_job or route_launcher_job),
            "audit": route_audit,
        },
        "viability": {
            "preparer_job_id": viability_preparer_job,
            "launcher_job_id": viability_launcher_job,
            "job_id": viability_job,
            "job_state": scheduler_job_state(
                viability_job or viability_launcher_job or viability_preparer_job
            ),
            "result": viability_result,
            "qualification": viability_qualification,
            "lm_sampled": bool(viability_result),
            "lm_job_launched": bool(viability_submission),
        },
    }


def ant_maze_progress(root: Path) -> dict[str, Any]:
    """Terminal Stage-B result plus conditional v18 and v19 extensions."""
    artifact_root = root / "var" / "artifacts"
    cells: dict[str, Any] = {}
    for arm in ANT_STAGE_B_ARMS:
        for seed in ANT_STAGE_B_SEEDS:
            stem = artifact_root / f"ant_maze_stage_b_05b_12pass_{arm}_s{seed}"
            metrics = Path(str(stem) + ".metrics.jsonl")
            receipt = Path(str(stem) + ".json")
            key = f"{arm}:s{seed}"
            cells[key] = {
                "arm": arm,
                "seed": seed,
                "updates": min(
                    max_ant_stage_b_update(metrics), ANT_STAGE_B_EXPECTED_UPDATES
                ),
                "complete": receipt.is_file(),
            }

    audit: dict[str, Any] = {}
    audit_path = artifact_root / "ant_maze_stage_b_05b_12pass_audit.json"
    if audit_path.is_file():
        try:
            audit = dict(json.loads(audit_path.read_text()))
        except (OSError, ValueError, TypeError):
            audit = {"status": "invalid"}

    controller: dict[str, Any] = {}
    controller_path = (
        root / "var/maze_runtime/controllers/ant_stable_handoff_v18.evaluation.json"
    )
    if controller_path.is_file():
        try:
            payload = json.loads(controller_path.read_text())
            summary = dict((payload.get("evaluation") or {}).get("summary") or {})
            controller = {
                "status": str(payload.get("status", "invalid")),
                "decision": str(payload.get("decision", "unknown")),
                "success_rate": summary.get("success_rate"),
                "minimum_heading_success_rate": summary.get(
                    "minimum_heading_success_rate"
                ),
                "minimum_map_success_rate": summary.get("minimum_map_success_rate"),
            }
        except (OSError, ValueError, TypeError):
            controller = {"status": "invalid", "decision": "invalid_artifact"}

    return {
        "launched": (
            artifact_root / "ant_maze_stage_b_05b_12pass_submission.json"
        ).is_file(),
        "cells": cells,
        "updates": sum(int(cell["updates"]) for cell in cells.values()),
        "expected_updates": (
            len(ANT_STAGE_B_ARMS)
            * len(ANT_STAGE_B_SEEDS)
            * ANT_STAGE_B_EXPECTED_UPDATES
        ),
        "complete": sum(bool(cell["complete"]) for cell in cells.values()),
        "audit": audit,
        "v18_controller": controller,
        "v19": ant_v19_progress(root),
    }


def waypoint_progress(
    root: Path,
    *,
    artifact_prefix: str = "e75",
    seed: int = 88501,
    data_name: str = "point_maze_waypoint_pilot_v1",
    sft_data_name: str = "point_maze_waypoint_warmstart_v1",
    sft_model_name: str = "point_maze_waypoint_warmstart_v1",
) -> dict[str, Any]:
    """Progress of one gated PointMaze data -> SFT -> dev -> online chain."""
    artifact_root = root / "var" / "artifacts"
    qualification_path = (
        artifact_root / f"{artifact_prefix}_point_maze_waypoint_dev_qualification.json"
    )
    qualification = "pending"
    qualification_observed: dict[str, Any] = {}
    qualification_reasons: list[str] = []
    if qualification_path.is_file():
        try:
            payload = json.loads(qualification_path.read_text())
            qualification = str(payload.get("status", "invalid"))
            qualification_observed = dict(payload.get("observed") or {})
            qualification_reasons = list(payload.get("reasons") or [])
        except (OSError, ValueError):
            qualification = "invalid"

    arms = {}
    for short, arm in WAYPOINT_ARMS:
        stem = artifact_root / f"{artifact_prefix}_point_maze_waypoint_{arm}_s{seed}"
        receipt = Path(str(stem) + ".json")
        metrics = Path(str(stem) + ".metrics.jsonl")
        arms[short] = {
            "arm": arm,
            "updates": min(max_waypoint_update(metrics), WAYPOINT_EXPECTED_UPDATES),
            "expected_updates": WAYPOINT_EXPECTED_UPDATES,
            "complete": receipt.is_file(),
        }

    jobs: dict[str, Any] = {}
    submission = artifact_root / f"{artifact_prefix}_point_maze_waypoint_05b_submission.json"
    if submission.is_file():
        try:
            jobs = dict(json.loads(submission.read_text()).get("jobs", {}))
        except (OSError, ValueError, TypeError):
            jobs = {}

    return {
        "launched": submission.is_file(),
        "data_certified": (
            root / "var/data" / data_name / "identity.json"
        ).is_file(),
        "train_routes_materialized": (
            root / "var/data" / sft_data_name / "identity.json"
        ).is_file(),
        "warmstart_complete": (
            artifact_root / f"{artifact_prefix}_point_maze_waypoint_warmstart.json"
        ).is_file()
        and (
            root / "var/models" / sft_model_name / "config.json"
        ).is_file(),
        "dev_gate": qualification,
        "dev_observed": qualification_observed,
        "dev_reasons": qualification_reasons,
        "arms": arms,
        "jobs": jobs,
    }


def waypoint_fullscale_progress(root: Path) -> dict[str, Any]:
    """Progress of the conditional E75F1 384/128, two-arm, five-seed cohort."""
    artifact_root = root / "var" / "artifacts"
    qualification_path = artifact_root / "e75f1_point_maze_waypoint_dev_qualification.json"
    qualification = "pending"
    observed: dict[str, Any] = {}
    if qualification_path.is_file():
        try:
            payload = json.loads(qualification_path.read_text())
            qualification = str(payload.get("status", "invalid"))
            observed = dict(payload.get("observed") or {})
        except (OSError, ValueError, TypeError):
            qualification = "invalid"

    cells: dict[str, Any] = {}
    for short, arm in (
        ("grpo", "grpo"),
        ("xmode", "verified_first_global_replay_canonical"),
    ):
        for seed in range(43, 48):
            key = f"{short}:s{seed}"
            stem = artifact_root / f"e75f1_point_maze_waypoint_{arm}_s{seed}"
            metrics = Path(str(stem) + ".metrics.jsonl")
            receipt = Path(str(stem) + ".json")
            cells[key] = {
                "arm": arm,
                "seed": seed,
                "updates": min(max_waypoint_update(metrics), 4608),
                "complete": receipt.is_file(),
            }

    jobs: dict[str, Any] = {}
    submission_path = artifact_root / "e75f1_point_maze_waypoint_05b_submission.json"
    if submission_path.is_file():
        try:
            jobs = dict(json.loads(submission_path.read_text()).get("jobs") or {})
        except (OSError, ValueError, TypeError):
            jobs = {}
    audit: dict[str, Any] = {}
    audit_path = artifact_root / "e75f1_point_maze_waypoint_05b_12pass_audit.json"
    if audit_path.is_file():
        try:
            audit = dict(json.loads(audit_path.read_text()))
        except (OSError, ValueError, TypeError):
            audit = {"status": "invalid"}
    return {
        "launched": submission_path.is_file(),
        "data_certified": (
            root / "var/data/point_maze_waypoint_e75f1/identity.json"
        ).is_file(),
        "dev_gate": qualification,
        "dev_observed": observed,
        "cells": cells,
        "updates": sum(int(cell["updates"]) for cell in cells.values()),
        "expected_updates": 46_080,
        "complete": sum(bool(cell["complete"]) for cell in cells.values()),
        "jobs": jobs,
        "audit": audit,
    }

def waypoint_final_progress(root: Path) -> dict[str, Any]:
    """One-shot E75R3 untouched-evaluation and paired-analysis state."""
    artifact_root = root / "var" / "artifacts"
    plan_path = artifact_root / "e75r3_point_maze_waypoint_final_eval_plan.json"
    submission_path = (
        artifact_root / "e75r3_point_maze_waypoint_final_eval_submission.json"
    )
    analysis_path = artifact_root / "e75r3_point_maze_waypoint_final_analysis.json"
    jobs: dict[str, Any] = {}
    if submission_path.is_file():
        try:
            jobs = dict(json.loads(submission_path.read_text()).get("jobs") or {})
        except (OSError, ValueError, TypeError):
            jobs = {}
    arms = {}
    for short, arm in WAYPOINT_ARMS:
        stem = artifact_root / f"e75r3_point_maze_waypoint_final_eval_{short}"
        arms[short] = {
            "arm": arm,
            "receipt": Path(str(stem) + ".json").is_file(),
            "metrics": Path(str(stem) + ".metrics.jsonl").is_file(),
        }
    analysis: dict[str, Any] = {}
    if analysis_path.is_file():
        try:
            analysis = dict(json.loads(analysis_path.read_text()))
        except (OSError, ValueError, TypeError):
            analysis = {"status": "invalid"}
    return {
        "launched": submission_path.is_file(),
        "plan_frozen": plan_path.is_file(),
        "arms": arms,
        "jobs": jobs,
        "analysis": analysis,
    }


def waypoint_gate_stop(entry: dict[str, Any]) -> str | None:
    """Compact terminal-gate result for the live status display."""
    if entry.get("dev_gate") not in {"fail", "invalid"}:
        return None
    observed = entry.get("dev_observed") or {}
    if not observed:
        return f"gate stop: {entry.get('dev_gate')} (no valid summary)"
    return (
        "gate stop: "
        f"pass maps={observed.get('maps_with_pass8', '?')}/32  "
        f"multi-route maps={observed.get('maps_with_two_routes', '?')}/32  "
        f"mean8={float(observed.get('mean8', 0.0)):.3f}"
    )


def queue_counts(user: str) -> dict[str, int]:
    """Live scheduler state, keyed by the job-name prefix each launcher uses."""
    try:
        out = subprocess.run(
            ["squeue", "-u", user, "-h", "-o", "%j %T"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    counts: dict[str, int] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        name, state = parts[0], parts[1]
        if name.startswith("e72b1aconf") or name.startswith("e72xgrpoconf"):
            key = "conf"
        elif name.startswith("e72b1b"):
            key = "b1b"
        elif name.startswith("e74"):
            key = "e74"
        elif "ant" in name:
            key = "ant"
        elif name.startswith("e76ctl"):
            key = "e76ctl"
        elif name.startswith("e76a"):
            key = "e76a"
        elif name.startswith("e76b"):
            key = "e76b"
        elif name.startswith("e76c"):
            key = "e76c"
        elif name.startswith("e77-"):
            key = "e77"
        elif name.startswith("e72b2bcal"):
            key = "b2bcal"
        elif name.startswith("e72b2b"):
            key = "b2b"
        elif name.startswith("e72b1a"):
            key = "b1a"
        elif name.startswith("e72b3a"):
            key = "b3a"
        elif name.startswith("e72f"):
            key = "cells"
        elif name.startswith("xdr_train"):
            key = "falcon"
        else:
            continue
        if state in ("RUNNING", "PENDING"):
            counts[f"{key}_{state.lower()}"] = (
                counts.get(f"{key}_{state.lower()}", 0) + 1
            )
    return counts


def pct(done: float, total: float) -> float:
    return 100.0 * done / total if total else 0.0


def elapsed_label(current: dict[str, Any], previous: dict[str, Any] | None) -> str:
    """How stale the comparison baseline is, for the change column header."""
    if not previous:
        return "was"
    minutes = (current["unix"] - previous.get("unix", current["unix"])) / 60
    if minutes < 90:
        return f"-{minutes:.0f}m"
    if minutes < 60 * 48:
        return f"-{minutes / 60:.1f}h"
    return f"-{minutes / 1440:.1f}d"


def render(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
    peak: dict[str, int] | None = None,
) -> str:
    frontier = current["frontier"]
    frontier_pct = pct(frontier["done"], frontier["total"])
    cohorts = current["cohorts"]

    previous_cohorts = (previous or {}).get("cohorts", {})
    # Deepest this cohort has *ever* been recorded at. "Launched then lost" has
    # to be judged against all of history, not against the baseline: with the
    # baseline a minute old, a cohort destroyed yesterday reads as never
    # started, which is the one thing the label exists to prevent.
    peak = peak or {}

    def prev_pct(name: str) -> float | None:
        entry = previous_cohorts.get(name)
        if not entry:
            return None
        return pct(entry["steps"], entry["total_steps"])

    lines = [
        "",
        f"  {'cohort':<30} {'now':>6}   {elapsed_label(current, previous):>6}   detail",
        f"  {'-' * 30} {'-' * 6}   {'-' * 6}   {'-' * 40}",
    ]

    done_gpu_hours = frontier_pct / 100 * FRONTIER_GPU_HOURS
    planned_gpu_hours = FRONTIER_GPU_HOURS

    for label, _, _, _ in COHORTS:
        entry = cohorts.get(label)
        if entry is None:
            continue
        now = pct(entry["steps"], entry["total_steps"])
        before = prev_pct(label)
        before_text = f"{before:5.1f}%" if before is not None else "    --"
        prev_entry = previous_cohorts.get(label) or {}
        if entry["runs_complete"] == entry["runs_total"]:
            state = "done"
        elif entry["runs_progressing"] == 0:
            # "not started" and "started, then lost" look identical on disk but
            # mean opposite things to whoever reads this. History tells them
            # apart: a cohort that once had steps and now has none was launched
            # and destroyed, which is a decision waiting rather than a to-do.
            state = (
                "ATTEMPTED, nothing retained"
                if peak.get(label, 0) > 0
                else "not started"
            )
        elif entry["steps"] < prev_entry.get("steps", 0):
            state = "running, LOST GROUND"
        else:
            state = "running"
        lines.append(
            f"  {label:<30} {now:5.1f}%   {before_text}   "
            f"{entry['runs_complete']}/{entry['runs_total']} to target, "
            f"{entry['steps']:,} steps ({state}"
            + (
                f", target {entry['target_passes']} passes"
                if entry.get("target_passes", 12) != 12
                else ""
            )
            + (
                f", {entry['runs_over_configured']} configured deeper"
                if entry.get("runs_over_configured")
                else ""
            )
            + ")"
        )
        gpu = entry["runs_total"] * HOURS_PER_TRAINING_RUN
        done_gpu_hours += now / 100 * gpu
        planned_gpu_hours += gpu

    falcon = current.get("falcon")
    if falcon and falcon["domains"]:
        now = pct(falcon["steps"], falcon["total_steps"])
        previous_falcon = (previous or {}).get("falcon") or {}
        before = (
            pct(previous_falcon["steps"], previous_falcon["total_steps"])
            if previous_falcon.get("total_steps")
            else None
        )
        before_text = f"{before:5.1f}%" if before is not None else "    --"
        state = (
            "done"
            if falcon["runs_complete"] == falcon["runs_total"]
            else "running"
        )
        lines.append(
            f"  {'E73 Falcon3-1B cross-family':<30} {now:5.1f}%   "
            f"{before_text}   {falcon['runs_complete']}/{falcon['runs_total']} "
            f"to target, {falcon['steps']:,} steps ({state})"
        )

    lines.append(
        f"  {'Frontier cells':<30} {frontier_pct:5.1f}%   "
        + (
            f"{pct(previous['frontier']['done'], previous['frontier']['total']):5.1f}%"
            if previous
            else "    --"
        )
        + "   "
        + "  ".join(
            f"{stage}:{entry['done']}/{entry['expected']}"
            for stage, entry in frontier["per_stage"].items()
        )
    )

    waypoint = current.get("waypoint")
    if waypoint:
        previous_waypoint = (previous or {}).get("waypoint") or {}
        previous_arms = previous_waypoint.get("arms", {})
        lines += [
            "",
            "  E75 PointMaze waypoint 0.5B",

            f"  {'-' * 30}",
            "  pipeline: "
            f"data={'done' if waypoint['data_certified'] else 'pending'}  "
            f"demos={'done' if waypoint['train_routes_materialized'] else 'pending'}  "
            f"sft={'done' if waypoint['warmstart_complete'] else 'pending'}  "
            f"dev={waypoint['dev_gate']}",
        ]
        gate_stop = waypoint_gate_stop(waypoint)
        if gate_stop:
            lines.append(f"  {gate_stop}")
        for short, _arm in WAYPOINT_ARMS:
            entry = waypoint["arms"][short]
            before = int(previous_arms.get(short, {}).get("updates", 0))
            moved = int(entry["updates"]) - before
            state = (
                "closed"
                if waypoint["dev_gate"] in {"fail", "invalid"}
                else "complete"
                if entry["complete"]
                else ("training" if entry["updates"] else "waiting")
            )
            lines.append(
                f"  {short:<10} {entry['updates']:>2}/{entry['expected_updates']} "
                f"updates ({state}, {moved:+} since last)"
            )
        if waypoint["jobs"]:
            job_labels = [
                ("prep", waypoint["jobs"].get("prepare")),
                ("sft", waypoint["jobs"].get("sft")),
                ("dev", waypoint["jobs"].get("dev")),
            ]
            job_labels.extend(
                (short, waypoint["jobs"].get(arm)) for short, arm in WAYPOINT_ARMS
            )
            lines.append(
                "  jobs: "
                + "  ".join(
                    f"{label}={job}" for label, job in job_labels if job is not None
                )
            )

    waypoint_r3 = current.get("waypoint_r3")
    if waypoint_r3 and waypoint_r3["launched"]:
        previous_waypoint = (previous or {}).get("waypoint_r3") or {}
        previous_arms = previous_waypoint.get("arms", {})
        lines += [
            "",
            "  E75R3 PointMaze calibrated warm start",
            f"  {'-' * 30}",
            "  pipeline: "
            f"data={'done' if waypoint_r3['data_certified'] else 'pending'}  "
            f"demos={'done' if waypoint_r3['train_routes_materialized'] else 'pending'}  "
            f"sft={'done' if waypoint_r3['warmstart_complete'] else 'pending'}"
            "/72 updates  "
            f"dev={waypoint_r3['dev_gate']}",
        ]
        gate_stop = waypoint_gate_stop(waypoint_r3)
        if gate_stop:
            lines.append(f"  {gate_stop}")
        for short, _arm in WAYPOINT_ARMS:
            entry = waypoint_r3["arms"][short]
            before = int(previous_arms.get(short, {}).get("updates", 0))
            moved = int(entry["updates"]) - before
            state = (
                "closed"
                if waypoint_r3["dev_gate"] in {"fail", "invalid"}
                else "complete"
                if entry["complete"]
                else ("training" if entry["updates"] else "waiting")
            )
            lines.append(
                f"  {short:<10} {entry['updates']:>2}/{entry['expected_updates']} "
                f"updates ({state}, {moved:+} since last)"
            )
        if waypoint_r3["jobs"]:
            job_labels = [
                ("prep", waypoint_r3["jobs"].get("prepare")),
                ("sft", waypoint_r3["jobs"].get("sft")),
                ("dev", waypoint_r3["jobs"].get("dev")),
            ]
            job_labels.extend(
                (short, waypoint_r3["jobs"].get(arm))
                for short, arm in WAYPOINT_ARMS
            )
            lines.append(
                "  jobs: "
                + "  ".join(
                    f"{label}={job}"
                    for label, job in job_labels
                    if job is not None
                )
            )

    waypoint_final = current.get("waypoint_final") or {}
    if waypoint_final.get("plan_frozen"):
        complete = sum(
            bool(entry.get("receipt"))
            for entry in waypoint_final.get("arms", {}).values()
        )
        analysis = waypoint_final.get("analysis") or {}
        analysis_state = (
            "complete"
            if analysis.get("status") == "complete"
            else "invalid"
            if analysis
            else "waiting"
        )
        lines += [
            "",
            "  E75R3 untouched final eval",
            "  " + "-" * 30,
            f"  plan=frozen  eval={complete}/3 complete  analysis={analysis_state}",
        ]
        if analysis.get("status") == "complete":
            summaries = analysis.get("arms") or {}
            summary_parts = []
            for short, arm in WAYPOINT_ARMS:
                metrics = (summaries.get(arm) or {}).get("metrics") or {}
                summary_parts.append(
                    short + ":pass=" + format(float(metrics.get("pass8", 0.0)), ".3f") + ","
                    + "distinct=" + format(float(metrics.get("distinct8", 0.0)), ".3f")
                )
            lines.append("  64 maps x 8: " + "  ".join(summary_parts))
            comparisons = analysis.get("comparisons") or {}
            paired_parts = []
            for label in ("current_minus_grpo", "delayed_minus_grpo"):
                metric = (
                    ((comparisons.get(label) or {}).get("metrics") or {})
                    .get("distinct8")
                    or {}
                )
                interval = metric.get("paired_map_bootstrap_95") or [0.0, 0.0]
                paired_parts.append(
                    label.replace("_minus_", "-") + "="
                    + format(float(metric.get("mean_delta", 0.0)), "+.3f") + " "
                    f"[{float(interval[0]):+.3f},{float(interval[1]):+.3f}]"
                )
            lines.append("  paired distinct8: " + "  ".join(paired_parts))
        jobs = waypoint_final.get("jobs") or {}
        if jobs:
            labels = [
                ("grpo", jobs.get("grpo")),
                ("current", jobs.get("verified_first_global_replay_canonical")),
                (
                    "delayed",
                    jobs.get(
                        "verified_first_delayed_singleton_replay_canonical"
                    ),
                ),
                ("analysis", jobs.get("analysis")),
            ]
            lines.append(
                "  jobs: "
                + "  ".join(
                    f"{label}={job}" for label, job in labels if job is not None
                )
            )

    e76 = (current.get("e76") or {}).get("stages", [])

    ant = current.get("ant_maze") or {}
    if ant.get("launched"):
        audit = ant.get("audit") or {}
        controller = ant.get("v18_controller") or {}
        v19 = ant.get("v19") or {}
        lines += [
            "",
            "  AntMaze 0.5B paper-scale record",
            "  " + "-" * 33,
            f"  Stage B: {ant['complete']}/10 terminal, "
            f"{ant['updates']:,}/{ant['expected_updates']:,} updates; "
            f"audit={audit.get('status', 'waiting')} "
            f"({audit.get('decision', 'decision pending')})",
        ]
        if controller:
            if controller.get("status") == "fail":
                success = controller.get("success_rate")
                heading = controller.get("minimum_heading_success_rate")
                detail = ""
                if success is not None and heading is not None:
                    detail = (
                        f"; success={float(success):.3f}, "
                        f"worst-heading={float(heading):.3f}"
                    )
                lines.append(
                    "  v15/v18 extension: stopped at controller gate "
                    f"({controller.get('decision', 'fail')}{detail}); "
                    "no LM cohort launched"
                )
            else:
                lines.append(
                    "  v15/v18 extension: controller gate="
                    f"{controller.get('status', 'pending')}"
                )
        if v19.get("launched"):
            v19_controller = v19.get("controller") or {}
            evaluation = v19_controller.get("evaluation") or {}
            controller_gate_stopped = evaluation.get("status") in {
                "fail",
                "invalid",
            }
            if evaluation:
                summary = dict(
                    (evaluation.get("evaluation") or {}).get("summary") or {}
                )
                detail = ""
                if (
                    summary.get("success_rate") is not None
                    and summary.get("minimum_heading_success_rate") is not None
                ):
                    detail = (
                        f"; success={float(summary['success_rate']):.3f}, "
                        "worst-heading="
                        f"{float(summary['minimum_heading_success_rate']):.3f}"
                    )
                lines.append(
                    "  v19 repair: controller gate="
                    f"{evaluation.get('status', 'invalid')} "
                    f"({evaluation.get('decision', 'decision pending')}{detail}); "
                    f"job={v19_controller.get('job_id')}"
                )
            else:
                timesteps = int(v19_controller.get("timesteps", 0))
                expected = int(
                    v19_controller.get(
                        "expected_timesteps", ANT_V19_EXPECTED_TIMESTEPS
                    )
                )
                lines.append(
                    "  v19 repair: controller="
                    f"{pct(timesteps, expected):.1f}% "
                    f"({timesteps:,}/{expected:,} transitions; "
                    f"{v19_controller.get('job_state', 'UNKNOWN')}; "
                    f"job={v19_controller.get('job_id')})"
                )

            route = v19.get("route") or {}
            route_audit = route.get("audit") or {}
            if controller_gate_stopped and not route_audit:
                lines.append(
                    "  v19 route admission: stopped at controller gate "
                    "(not launched; "
                    f"launcher={route.get('launcher_job_id')}); "
                    "no route sampled"
                )
            elif route_audit:
                lines.append(
                    "  v19 route admission: "
                    f"{route_audit.get('status', 'invalid')} "
                    f"({route_audit.get('decision', 'decision pending')}); "
                    f"job={route.get('job_id')}"
                )
            else:
                route_job = route.get("job_id")
                route_label = "job" if route_job else "launcher"
                lines.append(
                    "  v19 route admission: waiting on controller "
                    f"({route.get('job_state', 'UNKNOWN')}; "
                    f"{route_label}={route_job or route.get('launcher_job_id')})"
                )

            viability = v19.get("viability") or {}
            qualification = viability.get("qualification") or {}
            if controller_gate_stopped and not qualification:
                lines.append(
                    "  v19 0.5B viability: stopped at controller gate "
                    "(not launched); LM samples=none"
                )
            elif qualification:
                observed = qualification.get("summary") or {}
                rate = observed.get("verified_rate")
                rate_detail = (
                    f"; verified={float(rate):.3f}" if rate is not None else ""
                )
                lines.append(
                    "  v19 0.5B viability: "
                    f"{qualification.get('status', 'invalid')} "
                    f"({qualification.get('decision', 'decision pending')}"
                    f"{rate_detail}); job={viability.get('job_id')}"
                )
            elif viability.get("lm_job_launched"):
                lines.append(
                    "  v19 0.5B viability: LM dev gate "
                    f"{viability.get('job_state', 'UNKNOWN')}; "
                    f"job={viability.get('job_id')}"
                )
            else:
                waiting_job = (
                    viability.get("launcher_job_id")
                    or viability.get("preparer_job_id")
                )
                waiting_label = (
                    "launcher"
                    if viability.get("launcher_job_id")
                    else "preparer"
                )
                lines.append(
                    "  v19 0.5B viability: frozen, waiting on route "
                    f"({viability.get('job_state', 'UNKNOWN')}; "
                    f"{waiting_label}={waiting_job}); LM samples=none"
                )
    if e76:
        previous_e76 = {
            entry["label"]: entry
            for entry in ((previous or {}).get("e76") or {}).get("stages", [])
        }
        lines += [
            "",
            f"  {'E76 tuned-scale (separate study)':<30} {'now':>7}   "
            f"{elapsed_label(current, previous):>7}   gate / detail",
            f"  {'-' * 30} {'-' * 7}   {'-' * 7}   {'-' * 42}",
        ]
        for index, entry in enumerate(e76):
            if not entry["released"]:
                upstream = "E76A selection" if index == 1 else "E76B selection"
                lines.append(
                    f"  {entry['label']:<30} {'gated':>7}   {'--':>7}   "
                    f"waits for {upstream}"
                )
                continue
            now = pct(entry["steps"], entry["total_steps"])
            prior = previous_e76.get(entry["label"])
            before = (
                f"{pct(prior['steps'], prior['total_steps']):6.1f}%"
                if prior and prior.get("total_steps") else "     --"
            )
            if entry.get("malformed"):
                state = "MALFORMED LEDGER"
            elif entry["runs_registered"] != entry["runs_expected"]:
                state = (
                    f"INCOMPLETE LEDGER "
                    f"{entry['runs_registered']}/{entry['runs_expected']}"
                )
            elif entry.get("selected"):
                state = "selected; next gate released"
            elif entry["runs_complete"] == entry["runs_registered"]:
                state = "training done; selector queued/running"
            elif entry["runs_progressing"]:
                state = "running"
            else:
                state = "queued"
            lines.append(
                f"  {entry['label']:<30} {now:6.1f}%   {before}   "
                f"{entry['runs_complete']}/{entry['runs_registered']} to target, "
                f"{entry['steps']:,}/{entry['total_steps']:,} steps ({state})"
            )

    e77 = current.get("e77") or {}
    if e77.get("released"):
        previous_e77 = (previous or {}).get("e77") or {}
        now = pct(e77["steps"], e77["total_steps"])
        before = (
            f"{pct(previous_e77['steps'], previous_e77['total_steps']):6.1f}%"
            if previous_e77.get("total_steps")
            else "     --"
        )
        if e77.get("malformed"):
            state = "MALFORMED LEDGER"
        elif e77["runs_registered"] != e77["runs_expected"]:
            state = (
                "INCOMPLETE LEDGER "
                f"{e77['runs_registered']}/{e77['runs_expected']}"
            )
        elif e77["runs_complete"] == e77["runs_registered"]:
            state = "done"
        elif e77["runs_progressing"]:
            state = "running"
        else:
            state = "queued"
        domain_detail = ", ".join(
            f"{domain.replace('_', ' ')} "
            f"{entry['runs_complete']}/{entry['runs_registered']}"
            for domain, entry in sorted(e77.get("domains", {}).items())
        )
        lines += [
            "",
            f"  {'E77 fixed-component screen':<30} {now:6.1f}%   "
            f"{before}   {e77['runs_complete']}/{e77['runs_registered']} "
            f"to target, {e77['steps']:,}/{e77['total_steps']:,} steps "
            f"({state}; {domain_detail})",
        ]


    lines += [
        "",
        f"  launched work: {pct(done_gpu_hours, planned_gpu_hours):.1f}% "
        f"(~{done_gpu_hours:.0f} / {planned_gpu_hours:.0f} GPU-hours)",
        f"  Wave A + Tier 0: {pct(done_gpu_hours, WAVE_A_GPU_HOURS):.1f}% "
        f"(~{done_gpu_hours:.0f} / {WAVE_A_GPU_HOURS:.0f} GPU-hours, {WAVE_A_ARMS} arms)",
    ]

    queue = current.get("queue", {})
    if queue:
        lines.append(
            "  queue: "
            + "  ".join(
                f"{key.replace('_', ' ')} {value}"
                for key, value in sorted(queue.items())
            )
        )

    # A snapshot written before the multi-cohort schema has no per-cohort steps
    # to difference against; reporting a rate from it would invent one.
    if previous and previous_cohorts:
        hours = (current["unix"] - previous["unix"]) / 3600
        gained = sum(e["steps"] for e in cohorts.values()) - sum(
            e["steps"] for e in previous_cohorts.values()
        )
        remaining = sum(
            e["total_steps"] - e["steps"]
            for label, e in cohorts.items()
            if e["runs_progressing"] > 0 and e["runs_complete"] < e["runs_total"]
        )
        current_e76 = {
            entry["label"]: entry
            for entry in ((current.get("e76") or {}).get("stages") or [])
            if entry.get("released")
        }
        previous_e76 = {
            entry["label"]: entry
            for entry in (((previous or {}).get("e76") or {}).get("stages") or [])
            if entry.get("released")
        }
        for label, entry in current_e76.items():
            prior = previous_e76.get(label)
            if prior and prior.get("total_steps"):
                gained += entry["steps"] - prior["steps"]
            if (
                entry["runs_progressing"] > 0
                and entry["runs_complete"] < entry["runs_registered"]
            ):
                remaining += entry["total_steps"] - entry["steps"]
        current_e77 = current.get("e77") or {}
        previous_e77 = (previous or {}).get("e77") or {}
        if current_e77.get("released") and previous_e77.get("total_steps"):
            gained += current_e77["steps"] - previous_e77["steps"]
        if (
            current_e77.get("runs_progressing", 0) > 0
            and current_e77.get("runs_complete", 0)
            < current_e77.get("runs_registered", 0)
        ):
            remaining += current_e77["total_steps"] - current_e77["steps"]
        if hours > 0.01 and gained > 0:
            rate = gained / hours
            window = (
                f"{hours * 60:.0f} min" if hours < 1 else f"{hours:.1f} h"
            )
            eta = (
                f" -> ~{remaining / rate:.1f} h for cohorts in flight"
                if remaining
                else ""
            )
            lines.append(
                f"  rate:  {rate:,.0f} steps/hour over the last {window}{eta}"
            )
        elif hours > 0.01:
            window = (
                f"{hours * 60:.0f} min" if hours < 1 else f"{hours:.1f} h"
            )
            lines.append(f"  rate:  no training movement in the last {window}")

    # What is actually in the way. Everything above says how far things got;
    # this says what will not finish on its own, which is the question a status
    # check is usually being asked.
    blocked: list[str] = []
    for label, _, _, _ in COHORTS:
        entry = cohorts.get(label)
        if entry is None:
            continue
        prev_entry = previous_cohorts.get(label) or {}
        if entry["runs_progressing"] == 0 and peak.get(label, 0) > 0:
            blocked.append(
                f"{label.split()[0]}: launched but nothing on disk -- relaunch or drop it"
            )
        elif entry["steps"] < prev_entry.get("steps", 0):
            blocked.append(f"{label.split()[0]}: went backwards, runs are being lost")
        elif entry["runs_progressing"] == 0 and entry["runs_complete"] == 0:
            blocked.append(f"{label.split()[0]}: never started")
    for domain in (current.get("falcon") or {}).get("domains", []):
        short = FALCON_RUNS_PER_DOMAIN - domain["terminal"]
        if 0 < short <= 2:
            blocked.append(
                f"E73 {domain['domain']}: {short} run(s) short of a reportable domain"
            )
    if blocked:
        lines.append("")
        lines.append("  needs attention")
        lines.append(f"  {'-' * 30}")
        for item in blocked:
            lines.append(f"    - {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default=os.environ.get("USER", ""))
    parser.add_argument(
        "--since",
        type=float,
        default=None,
        help=(
            "compare against the newest snapshot at least this many hours old "
            "instead of the previous invocation; useful when status has been "
            "run several times in a row and the last one is minutes stale"
        ),
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="print without appending a snapshot (leaves the 'was' column alone)",
    )
    args = parser.parse_args()

    current = {
        "unix": time.time(),
        "cohorts": {
            label: cohort_progress(root, pattern, runs, passes)
            for label, pattern, runs, passes in COHORTS
        },
        "frontier": frontier_progress(root),
        "falcon": falcon_progress(root),
        "waypoint_f1": waypoint_fullscale_progress(root),
        "ant_maze": ant_maze_progress(root),
        "e76": e76_progress(root),
        "e77": e77_progress(root),
        "queue": queue_counts(args.user) if args.user else {},
    }

    history_path = root / HISTORY
    snapshots: list[dict[str, Any]] = []
    if history_path.is_file():
        for line in history_path.read_text().splitlines():
            try:
                snapshots.append(json.loads(line))
            except ValueError:
                continue
    # By default the baseline is simply the previous invocation. The column is
    # labelled with how long ago that was, so a row of zeros reads as "nothing
    # moved in four minutes" rather than as "nothing is moving".
    if args.since is None:
        previous = snapshots[-1] if snapshots else None
    else:
        cutoff = current["unix"] - args.since * 3600
        older = [s for s in snapshots if s.get("unix", 0) <= cutoff]
        previous = older[-1] if older else (snapshots[0] if snapshots else None)

    peak: dict[str, int] = {}
    for snapshot in snapshots:
        for label, entry in (snapshot.get("cohorts") or {}).items():
            peak[label] = max(peak.get(label, 0), int(entry.get("steps", 0)))

    print(render(current, previous, peak))

    if not args.no_record:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(current, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
