#!/usr/bin/env python3
"""Outcome-blind E111 audit for v7 discovery, Re:Dr, and entropy actuation."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_e102_full_open_bank_maxent_replay as shared  # noqa: E402
import audit_e108_admission_retention_mechanism_gate as retention_shared  # noqa: E402
import audit_e111_scheduler_amendments as scheduler_amendments  # noqa: E402
import audit_e111_proposal_retention_resume_recovery as resume_recovery  # noqa: E402
import audit_e111_qwen3_pantry_partial_checkpoint_quarantine as pantry_recovery  # noqa: E402
import audit_e111_qwen3_pantry_continuation as pantry_timeout  # noqa: E402
import audit_e111_qwen3_python_mathir_replacement as timeout_replacement  # noqa: E402
import launch_e111_verified_support_discovery_mechanism_gate_three_scale as launch  # noqa: E402
import status_e78 as status_shared  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / launch.LEDGER
OUT = ROOT / launch.AUDIT
OUTCOME_EXPOSURE_PROTOCOL = ROOT / (
    "paper/preregistration/"
    "e111_post_freeze_training_reward_exposure_deviation_20260818.md"
)
ADVANTAGE_BOUND = launch.SEMANTIC_COEFFICIENT + 1e-7
UNIFORM_TOLERANCE = 1e-7
RECOVERED_RESUME_FAILURE = (
    "proposal retention state refers to a non-proposal exemplar"
)
RECOVERED_PARTIAL_CHECKPOINT_FAILURE = (
    "PytorchStreamReader failed reading zip archive: failed finding central directory"
)
RECOVERED_TRITON_CLEANUP_FAILURE = (
    "FileNotFoundError: [Errno 2] No such file or directory: '/tmp/od2961'"
)
RECOVERED_PLASMA_CLEANUP_FAILURE = (
    "FileNotFoundError: [Errno 2] No such file or directory: '/tmp/test_plasma-"
)
FAILURE_MARKERS = (
    "Traceback (most recent call last)",
    "AssertionError",
    "CUDA out of memory",
    "OutOfMemoryError",
    "non-finite",
)


def scheduler_states(job_ids: list[int]) -> dict[int, str]:
    if not job_ids:
        return {}
    result = subprocess.run(
        [
            "sacct",
            "-n",
            "-P",
            "-j",
            ",".join(str(value) for value in job_ids),
            "--format=JobIDRaw,State",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    states: dict[int, str] = {}
    if result.returncode != 0:
        return states
    for line in result.stdout.splitlines():
        pieces = line.split("|")
        if len(pieces) >= 2 and pieces[0].isdigit():
            states.setdefault(int(pieces[0]), pieces[1])
    return states


def log_failures(path: Path) -> list[str]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    return [marker for marker in FAILURE_MARKERS if marker in text]


def unrecovered_stdout_failures(
    path: Path,
    *,
    job_id: int,
    recovered_occurrences: dict[str, int],
    partial_checkpoint_occurrences: dict[str, int],
    log_reinitialized_after: str | None = None,
) -> list[str]:
    """Return fatal stdout markers except exactly recorded recovery failures."""

    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    markers = [marker for marker in FAILURE_MARKERS if marker in text]
    observed = text.count(RECOVERED_RESUME_FAILURE)
    archived_expected = int(recovered_occurrences.get(str(job_id), 0))
    expected = archived_expected
    if log_reinitialized_after is not None:
        try:
            requeue_time = datetime.fromisoformat(log_reinitialized_after).timestamp()
        except ValueError:
            markers.append("invalid timeout-requeue timestamp")
        else:
            if path.stat().st_mtime > requeue_time:
                expected = 0
    observed_partial = text.count(RECOVERED_PARTIAL_CHECKPOINT_FAILURE)
    expected_partial = int(partial_checkpoint_occurrences.get(str(job_id), 0))
    tracebacks = text.count("Traceback (most recent call last)")
    recovered_cleanup_tracebacks = (
        text.count(RECOVERED_TRITON_CLEANUP_FAILURE)
        + text.count(RECOVERED_PLASMA_CLEANUP_FAILURE)
    )
    if observed != expected:
        markers.append(
            "proposal-retention resume failure count changed: "
            f"observed={observed} expected={expected}"
        )
    if observed_partial != expected_partial:
        markers.append(
            "partial-checkpoint failure count changed: "
            f"observed={observed_partial} expected={expected_partial}"
        )
    if (
        observed + observed_partial > 0
        and observed == expected
        and observed_partial == expected_partial
        and tracebacks
        == observed + observed_partial + recovered_cleanup_tracebacks
    ):
        markers = [
            marker
            for marker in markers
            if marker != "Traceback (most recent call last)"
        ]
    elif (
        observed + observed_partial == 0
        and recovered_cleanup_tracebacks > 0
        and tracebacks == recovered_cleanup_tracebacks
    ):
        markers = [
            marker
            for marker in markers
            if marker != "Traceback (most recent call last)"
        ]
    return markers


def completed_by_receipt(
    run_dir: Path,
    report: dict[str, Any],
    *,
    target_steps: int = launch.SMOKE_TARGET_STEPS,
) -> bool:
    return (
        status_shared.receipt_step(run_dir) >= target_steps
        and int(report.get("last_step", 0)) >= target_steps
    )


def parse_run(
    run_dir: Path,
    *,
    target_steps: int = launch.SMOKE_TARGET_STEPS,
) -> tuple[dict[str, Any], list[str]]:
    paths = shared.metric_paths(run_dir)
    report, violations = shared.parse_metrics(paths)
    # E111 preregisters uniform verified-likelihood Re:Dr with this E102-only
    # clipping mechanism disabled. Preserve every shared safety check except
    # the one that contradicts E111's frozen treatment identity.
    violations = [
        value
        for value in violations
        if value != "retention-safe balance was disabled on a replay update"
    ]
    v7_seen = False
    v7_active_min = 1.0
    replay_support_seen = False
    replay_support_active_min = 1.0
    v5_active_max = 0.0
    v5_seen = False
    v6_active_max = 0.0
    v6_seen = False
    controller_active_max = 0.0
    controller_seen = False
    semantic_advantage_abs_max = 0.0
    semantic_advantage_seen = False
    semantic_advantage_min = float("inf")
    semantic_advantage_max = float("-inf")
    semantic_both_sign_updates = 0
    semantic_rms_max = 0.0
    semantic_eligible_fraction_max = 0.0
    verified_support_size_max = 0.0
    support_at_least_two_eligible_fraction_max = 0.0
    external_support_size_max = 0.0
    external_support_nonempty_group_fraction_max = 0.0
    replay_gradient_l2_max = 0.0
    replay_groups_max = 0.0
    mass_weight_seen = False
    mass_weight_min = 1.0
    mass_weight_max = 1.0
    retention_tracking_seen = False
    retention_tracking_min = 1.0
    adaptive_seen = False
    adaptive_max = 0.0
    history_rows_added_max = 0.0

    for _path, _line_number, row in shared.rows(paths):
        if row.get("__invalid_json__"):
            continue
        v7 = shared.metric(
            row,
            "semantic_shannon_success_conditioned_verified_support_advantage_active",
        )
        if v7 is not None:
            v7_seen = True
            v7_active_min = min(v7_active_min, v7)
        include_replay = shared.metric(
            row,
            "semantic_shannon_verified_support_include_replay_bank_active",
        )
        if include_replay is not None:
            replay_support_seen = True
            replay_support_active_min = min(
                replay_support_active_min, include_replay
            )
        v5 = shared.metric(
            row,
            "semantic_shannon_success_conditioned_signed_advantage_active",
        )
        if v5 is not None:
            v5_seen = True
            v5_active_max = max(v5_active_max, abs(v5))
        v6 = shared.metric(
            row,
            "semantic_shannon_success_conditioned_group_centered_advantage_active",
        )
        if v6 is not None:
            v6_seen = True
            v6_active_max = max(v6_active_max, abs(v6))
        controller = shared.metric(row, "semantic_rms_controller_active")
        if controller is not None:
            controller_seen = True
            controller_active_max = max(controller_active_max, abs(controller))

        prefix = "semantic_shannon_success_conditioned_verified_support_"
        semantic_low = shared.metric(row, prefix + "effective_advantage_min")
        semantic_high = shared.metric(row, prefix + "effective_advantage_max")
        if semantic_low is not None and semantic_high is not None:
            semantic_advantage_seen = True
            semantic_advantage_min = min(semantic_advantage_min, semantic_low)
            semantic_advantage_max = max(semantic_advantage_max, semantic_high)
            semantic_both_sign_updates += int(semantic_low < 0.0 < semantic_high)
        for suffix in (
            prefix + "effective_advantage_min",
            prefix + "effective_advantage_max",
            "semantic_shannon_separate_semantic_advantage_min",
            "semantic_shannon_separate_semantic_advantage_max",
        ):
            semantic_advantage_abs_max = max(
                semantic_advantage_abs_max,
                abs(shared.metric(row, suffix) or 0.0),
            )
        semantic_rms_max = max(
            semantic_rms_max,
            shared.metric(row, prefix + "effective_advantage_rms") or 0.0,
        )
        semantic_eligible_fraction_max = max(
            semantic_eligible_fraction_max,
            shared.metric(row, prefix + "eligible_fraction") or 0.0,
        )
        verified_support_size_max = max(
            verified_support_size_max,
            shared.metric(row, prefix + "verified_support_size_mean") or 0.0,
        )
        support_at_least_two_eligible_fraction_max = max(
            support_at_least_two_eligible_fraction_max,
            shared.metric(
                row,
                prefix + "verified_support_at_least_two_eligible_fraction",
            )
            or 0.0,
        )
        external_support_size_max = max(
            external_support_size_max,
            shared.metric(
                row, prefix + "external_verified_support_size_mean"
            )
            or 0.0,
        )
        external_support_nonempty_group_fraction_max = max(
            external_support_nonempty_group_fraction_max,
            shared.metric(
                row,
                prefix + "external_verified_support_nonempty_group_fraction",
            )
            or 0.0,
        )
        history_rows_added_max = max(
            history_rows_added_max,
            shared.metric(row, prefix + "history_rows_added") or 0.0,
        )
        replay_gradient_l2_max = max(
            replay_gradient_l2_max,
            shared.metric(row, "canonical_replay_applied_score_gradient_l2")
            or 0.0,
        )
        groups = shared.metric(row, "canonical_replay_actuator_groups") or 0.0
        replay_groups_max = max(replay_groups_max, groups)
        if groups > 0.0:
            low = shared.metric(row, "canonical_replay_mass_weight_min")
            high = shared.metric(row, "canonical_replay_mass_weight_max")
            if low is not None and high is not None:
                mass_weight_seen = True
                mass_weight_min = min(mass_weight_min, low)
                mass_weight_max = max(mass_weight_max, high)
        tracking = retention_shared.retention_metric(row, "tracking_enabled")
        if tracking is not None:
            retention_tracking_seen = True
            retention_tracking_min = min(retention_tracking_min, tracking)
        adaptive = retention_shared.retention_metric(
            row, "adaptive_priority_enabled"
        )
        if adaptive is not None:
            adaptive_seen = True
            adaptive_max = max(adaptive_max, abs(adaptive))

    if report["last_step"] < target_steps:
        violations.append(
            f"only {report['last_step']}/{target_steps} optimizer steps"
        )
    if not v7_seen or v7_active_min < 1.0:
        violations.append("v7 verified-support estimator was inactive or absent")
    if not replay_support_seen or replay_support_active_min < 1.0:
        violations.append("replay-bank support inclusion was inactive or absent")
    if not v5_seen or v5_active_max > 0.0:
        violations.append("legacy v5 estimator was active or unreported")
    if not v6_seen or v6_active_max > 0.0:
        violations.append("v6 group-centered estimator was active or unreported")
    if not controller_seen or controller_active_max > 0.0:
        violations.append("semantic RMS controller was active or unreported")
    if semantic_advantage_abs_max > ADVANTAGE_BOUND:
        violations.append(
            f"semantic advantage exceeded eta bound: {semantic_advantage_abs_max}"
        )
    if report["priority_modes_updates"] > 0 or (
        report["priority_replay_groups_cumulative"] > 0
    ):
        violations.append("nonuniform proposal priority actuated")
    if mass_weight_seen and (
        abs(mass_weight_min - 1.0) > UNIFORM_TOLERANCE
        or abs(mass_weight_max - 1.0) > UNIFORM_TOLERANCE
    ):
        violations.append(
            f"Re:Dr mass weights were nonuniform: {mass_weight_min}, {mass_weight_max}"
        )
    if replay_groups_max > 0.0 and replay_gradient_l2_max <= 0.0:
        violations.append("materialized Re:Dr groups had no applied gradient")
    if not retention_tracking_seen or retention_tracking_min < 1.0:
        violations.append("diagnostic admission-retention tracking was absent")
    if not adaptive_seen or adaptive_max > 0.0:
        violations.append("adaptive proposal retention was active or unreported")

    report.update(
        {
            "v7_active_min": v7_active_min if v7_seen else 0.0,
            "replay_support_active_min": (
                replay_support_active_min if replay_support_seen else 0.0
            ),
            "v5_active_max": v5_active_max,
            "v6_active_max": v6_active_max,
            "controller_active_max": controller_active_max,
            "semantic_advantage_abs_max": semantic_advantage_abs_max,
            "semantic_advantage_min": (
                semantic_advantage_min if semantic_advantage_seen else 0.0
            ),
            "semantic_advantage_max": (
                semantic_advantage_max if semantic_advantage_seen else 0.0
            ),
            "semantic_both_sign_updates": semantic_both_sign_updates,
            "semantic_rms_max": semantic_rms_max,
            "semantic_eligible_fraction_max": semantic_eligible_fraction_max,
            "verified_support_size_max": verified_support_size_max,
            "support_at_least_two_eligible_fraction_max": (
                support_at_least_two_eligible_fraction_max
            ),
            "external_support_size_max": external_support_size_max,
            "external_support_nonempty_group_fraction_max": (
                external_support_nonempty_group_fraction_max
            ),
            "history_rows_added_max": history_rows_added_max,
            "replay_gradient_l2_max": replay_gradient_l2_max,
            "replay_groups_max": replay_groups_max,
            "mass_weight_seen": mass_weight_seen,
            "mass_weight_min": mass_weight_min,
            "mass_weight_max": mass_weight_max,
            "retention_tracking_min": (
                retention_tracking_min if retention_tracking_seen else 0.0
            ),
            "adaptive_priority_max": adaptive_max,
        }
    )
    return report, violations


def completed_causal_chain(report: dict[str, Any]) -> bool:
    return all(
        float(report.get(key, 0.0)) > 0.0
        for key in (
            "proposal_cumulative_admissions",
            "external_support_nonempty_group_fraction_max",
            "support_at_least_two_eligible_fraction_max",
            "semantic_rms_max",
            "replay_gradient_l2_max",
        )
    )


def main() -> int:
    if not LEDGER.is_file():
        raise SystemExit(f"E111 ledger is absent: {LEDGER}")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    violations: list[str] = []
    if ledger.get("schema") != (
        "e111_verified_support_discovery_mechanism_gate_jobs_v1"
    ):
        violations.append("unknown E111 ledger schema")
    if ledger.get("released") is not True:
        violations.append("E111 ledger was not atomically released")
    if ledger.get("pointmaze") != "excluded":
        violations.append("PointMaze exclusion drifted")
    if ledger.get("objective") != (
        "verified_support_v7_plus_uniform_replaydr_plus_support_discovery"
    ):
        violations.append("E111 objective identity drifted")
    if not OUTCOME_EXPOSURE_PROTOCOL.is_file():
        violations.append("post-freeze training-reward exposure record is absent")

    protocol = Path(str(ledger.get("protocol", "")))
    snapshot = Path(str(ledger.get("snapshot_root", "")))
    unit_path = Path(str(ledger.get("unit_evidence", "")))
    launcher_path = ROOT / (
        "ops/exp_scaling/"
        "launch_e111_verified_support_discovery_mechanism_gate_three_scale.py"
    )
    for path, expected, label in (
        (protocol, ledger.get("protocol_sha256"), "protocol"),
        (launcher_path, ledger.get("launcher_sha256"), "launcher"),
        (unit_path, ledger.get("unit_evidence_sha256"), "unit evidence"),
    ):
        if not path.is_file() or launch.digest(path) != expected:
            violations.append(f"frozen {label} digest mismatch")
    try:
        launch.verify_snapshot(snapshot)
    except (OSError, SystemExit) as exc:
        violations.append(f"snapshot validation failed: {exc}")

    if unit_path.is_file():
        try:
            unit = json.loads(unit_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            violations.append(f"unit evidence is invalid: {exc}")
        else:
            expected_unit = {
                "schema": "e111_verified_support_discovery_unit_tests_v1",
                "passed": True,
                "returncode": 0,
                "snapshot_root": str(snapshot),
                "outcomes_used": False,
                "pointmaze": "excluded",
            }
            for key, expected in expected_unit.items():
                if unit.get(key) != expected:
                    violations.append(f"unit evidence has invalid {key}")

    runs = ledger.get("runs", [])
    amendment_report, amendment_violations = scheduler_amendments.validate(
        ledger, runs
    )
    violations.extend(amendment_violations)
    recovery_report, recovery_violations = resume_recovery.validate()
    violations.extend(recovery_violations)
    pantry_recovery_report, pantry_recovery_violations = pantry_recovery.validate()
    violations.extend(pantry_recovery_violations)
    replacement_report, replacement_violations = timeout_replacement.validate(
        ledger, runs
    )
    violations.extend(replacement_violations)
    pantry_timeout_report, pantry_timeout_violations = pantry_timeout.validate(
        ledger, runs
    )
    violations.extend(pantry_timeout_violations)
    expected_cells = {
        (scale, domain)
        for scale in launch.SCALE_SEEDS
        for domain in launch.DOMAINS
    }
    observed_cells = {
        (str(run.get("scale")), str(run.get("domain"))) for run in runs
    }
    if len(runs) != 15 or observed_cells != expected_cells:
        violations.append("E111 ledger does not contain the frozen 3x5 cells")
    replacement_map = dict(
        replacement_report.get("continuation_by_original_job_id", {})
    )
    pantry_timeout_map = dict(
        pantry_timeout_report.get("continuation_by_original_job_id", {})
    )
    if set(replacement_map) & set(pantry_timeout_map):
        violations.append("E111 continuation maps overlap")
    replacement_map.update(pantry_timeout_map)
    effective_job_ids = [
        int(
            replacement_map.get(str(int(run["job_id"])), {}).get(
                "continuation_job_id", run["job_id"]
            )
        )
        for run in runs
    ]
    states = scheduler_states(effective_job_ids)
    recovered_occurrences = {
        str(key): int(value)
        for key, value in dict(
            recovery_report.get("trigger_occurrences", {})
        ).items()
    }
    partial_checkpoint_occurrences = {
        str(pantry_recovery_report.get("job_id")): int(
            pantry_recovery_report.get("partial_checkpoint_occurrences", 0)
        )
    }
    requeued_ids = {
        int(value)
        for value in amendment_report.get("exact_timeout_requeue_job_ids", [])
    }
    requeue_after = amendment_report.get("exact_timeout_requeue_recorded_after_at")
    reports: list[dict[str, Any]] = []
    for run in runs:
        original_job_id = int(run["job_id"])
        continuation = replacement_map.get(str(original_job_id), {})
        job_id = int(continuation.get("continuation_job_id", original_job_id))
        stdout = Path(str(continuation.get("stdout", run["stdout"])))
        stderr = Path(str(continuation.get("stderr", run["stderr"])))
        state = states.get(job_id, "UNKNOWN")
        run_dir = Path(str(run["run_dir"]))
        report, run_violations = parse_run(run_dir)
        if completed_by_receipt(run_dir, report):
            state = "COMPLETED"
        markers = log_failures(stderr)
        markers.extend(
            unrecovered_stdout_failures(
                stdout,
                job_id=job_id,
                recovered_occurrences=recovered_occurrences,
                partial_checkpoint_occurrences=partial_checkpoint_occurrences,
                log_reinitialized_after=(
                    str(requeue_after)
                    if job_id in requeued_ids and requeue_after is not None
                    else None
                ),
            )
        )
        if state.split("+", 1)[0] != "COMPLETED":
            run_violations.append(f"scheduler state is {state}")
        if markers:
            run_violations.append(f"failure markers in stderr: {markers}")
        reports.append(
            {
                "scale": str(run["scale"]),
                "domain": str(run["domain"]),
                "seed": int(run["seed"]),
                "original_job_id": original_job_id,
                "job_id": job_id,
                "effective_job_id": job_id,
                "continued_after_purged_timeout": job_id != original_job_id,
                "scheduler_state": state,
                "report": report,
                "completed_causal_chain": completed_causal_chain(report),
                "violations": run_violations,
            }
        )
        violations.extend(
            f"{run['scale']}/{run['domain']}: {value}"
            for value in run_violations
        )

    scale_chain_cells: dict[str, list[str]] = defaultdict(list)
    for row in reports:
        if row["completed_causal_chain"]:
            scale_chain_cells[row["scale"]].append(row["domain"])
    for scale in launch.SCALE_SEEDS:
        if not scale_chain_cells.get(scale):
            violations.append(
                f"{scale}: no cell completed discovery-to-Re:Dr-to-v7 actuation"
            )

    payload = {
        "schema": "e111_verified_support_discovery_mechanism_gate_audit_v1",
        "ledger": str(LEDGER),
        "terminal": all(
            row["scheduler_state"].split("+", 1)[0] == "COMPLETED"
            for row in reports
        ),
        "outcomes_used_for_gate": False,
        "post_freeze_training_reward_exposure": True,
        "outcome_exposure_protocol": str(OUTCOME_EXPOSURE_PROTOCOL),
        "outcome_exposure_protocol_sha256": (
            launch.digest(OUTCOME_EXPOSURE_PROTOCOL)
            if OUTCOME_EXPOSURE_PROTOCOL.is_file()
            else None
        ),
        "pointmaze": "excluded",
        "scheduler_amendments": amendment_report,
        "runtime_recovery_patch": recovery_report,
        "pantry_partial_checkpoint_recovery": pantry_recovery_report,
        "qwen3_pantry_timeout_continuation": pantry_timeout_report,
        "qwen3_python_mathir_timeout_replacement": replacement_report,
        "scale_chain_cells": dict(scale_chain_cells),
        "runs": reports,
        "violations": violations,
        "passed": not violations,
    }
    shared.atomic_json(OUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
