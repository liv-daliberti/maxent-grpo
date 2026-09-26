#!/usr/bin/env python3
"""Outcome-blind combined gate for E104 non-Python plus E106 Python cells."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_e102_full_open_bank_maxent_replay as shared  # noqa: E402
import audit_e104_group_centered_semantic_repair as e104_audit  # noqa: E402
import launch_e104_group_centered_semantic_repair_three_scale as e104  # noqa: E402
import launch_e106_python_lambda_normalization_three_scale as launch  # noqa: E402
import status_e78 as status  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / launch.AUDIT
CANCELLATION_NOTE = ROOT / (
    "paper/preregistration/"
    "e106_superseded_e104_falcon_python_cancellation_20260817.md"
)
QWEN3_PLACEMENT = ROOT / "var/artifacts/e106_qwen3_a6000_placement.json"
PRIOR_SURFACE_EVIDENCE = ROOT / "var/artifacts/e106_prior_surface_mode_evidence.json"
INTEGRATION_EVIDENCE = ROOT / (
    "var/artifacts/e106_parser_to_semantic_integration_tests.json"
)
CROSS_SCALE_SURFACE_EVIDENCE = ROOT / (
    "var/artifacts/e106_prior_python_surface_cross_scale_audit.json"
)
FALCON_BOOTSTRAP_EVIDENCE = ROOT / (
    "var/artifacts/e106_prior_falcon_python_bootstrap_audit.json"
)
EXTENDED_ESTIMATOR_EVIDENCE = ROOT / (
    "var/artifacts/e106_frozen_estimator_extended_tests.json"
)
POLICY_GRADIENT_EVIDENCE = ROOT / (
    "var/artifacts/e105_policy_gradient_direction_tests.json"
)
SEMANTIC_REPLAY_IDENTITY_EVIDENCE = ROOT / (
    "var/artifacts/e105_semantic_replay_identity_contract_tests.json"
)
TIME_LIMIT_AMENDMENT = ROOT / (
    "var/artifacts/e106_python_smoke_time_limit_amendment.json"
)
FALCON_POOL_AMENDMENT = ROOT / (
    "var/artifacts/e106_falcon_python_a6000_pool_amendment.json"
)
FALCON_ONE_HOUR_AMENDMENT = ROOT / (
    "var/artifacts/e106_falcon_one_hour_backfill_amendment.json"
)
FALCON_POOL_WIDENING_AMENDMENT = ROOT / (
    "var/artifacts/e106_falcon_same_gpu_pool_widening_amendment.json"
)
FALCON_ALL_POOL_AMENDMENT = ROOT / (
    "var/artifacts/e106_falcon_a6000_all_partition_pool_amendment.json"
)
QWEN3_CLEAN_RESTART_AMENDMENT = ROOT / (
    "var/artifacts/"
    "e106_qwen3_preemption_clean_restart_all_partition_amendment.json"
)
E110_LEDGER = ROOT / (
    "var/artifacts/e110_falcon_python_admission_horizon_jobs.json"
)
E110_TIME_LIMIT_AMENDMENT = ROOT / (
    "var/artifacts/e110_two_hour_backfill_amendment.json"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def python_admission_report(run_dir: Path) -> dict[str, float]:
    metrics = shared.metric_paths(run_dir)
    fields = {
        "bank_size_after_max": 0.0,
        "eligible_fraction_max": 0.0,
        "history_rows_added_max": 0.0,
        "available_groups_max": 0.0,
    }
    suffixes = {
        "bank_size_after_max": "online_canonical_bank_size_after_mean",
        "eligible_fraction_max": (
            "semantic_shannon_success_conditioned_group_centered_eligible_fraction"
        ),
        "history_rows_added_max": (
            "semantic_shannon_success_conditioned_group_centered_history_rows_added"
        ),
        "available_groups_max": "canonical_replay_available_groups",
    }
    for _path, _line, row in shared.rows(metrics):
        if row.get("__invalid_json__"):
            continue
        for key, suffix in suffixes.items():
            fields[key] = max(fields[key], shared.metric(row, suffix) or 0.0)
    return fields


def validate_unit_evidence(snapshot: Path) -> tuple[dict[str, Any], list[str]]:
    path = ROOT / launch.UNIT_EVIDENCE
    violations: list[str] = []
    if not path.is_file():
        return {}, ["E106 unit-test evidence is absent"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "e106_python_lambda_normalization_unit_tests_v1":
        violations.append("E106 unit-test evidence schema mismatch")
    if payload.get("passed") is not True or payload.get("returncode") != 0:
        violations.append("E106 unit tests did not pass")
    if payload.get("snapshot_root") != str(snapshot):
        violations.append("E106 unit tests used another snapshot")
    if payload.get("snapshot_sha256") != launch.SNAPSHOT_SHA256:
        violations.append("E106 unit-test snapshot hash mismatch")
    if "53 passed" not in str(payload.get("stdout", "")):
        violations.append("E106 unit tests lack the frozen 53-pass summary")
    return payload, violations


def validate_prior_surface_evidence() -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    if not PRIOR_SURFACE_EVIDENCE.is_file():
        return {}, ["E106 prior-response mode evidence is absent"]
    payload = json.loads(PRIOR_SURFACE_EVIDENCE.read_text(encoding="utf-8"))
    if payload.get("schema") != "e106_prior_surface_mode_evidence_v1":
        violations.append("E106 prior-response mode evidence schema mismatch")
    if payload.get("post_e104_or_e106_update_outcomes_inspected") is not False:
        violations.append("E106 prior-response mode evidence violated outcome blinding")
    if payload.get("verified_rows") != 3:
        violations.append("E106 prior-response evidence did not recover 3 verified rows")
    if payload.get("distinct_endpoint_keys") != 2:
        violations.append("E106 prior-response evidence lacks two endpoint modes")
    if payload.get("endpoint_multiplicities") != [2, 1]:
        violations.append("E106 prior-response endpoint multiplicities changed")
    source = ROOT / str(payload.get("input", ""))
    if not source.is_file() or payload.get("input_sha256") != digest(source):
        violations.append("E106 prior-response input digest mismatch")
    return payload, violations


def validate_integration_evidence(snapshot: Path) -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    if not INTEGRATION_EVIDENCE.is_file():
        return {}, ["E106 parser-to-semantic integration evidence is absent"]
    payload = json.loads(INTEGRATION_EVIDENCE.read_text(encoding="utf-8"))
    if payload.get("schema") != "e106_parser_to_semantic_integration_tests_v1":
        violations.append("E106 parser-to-semantic evidence schema mismatch")
    if payload.get("passed") is not True or payload.get("returncode") != 0:
        violations.append("E106 parser-to-semantic integration test did not pass")
    if payload.get("snapshot_root") != str(snapshot):
        violations.append("E106 parser-to-semantic test used another snapshot")
    if payload.get("snapshot_sha256") != launch.SNAPSHOT_SHA256:
        violations.append("E106 parser-to-semantic snapshot hash mismatch")
    if payload.get("post_e104_or_e106_update_outcomes_inspected") is not False:
        violations.append("E106 parser-to-semantic test violated outcome blinding")
    if payload.get("pointmaze") != "excluded":
        violations.append("E106 parser-to-semantic evidence does not exclude PointMaze")
    test_path = ROOT / str(payload.get("test", ""))
    if not test_path.is_file() or payload.get("test_sha256") != digest(test_path):
        violations.append("E106 parser-to-semantic test digest mismatch")
    expected_implementation = {
        relative: digest(snapshot / relative)
        for relative in (
            "src/oat_drgrpo/math_grader.py",
            "src/oat_drgrpo/semantic_shannon.py",
        )
    }
    if payload.get("implementation_sha256") != expected_implementation:
        violations.append("E106 parser-to-semantic implementation digests mismatch")
    if "1 passed" not in str(payload.get("stdout", "")):
        violations.append("E106 parser-to-semantic evidence lacks pass summary")
    return payload, violations


def validate_cross_scale_surface_evidence(
    snapshot: Path,
) -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    if not CROSS_SCALE_SURFACE_EVIDENCE.is_file():
        return {}, ["E106 cross-scale Python surface evidence is absent"]
    payload = json.loads(CROSS_SCALE_SURFACE_EVIDENCE.read_text(encoding="utf-8"))
    if payload.get("schema") != "e106_prior_python_surface_cross_scale_audit_v1":
        violations.append("E106 cross-scale Python surface schema mismatch")
    if payload.get("passed") is not True or payload.get("violations") != []:
        violations.append("E106 cross-scale Python surface audit did not pass")
    if payload.get("snapshot_root") != str(snapshot):
        violations.append("E106 cross-scale Python surface audit used another snapshot")
    if payload.get("snapshot_sha256") != launch.SNAPSHOT_SHA256:
        violations.append("E106 cross-scale Python surface snapshot hash mismatch")
    if payload.get("stored_scores_read") is not False:
        violations.append("E106 cross-scale Python surface audit read stored scores")
    if payload.get("post_e104_or_e106_update_outcomes_inspected") is not False:
        violations.append("E106 cross-scale Python surface audit violated blinding")
    if payload.get("pointmaze") != "excluded":
        violations.append("E106 cross-scale Python surface audit includes PointMaze")
    scales = payload.get("scales", {})
    qwen3 = scales.get("qwen3b", {})
    qwen3_totals = qwen3.get("totals", {})
    if qwen3_totals.get("recovered_verified") != 3:
        violations.append("E106 cross-scale audit did not recover 3 Qwen-3B rows")
    if qwen3.get("max_repaired_distinct_endpoint_keys_in_one_seed") != 2:
        violations.append("E106 cross-scale audit lacks two Qwen-3B endpoint modes")
    for scale in ("qwen05b", "falcon1b", "qwen3b"):
        report = scales.get(scale, {})
        if report.get("totals", {}).get("regressed_verified") != 0:
            violations.append(f"E106 cross-scale parser regressed {scale} rows")
        for item in report.get("files", []):
            source = ROOT / str(item.get("path", ""))
            if not source.is_file() or item.get("sha256") != digest(source):
                violations.append(f"E106 cross-scale input digest mismatch: {scale}")
    return payload, violations


def validate_falcon_bootstrap_evidence() -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    if not FALCON_BOOTSTRAP_EVIDENCE.is_file():
        return {}, ["E106 prior Falcon Python bootstrap evidence is absent"]
    payload = json.loads(FALCON_BOOTSTRAP_EVIDENCE.read_text(encoding="utf-8"))
    if payload.get("schema") != "e106_prior_falcon_python_bootstrap_audit_v1":
        violations.append("E106 prior Falcon bootstrap schema mismatch")
    if payload.get("passed") is not True or payload.get("violations") != []:
        violations.append("E106 prior Falcon bootstrap audit did not pass")
    if payload.get("evaluation_outcomes_read") is not False:
        violations.append("E106 prior Falcon bootstrap audit read outcomes")
    if payload.get("post_e104_or_e106_update_outcomes_inspected") is not False:
        violations.append("E106 prior Falcon bootstrap audit violated blinding")
    if payload.get("pointmaze") != "excluded":
        violations.append("E106 prior Falcon bootstrap audit includes PointMaze")
    runs = payload.get("runs", {})
    plain = runs.get("plain_grpo", {}).get("maxima", {})
    replay = runs.get("replay_semantic", {}).get("maxima", {})
    required = (
        (plain, "train/online_canonical_bank_size_after_mean", "plain bank"),
        (plain, "train/online_canonical_eligible_fraction", "plain admission"),
        (replay, "train/online_canonical_bank_size_after_mean", "replay bank"),
        (replay, "train/canonical_replay_available_groups", "replay groups"),
        (
            replay,
            "train/canonical_replay_applied_score_gradient_l2",
            "replay gradient",
        ),
        (
            replay,
            "train/semantic_shannon_success_conditioned_signed_eligible_fraction",
            "semantic eligibility",
        ),
    )
    for maxima, field, label in required:
        if float(maxima.get(field, 0.0)) <= 0.0:
            violations.append(f"E106 prior Falcon bootstrap lacks {label}")
    for name, report in runs.items():
        source = ROOT / str(report.get("path", ""))
        if not source.is_file() or report.get("sha256") != digest(source):
            violations.append(f"E106 prior Falcon bootstrap digest mismatch: {name}")
    return payload, violations


def validate_extended_estimator_evidence(
    snapshot: Path,
) -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    if not EXTENDED_ESTIMATOR_EVIDENCE.is_file():
        return {}, ["E106 extended frozen-estimator evidence is absent"]
    payload = json.loads(EXTENDED_ESTIMATOR_EVIDENCE.read_text(encoding="utf-8"))
    if payload.get("schema") != "e106_frozen_estimator_extended_tests_v1":
        violations.append("E106 extended estimator evidence schema mismatch")
    if payload.get("passed") is not True or payload.get("returncode") != 0:
        violations.append("E106 extended frozen-estimator tests did not pass")
    if payload.get("snapshot_root") != str(snapshot):
        violations.append("E106 extended estimator tests used another snapshot")
    if payload.get("snapshot_sha256") != launch.SNAPSHOT_SHA256:
        violations.append("E106 extended estimator snapshot hash mismatch")
    if payload.get("post_e104_or_e106_update_outcomes_inspected") is not False:
        violations.append("E106 extended estimator tests violated blinding")
    if payload.get("pointmaze") != "excluded":
        violations.append("E106 extended estimator evidence includes PointMaze")
    expected_tests = {
        relative: digest(ROOT / relative)
        for relative in (
            "tests/test_e106_parser_to_semantic_integration.py",
            "tests/test_semantic_shannon_group_centered.py",
            "tests/test_semantic_shannon_group_centered_theory.py",
        )
    }
    if payload.get("tests_sha256") != expected_tests:
        violations.append("E106 extended estimator test digests mismatch")
    expected_implementation = {
        relative: digest(snapshot / relative)
        for relative in (
            "src/oat_drgrpo/math_grader.py",
            "src/oat_drgrpo/semantic_shannon.py",
        )
    }
    if payload.get("implementation_sha256") != expected_implementation:
        violations.append("E106 extended estimator implementation digests mismatch")
    if "14 passed" not in str(payload.get("stdout", "")):
        violations.append("E106 extended estimator evidence lacks pass summary")
    return payload, violations


def validate_time_limit_amendment() -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    if not TIME_LIMIT_AMENDMENT.is_file():
        return {}, ["E106 Python smoke time-limit amendment record is absent"]
    payload = json.loads(TIME_LIMIT_AMENDMENT.read_text(encoding="utf-8"))
    if payload.get("schema") != "e106_python_smoke_time_limit_amendment_v1":
        violations.append("E106 Python smoke time-limit amendment schema mismatch")
    if payload.get("scheduler_only") is not True:
        violations.append("E106 Python smoke time-limit amendment was not scheduler-only")
    if payload.get("environment_changed") is not False:
        violations.append("E106 Python smoke time-limit amendment changed environment")
    if payload.get("post_e104_or_e106_update_outcomes_inspected") is not False:
        violations.append("E106 Python smoke time-limit amendment violated blinding")
    if payload.get("pointmaze") != "excluded":
        violations.append("E106 Python smoke time-limit amendment includes PointMaze")
    amendment = ROOT / str(payload.get("amendment", ""))
    if not amendment.is_file() or payload.get("amendment_sha256") != digest(amendment):
        violations.append("E106 Python smoke time-limit amendment digest mismatch")
    jobs = payload.get("jobs", [])
    if {item.get("job_id") for item in jobs} != {30640330, 30640331}:
        violations.append("E106 Python smoke time-limit amendment job set mismatch")
    for item in jobs:
        before = item.get("before", {})
        after = item.get("after", {})
        if before.get("time_limit") != "08:00:00":
            violations.append("E106 Python smoke prior time limit mismatch")
        if after.get("time_limit") != "02:00:00":
            violations.append("E106 Python smoke effective time limit mismatch")
        if before.get("runtime") != "00:00:00" or after.get("runtime") != "00:00:00":
            violations.append("E106 Python smoke time limit changed after runtime")
    return payload, violations


def validate_falcon_pool_amendment() -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    if not FALCON_POOL_AMENDMENT.is_file():
        return {}, ["E106 Falcon Python pool amendment record is absent"]
    payload = json.loads(FALCON_POOL_AMENDMENT.read_text(encoding="utf-8"))
    if payload.get("schema") != "e106_falcon_python_a6000_pool_amendment_v1":
        violations.append("E106 Falcon Python pool amendment schema mismatch")
    if payload.get("job_id") != 30640330:
        violations.append("E106 Falcon Python pool amendment names another job")
    if payload.get("scheduler_only") is not True:
        violations.append("E106 Falcon Python pool amendment was not scheduler-only")
    if payload.get("environment_changed") is not False:
        violations.append("E106 Falcon Python pool amendment changed environment")
    if payload.get("post_e104_or_e106_update_outcomes_inspected") is not False:
        violations.append("E106 Falcon Python pool amendment violated blinding")
    if payload.get("pointmaze") != "excluded":
        violations.append("E106 Falcon Python pool amendment includes PointMaze")
    amendment = ROOT / str(payload.get("amendment", ""))
    if not amendment.is_file() or payload.get("amendment_sha256") != digest(amendment):
        violations.append("E106 Falcon Python pool amendment digest mismatch")
    before = payload.get("before", {})
    effective = payload.get("effective", {})
    if before.get("node_list") != "node207":
        violations.append("E106 Falcon Python prior node constraint mismatch")
    if effective.get("node_list") != "node[205-207]":
        violations.append("E106 Falcon Python effective node pool mismatch")
    for field, expected in (
        ("gres", "gpu:a6000:1"),
        ("partition", "cs"),
        ("time_limit", "02:00:00"),
    ):
        if before.get(field) != expected or effective.get(field) != expected:
            violations.append(f"E106 Falcon Python pool changed {field}")
    if before.get("runtime") != "00:00:00" or effective.get("runtime") != "00:00:00":
        violations.append("E106 Falcon Python pool changed after runtime")
    if payload.get("proven_capacity_jobs") != {
        "node205": 30516431,
        "node206": 30516432,
        "node207": 30516430,
    }:
        violations.append("E106 Falcon Python pool capacity evidence mismatch")
    return payload, violations


def validate_falcon_one_hour_amendment() -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    if not FALCON_ONE_HOUR_AMENDMENT.is_file():
        return {}, ["E106 Falcon one-hour amendment record is absent"]
    payload = json.loads(FALCON_ONE_HOUR_AMENDMENT.read_text(encoding="utf-8"))
    if payload.get("schema") != "e106_falcon_one_hour_backfill_amendment_v1":
        violations.append("E106 Falcon one-hour amendment schema mismatch")
    if payload.get("scheduler_only") is not True:
        violations.append("E106 Falcon one-hour amendment was not scheduler-only")
    if payload.get("environment_changed") is not False:
        violations.append("E106 Falcon one-hour amendment changed environment")
    if payload.get("post_e104_or_e106_update_outcomes_inspected") is not False:
        violations.append("E106 Falcon one-hour amendment violated blinding")
    if payload.get("pointmaze") != "excluded":
        violations.append("E106 Falcon one-hour amendment includes PointMaze")
    if payload.get("old_time_limit") != "02:00:00" or payload.get(
        "new_time_limit"
    ) != "01:00:00":
        violations.append("E106 Falcon one-hour limits drifted")
    for path_key, digest_key in (
        ("protocol", "protocol_sha256"),
        ("script", "script_sha256"),
        ("e104_ledger", "e104_ledger_sha256"),
        ("e106_ledger", "e106_ledger_sha256"),
    ):
        path = ROOT / str(payload.get(path_key, ""))
        if not path.is_file() or payload.get(digest_key) != digest(path):
            violations.append(f"E106 Falcon one-hour digest mismatch: {path_key}")
    reference = payload.get("reference", {})
    if reference.get("job_id") != 30637791 or reference.get(
        "state"
    ) != "COMPLETED":
        violations.append("E106 Falcon one-hour reference job drifted")
    if reference.get("elapsed_seconds") != 1_206:
        violations.append("E106 Falcon one-hour reference runtime drifted")
    if float(reference.get("one_hour_margin", 0.0)) < 2.8:
        violations.append("E106 Falcon one-hour runtime margin is insufficient")
    jobs = payload.get("jobs", [])
    if {item.get("job_id") for item in jobs} != {
        30637790,
        30637793,
        30637794,
        30640330,
    }:
        violations.append("E106 Falcon one-hour target set drifted")
    for item in jobs:
        before = str(item.get("before", ""))
        after = str(item.get("after", ""))
        if "JobState=PENDING" not in before or "RunTime=00:00:00" not in before:
            violations.append("E106 Falcon one-hour target was not untouched before")
        if "TimeLimit=02:00:00" not in before:
            violations.append("E106 Falcon one-hour prior limit mismatch")
        if "JobState=PENDING" not in after or "RunTime=00:00:00" not in after:
            violations.append("E106 Falcon one-hour target started during amendment")
        if "TimeLimit=01:00:00" not in after:
            violations.append("E106 Falcon one-hour effective limit mismatch")
    return payload, violations


def validate_policy_gradient_evidence(
    snapshot: Path,
) -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    if not POLICY_GRADIENT_EVIDENCE.is_file():
        return {}, ["E105 policy-gradient direction evidence is absent"]
    payload = json.loads(POLICY_GRADIENT_EVIDENCE.read_text(encoding="utf-8"))
    if payload.get("schema") != "e105_policy_gradient_direction_tests_v1":
        violations.append("E105 policy-gradient direction schema mismatch")
    if payload.get("passed") is not True or payload.get("returncode") != 0:
        violations.append("E105 policy-gradient direction test did not pass")
    if payload.get("snapshot_root") != str(snapshot):
        violations.append("E105 policy-gradient direction snapshot path mismatch")
    if payload.get("snapshot_sha256") != launch.SNAPSHOT_SHA256:
        violations.append("E105 policy-gradient direction snapshot hash mismatch")
    if payload.get("post_e104_or_e106_update_outcomes_inspected") is not False:
        violations.append("E105 policy-gradient direction evidence violated blinding")
    if payload.get("pointmaze") != "excluded":
        violations.append("E105 policy-gradient direction evidence includes PointMaze")
    for path_key, digest_key in (
        ("test", "test_sha256"),
        ("script", "script_sha256"),
    ):
        path = ROOT / str(payload.get(path_key, ""))
        if not path.is_file() or payload.get(digest_key) != digest(path):
            violations.append(
                f"E105 policy-gradient direction digest mismatch: {path_key}"
            )
    expected_assertions = {
        "common_verified_mode_logit_decreases": True,
        "rare_verified_mode_logit_increases": True,
        "failed_mode_direct_gradient_is_zero": True,
        "centered_gradient_sum_is_zero": True,
        "production_ppo_ratio_loss_executed": True,
        "combined_production_replay_backward_executed": True,
        "replay_verified_mass_increases": True,
        "replay_failure_mass_decreases": True,
        "replay_preserves_semantic_rare_tilt": True,
        "microbatch_geometry_invariant": True,
    }
    if payload.get("assertions") != expected_assertions:
        violations.append("E105 policy-gradient direction assertions drifted")
    return payload, violations


def validate_semantic_replay_identity_evidence(
    snapshot: Path,
) -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    if not SEMANTIC_REPLAY_IDENTITY_EVIDENCE.is_file():
        return {}, ["E105 semantic/replay identity evidence is absent"]
    payload = json.loads(
        SEMANTIC_REPLAY_IDENTITY_EVIDENCE.read_text(encoding="utf-8")
    )
    if (
        payload.get("schema")
        != "e105_semantic_replay_identity_contract_tests_v1"
    ):
        violations.append("E105 semantic/replay identity schema mismatch")
    if payload.get("passed") is not True or payload.get("returncode") != 0:
        violations.append("E105 semantic/replay identity test did not pass")
    if payload.get("snapshot_root") != str(snapshot):
        violations.append("E105 semantic/replay identity snapshot path mismatch")
    if payload.get("snapshot_sha256") != launch.SNAPSHOT_SHA256:
        violations.append("E105 semantic/replay identity snapshot hash mismatch")
    if payload.get("post_e104_or_e106_update_outcomes_inspected") is not False:
        violations.append("E105 semantic/replay identity evidence violated blinding")
    if payload.get("pointmaze") != "excluded":
        violations.append("E105 semantic/replay identity evidence includes PointMaze")
    if payload.get("domains") != [
        "graph_coloring",
        "countdown",
        "python_factors",
        "mathir",
        "pantry_plan",
    ]:
        violations.append("E105 semantic/replay identity domain set drifted")
    if payload.get("group_size") != 16:
        violations.append("E105 semantic/replay identity group size drifted")
    for path_key, digest_key in (
        ("test", "test_sha256"),
        ("script", "script_sha256"),
    ):
        path = ROOT / str(payload.get(path_key, ""))
        if not path.is_file() or payload.get(digest_key) != digest(path):
            violations.append(
                f"E105 semantic/replay identity digest mismatch: {path_key}"
            )
    expected_implementation = {
        relative: digest(snapshot / relative)
        for relative in (
            "src/oat_drgrpo/learner/grpo.py",
            "src/oat_drgrpo/learner/run.py",
            "src/oat_drgrpo/math_grader.py",
            "src/oat_drgrpo/online_canonical_bank.py",
            "src/oat_drgrpo/semantic_shannon.py",
        )
    }
    if payload.get("implementation_sha256") != expected_implementation:
        violations.append("E105 semantic/replay implementation digests mismatch")
    expected_assertions = {
        "actor_reward_matches_replay_validator_admission": True,
        "semantic_and_replay_verified_keys_match": True,
        "semantic_and_replay_persistent_counts_match": True,
        "parseable_task_failures_are_excluded": True,
        "inactive_verified_rows_are_excluded": True,
        "rare_verified_mode_has_positive_semantic_pressure": True,
        "common_verified_mode_has_negative_semantic_pressure": True,
        "verified_replay_retains_both_modes": True,
        "joint_auto_resume_is_exact": True,
    }
    if payload.get("assertions") != expected_assertions:
        violations.append("E105 semantic/replay identity assertions drifted")
    if "6 passed" not in str(payload.get("stdout", "")):
        violations.append("E105 semantic/replay identity evidence lacks pass summary")
    return payload, violations


def validate_falcon_pool_widening_amendment() -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    if not FALCON_POOL_WIDENING_AMENDMENT.is_file():
        return {}, ["E106 Falcon same-GPU pool-widening record is absent"]
    payload = json.loads(
        FALCON_POOL_WIDENING_AMENDMENT.read_text(encoding="utf-8")
    )
    if payload.get("schema") != (
        "e106_falcon_same_gpu_pool_widening_amendment_v1"
    ):
        violations.append("E106 Falcon pool-widening schema mismatch")
    for key, expected in (
        ("scheduler_only", True),
        ("environment_changed", False),
        ("gpu_type_changed", False),
        ("post_e104_or_e106_update_outcomes_inspected", False),
        ("pointmaze", "excluded"),
    ):
        if payload.get(key) != expected:
            violations.append(f"E106 Falcon pool-widening field drifted: {key}")
    for path_key, digest_key in (
        ("protocol", "protocol_sha256"),
        ("script", "script_sha256"),
        ("e104_ledger", "e104_ledger_sha256"),
        ("one_hour_amendment", "one_hour_amendment_sha256"),
    ):
        path = ROOT / str(payload.get(path_key, ""))
        if not path.is_file() or payload.get(digest_key) != digest(path):
            violations.append(
                f"E106 Falcon pool-widening digest mismatch: {path_key}"
            )
    expected = {
        30637790: ("node202", "node[202-204]", "gpu:a5000:1"),
        30637793: ("node202", "node[202-204]", "gpu:a5000:1"),
        30637794: ("node206", "node[205-207]", "gpu:a6000:1"),
    }
    jobs = payload.get("jobs", [])
    if {item.get("job_id") for item in jobs} != set(expected):
        violations.append("E106 Falcon pool-widening target set drifted")
    for item in jobs:
        job_id = int(item.get("job_id", -1))
        if job_id not in expected:
            continue
        old, new, gres = expected[job_id]
        if (
            item.get("old_node_list"),
            item.get("new_node_list"),
            item.get("gres"),
        ) != (old, new, gres):
            violations.append(f"E106 Falcon pool-widening placement drifted: {job_id}")
        before = str(item.get("before", ""))
        after = str(item.get("after", ""))
        for record, node_list in ((before, old), (after, new)):
            if (
                "JobState=PENDING" not in record
                or "RunTime=00:00:00" not in record
                or "TimeLimit=01:00:00" not in record
                or f"ReqNodeList={node_list}" not in record
                or f"TresPerNode=gres/{gres}" not in record
            ):
                violations.append(
                    f"E106 Falcon pool-widening scheduler record drifted: {job_id}"
                )
    return payload, violations


def validate_falcon_all_pool_amendment() -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    if not FALCON_ALL_POOL_AMENDMENT.is_file():
        return {}, ["E106 Falcon all-partition A6000 record is absent"]
    payload = json.loads(FALCON_ALL_POOL_AMENDMENT.read_text(encoding="utf-8"))
    if payload.get("schema") != (
        "e106_falcon_a6000_all_partition_pool_amendment_v1"
    ):
        violations.append("E106 Falcon all-partition pool schema mismatch")
    for key, expected in (
        ("scheduler_only", True),
        ("environment_changed", False),
        ("gpu_type_changed", False),
        ("partition_changed", True),
        ("account_changed", False),
        ("post_e104_or_e106_update_outcomes_inspected", False),
        ("pointmaze", "excluded"),
    ):
        if payload.get(key) != expected:
            violations.append(
                f"E106 Falcon all-partition pool field drifted: {key}"
            )
    for path_key, digest_key in (
        ("protocol", "protocol_sha256"),
        ("script", "script_sha256"),
        ("e104_ledger", "e104_ledger_sha256"),
        ("e106_ledger", "e106_ledger_sha256"),
        ("prior_pool_amendment", "prior_pool_amendment_sha256"),
    ):
        path = ROOT / str(payload.get(path_key, ""))
        if not path.is_file() or payload.get(digest_key) != digest(path):
            violations.append(
                f"E106 Falcon all-partition digest mismatch: {path_key}"
            )
    expected_nodes = [
        "node103",
        "node104",
        "node205",
        "node206",
        "node207",
        "node208",
        "node805",
    ]
    if payload.get("authorized_nodes") != expected_nodes:
        violations.append("E106 Falcon all-partition node set drifted")
    inventory = str(payload.get("inventory", ""))
    for node in expected_nodes:
        if f"{node}|all|gpu:a6000:" not in inventory:
            violations.append(
                f"E106 Falcon all-partition inventory drifted: {node}"
            )
    jobs = payload.get("jobs", [])
    if {item.get("job_id") for item in jobs} != {30637794, 30640330}:
        violations.append("E106 Falcon all-partition target set drifted")
    for item in jobs:
        job_id = int(item.get("job_id", -1))
        if job_id not in {30637794, 30640330}:
            continue
        expected_domain = (
            "pantry_plan" if job_id == 30637794 else "python_factors"
        )
        expected_source = "e104" if job_id == 30637794 else "e106"
        if (
            item.get("scale"),
            item.get("domain"),
            item.get("source"),
            item.get("old_partition"),
            item.get("new_partition"),
            item.get("old_node_list"),
            item.get("new_node_list"),
            item.get("gres"),
        ) != (
            "falcon1b",
            expected_domain,
            expected_source,
            "cs",
            "all",
            "node[205-207]",
            "node[103-104,205-208,805]",
            "gpu:a6000:1",
        ):
            violations.append(
                f"E106 Falcon all-partition placement drifted: {job_id}"
            )
        before = str(item.get("before", ""))
        after = str(item.get("after", ""))
        for record, partition, node_list in (
            (before, "cs", "node[205-207]"),
            (after, "all", "node[103-104,205-208,805]"),
        ):
            if (
                "JobState=PENDING" not in record
                or "RunTime=00:00:00" not in record
                or "TimeLimit=01:00:00" not in record
                or f"Partition={partition}" not in record
                or f"ReqNodeList={node_list}" not in record
                or "TresPerNode=gres/gpu:a6000:1" not in record
            ):
                violations.append(
                    "E106 Falcon all-partition scheduler record drifted: "
                    f"{job_id}"
                )
    return payload, violations


def validate_qwen3_clean_restart_amendment() -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    if not QWEN3_CLEAN_RESTART_AMENDMENT.is_file():
        return {}, ["E106 Qwen-3B clean-restart amendment is absent"]
    payload = json.loads(
        QWEN3_CLEAN_RESTART_AMENDMENT.read_text(encoding="utf-8")
    )
    if payload.get("schema") != (
        "e106_qwen3_preemption_clean_restart_all_partition_amendment_v1"
    ):
        violations.append("E106 Qwen-3B clean-restart schema mismatch")
    for key, expected in (
        ("released", True),
        ("job_id", 30640331),
        ("scale", "qwen3b"),
        ("domain", "python_factors"),
        ("seed", 70),
        ("scheduler_only", True),
        ("clean_restart_required", True),
        ("interrupted_step", 22),
        ("checkpoint_present", False),
        ("old_partition", "lowprio"),
        ("new_partition", "all"),
        ("node_list", "node[103-104,205-208,805]"),
        ("gpu_type_changed", False),
        ("environment_changed", False),
        ("scientific_configuration_changed", False),
        ("post_e104_or_e106_update_outcomes_inspected", False),
        ("pointmaze", "excluded"),
        ("snapshot_sha256", launch.SNAPSHOT_SHA256),
    ):
        if payload.get(key) != expected:
            violations.append(f"E106 Qwen-3B clean-restart field drifted: {key}")
    for path_key, digest_key in (
        ("protocol", "protocol_sha256"),
        ("script", "script_sha256"),
        ("e106_ledger", "e106_ledger_sha256"),
        ("prior_placement", "prior_placement_sha256"),
        ("time_limit_amendment", "time_limit_amendment_sha256"),
        ("resume_identity_evidence", "resume_identity_evidence_sha256"),
    ):
        path = ROOT / str(payload.get(path_key, ""))
        if not path.is_file() or payload.get(digest_key) != digest(path):
            violations.append(
                f"E106 Qwen-3B clean-restart digest mismatch: {path_key}"
            )
    expected_nodes = [
        "node103",
        "node104",
        "node205",
        "node206",
        "node207",
        "node208",
        "node805",
    ]
    if payload.get("authorized_nodes") != expected_nodes:
        violations.append("E106 Qwen-3B clean-restart node set drifted")
    inventory = str(payload.get("inventory", ""))
    for node in expected_nodes:
        if f"{node}|all|gpu:a6000:" not in inventory:
            violations.append(
                f"E106 Qwen-3B clean-restart inventory drifted: {node}"
            )
    expected_files = {
        "debug_job30640331/eval_results/0_multi_answer.json",
        "debug_job30640331/eval_results/16_multi_answer.json",
        "debug_job30640331/eval_mode_coverage_draws.jsonl",
        "debug_job30640331/train_metrics.jsonl",
    }
    manifest = payload.get("archived_prefix_manifest", [])
    if {item.get("path") for item in manifest} != expected_files:
        violations.append("E106 Qwen-3B archived-prefix manifest drifted")
    archive = Path(str(payload.get("archive_dir", "")))
    run_dir = Path(str(payload.get("run_dir", "")))
    expected_archive = Path(f"{run_dir}_preempted_no_checkpoint_step22_restart1")
    if archive != expected_archive or not archive.is_dir():
        violations.append("E106 Qwen-3B interrupted-prefix archive is absent")
    else:
        for item in manifest:
            path = archive / str(item.get("path", ""))
            if (
                not path.is_file()
                or item.get("size") != path.stat().st_size
                or item.get("sha256") != digest(path)
            ):
                violations.append(
                    "E106 Qwen-3B interrupted-prefix file drifted: "
                    f"{item.get('path')}"
                )
    before = str(payload.get("before", ""))
    held = str(payload.get("held", ""))
    amended = str(payload.get("amended_held", ""))
    released = str(payload.get("released_scheduler_record", ""))
    for record, partition, held_state in (
        (before, "lowprio", False),
        (held, "lowprio", True),
        (amended, "all", True),
        (released, "all", False),
    ):
        required = (
            "JobState=PENDING",
            "RunTime=00:00:00",
            "TimeLimit=02:00:00",
            "Restarts=1",
            "ExitCode=0:0",
            f"Partition={partition}",
            "ReqNodeList=node[103-104,205-208,805]",
            "TresPerNode=gres/gpu:a6000:1",
        )
        if any(needle not in record for needle in required) or (
            held_state and "Reason=JobHeldUser" not in record
        ):
            violations.append(
                "E106 Qwen-3B clean-restart scheduler record drifted: "
                f"{partition}/held={held_state}"
            )
    return payload, violations


def validate_e110_replacement_ledger() -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    if not E110_LEDGER.is_file():
        return {}, ["E110 Falcon Python replacement ledger is absent"]
    payload = json.loads(E110_LEDGER.read_text(encoding="utf-8"))
    for key, expected in (
        ("schema", "e110_falcon_python_admission_horizon_jobs_v1"),
        ("released", True),
        ("snapshot_sha256", launch.SNAPSHOT_SHA256),
        ("supersedes_job_id", 30640330),
        ("replacement_scope", ["falcon1b", "python_factors", 55]),
        ("models", ["falcon1b"]),
        ("domains", ["python_factors"]),
        ("seeds", [55]),
        ("train_rows", 192),
        ("passes", 1),
        ("target_steps", 192),
        ("checkpoint_interval_steps", 64),
        ("submitted_partition", "cs"),
        ("effective_partition", "all"),
        ("mechanism_gate_used_outcome_metrics", False),
        ("post_update_outcome_metrics_inspected", False),
        ("pointmaze", "excluded"),
    ):
        if payload.get(key) != expected:
            violations.append(f"E110 replacement field drifted: {key}")
    for path_key, digest_key in (
        ("protocol", "protocol_sha256"),
        ("launcher", "launcher_sha256"),
        ("base_e79_ledger", "base_e79_ledger_sha256"),
        ("base_e106_ledger", "base_e106_ledger_sha256"),
    ):
        path = ROOT / str(payload.get(path_key, ""))
        if not path.is_file() or payload.get(digest_key) != digest(path):
            violations.append(f"E110 replacement digest mismatch: {path_key}")
    historical = payload.get("historical_admission_evidence", {})
    if (
        historical.get("job_id") != 30269053
        or historical.get("first_admission_step") != 179
        or historical.get("mechanism_only") is not True
        or historical.get("evaluation_outcomes_inspected") is not False
        or historical.get("surface")
        != {
            "model": "Falcon3-1B-Instruct",
            "domain": "python_factors",
            "seed": 55,
            "prompt_template": "falcon_boxed",
            "generate_max_length": 512,
            "num_samples": 16,
        }
    ):
        violations.append("E110 historical admission evidence drifted")
    historical_metrics = Path(str(historical.get("metrics", "")))
    if (
        not historical_metrics.is_file()
        or historical.get("metrics_sha256") != digest(historical_metrics)
    ):
        violations.append("E110 historical admission metrics drifted")
    failed = payload.get("failed_e106_mechanism_evidence", {})
    if (
        failed.get("job_id") != 30640330
        or failed.get("last_step") != 64
        or failed.get("bank_size_after_max") != 0.0
        or failed.get("history_rows_added_max") != 0.0
        or failed.get("replay_gradient_l2_max") != 0.0
        or failed.get("mechanism_only") is not True
        or failed.get("evaluation_outcomes_inspected") is not False
    ):
        violations.append("E110 failed-E106 diagnosis drifted")
    for path_key, digest_key in (
        ("metrics", "metrics_sha256"),
        ("completion_marker", "completion_marker_sha256"),
    ):
        path = Path(str(failed.get(path_key, "")))
        if not path.is_file() or failed.get(digest_key) != digest(path):
            violations.append(f"E110 failed-E106 evidence drifted: {path_key}")
    cancelled = payload.get("cancelled_zero_runtime_attempt", {})
    cancelled_record = str(cancelled.get("scheduler_record", ""))
    if cancelled.get("job_id") != 30647351 or any(
        needle not in cancelled_record
        for needle in (
            "JobState=CANCELLED",
            "RunTime=00:00:00",
            "Reason=JobHeldUser",
            "Partition=cs",
            "--partition=all",
        )
    ):
        violations.append("E110 cancelled zero-runtime attempt drifted")
    submitted = str(payload.get("submitted_held_scheduler_record", ""))
    if any(
        needle not in submitted
        for needle in (
            "JobState=PENDING",
            "Reason=JobHeldUser",
            "RunTime=00:00:00",
            "Partition=cs",
        )
    ):
        violations.append("E110 submit-time scheduler record drifted")
    runs = payload.get("runs", [])
    if len(runs) != 1:
        violations.append("E110 replacement run count drifted")
    else:
        run = runs[0]
        if (
            run.get("scale"),
            run.get("domain"),
            run.get("seed"),
            run.get("run_stamp"),
        ) != (
            "falcon1b",
            "python_factors",
            55,
            "e110_falcon1b_python_admission_horizon_s55",
        ):
            violations.append("E110 replacement scientific cell drifted")
        held = str(run.get("held_scheduler_record", ""))
        required = (
            "JobState=PENDING",
            "Reason=JobHeldUser",
            "RunTime=00:00:00",
            "Partition=all",
            "Account=allcs",
            "ReqNodeList=node[103-104,205-208,805]",
            "TresPerNode=gres/gpu:a6000:1",
            "TimeLimit=03:00:00",
            "OAT_ZERO_MAX_TRAIN=192",
            "OAT_ZERO_GENERATE_MAX_LENGTH=512",
            "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.1",
            "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.1",
        )
        if any(needle not in held for needle in required):
            violations.append("E110 effective held scheduler record drifted")
    return payload, violations


def validate_e110_time_limit_amendment() -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    if not E110_TIME_LIMIT_AMENDMENT.is_file():
        return {}, ["E110 two-hour backfill amendment is absent"]
    payload = json.loads(E110_TIME_LIMIT_AMENDMENT.read_text(encoding="utf-8"))
    for key, expected in (
        ("schema", "e110_two_hour_backfill_amendment_v1"),
        ("released", True),
        ("job_id", 30647379),
        ("old_time_limit", "03:00:00"),
        ("new_time_limit", "02:00:00"),
        ("scheduler_only", True),
        ("environment_changed", False),
        ("scientific_configuration_changed", False),
        ("post_e104_or_e106_or_e110_update_outcomes_inspected", False),
        ("pointmaze", "excluded"),
    ):
        if payload.get(key) != expected:
            violations.append(f"E110 time-limit field drifted: {key}")
    for path_key, digest_key in (
        ("protocol", "protocol_sha256"),
        ("script", "script_sha256"),
        ("e110_ledger", "e110_ledger_sha256"),
    ):
        path = ROOT / str(payload.get(path_key, ""))
        if not path.is_file() or payload.get(digest_key) != digest(path):
            violations.append(f"E110 time-limit digest mismatch: {path_key}")
    references = payload.get("runtime_references", {})
    expected_references = {
        "e106_v6_64_step": ("30640330", "00:39:03", "node208"),
        "e79_replay_3072_step": ("30269053", "11:45:28", "node207"),
    }
    for key, (job_id, elapsed, node) in expected_references.items():
        reference = references.get(key, {})
        if (
            reference.get("job_id"),
            reference.get("elapsed"),
            reference.get("node"),
            reference.get("state"),
            reference.get("exit_code"),
        ) != (job_id, elapsed, node, "COMPLETED", "0:0") or (
            "gres/gpu:a6000=1" not in str(reference.get("req_tres", ""))
            or "mem=64G" not in str(reference.get("req_tres", ""))
        ):
            violations.append(f"E110 runtime reference drifted: {key}")
    for record_key, limit, held in (
        ("before", "03:00:00", False),
        ("held", "03:00:00", True),
        ("amended_held", "02:00:00", True),
        ("released_scheduler_record", "02:00:00", False),
    ):
        record = str(payload.get(record_key, ""))
        required = (
            "JobState=PENDING",
            "RunTime=00:00:00",
            f"TimeLimit={limit}",
            "Partition=all",
            "Account=allcs",
            "ReqNodeList=node[103-104,205-208,805]",
            "TresPerNode=gres/gpu:a6000:1",
            "OAT_ZERO_MAX_TRAIN=192",
            "OAT_ZERO_GENERATE_MAX_LENGTH=512",
            "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.1",
            "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.1",
        )
        if any(needle not in record for needle in required) or (
            held and "Reason=JobHeldUser" not in record
        ):
            violations.append(f"E110 time-limit scheduler record drifted: {record_key}")
    return payload, violations


def main() -> int:
    ledger_path = ROOT / launch.LEDGER
    base_path = ROOT / launch.BASE_LEDGER
    violations: list[str] = []
    if not ledger_path.is_file():
        raise SystemExit(f"E106 ledger is absent: {ledger_path}")
    if not base_path.is_file():
        raise SystemExit(f"E104 base ledger is absent: {base_path}")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    base = json.loads(base_path.read_text(encoding="utf-8"))
    snapshot = Path(str(ledger.get("snapshot_root", ""))).resolve()
    try:
        launch.verify_snapshot(ROOT, snapshot)
    except SystemExit as exc:
        violations.append(str(exc))
    if ledger.get("schema") != "e106_python_lambda_normalization_three_scale_jobs_v1":
        violations.append("E106 ledger schema mismatch")
    if ledger.get("released") is not True:
        violations.append("E106 jobs were not durably released")
    for key, relative in (
        ("protocol_sha256", launch.PROTOCOL),
        ("diagnosis_sha256", launch.DIAGNOSIS),
        ("launcher_sha256", "ops/exp_scaling/launch_e106_python_lambda_normalization_three_scale.py"),
        ("base_e104_ledger_sha256", launch.BASE_LEDGER),
    ):
        path = ROOT / relative
        if not path.is_file() or ledger.get(key) != digest(path):
            violations.append(f"E106 provenance digest mismatch: {key}")
    diagnosis = json.loads((ROOT / launch.DIAGNOSIS).read_text(encoding="utf-8"))
    if diagnosis.get("post_e104_update_outcomes_inspected") is not False:
        violations.append("E106 diagnosis outcome blinding was violated")
    if not CANCELLATION_NOTE.is_file():
        violations.append("E106 superseded-job cancellation note is absent")
    else:
        note = CANCELLATION_NOTE.read_text(encoding="utf-8")
        for marker in (
            "30637792",
            "30640330",
            "00:00:00",
            "non-Python E104 job was cancelled",
        ):
            if marker not in note:
                violations.append(f"E106 cancellation note lacks {marker!r}")
    qwen3_placement: dict[str, Any] = {}
    if not QWEN3_PLACEMENT.is_file():
        violations.append("E106 Qwen-3B placement record is absent")
    else:
        qwen3_placement = json.loads(QWEN3_PLACEMENT.read_text(encoding="utf-8"))
        if qwen3_placement.get("schema") != "e106_qwen3_a6000_placement_v1":
            violations.append("E106 Qwen-3B placement schema mismatch")
        if qwen3_placement.get("job_id") != 30640331:
            violations.append("E106 Qwen-3B placement names another job")
        if qwen3_placement.get("environment_changed") is not False:
            violations.append("E106 Qwen-3B placement changed the environment")
        if qwen3_placement.get("post_update_outcomes_inspected") is not False:
            violations.append("E106 Qwen-3B placement inspected outcomes")
        for path_key, digest_key in (
            ("amendment", "amendment_sha256"),
            ("capacity_preflight_ledger", "capacity_preflight_ledger_sha256"),
            ("capacity_preflight_audit", "capacity_preflight_audit_sha256"),
        ):
            evidence = ROOT / str(qwen3_placement.get(path_key, ""))
            if (
                not evidence.is_file()
                or qwen3_placement.get(digest_key) != digest(evidence)
            ):
                violations.append(f"E106 Qwen-3B placement digest mismatch: {path_key}")
    unit, unit_violations = validate_unit_evidence(snapshot)
    violations.extend(unit_violations)
    unit_path = ROOT / launch.UNIT_EVIDENCE
    if (
        not unit_path.is_file()
        or ledger.get("unit_evidence_sha256") != digest(unit_path)
    ):
        violations.append("E106 ledger unit-test evidence digest mismatch")
    prior_surface, prior_surface_violations = validate_prior_surface_evidence()
    violations.extend(prior_surface_violations)
    integration, integration_violations = validate_integration_evidence(snapshot)
    violations.extend(integration_violations)
    cross_scale, cross_scale_violations = validate_cross_scale_surface_evidence(
        snapshot
    )
    violations.extend(cross_scale_violations)
    falcon_bootstrap, falcon_bootstrap_violations = (
        validate_falcon_bootstrap_evidence()
    )
    violations.extend(falcon_bootstrap_violations)
    extended, extended_violations = validate_extended_estimator_evidence(snapshot)
    violations.extend(extended_violations)
    policy_gradient, policy_gradient_violations = (
        validate_policy_gradient_evidence(snapshot)
    )
    violations.extend(policy_gradient_violations)
    semantic_replay_identity, semantic_replay_identity_violations = (
        validate_semantic_replay_identity_evidence(snapshot)
    )
    violations.extend(semantic_replay_identity_violations)
    time_limit, time_limit_violations = validate_time_limit_amendment()
    violations.extend(time_limit_violations)
    falcon_pool, falcon_pool_violations = validate_falcon_pool_amendment()
    violations.extend(falcon_pool_violations)
    falcon_one_hour, falcon_one_hour_violations = (
        validate_falcon_one_hour_amendment()
    )
    violations.extend(falcon_one_hour_violations)
    falcon_pool_widening, falcon_pool_widening_violations = (
        validate_falcon_pool_widening_amendment()
    )
    violations.extend(falcon_pool_widening_violations)
    falcon_all_pool, falcon_all_pool_violations = (
        validate_falcon_all_pool_amendment()
    )
    violations.extend(falcon_all_pool_violations)
    qwen3_clean_restart, qwen3_clean_restart_violations = (
        validate_qwen3_clean_restart_amendment()
    )
    violations.extend(qwen3_clean_restart_violations)
    e110, e110_violations = validate_e110_replacement_ledger()
    violations.extend(e110_violations)
    e110_time, e110_time_violations = validate_e110_time_limit_amendment()
    violations.extend(e110_time_violations)

    e106_runs = list(ledger.get("runs", []))
    e104_runs = [run for run in base.get("runs", []) if run.get("domain") != launch.DOMAIN]
    if len(e106_runs) != 3:
        violations.append(f"E106 expected 3 Python cells, found {len(e106_runs)}")
    if len(e104_runs) != 12:
        violations.append(f"combined gate expected 12 non-Python E104 cells, found {len(e104_runs)}")
    failed_falcon = [
        run for run in e106_runs if int(run.get("job_id", -1)) == 30640330
    ]
    e110_runs = list(e110.get("runs", []))
    if len(failed_falcon) != 1 or len(e110_runs) != 1:
        violations.append("combined gate Falcon/Python replacement set drifted")
    effective_e106_runs = [
        run for run in e106_runs if int(run.get("job_id", -1)) != 30640330
    ] + e110_runs
    if len(effective_e106_runs) != 3:
        violations.append("combined gate expected three effective Python cells")
    all_runs = e104_runs + effective_e106_runs
    states = e104_audit.scheduler_states([int(run["job_id"]) for run in all_runs])
    reports: list[dict[str, Any]] = []
    nonzero_semantic_scales: set[str] = set()
    completed_zero_semantic_cells: list[dict[str, Any]] = []

    for source, runs in (
        ("e104_non_python", e104_runs),
        (
            "e106_python",
            [
                run
                for run in effective_e106_runs
                if int(run.get("job_id", -1)) != int(e110_runs[0].get("job_id", -2))
            ]
            if e110_runs
            else effective_e106_runs,
        ),
        ("e110_python", e110_runs),
    ):
        for run in runs:
            scale = str(run["scale"])
            domain = str(run["domain"])
            seed = int(run["seed"])
            prefix = f"{source}/{scale}/{domain}/s{seed}"
            target_steps = (
                int(e110.get("target_steps", 192))
                if source == "e110_python"
                else e104.SMOKE_TARGET_STEPS
            )
            run_dir = Path(str(run["run_dir"]))
            report, local = e104_audit.parse_run(
                run_dir,
                target_steps=target_steps,
            )
            violations.extend(f"{prefix}: {message}" for message in local)
            scheduler_state = states.get(int(run["job_id"]), "UNKNOWN")
            receipt_step = status.receipt_step(run_dir)
            runtime_complete = (
                report["last_step"] >= target_steps
                and receipt_step >= target_steps
            )
            if scheduler_state != "COMPLETED" and not runtime_complete:
                violations.append(f"{prefix}: scheduler state is {scheduler_state}")
            failures = e104_audit.log_failures(Path(str(run["stderr"])))
            if failures:
                violations.append(f"{prefix}: failure markers {failures}")
            semantic_rms = float(report["semantic_rms_max"])
            if semantic_rms > 0.0:
                nonzero_semantic_scales.add(scale)
            elif (
                runtime_complete
            ):
                completed_zero_semantic_cells.append(
                    {
                        "source": source,
                        "scale": scale,
                        "domain": domain,
                        "seed": seed,
                        "job_id": int(run["job_id"]),
                        "mechanism_context": {
                            key: report[key]
                            for key in (
                                "bank_size_after_max",
                                "online_eligible_fraction_max",
                                "semantic_eligible_fraction_max",
                                "support_at_least_two_prompt_fraction_max",
                                "mean_support_per_prompt_max",
                                "history_rows_added_max",
                                "replay_groups_max",
                                "replay_gradient_l2_max",
                            )
                        },
                    }
                )
            admission: dict[str, float] | None = None
            if source in {"e106_python", "e110_python"}:
                admission = python_admission_report(Path(str(run["run_dir"])))
                if admission["eligible_fraction_max"] <= 0.0:
                    violations.append(f"{prefix}: no verified Python row was admitted")
                if admission["bank_size_after_max"] <= 0.0:
                    violations.append(f"{prefix}: verified Python bank stayed empty")
                if admission["history_rows_added_max"] <= 0.0:
                    violations.append(f"{prefix}: semantic history received no Python row")
            reports.append(
                {
                    "source": source,
                    "scale": scale,
                    "domain": domain,
                    "seed": seed,
                    "job_id": int(run["job_id"]),
                    "scheduler_state": scheduler_state,
                    "completion_receipt_step": receipt_step,
                    "runtime_complete": runtime_complete,
                    "run_dir": str(run["run_dir"]),
                    "report": report,
                    "python_admission": admission,
                    "target_steps": target_steps,
                }
            )

    complete = (
        len(reports) == 15
        and all(item["runtime_complete"] for item in reports)
    )
    missing_semantic_scales = set(launch.SCALES) - nonzero_semantic_scales
    if complete and missing_semantic_scales:
        violations.append(
            "no live nonzero semantic update for scales: "
            + ", ".join(sorted(missing_semantic_scales))
        )
    payload = {
        "schema": "e106_python_lambda_normalization_combined_gate_v1",
        "complete": complete,
        "passed": complete and not violations,
        "mechanism_gate_used_outcome_metrics": False,
        "post_update_outcome_metrics_inspected": False,
        "pointmaze": "excluded",
        "surface_version": launch.SURFACE_VERSION,
        "nonzero_semantic_scales": sorted(nonzero_semantic_scales),
        "completed_zero_semantic_cells": completed_zero_semantic_cells,
        "snapshot_root": str(snapshot),
        "snapshot_sha256": launch.SNAPSHOT_SHA256,
        "ledger": str(ledger_path),
        "base_e104_ledger": str(base_path),
        "superseded_e104_falcon_python_cancellation": {
            "job_id": 30637792,
            "replacement_job_id": 30640330,
            "runtime": "00:00:00",
            "path": str(CANCELLATION_NOTE),
            "sha256": digest(CANCELLATION_NOTE) if CANCELLATION_NOTE.is_file() else None,
        },
        "qwen3_a6000_placement": {
            "path": str(QWEN3_PLACEMENT),
            "sha256": digest(QWEN3_PLACEMENT) if QWEN3_PLACEMENT.is_file() else None,
            "record": qwen3_placement,
        },
        "qwen3_clean_restart_all_partition_amendment": {
            "path": str(QWEN3_CLEAN_RESTART_AMENDMENT),
            "sha256": (
                digest(QWEN3_CLEAN_RESTART_AMENDMENT)
                if QWEN3_CLEAN_RESTART_AMENDMENT.is_file()
                else None
            ),
            "record": qwen3_clean_restart,
        },
        "e110_falcon_python_admission_horizon_replacement": {
            "path": str(E110_LEDGER),
            "sha256": digest(E110_LEDGER) if E110_LEDGER.is_file() else None,
            "record": e110,
        },
        "e110_two_hour_backfill_amendment": {
            "path": str(E110_TIME_LIMIT_AMENDMENT),
            "sha256": (
                digest(E110_TIME_LIMIT_AMENDMENT)
                if E110_TIME_LIMIT_AMENDMENT.is_file()
                else None
            ),
            "record": e110_time,
        },
        "superseded_failed_e106_falcon_python_cell": (
            failed_falcon[0] if len(failed_falcon) == 1 else None
        ),
        "unit_test_evidence": {
            "path": str(ROOT / launch.UNIT_EVIDENCE),
            "sha256": digest(ROOT / launch.UNIT_EVIDENCE) if (ROOT / launch.UNIT_EVIDENCE).is_file() else None,
            "passed": unit.get("passed", False),
        },
        "prior_surface_mode_evidence": {
            "path": str(PRIOR_SURFACE_EVIDENCE),
            "sha256": digest(PRIOR_SURFACE_EVIDENCE) if PRIOR_SURFACE_EVIDENCE.is_file() else None,
            "verified_rows": prior_surface.get("verified_rows"),
            "distinct_endpoint_keys": prior_surface.get("distinct_endpoint_keys"),
            "endpoint_multiplicities": prior_surface.get("endpoint_multiplicities"),
        },
        "parser_to_semantic_integration_evidence": {
            "path": str(INTEGRATION_EVIDENCE),
            "sha256": digest(INTEGRATION_EVIDENCE) if INTEGRATION_EVIDENCE.is_file() else None,
            "passed": integration.get("passed", False),
            "snapshot_sha256": integration.get("snapshot_sha256"),
        },
        "cross_scale_python_surface_evidence": {
            "path": str(CROSS_SCALE_SURFACE_EVIDENCE),
            "sha256": digest(CROSS_SCALE_SURFACE_EVIDENCE) if CROSS_SCALE_SURFACE_EVIDENCE.is_file() else None,
            "passed": cross_scale.get("passed", False),
            "scales": {
                scale: report.get("totals", {})
                for scale, report in cross_scale.get("scales", {}).items()
            },
        },
        "prior_falcon_python_bootstrap_evidence": {
            "path": str(FALCON_BOOTSTRAP_EVIDENCE),
            "sha256": digest(FALCON_BOOTSTRAP_EVIDENCE) if FALCON_BOOTSTRAP_EVIDENCE.is_file() else None,
            "passed": falcon_bootstrap.get("passed", False),
            "runs": falcon_bootstrap.get("runs", {}),
        },
        "extended_frozen_estimator_evidence": {
            "path": str(EXTENDED_ESTIMATOR_EVIDENCE),
            "sha256": digest(EXTENDED_ESTIMATOR_EVIDENCE) if EXTENDED_ESTIMATOR_EVIDENCE.is_file() else None,
            "passed": extended.get("passed", False),
            "snapshot_sha256": extended.get("snapshot_sha256"),
        },
        "frozen_policy_gradient_direction_evidence": {
            "path": str(POLICY_GRADIENT_EVIDENCE),
            "sha256": (
                digest(POLICY_GRADIENT_EVIDENCE)
                if POLICY_GRADIENT_EVIDENCE.is_file()
                else None
            ),
            "passed": policy_gradient.get("passed", False),
            "snapshot_sha256": policy_gradient.get("snapshot_sha256"),
            "assertions": policy_gradient.get("assertions", {}),
        },
        "semantic_replay_identity_contract_evidence": {
            "path": str(SEMANTIC_REPLAY_IDENTITY_EVIDENCE),
            "sha256": (
                digest(SEMANTIC_REPLAY_IDENTITY_EVIDENCE)
                if SEMANTIC_REPLAY_IDENTITY_EVIDENCE.is_file()
                else None
            ),
            "passed": semantic_replay_identity.get("passed", False),
            "snapshot_sha256": semantic_replay_identity.get("snapshot_sha256"),
            "domains": semantic_replay_identity.get("domains", []),
            "assertions": semantic_replay_identity.get("assertions", {}),
        },
        "python_smoke_time_limit_amendment": {
            "path": str(TIME_LIMIT_AMENDMENT),
            "sha256": digest(TIME_LIMIT_AMENDMENT) if TIME_LIMIT_AMENDMENT.is_file() else None,
            "record": time_limit,
        },
        "falcon_python_a6000_pool_amendment": {
            "path": str(FALCON_POOL_AMENDMENT),
            "sha256": digest(FALCON_POOL_AMENDMENT) if FALCON_POOL_AMENDMENT.is_file() else None,
            "record": falcon_pool,
        },
        "falcon_one_hour_backfill_amendment": {
            "path": str(FALCON_ONE_HOUR_AMENDMENT),
            "sha256": (
                digest(FALCON_ONE_HOUR_AMENDMENT)
                if FALCON_ONE_HOUR_AMENDMENT.is_file()
                else None
            ),
            "record": falcon_one_hour,
        },
        "falcon_same_gpu_pool_widening_amendment": {
            "path": str(FALCON_POOL_WIDENING_AMENDMENT),
            "sha256": (
                digest(FALCON_POOL_WIDENING_AMENDMENT)
                if FALCON_POOL_WIDENING_AMENDMENT.is_file()
                else None
            ),
            "record": falcon_pool_widening,
        },
        "falcon_a6000_all_partition_pool_amendment": {
            "path": str(FALCON_ALL_POOL_AMENDMENT),
            "sha256": (
                digest(FALCON_ALL_POOL_AMENDMENT)
                if FALCON_ALL_POOL_AMENDMENT.is_file()
                else None
            ),
            "record": falcon_all_pool,
        },
        "runs": reports,
        "violations": violations,
    }
    shared.atomic_json(OUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
