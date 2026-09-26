#!/usr/bin/env python3
"""Submit E105's 75 repaired semantic-on-replay cells after the E104 gate."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e79_falcon1b_aligned_verified_replay as e79  # noqa: E402
import launch_e80r1_qwen3b_aligned_verified_replay as e80  # noqa: E402
import launch_e81_semantic_maxent_verified_replay_05b as e81  # noqa: E402
import launch_e104_group_centered_semantic_repair_three_scale as e104  # noqa: E402
import launch_e106_python_lambda_normalization_three_scale as e106  # noqa: E402
import audit_e106_python_lambda_normalization as e106_gate  # noqa: E402


DOMAINS = e104.DOMAINS
SCALE_SEEDS = {
    "qwen05b": (43, 44, 45, 46, 47),
    "falcon1b": (55, 56, 57, 58, 59),
    "qwen3b": (70, 71, 72, 73, 74),
}
MODEL_TAGS = e104.MODEL_TAGS
VARIANT = e104.VARIANT
ARM = e104.ARM
TRAIN_ROWS = 384
PASSES = 8
TARGET_STEPS = TRAIN_ROWS * PASSES
CHECKPOINT_INTERVAL = 192
LEDGER = "var/artifacts/e105_group_centered_semantic_repair_full_three_scale_jobs.json"
PROTOCOL = (
    "paper/preregistration/"
    "e105_group_centered_semantic_repair_full_three_scale_20260817.md"
)
REPAIR_AMENDMENT = (
    "paper/preregistration/"
    "e105_python_lambda_normalization_amendment_20260817.md"
)
PYTHON_COMPARATOR_AMENDMENT = (
    "paper/preregistration/"
    "e109_repaired_python_replay_comparators_20260817.md"
)
ANALYSIS_PROTOCOL = (
    "paper/preregistration/e105_paired_analysis_specification_20260817.md"
)
ANALYSIS_BUILDER = (
    "ops/exp_scaling/build_e105_group_centered_semantic_repair_results.py"
)
ANALYSIS_PLOTTER = (
    "ops/exp_scaling/plot_e105_group_centered_semantic_endpoint_effects.py"
)
POST_CAMPAIGN_ANALYSIS_PROTOCOL = (
    "paper/preregistration/e105_post_campaign_analysis_20260818.md"
)
POST_CAMPAIGN_ANALYSIS_SCRIPT = (
    "ops/slurm/e105_post_campaign_analysis.slurm"
)
E104_LEDGER = e104.LEDGER
E104_AUDIT = e104.AUDIT
E106_LEDGER = e106.LEDGER
E106_AUDIT = e106.AUDIT
E106_UNIT_EVIDENCE = e106.UNIT_EVIDENCE
E104_DEVIATION = (
    "paper/preregistration/e104_step0_exposure_deviation_20260817.md"
)
E104_THEORY_CLARIFICATION = (
    "paper/preregistration/e104_theory_clarification_20260817.md"
)
E104_PLACEMENT_AMENDMENT = (
    "paper/preregistration/e104_static_smoke_placement_amendment_20260817.md"
)
E104_UNIT_EVIDENCE = (
    "var/artifacts/e104_group_centered_semantic_repair_unit_tests.json"
)
SPARSE_THEORY_EVIDENCE = (
    "var/artifacts/e105_sparse_success_theory_tests.json"
)
QWEN3_PLACEMENT_PROTOCOL = (
    "paper/preregistration/"
    "e105_qwen3_paired_a6000_placement_amendment_20260817.md"
)
QWEN3_PLACEMENT_SCRIPT = (
    "ops/exp_scaling/apply_e105_qwen3_paired_a6000_placement_amendment.py"
)
QWEN3_PLACEMENT_ARTIFACT = (
    "var/artifacts/e105_qwen3_paired_a6000_placement_amendment.json"
)
QWEN3_A6000_NODE_LIST = "node[103-104,205-208,805]"
COMPARATOR_LEDGERS = {
    "qwen05b": "var/artifacts/e78_verified_replay_only_05b_jobs.json",
    "falcon1b": "var/artifacts/e79_falcon1b_aligned_verified_replay_jobs.json",
    "qwen3b": "var/artifacts/e80r1_qwen3b_aligned_verified_replay_jobs.json",
}
REPAIRED_PYTHON_COMPARATOR_LEDGER = (
    "var/artifacts/e109_repaired_python_replay_comparators_jobs.json"
)
E106_SUPPLEMENTAL_LEDGER_KEYS = (
    "qwen3_a6000_placement",
    "superseded_e104_falcon_python_cancellation",
    "prior_surface_mode_evidence",
    "parser_to_semantic_integration_evidence",
    "cross_scale_python_surface_evidence",
    "prior_falcon_python_bootstrap_evidence",
    "extended_frozen_estimator_evidence",
    "frozen_policy_gradient_direction_evidence",
    "semantic_replay_identity_contract_evidence",
    "python_smoke_time_limit_amendment",
    "falcon_python_a6000_pool_amendment",
    "falcon_one_hour_backfill_amendment",
    "falcon_same_gpu_pool_widening_amendment",
    "falcon_a6000_all_partition_pool_amendment",
    "qwen3_clean_restart_all_partition_amendment",
    "e110_falcon_python_admission_horizon_replacement",
    "e110_two_hour_backfill_amendment",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _require_current_audit_artifact(
    audit: dict[str, Any], key: str, artifact: Path
) -> dict[str, Any]:
    record = audit.get(key)
    if not isinstance(record, dict):
        raise SystemExit(f"E106 combined audit lacks {key}")
    recorded_path = Path(str(record.get("path", ""))).resolve()
    if recorded_path != artifact.resolve():
        raise SystemExit(f"E106 {key} path mismatch")
    if not artifact.is_file() or record.get("sha256") != e104.digest(artifact):
        raise SystemExit(f"E106 {key} digest mismatch")
    return record


def require_e106_supplemental_evidence(
    root: Path, audit: dict[str, Any], snapshot: Path
) -> None:
    """Revalidate current E106 evidence instead of trusting a stale pass bit."""

    if root.resolve() != e106_gate.ROOT.resolve():
        raise SystemExit("E105 and E106 validators resolve different repository roots")
    cancellation = _require_current_audit_artifact(
        audit,
        "superseded_e104_falcon_python_cancellation",
        e106_gate.CANCELLATION_NOTE,
    )
    if cancellation.get("job_id") != 30637792 or cancellation.get(
        "replacement_job_id"
    ) != 30640330:
        raise SystemExit("E106 cancellation record names the wrong jobs")
    if cancellation.get("runtime") != "00:00:00":
        raise SystemExit("E106 superseded Falcon Python job accrued runtime")
    validators = (
        (
            "prior_surface_mode_evidence",
            e106_gate.PRIOR_SURFACE_EVIDENCE,
            e106_gate.validate_prior_surface_evidence,
            (),
        ),
        (
            "parser_to_semantic_integration_evidence",
            e106_gate.INTEGRATION_EVIDENCE,
            e106_gate.validate_integration_evidence,
            (snapshot,),
        ),
        (
            "cross_scale_python_surface_evidence",
            e106_gate.CROSS_SCALE_SURFACE_EVIDENCE,
            e106_gate.validate_cross_scale_surface_evidence,
            (snapshot,),
        ),
        (
            "prior_falcon_python_bootstrap_evidence",
            e106_gate.FALCON_BOOTSTRAP_EVIDENCE,
            e106_gate.validate_falcon_bootstrap_evidence,
            (),
        ),
        (
            "extended_frozen_estimator_evidence",
            e106_gate.EXTENDED_ESTIMATOR_EVIDENCE,
            e106_gate.validate_extended_estimator_evidence,
            (snapshot,),
        ),
        (
            "frozen_policy_gradient_direction_evidence",
            e106_gate.POLICY_GRADIENT_EVIDENCE,
            e106_gate.validate_policy_gradient_evidence,
            (snapshot,),
        ),
        (
            "semantic_replay_identity_contract_evidence",
            e106_gate.SEMANTIC_REPLAY_IDENTITY_EVIDENCE,
            e106_gate.validate_semantic_replay_identity_evidence,
            (snapshot,),
        ),
        (
            "python_smoke_time_limit_amendment",
            e106_gate.TIME_LIMIT_AMENDMENT,
            e106_gate.validate_time_limit_amendment,
            (),
        ),
        (
            "falcon_python_a6000_pool_amendment",
            e106_gate.FALCON_POOL_AMENDMENT,
            e106_gate.validate_falcon_pool_amendment,
            (),
        ),
        (
            "falcon_one_hour_backfill_amendment",
            e106_gate.FALCON_ONE_HOUR_AMENDMENT,
            e106_gate.validate_falcon_one_hour_amendment,
            (),
        ),
        (
            "falcon_same_gpu_pool_widening_amendment",
            e106_gate.FALCON_POOL_WIDENING_AMENDMENT,
            e106_gate.validate_falcon_pool_widening_amendment,
            (),
        ),
        (
            "falcon_a6000_all_partition_pool_amendment",
            e106_gate.FALCON_ALL_POOL_AMENDMENT,
            e106_gate.validate_falcon_all_pool_amendment,
            (),
        ),
        (
            "qwen3_clean_restart_all_partition_amendment",
            e106_gate.QWEN3_CLEAN_RESTART_AMENDMENT,
            e106_gate.validate_qwen3_clean_restart_amendment,
            (),
        ),
        (
            "e110_falcon_python_admission_horizon_replacement",
            e106_gate.E110_LEDGER,
            e106_gate.validate_e110_replacement_ledger,
            (),
        ),
        (
            "e110_two_hour_backfill_amendment",
            e106_gate.E110_TIME_LIMIT_AMENDMENT,
            e106_gate.validate_e110_time_limit_amendment,
            (),
        ),
    )
    for key, artifact, validator, args in validators:
        record = _require_current_audit_artifact(audit, key, artifact)
        payload, violations = validator(*args)
        if violations:
            raise SystemExit(f"E106 {key} failed fresh validation: {violations}")
        if "passed" in record and record.get("passed") is not True:
            raise SystemExit(f"E106 combined audit did not pass {key}")
        if "record" in record and record.get("record") != payload:
            raise SystemExit(f"E106 combined audit has stale {key} content")

    placement = _require_current_audit_artifact(
        audit, "qwen3_a6000_placement", e106_gate.QWEN3_PLACEMENT
    )
    placement_payload = json.loads(
        e106_gate.QWEN3_PLACEMENT.read_text(encoding="utf-8")
    )
    if placement.get("record") != placement_payload:
        raise SystemExit("E106 combined audit has stale Qwen-3B placement content")
    if placement_payload.get("schema") != "e106_qwen3_a6000_placement_v1":
        raise SystemExit("E106 Qwen-3B placement schema mismatch")
    if placement_payload.get("job_id") != 30640331:
        raise SystemExit("E106 Qwen-3B placement names another job")
    if placement_payload.get("environment_changed") is not False:
        raise SystemExit("E106 Qwen-3B placement changed the environment")
    if placement_payload.get("post_update_outcomes_inspected") is not False:
        raise SystemExit("E106 Qwen-3B placement inspected outcomes")
    for path_key, digest_key in (
        ("amendment", "amendment_sha256"),
        ("capacity_preflight_ledger", "capacity_preflight_ledger_sha256"),
        ("capacity_preflight_audit", "capacity_preflight_audit_sha256"),
    ):
        evidence = root / str(placement_payload.get(path_key, ""))
        if not evidence.is_file() or placement_payload.get(
            digest_key
        ) != e104.digest(evidence):
            raise SystemExit(f"E106 Qwen-3B placement digest mismatch: {path_key}")


def require_sparse_success_theory_evidence(root: Path, snapshot: Path) -> dict[str, Any]:
    sparse_path = root / SPARSE_THEORY_EVIDENCE
    if not sparse_path.is_file():
        raise SystemExit("E105 sparse-success theory evidence is absent")
    sparse = json.loads(sparse_path.read_text(encoding="utf-8"))
    if sparse.get("schema") != "e105_sparse_success_theory_tests_v1":
        raise SystemExit("E105 sparse-success theory evidence schema mismatch")
    if sparse.get("passed") is not True or sparse.get("returncode") != 0:
        raise SystemExit("E105 sparse-success theory test did not pass")
    if sparse.get("snapshot_root") != str(snapshot):
        raise SystemExit("E105 sparse-success theory test used another snapshot")
    if sparse.get("snapshot_sha256") != e106.SNAPSHOT_SHA256:
        raise SystemExit("E105 sparse-success theory snapshot hash mismatch")
    if sparse.get("post_e104_or_e106_update_outcomes_inspected") is not False:
        raise SystemExit("E105 sparse-success theory evidence violated blinding")
    if sparse.get("pointmaze") != "excluded":
        raise SystemExit("E105 sparse-success theory evidence includes PointMaze")
    sparse_test = root / str(sparse.get("test", ""))
    if not sparse_test.is_file() or sparse.get("test_sha256") != e104.digest(
        sparse_test
    ):
        raise SystemExit("E105 sparse-success theory test digest mismatch")
    if sparse.get("implementation_sha256") != {
        "src/oat_drgrpo/semantic_shannon.py": e104.digest(
            snapshot / "src/oat_drgrpo/semantic_shannon.py"
        )
    }:
        raise SystemExit("E105 sparse-success implementation digest mismatch")
    if sparse.get("group_size") != 16 or "2 passed" not in str(
        sparse.get("stdout", "")
    ):
        raise SystemExit("E105 sparse-success theory evidence is incomplete")
    return sparse


def check_gate(root: Path) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    e104_ledger_path = root / E104_LEDGER
    e104_audit_path = root / E104_AUDIT
    e106_ledger_path = root / E106_LEDGER
    e106_audit_path = root / E106_AUDIT
    for required in (
        e104_ledger_path,
        e104_audit_path,
        e106_ledger_path,
        e106_audit_path,
    ):
        if not required.is_file():
            raise SystemExit(f"E105 gate input is absent: {required}")
    e104_ledger = json.loads(e104_ledger_path.read_text(encoding="utf-8"))
    e104_audit = json.loads(e104_audit_path.read_text(encoding="utf-8"))
    e106_ledger = json.loads(e106_ledger_path.read_text(encoding="utf-8"))
    audit = json.loads(e106_audit_path.read_text(encoding="utf-8"))
    if e104_ledger.get("released") is not True:
        raise SystemExit("E104 was not released")
    if e106_ledger.get("released") is not True:
        raise SystemExit("E106 was not released")
    if e106_ledger.get("base_e104_ledger_sha256") != e104.digest(e104_ledger_path):
        raise SystemExit("E106 does not bind the current immutable E104 ledger")
    for key, relative in (
        ("protocol_sha256", e106.PROTOCOL),
        ("diagnosis_sha256", e106.DIAGNOSIS),
        (
            "launcher_sha256",
            "ops/exp_scaling/launch_e106_python_lambda_normalization_three_scale.py",
        ),
    ):
        evidence = root / relative
        if not evidence.is_file() or e106_ledger.get(key) != e104.digest(evidence):
            raise SystemExit(f"E106 ledger provenance digest mismatch: {key}")
    diagnosis = json.loads((root / e106.DIAGNOSIS).read_text(encoding="utf-8"))
    if diagnosis.get("post_e104_update_outcomes_inspected") is not False:
        raise SystemExit("E106 diagnosis violated outcome blinding")
    if e104_audit.get("schema") != "e104_group_centered_semantic_repair_gate_v1":
        raise SystemExit("E104 audit schema mismatch")
    if audit.get("schema") != "e106_python_lambda_normalization_combined_gate_v1":
        raise SystemExit("E106 combined audit schema mismatch")
    if audit.get("passed") is not True or audit.get("complete") is not True:
        raise SystemExit("E104+E106 combined mechanism gate did not pass")
    if audit.get("pointmaze") != "excluded":
        raise SystemExit("E106 combined gate did not exclude PointMaze")
    if audit.get("surface_version") != e106.SURFACE_VERSION:
        raise SystemExit("E106 combined gate names another parser surface")
    if e104_audit.get("outcome_metrics_inspected") is not True:
        raise SystemExit("E104 audit did not record the known step-0 exposure")
    if e104_audit.get("post_update_outcome_metrics_inspected") is not False:
        raise SystemExit("E104 audit inspected a forbidden post-update outcome")
    if audit.get("post_update_outcome_metrics_inspected") is not False:
        raise SystemExit("combined gate inspected a forbidden post-update outcome")
    if audit.get("mechanism_gate_used_outcome_metrics") is not False:
        raise SystemExit("combined mechanism decision used outcome metrics")
    if set(audit.get("nonzero_semantic_scales", [])) != set(SCALE_SEEDS):
        raise SystemExit(
            "combined gate lacks a live nonzero semantic update at every scale"
        )
    exposure = e104_audit.get("manual_pre_treatment_outcome_exposure", {})
    deviation = root / E104_DEVIATION
    if exposure.get("scope") != "one aggregate correctness field at step 0":
        raise SystemExit("E104 exposure scope is not the registered exception")
    if exposure.get("deviation") != str(deviation):
        raise SystemExit("E104 exposure note path mismatch")
    if not deviation.is_file() or exposure.get("deviation_sha256") != e104.digest(
        deviation
    ):
        raise SystemExit("E104 exposure note digest mismatch")
    clarification = root / E104_THEORY_CLARIFICATION
    if e104_audit.get("theory_clarification") != str(clarification):
        raise SystemExit("E104 theory clarification path mismatch")
    if not clarification.is_file() or e104_audit.get(
        "theory_clarification_sha256"
    ) != e104.digest(clarification):
        raise SystemExit("E104 theory clarification digest mismatch")
    placement_amendment = root / E104_PLACEMENT_AMENDMENT
    if e104_audit.get("placement_amendment") != str(placement_amendment):
        raise SystemExit("E104 placement amendment path mismatch")
    if not placement_amendment.is_file() or e104_audit.get(
        "placement_amendment_sha256"
    ) != e104.digest(placement_amendment):
        raise SystemExit("E104 placement amendment digest mismatch")
    qwen3_amendment = e104_audit.get("qwen3_a6000_placement_amendment")
    if qwen3_amendment is not None:
        qwen3_path = Path(str(qwen3_amendment.get("path", "")))
        if qwen3_amendment.get("validated") is not True:
            raise SystemExit("E104 Qwen-3B placement amendment was not validated")
        if not qwen3_path.is_file() or qwen3_amendment.get(
            "sha256"
        ) != e104.digest(qwen3_path):
            raise SystemExit("E104 Qwen-3B placement amendment digest mismatch")
    unit_record = e104_audit.get("unit_test_evidence", {})
    unit_evidence = root / E104_UNIT_EVIDENCE
    if unit_record.get("path") != str(unit_evidence):
        raise SystemExit("E104 unit-test evidence path mismatch")
    if unit_record.get("passed") is not True:
        raise SystemExit("E104 snapshot unit tests did not pass")
    if not unit_evidence.is_file() or unit_record.get("sha256") != e104.digest(
        unit_evidence
    ):
        raise SystemExit("E104 unit-test evidence digest mismatch")
    e106_unit = root / E106_UNIT_EVIDENCE
    combined_unit = audit.get("unit_test_evidence", {})
    if combined_unit.get("path") != str(e106_unit):
        raise SystemExit("E106 unit-test evidence path mismatch")
    if combined_unit.get("passed") is not True:
        raise SystemExit("E106 snapshot unit tests did not pass")
    if not e106_unit.is_file() or combined_unit.get("sha256") != e104.digest(e106_unit):
        raise SystemExit("E106 unit-test evidence digest mismatch")
    snapshot = Path(str(e106_ledger["snapshot_root"])).resolve()
    if Path(str(audit.get("snapshot_root"))).resolve() != snapshot:
        raise SystemExit("E106 ledger and combined audit name different snapshots")
    if audit.get("snapshot_sha256") != e106.SNAPSHOT_SHA256:
        raise SystemExit("E106 combined audit snapshot hash mismatch")
    e106.verify_snapshot(root, snapshot)
    require_e106_supplemental_evidence(root, audit, snapshot)
    require_sparse_success_theory_evidence(root, snapshot)
    return snapshot, e106_audit_path, audit, e104_audit


def comparator_index(
    root: Path,
    scale: str,
    *,
    allow_legacy_python: bool = False,
) -> dict[tuple[str, int], dict[str, Any]]:
    path = root / COMPARATOR_LEDGERS[scale]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("released") is not True:
        raise SystemExit(f"{scale} Re:Dr comparator is not released")
    replay = [run for run in payload["runs"] if run.get("arm") == "replay"]
    expected = {
        (domain, seed) for domain in DOMAINS for seed in SCALE_SEEDS[scale]
    }
    index = {
        (str(run["domain"]), int(run["seed"])): run for run in replay
    }
    if set(index) != expected or len(replay) != len(expected):
        raise SystemExit(f"{scale} comparator does not cover the E105 grid")
    repaired_path = root / REPAIRED_PYTHON_COMPARATOR_LEDGER
    if not repaired_path.is_file():
        if allow_legacy_python:
            return index
        raise SystemExit(
            "E105 repaired Python Re:Dr comparator ledger is absent"
        )
    repaired = json.loads(repaired_path.read_text(encoding="utf-8"))
    if (
        repaired.get("schema")
        != "e109_repaired_python_replay_comparators_jobs_v1"
        or repaired.get("released") is not True
    ):
        raise SystemExit("E105 repaired Python comparator is not released")
    if repaired.get("snapshot_sha256") != e106.SNAPSHOT_SHA256:
        raise SystemExit("E105 repaired Python comparator snapshot hash mismatch")
    if repaired.get("parser_surface_version") != e106.SURFACE_VERSION:
        raise SystemExit("E105 repaired Python comparator parser surface mismatch")
    if repaired.get("semantic_coefficient") != 0.0:
        raise SystemExit("E105 repaired Python comparator activates semantics")
    if repaired.get("pointmaze") != "excluded":
        raise SystemExit("E105 repaired Python comparator includes PointMaze")
    repaired_runs = list(repaired.get("runs", []))
    expected_repaired_cells = {
        (candidate_scale, "python_factors", seed)
        for candidate_scale, seeds in SCALE_SEEDS.items()
        for seed in seeds
    }
    observed_repaired_cells = {
        (
            str(run.get("scale")),
            str(run.get("domain")),
            int(run.get("seed")),
        )
        for run in repaired_runs
        if run.get("arm") == "replay"
    }
    if (
        len(repaired_runs) != 15
        or observed_repaired_cells != expected_repaired_cells
        or any(run.get("arm") != "replay" for run in repaired_runs)
    ):
        raise SystemExit("E105 repaired Python comparator cohort drifted")
    repaired_scale = [
        run
        for run in repaired_runs
        if str(run.get("scale")) == scale
    ]
    expected_python = {
        ("python_factors", seed) for seed in SCALE_SEEDS[scale]
    }
    repaired_index = {
        (str(run.get("domain")), int(run.get("seed"))): run
        for run in repaired_scale
        if run.get("arm") == "replay"
    }
    if set(repaired_index) != expected_python or len(repaired_scale) != 5:
        raise SystemExit(f"E105 repaired Python comparator drifted for {scale}")
    index.update(repaired_index)
    return index


def require_qwen3_paired_placement(
    root: Path,
) -> tuple[dict[str, Any], set[tuple[str, int]]]:
    """Validate the post-gate scheduler amendment that preserves pair hardware."""

    artifact_path = root / QWEN3_PLACEMENT_ARTIFACT
    protocol_path = root / QWEN3_PLACEMENT_PROTOCOL
    script_path = root / QWEN3_PLACEMENT_SCRIPT
    for required in (artifact_path, protocol_path, script_path):
        if not required.is_file():
            raise SystemExit(f"E105 Qwen-3B placement input is absent: {required}")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    checks = {
        "schema": payload.get("schema")
        == "e105_qwen3_paired_a6000_placement_amendment_v1",
        "protocol": Path(str(payload.get("protocol", ""))).resolve()
        == protocol_path.resolve(),
        "protocol_sha256": payload.get("protocol_sha256") == e104.digest(protocol_path),
        "script": Path(str(payload.get("script", ""))).resolve()
        == script_path.resolve(),
        "script_sha256": payload.get("script_sha256") == e104.digest(script_path),
        "e80r1_ledger": Path(str(payload.get("e80r1_ledger", ""))).resolve()
        == (root / COMPARATOR_LEDGERS["qwen3b"]).resolve(),
        "e80r1_ledger_sha256": payload.get("e80r1_ledger_sha256")
        == e104.digest(root / COMPARATOR_LEDGERS["qwen3b"]),
        "combined_gate": Path(str(payload.get("combined_gate", ""))).resolve()
        == (root / E106_AUDIT).resolve(),
        "combined_gate_sha256": payload.get("combined_gate_sha256")
        == e104.digest(root / E106_AUDIT),
        "combined_gate_complete": payload.get("combined_gate_complete") is True,
        "combined_gate_passed": payload.get("combined_gate_passed") is True,
        "outcome_blind": payload.get("outcome_metrics_inspected") is False,
        "scientific_configuration": payload.get("scientific_configuration_changed")
        is False,
        "pointmaze": payload.get("pointmaze") == "excluded",
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"E105 Qwen-3B placement artifact failed: {failed}")

    expected_candidates = {
        ("python_factors", 73),
        ("python_factors", 74),
        *(('mathir', seed) for seed in (71, 72, 73, 74)),
        *(('pantry_plan', seed) for seed in (71, 72, 73, 74)),
    }
    candidates = {
        (str(row["domain"]), int(row["seed"]))
        for row in payload.get("candidate_cells", [])
    }
    historical_candidates = {
        (str(row["domain"]), int(row["seed"]))
        for row in payload.get("historical_candidate_cells", [])
    }
    prospective_python = {
        (str(row["domain"]), int(row["seed"]))
        for row in payload.get("prospective_python_cells", [])
    }
    paired_rows = payload.get("paired_a6000_cells", [])
    moved_rows = payload.get("moved_pairs", [])
    skipped_rows = payload.get("skipped_pairs", [])
    paired = {(str(row["domain"]), int(row["seed"])) for row in paired_rows}
    moved = {(str(row["domain"]), int(row["seed"])) for row in moved_rows}
    skipped = {(str(row["domain"]), int(row["seed"])) for row in skipped_rows}
    if (
        candidates != expected_candidates
        or prospective_python
        != {("python_factors", 73), ("python_factors", 74)}
        or historical_candidates != expected_candidates - prospective_python
        or moved & skipped
        or moved | skipped != historical_candidates
        or paired != prospective_python | moved
        or len(paired_rows) != len(paired)
        or len(moved_rows) != len(moved)
        or len(skipped_rows) != len(skipped)
    ):
        raise SystemExit("E105 Qwen-3B placement pair partition drifted")

    control_payload = json.loads(
        (root / COMPARATOR_LEDGERS["qwen3b"]).read_text(encoding="utf-8")
    )
    controls = {
        (str(run["domain"]), int(run["seed"])): run
        for run in control_payload["runs"]
        if run.get("arm") == "control"
    }
    legacy_replay = {
        (str(run["domain"]), int(run["seed"])): run
        for run in control_payload["runs"]
        if run.get("arm") == "replay"
    }
    for row in moved_rows:
        cell = (str(row["domain"]), int(row["seed"]))
        if int(row["control_job_id"]) != int(controls[cell]["job_id"]):
            raise SystemExit(f"E105 placement control binding drifted: {cell}")
        if int(row["replay_job_id"]) != int(legacy_replay[cell]["job_id"]):
            raise SystemExit(f"E105 placement replay binding drifted: {cell}")
    placement = payload.get("placement", {})
    if placement != {
        "partition": "lowprio",
        "account": "mltheory",
        "node_list": QWEN3_A6000_NODE_LIST,
        "gres": "gpu:a6000:1",
        "cpus": 16,
        "memory": "128G",
        "time_limit": "3-00:00:00",
    }:
        raise SystemExit("E105 Qwen-3B placement resources drifted")
    return payload, paired


def references(root: Path, scale: str) -> list[dict[str, Any]]:
    if scale == "qwen05b":
        runs = e81.references(root)
    elif scale == "falcon1b":
        runs = e79.references(root)
    else:
        runs = e80.references(root)
    selected = [
        run
        for run in runs
        if str(run["domain"]) in DOMAINS
        and int(run["seed"]) in SCALE_SEEDS[scale]
    ]
    expected = {
        (domain, seed) for domain in DOMAINS for seed in SCALE_SEEDS[scale]
    }
    if {(str(run["domain"]), int(run["seed"])) for run in selected} != expected:
        raise SystemExit(f"{scale} source templates do not cover E105")
    return selected


def run_stamp(scale: str, domain: str, seed: int) -> str:
    return f"e105_{scale}_{e81.DOMAIN_TAGS[domain]}_{ARM}_s{seed}"


def save_path(root: Path, scale: str, domain: str, seed: int) -> Path:
    return root / "var/data" / (
        f"xdr_{MODEL_TAGS[scale]}_{VARIANT}_{run_stamp(scale, domain, seed)}"
    )


def build_env(
    root: Path,
    scale: str,
    run: dict[str, Any],
    snapshot: Path,
) -> tuple[dict[str, str], Path]:
    domain = str(run["domain"])
    seed = int(run["seed"])
    if scale == "qwen05b":
        env, _ = e81.build_env(root, run, snapshot)
    elif scale == "falcon1b":
        env, _ = e79.build_env(
            root, run, "replay", snapshot, e79.model_root(root)
        )
    else:
        env, _ = e80.build_env(
            root, run, "replay", snapshot, e80.model_root(root)
        )
    target = save_path(root, scale, domain, seed)
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": run_stamp(scale, domain, seed),
            "OAT_ZERO_SOURCE_ROOT": str(snapshot / "src"),
            "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot / "ops"),
            "OAT_ZERO_MAX_TRAIN": str(TRAIN_ROWS),
            "OAT_ZERO_NUM_PROMPT_EPOCH": str(PASSES),
            "OAT_ZERO_MAX_PROMPT_EPOCHS": str(PASSES),
            "OAT_ZERO_EVAL_PROMPT_INTERVAL": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_SAVE_STEPS": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_SAVE_FROM": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_RESUME_STEPS": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_MAX_SAVE_NUM": "1",
            "OAT_ZERO_MAX_RESUME_NUM": "1",
            "OAT_ZERO_AUTO_RESUME": "1",
            "OAT_ZERO_WATCHDOG_REQUEUE": "1",
            "OAT_ZERO_USE_WB": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    env.update(e104.fixed_objective())
    return env, target


def base_command(
    root: Path, scale: str, run: dict[str, Any], env: dict[str, str]
) -> list[str]:
    if scale == "qwen05b":
        return e81.sbatch_command(root, run, env)
    if scale == "falcon1b":
        return e79.sbatch_command(root, run, "replay", env)
    return e80.sbatch_command(root, run, "replay", env)


def job_name(scale: str, domain: str, seed: int) -> str:
    scale_tag = {"qwen05b": "q05", "falcon1b": "f1", "qwen3b": "q3"}[scale]
    return f"e105-{scale_tag}-{e81.DOMAIN_TAGS[domain][:5]}-s{seed}"


def sbatch_command(
    root: Path,
    scale: str,
    run: dict[str, Any],
    env: dict[str, str],
    qwen3_a6000_cells: set[tuple[str, int]] | frozenset[tuple[str, int]] = frozenset(),
) -> list[str]:
    domain = str(run["domain"])
    seed = int(run["seed"])
    command = base_command(root, scale, run, env)
    use_qwen3_a6000 = scale == "qwen3b" and (domain, seed) in qwen3_a6000_cells
    output: list[str] = []
    for token in command:
        if token.startswith("--job-name="):
            output.append(f"--job-name={job_name(scale, domain, seed)}")
        elif token.startswith("--nice="):
            output.append("--nice=100")
        elif use_qwen3_a6000 and token.startswith("--partition="):
            output.append("--partition=lowprio")
        elif use_qwen3_a6000 and token.startswith("--nodelist="):
            output.append(f"--nodelist={QWEN3_A6000_NODE_LIST}")
        elif use_qwen3_a6000 and token.startswith("--gres="):
            output.append("--gres=gpu:a6000:1")
        else:
            output.append(token)
    if not any(token.startswith("--nice=") for token in output):
        output.insert(-1, "--nice=100")
    return output


def held_job_audit(
    job_id: str,
    scale: str,
    run: dict[str, Any],
    snapshot: Path,
    qwen3_a6000_cells: set[tuple[str, int]] | frozenset[tuple[str, int]] = frozenset(),
) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot inspect held E105 job {job_id}")
    record = result.stdout
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"OAT_ZERO_VARIANT={VARIANT}",
        f"OAT_ZERO_SEED={run['seed']}",
        f"OAT_ZERO_MAX_TRAIN={TRAIN_ROWS}",
        f"OAT_ZERO_NUM_PROMPT_EPOCH={PASSES}",
        f"OAT_ZERO_EVAL_PROMPT_INTERVAL={CHECKPOINT_INTERVAL}",
        f"OAT_ZERO_SOURCE_ROOT={snapshot / 'src'}",
        f"OAT_ZERO_OPS_SNAPSHOT_ROOT={snapshot / 'ops'}",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.1",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE=0",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE=1",
        "OAT_ZERO_SEMANTIC_RMS_CONTROL=0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=verified_likelihood_per_rollout",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=0",
    )
    missing = [needle for needle in required if needle not in record]
    if scale == "qwen3b":
        cell = (str(run["domain"]), int(run["seed"]))
        if cell in qwen3_a6000_cells:
            placement_required = (
                "Partition=lowprio",
                f"ReqNodeList={QWEN3_A6000_NODE_LIST}",
                "TresPerNode=gres/gpu:a6000:1",
            )
        else:
            placement_required = (
                "Partition=mltheory",
                "ReqNodeList=node302",
                "TresPerNode=gres/gpu:a100:1",
            )
        missing.extend(
            needle for needle in placement_required if needle not in record
        )
    if missing:
        raise RuntimeError(f"held E105 job {job_id} lacks {missing}")
    return record


def submit_post_campaign_analysis(
    root: Path,
    *,
    e105_job_ids: list[str],
) -> dict[str, Any]:
    repaired = json.loads(
        (root / REPAIRED_PYTHON_COMPARATOR_LEDGER).read_text(encoding="utf-8")
    )
    e109_job_ids = [str(int(run["job_id"])) for run in repaired.get("runs", [])]
    if len(e105_job_ids) != 75 or len(e109_job_ids) != 15:
        raise RuntimeError("post-campaign analysis requires exactly 75+15 jobs")
    dependency_ids = [*e109_job_ids, *e105_job_ids]
    dependency = "afterany:" + ":".join(dependency_ids)
    script = root / POST_CAMPAIGN_ANALYSIS_SCRIPT
    export = ",".join(
        (
            "ALL",
            f"OAT_ZERO_REPO_ROOT={root}",
            f"OAT_ZERO_ANALYSIS_SCRIPT_SHA256={e104.digest(script)}",
            f"OAT_ZERO_ANALYSIS_PYTHON={Path(sys.executable).resolve()}",
        )
    )
    command = [
        "sbatch",
        "--parsable",
        f"--dependency={dependency}",
        "--job-name=e105-analysis",
        f"--export={export}",
        "--partition=all",
        "--account=allcs",
        "--cpus-per-task=2",
        "--mem=16G",
        "--time=02:00:00",
        "--nice=0",
        "--no-requeue",
        str(script),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"post-campaign analysis submission failed: {result.stderr.strip()}"
        )
    raw = result.stdout.strip().split(";", 1)[0]
    if not raw.isdigit():
        raise RuntimeError(f"invalid post-campaign analysis job id: {result.stdout!r}")
    job_id = int(raw)
    inspected = subprocess.run(
        ["scontrol", "show", "job", "-o", str(job_id)],
        capture_output=True,
        text=True,
        check=False,
    )
    record = inspected.stdout.strip()
    required = (
        "JobState=PENDING",
        "Reason=Dependency",
        "RunTime=00:00:00",
        "Account=allcs",
        "ReqTRES=cpu=2,mem=16G,node=1",
        "TimeLimit=02:00:00",
        f"Command={script}",
    )
    missing = [needle for needle in required if needle not in record]
    if "--partition=all" not in record:
        missing.append("SubmitLine=... --partition=all")
    if not any(value in record for value in ("Partition=all", "Partition=cs")):
        missing.append("Partition=all|cs")
    if inspected.returncode != 0 or missing or "gres/gpu" in record:
        subprocess.run(["scancel", str(job_id)], check=False)
        raise RuntimeError(
            f"post-campaign analysis scheduler record invalid: {missing}"
        )
    return {
        "job_id": job_id,
        "dependency_type": "afterany",
        "dependency_job_ids": [int(value) for value in dependency_ids],
        "dependency_cells": 90,
        "cpu_only": True,
        "scheduler_record": record,
    }


def repair_post_campaign_analysis(root: Path) -> dict[str, Any]:
    """Attach the missing CPU analysis receipt after scientific-job release."""

    ledger_path = root / LEDGER
    if not ledger_path.is_file():
        raise RuntimeError("E105 recovery requires the released campaign ledger")
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    runs = list(payload.get("runs", []))
    job_ids = [str(int(run["job_id"])) for run in runs]
    if (
        payload.get("schema")
        != "e105_group_centered_semantic_repair_full_three_scale_jobs_v1"
        or payload.get("released") is not True
        or len(job_ids) != 75
        or len(set(job_ids)) != 75
        or payload.get("pointmaze") != "excluded"
    ):
        raise RuntimeError("E105 recovery ledger is not the released 75-cell campaign")
    if payload.get("post_campaign_analysis") is not None:
        raise RuntimeError("E105 post-campaign analysis is already attached")
    receipt = submit_post_campaign_analysis(root, e105_job_ids=job_ids)
    payload["post_campaign_analysis"] = receipt
    payload["post_campaign_analysis_recovery"] = {
        "reason": "requested all partition canonicalized to effective cs",
        "scientific_jobs_resubmitted": False,
        "scientific_configuration_changed": False,
        "recovery_launcher_sha256": e104.digest(Path(__file__)),
    }
    e81.atomic_json(ledger_path, payload)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--repair-post-campaign-analysis", action="store_true")
    args = parser.parse_args()
    selected = sum(
        (args.submit, args.dry_run, args.repair_post_campaign_analysis)
    )
    if selected > 1:
        raise SystemExit("choose exactly one E105 action")
    root = repo_root()
    protocol = root / PROTOCOL
    repair_amendment = root / REPAIR_AMENDMENT
    python_comparator_amendment = root / PYTHON_COMPARATOR_AMENDMENT
    analysis_protocol = root / ANALYSIS_PROTOCOL
    analysis_builder = root / ANALYSIS_BUILDER
    analysis_plotter = root / ANALYSIS_PLOTTER
    post_campaign_analysis_protocol = root / POST_CAMPAIGN_ANALYSIS_PROTOCOL
    post_campaign_analysis_script = root / POST_CAMPAIGN_ANALYSIS_SCRIPT
    qwen3_placement_protocol = root / QWEN3_PLACEMENT_PROTOCOL
    qwen3_placement_script = root / QWEN3_PLACEMENT_SCRIPT
    for required in (
        protocol,
        repair_amendment,
        python_comparator_amendment,
        analysis_protocol,
        analysis_builder,
        analysis_plotter,
        post_campaign_analysis_protocol,
        post_campaign_analysis_script,
        qwen3_placement_protocol,
        qwen3_placement_script,
    ):
        if not required.is_file():
            raise SystemExit(f"E105 frozen input is absent: {required}")
    ledger_path = root / LEDGER
    if args.repair_post_campaign_analysis:
        receipt = repair_post_campaign_analysis(root)
        print(
            f"[e105-analysis-recovery] job={receipt['job_id']} "
            f"dependency_cells={receipt['dependency_cells']} ledger={ledger_path}"
        )
        return 0
    if args.submit and ledger_path.exists():
        raise SystemExit(f"refusing duplicate E105 submission: {ledger_path}")
    if args.submit:
        snapshot, audit_path, audit, e104_audit = check_gate(root)
        _placement, qwen3_a6000_cells = require_qwen3_paired_placement(root)
    else:
        if args.snapshot_root is None:
            raise SystemExit("E105 dry-run requires the exact repaired --snapshot-root")
        snapshot = args.snapshot_root.resolve()
        if snapshot != (root / e106.SNAPSHOT).resolve():
            raise SystemExit("E105 dry-run requires E106's exact repaired snapshot")
        e106.verify_snapshot(root, snapshot)
        audit_path = root / E106_AUDIT
        audit = {}
        e104_audit = {}
        qwen3_a6000_cells = set()

    comparators = {
        scale: comparator_index(
            root,
            scale,
            allow_legacy_python=not args.submit,
        )
        for scale in SCALE_SEEDS
    }
    planned: list[dict[str, Any]] = []
    for scale in SCALE_SEEDS:
        for run in references(root, scale):
            domain = str(run["domain"])
            seed = int(run["seed"])
            env, target = build_env(root, scale, run, snapshot)
            if args.submit and target.exists():
                raise SystemExit(f"refusing to overwrite E105 run: {target}")
            comparator = comparators[scale][(domain, seed)]
            planned.append(
                {
                    "scale": scale,
                    "model_tag": MODEL_TAGS[scale],
                    "arm": ARM,
                    "domain": domain,
                    "seed": seed,
                    "run_stamp": run_stamp(scale, domain, seed),
                    "run_dir": str(target),
                    "paired_replay": {
                        "run_stamp": str(comparator["run_stamp"]),
                        "run_dir": str(comparator["run_dir"]),
                        "job_id": int(comparator["job_id"]),
                    },
                    "command": sbatch_command(
                        root, scale, run, env, qwen3_a6000_cells
                    ),
                    "template": run,
                }
            )
    if len(planned) != 75 or Counter(cell["scale"] for cell in planned) != {
        "qwen05b": 25,
        "falcon1b": 25,
        "qwen3b": 25,
    }:
        raise SystemExit("E105 did not materialize exactly 25 cells per scale")
    if args.dry_run or not args.submit:
        for cell in planned:
            print(" ".join(shlex.quote(token) for token in cell["command"]))
        print(f"[e105] dry_run=True cells=75 snapshot={snapshot}")
        return 0

    submitted: list[str] = []
    records: list[dict[str, Any]] = []
    try:
        for cell in planned:
            result = subprocess.run(
                cell["command"], capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"submission failed for {cell['run_stamp']}: "
                    f"{result.stderr.strip()}"
                )
            job_id = result.stdout.strip().split(";", 1)[0]
            if not job_id.isdigit():
                raise RuntimeError(f"invalid job id: {result.stdout!r}")
            submitted.append(job_id)
            held = held_job_audit(
                job_id,
                str(cell["scale"]),
                cell["template"],
                snapshot,
                qwen3_a6000_cells,
            )
            records.append(
                {
                    key: cell[key]
                    for key in (
                        "scale",
                        "model_tag",
                        "arm",
                        "domain",
                        "seed",
                        "run_stamp",
                        "run_dir",
                        "paired_replay",
                    )
                }
                | {"job_id": int(job_id), "held_scheduler_record": held}
            )
        payload = {
            "schema": "e105_group_centered_semantic_repair_full_three_scale_jobs_v1",
            "protocol": str(protocol),
            "protocol_sha256": e104.digest(protocol),
            "repair_amendment": str(repair_amendment),
            "repair_amendment_sha256": e104.digest(repair_amendment),
            "python_comparator_amendment": str(python_comparator_amendment),
            "python_comparator_amendment_sha256": e104.digest(
                python_comparator_amendment
            ),
            "repaired_python_comparator_ledger": str(
                root / REPAIRED_PYTHON_COMPARATOR_LEDGER
            ),
            "repaired_python_comparator_ledger_sha256": e104.digest(
                root / REPAIRED_PYTHON_COMPARATOR_LEDGER
            ),
            "historical_comparator_ledgers": {
                scale: {
                    "path": str(root / path),
                    "sha256": e104.digest(root / path),
                }
                for scale, path in COMPARATOR_LEDGERS.items()
            },
            "analysis_protocol": str(analysis_protocol),
            "analysis_protocol_sha256": e104.digest(analysis_protocol),
            "analysis_builder": str(analysis_builder),
            "analysis_builder_sha256": e104.digest(analysis_builder),
            "analysis_plotter": str(analysis_plotter),
            "analysis_plotter_sha256": e104.digest(analysis_plotter),
            "post_campaign_analysis_protocol": str(
                post_campaign_analysis_protocol
            ),
            "post_campaign_analysis_protocol_sha256": e104.digest(
                post_campaign_analysis_protocol
            ),
            "post_campaign_analysis_script": str(post_campaign_analysis_script),
            "post_campaign_analysis_script_sha256": e104.digest(
                post_campaign_analysis_script
            ),
            "qwen3_paired_placement_protocol": str(qwen3_placement_protocol),
            "qwen3_paired_placement_protocol_sha256": e104.digest(
                qwen3_placement_protocol
            ),
            "qwen3_paired_placement_script": str(qwen3_placement_script),
            "qwen3_paired_placement_script_sha256": e104.digest(
                qwen3_placement_script
            ),
            "qwen3_paired_placement_artifact": str(
                root / QWEN3_PLACEMENT_ARTIFACT
            ),
            "qwen3_paired_placement_artifact_sha256": e104.digest(
                root / QWEN3_PLACEMENT_ARTIFACT
            ),
            "qwen3_a6000_cells": [
                {"domain": domain, "seed": seed}
                for domain, seed in sorted(qwen3_a6000_cells)
            ],
            "launcher_sha256": e104.digest(Path(__file__)),
            "e104_ledger": str(root / E104_LEDGER),
            "e104_ledger_sha256": e104.digest(root / E104_LEDGER),
            "e104_audit": str(root / E104_AUDIT),
            "e104_audit_sha256": e104.digest(root / E104_AUDIT),
            "e106_ledger": str(root / E106_LEDGER),
            "e106_ledger_sha256": e104.digest(root / E106_LEDGER),
            "e106_combined_audit": str(audit_path),
            "e106_combined_audit_sha256": e104.digest(audit_path),
            "e104_deviation": str(root / E104_DEVIATION),
            "e104_deviation_sha256": e104.digest(root / E104_DEVIATION),
            "e104_theory_clarification": str(
                root / E104_THEORY_CLARIFICATION
            ),
            "e104_theory_clarification_sha256": e104.digest(
                root / E104_THEORY_CLARIFICATION
            ),
            "e104_placement_amendment": str(
                root / E104_PLACEMENT_AMENDMENT
            ),
            "e104_placement_amendment_sha256": e104.digest(
                root / E104_PLACEMENT_AMENDMENT
            ),
            "e104_qwen3_a6000_placement_amendment": e104_audit.get(
                "qwen3_a6000_placement_amendment"
            ),
            "e104_unit_test_evidence": str(root / E104_UNIT_EVIDENCE),
            "e104_unit_test_evidence_sha256": e104.digest(
                root / E104_UNIT_EVIDENCE
            ),
            "e106_unit_test_evidence": str(root / E106_UNIT_EVIDENCE),
            "e106_unit_test_evidence_sha256": e104.digest(
                root / E106_UNIT_EVIDENCE
            ),
            "sparse_success_theory_evidence": str(
                root / SPARSE_THEORY_EVIDENCE
            ),
            "sparse_success_theory_evidence_sha256": e104.digest(
                root / SPARSE_THEORY_EVIDENCE
            ),
            "e106_supplemental_evidence": {
                key: audit[key]
                for key in E106_SUPPLEMENTAL_LEDGER_KEYS
            },
            "snapshot_root": str(snapshot),
            "snapshot_sha256": e106.SNAPSHOT_SHA256,
            "python_response_surface_version": e106.SURFACE_VERSION,
            "python_comparator_repaired": True,
            "models": list(SCALE_SEEDS),
            "domains": list(DOMAINS),
            "seeds": {key: list(value) for key, value in SCALE_SEEDS.items()},
            "arms": [ARM],
            "train_rows": TRAIN_ROWS,
            "passes": PASSES,
            "target_steps": TARGET_STEPS,
            "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
            "semantic_coefficient": e104.SEMANTIC_COEFFICIENT,
            "replay_weight": e104.REPLAY_WEIGHT,
            "objective": "sampled_group_centered_semantic_score_plus_verified_replay",
            "pointmaze": "excluded",
            "runs": records,
            "released": False,
        }
        e81.atomic_json(ledger_path, payload)
        for job_id in submitted:
            release = subprocess.run(
                ["scontrol", "release", job_id],
                capture_output=True,
                text=True,
                check=False,
            )
            if release.returncode != 0:
                raise RuntimeError(f"release failed for {job_id}")
        payload["released"] = True
        e81.atomic_json(ledger_path, payload)
    except Exception:
        e81.cancel(submitted)
        if ledger_path.exists():
            ledger_path.unlink()
        raise
    post_campaign_analysis = submit_post_campaign_analysis(
        root,
        e105_job_ids=submitted,
    )
    payload["post_campaign_analysis"] = post_campaign_analysis
    e81.atomic_json(ledger_path, payload)
    print(f"[e105] cells=75 released=75 snapshot={snapshot} ledger={ledger_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
