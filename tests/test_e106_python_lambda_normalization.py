from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_e106_python_lambda_normalization_three_scale.py"
AUDITOR = ROOT / "ops/exp_scaling/audit_e106_python_lambda_normalization.py"
PROTOCOL = ROOT / "paper/preregistration/e106_python_lambda_normalization_three_scale_20260817.md"
DIAGNOSIS = ROOT / "var/artifacts/e106_python_lambda_normalization_diagnosis.json"
CANCELLATION = ROOT / "paper/preregistration/e106_superseded_e104_falcon_python_cancellation_20260817.md"
PLACEMENT = ROOT / "var/artifacts/e106_qwen3_a6000_placement.json"
PRIOR_SURFACE = ROOT / "var/artifacts/e106_prior_surface_mode_evidence.json"
INTEGRATION = ROOT / "var/artifacts/e106_parser_to_semantic_integration_tests.json"
CROSS_SCALE_SURFACE = ROOT / (
    "var/artifacts/e106_prior_python_surface_cross_scale_audit.json"
)
FALCON_BOOTSTRAP = ROOT / (
    "var/artifacts/e106_prior_falcon_python_bootstrap_audit.json"
)
EXTENDED_ESTIMATOR = ROOT / (
    "var/artifacts/e106_frozen_estimator_extended_tests.json"
)
TIME_LIMIT_AMENDMENT = ROOT / (
    "var/artifacts/e106_python_smoke_time_limit_amendment.json"
)
FALCON_POOL_AMENDMENT = ROOT / (
    "var/artifacts/e106_falcon_python_a6000_pool_amendment.json"
)
POLICY_GRADIENT_EVIDENCE = ROOT / (
    "var/artifacts/e105_policy_gradient_direction_tests.json"
)
SEMANTIC_REPLAY_IDENTITY_EVIDENCE = ROOT / (
    "var/artifacts/e105_semantic_replay_identity_contract_tests.json"
)
FALCON_POOL_WIDENING = ROOT / (
    "var/artifacts/e106_falcon_same_gpu_pool_widening_amendment.json"
)
FALCON_ALL_POOL = ROOT / (
    "var/artifacts/e106_falcon_a6000_all_partition_pool_amendment.json"
)
QWEN3_CLEAN_RESTART = ROOT / (
    "var/artifacts/"
    "e106_qwen3_preemption_clean_restart_all_partition_amendment.json"
)
E110_LEDGER = ROOT / (
    "var/artifacts/e110_falcon_python_admission_horizon_jobs.json"
)
E110_TIME_LIMIT_AMENDMENT = ROOT / (
    "var/artifacts/e110_two_hour_backfill_amendment.json"
)


def _load():
    spec = importlib.util.spec_from_file_location("e106_launcher", LAUNCHER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_auditor():
    spec = importlib.util.spec_from_file_location("e106_auditor", AUDITOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e106_is_python_only_at_all_three_scales():
    module = _load()
    assert module.DOMAIN == "python_factors"
    assert module.SCALES == ("qwen05b", "falcon1b", "qwen3b")
    assert module.SURFACE_VERSION == "python-factor-response-v2-latex-lambda"
    assert module.SNAPSHOT_SHA256.startswith("b853595e3b158046")
    assert "unit_evidence_sha256" in LAUNCHER.read_text()


def test_e106_snapshot_is_exactly_one_source_file_from_e104():
    module = _load()
    snapshot = ROOT / module.SNAPSHOT
    module.verify_snapshot(ROOT, snapshot)
    assert (
        "python-factor-response-v2-latex-lambda"
        in (snapshot / "src/oat_drgrpo/math_grader.py").read_text()
    )


def test_e106_diagnosis_is_outcome_blind_and_recovers_verified_seeds():
    payload = json.loads(DIAGNOSIS.read_text())
    assert payload["post_e104_update_outcomes_inspected"] is False
    assert payload["e104_qwen3b_python_mechanism"]["available_groups_max"] == 0
    assert payload["prior_analyzed_surface_replay"] == {
        "normalized_lambda_candidates": 128,
        "rows": 128,
        "source": "var/data/xdr_qwen25_3b_instruct_grpo_compute_matched_e80r1_qwen3b_aligned_python_control_s70/debug_job30277392/eval_results/0_multi_answer.json",
        "source_sha256": "d6d3964a548d58204f9326d879b605cff4378323d443a9a02fe39d5ac4305f09",
        "validated_after_fix": 3,
    }
    prior = json.loads(PRIOR_SURFACE.read_text())
    assert prior["post_e104_or_e106_update_outcomes_inspected"] is False
    assert prior["verified_rows"] == 3
    assert prior["distinct_endpoint_keys"] == 2
    assert prior["endpoint_multiplicities"] == [2, 1]


def test_e106_frozen_snapshot_passes_parser_to_semantic_integration():
    module = _load()
    evidence = json.loads(INTEGRATION.read_text())
    assert evidence["passed"] is True
    assert evidence["returncode"] == 0
    assert evidence["snapshot_sha256"] == module.SNAPSHOT_SHA256
    assert evidence["post_e104_or_e106_update_outcomes_inspected"] is False
    assert evidence["pointmaze"] == "excluded"
    assert "1 passed" in evidence["stdout"]
    assert "PRIOR_SURFACE_EVIDENCE" in AUDITOR.read_text()
    assert "INTEGRATION_EVIDENCE" in AUDITOR.read_text()


def test_e106_prior_evidence_covers_python_surfaces_at_all_three_scales():
    evidence = json.loads(CROSS_SCALE_SURFACE.read_text())
    assert evidence["passed"] is True
    assert evidence["stored_scores_read"] is False
    assert evidence["post_e104_or_e106_update_outcomes_inspected"] is False
    assert evidence["pointmaze"] == "excluded"
    assert set(evidence["scales"]) == {"qwen05b", "falcon1b", "qwen3b"}
    qwen3 = evidence["scales"]["qwen3b"]
    assert qwen3["totals"]["legacy_verified"] == 0
    assert qwen3["totals"]["recovered_verified"] == 3
    assert qwen3["max_repaired_distinct_endpoint_keys_in_one_seed"] == 2
    for scale in evidence["scales"].values():
        assert scale["totals"]["regressed_verified"] == 0


def test_e106_prior_falcon_python_can_bootstrap_bank_and_replay():
    evidence = json.loads(FALCON_BOOTSTRAP.read_text())
    assert evidence["passed"] is True
    assert evidence["evaluation_outcomes_read"] is False
    assert evidence["post_e104_or_e106_update_outcomes_inspected"] is False
    assert evidence["pointmaze"] == "excluded"
    plain = evidence["runs"]["plain_grpo"]["maxima"]
    replay = evidence["runs"]["replay_semantic"]["maxima"]
    assert plain["train/online_canonical_bank_size_after_mean"] > 0.0
    assert plain["train/online_canonical_eligible_fraction"] > 0.0
    assert replay["train/canonical_replay_available_groups"] > 0.0
    assert replay["train/canonical_replay_applied_score_gradient_l2"] > 0.0
    assert (
        replay[
            "train/semantic_shannon_success_conditioned_signed_eligible_fraction"
        ]
        > 0.0
    )
    assert "CROSS_SCALE_SURFACE_EVIDENCE" in AUDITOR.read_text()
    assert "FALCON_BOOTSTRAP_EVIDENCE" in AUDITOR.read_text()


def test_e106_extended_estimator_tests_are_snapshot_bound():
    module = _load()
    evidence = json.loads(EXTENDED_ESTIMATOR.read_text())
    assert evidence["passed"] is True
    assert evidence["returncode"] == 0
    assert evidence["snapshot_sha256"] == module.SNAPSHOT_SHA256
    assert evidence["post_e104_or_e106_update_outcomes_inspected"] is False
    assert evidence["pointmaze"] == "excluded"
    assert "14 passed" in evidence["stdout"]
    assert set(evidence["tests_sha256"]) == {
        "tests/test_e106_parser_to_semantic_integration.py",
        "tests/test_semantic_shannon_group_centered.py",
        "tests/test_semantic_shannon_group_centered_theory.py",
    }
    assert "EXTENDED_ESTIMATOR_EVIDENCE" in AUDITOR.read_text()


def test_e106_pending_python_scheduler_amendments_change_no_environment():
    time_limit = json.loads(TIME_LIMIT_AMENDMENT.read_text())
    assert time_limit["scheduler_only"] is True
    assert time_limit["environment_changed"] is False
    assert time_limit["post_e104_or_e106_update_outcomes_inspected"] is False
    assert {item["job_id"] for item in time_limit["jobs"]} == {
        30640330,
        30640331,
    }
    assert all(
        item["before"]["time_limit"] == "08:00:00"
        and item["after"]["time_limit"] == "02:00:00"
        and item["before"]["runtime"] == "00:00:00"
        and item["after"]["runtime"] == "00:00:00"
        for item in time_limit["jobs"]
    )

    pool = json.loads(FALCON_POOL_AMENDMENT.read_text())
    assert pool["job_id"] == 30640330
    assert pool["scheduler_only"] is True
    assert pool["environment_changed"] is False
    assert pool["post_e104_or_e106_update_outcomes_inspected"] is False
    assert pool["before"]["node_list"] == "node207"
    assert pool["effective"]["node_list"] == "node[205-207]"
    assert pool["before"]["gres"] == pool["effective"]["gres"] == "gpu:a6000:1"
    assert "TIME_LIMIT_AMENDMENT" in AUDITOR.read_text()
    assert "FALCON_POOL_AMENDMENT" in AUDITOR.read_text()


def test_e106_protocol_and_gate_exclude_pointmaze_and_lock_e105():
    text = (PROTOCOL.read_text() + LAUNCHER.read_text() + AUDITOR.read_text()).lower()
    assert "pointmaze" in text
    assert '"excluded"' in text
    assert "post_update_outcome_metrics_inspected" in AUDITOR.read_text()
    assert "mechanism_gate_used_outcome_metrics" in AUDITOR.read_text()
    assert "nonzero_semantic_scales" in AUDITOR.read_text()
    assert "completed_zero_semantic_cells" in AUDITOR.read_text()
    assert "no live nonzero semantic update for scales" in AUDITOR.read_text()
    assert "12" in AUDITOR.read_text()
    assert "e105" in PROTOCOL.read_text().lower()


def test_falcon_one_hour_amendment_is_outcome_blind_and_current():
    audit = _load_auditor()
    payload, violations = audit.validate_falcon_one_hour_amendment()

    assert not violations
    assert payload["scheduler_only"] is True
    assert payload["environment_changed"] is False
    assert payload["post_e104_or_e106_update_outcomes_inspected"] is False
    assert payload["pointmaze"] == "excluded"
    assert payload["reference"]["elapsed_seconds"] == 1_206
    assert payload["reference"]["one_hour_margin"] >= 2.8
    assert "30637792" in CANCELLATION.read_text()
    assert "30640330" in CANCELLATION.read_text()
    assert "CANCELLATION_NOTE" in AUDITOR.read_text()
    placement = json.loads(PLACEMENT.read_text())
    assert placement["job_id"] == 30640331
    assert placement["effective"]["gres"] == "gpu:a6000:1"
    assert placement["effective"]["runtime"] == "00:00:00"
    assert placement["environment_changed"] is False
    assert "QWEN3_PLACEMENT" in AUDITOR.read_text()


def test_policy_gradient_evidence_executes_frozen_production_loss_direction():
    audit = _load_auditor()
    snapshot = ROOT / (
        "var/artifacts/source_snapshots/"
        "e106_python_lambda_b853595e3b158046"
    )
    payload, violations = audit.validate_policy_gradient_evidence(snapshot)

    assert not violations
    assert payload["passed"] is True
    assert payload["returncode"] == 0
    assert payload["post_e104_or_e106_update_outcomes_inspected"] is False
    assert payload["pointmaze"] == "excluded"
    assert payload["assertions"] == {
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
    assert payload == json.loads(POLICY_GRADIENT_EVIDENCE.read_text())


def test_semantic_replay_identity_evidence_covers_all_e105_domains():
    audit = _load_auditor()
    snapshot = ROOT / (
        "var/artifacts/source_snapshots/"
        "e106_python_lambda_b853595e3b158046"
    )
    payload, violations = audit.validate_semantic_replay_identity_evidence(
        snapshot
    )

    assert not violations
    assert payload["passed"] is True
    assert payload["returncode"] == 0
    assert payload["group_size"] == 16
    assert payload["domains"] == [
        "graph_coloring",
        "countdown",
        "python_factors",
        "mathir",
        "pantry_plan",
    ]
    assert payload["post_e104_or_e106_update_outcomes_inspected"] is False
    assert payload["pointmaze"] == "excluded"
    assert payload["assertions"] == {
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
    assert "6 passed" in payload["stdout"]
    assert payload == json.loads(SEMANTIC_REPLAY_IDENTITY_EVIDENCE.read_text())


def test_falcon_pool_widening_is_outcome_blind_and_same_gpu():
    audit = _load_auditor()
    payload, violations = audit.validate_falcon_pool_widening_amendment()

    assert not violations
    assert payload["scheduler_only"] is True
    assert payload["environment_changed"] is False
    assert payload["gpu_type_changed"] is False
    assert payload["post_e104_or_e106_update_outcomes_inspected"] is False
    assert payload["pointmaze"] == "excluded"
    assert {
        item["job_id"]: (
            item["old_node_list"],
            item["new_node_list"],
            item["gres"],
        )
        for item in payload["jobs"]
    } == {
        30637790: ("node202", "node[202-204]", "gpu:a5000:1"),
        30637793: ("node202", "node[202-204]", "gpu:a5000:1"),
        30637794: ("node206", "node[205-207]", "gpu:a6000:1"),
    }
    assert payload == json.loads(FALCON_POOL_WIDENING.read_text())


def test_falcon_all_partition_pool_is_outcome_blind_and_same_a6000():
    audit = _load_auditor()
    payload, violations = audit.validate_falcon_all_pool_amendment()

    assert not violations
    assert payload["scheduler_only"] is True
    assert payload["environment_changed"] is False
    assert payload["gpu_type_changed"] is False
    assert payload["partition_changed"] is True
    assert payload["account_changed"] is False
    assert payload["post_e104_or_e106_update_outcomes_inspected"] is False
    assert payload["pointmaze"] == "excluded"
    assert {
        item["job_id"]: (
            item["old_partition"],
            item["new_partition"],
            item["old_node_list"],
            item["new_node_list"],
            item["gres"],
        )
        for item in payload["jobs"]
    } == {
        30637794: (
            "cs",
            "all",
            "node[205-207]",
            "node[103-104,205-208,805]",
            "gpu:a6000:1",
        ),
        30640330: (
            "cs",
            "all",
            "node[205-207]",
            "node[103-104,205-208,805]",
            "gpu:a6000:1",
        ),
    }
    assert payload == json.loads(FALCON_ALL_POOL.read_text())


def test_qwen3_preempted_prefix_is_archived_and_clean_restart_is_audited():
    audit = _load_auditor()
    payload, violations = audit.validate_qwen3_clean_restart_amendment()

    assert not violations
    assert payload == json.loads(QWEN3_CLEAN_RESTART.read_text())
    assert payload["released"] is True
    assert payload["interrupted_step"] == 22
    assert payload["checkpoint_present"] is False
    assert payload["clean_restart_required"] is True
    assert payload["old_partition"] == "lowprio"
    assert payload["new_partition"] == "all"
    assert payload["environment_changed"] is False
    assert payload["scientific_configuration_changed"] is False
    assert payload["post_e104_or_e106_update_outcomes_inspected"] is False
    assert payload["pointmaze"] == "excluded"
    assert Path(payload["archive_dir"]).is_dir()
    assert {
        item["path"] for item in payload["archived_prefix_manifest"]
    } == {
        "debug_job30640331/eval_results/0_multi_answer.json",
        "debug_job30640331/eval_results/16_multi_answer.json",
        "debug_job30640331/eval_mode_coverage_draws.jsonl",
        "debug_job30640331/train_metrics.jsonl",
    }


def test_e110_replaces_only_failed_falcon_python_at_admission_horizon():
    audit = _load_auditor()
    payload, violations = audit.validate_e110_replacement_ledger()

    assert not violations
    assert payload == json.loads(E110_LEDGER.read_text())
    assert payload["released"] is True
    assert payload["supersedes_job_id"] == 30640330
    assert payload["replacement_scope"] == ["falcon1b", "python_factors", 55]
    assert payload["target_steps"] == 192
    assert payload["historical_admission_evidence"]["first_admission_step"] == 179
    assert payload["historical_admission_evidence"]["mechanism_only"] is True
    assert (
        payload["historical_admission_evidence"]["evaluation_outcomes_inspected"]
        is False
    )
    assert payload["failed_e106_mechanism_evidence"]["last_step"] == 64
    assert payload["failed_e106_mechanism_evidence"]["bank_size_after_max"] == 0.0
    assert payload["cancelled_zero_runtime_attempt"]["job_id"] == 30647351
    assert payload["submitted_partition"] == "cs"
    assert payload["effective_partition"] == "all"
    assert payload["pointmaze"] == "excluded"


def test_e110_two_hour_backfill_amendment_is_scheduler_only_and_audited():
    audit = _load_auditor()
    payload, violations = audit.validate_e110_time_limit_amendment()

    assert not violations
    assert payload == json.loads(E110_TIME_LIMIT_AMENDMENT.read_text())
    assert payload["released"] is True
    assert payload["job_id"] == 30647379
    assert payload["old_time_limit"] == "03:00:00"
    assert payload["new_time_limit"] == "02:00:00"
    assert payload["scheduler_only"] is True
    assert payload["environment_changed"] is False
    assert payload["scientific_configuration_changed"] is False
    assert payload["post_e104_or_e106_or_e110_update_outcomes_inspected"] is False
    assert payload["pointmaze"] == "excluded"
    assert payload["runtime_references"]["e106_v6_64_step"]["elapsed"] == "00:39:03"
    assert payload["runtime_references"]["e79_replay_3072_step"]["elapsed"] == "11:45:28"
