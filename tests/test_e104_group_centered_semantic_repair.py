from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / (
    "ops/exp_scaling/"
    "launch_e104_group_centered_semantic_repair_three_scale.py"
)
AUDITOR = ROOT / (
    "ops/exp_scaling/audit_e104_group_centered_semantic_repair.py"
)
PROTOCOL = ROOT / (
    "paper/preregistration/"
    "e104_group_centered_semantic_repair_three_scale_20260817.md"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e104_is_three_scales_five_static_domains_and_no_pointmaze():
    launch = _load(LAUNCHER, "e104_launch")
    assert launch.SCALE_SEEDS == {
        "qwen05b": 43,
        "falcon1b": 55,
        "qwen3b": 70,
    }
    assert launch.DOMAINS == (
        "graph_coloring",
        "countdown",
        "python_factors",
        "mathir",
        "pantry_plan",
    )
    assert launch.SMOKE_TARGET_STEPS == 64
    assert "point" not in " ".join(launch.DOMAINS).lower()


def test_e104_objective_is_replay_plus_only_the_repaired_fixed_semantic_term():
    launch = _load(LAUNCHER, "e104_objective")
    objective = launch.fixed_objective()
    assert objective["OAT_ZERO_VARIANT"] == (
        "verified_replay_semantic_maxent_group_centered"
    )
    assert objective["OAT_ZERO_SEMANTIC_SHANNON_COEF"] == "0.1"
    assert objective[
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE"
    ] == "0"
    assert objective[
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE"
    ] == "1"
    assert objective["OAT_ZERO_SEMANTIC_RMS_CONTROL"] == "0"
    assert objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY"] == "1"
    assert objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE"] == (
        "verified_likelihood_per_rollout"
    )
    assert objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA"] == "0.1"
    assert objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY"] == "0"
    assert objective["OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS"] == "0"
    assert objective["OAT_ZERO_MAXENT_ALPHA"] == "0.0"
    assert objective["OAT_ZERO_POLICY_ENTROPY_COEF"] == "0.0"


def test_e104_protocol_freezes_outcome_blind_gate_and_full_followup():
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "Frozen before any repaired model was trained or evaluated" in text
    assert "No evaluation correctness" in text
    assert "E105 contains 75" in text
    assert "repaired treatment cells" in text
    assert "PointMaze is excluded" in text
    auditor = AUDITOR.read_text(encoding="utf-8")
    for forbidden in ("pass@", "distinct@", "mean_correct", "eval/pass"):
        assert forbidden not in auditor
    assert "snapshot unit-test evidence is absent" in auditor
    assert "unit_evidence.validate_evidence" in auditor
    assert "validate_qwen3_amendment" in auditor
    assert '"qwen3_a6000_placement_amendment"' in auditor
    assert auditor.index("snapshot = Path") < auditor.index(
        "unit_evidence.validate_evidence"
    )


def test_e104_auditor_accepts_only_centered_bounded_live_semantic_updates(tmp_path):
    audit = _load(AUDITOR, "e104_audit")
    attempt = tmp_path / "debug_job1"
    attempt.mkdir()
    rows = []
    for step in range(1, 65):
        rows.append(
            {
                "misc/global_step": step,
                "train/semantic_shannon_success_conditioned_group_centered_advantage_active": 1.0,
                "train/semantic_shannon_success_conditioned_signed_advantage_active": 0.0,
                "train/semantic_rms_controller_active": 0.0,
                "train/semantic_shannon_success_conditioned_group_centered_effective_advantage_mean": 0.0,
                "train/semantic_shannon_success_conditioned_group_centered_effective_advantage_min": -0.02,
                "train/semantic_shannon_success_conditioned_group_centered_effective_advantage_max": 0.06,
                "train/semantic_shannon_success_conditioned_group_centered_effective_advantage_rms": 0.01,
                "train/semantic_shannon_separate_semantic_advantage_min": -0.02,
                "train/semantic_shannon_separate_semantic_advantage_max": 0.06,
                "train/canonical_replay_applied_score_gradient_l2": 0.1,
                "train/canonical_replay_actuator_groups": 1.0,
                "train/online_canonical_bank_size_after_mean": 2.0,
                "train/online_canonical_eligible_fraction": 0.5,
                "train/online_canonical_support_at_least_two_prompt_fraction": 0.25,
                "train/verified_discovery_mean_support_per_prompt": 1.25,
                "train/semantic_shannon_success_conditioned_group_centered_eligible_fraction": 0.5,
                "train/semantic_shannon_success_conditioned_group_centered_history_rows_added": 8.0,
            }
        )
    (attempt / "train_metrics.jsonl").write_text(
        "".join(__import__("json").dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    report, violations = audit.parse_run(tmp_path)
    assert violations == []
    assert report["last_step"] == 64
    assert report["semantic_rms_max"] == 0.01
    assert report["bank_size_after_max"] == 2.0
    assert report["online_eligible_fraction_max"] == 0.5
    assert report["support_at_least_two_prompt_fraction_max"] == 0.25
    assert report["mean_support_per_prompt_max"] == 1.25
    assert report["semantic_eligible_fraction_max"] == 0.5
    assert report["history_rows_added_max"] == 8.0

    _, long_horizon_violations = audit.parse_run(tmp_path, target_steps=192)
    assert "only 64/192 optimizer steps" in long_horizon_violations


def test_e104_auditor_rejects_the_legacy_uniform_negative_failure(tmp_path):
    audit = _load(AUDITOR, "e104_audit_failure")
    attempt = tmp_path / "debug_job1"
    attempt.mkdir()
    row = {
        "misc/global_step": 64,
        "train/semantic_shannon_success_conditioned_group_centered_advantage_active": 1.0,
        "train/semantic_shannon_success_conditioned_signed_advantage_active": 0.0,
        "train/semantic_rms_controller_active": 0.0,
        "train/semantic_shannon_success_conditioned_group_centered_effective_advantage_mean": -0.001,
        "train/semantic_shannon_success_conditioned_group_centered_effective_advantage_min": -0.001,
        "train/semantic_shannon_success_conditioned_group_centered_effective_advantage_max": -0.001,
        "train/semantic_shannon_success_conditioned_group_centered_effective_advantage_rms": 0.001,
        "train/canonical_replay_applied_score_gradient_l2": 0.1,
        "train/canonical_replay_actuator_groups": 1.0,
    }
    (attempt / "train_metrics.jsonl").write_text(
        __import__("json").dumps(row) + "\n", encoding="utf-8"
    )
    _, violations = audit.parse_run(tmp_path)
    assert any("mean exceeded tolerance" in value for value in violations)
