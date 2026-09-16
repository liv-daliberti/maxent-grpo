from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_e111_verified_support_discovery_mechanism_gate_three_scale.py"
AUDITOR = ROOT / "ops/exp_scaling/audit_e111_verified_support_discovery_mechanism_gate.py"
PROTOCOL = ROOT / "paper/preregistration/e111_verified_support_discovery_mechanism_gate_three_scale_20260818.md"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e111_is_recurrence_matched_three_scales_five_domains_no_pointmaze():
    launch = _load(LAUNCHER, "e111_launch")
    assert launch.SCALE_SEEDS == {"qwen05b": 43, "falcon1b": 55, "qwen3b": 70}
    assert launch.DOMAINS == (
        "graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan"
    )
    assert launch.SMOKE_TRAIN_ROWS == 8
    assert launch.SMOKE_PASSES == 8
    assert launch.SMOKE_TARGET_STEPS == 64
    assert "point" not in " ".join(launch.DOMAINS).lower()


def test_e111_objective_is_v7_uniform_replaydr_and_support_discovery():
    launch = _load(LAUNCHER, "e111_objective")
    objective = launch.fixed_objective()
    assert objective["OAT_ZERO_VARIANT"] == "verified_replay_semantic_maxent_verified_support_discovery"
    assert objective["OAT_ZERO_SEMANTIC_SHANNON_COEF"] == "0.1"
    assert objective["OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE"] == "0"
    assert objective["OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE"] == "0"
    assert objective["OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_VERIFIED_SUPPORT_ADVANTAGE"] == "1"
    assert objective["OAT_ZERO_SEMANTIC_SHANNON_VERIFIED_SUPPORT_INCLUDE_REPLAY_BANK"] == "1"
    assert objective["OAT_ZERO_SEMANTIC_RMS_CONTROL"] == "0"
    assert objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY"] == "1"
    assert objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE"] == "verified_likelihood_per_rollout"
    assert objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA"] == "0.1"
    assert objective["OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS"] == "1"
    assert objective["OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT"] == "1"
    assert objective["OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_TRANSFORM_PROPOSALS"] == "0"
    assert objective["OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_EXACT_GRAMMAR_TRANSFORMS"] == "0"
    assert objective["OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SAMPLING_TEMPERATURE"] == "1.2"
    assert objective["OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_VISITS"] == "0"
    assert objective["OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_MULTIPLIER"] == "1.0"
    assert objective["OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_RETENTION_TRACKING"] == "1"
    assert objective["OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_ADAPTIVE_RETENTION_PRIORITY"] == "0"


def test_e111_protocol_and_auditor_are_outcome_blind():
    text = PROTOCOL.read_text(encoding="utf-8")
    for required in (
        "before submission of any E111 job",
        "There is no structural unseen bucket",
        "Proposal rows never enter PPO",
        "first eight training prompts",
        "giving 64 optimizer updates",
        "each model scale",
        "E105 is permanently superseded",
        "PointMaze",
    ):
        assert required in text
    auditor = AUDITOR.read_text(encoding="utf-8")
    for forbidden in ("pass@", "distinct@", "mean_correct", "eval/pass"):
        assert forbidden not in auditor
    assert "\"outcomes_used_for_gate\": False" in auditor
    assert "post_freeze_training_reward_exposure" in auditor
    assert "completed_causal_chain" in auditor


def test_e111_build_env_preserves_recurrence_and_action_interface(tmp_path):
    launch = _load(LAUNCHER, "e111_env")
    runs = launch.references(ROOT, "qwen05b")
    graph = next(run for run in runs if run["domain"] == "graph_coloring")
    pantry = next(run for run in runs if run["domain"] == "pantry_plan")
    graph_env, _ = launch.build_env(ROOT, "qwen05b", graph, tmp_path)
    pantry_env, _ = launch.build_env(ROOT, "qwen05b", pantry, tmp_path)
    for env in (graph_env, pantry_env):
        assert env["OAT_ZERO_MAX_TRAIN"] == "8"
        assert env["OAT_ZERO_NUM_PROMPT_EPOCH"] == "8"
        assert env["OAT_ZERO_MAX_PROMPT_EPOCHS"] == "8"
        assert env["OAT_ZERO_EVAL_PROMPT_INTERVAL"] == "64"
        assert env["OAT_ZERO_SAVE_STEPS"] == "32"
        assert env["OAT_ZERO_VARIANT"] == launch.VARIANT
    assert graph_env["OAT_ZERO_REPLICATED_FREEFORM_SAMPLING"] == "1"
    assert graph_env["OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC"] == "1"
    assert pantry_env["OAT_ZERO_REPLICATED_FREEFORM_SAMPLING"] == "0"
    assert pantry_env["OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC"] == "0"


def test_e111_qwen3_uses_proven_a6000_pool(tmp_path):
    launch = _load(LAUNCHER, "e111_placement")
    run = launch.references(ROOT, "qwen3b")[0]
    env, _ = launch.build_env(ROOT, "qwen3b", run, tmp_path)
    command = launch.sbatch_command(ROOT, "qwen3b", run, env)
    assert "--partition=all" in command
    assert "--account=mltheory" in command
    assert f"--nodelist={launch.QWEN3_A6000_NODES}" in command
    assert "--gres=gpu:a6000:1" in command
    assert "--time=12:00:00" in command
    source = LAUNCHER.read_text(encoding="utf-8")
    assert (chr(34) + "Partition=mltheory" + chr(34) + ",") in source
    assert (chr(34) + "Account=mltheory" + chr(34) + ",") in source


def _mechanism_row(*, mass_weight_max: float = 1.0) -> dict[str, float]:
    prefix = "train/semantic_shannon_success_conditioned_verified_support_"
    return {
        "misc/global_step": 64,
        "train/semantic_shannon_success_conditioned_verified_support_advantage_active": 1.0,
        "train/semantic_shannon_verified_support_include_replay_bank_active": 1.0,
        "train/semantic_shannon_success_conditioned_signed_advantage_active": 0.0,
        "train/semantic_shannon_success_conditioned_group_centered_advantage_active": 0.0,
        "train/semantic_rms_controller_active": 0.0,
        prefix + "effective_advantage_min": -0.02,
        prefix + "effective_advantage_max": 0.04,
        prefix + "effective_advantage_rms": 0.01,
        prefix + "eligible_fraction": 0.25,
        prefix + "verified_support_size_mean": 2.0,
        prefix + "verified_support_at_least_two_eligible_fraction": 1.0,
        prefix + "external_verified_support_size_mean": 2.0,
        prefix + "external_verified_support_nonempty_group_fraction": 1.0,
        prefix + "history_rows_added": 4.0,
        "train/canonical_replay_actuator_groups": 1.0,
        "train/canonical_replay_applied_score_gradient_l2": 0.1,
        "train/canonical_replay_mass_weight_min": 1.0,
        "train/canonical_replay_mass_weight_max": mass_weight_max,
        "train/canonical_replay_priority_modes": 0.0,
        "train/canonical_replay_priority_replay_groups_cumulative": 0.0,
        "train/canonical_replay_priority_replay_modes_cumulative": 0.0,
        "train/canonical_replay_applied_positive_gradient_max": 0.0,
        "train/counterfactual_proposal_groups_generated": 1.0,
        "train/counterfactual_proposal_rows_generated": 16.0,
        "train/counterfactual_proposal_cumulative_new_outcomes": 1.0,
        "train/counterfactual_proposal_conditioned_rows_sent_to_ppo": 0.0,
        "train/counterfactual_proposal_transform_rows_sent_to_ppo": 0.0,
        "train/counterfactual_proposal_objective_outcome_delta": 0.0,
        "train/counterfactual_proposal_gold_support_feedback": 0.0,
        "train/counterfactual_proposal_desired_mode_count_feedback": 0.0,
        "train/counterfactual_proposal_eval_feedback": 0.0,
        "train/counterfactual_proposal_transform_enabled": 0.0,
        "train/counterfactual_proposal_exact_grammar_transform_enabled": 0.0,
        "train/counterfactual_proposal_objective_support_separated": 1.0,
        "train/canonical_replay_proposal_retention_tracking_enabled": 1.0,
        "train/canonical_replay_proposal_retention_adaptive_priority_enabled": 0.0,
    }


def test_e111_auditor_accepts_complete_uniform_causal_chain(tmp_path):
    audit = _load(AUDITOR, "e111_audit")
    attempt = tmp_path / "debug_job1"
    attempt.mkdir()
    (attempt / "train_metrics.jsonl").write_text(
        json.dumps(_mechanism_row()) + "\n", encoding="utf-8"
    )
    report, violations = audit.parse_run(tmp_path)
    assert violations == []
    assert audit.completed_causal_chain(report)
    assert report["proposal_cumulative_admissions"] == 1.0
    assert report["external_support_nonempty_group_fraction_max"] == 1.0
    assert report["support_at_least_two_eligible_fraction_max"] == 1.0
    assert report["semantic_rms_max"] == 0.01
    assert report["replay_gradient_l2_max"] == 0.1
    assert report["mass_weight_min"] == 1.0
    assert report["mass_weight_max"] == 1.0


def test_e111_auditor_rejects_nonuniform_replay_priority(tmp_path):
    audit = _load(AUDITOR, "e111_audit_nonuniform")
    attempt = tmp_path / "debug_job1"
    attempt.mkdir()
    (attempt / "train_metrics.jsonl").write_text(
        json.dumps(_mechanism_row(mass_weight_max=4.0)) + "\n", encoding="utf-8"
    )
    _, violations = audit.parse_run(tmp_path)
    assert any("mass weights were nonuniform" in value for value in violations)



def test_e111_small_scales_use_current_healthy_general_pool(tmp_path):
    launch = _load(LAUNCHER, "e111_general_placement")
    for scale in ("qwen05b", "falcon1b"):
        run = launch.references(ROOT, scale)[0]
        env, _ = launch.build_env(ROOT, scale, run, tmp_path)
        command = launch.sbatch_command(ROOT, scale, run, env)
        assert "--partition=all" in command
        assert "--account=mltheory" in command
        assert f"--nodelist={launch.GENERAL_GPU_NODES}" in command
        assert "--gres=gpu:1" in command
        assert "--time=08:00:00" in command
