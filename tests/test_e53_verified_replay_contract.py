from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]
PROTOCOL = (
    ROOT / "paper/preregistration/e53_verified_exemplar_replay_05b.md"
)
LAUNCHER = (
    ROOT / "ops/exp_scaling/launch_e53_verified_exemplar_replay_05b.sh"
)
AUDITOR = ROOT / "ops/exp_scaling/audit_e53_sentinel.py"
SPEC = importlib.util.spec_from_file_location("audit_e53_sentinel", AUDITOR)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _base_record(step: int, *, active: bool = True) -> dict[str, float]:
    record = {
        "trainer/global_step": float(step),
        "misc/prompt_consumed": float(step),
        "train/policy_grad_norm": 0.25,
        "actor/no_eos_count": 0.0,
        "actor/response_tok_len": 24.0,
        "train/online_canonical_entropy_alpha_used": 0.0,
        "train/maxent_conditional_token_entropy": 2.0,
        "train/maxent_inverse_observed_entropy": 2.0,
        "train/maxent_inverse_entropy_ema": 2.0,
        "train/maxent_inverse_multiplier": 1.0,
        "train/maxent_alpha_used": 0.000075,
        "train/maxent_inverse_next_alpha": 0.000075,
        "train/maxent_inverse_observations": float(step),
        "train/maxent_inverse_projection_active": 0.0,
        "train/maxent_entropy_loss": -0.001,
        "train/canonical_replay_available_groups": float(active),
        "train/canonical_replay_available_modes": 2.0 if active else 0.0,
        "train/canonical_replay_capacity": 16.0,
        "train/canonical_replay_gold_support_feedback": 0.0,
        "train/canonical_replay_alpha_projection_active": 0.0,
        "train/canonical_replay_observation_skipped": 0.0 if active else 1.0,
        "train/canonical_replay_observations": float(step if active else 0),
        "train/canonical_replay_next_alpha": 0.1,
        "train/canonical_replay_projection_active": 0.0,
    }
    if active:
        record.update(
            {
                "train/canonical_replay_balance_loss": 0.2,
                "train/canonical_replay_weighted_loss": 0.01875,
                "train/canonical_replay_backward_scale": 1.5,
                "train/canonical_replay_chunk_size": 1.0,
                "train/canonical_replay_score_passes": 2.0,
                "train/canonical_replay_normalized_model_entropy": 0.8,
                "train/canonical_replay_cross_entropy_excess": 0.2,
                "train/canonical_replay_alpha_used": 0.1,
                "train/canonical_replay_eligible_groups": 1.0,
                "train/canonical_replay_retained_modes": 2.0,
                "train/canonical_replay_reward_estimator_scale": 0.9375,
                "train/canonical_replay_observed_normalized_entropy": 0.8,
                "train/canonical_replay_entropy_ema": 0.8,
                "train/canonical_replay_inverse_multiplier": 1.0,
                "train/canonical_replay_alpha_before": 0.1,
            }
        )
    return record


def test_e53_contract_is_target_free_unbounded_and_cross_domain():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    for required in (
        "FROZEN BEFORE E53 SENTINEL SUBMISSION",
        "KL(U(B_x) || q_theta(. | x))",
        "`alpha_(t+1) = 0.10 * z_ref / z_ema,t`",
        "There is no lower or upper projection",
        "gold list or count of valid outcomes",
        "exactly 50 complete prompt-pool passes",
        "retain at least 75%",
    ):
        assert required in protocol
    for required in (
        "EXPECTED_JOBS=9",
        "PROMPT_EPOCHS=50",
        "OAT_ZERO_TRAIN_SEEDS=9010",
        "OAT_ZERO_ONLY_ARMS=grpo,maxent_inverse,maxent_inverse_canonical_replay",
        "OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.0",
        "OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.50",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.10",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY=16",
        "auditor_sha256",
        "scontrol release",
    ):
        assert required in launcher


def test_e53_replay_audit_accepts_exact_active_and_idle_arithmetic():
    active = MODULE.audit_run(
        [_base_record(step) for step in range(1, 5)],
        arm=MODULE.REPLAY_ARM,
        prompt_pool_size=384,
    )
    assert active["violations"] == []
    assert active["replay_activations"] == 4
    assert active["latest"]["replay_alpha_used"] == 0.1

    idle = MODULE.audit_run(
        [_base_record(1, active=False)],
        arm=MODULE.REPLAY_ARM,
        prompt_pool_size=384,
    )
    assert idle["violations"] == []
    assert idle["replay_activations"] == 0


def test_e53_replay_audit_rejects_projection_sensor_and_scaling_drift():
    record = _base_record(1)
    record["train/canonical_replay_alpha_projection_active"] = 1.0
    record["train/canonical_replay_observed_normalized_entropy"] = 0.7
    record["train/canonical_replay_backward_scale"] = 0.1
    result = MODULE.audit_run(
        [record],
        arm=MODULE.REPLAY_ARM,
        prompt_pool_size=384,
    )
    assert any("projection/gold-support" in item for item in result["violations"])
    assert any("entropy sensor mismatch" in item for item in result["violations"])
    assert any("execution scaling mismatch" in item for item in result["violations"])


def test_e53_behavior_requires_75pct_self_retention_and_multiplicity():
    control_rows = [
        {"step": step, "distinct8": 0.2, "pass8": 0.1, "mean8": 0.1}
        for step in range(1, 17)
    ]
    replay_rows = [
        {
            "step": step,
            "distinct8": 1.0 if step <= 8 else 0.7,
            "pass8": 0.2,
            "mean8": 0.2,
        }
        for step in range(1, 17)
    ]
    result = MODULE.behavioral_gate(
        {"status": "complete", "evaluations": control_rows},
        {"status": "complete", "evaluations": replay_rows},
    )
    assert result["status"] == "fail"
    assert result["checks"]["retains_75pct_of_own_best_rolling_eight"] is False


def test_e53_only_terminal_pass_persists_stage_a_approval(tmp_path):
    audit_out = tmp_path / "audit.json"
    approval_out = tmp_path / "approval.json"
    approval_out.write_text('{"status":"pass"}\n', encoding="utf-8")
    MODULE.write_audit_outputs(
        payload={"status": "in_progress"},
        audit_out=audit_out,
        approval_out=approval_out,
    )
    assert not approval_out.exists()
    MODULE.write_audit_outputs(
        payload={"status": "pass", "authorizes_stage_a": True},
        audit_out=audit_out,
        approval_out=approval_out,
    )
    assert MODULE.json.loads(approval_out.read_text())["authorizes_stage_a"] is True


def test_e53_checkpoint_gate_fails_closed_when_terminal_state_is_missing(
    tmp_path,
):
    result = MODULE.checkpoint_gate(
        tmp_path,
        run_stamp="missing",
        arm=MODULE.REPLAY_ARM,
        terminal_step=192 * 50,
    )
    assert result["status"] == "fail"
    assert "missing terminal checkpoint" in result["violations"][0]
