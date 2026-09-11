from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "ops/exp_scaling/audit_e52_sentinel.py"
SPEC = importlib.util.spec_from_file_location("audit_e52_sentinel", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _inverse_record(step: int, entropy: float, *, hybrid: bool):
    reference = 2.0
    ema = entropy
    multiplier = reference / ema if step > 64 else 1.0
    record = {
        "trainer/global_step": step,
        "misc/prompt_consumed": step,
        "train/maxent_conditional_token_entropy": entropy,
        "train/maxent_inverse_observed_entropy": entropy,
        "train/maxent_inverse_entropy_ema": ema,
        "train/maxent_inverse_multiplier": multiplier,
        "train/maxent_alpha_used": 0.000075,
        "train/maxent_inverse_next_alpha": 0.000075 * multiplier,
        "train/maxent_inverse_observations": step,
        "train/maxent_inverse_projection_active": 0.0,
        "train/maxent_entropy_loss": -0.0001,
        "train/online_canonical_entropy_alpha_used": 0.1 if hybrid else 0.0,
        "train/policy_grad_norm": 0.01,
        "actor/no_eos_count": 0.0,
        "actor/response_tok_len": 24.0,
        "train/online_canonical_task_reward_mean": 0.25,
        "train/online_canonical_canonicalizable_correct_fraction": 0.5,
        "train/online_canonical_support_at_least_two_prompt_fraction": 0.2,
        "train/online_canonical_entropy_advantage_rms": 0.01,
        "train/online_canonical_novelty_advantage_rms": 0.02,
        "train/online_canonical_tracked_outcomes": 10.0,
    }
    if step >= 64:
        record["train/maxent_inverse_reference_entropy"] = reference
    return record


def test_runtime_audit_accepts_unbounded_inverse_arithmetic_and_fixed_bank():
    records = [
        _inverse_record(step, entropy=2.0 if step <= 64 else 0.5, hybrid=True)
        for step in range(1, 67)
    ]

    result = MODULE.audit_run(
        records,
        arm="maxent_inverse_canonical",
        prompt_pool_size=384,
    )

    assert result["violations"] == []
    assert result["latest"]["multiplier"] == 4.0
    assert result["latest"]["alpha_next"] == 0.0003
    assert result["latest"]["canonical_alpha"] == 0.1
    assert result["training_passes"] == 66 / (384 * 16)
    assert result["mechanism_window"]["online_canonical_task_reward_mean"] == 0.25
    assert result["mechanism_window"]["tracked_outcomes"] == 10.0


def test_runtime_audit_rejects_sensor_mismatch_projection_and_bank_drift():
    record = _inverse_record(65, entropy=0.5, hybrid=True)
    record["train/maxent_inverse_observed_entropy"] = 0.4
    record["train/maxent_inverse_projection_active"] = 1.0
    record["train/online_canonical_entropy_alpha_used"] = 0.2

    result = MODULE.audit_run(
        [record],
        arm="maxent_inverse_canonical",
        prompt_pool_size=384,
    )

    assert any("sensor mismatch" in item for item in result["violations"])
    assert any("projection" in item for item in result["violations"])
    assert any("canonical alpha" in item for item in result["violations"])


def test_behavioral_gate_requires_stable_last_eight_boundary_wins():
    def rows(offset: float):
        return [
            {
                "step": step,
                "distinct8": 1.0 + offset,
                "pass8": 0.8,
                "mean8": 0.5,
            }
            for step in range(1, 9)
        ]

    result = MODULE.behavioral_gate(
        {"status": "complete", "evaluations": rows(0.0)},
        {"status": "complete", "evaluations": rows(0.2)},
    )

    assert result["status"] == "pass"
    assert result["hybrid_wins"] == 8
    assert result["hybrid_excess_wins"] == 8
    assert result["hybrid_positive_multiplicity_boundaries"] == 8
    assert result["hybrid_self_retention_ratio"] == 1.0


def test_behavioral_gate_never_decides_from_an_early_eight_boundary_window():
    def rows(offset: float):
        return [
            {
                "step": step,
                "distinct8": 1.0 + offset,
                "pass8": 0.8,
                "mean8": 0.5,
            }
            for step in range(1, 9)
        ]

    result = MODULE.behavioral_gate(
        {"status": "running", "evaluations": rows(0.0)},
        {"status": "running", "evaluations": rows(-0.2)},
    )

    assert result["status"] == "pending"
    assert result["paired_boundaries"] == 8
    assert result["provisional_last_eight"]["status"] == "fail"


def test_behavioral_gate_rejects_a_single_mode_plateau():
    control_rows = [
        {
            "step": step,
            "distinct8": 0.4,
            "pass8": 0.4,
            "mean8": 0.3,
        }
        for step in range(1, 9)
    ]
    hybrid_rows = [
        {
            "step": step,
            "distinct8": 0.5,
            "pass8": 0.5,
            "mean8": 0.4,
        }
        for step in range(1, 9)
    ]

    result = MODULE.behavioral_gate(
        {"status": "complete", "evaluations": control_rows},
        {"status": "complete", "evaluations": hybrid_rows},
    )

    assert result["status"] == "fail"
    assert result["checks"]["higher_mean_distinct8"] is True
    assert result["checks"]["positive_multiplicity_at_least_six"] is False
    assert result["checks"]["higher_mean_distinct_excess_over_pass"] is False


def test_behavioral_gate_rejects_late_self_collapse_above_a_weak_control():
    control_rows = [
        {
            "step": step,
            "distinct8": 0.2,
            "pass8": 0.15,
            "mean8": 0.1,
        }
        for step in range(1, 17)
    ]
    hybrid_rows = [
        {
            "step": step,
            "distinct8": 1.0 if step <= 8 else 0.4,
            "pass8": 0.25 if step <= 8 else 0.3,
            "mean8": 0.2,
        }
        for step in range(1, 17)
    ]

    result = MODULE.behavioral_gate(
        {"status": "complete", "evaluations": control_rows},
        {"status": "complete", "evaluations": hybrid_rows},
    )

    assert result["status"] == "fail"
    assert result["checks"]["higher_mean_distinct8"] is True
    assert result["checks"]["wins_at_least_six"] is True
    assert result["checks"]["retains_half_of_own_best_rolling_eight"] is False


def test_prompt_horizon_alone_cannot_claim_terminal_completion():
    terminal_step = MODULE.MAX_PASSES * 384
    record = _inverse_record(
        terminal_step,
        entropy=2.0,
        hybrid=True,
    )
    record["misc/prompt_consumed"] = (
        MODULE.MAX_PASSES * 384 * MODULE.NUM_SAMPLES
    )

    pending = MODULE.audit_run(
        [record],
        arm="maxent_inverse_canonical",
        prompt_pool_size=384,
    )

    assert pending["status"] == "terminal_eval_pending"
    assert pending["terminal_evaluation_present"] is False

    record.update(
        {
            MODULE.DISTINCT_KEY: 1.0,
            MODULE.PASS8_KEY: 0.75,
            MODULE.MEAN8_KEY: 0.5,
        }
    )
    complete = MODULE.audit_run(
        [record],
        arm="maxent_inverse_canonical",
        prompt_pool_size=384,
    )

    assert complete["status"] == "complete"
    assert complete["terminal_evaluation_present"] is True


def test_only_positive_terminal_audit_persists_stage_a_approval(tmp_path):
    audit_out = tmp_path / "audit.json"
    approval_out = tmp_path / "approval.json"
    approval_out.write_text('{"status":"pass"}\n', encoding="utf-8")

    MODULE.write_audit_outputs(
        payload={"status": "in_progress"},
        audit_out=audit_out,
        approval_out=approval_out,
    )

    assert not approval_out.exists()
    assert MODULE.json.loads(audit_out.read_text())["status"] == "in_progress"

    MODULE.write_audit_outputs(
        payload={"status": "pass", "authorizes_stage_a": True},
        audit_out=audit_out,
        approval_out=approval_out,
    )

    persisted = MODULE.json.loads(approval_out.read_text())
    assert persisted == {"authorizes_stage_a": True, "status": "pass"}
