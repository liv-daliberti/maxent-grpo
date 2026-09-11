from pathlib import Path
import importlib.util

import pytest
import torch

from oat_drgrpo.canonical_replay import (
    canonical_replay_split_mass_balance_loss,
)
from oat_drgrpo.semantic_shannon import (
    open_set_success_semantic_signal,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_e56_auditor():
    path = ROOT / "ops/exp_scaling/audit_e56_sentinel.py"
    spec = importlib.util.spec_from_file_location("e56_sentinel_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e56_open_set_actuator_penalizes_common_and_rewards_unseen_success():
    common = open_set_success_semantic_signal(
        explicit_counts={"known": 20},
        sampled_key="known",
    )
    unseen = open_set_success_semantic_signal(
        explicit_counts={"known": 20},
        sampled_key="new-validator-positive",
    )

    assert common.centered_clipped_surprisal < 0
    assert unseen.centered_clipped_surprisal > 0
    assert common.normalized_predictive_entropy == pytest.approx(
        unseen.normalized_predictive_entropy
    )


def test_e56_split_replay_has_independent_mass_and_balance_geometry():
    scores = torch.tensor([-0.2, -1.5, -3.0, -0.7])
    result = canonical_replay_split_mass_balance_loss(scores, [1, 3])

    assert result.mass_score_gradients.sum().item() == pytest.approx(-1.0)
    assert result.balance_score_gradients.sum().item() == pytest.approx(
        0.0, abs=1e-7
    )
    assert result.balance_score_gradients[0].item() == 0.0
    assert result.actuator_groups == 2
    assert result.balance_eligible_groups == 1


def test_e56_design_smoke_and_sentinel_are_target_free_and_unprojected():
    protocol = (
        ROOT
        / "paper/preregistration/e56_open_set_split_controller_05b.md"
    ).read_text(encoding="utf-8")
    launcher = (
        ROOT
        / "ops/exp_scaling/launch_e56_open_set_split_smoke.sh"
    ).read_text(encoding="utf-8")
    auditor = (
        ROOT / "ops/exp_scaling/audit_e56_smokes.py"
    ).read_text(encoding="utf-8")
    sentinel_auditor = (
        ROOT / "ops/exp_scaling/audit_e56_sentinel.py"
    ).read_text(encoding="utf-8")
    parser = (
        ROOT / "ops/exp_scaling/parse_scaling_curve.py"
    ).read_text(encoding="utf-8")

    assert "FROZEN FOR ONE THREE-DOMAIN SENTINEL" in protocol
    assert "No pair uses evaluation behavior" in protocol
    assert "desired mode count" in protocol
    assert "with no lower or upper projection" in protocol
    assert "split_mass_balance_per_rollout" in launcher
    assert "OAT_ZERO_INCLUDE_OPEN_SET_SPLIT_CANONICAL_ARM=1" in launcher
    assert "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10" in launcher
    assert "gold_support_feedback" in launcher
    assert "config|graph|python|sentinel_config|sentinel" in launcher
    assert "E56 sentinel requires a clean terminal smoke audit" in launcher
    assert "OAT_ZERO_TRAIN_SEEDS=9010" in launcher
    assert "OAT_ZERO_SBATCH_HOLD=1" in launcher
    assert 'scontrol release "${job_ids[@]}"' in launcher
    assert "e56_open_set_split_canonical_05b_sentinel_v1" in launcher
    assert "smoke_audit_sha256" in launcher
    assert '"attempt_selection": "exact_manifest_job_id"' in launcher
    assert 'write_sentinel_identity "${job_ids[@]}"' in launcher
    assert "canonical_replay_mass_score_gradient_sum" in auditor
    assert "open_set_projection_active" in auditor
    assert 'frozenset({"graph_coloring"})' in auditor
    assert (
        "semantic_shannon_success_conditioned_signed_open_set_entropy_ema"
        in parser
    )
    assert "canonical_replay_mass_next_alpha" in parser
    assert "mass_raw_score_gradient_sum" in launcher
    assert "balance_raw_score_gradient_sum" in launcher
    assert "expected_multiplier = reference / ema if inverse else ema / reference" in (
        sentinel_auditor
    )
    assert "terminal run never rewarded a new valid mode" in sentinel_auditor
    assert "terminal run never penalized a common valid mode" in (
        sentinel_auditor
    )
    assert "canonical_replay_mass_controller_state" in sentinel_auditor
    assert "unexpected debug attempts" in sentinel_auditor
    assert "job_manifest_sha256" in sentinel_auditor


def test_e56_open_set_telemetry_is_emitted_by_the_active_signed_branch():
    learner = (
        ROOT / "src/oat_drgrpo/learner/grpo.py"
    ).read_text(encoding="utf-8")
    quality_block, signed_and_later = learner.split(
        "if quality_gated_diagnostics is not None:", 1
    )[1].split(
        "if success_conditioned_signed_diagnostics is not None:", 1
    )
    signed_block = signed_and_later.split(
        "if (\n                semantic_shannon_use_separate_advantage", 1
    )[0]

    assert '"open_set_coefficient_used"' not in quality_block
    assert '"open_set_coefficient_used"' in signed_block
    assert '"open_set_observations"' in signed_block
    assert '"open_set_projection_active"' in signed_block


def test_e56_safety_gate_is_pending_before_first_treatment_evaluation():
    auditor = _load_e56_auditor()
    control = {
        "safety_window": {
            "points": 1,
            "mean_no_eos_count": 0.0,
            "mean_response_length": 10.0,
        }
    }
    treatment = {
        "safety_window": {
            "points": 0,
            "mean_no_eos_count": None,
            "mean_response_length": None,
        }
    }

    assert auditor._safety_gate(control, treatment) == {"status": "pending"}


def test_e56_safety_gate_does_not_terminal_fail_a_startup_window():
    auditor = _load_e56_auditor()
    control = {
        "status": "running",
        "safety_window": {
            "points": 64,
            "mean_no_eos_count": 0.0,
            "mean_response_length": 4.0,
        },
    }
    treatment = {
        "status": "running",
        "safety_window": {
            "points": 13,
            "mean_no_eos_count": 1.23,
            "mean_response_length": 19.2,
        },
    }

    result = auditor._safety_gate(control, treatment)
    assert result["status"] == "pending"
    assert result["provisional_status"] == "fail"
    assert result["checks"]["no_eos"] is False


def test_e56_auditor_rejects_an_unbound_debug_attempt(tmp_path):
    auditor = _load_e56_auditor()
    run_stamp = "gce56_test_open_set_split_canonical_s9010"
    run_root = tmp_path / f"xdr_model_{run_stamp}"
    expected = run_root / "debug_job200"
    stale = run_root / "debug_job100"
    expected.mkdir(parents=True)
    stale.mkdir()
    record = '{"trainer/global_step": 1}\n'
    (expected / "train_metrics.jsonl").write_text(record, encoding="utf-8")
    (stale / "train_metrics.jsonl").write_text(record, encoding="utf-8")

    records, violations = auditor._load_bound_records(
        tmp_path,
        run_stamp=run_stamp,
        job_id=200,
    )

    assert len(records) == 1
    assert any("unexpected debug attempts" in item for item in violations)
