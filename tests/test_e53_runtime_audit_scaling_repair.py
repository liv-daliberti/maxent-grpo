from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
AUDITOR = ROOT / "ops/exp_scaling/audit_e53_sentinel_v2.py"
BASE_AUDITOR = ROOT / "ops/exp_scaling/audit_e53_sentinel.py"
IDENTITY = ROOT / "var/artifacts/e53_verified_replay_05b_sentinel_identity.json"
REPAIR = (
    ROOT
    / "paper/preregistration/e53_runtime_audit_scaling_repair_20260726.md"
)
SPEC = importlib.util.spec_from_file_location("audit_e53_sentinel_v2_test", AUDITOR)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _record(width: float) -> dict[str, float]:
    return {
        "train/canonical_replay_available_groups": 1.0,
        "train/canonical_replay_available_modes": 2.0,
        "train/canonical_replay_capacity": 16.0,
        "train/canonical_replay_gold_support_feedback": 0.0,
        "train/canonical_replay_alpha_projection_active": 0.0,
        "train/canonical_replay_observation_skipped": 0.0,
        "train/canonical_replay_balance_loss": 0.2,
        "train/canonical_replay_weighted_loss": 0.01875,
        "train/canonical_replay_backward_scale": width,
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
        "train/canonical_replay_next_alpha": 0.1,
        "train/canonical_replay_observations": 1.0,
        "train/canonical_replay_projection_active": 0.0,
    }


def test_runtime_repair_accepts_logged_accumulation_width_16():
    _reference, violations, _latest, active = MODULE._check_replay_controller(
        _record(16.0),
        step=1,
        reference=None,
    )
    assert active is True
    assert violations == []


def test_runtime_repair_rejects_fully_weighted_scalar_in_width_field():
    _reference, violations, _latest, active = MODULE._check_replay_controller(
        _record(1.5),
        step=1,
        reference=None,
    )
    assert active is True
    assert any("accumulation width" in item for item in violations)


def test_bound_base_auditor_remains_exactly_the_identity_version():
    identity = json.loads(IDENTITY.read_text(encoding="utf-8"))
    observed = hashlib.sha256(BASE_AUDITOR.read_bytes()).hexdigest()
    assert observed == identity["auditor_sha256"]
    repair = REPAIR.read_text(encoding="utf-8")
    assert "repairs only the fail-closed" in repair
    assert "It does not\nchange source, execution snapshots, jobs, coefficients" in repair
