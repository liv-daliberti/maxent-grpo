from pathlib import Path

import pytest
import torch

from oat_drgrpo.canonical_replay import (
    canonical_replay_uniform_loss,
    canonical_replay_uniform_verified_likelihood_loss,
)


ROOT = Path(__file__).resolve().parents[1]


def test_e54_actuator_adds_common_mass_without_changing_e53_balance():
    scores = torch.tensor([-0.5, -2.0, -3.0], requires_grad=True)
    balance = canonical_replay_uniform_loss(scores, [3])
    likelihood = canonical_replay_uniform_verified_likelihood_loss(
        scores,
        [3],
    )

    assert balance.cross_entropy_excess.item() == pytest.approx(
        balance.loss.item()
    )
    assert balance.score_gradients.sum().item() == pytest.approx(0.0, abs=1e-7)
    assert likelihood.cross_entropy_excess.item() == pytest.approx(
        balance.loss.item()
    )
    assert likelihood.score_gradients.sum().item() == pytest.approx(-1.0)
    assert torch.allclose(
        likelihood.score_gradients,
        torch.full((3,), -1.0 / 3.0),
    )


def test_e54_protocol_forbids_gold_targets_and_coefficient_retuning():
    protocol = (
        ROOT
        / "paper/preregistration/e54_uniform_verified_likelihood_05b.md"
    ).read_text(encoding="utf-8")

    assert "FROZEN BEFORE E54 ENGINEERING SMOKE OR SENTINEL SUBMISSION" in protocol
    assert "score-gradient sum of `-1`" in protocol
    assert "There is no lower or upper projection" in protocol
    assert "gold support count" in protocol
    assert "coefficient chosen from E53" in protocol
    assert "inherits `0.10`" in protocol


def test_e54_launcher_is_treatment_only_and_identity_bound():
    launcher = (
        ROOT
        / "ops/exp_scaling/launch_e54_uniform_verified_likelihood_05b.sh"
    ).read_text(encoding="utf-8")

    assert "OAT_ZERO_ONLY_ARMS=maxent_inverse_canonical_replay" in launcher
    assert "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=verified_likelihood" in launcher
    assert "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.10" in launcher
    assert "OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY=16" in launcher
    assert "OAT_ZERO_MAX_PROMPT_EPOCHS=50" in launcher
    assert "OAT_ZERO_SBATCH_HOLD=1" in launcher
    assert "e53_control_identity_sha256" in launcher
    assert "scontrol release" in launcher


def test_e54_auditor_checks_exact_common_mass_gradient_and_frozen_control():
    auditor = (
        ROOT / "ops/exp_scaling/audit_e54_sentinel.py"
    ).read_text(encoding="utf-8")

    assert '"train/canonical_replay_actuator_loss"' in auditor
    assert '"train/canonical_replay_score_gradient_sum"' in auditor
    assert "common-mass actuator is not exact" in auditor
    assert "e53_control_identity_sha256" in auditor
    assert "BASE.behavioral_gate(control, treatment)" in auditor
