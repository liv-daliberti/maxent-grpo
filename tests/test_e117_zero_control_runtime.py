from dataclasses import fields
from types import SimpleNamespace

import torch

from oat_drgrpo.learner.init import build_semantic_shannon_tracker
from oat_drgrpo.semantic_shannon import (
    SemanticShannonSuccessConditionedSignedDiagnostics,
    add_semantic_shannon_separate_advantage,
    success_conditioned_semantic_metric_values,
)


PREFIX = "train/semantic_shannon_success_conditioned_verified_support"


def semantic_args(coefficient: float, *, allow_zero: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        semantic_shannon_coef=coefficient,
        semantic_shannon_allow_zero_coefficient_control=allow_zero,
        semantic_shannon_surprisal_clip=5.0,
        semantic_shannon_pseudocount=1.0,
        semantic_shannon_quality_gated_advantage=False,
        semantic_shannon_quality_gated_cap=0.05,
        semantic_shannon_success_conditioned_signed_advantage=False,
        semantic_shannon_success_conditioned_signed_cap=0.05,
        semantic_shannon_success_conditioned_group_centered_advantage=False,
        semantic_shannon_success_conditioned_verified_support_advantage=True,
    )


def exercise(coefficient: float):
    tracker = build_semantic_shannon_tracker(semantic_args(coefficient))
    assert tracker is not None
    (
        advantages,
        diagnostics,
    ) = tracker.score_success_conditioned_signed_advantages_and_update(
        prompt_token_ids=[[11, 12]] * 4,
        answer_keys=["a", "a", "a", "b"],
        task_rewards=[1.0] * 4,
        active_mask=[True] * 4,
        num_samples=4,
        verified_support_keys_by_group=[["a", "b"]],
    )
    metrics = success_conditioned_semantic_metric_values(PREFIX, diagnostics)
    return tracker, advantages, diagnostics, metrics


def test_zero_control_factory_is_explicit():
    assert build_semantic_shannon_tracker(semantic_args(0.0, allow_zero=False)) is None
    assert build_semantic_shannon_tracker(semantic_args(0.0)) is not None
    assert build_semantic_shannon_tracker(semantic_args(0.1)) is not None


def test_cpf_execute_one_estimator_and_emit_one_complete_namespace():
    arms = {"c": exercise(0.0), "p": exercise(0.0), "f": exercise(0.1)}
    expected_keys = {
        f"{PREFIX}_{field.name}"
        for field in fields(SemanticShannonSuccessConditionedSignedDiagnostics)
    }
    assert all(set(result[3]) == expected_keys for result in arms.values())

    for arm in ("c", "p"):
        tracker, advantages, diagnostics, metrics = arms[arm]
        assert advantages == [0.0] * 4
        assert all(value.hex() == "0x0.0p+0" for value in advantages)
        assert diagnostics.eligible_fraction == 1.0
        assert diagnostics.verified_support_at_least_two_eligible_fraction == 1.0
        assert diagnostics.history_rows_added == 4.0
        assert diagnostics.tracked_outcomes == 2.0
        assert metrics[PREFIX + "_open_set_coefficient_used"] == 0.0
        assert metrics[PREFIX + "_raw_eligible_advantage_rms"] == 0.0
        assert metrics[PREFIX + "_effective_advantage_rms"] == 0.0

        task_advantage = torch.tensor([0.25, -0.5, 1.0, -2.0], dtype=torch.float32)
        semantic_advantage = torch.tensor(advantages, dtype=torch.float32)
        combined = add_semantic_shannon_separate_advantage(
            task_advantage, semantic_advantage
        )
        assert torch.equal(combined.view(torch.int32), task_advantage.view(torch.int32))
        assert tracker.tracked_prompt_count == 1

    assert arms["c"][1] == arms["p"][1]
    assert arms["c"][3] == arms["p"][3]
    f_diagnostics = arms["f"][2]
    assert f_diagnostics.open_set_coefficient_used == 0.1
    assert f_diagnostics.raw_eligible_advantage_rms > 0.0
    assert f_diagnostics.effective_advantage_rms > 0.0

    states = []
    for tracker, _, _, _ in arms.values():
        state = tracker.state_dict()
        state.pop("coefficient")
        states.append(state)
    assert states[0] == states[1] == states[2]
