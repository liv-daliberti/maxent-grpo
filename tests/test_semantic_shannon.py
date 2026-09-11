from __future__ import annotations

import copy
import math

import pytest

from oat_drgrpo.learner.run import ZeroMathRunMixin
from oat_drgrpo.outcome_collision import INVALID_OUTCOME_KEY
from oat_drgrpo.semantic_shannon import (
    OpenSetSemanticInverseController,
    SemanticShannonTracker,
    open_set_success_semantic_signal,
)


PROMPT_A = [10, 20, 30]
PROMPT_B = [10, 20, 31]


def _prompts(prompt, count):
    return [list(prompt) for _ in range(count)]


def test_duplicate_group_matches_predictive_unseen_bucket_formula():
    tracker = SemanticShannonTracker(
        coefficient=0.1,
        surprisal_clip=5.0,
        pseudocount=1.0,
    )

    bonuses, diagnostics = tracker.score_and_update(
        prompt_token_ids=_prompts(PROMPT_A, 4),
        answer_keys=["a"] * 4,
        num_samples=4,
    )

    # Three same-answer peers plus alpha, divided by those three observations,
    # alpha for the explicit answer, and alpha for the unseen bucket.
    probability = 4.0 / 5.0
    surprisal = -math.log(probability)
    expected_bonus = 0.1 * (surprisal / 5.0 - 1.0)
    assert bonuses == pytest.approx([expected_bonus] * 4)
    assert diagnostics.predictive_probability_mean == pytest.approx(probability)
    assert diagnostics.surprisal_mean == pytest.approx(surprisal)
    assert diagnostics.normalized_surprisal_mean == pytest.approx(
        surprisal / 5.0
    )
    assert diagnostics.entropy_mean == pytest.approx(
        -(0.8 * math.log(0.8) + 0.2 * math.log(0.2))
    )
    assert diagnostics.normalization_error_max <= 1e-12
    assert diagnostics.unseen_fraction == pytest.approx(0.0)
    assert diagnostics.distinct_outcomes_mean == pytest.approx(1.0)
    assert diagnostics.distinct_fraction == pytest.approx(0.25)


def test_unique_group_scores_each_singleton_through_one_unseen_bucket():
    tracker = SemanticShannonTracker()

    bonuses, diagnostics = tracker.score_and_update(
        prompt_token_ids=_prompts(PROMPT_A, 4),
        answer_keys=["a", "b", "c", "d"],
        num_samples=4,
    )

    # Each row sees three leave-one-out outcomes plus one unseen bucket:
    # D = 3 peer observations + alpha * (3 explicit + 1 unseen) = 7.
    probability = 1.0 / 7.0
    surprisal = -math.log(probability)
    assert diagnostics.predictive_probability_mean == pytest.approx(probability)
    assert diagnostics.surprisal_mean == pytest.approx(surprisal)
    assert diagnostics.unseen_fraction == pytest.approx(1.0)
    assert diagnostics.distinct_outcomes_mean == pytest.approx(4.0)
    assert diagnostics.distinct_fraction == pytest.approx(1.0)
    assert all(-0.1 <= value <= 0.0 for value in bonuses)


def test_separate_advantage_uses_predictive_not_current_group_centering():
    tracker = SemanticShannonTracker(
        coefficient=0.1,
        surprisal_clip=5.0,
        pseudocount=1.0,
    )

    advantages, diagnostics, advantage_diagnostics = (
        tracker.score_separate_advantages_and_update(
            prompt_token_ids=_prompts(PROMPT_A, 4),
            answer_keys=["a"] * 4,
            num_samples=4,
        )
    )

    observed_surprisal = -math.log(4.0 / 5.0)
    unseen_surprisal = -math.log(1.0 / 5.0)
    predictive_baseline = (
        4.0 / 5.0 * observed_surprisal
        + 1.0 / 5.0 * unseen_surprisal
    )
    expected_advantage = 0.1 / 5.0 * (
        observed_surprisal - predictive_baseline
    )

    assert advantages == pytest.approx([expected_advantage] * 4)
    assert expected_advantage < 0.0
    # Empirically centering this collapsed current group would erase it.
    current_group_centered = [
        value - sum(advantages) / len(advantages) for value in advantages
    ]
    assert current_group_centered == pytest.approx([0.0] * 4)
    assert sum(advantages) / len(advantages) == pytest.approx(
        expected_advantage
    )
    assert diagnostics.normalization_error_max <= 1e-12
    assert advantage_diagnostics.predictive_baseline_mean == pytest.approx(
        predictive_baseline
    )
    assert (
        advantage_diagnostics.predictive_baseline_normalized_mean
        == pytest.approx(predictive_baseline / 5.0)
    )
    assert advantage_diagnostics.predictive_centering_error_max <= 1e-15
    assert advantage_diagnostics.advantage_scale == pytest.approx(0.02)
    assert advantage_diagnostics.advantage_mean == pytest.approx(
        expected_advantage
    )
    assert advantage_diagnostics.advantage_rms == pytest.approx(
        abs(expected_advantage)
    )
    assert advantage_diagnostics.advantage_negative_fraction == 1.0
    assert advantage_diagnostics.advantage_positive_fraction == 0.0


def test_separate_advantage_rewards_rare_and_penalizes_common_outcomes():
    tracker = SemanticShannonTracker(
        coefficient=0.1,
        surprisal_clip=5.0,
        pseudocount=1.0,
    )
    tracker.score_and_update(
        prompt_token_ids=_prompts(PROMPT_A, 4),
        answer_keys=["a"] * 4,
        num_samples=4,
    )

    advantages, diagnostics, advantage_diagnostics = (
        tracker.score_separate_advantages_and_update(
            prompt_token_ids=_prompts(PROMPT_A, 4),
            answer_keys=["a", "a", "a", "b"],
            num_samples=4,
        )
    )

    assert advantages[:3] == pytest.approx([advantages[0]] * 3)
    assert advantages[0] < 0.0
    assert advantages[3] > 0.0
    assert diagnostics.unseen_fraction == pytest.approx(0.25)
    assert advantage_diagnostics.advantage_negative_fraction == 0.75
    assert advantage_diagnostics.advantage_positive_fraction == 0.25
    assert advantage_diagnostics.advantage_zero_fraction == 0.0
    assert advantage_diagnostics.predictive_centering_error_max <= 1e-15


def test_history_and_peers_are_used_before_full_group_update():
    tracker = SemanticShannonTracker()
    tracker.score_and_update(
        prompt_token_ids=_prompts(PROMPT_A, 2),
        answer_keys=["a", "b"],
        num_samples=2,
    )

    bonuses, diagnostics = tracker.score_and_update(
        prompt_token_ids=_prompts(PROMPT_A, 2),
        answer_keys=["a", "c"],
        num_samples=2,
    )

    # For a: H={a,b}, peer adds c, D=2+1+alpha*(3+1)=7, numerator=2.
    # For c: H={a,b}, peer a is already explicit, D=2+1+alpha*(2+1)=6;
    # c itself is absent and therefore uses the unseen bucket numerator=1.
    assert diagnostics.predictive_probability_min == pytest.approx(1.0 / 6.0)
    assert diagnostics.predictive_probability_max == pytest.approx(2.0 / 7.0)
    assert bonuses[1] > bonuses[0]
    assert diagnostics.unseen_fraction == pytest.approx(0.5)
    state = tracker.state_dict()
    counts = next(iter(state["counts"].values()))
    assert counts == {"a": 2, "b": 1, "c": 1}
    assert state["groups_scored"] == 2
    assert state["rows_scored"] == 4


def test_prompt_histories_are_isolated_by_unpadded_token_ids():
    tracker = SemanticShannonTracker()
    tracker.score_and_update(
        prompt_token_ids=_prompts(PROMPT_A, 2),
        answer_keys=["a", "a"],
        num_samples=2,
    )

    _, diagnostics = tracker.score_and_update(
        prompt_token_ids=_prompts(PROMPT_B, 2),
        answer_keys=["a", "b"],
        num_samples=2,
    )

    assert diagnostics.history_total_mean == pytest.approx(0.0)
    assert diagnostics.tracked_prompts == pytest.approx(2.0)


def test_none_uses_one_shared_invalid_outcome_key():
    tracker = SemanticShannonTracker()

    _, diagnostics = tracker.score_and_update(
        prompt_token_ids=_prompts(PROMPT_A, 4),
        answer_keys=[None, None, "a", "b"],
        num_samples=4,
    )

    counts = next(iter(tracker.state_dict()["counts"].values()))
    assert counts[INVALID_OUTCOME_KEY] == 2
    assert len(counts) == 3
    assert diagnostics.invalid_fraction == pytest.approx(0.5)
    assert diagnostics.parseable_fraction == pytest.approx(0.5)


def test_bonus_is_clipped_to_closed_nonpositive_interval():
    tracker = SemanticShannonTracker(
        coefficient=0.1,
        surprisal_clip=0.01,
        pseudocount=1.0,
    )

    bonuses, diagnostics = tracker.score_and_update(
        prompt_token_ids=_prompts(PROMPT_A, 4),
        answer_keys=["a", "b", "c", "d"],
        num_samples=4,
    )

    assert bonuses == pytest.approx([0.0] * 4)
    assert diagnostics.clip_fraction == pytest.approx(1.0)
    assert diagnostics.normalized_surprisal_mean == pytest.approx(1.0)
    assert diagnostics.bonus_min >= -0.1
    assert diagnostics.bonus_max <= 0.0


def test_state_round_trip_preserves_exact_next_predictive_scores():
    tracker = SemanticShannonTracker()
    tracker.score_and_update(
        prompt_token_ids=_prompts(PROMPT_A, 4),
        answer_keys=["a", "a", "b", None],
        num_samples=4,
    )
    restored = SemanticShannonTracker()
    restored.load_state_dict(tracker.state_dict())

    kwargs = {
        "prompt_token_ids": _prompts(PROMPT_A, 4),
        "answer_keys": ["a", "b", "c", None],
        "num_samples": 4,
    }
    expected_bonuses, expected_diagnostics = tracker.score_and_update(**kwargs)
    observed_bonuses, observed_diagnostics = restored.score_and_update(**kwargs)

    assert observed_bonuses == expected_bonuses
    assert observed_diagnostics == expected_diagnostics
    assert restored.state_dict() == tracker.state_dict()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda state: state.update(coefficient=0.2), "coefficient"),
        (lambda state: state.update(surprisal_clip=4.0), "surprisal_clip"),
        (lambda state: state.update(pseudocount=0.5), "pseudocount"),
        (lambda state: state.update(schema="wrong"), "invalid"),
        (lambda state: state.update(counts=[]), "counts"),
        (
            lambda state: state.update(rows_scored=1, groups_scored=1),
            "rows_scored",
        ),
    ],
)
def test_state_restore_rejects_contract_or_shape_mismatch(mutation, message):
    tracker = SemanticShannonTracker()
    state = tracker.state_dict()
    mutation(state)

    with pytest.raises(ValueError, match=message):
        tracker.load_state_dict(state)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"coefficient": float("nan")}, "coefficient"),
        ({"coefficient": -0.1}, "coefficient"),
        ({"surprisal_clip": 0.0}, "surprisal_clip"),
        ({"pseudocount": float("inf")}, "pseudocount"),
    ],
)
def test_constructor_rejects_invalid_hyperparameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SemanticShannonTracker(**kwargs)


def test_incomplete_or_mixed_prompt_groups_fail_before_mutating_state():
    tracker = SemanticShannonTracker()

    with pytest.raises(ValueError, match="complete"):
        tracker.score_and_update(
            prompt_token_ids=_prompts(PROMPT_A, 3),
            answer_keys=["a", "b", "c"],
            num_samples=2,
        )
    with pytest.raises(ValueError, match="share one"):
        tracker.score_and_update(
            # The malformed group is second: validation must still be atomic.
            prompt_token_ids=[PROMPT_A, PROMPT_A, PROMPT_A, PROMPT_B],
            answer_keys=["a", "b", "c", "d"],
            num_samples=2,
        )

    assert tracker.state_dict()["counts"] == {}


def test_open_set_signal_pushes_down_common_mode_and_up_unseen_success():
    common = open_set_success_semantic_signal(
        explicit_counts={"known": 32},
        sampled_key="known",
    )
    novel = open_set_success_semantic_signal(
        explicit_counts={"known": 32},
        sampled_key="new-valid-mode",
    )

    assert common.centered_clipped_surprisal < 0
    assert novel.centered_clipped_surprisal > 0
    assert 0 < common.normalized_predictive_entropy < 1
    assert novel.normalized_predictive_entropy == pytest.approx(
        common.normalized_predictive_entropy
    )
    assert common.explicit_support_size == 1


def test_open_set_inverse_keeps_singleton_pressure_alive_without_projection():
    ratios = []
    for count in (1, 10, 100, 1_000, 10_000):
        signal = open_set_success_semantic_signal(
            explicit_counts={"known": count},
            sampled_key="known",
        )
        # With beta proportional to 1 / normalized entropy, this ratio is the
        # coefficient-free effective pressure. It remains finite and nonzero
        # even though the singleton entropy itself tends toward zero.
        ratios.append(
            abs(signal.centered_clipped_surprisal)
            / signal.normalized_predictive_entropy
        )

    assert min(ratios) > 0.04
    assert max(ratios) < 0.12


def test_open_set_controller_is_unprojected_and_checkpoint_exact():
    controller = OpenSetSemanticInverseController(
        base_coefficient=0.1,
        warmup_steps=2,
        ema_decay=0.0,
    )

    assert controller.observe(0.8)[
        "semantic_open_set_next_coefficient"
    ] == pytest.approx(0.1)
    warmup = controller.observe(0.6)
    assert warmup["semantic_open_set_reference_entropy"] == pytest.approx(0.7)
    collapsed = controller.observe(0.07)
    assert collapsed["semantic_open_set_next_coefficient"] == pytest.approx(1.0)
    assert collapsed["semantic_open_set_projection_active"] == 0.0

    state = controller.state_dict()
    restored = OpenSetSemanticInverseController(
        base_coefficient=0.1,
        warmup_steps=2,
        ema_decay=0.0,
    )
    restored.load_state_dict(state)
    assert restored.state_dict() == state


def _open_set_tracker(*, coefficient=0.1, warmup_steps=1, ema_decay=0.0):
    return SemanticShannonTracker(
        coefficient=coefficient,
        success_conditioned_signed_advantage=True,
        open_set_inverse_adaptation=True,
        open_set_warmup_steps=warmup_steps,
        open_set_ema_decay=ema_decay,
    )


def test_open_set_tracker_bootstraps_without_gold_support_and_then_separates_modes():
    tracker = _open_set_tracker()

    first, first_diagnostics = (
        tracker.score_success_conditioned_signed_advantages_and_update(
            prompt_token_ids=_prompts(PROMPT_A, 4),
            answer_keys=["known", None, None, None],
            task_rewards=[1.0, 0.0, 0.0, 0.0],
            active_mask=[1.0] * 4,
            num_samples=4,
        )
    )
    assert first == [0.0] * 4
    assert first_diagnostics.open_set_observation_skipped == 1.0
    assert first_diagnostics.open_set_observations == 0.0

    second, diagnostics = (
        tracker.score_success_conditioned_signed_advantages_and_update(
            prompt_token_ids=_prompts(PROMPT_A, 4),
            answer_keys=["known", "novel", None, None],
            task_rewards=[1.0, 1.0, 0.0, 0.0],
            active_mask=[1.0] * 4,
            num_samples=4,
        )
    )
    assert second[0] < 0.0
    assert second[1] > 0.0
    assert second[2:] == [0.0, 0.0]
    assert diagnostics.open_set_inverse_adaptation_active == 1.0
    assert diagnostics.open_set_observations == 1.0
    assert diagnostics.open_set_warmup_complete == 1.0
    assert diagnostics.open_set_projection_active == 0.0
    assert diagnostics.advantage_cap == 0.0


def test_open_set_tracker_does_not_cap_effective_semantic_advantage():
    tracker = _open_set_tracker(coefficient=10.0)
    tracker.score_success_conditioned_signed_advantages_and_update(
        prompt_token_ids=_prompts(PROMPT_A, 2),
        answer_keys=["known", None],
        task_rewards=[1.0, 0.0],
        active_mask=[1.0, 1.0],
        num_samples=2,
    )

    advantages, diagnostics = (
        tracker.score_success_conditioned_signed_advantages_and_update(
            prompt_token_ids=_prompts(PROMPT_A, 2),
            answer_keys=["known", None],
            task_rewards=[1.0, 0.0],
            active_mask=[1.0, 1.0],
            num_samples=2,
        )
    )
    assert advantages[0] < -0.05
    assert advantages[0] == pytest.approx(
        diagnostics.raw_eligible_advantage_mean
    )
    assert diagnostics.negative_cap_fraction == 0.0
    assert diagnostics.advantage_cap == 0.0


def test_open_set_tracker_checkpoint_preserves_controller_and_next_scores():
    tracker = _open_set_tracker(warmup_steps=2, ema_decay=0.5)
    for answer_keys in (["a", "a"], ["a", "b"]):
        tracker.score_success_conditioned_signed_advantages_and_update(
            prompt_token_ids=_prompts(PROMPT_A, 2),
            answer_keys=answer_keys,
            task_rewards=[1.0, 1.0],
            active_mask=[1.0, 1.0],
            num_samples=2,
        )
    state = tracker.state_dict()
    assert state["schema"] == "semantic_shannon_tracker_v4_open_set_inverse"

    restored = _open_set_tracker(warmup_steps=2, ema_decay=0.5)
    restored.load_state_dict(copy.deepcopy(state))
    kwargs = {
        "prompt_token_ids": _prompts(PROMPT_A, 2),
        "answer_keys": ["a", "c"],
        "task_rewards": [1.0, 1.0],
        "active_mask": [1.0, 1.0],
        "num_samples": 2,
    }
    expected = tracker.score_success_conditioned_signed_advantages_and_update(
        **kwargs
    )
    observed = restored.score_success_conditioned_signed_advantages_and_update(
        **kwargs
    )
    assert observed == expected
    assert restored.state_dict() == tracker.state_dict()


class _CheckpointHarness(ZeroMathRunMixin):
    pass


def _set_checkpoint_fields(learner):
    learner.global_step = 2
    learner.policy_sgd_step = 2.0
    learner.query_step = 2
    learner.prompt_consumed = 2
    learner.prompt_epoch = 0
    learner.steps = 2
    learner._prompt_batches_consumed_total = 2
    learner.update_interval = 1
    learner._xdr_tau_controller = None
    learner._maxent_alpha_controller = None
    learner._maxent_length_controller = None
    learner._diayn_mi_tracker = None
    learner._wandb_run_id = None
    learner._wandb_run_name = None


def test_checkpoint_client_state_round_trip_persists_semantic_counts():
    source = _CheckpointHarness()
    _set_checkpoint_fields(source)
    source._semantic_shannon_tracker = SemanticShannonTracker()
    source._semantic_shannon_tracker.score_and_update(
        prompt_token_ids=_prompts(PROMPT_A, 2),
        answer_keys=["a", "b"],
        num_samples=2,
    )

    checkpoint = source._checkpoint_client_state()
    assert checkpoint["semantic_shannon_tracker_state"] == (
        source._semantic_shannon_tracker.state_dict()
    )

    restored = _CheckpointHarness()
    _set_checkpoint_fields(restored)
    restored._semantic_shannon_tracker = SemanticShannonTracker()
    restored._restore_training_progress_state(checkpoint)

    assert restored._semantic_shannon_tracker.state_dict() == (
        source._semantic_shannon_tracker.state_dict()
    )


def test_resume_requires_tracker_state_presence_to_match_active_treatment():
    learner = _CheckpointHarness()
    _set_checkpoint_fields(learner)
    learner._semantic_shannon_tracker = SemanticShannonTracker()

    with pytest.raises(ValueError, match="cannot resume without"):
        learner._restore_training_progress_state({})

    learner._semantic_shannon_tracker = None
    state = SemanticShannonTracker().state_dict()
    with pytest.raises(ValueError, match="non-semantic-Shannon"):
        learner._restore_training_progress_state(
            {"semantic_shannon_tracker_state": copy.deepcopy(state)}
        )
