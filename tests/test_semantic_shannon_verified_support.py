from __future__ import annotations

import copy
import math
from typing import Iterator

import pytest

from oat_drgrpo.semantic_shannon import SemanticShannonTracker


PROMPT = [101, 202, 303]


def _tracker(*, coefficient: float = 0.1, clip: float = 5.0) -> SemanticShannonTracker:
    return SemanticShannonTracker(
        coefficient=coefficient,
        surprisal_clip=clip,
        pseudocount=1.0,
        success_conditioned_verified_support_advantage=True,
    )


def _score(
    tracker: SemanticShannonTracker,
    answer_keys: list[str | None],
    task_rewards: list[float],
) -> tuple[list[float], object]:
    return tracker.score_success_conditioned_signed_advantages_and_update(
        prompt_token_ids=[list(PROMPT) for _ in answer_keys],
        answer_keys=answer_keys,
        task_rewards=task_rewards,
        active_mask=[True] * len(answer_keys),
        num_samples=len(answer_keys),
    )


def test_all_wrong_and_first_singleton_are_safe_exact_noops() -> None:
    tracker = _tracker()
    before = copy.deepcopy(tracker.state_dict())

    wrong, wrong_diagnostics = _score(
        tracker,
        [None] * 4,
        [0.0] * 4,
    )
    assert wrong == [0.0] * 4
    assert tracker.state_dict() == before
    assert wrong_diagnostics.effective_advantage_rms == 0.0

    singleton, _ = _score(
        tracker,
        ["only", None, None, None],
        [1.0, 0.0, 0.0, 0.0],
    )
    assert singleton == pytest.approx([0.0] * 4, abs=1e-15)
    assert next(iter(tracker.state_dict()["counts"].values())) == {"only": 1}


def test_rare_singleton_uses_verified_history_instead_of_being_erased() -> None:
    tracker = _tracker()
    _score(tracker, ["common"] * 4, [1.0] * 4)

    advantages, diagnostics = _score(
        tracker,
        ["rare", None, None, None],
        [1.0, 0.0, 0.0, 0.0],
    )

    assert advantages[0] > 0.0
    assert advantages[1:] == [0.0, 0.0, 0.0]
    assert diagnostics.effective_advantage_rms > 0.0
    assert diagnostics.effective_advantage_positive_fraction == pytest.approx(0.25)


def test_known_mode_singletons_receive_predictor_centered_signed_pressure() -> None:
    seeded = _tracker()
    _score(
        seeded,
        ["common"] * 12 + ["rare"] * 4,
        [1.0] * 16,
    )
    state = copy.deepcopy(seeded.state_dict())

    common = _tracker()
    common.load_state_dict(copy.deepcopy(state))
    common_advantages, _ = _score(
        common,
        ["common"] + [None] * 15,
        [1.0] + [0.0] * 15,
    )

    rare = _tracker()
    rare.load_state_dict(copy.deepcopy(state))
    rare_advantages, _ = _score(
        rare,
        ["rare"] + [None] * 15,
        [1.0] + [0.0] * 15,
    )

    assert common_advantages[0] < 0.0
    assert rare_advantages[0] > 0.0
    assert common_advantages[1:] == [0.0] * 15
    assert rare_advantages[1:] == [0.0] * 15


def test_uniform_verified_history_is_stationary_for_known_modes() -> None:
    seeded = _tracker()
    _score(
        seeded,
        ["left"] * 8 + ["right"] * 8,
        [1.0] * 16,
    )
    state = copy.deepcopy(seeded.state_dict())

    for mode in ("left", "right"):
        tracker = _tracker()
        tracker.load_state_dict(copy.deepcopy(state))
        advantages, _ = _score(
            tracker,
            [mode] + [None] * 15,
            [1.0] + [0.0] * 15,
        )
        assert advantages == pytest.approx([0.0] * 16, abs=1e-15)


def test_v7_state_round_trips_and_rejects_v6_aliasing() -> None:
    tracker = _tracker()
    _score(
        tracker,
        ["common", "common", "rare", None],
        [1.0, 1.0, 1.0, 0.0],
    )
    state = copy.deepcopy(tracker.state_dict())

    assert state["schema"] == "semantic_shannon_tracker_v7_verified_support"
    assert state["verified_support_only"] is True
    assert state["structural_unseen_bucket"] is False

    restored = _tracker()
    restored.load_state_dict(state)
    assert restored.state_dict() == state

    legacy = SemanticShannonTracker(
        coefficient=0.1,
        success_conditioned_group_centered_advantage=True,
    )
    with pytest.raises(ValueError, match="invalid semantic Shannon tracker state"):
        legacy.load_state_dict(state)


def _count_vectors(total: int, width: int) -> Iterator[tuple[int, ...]]:
    if width == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for suffix in _count_vectors(total - first, width - 1):
            yield (first,) + suffix


def _multinomial_probability(
    counts: tuple[int, ...], probabilities: tuple[float, ...]
) -> float:
    total = sum(counts)
    coefficient = math.factorial(total)
    for count in counts:
        coefficient //= math.factorial(count)
    return float(coefficient) * math.prod(
        probability**count
        for probability, count in zip(probabilities, counts)
    )


def _seeded_state(*, verified_support: bool) -> dict[str, object]:
    tracker = SemanticShannonTracker(
        coefficient=1.0,
        surprisal_clip=20.0,
        pseudocount=1.0,
        success_conditioned_group_centered_advantage=not verified_support,
        success_conditioned_verified_support_advantage=verified_support,
    )
    _score(
        tracker,
        ["common"] * 8 + ["rare"] * 2 + [None] * 6,
        [1.0] * 10 + [0.0] * 6,
    )
    return copy.deepcopy(tracker.state_dict())


def _expected_update(
    probabilities: tuple[float, ...],
    *,
    verified_support: bool,
    group_size: int = 16,
) -> list[float]:
    state = _seeded_state(verified_support=verified_support)
    update = [0.0] * len(probabilities)
    for counts in _count_vectors(group_size, len(probabilities)):
        group_probability = _multinomial_probability(counts, probabilities)
        actions = [
            action
            for action, count in enumerate(counts)
            for _ in range(count)
        ]
        tracker = SemanticShannonTracker(
            coefficient=1.0,
            surprisal_clip=20.0,
            pseudocount=1.0,
            success_conditioned_group_centered_advantage=not verified_support,
            success_conditioned_verified_support_advantage=verified_support,
        )
        tracker.load_state_dict(copy.deepcopy(state))
        advantages, _ = tracker.score_success_conditioned_signed_advantages_and_update(
            prompt_token_ids=[list(PROMPT)] * group_size,
            answer_keys=[
                None if action == 0 else ("common" if action == 1 else "rare")
                for action in actions
            ],
            task_rewards=[0.0 if action == 0 else 1.0 for action in actions],
            active_mask=[True] * group_size,
            num_samples=group_size,
        )
        for action, advantage in zip(actions, advantages):
            for logit in range(len(probabilities)):
                update[logit] += (
                    group_probability
                    * advantage
                    * (float(action == logit) - probabilities[logit])
                )
    return update


def _conditional_entropy_gradient(
    probabilities: tuple[float, ...],
) -> list[float]:
    success_mass = sum(probabilities[1:])
    conditional = [value / success_mass for value in probabilities[1:]]
    entropy = -sum(value * math.log(value) for value in conditional)
    return [
        0.0,
        *[
            value * (-math.log(value) - entropy)
            for value in conditional
        ],
    ]


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right)) / (
        math.sqrt(sum(value * value for value in left))
        * math.sqrt(sum(value * value for value in right))
    )


def test_sparse_success_v7_preserves_direction_and_restores_first_order_dose() -> None:
    probabilities = (0.98, 0.016, 0.004)
    v6 = _expected_update(probabilities, verified_support=False)
    v7 = _expected_update(probabilities, verified_support=True)
    exact = _conditional_entropy_gradient(probabilities)

    assert _cosine(v7, exact) > 0.97
    assert abs(v7[0]) < 0.2 * math.sqrt(
        sum(value * value for value in v7)
    )
    v6_norm = math.sqrt(sum(value * value for value in v6))
    v7_norm = math.sqrt(sum(value * value for value in v7))
    assert v7_norm > 8.0 * v6_norm



def test_external_verified_membership_actuates_common_without_count_leakage() -> None:
    tracker = _tracker()
    _score(tracker, ["common"] * 4, [1.0] * 4)

    advantages, diagnostics = (
        tracker.score_success_conditioned_signed_advantages_and_update(
            prompt_token_ids=[list(PROMPT)] * 4,
            answer_keys=["common", None, None, None],
            task_rewards=[1.0, 0.0, 0.0, 0.0],
            active_mask=[True] * 4,
            num_samples=4,
            verified_support_keys_by_group=[("common", "proposal_rare")],
        )
    )

    assert advantages[0] < 0.0
    assert advantages[1:] == [0.0, 0.0, 0.0]
    assert diagnostics.verified_support_size_mean == pytest.approx(2.0)
    assert (
        diagnostics.verified_support_at_least_two_eligible_fraction
        == pytest.approx(1.0)
    )
    assert diagnostics.external_verified_support_size_mean == pytest.approx(2.0)
    assert (
        diagnostics.external_verified_support_nonempty_group_fraction
        == pytest.approx(1.0)
    )
    counts = next(iter(tracker.state_dict()["counts"].values()))
    assert counts == {"common": 5}
    assert "proposal_rare" not in counts


def test_external_verified_membership_is_rejected_outside_v7() -> None:
    legacy = SemanticShannonTracker(
        coefficient=0.1,
        success_conditioned_group_centered_advantage=True,
    )
    with pytest.raises(ValueError, match="external verified support"):
        legacy.score_success_conditioned_signed_advantages_and_update(
            prompt_token_ids=[list(PROMPT)] * 4,
            answer_keys=["common", None, None, None],
            task_rewards=[1.0, 0.0, 0.0, 0.0],
            active_mask=[True] * 4,
            num_samples=4,
            verified_support_keys_by_group=[("common", "rare")],
        )
