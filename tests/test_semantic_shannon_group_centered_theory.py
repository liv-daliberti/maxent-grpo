from __future__ import annotations

import itertools
import math

import pytest

from oat_drgrpo.semantic_shannon import SemanticShannonTracker


def _expected_semantic_logit_update(
    probabilities: tuple[float, ...],
    *,
    successful_actions: frozenset[int],
    group_size: int = 4,
) -> list[float]:
    """Enumerate the repaired estimator under a finite categorical policy."""

    update = [0.0 for _ in probabilities]
    for group in itertools.product(range(len(probabilities)), repeat=group_size):
        group_probability = math.prod(probabilities[action] for action in group)
        tracker = SemanticShannonTracker(
            coefficient=1.0,
            surprisal_clip=20.0,
            pseudocount=1.0,
            success_conditioned_group_centered_advantage=True,
        )
        advantages, _ = tracker.score_success_conditioned_signed_advantages_and_update(
            prompt_token_ids=[[17, 23]] * group_size,
            answer_keys=[
                f"mode-{action}" if action in successful_actions else None
                for action in group
            ],
            task_rewards=[
                1.0 if action in successful_actions else 0.0
                for action in group
            ],
            active_mask=[True] * group_size,
            num_samples=group_size,
        )
        assert sum(advantages) == pytest.approx(0.0, abs=1e-14)
        for action, advantage in zip(group, advantages):
            for logit in range(len(probabilities)):
                score = float(action == logit) - probabilities[logit]
                update[logit] += group_probability * advantage * score
    return update


def _conditional_entropy_logit_gradient(
    probabilities: tuple[float, ...],
    *,
    successful_actions: frozenset[int],
) -> list[float]:
    success_mass = sum(probabilities[action] for action in successful_actions)
    conditional = {
        action: probabilities[action] / success_mass
        for action in successful_actions
    }
    entropy = -sum(value * math.log(value) for value in conditional.values())
    return [
        (
            conditional[action]
            * (-math.log(conditional[action]) - entropy)
            if action in successful_actions
            else 0.0
        )
        for action in range(len(probabilities))
    ]


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(
        sum(b * b for b in right)
    )
    return numerator / denominator


@pytest.mark.parametrize(
    ("probabilities", "successful_actions"),
    [
        ((0.78, 0.12, 0.10), frozenset({0, 1})),
        ((0.65, 0.18, 0.07, 0.10), frozenset({0, 1, 2})),
        ((0.48, 0.31, 0.16, 0.05), frozenset({0, 1, 2})),
    ],
)
def test_expected_repaired_update_tracks_conditional_entropy_gradient(
    probabilities: tuple[float, ...],
    successful_actions: frozenset[int],
) -> None:
    repaired = _expected_semantic_logit_update(
        probabilities,
        successful_actions=successful_actions,
    )
    exact = _conditional_entropy_logit_gradient(
        probabilities,
        successful_actions=successful_actions,
    )

    assert _cosine(repaired, exact) > 0.95
    for action in range(len(probabilities)):
        if action not in successful_actions:
            assert repaired[action] == pytest.approx(0.0, abs=1e-14)


def test_expected_repaired_update_is_zero_at_uniform_success_distribution() -> None:
    repaired = _expected_semantic_logit_update(
        (0.30, 0.30, 0.30, 0.10),
        successful_actions=frozenset({0, 1, 2}),
    )

    assert repaired == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1e-14)
