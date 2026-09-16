from __future__ import annotations

import math
from typing import Iterator

import pytest

from oat_drgrpo.semantic_shannon import SemanticShannonTracker


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


def _expected_group_centered_update(
    probabilities: tuple[float, ...],
    *,
    successful_actions: frozenset[int],
    group_size: int,
) -> list[float]:
    update = [0.0 for _ in probabilities]
    total_probability = 0.0
    for counts in _count_vectors(group_size, len(probabilities)):
        group_probability = _multinomial_probability(counts, probabilities)
        total_probability += group_probability
        actions = [
            action
            for action, count in enumerate(counts)
            for _ in range(count)
        ]
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
                for action in actions
            ],
            task_rewards=[
                1.0 if action in successful_actions else 0.0
                for action in actions
            ],
            active_mask=[True] * group_size,
            num_samples=group_size,
        )
        assert sum(advantages) == pytest.approx(0.0, abs=1e-14)
        for action, advantage in zip(actions, advantages):
            for logit in range(len(probabilities)):
                update[logit] += (
                    group_probability
                    * advantage
                    * (float(action == logit) - probabilities[logit])
                )
    assert total_probability == pytest.approx(1.0, abs=1e-12)
    return update


def _conditional_entropy_gradient(
    probabilities: tuple[float, ...], successful_actions: frozenset[int]
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
    return sum(a * b for a, b in zip(left, right)) / (
        math.sqrt(sum(value * value for value in left))
        * math.sqrt(sum(value * value for value in right))
    )


@pytest.mark.parametrize(
    ("probabilities", "successful_actions"),
    [
        ((0.90, 0.08, 0.02), frozenset({1, 2})),
        ((0.88, 0.07, 0.035, 0.015), frozenset({1, 2, 3})),
    ],
)
def test_exact_group16_sparse_success_update_tracks_conditional_entropy(
    probabilities: tuple[float, ...], successful_actions: frozenset[int]
) -> None:
    repaired = _expected_group_centered_update(
        probabilities,
        successful_actions=successful_actions,
        group_size=16,
    )
    exact = _conditional_entropy_gradient(probabilities, successful_actions)

    assert _cosine(repaired, exact) > 0.95
    assert math.sqrt(sum(value * value for value in repaired)) > 1e-6
    for action in range(len(probabilities)):
        if action not in successful_actions:
            assert repaired[action] == pytest.approx(0.0, abs=1e-13)

