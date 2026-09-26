from __future__ import annotations

import copy
import math

import pytest

from oat_drgrpo.semantic_shannon import SemanticShannonTracker


PROMPT = [101, 202, 303]


def _tracker() -> SemanticShannonTracker:
    return SemanticShannonTracker(
        coefficient=0.1,
        surprisal_clip=5.0,
        pseudocount=1.0,
        success_conditioned_group_centered_advantage=True,
    )


def _score(
    tracker: SemanticShannonTracker,
    *,
    answer_keys,
    task_rewards,
    active_mask=None,
):
    row_count = len(answer_keys)
    return tracker.score_success_conditioned_signed_advantages_and_update(
        prompt_token_ids=[list(PROMPT) for _ in range(row_count)],
        answer_keys=answer_keys,
        task_rewards=task_rewards,
        active_mask=(
            [True] * row_count if active_mask is None else active_mask
        ),
        num_samples=row_count,
    )


def test_all_wrong_group_is_an_exact_state_noop():
    tracker = _tracker()
    before = copy.deepcopy(tracker.state_dict())

    advantages, diagnostics = _score(
        tracker,
        answer_keys=["wrong-a", "wrong-b", None, "wrong-c"],
        task_rewards=[0.0, 0.0, 1.0, 0.0],
    )

    assert advantages == [0.0] * 4
    assert tracker.state_dict() == before
    assert diagnostics.effective_advantage_rms == 0.0
    assert diagnostics.history_groups_skipped == 1.0


def test_all_same_success_group_is_exact_zero_not_uniformly_negative():
    tracker = _tracker()

    first, _ = _score(
        tracker,
        answer_keys=["common"] * 4,
        task_rewards=[1.0] * 4,
    )
    second, diagnostics = _score(
        tracker,
        answer_keys=["common"] * 4,
        task_rewards=[1.0] * 4,
    )

    assert first == pytest.approx([0.0] * 4, abs=1e-12)
    assert second == pytest.approx([0.0] * 4, abs=1e-12)
    assert diagnostics.effective_advantage_mean == pytest.approx(0.0, abs=1e-12)
    assert diagnostics.effective_advantage_zero_fraction == 1.0


def test_single_success_among_failures_is_exact_zero():
    tracker = _tracker()

    advantages, diagnostics = _score(
        tracker,
        answer_keys=["only", "wrong-a", "wrong-b", None],
        task_rewards=[1.0, 0.0, 0.0, 1.0],
    )

    assert advantages == pytest.approx([0.0] * 4, abs=1e-12)
    assert diagnostics.eligible_fraction == pytest.approx(0.25)


def test_rare_success_is_positive_common_successes_balance_it_exactly():
    tracker = _tracker()
    _score(
        tracker,
        answer_keys=["common"] * 4,
        task_rewards=[1.0] * 4,
    )

    advantages, diagnostics = _score(
        tracker,
        answer_keys=["common", "common", "common", "rare"],
        task_rewards=[1.0] * 4,
    )

    assert all(value < 0.0 for value in advantages[:3])
    assert advantages[3] > 0.0
    assert sum(advantages) == pytest.approx(0.0, abs=1e-12)
    assert diagnostics.effective_advantage_mean == pytest.approx(0.0, abs=1e-12)

    common_score = -math.log(7.0 / 10.0) / 5.0
    rare_score = -math.log(1.0 / 9.0) / 5.0
    sampled_baseline = (3.0 * common_score + rare_score) / 4.0
    expected = [
        0.1 * (common_score - sampled_baseline),
        0.1 * (common_score - sampled_baseline),
        0.1 * (common_score - sampled_baseline),
        0.1 * (rare_score - sampled_baseline),
    ]
    assert advantages == pytest.approx(expected, abs=1e-12)


def test_ineligible_rows_are_zero_and_eligible_rows_center_separately():
    tracker = _tracker()
    _score(
        tracker,
        answer_keys=["common"] * 4,
        task_rewards=[1.0] * 4,
    )

    advantages, _ = _score(
        tracker,
        answer_keys=["common", "rare", "wrong", None],
        task_rewards=[1.0, 1.0, 0.0, 1.0],
    )

    assert advantages[0] < 0.0
    assert advantages[1] > 0.0
    assert advantages[2:] == [0.0, 0.0]
    assert advantages[0] + advantages[1] == pytest.approx(0.0, abs=1e-12)


def test_group_centered_state_round_trips_under_a_distinct_schema():
    tracker = _tracker()
    _score(
        tracker,
        answer_keys=["common", "common", "rare", "wrong"],
        task_rewards=[1.0, 1.0, 1.0, 0.0],
    )
    state = copy.deepcopy(tracker.state_dict())

    assert state["schema"] == "semantic_shannon_tracker_v6_group_centered"
    assert state["success_conditioned_group_centered_advantage"] is True

    restored = _tracker()
    restored.load_state_dict(state)
    assert restored.state_dict() == state

    legacy = SemanticShannonTracker(
        coefficient=0.1,
        success_conditioned_signed_advantage=True,
    )
    with pytest.raises(ValueError, match="invalid semantic Shannon tracker state"):
        legacy.load_state_dict(state)


def test_multiple_prompt_groups_center_and_update_history_independently():
    tracker = _tracker()
    prompt_a = [11, 12]
    prompt_b = [21, 22]

    advantages, diagnostics = (
        tracker.score_success_conditioned_signed_advantages_and_update(
            prompt_token_ids=[prompt_a] * 4 + [prompt_b] * 4,
            answer_keys=["a", "a", "b", None, "x", "x", "x", "y"],
            task_rewards=[1.0] * 8,
            active_mask=[True] * 8,
            num_samples=4,
        )
    )

    assert sum(advantages[:4]) == pytest.approx(0.0, abs=1e-12)
    assert sum(advantages[4:]) == pytest.approx(0.0, abs=1e-12)
    assert advantages[2] > 0.0
    assert advantages[7] > 0.0
    assert advantages[3] == 0.0
    assert diagnostics.effective_advantage_mean == pytest.approx(0.0, abs=1e-12)
    assert diagnostics.history_groups_updated == 2.0
    assert diagnostics.history_rows_added == 7.0
    assert sorted(tracker.state_dict()["counts"].values(), key=str) == sorted(
        [{"a": 2, "b": 1}, {"x": 3, "y": 1}],
        key=str,
    )


def test_inactive_verified_row_is_zero_and_never_enters_history():
    tracker = _tracker()

    advantages, diagnostics = _score(
        tracker,
        answer_keys=["common", "common", "rare", "inactive"],
        task_rewards=[1.0] * 4,
        active_mask=[True, True, True, False],
    )

    assert advantages[3] == 0.0
    assert sum(advantages[:3]) == pytest.approx(0.0, abs=1e-12)
    assert diagnostics.eligible_fraction == pytest.approx(0.75)
    counts = next(iter(tracker.state_dict()["counts"].values()))
    assert counts == {"common": 2, "rare": 1}


def test_within_group_row_permutation_only_permutes_advantages():
    seeded = _tracker()
    _score(
        seeded,
        answer_keys=["common", "common", "common", "rare"],
        task_rewards=[1.0] * 4,
    )
    state = copy.deepcopy(seeded.state_dict())
    answers = ["common", "rare", "rare", "new"]
    permutation = [2, 0, 3, 1]

    ordered = _tracker()
    ordered.load_state_dict(copy.deepcopy(state))
    ordered_advantages, _ = _score(
        ordered,
        answer_keys=answers,
        task_rewards=[1.0] * 4,
    )

    permuted = _tracker()
    permuted.load_state_dict(copy.deepcopy(state))
    permuted_advantages, _ = _score(
        permuted,
        answer_keys=[answers[index] for index in permutation],
        task_rewards=[1.0] * 4,
    )
    restored_order = [
        permuted_advantages[permutation.index(index)] for index in range(4)
    ]

    assert restored_order == pytest.approx(ordered_advantages, abs=1e-12)
    assert permuted.state_dict() == ordered.state_dict()
