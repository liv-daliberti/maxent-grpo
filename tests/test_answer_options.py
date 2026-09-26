import math

import pytest

from oat_drgrpo.answer_options import (
    AnswerOptionMITracker,
    conditional_answer_repr,
    crossfit_option_answer_mi,
    empirical_option_answer_mi,
    format_answer_option_prompt,
    option_id_for_sample,
)


def test_qwen_answer_option_is_inserted_inside_system_message():
    prompt = (
        "<|im_start|>system\nReturn only an answer.<|im_end|>\n"
        "<|im_start|>user\nproblem<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    conditioned = format_answer_option_prompt(prompt, 2, 4)

    assert "Answer-option latent: z=2 of 4." in conditioned
    assert conditioned.index("Answer-option latent") < conditioned.index("<|im_end|>")
    assert conditioned.endswith("<|im_start|>assistant\n")


def test_option_ids_partition_a_group_contiguously():
    assert [option_id_for_sample(index, 4, 2) for index in range(8)] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]


def test_conditional_answer_repr_namespaces_identical_answers_by_prompt():
    first = conditional_answer_repr('{"problem": 1}', "graph_coloring:123")
    second = conditional_answer_repr('{"problem": 2}', "graph_coloring:123")

    assert first != second
    assert first.endswith("|graph_coloring:123")


def test_tracker_scores_before_updating_and_then_learns_binding():
    tracker = AnswerOptionMITracker(
        num_options=2,
        ema_decay=0.0,
        smoothing=1.0,
        bonus_clip=5.0,
    )
    rows = ["x|answer-a", "x|answer-b"]
    options = [0, 1]
    kwargs = {
        "answer_reprs": rows,
        "option_ids": options,
        "correct": [True, True],
        "loss_masks": [1.0, 1.0],
        "beta": 0.25,
        "correct_only": True,
    }

    first_bonus, first_diagnostics = tracker.update_and_score(**kwargs)
    second_bonus, second_diagnostics = tracker.update_and_score(**kwargs)

    assert first_bonus == pytest.approx([0.0, 0.0])
    assert first_diagnostics.lower_bound_nats == pytest.approx(0.0)
    assert second_bonus[0] > 0.0
    assert second_bonus[1] > 0.0
    assert second_diagnostics.lower_bound_nats > 0.0


def test_leave_one_out_tracker_rewards_repeated_binding_on_first_visit():
    tracker = AnswerOptionMITracker(
        num_options=2,
        ema_decay=0.9,
        smoothing=1.0,
        bonus_clip=5.0,
        leave_one_out=True,
    )
    bonuses, diagnostics = tracker.update_and_score(
        answer_reprs=["x|a", "x|a", "x|b", "x|b"],
        option_ids=[0, 0, 1, 1],
        correct=[True, True, True, True],
        loss_masks=[1.0] * 4,
        beta=0.1,
        correct_only=True,
    )

    assert all(value > 0.0 for value in bonuses)
    assert diagnostics.lower_bound_nats > 0.0
    assert diagnostics.leave_one_out_supported_fraction == pytest.approx(1.0)


def test_leave_one_out_tracker_cannot_reward_singletons_from_their_own_rows():
    tracker = AnswerOptionMITracker(
        num_options=2,
        ema_decay=0.9,
        smoothing=1.0,
        leave_one_out=True,
    )
    bonuses, diagnostics = tracker.update_and_score(
        answer_reprs=["x|a", "x|b"],
        option_ids=[0, 1],
        correct=[True, True],
        loss_masks=[1.0, 1.0],
        beta=0.1,
        correct_only=True,
    )

    assert bonuses == pytest.approx([0.0, 0.0])
    assert diagnostics.lower_bound_nats == pytest.approx(0.0)
    assert diagnostics.leave_one_out_supported_fraction == pytest.approx(0.0)


def test_tracker_state_round_trip_rejects_estimator_mismatch():
    history = AnswerOptionMITracker(num_options=2, leave_one_out=False)
    loo = AnswerOptionMITracker(num_options=2, leave_one_out=True)

    with pytest.raises(ValueError, match="leave_one_out"):
        loo.load_state_dict(history.state_dict())


def test_tracker_never_rewards_incorrect_rows_in_correct_only_mode():
    tracker = AnswerOptionMITracker(num_options=2, ema_decay=0.0)
    bonuses, diagnostics = tracker.update_and_score(
        answer_reprs=["x|a", "x|b"],
        option_ids=[0, 1],
        correct=[False, True],
        loss_masks=[1.0, 1.0],
        beta=0.5,
        correct_only=True,
    )

    assert bonuses[0] == 0.0
    assert diagnostics.eligible_fraction == pytest.approx(0.5)


def test_tracker_state_round_trip_preserves_delayed_discriminator():
    tracker = AnswerOptionMITracker(num_options=2, ema_decay=0.5)
    tracker.update_and_score(
        answer_reprs=["x|a", "x|b"],
        option_ids=[0, 1],
        correct=[True, True],
        loss_masks=[1.0, 1.0],
        beta=0.1,
        correct_only=True,
    )
    restored = AnswerOptionMITracker(num_options=2, ema_decay=0.5)
    restored.load_state_dict(tracker.state_dict())

    expected, _ = tracker.update_and_score(
        answer_reprs=["x|a", "x|b"],
        option_ids=[0, 1],
        correct=[True, True],
        loss_masks=[1.0, 1.0],
        beta=0.1,
        correct_only=True,
    )
    observed, _ = restored.update_and_score(
        answer_reprs=["x|a", "x|b"],
        option_ids=[0, 1],
        correct=[True, True],
        loss_masks=[1.0, 1.0],
        beta=0.1,
        correct_only=True,
    )

    assert observed == pytest.approx(expected)


def test_empirical_option_answer_mi_detects_perfect_semantic_binding():
    metrics = empirical_option_answer_mi(
        answer_reprs=["a", "a", "b", "b"],
        option_ids=[0, 0, 1, 1],
        correct=[True, True, True, True],
        num_options=2,
    )

    assert metrics["option_answer_mi_nats"] == pytest.approx(math.log(2.0))
    assert metrics["option_classifier_accuracy"] == pytest.approx(1.0)
    assert metrics["option_eligible_fraction"] == pytest.approx(1.0)
    assert metrics["option_correct_rate_range"] == pytest.approx(0.0)


def test_empirical_option_answer_mi_does_not_confuse_failure_with_binding():
    metrics = empirical_option_answer_mi(
        answer_reprs=["a", "a", None, None],
        option_ids=[0, 0, 1, 1],
        correct=[True, True, False, False],
        num_options=2,
    )

    assert metrics["option_answer_mi_nats"] == pytest.approx(0.0)
    assert metrics["option_correct_rate_range"] == pytest.approx(1.0)


def test_crossfit_option_mi_does_not_reward_singleton_memorization():
    metrics = crossfit_option_answer_mi(
        answer_reprs_by_draw=[["only-in-zero", "only-in-one"], ["new-a", "new-b"]],
        option_ids_by_draw=[[0, 1], [0, 1]],
        correct_by_draw=[[True, True], [True, True]],
        num_options=2,
    )

    assert [row["option_answer_mi_lower_bound_nats"] for row in metrics] == (
        pytest.approx([0.0, 0.0])
    )


def test_crossfit_option_mi_recovers_repeated_binding():
    metrics = crossfit_option_answer_mi(
        answer_reprs_by_draw=[["a", "b"], ["a", "b"], ["a", "b"]],
        option_ids_by_draw=[[0, 1], [0, 1], [0, 1]],
        correct_by_draw=[[True, True], [True, True], [True, True]],
        num_options=2,
    )

    assert all(row["option_answer_mi_lower_bound_nats"] > 0 for row in metrics)
    assert all(row["option_classifier_accuracy"] == 1.0 for row in metrics)
