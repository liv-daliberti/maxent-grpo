from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest
import torch

from oat_drgrpo.learner.grpo import ZeroMathGrpoMixin
from oat_drgrpo.semantic_shannon import SemanticShannonTracker


PROMPT = [101, 202, 303]


def _prompts(count: int) -> list[list[int]]:
    return [list(PROMPT) for _ in range(count)]


def _tracker(*, cap: float = 0.05) -> SemanticShannonTracker:
    return SemanticShannonTracker(
        coefficient=0.1,
        surprisal_clip=5.0,
        pseudocount=1.0,
        quality_gated_advantage=True,
        quality_gated_cap=cap,
    )


def _score(
    tracker: SemanticShannonTracker,
    *,
    answer_keys,
    task_rewards,
    active_mask=None,
):
    row_count = len(answer_keys)
    return tracker.score_quality_gated_advantages_and_update(
        prompt_token_ids=_prompts(row_count),
        answer_keys=answer_keys,
        task_rewards=task_rewards,
        active_mask=(
            [True] * row_count if active_mask is None else active_mask
        ),
        num_samples=row_count,
    )


def test_all_wrong_distinct_math_is_exact_zero_and_does_not_update_history():
    tracker = _tracker()
    before = copy.deepcopy(tracker.state_dict())

    advantages, diagnostics = _score(
        tracker,
        answer_keys=["17", "23", "91", "144"],
        task_rewards=[0.0, 0.0, 0.0, 0.0],
    )

    assert advantages == [0.0, 0.0, 0.0, 0.0]
    assert tracker.state_dict() == before
    assert diagnostics.raw_all_row_advantage_rms == 0.0
    assert diagnostics.effective_advantage_rms == 0.0
    assert diagnostics.eligible_fraction == 0.0
    assert diagnostics.gated_fraction == 1.0
    assert diagnostics.history_rows_added == 0.0
    assert diagnostics.history_groups_updated == 0.0
    assert diagnostics.history_groups_skipped == 1.0
    assert diagnostics.tracked_prompts == 0.0


def test_common_correct_mode_is_not_penalized_and_only_updates_success_history():
    tracker = _tracker()

    advantages, diagnostics = _score(
        tracker,
        answer_keys=["42"] * 4,
        task_rewards=[1.0] * 4,
    )

    assert advantages == [0.0] * 4
    assert diagnostics.raw_all_row_advantage_max < 0.0
    assert diagnostics.positive_only_zeroed_fraction == 1.0
    assert diagnostics.effective_advantage_zero_fraction == 1.0
    state = tracker.state_dict()
    assert next(iter(state["counts"].values())) == {"42": 4}
    assert state["groups_scored"] == 1
    assert state["rows_scored"] == 4


def test_rare_correct_mode_receives_positive_novelty_but_common_mode_does_not():
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

    assert advantages[:3] == [0.0] * 3
    assert 0.0 < advantages[3] <= 0.05
    assert diagnostics.raw_all_row_advantage_min < 0.0
    assert diagnostics.raw_all_row_advantage_max > 0.0
    assert diagnostics.effective_advantage_positive_fraction == 0.25
    assert diagnostics.positive_only_zeroed_fraction == 0.75
    counts = next(iter(tracker.state_dict()["counts"].values()))
    assert counts == {"common": 7, "rare": 1}


def test_mixed_group_excludes_wrong_and_unparseable_rows_from_support_and_history():
    tracker = _tracker()

    advantages, diagnostics = _score(
        tracker,
        answer_keys=["correct-a", "wrong-random", "correct-b", None],
        task_rewards=[1.0, 0.0, 1.0, 1.0],
    )

    assert advantages[1] == 0.0
    assert advantages[3] == 0.0
    assert advantages[0] > 0.0
    assert advantages[2] > 0.0
    assert diagnostics.eligible_fraction == pytest.approx(0.5)
    assert diagnostics.reward_positive_fraction == pytest.approx(0.75)
    assert diagnostics.parseable_fraction == pytest.approx(0.75)
    assert diagnostics.history_rows_added == 2.0
    counts = next(iter(tracker.state_dict()["counts"].values()))
    assert counts == {"correct-a": 1, "correct-b": 1}


def test_explicit_cap_clamps_a_rare_success():
    tracker = _tracker(cap=0.001)
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

    assert advantages[3] == pytest.approx(0.001)
    assert max(advantages) == pytest.approx(0.001)
    assert (
        diagnostics.raw_all_row_advantage_max
        > diagnostics.advantage_cap
    )
    assert diagnostics.effective_advantage_max == pytest.approx(0.001)
    assert diagnostics.cap_fraction == pytest.approx(0.25)


def test_inactive_rows_cannot_change_effective_advantages_or_history():
    left = _tracker()
    right = _tracker()
    for tracker in (left, right):
        _score(
            tracker,
            answer_keys=["common"] * 4,
            task_rewards=[1.0] * 4,
        )

    left_advantages, left_diagnostics = _score(
        left,
        answer_keys=["rare", "common", "junk-a", None],
        task_rewards=[1.0, 1.0, 1.0, 1.0],
        active_mask=[1.0, 0.0, 0.0, 0.0],
    )
    right_advantages, right_diagnostics = _score(
        right,
        answer_keys=["rare", "junk-b", "junk-c", "junk-d"],
        task_rewards=[1.0, 1.0, 1.0, 1.0],
        active_mask=[1.0, 0.0, 0.0, 0.0],
    )

    assert left_advantages == pytest.approx(right_advantages)
    assert left_advantages[0] > 0.0
    assert left_advantages[1:] == [0.0, 0.0, 0.0]
    assert left.state_dict() == right.state_dict()
    assert left_diagnostics.active_fraction == pytest.approx(0.25)
    assert right_diagnostics.eligible_fraction == pytest.approx(0.25)
    counts = next(iter(left.state_dict()["counts"].values()))
    assert counts == {"common": 4, "rare": 1}


def test_quality_gated_history_round_trip_and_cap_mismatch_rejection():
    tracker = _tracker(cap=0.05)
    _score(
        tracker,
        answer_keys=["correct", "wrong", None, "also-correct"],
        task_rewards=[1.0, 0.0, 1.0, 1.0],
    )
    state = copy.deepcopy(tracker.state_dict())
    assert state["schema"] == "semantic_shannon_tracker_v2_quality_gated"
    assert state["quality_gated_cap"] == pytest.approx(0.05)
    assert state["rows_scored"] == 2

    restored = _tracker(cap=0.05)
    restored.load_state_dict(state)
    kwargs = {
        "answer_keys": ["correct", "new-correct", "wrong", None],
        "task_rewards": [1.0, 1.0, 0.0, 1.0],
    }
    expected = _score(tracker, **kwargs)
    observed = _score(restored, **kwargs)
    assert observed == expected
    assert restored.state_dict() == tracker.state_dict()

    with pytest.raises(ValueError, match="quality_gated_cap"):
        _tracker(cap=0.04).load_state_dict(state)
    with pytest.raises(ValueError, match="invalid"):
        SemanticShannonTracker().load_state_dict(state)


def test_e41_path_and_v1_resume_schema_are_exact_when_quality_gate_is_disabled():
    baseline = SemanticShannonTracker(
        coefficient=0.1,
        surprisal_clip=5.0,
        pseudocount=1.0,
    )
    explicit_disabled = SemanticShannonTracker(
        coefficient=0.1,
        surprisal_clip=5.0,
        pseudocount=1.0,
        quality_gated_advantage=False,
        quality_gated_cap=0.001,
    )
    kwargs = {
        "prompt_token_ids": _prompts(4),
        "answer_keys": ["same"] * 4,
        "num_samples": 4,
    }

    baseline_result = baseline.score_separate_advantages_and_update(**kwargs)
    disabled_result = explicit_disabled.score_separate_advantages_and_update(
        **kwargs
    )

    assert disabled_result == baseline_result
    assert explicit_disabled.state_dict() == baseline.state_dict()
    assert baseline.state_dict()["schema"] == "semantic_shannon_tracker_v1"
    assert "quality_gated_cap" not in baseline.state_dict()


class _Policy:
    training = True

    def __call__(self, input_ids, *, attention_mask):
        del attention_mask
        return {
            "logits": torch.zeros(
                (*input_ids.shape, 8),
                dtype=torch.float32,
                device=input_ids.device,
            )
        }


class _QualityGateLearner(ZeroMathGrpoMixin):
    def __init__(self):
        self.args = SimpleNamespace(
            canonical_action_task="none",
            canonical_graph_actions=False,
            canonical_graph_action_count=3,
            critic_type="drgrpo",
            num_samples=4,
            reward_scale=1.0,
            prompt_template="qwen_boxed",
            outcome_collision_coef=0.0,
            outcome_collision_outside_centering=False,
            semantic_shannon_coef=0.1,
            semantic_shannon_surprisal_clip=5.0,
            semantic_shannon_pseudocount=1.0,
            semantic_shannon_separate_advantage=True,
            semantic_shannon_quality_gated_advantage=True,
            semantic_shannon_quality_gated_cap=0.05,
            temperature=1.0,
            train_batch_size_per_device=4,
            xdr_tau=math.inf,
            seed_entropy_alpha=0.0,
            generate_max_length=8,
        )
        self.model = _Policy()
        self.ref_model = None
        self._semantic_shannon_tracker = _tracker()
        self.captured = None

    def get_completion_mask(self, att_mask, prompt_id_lens):
        del prompt_id_lens
        mask = torch.zeros_like(att_mask, dtype=torch.bool)
        mask[:, 1] = True
        return mask

    def _seed_answer_keys_grouped(
        self,
        input_ids,
        response_masks,
        group_size,
        references_grouped=None,
    ):
        del input_ids, response_masks, references_grouped
        return [["17", "23", "91", "144"][:group_size]]

    def _resolve_scoring_vocab_upper_bound(self, model):
        del model
        return 8

    def _sanitize_scoring_token_ids(self, input_ids, **kwargs):
        del kwargs
        return input_ids

    def _mask_invalid_scoring_logit_columns(self, logits, **kwargs):
        del kwargs
        return logits

    def _policy_logps_and_optional_entropy(
        self,
        logits,
        input_ids,
        response_masks,
        *,
        need_entropy,
    ):
        del logits, input_ids, need_entropy
        return torch.zeros_like(response_masks, dtype=torch.float32), None

    def _baseline_update_with_precomputed_advantages(self, **kwargs):
        self.captured = kwargs
        return kwargs["extra_infos"]


def test_learner_routes_task_reward_and_active_mask_through_quality_gate(
    monkeypatch,
):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    learner = _QualityGateLearner()
    trajectory = {
        "input_ids": torch.zeros((4, 3), dtype=torch.long),
        "attention_mask": torch.ones((4, 3), dtype=torch.long),
        "rewards": [[0.0], [0.0], [0.0], [1.0]],
        "prompt_ids_lens": [1] * 4,
        "loss_masks": [1.0, 1.0, 1.0, 0.0],
        "references": [None] * 4,
    }

    infos = learner._grpo_learning_step_with_progress(trajectory)

    assert learner.captured is not None
    torch.testing.assert_close(
        learner.captured["final_rewards"],
        torch.tensor([[0.0], [0.0], [0.0], [1.0]]),
    )
    assert learner.captured["advantages"] == pytest.approx(
        torch.tensor([[-0.25], [-0.25], [-0.25], [0.75]])
    )
    assert (
        infos["semantic_shannon_quality_gated_advantage_active"].item()
        == 1.0
    )
    assert infos["semantic_shannon_quality_gated_eligible_fraction"].item() == 0.0
    assert infos["semantic_shannon_quality_gated_active_fraction"].item() == 0.75
    assert (
        infos["semantic_shannon_quality_gated_effective_advantage_rms"].item()
        == 0.0
    )
    assert infos["semantic_shannon_quality_gated_history_rows_added"].item() == 0.0
    assert learner._semantic_shannon_tracker.state_dict()["counts"] == {}
