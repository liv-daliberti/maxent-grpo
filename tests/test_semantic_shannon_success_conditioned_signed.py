from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest
import torch

import oat_drgrpo.learner.grpo as grpo_module
from oat_drgrpo.args import ZeroMathArgs, validate_zero_math_args
from oat_drgrpo.learner.grpo import ZeroMathGrpoMixin
from oat_drgrpo.semantic_shannon import SemanticShannonTracker


PROMPT = [101, 202, 303]


def _tracker(*, cap: float = 0.05) -> SemanticShannonTracker:
    return SemanticShannonTracker(
        coefficient=0.1,
        surprisal_clip=5.0,
        pseudocount=1.0,
        success_conditioned_signed_advantage=True,
        success_conditioned_signed_cap=cap,
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


def test_all_wrong_group_is_exact_zero_and_state_noop():
    tracker = _tracker()
    before = copy.deepcopy(tracker.state_dict())

    advantages, diagnostics = _score(
        tracker,
        answer_keys=["17", "23", "91", "144"],
        task_rewards=[0.0] * 4,
    )

    assert advantages == [0.0] * 4
    assert tracker.state_dict() == before
    assert diagnostics.raw_eligible_advantage_rms == 0.0
    assert diagnostics.effective_advantage_rms == 0.0
    assert diagnostics.effective_advantage_zero_fraction == 1.0
    assert diagnostics.eligible_fraction == 0.0
    assert diagnostics.history_rows_added == 0.0
    assert diagnostics.history_groups_updated == 0.0
    assert diagnostics.history_groups_skipped == 1.0


def test_common_success_is_negative_and_rare_success_is_positive():
    tracker = _tracker()

    common_advantages, common_diagnostics = _score(
        tracker,
        answer_keys=["common"] * 4,
        task_rewards=[1.0] * 4,
    )
    assert all(value < 0.0 for value in common_advantages)
    assert common_diagnostics.effective_advantage_negative_fraction == 1.0

    advantages, diagnostics = _score(
        tracker,
        answer_keys=["common", "common", "common", "rare"],
        task_rewards=[1.0] * 4,
    )

    assert all(value < 0.0 for value in advantages[:3])
    assert advantages[3] > 0.0
    assert diagnostics.effective_advantage_min < 0.0
    assert diagnostics.effective_advantage_max > 0.0
    assert diagnostics.effective_advantage_negative_fraction == 0.75
    assert diagnostics.effective_advantage_positive_fraction == 0.25
    counts = next(iter(tracker.state_dict()["counts"].values()))
    assert counts == {"common": 7, "rare": 1}


def test_ineligible_rows_neither_change_support_nor_history():
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
        answer_keys=["rare", "wrong-a", None, "inactive-a"],
        task_rewards=[1.0, 0.0, 1.0, 1.0],
        active_mask=[1.0, 1.0, 1.0, 0.0],
    )
    right_advantages, right_diagnostics = _score(
        right,
        answer_keys=["rare", "wrong-b", None, "inactive-b"],
        task_rewards=[1.0, 0.0, 1.0, 1.0],
        active_mask=[1.0, 1.0, 1.0, 0.0],
    )

    assert left_advantages == pytest.approx(right_advantages)
    assert left_advantages[0] > 0.0
    assert left_advantages[1:] == [0.0, 0.0, 0.0]
    assert left.state_dict() == right.state_dict()
    assert left_diagnostics == right_diagnostics
    assert left_diagnostics.eligible_fraction == pytest.approx(0.25)
    assert left_diagnostics.history_rows_added == 1.0
    counts = next(iter(left.state_dict()["counts"].values()))
    assert counts == {"common": 4, "rare": 1}


def test_symmetric_cap_clamps_both_common_and_rare_successes():
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

    assert advantages[:3] == pytest.approx([-0.001] * 3)
    assert advantages[3] == pytest.approx(0.001)
    assert diagnostics.effective_advantage_min == pytest.approx(-0.001)
    assert diagnostics.effective_advantage_max == pytest.approx(0.001)
    assert diagnostics.negative_cap_fraction == pytest.approx(0.75)
    assert diagnostics.positive_cap_fraction == pytest.approx(0.25)


def test_signed_history_round_trip_and_disabled_schema_compatibility():
    tracker = _tracker(cap=0.05)
    _score(
        tracker,
        answer_keys=["correct", "wrong", None, "also-correct"],
        task_rewards=[1.0, 0.0, 1.0, 1.0],
    )
    state = copy.deepcopy(tracker.state_dict())
    assert (
        state["schema"]
        == "semantic_shannon_tracker_v3_success_conditioned_signed"
    )
    assert state["success_conditioned_signed_cap"] == pytest.approx(0.05)
    assert state["rows_scored"] == 2

    restored = _tracker(cap=0.05)
    restored.load_state_dict(state)
    kwargs = {
        "answer_keys": ["correct", "new-correct", "wrong", None],
        "task_rewards": [1.0, 1.0, 0.0, 1.0],
    }
    assert _score(restored, **kwargs) == _score(tracker, **kwargs)
    assert restored.state_dict() == tracker.state_dict()

    with pytest.raises(ValueError, match="success_conditioned_signed_cap"):
        _tracker(cap=0.04).load_state_dict(state)

    baseline = SemanticShannonTracker()
    explicit_disabled = SemanticShannonTracker(
        success_conditioned_signed_advantage=False,
        success_conditioned_signed_cap=0.001,
    )
    assert baseline.state_dict() == explicit_disabled.state_dict()
    assert baseline.state_dict()["schema"] == "semantic_shannon_tracker_v1"


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


class _SignedLearner(ZeroMathGrpoMixin):
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
            semantic_shannon_quality_gated_advantage=False,
            semantic_shannon_quality_gated_cap=0.05,
            semantic_shannon_success_conditioned_signed_advantage=True,
            semantic_shannon_success_conditioned_signed_cap=0.05,
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
        return [["common"] * group_size]

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


def test_learner_keeps_task_centering_unchanged_and_adds_signed_once(
    monkeypatch,
):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    learner = _SignedLearner()
    trajectory = {
        "input_ids": torch.zeros((4, 3), dtype=torch.long),
        "attention_mask": torch.ones((4, 3), dtype=torch.long),
        "rewards": [[1.0]] * 4,
        "prompt_ids_lens": [1] * 4,
        "loss_masks": [1.0] * 4,
        "references": [None] * 4,
    }
    expected_tracker = _tracker()
    expected, _ = (
        expected_tracker.score_success_conditioned_signed_advantages_and_update(
            prompt_token_ids=[[0]] * 4,
            answer_keys=["common"] * 4,
            task_rewards=[1.0] * 4,
            active_mask=[1.0] * 4,
            num_samples=4,
        )
    )

    infos = learner._grpo_learning_step_with_progress(trajectory)

    assert learner.captured is not None
    torch.testing.assert_close(
        learner.captured["final_rewards"], torch.ones((4, 1))
    )
    torch.testing.assert_close(
        learner.captured["advantages"],
        torch.tensor(expected).reshape(4, 1),
    )
    assert infos["semantic_shannon_augmented_reward_mean"].item() == 1.0
    assert (
        infos["semantic_shannon_reward_sent_to_centering_mean"].item() == 1.0
    )
    assert (
        infos[
            "semantic_shannon_success_conditioned_signed_advantage_active"
        ].item()
        == 1.0
    )
    assert (
        infos[
            "semantic_shannon_success_conditioned_signed_"
            "effective_advantage_negative_fraction"
        ].item()
        == 1.0
    )


def test_e44_xdr_weights_use_task_advantage_before_semantic_addition(
    monkeypatch,
):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    learner = _SignedLearner()
    learner.args.xdr_tau = 0.05
    learner.args.xdr_task_advantage_weights = True
    observed: dict[str, torch.Tensor] = {}

    def _capture_weights(advantages, token_counts, **kwargs):
        del token_counts, kwargs
        observed["advantages"] = advantages.detach().clone()
        return torch.ones(advantages.shape[0], dtype=torch.float32)

    monkeypatch.setattr(
        grpo_module,
        "compute_xdr_row_weights",
        _capture_weights,
    )
    trajectory = {
        "input_ids": torch.zeros((4, 3), dtype=torch.long),
        "attention_mask": torch.ones((4, 3), dtype=torch.long),
        "rewards": [[1.0]] * 4,
        "prompt_ids_lens": [1] * 4,
        "loss_masks": [1.0] * 4,
        "references": [None] * 4,
    }

    infos = learner._grpo_learning_step_with_progress(trajectory)

    # An all-correct group has zero ordinary Dr.GRPO advantage. The semantic
    # term is negative for the common success, but it must not enter xDr's
    # detached aggregation weights a second time.
    torch.testing.assert_close(
        observed["advantages"],
        torch.zeros((4, 1), dtype=torch.float32),
    )
    assert learner.captured is not None
    assert torch.all(learner.captured["advantages"] < 0)
    assert infos["xdr_task_advantage_weights_active"].item() == 1.0
    assert infos["xdr_weight_advantage_rms"].item() == 0.0


def test_e44_task_only_xdr_weighting_has_strict_configuration_gate():
    args = ZeroMathArgs(
        critic_type="drgrpo",
        num_samples=16,
        xdr_tau=0.05,
        xdr_task_advantage_weights=True,
        semantic_shannon_coef=0.10,
        semantic_shannon_separate_advantage=True,
        semantic_shannon_success_conditioned_signed_advantage=True,
    )
    assert validate_zero_math_args(args) is args

    args.semantic_shannon_success_conditioned_signed_advantage = False
    with pytest.raises(
        ValueError,
        match="requires the success-conditioned signed semantic advantage",
    ):
        validate_zero_math_args(args)
