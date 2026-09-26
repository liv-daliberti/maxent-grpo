from __future__ import annotations

import json
from itertools import product
from types import SimpleNamespace

from datasets import load_from_disk
import pytest
import torch

from oat_drgrpo.canonical_actions import (
    canonical_action_code_from_verified_response,
    canonical_action_code_token_ids,
    decode_canonical_action_response,
    resolve_canonical_action_space,
)
from oat_drgrpo.learner import run as run_module
from oat_drgrpo.learner.grpo import _task_bound_canonicalization_surfaces
from oat_drgrpo.learner.run import ZeroMathRunMixin
from oat_drgrpo.math_grader import boxed_reward_fn, validated_modebench_outcome_key
from oat_drgrpo.online_canonical_bank import OnlineCanonicalBank
from oat_drgrpo.pantry_support_action import (
    PANTRY_SUPPORT_MASK_INVALID,
    decode_pantry_support_mask,
    pantry_support_mask_from_allocation,
    pantry_support_from_mask,
)
from oat_drgrpo.pantry_plan import PantryPlanError
from oat_drgrpo.templates import (
    apply_qwen_pantry_support_mask_template,
    validate_qwen_pantry_support_mask_materialization,
)


class _BinaryTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [{"0": 10, "1": 11}[text]]

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join({10: "0", 11: "1"}[int(token_id)] for token_id in token_ids)


class _UniformBinaryModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids, *, attention_mask):
        torch.testing.assert_close(attention_mask, torch.ones_like(input_ids))
        logits = torch.zeros(
            (*input_ids.shape, 16), dtype=torch.float32, device=input_ids.device
        )
        return {"logits": logits + self.anchor * 0.0}


class _RunHarness(ZeroMathRunMixin):
    pass


def _first_dev_row():
    return load_from_disk("var/data/pantry_plan_modebench_v2/dev")[
        "multi_answer"
    ][0]


def test_pantry_support_mask_template_is_exact_and_stale_safe():
    question = str(_first_dev_row()["problem"])
    prompt = apply_qwen_pantry_support_mask_template(question)

    assert "Return only six binary digits" in prompt
    assert "corresponding Pantry row" in prompt
    assert "ingredient_id=grams" not in prompt
    assert "\\boxed{}" not in prompt
    validate_qwen_pantry_support_mask_materialization([question], [prompt])


def test_pantry_support_mask_action_space_is_fixed_six_binary_decisions():
    space = resolve_canonical_action_space(_BinaryTokenizer(), "pantry_support_mask")

    assert space.horizon == 6
    assert space.sequence_count == 64
    assert space.action_strings_by_position == (("0", "1"),) * 6
    assert space.token_ids_by_position == ((10, 11),) * 6


def test_pantry_learner_sampler_covers_all_six_steps_and_emits_task_identity(
    monkeypatch,
):
    monkeypatch.setattr(run_module.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        run_module,
        "boxed_reward_fn",
        lambda response, reference, *, fast: ({"formatted": True}, 1.0),
    )
    harness = _RunHarness()
    harness.args = SimpleNamespace(
        canonical_graph_actions=False,
        canonical_action_task="pantry_support_mask",
        canonical_graph_learner_sampling=True,
        canonical_graph_fixed_shape_sampling=True,
        canonical_graph_action_count=6,
        num_samples=4,
        train_batch_size_per_device=2,
        seed=76201,
        prompt_max_length=32,
        verifier_version="fast",
        temperature=1.0,
    )
    harness.model = _UniformBinaryModel()
    harness.tokenizer = _BinaryTokenizer()
    harness._canonical_action_space = resolve_canonical_action_space(
        harness.tokenizer, "pantry_support_mask"
    )
    harness._canonical_action_token_ids = (
        harness._canonical_action_space.union_token_ids
    )
    harness._canonical_action_token_ids_by_position = (
        harness._canonical_action_space.token_ids_by_position
    )
    harness.update_interval = 1
    harness.steps = 0
    harness._prompt_batches_consumed_total = 0
    reference = str(_first_dev_row()["answer"])

    trajectories, info = harness._sample_canonical_feedback_with_learner(
        ["raw"], ["0"], [reference]
    )

    assert len(trajectories) == 4
    assert info["actor/canonical_pantry_support_mask_actions"] == 1.0
    assert info["actor/canonical_action_count"] == 6.0
    assert info["actor/canonical_action_support_size"] == 2.0
    assert info["actor/canonical_sequence_support_size"] == 64.0
    assert info["actor/canonical_behavior_q_row_count"] == 24.0
    for trajectory in trajectories:
        assert len(trajectory.response_ids) == 6
        assert [len(row) for row in trajectory.canonical_behavior_action_logprobs] == [
            2,
        ] * 6
        assert trajectory.canonical_behavior_action_token_ids_by_position == [
            [10, 11],
        ] * 6


def test_pantry_proposal_sampler_uses_isolated_seed_and_temperature(monkeypatch):
    monkeypatch.setattr(run_module.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        run_module,
        "boxed_reward_fn",
        lambda response, reference, *, fast: ({"formatted": True}, 1.0),
    )
    harness = _RunHarness()
    harness.args = SimpleNamespace(
        canonical_graph_actions=False,
        canonical_action_task="pantry_support_mask",
        canonical_graph_learner_sampling=True,
        canonical_graph_fixed_shape_sampling=True,
        canonical_graph_action_count=6,
        num_samples=4,
        train_batch_size_per_device=2,
        seed=76201,
        prompt_max_length=32,
        verifier_version="fast",
        temperature=1.0,
    )
    harness.model = _UniformBinaryModel()
    harness.tokenizer = _BinaryTokenizer()
    harness._canonical_action_space = resolve_canonical_action_space(
        harness.tokenizer, "pantry_support_mask"
    )
    harness._canonical_action_token_ids = (
        harness._canonical_action_space.union_token_ids
    )
    harness._canonical_action_token_ids_by_position = (
        harness._canonical_action_space.token_ids_by_position
    )
    harness.update_interval = 1
    harness.steps = 0
    harness._prompt_batches_consumed_total = 0
    harness._canonical_entropy_prompt_pending = "neutral-prompt"

    _, info = harness._sample_canonical_feedback_with_learner(
        ["raw"],
        ["0"],
        [str(_first_dev_row()["answer"])],
        sampling_seed=99123,
        sampling_temperature=1.2,
        set_entropy_pending=False,
    )

    assert info["actor/canonical_request_seed"] == 99123.0
    assert info["actor/sampling_temperature"] == 1.2
    assert harness._canonical_entropy_prompt_pending == "neutral-prompt"


def test_pantry_proposal_admits_only_to_priority_replay_support():
    harness = _RunHarness()
    harness.args = SimpleNamespace(
        canonical_graph_actions=False,
        canonical_action_task="pantry_support_mask",
    )
    bank = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        replay_capacity=16,
        global_replay_groups_per_step=1,
        separate_proposal_objective_support=True,
        proposal_replay_priority_visits=4,
        proposal_replay_priority_multiplier=4.0,
    )
    bank.score_and_update(
        prompt_token_ids=[[7, 8]] * 2,
        outcome_keys=["known", None],
        task_rewards=[1.0, 0.0],
        active_mask=[1, 1],
        response_token_ids=[[11], [99]],
        num_samples=2,
    )
    harness._online_canonical_bank = bank
    sampler_calls = []

    def fake_sampler(*args, **kwargs):
        sampler_calls.append((args, kwargs))
        return [object(), object()], {"actor/canonical_request_seed": 99123.0}

    def fake_explorer(**kwargs):
        generated = kwargs["proposal_group_factory"](0, 1.2, 99123)
        assert len(generated) == 2
        return {
            "prompt_token_ids": [[7, 8]],
            "outcome_keys": ["new"],
            "response_token_ids": [[12]],
        }, {
            "actor/counterfactual_proposal_groups_generated": 1.0,
            "actor/counterfactual_proposal_rows_generated": 2.0,
        }

    harness._sample_canonical_feedback_with_learner = fake_sampler
    harness._generate_verified_counterfactual_proposals = fake_explorer
    info = harness._sample_and_admit_canonical_counterfactual_proposals(
        raw_prompts=["raw"],
        processed_prompts=["processed"],
        refs=["reference"],
        neutral_feedback=[object(), object()],
        actor_info={"actor/rewards": 0.5},
    )

    assert len(sampler_calls) == 1
    assert sampler_calls[0][1] == {
        "sampling_seed": 99123,
        "sampling_temperature": 1.2,
        "set_entropy_pending": False,
    }
    assert bank.tracked_outcome_count == 1
    assert bank.replay_mean_support_per_prompt == 2.0
    assert info["actor/counterfactual_proposal_objective_outcome_delta"] == 0.0
    assert info["actor/counterfactual_proposal_conditioned_rows_sent_to_ppo"] == 0.0
    assert info["actor/counterfactual_proposal_cumulative_new_outcomes"] == 1.0
    prioritized = bank.scheduled_global_replay_groups()[0]
    assert prioritized.outcome_keys == ("known", "new")
    assert prioritized.mass_weights == pytest.approx((0.4, 1.6))
    assert prioritized.priority_modes == 1


def test_pantry_support_mask_projects_without_certified_support_catalogue():
    row = _first_dev_row()
    spec = json.loads(row["answer"])
    stripped = dict(spec)
    stripped.pop("certified_support_sha256", None)
    stripped["certified_mode_count"] = 2

    decoded = {}
    for bits in product("01", repeat=6):
        mask = "".join(bits)
        allocation = decode_pantry_support_mask(mask, stripped)
        if allocation != PANTRY_SUPPORT_MASK_INVALID:
            decoded[mask] = allocation

    assert len(decoded) >= 2
    for mask, allocation in decoded.items():
        support = pantry_support_from_mask(mask, stripped)
        assert support is not None
        response = decode_canonical_action_response(
            "pantry_support_mask", mask, json.dumps(stripped)
        )
        assert response == allocation
        info, reward = boxed_reward_fn(response, json.dumps(stripped))
        assert info == {"formatted": True}
        assert reward == 1.0
        assert validated_modebench_outcome_key(response, stripped) == (
            "pantry_plan:pantry-v1:" + "+".join(support)
        )


def test_pantry_support_mask_invalid_width_and_infeasible_masks_fail_closed():
    row = _first_dev_row()
    spec = json.loads(row["answer"])

    assert decode_pantry_support_mask("000000", spec) == PANTRY_SUPPORT_MASK_INVALID
    assert decode_pantry_support_mask("111111", spec) == PANTRY_SUPPORT_MASK_INVALID
    assert decode_pantry_support_mask("01010", spec) == PANTRY_SUPPORT_MASK_INVALID


def test_verified_allocation_round_trips_to_exact_policy_mask_tokens():
    row = _first_dev_row()
    spec = json.loads(row["answer"])
    space = resolve_canonical_action_space(_BinaryTokenizer(), "pantry_support_mask")
    mask = next(
        candidate
        for candidate in ("".join(bits) for bits in product("01", repeat=6))
        if decode_pantry_support_mask(candidate, spec) != PANTRY_SUPPORT_MASK_INVALID
    )
    allocation = decode_pantry_support_mask(mask, spec)

    assert pantry_support_mask_from_allocation(allocation, spec) == mask
    assert canonical_action_code_from_verified_response(
        "pantry_support_mask", allocation, row["answer"]
    ) == mask
    assert canonical_action_code_token_ids(space, mask) == tuple(
        10 if bit == "0" else 11 for bit in mask
    )


def test_verified_allocation_projection_fails_closed():
    spec = json.loads(_first_dev_row()["answer"])
    with pytest.raises(PantryPlanError, match="fails its verifier"):
        pantry_support_mask_from_allocation("carrots=9999", spec)


def test_pantry_online_tracking_uses_decoded_witness_but_keeps_mask_tokens():
    decoded = ["001010", "110100"]
    witnesses = ["witness-a", "witness-b"]
    assert _task_bound_canonicalization_surfaces(
        decoded,
        {"responses": witnesses},
        canonical_task="pantry_support_mask",
    ) == witnesses
    assert _task_bound_canonicalization_surfaces(
        decoded,
        {"responses": witnesses},
        canonical_task="none",
    ) is decoded
