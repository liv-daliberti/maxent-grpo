from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest
import torch

from oat_drgrpo.actor import (
    ZeroMathActor,
    _CanonicalPositionLogitsProcessor,
    _position_canonical_action_violation,
)
from oat_drgrpo.canonical_actions import (
    canonical_behavior_overlap_diagnostics,
    decode_countdown_action_code,
    enumerate_countdown_action_codes,
    materialize_position_canonical_behavior_policy,
    resolve_canonical_action_space,
    restricted_position_action_log_probs_entropy_and_distribution,
)
from oat_drgrpo.learner import run as run_module
from oat_drgrpo.learner.run import ZeroMathRunMixin
from oat_drgrpo.math_grader import _canonical_countdown_expression_key
from oat_drgrpo.maxent_controllers import MaxEntProportionalController
from oat_drgrpo.templates import (
    apply_qwen_countdown_digits_template,
    validate_qwen_countdown_digits_materialization,
)


class _DigitTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == "processed-countdown":
            return [9]
        return [int(text)]

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(str(int(token_id)) for token_id in token_ids)


def _countdown_reference() -> str:
    return json.dumps(
        {
            "verifier": "countdown",
            "numbers": [2, 5, 9],
            "target": 23,
            "num_completions": 2,
        }
    )


def test_countdown_template_materializes_only_the_audited_three_digit_contract():
    question = (
        "Using the numbers [2, 5, 9], create an expression that equals 23. "
        "Use each given number exactly once. Put the expression inside \\boxed{}."
    )
    prompt = apply_qwen_countdown_digits_template(question)

    assert "Return only three digits" in prompt
    assert "1=pair+s" in prompt
    assert "6=s/pair" in prompt
    assert "\\boxed{}" not in prompt
    validate_qwen_countdown_digits_materialization([question], [prompt])
    with pytest.raises(RuntimeError, match="stale or foreign"):
        validate_qwen_countdown_digits_materialization([question], [question])


def test_countdown_codec_has_108_nonaliasing_label_free_actions():
    reference = json.loads(_countdown_reference())
    reference["target"] = object()
    reference["solutions"] = object()
    codes = enumerate_countdown_action_codes()
    expressions = [decode_countdown_action_code(code, reference) for code in codes]
    keys = [
        _canonical_countdown_expression_key(expression, reference)
        for expression in expressions
    ]

    assert len(codes) == len(set(codes)) == 108
    assert codes[0] == "111"
    assert codes[-1] == "636"
    assert len(set(expressions)) == 108
    assert None not in keys
    assert len(set(keys)) == 108


def test_countdown_greedy_eval_decodes_action_codes_before_oracle_grading():
    references = [_countdown_reference(), _countdown_reference()]
    candidates = [["111", "123"], ["212", "636"]]
    observed = {}

    class _Oracle:
        def compare(
            self,
            prompts,
            responses,
            oracle_references,
            *,
            batch_size,
            return_probs,
            disable_tqdm,
        ):
            observed.update(
                prompts=prompts,
                responses=responses,
                references=oracle_references,
                batch_size=batch_size,
                return_probs=return_probs,
                disable_tqdm=disable_tqdm,
            )
            assert all(response.startswith("(") for response in responses)
            return [1.0, 0.0, 0.0, 1.0], {}

    actor = object.__new__(ZeroMathActor)
    actor.eval_mode = True
    actor._canonical_action_space = SimpleNamespace(task="countdown")
    actor.eval_sampling_params = SimpleNamespace(n=2)
    actor.oracle = _Oracle()
    actor.oracle_batch_size = 7
    actor.generate = lambda prompts, params: (prompts, params)
    actor.extract_candidates_from_output = lambda outputs, params: candidates

    responses, rewards = ZeroMathActor.generate_and_maybe_eval(
        actor,
        ["raw-1", "raw-2"],
        ["formatted-1", "formatted-2"],
        references,
    )

    assert responses == [
        [
            decode_countdown_action_code("111", references[0]),
            decode_countdown_action_code("123", references[0]),
        ],
        [
            decode_countdown_action_code("212", references[1]),
            decode_countdown_action_code("636", references[1]),
        ],
    ]
    assert rewards.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert observed == {
        "prompts": ["raw-1", "raw-2", "raw-1", "raw-2"],
        "responses": [
            responses[0][0],
            responses[1][0],
            responses[0][1],
            responses[1][1],
        ],
        "references": references * 2,
        "batch_size": 7,
        "return_probs": True,
        "disable_tqdm": True,
    }


@pytest.mark.parametrize("bad_code", ["", "11", "1111", "101", "141", "117"])
def test_countdown_codec_rejects_actions_outside_the_positional_grammar(bad_code):
    with pytest.raises(ValueError, match="invalid canonical Countdown"):
        decode_countdown_action_code(bad_code, _countdown_reference())


def test_countdown_positional_scorer_is_normalized_masked_and_differentiable():
    supports = ((1, 2, 3, 4, 5, 6), (1, 2, 3), (1, 2, 3, 4, 5, 6))
    labels = torch.tensor([[9, 1, 2, 6, 0], [9, 6, 3, 1, 0]])
    response_masks = torch.tensor(
        [[True, True, True, False], [True, True, True, False]]
    )
    logits = torch.zeros((2, 5, 12), requires_grad=True)

    selected, entropy, full, support_mask = (
        restricted_position_action_log_probs_entropy_and_distribution(
            logits,
            labels,
            response_masks,
            allowed_token_ids_by_position=supports,
        )
    )

    expected = torch.tensor([-math.log(6), -math.log(3), -math.log(6)])
    torch.testing.assert_close(selected[0, :3], expected)
    torch.testing.assert_close(entropy[0, :3], -expected)
    assert support_mask[0, :3].sum(dim=-1).tolist() == [6, 3, 6]
    assert torch.isneginf(full[0, 1, 3:]).all()
    assert torch.equal(full[:, 3], torch.zeros_like(full[:, 3]))
    (selected[response_masks].sum() + entropy[response_masks].sum()).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_positional_behavior_transport_matches_the_learner_distribution():
    supports = ((1, 2, 3, 4, 5, 6), (1, 2, 3), (1, 2, 3, 4, 5, 6))
    labels = torch.tensor([[9, 1, 2, 6, 0]])
    response_masks = torch.tensor([[True, True, True, False]])
    logits = torch.zeros((1, 5, 12))
    learner_selected, _, learner_full, learner_support = (
        restricted_position_action_log_probs_entropy_and_distribution(
            logits,
            labels,
            response_masks,
            allowed_token_ids_by_position=supports,
        )
    )
    compact_full = [
        learner_full[0, position, : len(support)].tolist()
        for position, support in enumerate(supports)
    ]
    compact_selected = learner_selected[0, :3].tolist()
    selected, full, support_mask, norm_error, echo_error = (
        materialize_position_canonical_behavior_policy(
            labels[:, 1:],
            response_masks,
            action_ids=[[1, 2, 6]],
            selected_log_probs=[compact_selected],
            full_log_probs=[compact_full],
            behavior_action_token_ids_by_position=[[list(row) for row in supports]],
            allowed_token_ids_by_position=supports,
        )
    )
    diagnostics = canonical_behavior_overlap_diagnostics(
        selected,
        full,
        learner_selected,
        learner_full,
        response_masks,
        support_mask=support_mask,
    )

    assert torch.equal(support_mask, learner_support)
    assert norm_error < 1e-6
    assert echo_error == 0.0
    assert diagnostics["canonical_behavior_ratio_min"] == pytest.approx(1.0)
    assert diagnostics["canonical_behavior_ratio_max"] == pytest.approx(1.0)
    assert diagnostics["canonical_behavior_tv_max"] == pytest.approx(0.0)


def test_position_logits_processor_enforces_6_by_3_by_6_support():
    supports = ((1, 2, 3, 4, 5, 6), (1, 2, 3), (1, 2, 3, 4, 5, 6))
    processor = _CanonicalPositionLogitsProcessor(supports)
    scores = torch.arange(10, dtype=torch.float32)

    middle = processor([1], scores)
    assert torch.equal(middle[1:4], scores[1:4])
    assert torch.isneginf(middle[0])
    assert torch.isneginf(middle[4:]).all()
    # vLLM V0 invokes processors once at max_tokens before stopping.  That
    # terminal bookkeeping callback must not request a nonexistent position.
    assert processor([6, 3, 6], scores) is scores
    with pytest.raises(RuntimeError, match="exceeded its fixed horizon"):
        processor([6, 3, 6, 1], scores)
    assert _position_canonical_action_violation(
        [6, 3, 6], action_token_ids_by_position=supports
    ) is None
    assert "position 1" in _position_canonical_action_violation(
        [6, 4, 6], action_token_ids_by_position=supports
    )


class _UniformCanonicalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.call_training_modes: list[bool] = []
        self.call_row_counts: list[int] = []

    def forward(self, input_ids, *, attention_mask):
        torch.testing.assert_close(attention_mask, torch.ones_like(input_ids))
        self.call_training_modes.append(bool(self.training))
        self.call_row_counts.append(int(input_ids.shape[0]))
        logits = torch.zeros(
            (*input_ids.shape, 12), dtype=torch.float32, device=input_ids.device
        )
        return {"logits": logits + self.anchor * 0.0}


class _RunHarness(ZeroMathRunMixin):
    pass


@pytest.mark.parametrize(
    ("task", "leaf_count", "prefix_count"),
    (("graph_coloring", 27, 13), ("countdown", 108, 25)),
)
def test_post_update_sensor_exactly_enumerates_the_finite_policy_tree(
    monkeypatch, task, leaf_count, prefix_count
):
    monkeypatch.setattr(run_module.dist, "get_world_size", lambda: 1)
    harness = _RunHarness()
    harness.model = _UniformCanonicalModel()
    harness.model.train()
    harness.tokenizer = _DigitTokenizer()
    harness.args = SimpleNamespace(temperature=1.0, train_batch_size_per_device=7)
    harness._canonical_action_space = resolve_canonical_action_space(
        harness.tokenizer, task
    )

    result = harness._compute_exact_canonical_sequence_entropy(
        "processed-countdown"
    )

    assert result["canonical_exact_sequence_entropy"] == pytest.approx(
        math.log(leaf_count), abs=1e-12
    )
    assert result["canonical_exact_sequence_entropy_ratio"] == pytest.approx(1.0)
    assert result["canonical_exact_prefix_row_count"] == prefix_count
    assert result["canonical_exact_leaf_count"] == leaf_count
    assert result["canonical_exact_leaf_mass"] == pytest.approx(1.0, abs=1e-12)
    assert sum(harness.model.call_row_counts) == prefix_count
    assert harness.model.call_training_modes
    assert not any(harness.model.call_training_modes)
    assert harness.model.training


def test_countdown_learner_sampler_decodes_before_reward_and_transports_ragged_q(
    monkeypatch,
):
    monkeypatch.setattr(run_module.dist, "get_world_size", lambda: 1)
    observed_responses = []

    def reward(response, reference, *, fast):
        observed_responses.append(response)
        assert reference == _countdown_reference()
        assert fast is True
        assert len(response) > 3 and response.startswith("(")
        return {"formatted": True}, 1.0

    monkeypatch.setattr(run_module, "boxed_reward_fn", reward)
    harness = _RunHarness()
    harness.args = SimpleNamespace(
        canonical_graph_actions=False,
        canonical_action_task="countdown",
        canonical_graph_learner_sampling=True,
        canonical_graph_fixed_shape_sampling=True,
        canonical_graph_action_count=3,
        num_samples=4,
        train_batch_size_per_device=2,
        seed=17,
        prompt_max_length=32,
        verifier_version="fast",
        temperature=1.0,
    )
    harness.model = _UniformCanonicalModel()
    harness.tokenizer = _DigitTokenizer()
    harness._canonical_action_space = resolve_canonical_action_space(
        harness.tokenizer, "countdown"
    )
    harness._canonical_action_token_ids = (
        harness._canonical_action_space.union_token_ids
    )
    harness._canonical_action_token_ids_by_position = (
        harness._canonical_action_space.token_ids_by_position
    )
    harness.update_interval = 1
    harness.steps = 2
    harness._prompt_batches_consumed_total = 3

    trajectories, info = harness._sample_canonical_feedback_with_learner(
        ["raw"], ["processed-countdown"], [_countdown_reference()]
    )

    assert len(trajectories) == len(observed_responses) == 4
    assert info["actor/canonical_countdown_actions"] == 1.0
    assert info["actor/canonical_sequence_support_size"] == 108.0
    assert info["actor/canonical_behavior_q_support_min"] == 3.0
    assert info["actor/canonical_behavior_q_support_max"] == 6.0
    for trajectory in trajectories:
        assert trajectory.response in observed_responses
        assert len(trajectory.response_ids) == 3
        assert [len(row) for row in trajectory.canonical_behavior_action_logprobs] == [
            6,
            3,
            6,
        ]
        assert trajectory.canonical_behavior_action_token_ids_by_position == [
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3],
            [1, 2, 3, 4, 5, 6],
        ]


def test_canonical_controller_checkpoint_rejects_legacy_entropy_units():
    controller = MaxEntProportionalController(
        base_alpha=0.05,
        max_alpha=0.5,
        target_ratio=0.8,
        warmup_steps=64,
        ema_decay=0.9,
        gain=1.0,
        configured_target_entropy=3.0,
        entropy_units="canonical_action_nats_exact_v1",
        observation_metric_key="canonical_exact_sequence_entropy",
    )
    state = controller.state_dict()
    assert state["entropy_units"] == "canonical_action_nats_exact_v1"

    for legacy_units in ("canonical_sequence_nats_v1", "sequence_nats_v1"):
        legacy = dict(state, entropy_units=legacy_units)
        with pytest.raises(ValueError, match="incompatible MaxEnt entropy units"):
            controller.load_state_dict(legacy)


def test_canonical_controller_never_falls_back_to_rollout_entropy():
    controller = MaxEntProportionalController(
        base_alpha=0.05,
        max_alpha=0.5,
        target_ratio=0.8,
        warmup_steps=64,
        ema_decay=0.9,
        gain=1.0,
        configured_target_entropy=3.0,
        entropy_units="canonical_action_nats_exact_v1",
        observation_metric_key="canonical_exact_sequence_entropy",
    )
    harness = _RunHarness()
    harness._maxent_alpha_controller = controller
    harness.strategy = SimpleNamespace(all_reduce=lambda values: values)

    with pytest.raises(RuntimeError, match="sequence-entropy observation"):
        harness._update_maxent_alpha_controller({"maxent_sequence_entropy": 3.0})

    train_info = {"canonical_exact_sequence_entropy": 3.0}
    harness._update_maxent_alpha_controller(train_info)
    assert controller.observation_count == 1


class _LearnOrderingHarness(ZeroMathRunMixin):
    def __init__(self):
        self.events = []

    def learning_step(self, _data):
        self.events.append("optimizer_step")
        return {"loss": torch.tensor(0.0)}

    def _compute_exact_canonical_sequence_entropy(self, processed_prompt):
        assert processed_prompt == "processed-countdown"
        assert self.events == ["optimizer_step"]
        self.events.append("exact_post_update_entropy")
        return {"canonical_exact_sequence_entropy": math.log(108)}

    def _update_maxent_alpha_controller(self, train_info):
        assert train_info["canonical_exact_sequence_entropy"] == pytest.approx(
            math.log(108)
        )
        self.events.append("controller_observe")


def test_learn_observes_exact_entropy_after_update_and_before_controller(monkeypatch):
    monkeypatch.setattr(run_module.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(run_module.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(run_module.dist, "barrier", lambda: None)
    monkeypatch.setattr(run_module, "DataLoader", lambda *_args, **_kwargs: [object()])
    harness = _LearnOrderingHarness()
    harness.pi_buffer = [object()]
    harness.tokenizer = _DigitTokenizer()
    harness.strategy = SimpleNamespace(
        grad_acc_step=1,
        is_rank_0=lambda: False,
        print=lambda *_args, **_kwargs: None,
    )
    class _Dataset:
        def __len__(self):
            return 1

        def collate_fn(self, items):
            return items

    harness.dataset_builder = lambda *_args, **_kwargs: _Dataset()
    harness.args = SimpleNamespace(
        critic_type="drgrpo",
        max_sgd_steps=1,
        num_ppo_epochs=1,
        train_batch_size_per_device=1,
    )
    harness.model = _UniformCanonicalModel()
    harness.critic = None
    harness.policy_sgd_step = 0
    harness.global_step = 0
    harness.gradient_update_st = 0.0
    harness._xdr_tau_controller = None
    harness._maxent_length_controller = None
    harness._canonical_entropy_prompt_pending = "processed-countdown"

    harness.learn(2)

    assert harness.events == [
        "optimizer_step",
        "exact_post_update_entropy",
        "controller_observe",
    ]
