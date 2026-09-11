from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from oat_drgrpo.learner import run as run_module
from oat_drgrpo.learner.grpo import _temporary_eval_mode
from oat_drgrpo.learner.run import ZeroMathRunMixin


_SUPPORT = (16, 17, 18)
_TOKEN_TO_DIGIT = {16: "1", 17: "2", 18: "3"}


class _Tokenizer:
    _processed = {
        "processed-a": [40],
        "processed-b": [50, 51],
    }

    def encode(self, text):
        return list(self._processed[text])

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens,
        clean_up_tokenization_spaces,
    ):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(_TOKEN_TO_DIGIT[int(token_id)] for token_id in token_ids)


class _MalformedDecodeTokenizer(_Tokenizer):
    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens,
        clean_up_tokenization_spaces,
    ):
        super().decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        return "1 2 3"


class _PrefixPolicy(torch.nn.Module):
    """Small causal policy whose next-action law depends on the full prefix."""

    vocab_size = 64
    _prompt_lengths = {40: 1, 50: 2}

    def __init__(self, *, uniform: bool = False):
        super().__init__()
        self.uniform = uniform
        self.call_inputs: list[list[tuple[int, ...]]] = []
        self.call_attention_masks: list[list[tuple[int, ...]]] = []
        self.call_training_modes: list[bool] = []

    def action_logits(self, sequence: list[int] | tuple[int, ...]) -> torch.Tensor:
        sequence = tuple(int(token_id) for token_id in sequence)
        marker = sequence[0]
        prompt_length = self._prompt_lengths[marker]
        prefix = sequence[prompt_length:]
        if self.uniform:
            probabilities = (1 / 3, 1 / 3, 1 / 3)
        elif not prefix:
            probabilities = (0.15, 0.35, 0.50) if marker == 40 else (0.55, 0.25, 0.20)
        elif len(prefix) == 1:
            probabilities = {
                16: (0.10, 0.80, 0.10),
                17: (0.65, 0.10, 0.25),
                18: (0.20, 0.30, 0.50),
            }[prefix[0]]
        elif len(prefix) == 2:
            first = _SUPPORT.index(prefix[0])
            second = _SUPPORT.index(prefix[1])
            peak = (first + 2 * second + int(marker == 50)) % len(_SUPPORT)
            rows = (
                (0.70, 0.20, 0.10),
                (0.10, 0.70, 0.20),
                (0.20, 0.10, 0.70),
            )
            probabilities = rows[peak]
        else:
            raise AssertionError(f"unexpected canonical prefix: {prefix}")
        return torch.tensor(probabilities, dtype=torch.float32).log()

    def forward(self, input_ids, *, attention_mask):
        torch.testing.assert_close(attention_mask, torch.ones_like(input_ids))
        rows = [tuple(int(value) for value in row) for row in input_ids.tolist()]
        self.call_inputs.append(rows)
        self.call_attention_masks.append(
            [tuple(int(value) for value in row) for row in attention_mask.tolist()]
        )
        self.call_training_modes.append(bool(self.training))
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], self.vocab_size),
            123.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        for row_index, sequence in enumerate(rows):
            prompt_length = self._prompt_lengths[sequence[0]]
            for realized_action_count in range(3):
                prediction_position = prompt_length + realized_action_count - 1
                prefix = sequence[: prompt_length + realized_action_count]
                logits[row_index, prediction_position, list(_SUPPORT)] = (
                    self.action_logits(prefix).to(input_ids.device)
                )
        return {"logits": logits}


class _Sampler(ZeroMathRunMixin):
    pass


def _make_sampler(
    *,
    model: _PrefixPolicy | None = None,
    tokenizer: _Tokenizer | None = None,
    seed: int = 123,
    steps: int = 7,
    consumed: int = 4,
    num_samples: int = 2,
    temperature: float = 1.0,
) -> _Sampler:
    sampler = _Sampler()
    sampler.args = SimpleNamespace(
        canonical_graph_actions=True,
        canonical_graph_learner_sampling=True,
        canonical_graph_fixed_shape_sampling=True,
        canonical_graph_action_count=3,
        num_samples=num_samples,
        train_batch_size_per_device=2,
        seed=seed,
        prompt_max_length=32,
        verifier_version="fast",
        temperature=temperature,
    )
    sampler.model = model or _PrefixPolicy()
    sampler.tokenizer = tokenizer or _Tokenizer()
    sampler._canonical_action_token_ids = _SUPPORT
    sampler.update_interval = 1
    sampler.steps = steps
    sampler._prompt_batches_consumed_total = consumed
    return sampler


@pytest.fixture(autouse=True)
def _force_cpu_sampler(monkeypatch):
    real_torch_device = torch.device
    monkeypatch.setattr(run_module.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(run_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        run_module.torch,
        "device",
        lambda *_args, **_kwargs: real_torch_device("cpu"),
    )


def _binary_reward(response: str, reference: str, *, fast: bool):
    assert fast is True
    target_digit = {"ref-a": "1", "ref-b": "3"}[reference]
    return {"formatted": len(response) == 3}, float(response[-1] == target_digit)


def _trajectory_signature(trajectories):
    return [
        (
            trajectory.prompt,
            tuple(trajectory.prompt_ids),
            trajectory.response,
            tuple(trajectory.response_ids),
            tuple(float(value) for value in trajectory.response_logprobs),
            tuple(float(value) for value in trajectory.rewards),
            tuple(
                tuple(float(value) for value in row)
                for row in trajectory.canonical_behavior_action_logprobs
            ),
            tuple(trajectory.canonical_behavior_action_token_ids),
            trajectory.reference,
        )
        for trajectory in trajectories
    ]


def test_learner_sampler_preserves_autoregressive_group_and_policy_transport(
    monkeypatch,
):
    grader_calls = []

    def grader(response, reference, *, fast):
        grader_calls.append((response, reference, fast))
        return _binary_reward(response, reference, fast=fast)

    monkeypatch.setattr(run_module, "boxed_reward_fn", grader)
    model = _PrefixPolicy()
    sampler = _make_sampler(model=model)

    trajectories, info = sampler._sample_canonical_feedback_with_learner(
        ["raw-a"],
        ["processed-a"],
        ["ref-a"],
    )

    assert [trajectory.prompt for trajectory in trajectories] == ["raw-a", "raw-a"]
    assert [trajectory.prompt_ids for trajectory in trajectories] == [[40], [40]]
    assert [trajectory.reference for trajectory in trajectories] == [
        "ref-a",
        "ref-a",
    ]
    assert all(len(trajectory.response_ids) == 3 for trajectory in trajectories)
    assert all(
        trajectory.response
        == "".join(_TOKEN_TO_DIGIT[token] for token in trajectory.response_ids)
        for trajectory in trajectories
    )
    assert grader_calls == [
        (trajectory.response, trajectory.reference, True) for trajectory in trajectories
    ]

    # Every depth uses the eventual teacher-forcing width and microbatch.  A
    # support-token suffix is overwritten one action at a time, while all-ones
    # attention preserves the teacher-forcing attention kernel.
    assert len(model.call_inputs) == 3
    for depth in range(3):
        expected_inputs = [
            tuple(
                [40]
                + list(trajectory.response_ids[:depth])
                + [_SUPPORT[0]] * (3 - depth)
            )
            for trajectory in trajectories
        ]
        assert model.call_inputs[depth] == expected_inputs
        assert model.call_attention_masks[depth] == [
            (1,) * 4 for _ in trajectories
        ]
    assert model.call_training_modes == [False, False, False]

    for trajectory in trajectories:
        prefix = list(trajectory.prompt_ids)
        assert trajectory.canonical_behavior_action_token_ids == list(_SUPPORT)
        assert trajectory.rewards[:2] == [0.0, 0.0]
        assert trajectory.rewards[-1] in {0.0, 1.0}
        for position, token_id in enumerate(trajectory.response_ids):
            expected_full = torch.log_softmax(model.action_logits(prefix), dim=-1)
            observed_full = torch.tensor(
                trajectory.canonical_behavior_action_logprobs[position]
            )
            torch.testing.assert_close(observed_full, expected_full)
            support_index = _SUPPORT.index(token_id)
            assert float(trajectory.response_logprobs[position]) == pytest.approx(
                float(expected_full[support_index])
            )
            prefix.append(token_id)

    expected_rewards = [float(trajectory.rewards[-1]) for trajectory in trajectories]
    assert info["actor/rewards"] == pytest.approx(sum(expected_rewards) / 2)
    assert info["actor/num_data"] == 2.0
    assert info["actor/canonical_behavior_q_row_count"] == 6.0
    assert info["actor/canonical_behavior_q_norm_error_max"] < 1e-6
    assert info["actor/canonical_sampler_learner"] == 1.0
    assert info["actor/canonical_sampler_fixed_shape"] == 1.0
    assert info["actor/canonical_request_seed"] == float(123 + 1_000_003 * 7 + 4)
    assert model.training is True


def test_fixed_shape_behavior_trace_matches_completed_teacher_forcing(monkeypatch):
    monkeypatch.setattr(run_module, "boxed_reward_fn", _binary_reward)
    model = _PrefixPolicy()
    sampler = _make_sampler(model=model, num_samples=4)

    trajectories, _ = sampler._sample_canonical_feedback_with_learner(
        ["raw-a"], ["processed-a"], ["ref-a"]
    )

    model.eval()
    for batch_start in range(0, len(trajectories), 2):
        batch = trajectories[batch_start : batch_start + 2]
        input_ids = torch.tensor(
            [
                list(trajectory.prompt_ids) + list(trajectory.response_ids)
                for trajectory in batch
            ],
            dtype=torch.long,
        )
        logits = model(input_ids, attention_mask=torch.ones_like(input_ids))["logits"]
        for row_index, trajectory in enumerate(batch):
            for position in range(3):
                teacher_forced = torch.log_softmax(
                    logits[row_index, len(trajectory.prompt_ids) + position - 1]
                    .index_select(-1, torch.tensor(_SUPPORT))
                    .float(),
                    dim=-1,
                )
                transported = torch.tensor(
                    trajectory.canonical_behavior_action_logprobs[position]
                )
                torch.testing.assert_close(transported, teacher_forced)


def test_temporary_eval_mode_restores_model_state():
    model = _PrefixPolicy()
    assert model.training is True
    with _temporary_eval_mode(model, enabled=True):
        assert model.training is False
    assert model.training is True

    model.eval()
    with _temporary_eval_mode(model, enabled=True):
        assert model.training is False
    assert model.training is False

    model.train()
    with _temporary_eval_mode(model, enabled=False):
        assert model.training is True
    assert model.training is True


def test_learner_sampler_rejects_multiple_prompt_microbatch(monkeypatch):
    monkeypatch.setattr(run_module, "boxed_reward_fn", _binary_reward)
    sampler = _make_sampler(model=_PrefixPolicy(), num_samples=2)

    with pytest.raises(RuntimeError, match="exactly one prompt|microbatch"):
        sampler._sample_canonical_feedback_with_learner(
            ["raw-a", "raw-b"],
            ["processed-a", "processed-b"],
            ["ref-a", "ref-b"],
        )


def test_learner_sampler_rejects_non_fixed_shape_path(monkeypatch):
    monkeypatch.setattr(run_module, "boxed_reward_fn", _binary_reward)
    sampler = _make_sampler(model=_PrefixPolicy(), num_samples=2)
    sampler.args.canonical_graph_fixed_shape_sampling = False

    with pytest.raises(RuntimeError, match="fixed-shape|placeholder"):
        sampler._sample_canonical_feedback_with_learner(
            ["raw-a"], ["processed-a"], ["ref-a"]
        )


def test_learner_sampler_seed_is_reproducible_and_changes_with_progress(monkeypatch):
    monkeypatch.setattr(run_module, "boxed_reward_fn", _binary_reward)
    first = _make_sampler(model=_PrefixPolicy(uniform=True), num_samples=8)
    replay = _make_sampler(model=_PrefixPolicy(uniform=True), num_samples=8)
    advanced = _make_sampler(model=_PrefixPolicy(uniform=True), num_samples=8, steps=8)

    first_data, first_info = first._sample_canonical_feedback_with_learner(
        ["raw-a"], ["processed-a"], ["ref-a"]
    )
    replay_data, replay_info = replay._sample_canonical_feedback_with_learner(
        ["raw-a"], ["processed-a"], ["ref-a"]
    )
    advanced_data, advanced_info = advanced._sample_canonical_feedback_with_learner(
        ["raw-a"], ["processed-a"], ["ref-a"]
    )

    assert _trajectory_signature(first_data) == _trajectory_signature(replay_data)
    assert (
        first_info["actor/canonical_request_seed"]
        == replay_info["actor/canonical_request_seed"]
    )
    assert advanced_info["actor/canonical_request_seed"] == (
        first_info["actor/canonical_request_seed"] + 1_000_003
    )
    assert [trajectory.response_ids for trajectory in advanced_data] != [
        trajectory.response_ids for trajectory in first_data
    ]


def test_learner_sampler_applies_temperature_to_behavior_policy(monkeypatch):
    monkeypatch.setattr(run_module, "boxed_reward_fn", _binary_reward)
    model = _PrefixPolicy()
    sampler = _make_sampler(model=model, num_samples=1, temperature=2.0)

    trajectories, info = sampler._sample_canonical_feedback_with_learner(
        ["raw-a"], ["processed-a"], ["ref-a"]
    )

    expected = torch.log_softmax(model.action_logits([40]) / 2.0, dim=-1)
    observed = torch.tensor(trajectories[0].canonical_behavior_action_logprobs[0])
    torch.testing.assert_close(observed, expected)
    assert info["actor/sampling_temperature"] == 2.0


def test_learner_sampler_rejects_malformed_decoded_response(monkeypatch):
    def grader(*_args, **_kwargs):
        raise AssertionError("a malformed canonical response reached the verifier")

    monkeypatch.setattr(run_module, "boxed_reward_fn", grader)
    sampler = _make_sampler(
        tokenizer=_MalformedDecodeTokenizer(),
        model=_PrefixPolicy(uniform=True),
        num_samples=1,
    )

    with pytest.raises((RuntimeError, ValueError), match="canonical|decoded|response"):
        sampler._sample_canonical_feedback_with_learner(
            ["raw-a"], ["processed-a"], ["ref-a"]
        )


def test_learner_sampler_rejects_nonbinary_reward(monkeypatch):
    monkeypatch.setattr(
        run_module,
        "boxed_reward_fn",
        lambda *_args, **_kwargs: ({"formatted": True}, 0.25),
    )
    sampler = _make_sampler(model=_PrefixPolicy(uniform=True), num_samples=1)

    with pytest.raises((RuntimeError, ValueError), match="binary|reward"):
        sampler._sample_canonical_feedback_with_learner(
            ["raw-a"], ["processed-a"], ["ref-a"]
        )
