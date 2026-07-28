import math
import sys
from types import SimpleNamespace

import pytest

from oat_drgrpo.actor import (
    _canonical_action_violation,
    _configure_canonical_sampling_params,
    _extract_canonical_behavior_logprobs,
    _extract_selected_action_logprobs,
    _resolve_mode_coverage_allowed_token_ids,
    ZeroMathActor,
)


def test_configure_canonical_sampling_params_fixes_support_and_horizon():
    params = SimpleNamespace(
        allowed_token_ids=None,
        max_tokens=192,
        min_tokens=0,
        ignore_eos=False,
        stop=["stop"],
        stop_token_ids=[2],
    )

    _configure_canonical_sampling_params(
        params,
        action_token_ids=(16, 17, 18),
        action_count=3,
    )

    assert params.allowed_token_ids == [16, 17, 18]
    assert params.max_tokens == 3
    assert params.min_tokens == 3
    assert params.ignore_eos is True
    assert params.stop is None
    assert params.stop_token_ids is None
    assert params.logprobs == 3


def test_canonical_action_validation_accepts_only_exact_supported_actions():
    kwargs = {"action_token_ids": (16, 17, 18), "action_count": 3}
    assert _canonical_action_violation([16, 18, 17], **kwargs) is None
    assert "expected 3 tokens" in _canonical_action_violation([16, 17], **kwargs)
    assert "unsupported token ids [19]" in _canonical_action_violation(
        [16, 19, 17], **kwargs
    )


def test_actor_logprob_extraction_requires_finite_selected_entries():
    rows = [
        {16: SimpleNamespace(logprob=-0.1)},
        {17: SimpleNamespace(logprob=-0.2)},
        {18: SimpleNamespace(logprob=-0.3)},
    ]
    assert _extract_selected_action_logprobs([16, 17, 18], rows) == [
        -0.1,
        -0.2,
        -0.3,
    ]

    rows[1][17].logprob = math.nan
    with pytest.raises(RuntimeError, match="nonfinite"):
        _extract_selected_action_logprobs([16, 17, 18], rows)

    with pytest.raises(RuntimeError, match="missing"):
        _extract_selected_action_logprobs([16], [{}])


def test_actor_extracts_complete_normalized_canonical_behavior_policy():
    rows = [
        {
            16: SimpleNamespace(logprob=math.log(0.2)),
            17: SimpleNamespace(logprob=math.log(0.3)),
            18: SimpleNamespace(logprob=math.log(0.5)),
        },
        {
            16: SimpleNamespace(logprob=math.log(0.6)),
            17: SimpleNamespace(logprob=math.log(0.1)),
            18: SimpleNamespace(logprob=math.log(0.3)),
        },
    ]

    selected, full, norm_error = _extract_canonical_behavior_logprobs(
        [18, 16], rows, action_token_ids=(16, 17, 18)
    )

    assert selected == pytest.approx([math.log(0.5), math.log(0.6)])
    expected_full = [
        [math.log(0.2), math.log(0.3), math.log(0.5)],
        [math.log(0.6), math.log(0.1), math.log(0.3)],
    ]
    for observed, expected in zip(full, expected_full):
        assert observed == pytest.approx(expected)
    assert norm_error <= 1e-6


def test_actor_rejects_incomplete_or_unnormalized_behavior_policy():
    missing = [
        {
            16: SimpleNamespace(logprob=math.log(0.5)),
            17: SimpleNamespace(logprob=math.log(0.5)),
        }
    ]
    with pytest.raises(RuntimeError, match="support mismatch"):
        _extract_canonical_behavior_logprobs(
            [16], missing, action_token_ids=(16, 17, 18)
        )

    unnormalized = [
        {
            16: SimpleNamespace(logprob=math.log(0.5)),
            17: SimpleNamespace(logprob=math.log(0.5)),
            18: SimpleNamespace(logprob=math.log(0.5)),
        }
    ]
    with pytest.raises(RuntimeError, match="not normalized"):
        _extract_canonical_behavior_logprobs(
            [16], unnormalized, action_token_ids=(16, 17, 18)
        )


def test_mode_coverage_recovers_canonical_support_after_vllm_clears_field():
    # vLLM V0 clears this field after converting it to a logits processor on
    # the initial greedy evaluation request.  The later coverage request must
    # rebuild its support from the actor's immutable canonical token tuple.
    mutated_eval_params = SimpleNamespace(allowed_token_ids=None)

    assert _resolve_mode_coverage_allowed_token_ids(
        mutated_eval_params, (16, 17, 18)
    ) == [16, 17, 18]


def test_mode_coverage_preserves_noncanonical_allowed_tokens_when_present():
    eval_params = SimpleNamespace(allowed_token_ids=[7, 9])

    assert _resolve_mode_coverage_allowed_token_ids(eval_params, None) == [7, 9]
    assert _resolve_mode_coverage_allowed_token_ids(
        SimpleNamespace(allowed_token_ids=None), None
    ) is None


def test_mode_coverage_forwards_explicit_reproducible_seed(monkeypatch):
    captured = {}

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(SamplingParams=FakeSamplingParams)
    )
    sample = SimpleNamespace(
        text=r"\boxed{1}", token_ids=[1], finish_reason="stop"
    )
    actor = SimpleNamespace(
        eval_sampling_params=SimpleNamespace(
            max_tokens=8,
            min_tokens=0,
            ignore_eos=False,
            stop=None,
            stop_token_ids=None,
            allowed_token_ids=None,
            include_stop_str_in_output=False,
        ),
        args=SimpleNamespace(prompt_template="qwen_boxed"),
        generate=lambda _prompts, _params: [SimpleNamespace(outputs=[sample])],
        oracle=SimpleNamespace(
            get_reward=lambda _prompts, _responses, _refs: (
                SimpleNamespace(tolist=lambda: [1.0]),
                {},
            )
        ),
    )

    result = ZeroMathActor.generate_for_mode_coverage(
        actor, ["prompt"], ["1"], 1, 1.0, seed=1003
    )

    assert captured["seed"] == 1003
    assert result["rewards"] == [[1.0]]
    assert result["responses"] == [[r"\boxed{1}"]]


def test_diayn_neutral_quality_keeps_the_control_request_shape(monkeypatch):
    calls = []

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(SamplingParams=FakeSamplingParams)
    )
    sample = SimpleNamespace(
        text=r"\boxed{1}", token_ids=[1], finish_reason="stop"
    )

    def generate(prompts, params):
        calls.append((list(prompts), params))
        return [
            SimpleNamespace(outputs=[sample for _ in range(params.n)])
            for _ in prompts
        ]

    actor = SimpleNamespace(
        eval_sampling_params=SimpleNamespace(
            max_tokens=8,
            min_tokens=0,
            ignore_eos=False,
            stop=None,
            stop_token_ids=None,
            allowed_token_ids=None,
            include_stop_str_in_output=False,
        ),
        args=SimpleNamespace(prompt_template="qwen_boxed", diayn_num_options=4),
        generate=generate,
        oracle=SimpleNamespace(
            get_reward=lambda _prompts, responses, _refs: (
                SimpleNamespace(tolist=lambda: [1.0] * len(responses)),
                {},
            )
        ),
    )

    result = ZeroMathActor.generate_for_mode_coverage(
        actor, ["neutral prompt"], ["1"], 8, 1.0, seed=360103
    )

    prompts, params = calls[0]
    assert prompts == ["neutral prompt"]
    assert params.n == 8
    assert params.seed == 360103
    assert result["option_ids"] == [[None] * 8]
    assert result["request_seeds_by_prompt"] is None


def test_diayn_binding_uses_independent_prompt_option_seeds(monkeypatch):
    calls = []

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(SamplingParams=FakeSamplingParams)
    )
    sample = SimpleNamespace(
        text=r"\boxed{1}", token_ids=[1], finish_reason="stop"
    )

    def generate(prompts, params):
        calls.append((list(prompts), list(params)))
        return [
            SimpleNamespace(outputs=[sample for _ in range(request_params.n)])
            for request_params in params
        ]

    actor = SimpleNamespace(
        eval_sampling_params=SimpleNamespace(
            max_tokens=8,
            min_tokens=0,
            ignore_eos=False,
            stop=None,
            stop_token_ids=None,
            allowed_token_ids=None,
            include_stop_str_in_output=False,
        ),
        args=SimpleNamespace(prompt_template="qwen_boxed", diayn_num_options=4),
        generate=generate,
        oracle=SimpleNamespace(
            get_reward=lambda _prompts, responses, _refs: (
                SimpleNamespace(tolist=lambda: [1.0] * len(responses)),
                {},
            )
        ),
    )

    result = ZeroMathActor.generate_for_mode_coverage(
        actor,
        ["prompt 5", "prompt 9"],
        ["1", "1"],
        8,
        1.0,
        seed=360103,
        condition_on_answer_options=True,
        prompt_indices=[5, 9],
    )

    prompts, params = calls[0]
    assert len(prompts) == 8
    assert {request_params.n for request_params in params} == {2}
    expected = [
        360103_000_020,
        360103_000_021,
        360103_000_022,
        360103_000_023,
        360103_000_036,
        360103_000_037,
        360103_000_038,
        360103_000_039,
    ]
    assert [request_params.seed for request_params in params] == expected
    assert len(set(expected)) == 8
    assert result["request_seeds_by_prompt"] == [expected[:4], expected[4:]]
    assert result["option_ids"] == [
        [0, 0, 1, 1, 2, 2, 3, 3],
        [0, 0, 1, 1, 2, 2, 3, 3],
    ]
