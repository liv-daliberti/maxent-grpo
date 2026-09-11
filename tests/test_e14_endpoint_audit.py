from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ops.exp_scaling.audit_e14_checkpoint import (
    ACTION_INDICES,
    LOG_PROBABILITY_TOLERANCE,
    PREFIXES,
    compare_tree_and_teacher_forced,
    enumerate_exact_policy,
    infer_prefix_tree_log_probabilities,
    infer_teacher_forced_log_probabilities,
    validate_endpoint_checkpoint_tag,
    valid_policy_metrics,
)


def _uniform_tree() -> dict[tuple[int, ...], torch.Tensor]:
    values = torch.full((3,), -math.log(3.0), dtype=torch.float64)
    return {prefix: values.clone() for prefix in PREFIXES}


def _teacher_forced_from_tree(
    tree: dict[tuple[int, ...], torch.Tensor],
) -> torch.Tensor:
    return torch.stack(
        [
            torch.stack([tree[action[:depth]][action[depth]] for depth in range(3)])
            for action in ACTION_INDICES
        ]
    )


def test_exact_uniform_tree_recovers_all_preregistered_quantities():
    tree = _uniform_tree()
    exact = enumerate_exact_policy(tree)
    # Three valid leaves under a uniform 27-action policy.
    rewards = [int(index in (0, 7, 26)) for index in range(27)]
    valid = valid_policy_metrics(exact, rewards)

    assert exact.probability_sum == pytest.approx(1.0, abs=1e-12)
    assert exact.leaf_entropy == pytest.approx(math.log(27.0), abs=1e-12)
    assert exact.conditional_entropy == pytest.approx(math.log(27.0), abs=1e-12)
    assert valid["valid_action_count"] == 3
    assert valid["p_valid"] == pytest.approx(3.0 / 27.0, abs=1e-12)
    assert valid["h_valid"] == pytest.approx(math.log(3.0), abs=1e-12)
    assert valid["n_eff_valid"] == pytest.approx(3.0, abs=1e-12)
    assert sum(valid["q_plus"]) == pytest.approx(1.0, abs=1e-12)


def test_nonuniform_tree_leaf_entropy_matches_conditional_identity():
    tree = {}
    for ordinal, prefix in enumerate(PREFIXES):
        logits = torch.tensor(
            [0.13 * ordinal, -0.19 * (ordinal + 1), 0.07 * (ordinal - 2)],
            dtype=torch.float64,
        )
        tree[prefix] = torch.log_softmax(logits, dim=0)

    exact = enumerate_exact_policy(tree)
    assert exact.probability_sum == pytest.approx(1.0, abs=1e-12)
    assert exact.leaf_entropy == pytest.approx(
        exact.conditional_entropy, abs=1e-12
    )

    crosscheck = compare_tree_and_teacher_forced(
        tree, _teacher_forced_from_tree(tree)
    )
    assert crosscheck["per_token_max_abs_error"] == 0.0
    assert crosscheck["sequence_max_abs_error"] == 0.0


def test_audit_math_fails_closed_on_tree_grader_or_crosscheck_drift():
    tree = _uniform_tree()
    missing = dict(tree)
    missing.pop((0, 0))
    with pytest.raises(ValueError, match="prefix tree mismatch"):
        enumerate_exact_policy(missing)

    unnormalized = dict(tree)
    unnormalized[()] = torch.zeros(3, dtype=torch.float64)
    with pytest.raises(ValueError, match="not normalized"):
        enumerate_exact_policy(unnormalized)

    exact = enumerate_exact_policy(tree)
    with pytest.raises(ValueError, match="at least one valid"):
        valid_policy_metrics(exact, [0] * 27)
    with pytest.raises(ValueError, match="binary"):
        valid_policy_metrics(exact, [2] + [0] * 26)

    teacher_forced = _teacher_forced_from_tree(tree)
    assert LOG_PROBABILITY_TOLERANCE == 5e-3
    teacher_forced[11, 2] += LOG_PROBABILITY_TOLERANCE + 1e-4
    with pytest.raises(ValueError, match="disagree"):
        compare_tree_and_teacher_forced(tree, teacher_forced)


def test_endpoint_checkpoint_requires_step_128_and_makes_alias_explicit():
    assert validate_endpoint_checkpoint_tag(
        Path("step_00128"),
        expected_optimizer_updates=128,
        allow_terminal_alias=False,
    ) == (128, "scheduled_update_boundary")

    with pytest.raises(ValueError, match="allow-terminal-alias"):
        validate_endpoint_checkpoint_tag(
            Path("step_00129"),
            expected_optimizer_updates=128,
            allow_terminal_alias=False,
        )

    assert validate_endpoint_checkpoint_tag(
        Path("step_00129"),
        expected_optimizer_updates=128,
        allow_terminal_alias=True,
    ) == (129, "forced_terminal_alias")


class _CausalToyModel(torch.nn.Module):
    """Causal table whose next-token logits depend on the observed prefix."""

    def __init__(self):
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.Tensor, int]] = []

    def forward(
        self, *, input_ids, attention_mask, use_cache, logits_to_keep
    ):
        del use_cache
        self.calls.append(
            (input_ids.detach().clone(), attention_mask.detach().clone(), logits_to_keep)
        )
        batch, sequence = input_ids.shape
        logits = torch.full((batch, sequence, 8), -100.0, device=input_ids.device)
        cumulative = input_ids.cumsum(dim=1).float()
        logits[:, :, 1] = 0.11 * cumulative
        logits[:, :, 2] = -0.07 * cumulative + 0.4
        logits[:, :, 3] = 0.03 * cumulative - 0.2
        return SimpleNamespace(logits=logits[:, -logits_to_keep:])


def test_model_inference_crosschecks_prefix_tree_against_full_teacher_forcing():
    model = _CausalToyModel()
    prompts = [[4], [5, 6]]
    trees = infer_prefix_tree_log_probabilities(
        model,
        prompts,
        action_token_ids=(1, 2, 3),
        pad_token_id=0,
        device=torch.device("cpu"),
        batch_size=5,
    )
    prefix_calls = list(model.calls)
    model.calls.clear()
    teacher_forced = infer_teacher_forced_log_probabilities(
        model,
        prompts,
        action_token_ids=(1, 2, 3),
        pad_token_id=0,
        device=torch.device("cpu"),
        batch_size=5,
    )
    teacher_calls = list(model.calls)

    # Both paths always present the model with prompt + all three canonical
    # positions, an all-one mask, the same four-logit suffix, and a full batch.
    assert prefix_calls
    assert teacher_calls
    for input_ids, attention_mask, logits_to_keep in prefix_calls:
        assert input_ids.shape[0] == 5
        assert input_ids.shape[1] in {4, 5}
        assert torch.equal(attention_mask, torch.ones_like(attention_mask))
        assert logits_to_keep == 4
    for input_ids, attention_mask, logits_to_keep in teacher_calls:
        assert input_ids.shape[0] == 5
        assert input_ids.shape[1] in {4, 5}
        assert torch.equal(attention_mask, torch.ones_like(attention_mask))
        assert logits_to_keep == 4

    for tree, observed in zip(trees, teacher_forced, strict=True):
        exact = enumerate_exact_policy(tree)
        crosscheck = compare_tree_and_teacher_forced(tree, observed)
        # Runtime logits are intentionally reduced in float32, matching the
        # learner, before the exact tree arithmetic switches to float64.
        assert exact.probability_sum == pytest.approx(1.0, abs=1e-6)
        assert crosscheck["per_token_max_abs_error"] == pytest.approx(0.0)
        assert crosscheck["sequence_max_abs_error"] == pytest.approx(0.0)
