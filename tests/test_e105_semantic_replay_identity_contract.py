from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from oat_drgrpo.learner.grpo import (
    ZeroMathGrpoMixin,
    _task_bound_canonicalization_surfaces,
)
from oat_drgrpo.learner.run import ZeroMathRunMixin
from oat_drgrpo.math_grader import (
    boxed_reward_fn,
    validated_modebench_outcome_key,
)
from oat_drgrpo.online_canonical_bank import OnlineCanonicalBank
from oat_drgrpo.semantic_shannon import SemanticShannonTracker


def _ingredient(
    ingredient_id: str,
    *,
    energy: str,
    protein: str,
    fiber: str,
    sodium: str,
) -> dict[str, object]:
    return {
        "id": ingredient_id,
        "available_g": 300,
        "step_g": 25,
        "min_if_used_g": 50,
        "attributes_per_100g": {
            "energy_kcal": energy,
            "protein_g": protein,
            "fiber_g": fiber,
            "sodium_mg": sodium,
        },
        "tags": [],
    }


def _domain_cases() -> tuple[tuple[str, str, str, str], ...]:
    graph = {
        "verifier": "graph_coloring",
        "n": 4,
        "edges": [[1, 2], [2, 3], [3, 4]],
    }
    countdown = {
        "verifier": "countdown",
        "numbers": [2, 3, 4],
        "target": 14,
    }
    python = {
        "verifier": "python_factor_function",
        "python_version": "factor-v1",
        "cases": [6, 10, 15],
        "source": "e105-semantic-replay-identity-contract",
    }
    mathir = {
        "verifier": "mathir_algebra",
        "mathir_version": "linear-v0",
        "bindings": {"a": 3, "b": 5, "c": 14},
        "initial_lhs": "add(mul(a,x),b)",
        "initial_rhs": "c",
        "max_steps": 4,
        "support_is_open": True,
    }
    pantry = {
        "verifier": "pantry_plan",
        "pantry_version": "pantry-v1",
        "ingredients": [
            _ingredient(
                "lentils",
                energy="116",
                protein="9",
                fiber="7.9",
                sodium="2",
            ),
            _ingredient(
                "chickpeas",
                energy="164",
                protein="8.9",
                fiber="7.6",
                sodium="7",
            ),
            _ingredient(
                "brown_rice",
                energy="123",
                protein="2.7",
                fiber="1.6",
                sodium="4",
            ),
        ],
        "targets": {
            "mass_g": {"min": "250", "max": "450"},
            "energy_kcal": {"min": "300", "max": "650"},
            "protein_g": {"min": "20"},
            "fiber_g": {"min": "10"},
            "sodium_mg": {"max": "100"},
        },
        "min_ingredients": 2,
        "max_ingredients": 3,
        "forbidden_tags": [],
        "certified_mode_count": 3,
    }
    return (
        ("graph_coloring", json.dumps(graph), r"\boxed{1212}", r"\boxed{2121}"),
        (
            "countdown",
            json.dumps(countdown),
            r"\boxed{2 + 3 * 4}",
            r"\boxed{(4 + 3) * 2}",
        ),
        (
            "python_factors",
            json.dumps(python),
            r"\boxed{\lambda n: 2 if n % 2 == 0 else 3}",
            r"\boxed{\lambda n: 3 if n % 3 == 0 else 5}",
        ),
        (
            "mathir",
            json.dumps(mathir),
            r"\boxed{sub(b);div(a)}",
            r"\boxed{div(a);sub(div(b,a))}",
        ),
        (
            "pantry_plan",
            json.dumps(pantry),
            r"\boxed{lentils=200;brown_rice=100}",
            r"\boxed{chickpeas=200;brown_rice=100}",
        ),
    )


def _invalid_response(domain: str) -> str:
    return {
        # These two are deliberately parseable by the semantic surface parser
        # but fail the executable task verifier. The shared positive-reward
        # gate must keep them out of both persistent histories.
        "graph_coloring": r"\boxed{1123}",
        "countdown": r"\boxed{2 + 3 + 4}",
        "python_factors": r"\boxed{lambda n: 4}",
        "mathir": r"\boxed{sub(b)}",
        "pantry_plan": r"\boxed{brown_rice=100;chickpeas=50}",
    }[domain]


class _SurfaceTokenizer:
    def __init__(self, surfaces: list[str]) -> None:
        self.surfaces = surfaces

    def decode(self, token_ids, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return self.surfaces[int(token_ids[0])]


class _KeyHarness(ZeroMathGrpoMixin):
    def __init__(self, surfaces: list[str], *, prompt_template: str) -> None:
        self.args = SimpleNamespace(prompt_template=prompt_template)
        self.tokenizer = _SurfaceTokenizer(surfaces)


class _RunHarness(ZeroMathRunMixin):
    pass


def _set_checkpoint_fields(learner: _RunHarness) -> None:
    learner.global_step = 64
    learner.policy_sgd_step = 64.0
    learner.query_step = 64
    learner.prompt_consumed = 64
    learner.prompt_epoch = 1
    learner.steps = 64
    learner._prompt_batches_consumed_total = 64
    learner.update_interval = 1
    learner._xdr_tau_controller = None
    learner._maxent_alpha_controller = None
    learner._maxent_length_controller = None
    learner._diayn_mi_tracker = None
    learner._semantic_rms_controller = None
    learner._wandb_run_id = None
    learner._wandb_run_name = None


def _semantic_tracker() -> SemanticShannonTracker:
    return SemanticShannonTracker(
        coefficient=0.1,
        surprisal_clip=5.0,
        pseudocount=1.0,
        success_conditioned_group_centered_advantage=True,
    )


def _replay_bank() -> OnlineCanonicalBank:
    return OnlineCanonicalBank(
        entropy_alpha=0.0,
        pseudocount=1.0,
        surprisal_clip=5.0,
        retain_exemplars=True,
        replay_capacity=16,
        global_replay_groups_per_step=1,
    )


@pytest.mark.parametrize(
    ("domain", "reference", "common", "rare"),
    _domain_cases(),
    ids=[case[0] for case in _domain_cases()],
)
def test_e105_semantic_and_replay_commit_identical_verified_support(
    domain: str,
    reference: str,
    common: str,
    rare: str,
) -> None:
    """The exact E105 group path shares one verified outcome identity."""

    invalid = _invalid_response(domain)
    # Exact E105 geometry: sixteen candidates. Seven common verified rows and
    # one rare verified row learn; seven task failures and one masked verified
    # row must be exact no-ops for both histories.
    surfaces = [common] * 7 + [rare] + [invalid] * 7 + [common]
    active = [True] * 15 + [False]
    input_ids = torch.tensor(
        [[101, row_index] for row_index in range(16)],
        dtype=torch.long,
    )
    response_masks = torch.ones((16, 1), dtype=torch.float32)
    references_grouped = [[reference] * 16]
    prompt_template = (
        "qwen_pantry_support_mask" if domain == "pantry_plan" else "qwen_boxed"
    )
    harness = _KeyHarness(surfaces, prompt_template=prompt_template)

    semantic_kwargs: dict[str, object] = {}
    if domain == "pantry_plan":
        task_surfaces = _task_bound_canonicalization_surfaces(
            ["unusable-action-token-surface"] * 16,
            {"responses": surfaces},
            canonical_task="pantry_support_mask",
            expected_count=16,
        )
        assert task_surfaces == surfaces
        semantic_kwargs["response_surfaces"] = task_surfaces
    semantic_keys = harness._seed_answer_keys_grouped(
        input_ids,
        response_masks,
        16,
        references_grouped,
        **semantic_kwargs,
    )[0]

    rewards = [
        float(boxed_reward_fn(surface, reference, fast=True)[1])
        for surface in surfaces
    ]
    replay_validator_keys = [
        validated_modebench_outcome_key(surface, reference)
        for surface in surfaces
    ]
    assert [reward > 0.0 for reward in rewards] == [
        key is not None for key in replay_validator_keys
    ]
    # This is the production bank's fail-closed reward/validator intersection.
    replay_keys = [
        key if key is not None and reward > 0.0 else None
        for key, reward in zip(replay_validator_keys, rewards)
    ]
    for semantic_key, replay_key, reward in zip(
        semantic_keys, replay_keys, rewards
    ):
        if reward > 0.0:
            assert semantic_key == replay_key

    prompt_token_ids = [[17, 23]] * 16
    semantic = SemanticShannonTracker(
        coefficient=0.1,
        surprisal_clip=5.0,
        pseudocount=1.0,
        success_conditioned_group_centered_advantage=True,
    )
    advantages, semantic_diagnostics = (
        semantic.score_success_conditioned_signed_advantages_and_update(
            prompt_token_ids=prompt_token_ids,
            answer_keys=semantic_keys,
            task_rewards=rewards,
            active_mask=active,
            num_samples=16,
        )
    )
    bank = OnlineCanonicalBank(
        entropy_alpha=0.0,
        pseudocount=1.0,
        surprisal_clip=5.0,
        retain_exemplars=True,
        replay_capacity=16,
        global_replay_groups_per_step=1,
    )
    _bank_advantages, bank_diagnostics = bank.score_and_update(
        prompt_token_ids=prompt_token_ids,
        outcome_keys=replay_keys,
        task_rewards=rewards,
        active_mask=active,
        num_samples=16,
        response_token_ids=[[row_index + 1] for row_index in range(16)],
    )

    assert semantic.state_dict()["counts"] == bank.state_dict()["counts"]
    assert semantic_diagnostics.history_rows_added == 8
    assert semantic_diagnostics.eligible_fraction == pytest.approx(0.5)
    assert bank_diagnostics.eligible_fraction == pytest.approx(0.5)
    assert semantic.state_dict()["counts"] == {
        next(iter(semantic.state_dict()["counts"])): {
            str(replay_keys[0]): 7,
            str(replay_keys[7]): 1,
        }
    }

    assert advantages[:7] == pytest.approx([advantages[0]] * 7)
    assert advantages[0] < 0.0
    assert advantages[7] > 0.0
    assert advantages[8:] == pytest.approx([0.0] * 8)
    assert sum(advantages) == pytest.approx(0.0, abs=1e-12)

    replay_groups = bank.scheduled_global_replay_groups(min_modes=1)
    assert len(replay_groups) == 1
    assert set(replay_groups[0].outcome_keys) == {
        replay_keys[0],
        replay_keys[7],
    }


def test_e105_auto_resume_restores_semantic_and_replay_state_together() -> None:
    """The two histories and replay scheduler continue exactly after resume."""

    prompt_token_ids = [[17, 23]] * 16
    answer_keys = ["common"] * 14 + ["rare", None]
    rewards = [1.0] * 15 + [0.0]
    active = [True] * 16
    response_token_ids = [[1]] * 14 + [[2], [3]]

    source = _RunHarness()
    _set_checkpoint_fields(source)
    source._semantic_shannon_tracker = _semantic_tracker()
    source._online_canonical_bank = _replay_bank()
    source._semantic_shannon_tracker.score_success_conditioned_signed_advantages_and_update(
        prompt_token_ids=prompt_token_ids,
        answer_keys=answer_keys,
        task_rewards=rewards,
        active_mask=active,
        num_samples=16,
    )
    source._online_canonical_bank.score_and_update(
        prompt_token_ids=prompt_token_ids,
        outcome_keys=answer_keys,
        task_rewards=rewards,
        active_mask=active,
        num_samples=16,
        response_token_ids=response_token_ids,
    )

    checkpoint = source._checkpoint_client_state()
    assert checkpoint["semantic_shannon_tracker_state"] == (
        source._semantic_shannon_tracker.state_dict()
    )
    assert checkpoint["online_canonical_bank_state"] == (
        source._online_canonical_bank.state_dict()
    )

    restored = _RunHarness()
    _set_checkpoint_fields(restored)
    restored._semantic_shannon_tracker = _semantic_tracker()
    restored._online_canonical_bank = _replay_bank()
    restored._restore_training_progress_state(checkpoint)

    assert restored._semantic_shannon_tracker.state_dict() == (
        source._semantic_shannon_tracker.state_dict()
    )
    assert restored._online_canonical_bank.state_dict() == (
        source._online_canonical_bank.state_dict()
    )

    source_advantages, _ = source._semantic_shannon_tracker.score_success_conditioned_signed_advantages_and_update(
        prompt_token_ids=prompt_token_ids,
        answer_keys=answer_keys,
        task_rewards=rewards,
        active_mask=active,
        num_samples=16,
    )
    restored_advantages, _ = restored._semantic_shannon_tracker.score_success_conditioned_signed_advantages_and_update(
        prompt_token_ids=prompt_token_ids,
        answer_keys=answer_keys,
        task_rewards=rewards,
        active_mask=active,
        num_samples=16,
    )
    assert restored_advantages == pytest.approx(source_advantages, abs=1e-12)

    source._online_canonical_bank.score_and_update(
        prompt_token_ids=prompt_token_ids,
        outcome_keys=answer_keys,
        task_rewards=rewards,
        active_mask=active,
        num_samples=16,
        response_token_ids=response_token_ids,
    )
    restored._online_canonical_bank.score_and_update(
        prompt_token_ids=prompt_token_ids,
        outcome_keys=answer_keys,
        task_rewards=rewards,
        active_mask=active,
        num_samples=16,
        response_token_ids=response_token_ids,
    )
    assert restored._semantic_shannon_tracker.state_dict() == (
        source._semantic_shannon_tracker.state_dict()
    )
    assert restored._online_canonical_bank.state_dict() == (
        source._online_canonical_bank.state_dict()
    )
    assert restored._online_canonical_bank.scheduled_global_replay_groups(
        min_modes=1
    ) == source._online_canonical_bank.scheduled_global_replay_groups(min_modes=1)
