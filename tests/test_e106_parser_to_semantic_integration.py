from __future__ import annotations

import json

import pytest

from oat_drgrpo.math_grader import validated_modebench_outcome_key
from oat_drgrpo.semantic_shannon import SemanticShannonTracker


def _reference() -> str:
    return json.dumps(
        {
            "verifier": "python_factor_function",
            "python_version": "factor-v1",
            "cases": [6, 10, 15],
            "source": "e106-parser-to-semantic-regression",
        }
    )


def test_boxed_latex_python_modes_reach_group_centered_entropy_score() -> None:
    reference = _reference()
    common_response = r"\boxed{\lambda n: 2 if n % 2 == 0 else 3}"
    rare_response = r"\boxed{\lambda n: 3 if n % 3 == 0 else 5}"
    common_key = validated_modebench_outcome_key(common_response, reference)
    rare_key = validated_modebench_outcome_key(rare_response, reference)

    assert common_key == "python_factor:2,2,3"
    assert rare_key == "python_factor:3,5,3"
    assert common_key != rare_key

    tracker = SemanticShannonTracker(
        coefficient=0.1,
        surprisal_clip=5.0,
        pseudocount=1.0,
        success_conditioned_group_centered_advantage=True,
    )
    advantages, telemetry = (
        tracker.score_success_conditioned_signed_advantages_and_update(
            prompt_token_ids=[[17, 23]] * 4,
            answer_keys=[common_key, common_key, rare_key, None],
            task_rewards=[1.0, 1.0, 1.0, 0.0],
            active_mask=[True] * 4,
            num_samples=4,
        )
    )

    assert advantages[0] == pytest.approx(advantages[1])
    assert advantages[0] < 0.0
    assert advantages[2] > 0.0
    assert advantages[3] == 0.0
    assert sum(advantages) == pytest.approx(0.0, abs=1e-12)
    assert max(abs(value) for value in advantages) <= 0.1
    assert telemetry.effective_advantage_mean == pytest.approx(0.0, abs=1e-12)
