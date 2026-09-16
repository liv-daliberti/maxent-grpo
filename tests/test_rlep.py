from __future__ import annotations

import json

import pytest
import torch

from oat_drgrpo.rlep import RLEPExperiencePool, rlep_mixed_advantages
from ops.exp_scaling.audit_e98_rlep_pool import audit_pool


def _write_pool(tmp_path, *, successes_per_prompt: int = 4, sparse: bool = False):
    path = tmp_path / "debug_job1" / "eval_mode_coverage_draws.jsonl"
    path.parent.mkdir()
    rows = []
    for draw in range(4):
        prompt_rows = []
        for prompt in range(2):
            rewards = [0.0] * 16
            prompt_successes = 0 if sparse and prompt == 1 else successes_per_prompt
            if draw < prompt_successes:
                rewards[draw] = 1.0
            prompt_rows.append(
                {
                    "prompt": f"prompt-{prompt}",
                    "reference": {"id": prompt},
                    "responses": [
                        f"response-{prompt}-{draw}-{index}" for index in range(16)
                    ],
                    "rewards": rewards,
                }
            )
        rows.append(
            {
                "draw_index": draw,
                "evaluation_kind": "fixed_seed_sampled_k_neutral",
                "sample_count": 16,
                "temperature": 0.7,
                "top_p": 0.95,
                "prompts": prompt_rows,
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return tmp_path


def test_pool_loads_verified_rows_and_samples_deterministically(tmp_path):
    pool = RLEPExperiencePool.from_directory(_write_pool(tmp_path))
    assert pool.diagnostics.prompts == 2
    assert pool.diagnostics.trajectories == 8
    assert pool.diagnostics.minimum_trajectories_per_prompt == 4
    first = pool.sample({"id": 0}, count=2, experiment_seed=43, learner_step=9)
    second = pool.sample({"id": 0}, count=2, experiment_seed=43, learner_step=9)
    assert first == second
    assert len(set(first)) == 2


def test_pool_fails_closed_when_a_prompt_has_fewer_than_two_successes(tmp_path):
    with pytest.raises(ValueError, match="at least two"):
        RLEPExperiencePool.from_directory(
            _write_pool(tmp_path, successes_per_prompt=1)
        )

def test_sparse_pool_preserves_ineligible_prompts_for_drgrpo_fallback():
    pool = RLEPExperiencePool(
        {"eligible": ["answer-a", "answer-b"], "ineligible": []},
        allow_sparse=True,
    )
    diagnostics = pool.diagnostics
    assert diagnostics.prompts == 2
    assert diagnostics.eligible_prompts == 1
    assert diagnostics.ineligible_prompts == 1
    assert pool.can_sample("eligible", count=2)
    assert not pool.can_sample("ineligible", count=2)
    with pytest.raises(ValueError, match="fewer verified"):
        pool.sample(
            "ineligible", count=2, experiment_seed=43, learner_step=1
        )


def test_mixed_advantages_equal_a_single_four_row_drgrpo_batch():
    mixed = rlep_mixed_advantages(torch.tensor([[1.0], [0.0]]), replay_count=2)
    assert mixed.mixed_reward_mean == pytest.approx(0.75)
    assert mixed.replay_advantage == pytest.approx(0.25)
    assert mixed.fresh[:, 0].tolist() == pytest.approx([0.125, -0.375])

    fresh_score = torch.tensor([2.0, 3.0])
    replay_score = torch.tensor([5.0, 7.0])
    split_objective = (
        (mixed.fresh[:, 0] * fresh_score).mean()
        + mixed.replay_advantage * replay_score.sum() / 4.0
    )
    rewards = torch.tensor([1.0, 0.0, 1.0, 1.0])
    scores = torch.cat((fresh_score, replay_score))
    direct_objective = ((rewards - rewards.mean()) * scores).mean()
    assert split_objective.item() == pytest.approx(direct_objective.item())


def test_pool_audit_writes_a_content_bound_completion_receipt(tmp_path):
    root = _write_pool(tmp_path)
    marker = root / "debug_job1" / "EVAL_ONLY_COMPLETE.json"
    marker.write_text(
        json.dumps(
            {
                "eval_mode_coverage_k": 16,
                "eval_mode_coverage_temperature": 0.7,
                "eval_mode_coverage_top_p": 0.95,
                "eval_mode_coverage_draws": 4,
                "test_split": "multi_answer",
            }
        )
    )
    payload = audit_pool(root, expected_prompts=2)
    receipt = json.loads((root / "RLEP_POOL_COMPLETE.json").read_text())
    assert receipt == payload
    assert payload["prompts"] == 2
    assert payload["minimum_trajectories_per_prompt"] == 4
    assert len(payload["sidecar_sha256"]) == 64
