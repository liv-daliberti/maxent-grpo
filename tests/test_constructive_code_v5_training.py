from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "ops", ROOT / "ops/exp_scaling", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_constructive_code_v5 as audit
import launch_constructive_code_v5_experiment as launch
import train_constructive_code_v5 as train


def test_frozen_task_schedule_and_compute_rectangle() -> None:
    assert train.DEVELOPMENT_PROBLEMS == ("359_B", "988_A", "1283_C", "1399_D")
    assert train.TRAIN_PROBLEMS == ("327_B", "659_C", "1208_C", "1102_B")
    assert train.EVALUATION_PROBLEMS == ("361_B", "1294_C", "1408_A", "149_C")
    assert train.EXPECTED_FAMILIES == (
        "ordered_sequence",
        "unordered_set",
        "assignment",
        "unordered_partition",
    )
    assert train.SAMPLES == train.REPLAY_CAPACITY == 16
    assert train.RESPONSE_TOKENS == 1024


def test_request_namespaces_are_paired_and_phase_disjoint() -> None:
    shared = train.request_namespace(
        mode=train.PAIRED_MODE,
        seed=78101,
        phase="train",
        update=2,
        task_index=1,
        draw=0,
    )
    assert shared == train.request_namespace(
        mode=train.PAIRED_MODE,
        seed=78101,
        phase="train",
        update=2,
        task_index=1,
        draw=0,
    )
    assert shared != train.request_namespace(
        mode=train.STAGE_B_MODE,
        seed=43,
        phase="eval_sample",
        update=2,
        task_index=1,
        draw=0,
    )
    with pytest.raises(ValueError):
        train.request_namespace(
            mode=train.STAGE_B_MODE,
            seed=43,
            phase="future",
            update=0,
            task_index=0,
            draw=0,
        )


def test_only_exact_complete_markdown_fence_is_stripped() -> None:
    assert train.strip_exact_surrounding_fence("```python\nprint(1)\n```") == (
        "print(1)",
        True,
    )
    assert train.strip_exact_surrounding_fence("text\n```python\nprint(1)\n```") == (
        "text\n```python\nprint(1)\n```",
        False,
    )


def test_selected_suite_prefers_overlay_and_falls_back_to_plus() -> None:
    problem_keys = {"359_B": "k1", "988_A": "k2"}
    gate = {
        "status": "pass",
        "expected_replay_count": 2304,
        "observed_replay_count": 2304,
        "violations": [],
        "checker_equivalence_violations": [],
        "task_results": [
            *[
                {"problem_key": key, "status": "pass"}
                for key in ("k1", "k2")
            ],
            *[
                {"problem_key": f"unused-{index}", "status": "pass"}
                for index in range(10)
            ],
        ],
        "suite_results": [
            {
                "problem_key": "k1",
                "suite_id": train.OVERLAY_SUITE,
                "status": "pass",
            },
            {
                "problem_key": "k1",
                "suite_id": train.PLUS_SUITE,
                "status": "pass",
            },
            {
                "problem_key": "k2",
                "suite_id": train.OVERLAY_SUITE,
                "status": "fail",
            },
            {
                "problem_key": "k2",
                "suite_id": train.PLUS_SUITE,
                "status": "pass",
            },
        ],
    }
    assert train._selected_suites(gate, problem_keys) == {
        "359_B": train.OVERLAY_SUITE,
        "988_A": train.PLUS_SUITE,
    }


def test_fixed_rectangle_has_exact_1024_response_slots() -> None:
    input_ids, attention, mask, prompt_length = train._fixed_rectangle(
        prompt_token_ids=(10, 11, 12),
        response_rows=((20, 21), (30,)),
        pad_token_id=0,
        device="cpu",
    )
    assert input_ids.shape == (2, 3 + 1024)
    assert attention.shape == input_ids.shape
    assert mask.shape == (2, 1024)
    assert prompt_length == 3
    assert mask.sum(dim=1).tolist() == [2, 1]
    assert input_ids[0, :5].tolist() == [10, 11, 12, 20, 21]


def test_split_replay_has_mass_for_singleton_and_balance_for_multimode() -> None:
    singleton = train.canonical_replay_split_mass_balance_loss(
        torch.tensor([-2.0], dtype=torch.float64), [1]
    )
    assert singleton.actuator_groups == 1
    assert singleton.balance_eligible_groups == 0
    assert singleton.mass_score_gradients.tolist() == [-1.0]
    multimode = train.canonical_replay_split_mass_balance_loss(
        torch.tensor([-2.0, -5.0], dtype=torch.float64), [2]
    )
    assert multimode.actuator_groups == 1
    assert multimode.balance_eligible_groups == 1
    assert float(torch.linalg.vector_norm(multimode.balance_score_gradients)) > 0


def test_identity_record_sets_are_exact() -> None:
    paired = {
        "runs": {arm: {"job_id": index} for index, arm in enumerate(train.ARMS)}
    }
    assert set(audit.identity_records(paired, train.PAIRED_MODE)) == set(train.ARMS)
    cells = {
        f"{arm}/s{seed}": {"job_id": seed}
        for arm in train.ARMS
        for seed in (43, 44, 45, 46, 47)
    }
    assert len(audit.identity_records({"cells": cells}, train.STAGE_B_MODE)) == 10
    cells.pop(next(iter(cells)))
    with pytest.raises(ValueError):
        audit.identity_records({"cells": cells}, train.STAGE_B_MODE)


def test_shell_tree_hash_matches_frozen_gate_identity() -> None:
    identity = json.loads(
        (ROOT / "var/artifacts/constructive_code_v5_gate_identity.json").read_text()
    )
    source_hash = identity["source_hash"]
    source_root = (
        ROOT
        / f"var/artifacts/source_snapshots/constructive_code_v5_source_{source_hash}/src"
    )
    assert launch.shell_tree_sha256(source_root) == source_hash


def test_dependency_chain_is_afterok_and_fail_closed() -> None:
    launcher = (
        ROOT / "ops/exp_scaling/launch_constructive_code_v5_experiment.py"
    ).read_text()
    assert "--dependency=afterok:{audit_job}" in launcher
    assert "stage_b_dependency_job_id" in launcher
    preparer = (
        ROOT
        / "ops/exp_scaling/launch_constructive_code_v5_paired_dependency_preparer.sh"
    ).read_text()
    assert '--dependency="afterok:$upstream_job"' in preparer
    assert "fail_closed" in preparer
