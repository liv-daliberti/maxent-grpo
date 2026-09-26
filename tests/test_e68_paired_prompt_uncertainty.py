import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v1 = _load(
    "e68_paired_prompt_uncertainty_v1_test",
    "ops/exp_scaling/analyze_e68_paired_prompt_uncertainty.py",
)
v2 = _load(
    "e68_paired_prompt_uncertainty_v2_test",
    "ops/exp_scaling/analyze_e68_paired_prompt_uncertainty_v2.py",
)


def _prompt(index, *, mean, any_correct, distinct):
    return {
        "prompt_index": index,
        "prompt": f"problem {index}",
        "reference": f"reference {index}",
        "metrics": {
            "mean_at_k": mean,
            "any_correct_at_k": any_correct,
            "distinct_correct_modes_at_k": distinct,
        },
    }


def _records(*, greedy, sampled):
    records = [
        {
            "step": 0,
            "evaluation_kind": "deterministic_greedy_trace_neutral",
            "draw_index": None,
            "seed": 0,
            "sample_count": 1,
            "prompts": [
                _prompt(
                    index,
                    mean=value,
                    any_correct=value,
                    distinct=value,
                )
                for index, value in enumerate(greedy)
            ],
        }
    ]
    for draw in range(4):
        records.append(
            {
                "step": 0,
                "evaluation_kind": "fixed_seed_sampled_k_neutral",
                "draw_index": draw,
                "seed": 1000 + draw,
                "sample_count": 8,
                "prompts": [
                    _prompt(
                        index,
                        mean=value,
                        any_correct=float(value > 0),
                        distinct=2 * float(value > 0),
                    )
                    for index, value in enumerate(sampled)
                ],
            }
        )
    return records


def _primary(path, scores):
    path.write_text(
        json.dumps(
            [
                {
                    "problem": f"problem {index}",
                    "reference": f"reference {index}",
                    "scores": [score],
                }
                for index, score in enumerate(scores)
            ]
        ),
        encoding="utf-8",
    )


def test_crossed_bootstrap_keeps_constant_paired_difference():
    difference = np.full((3, 11), 0.25)
    lower, upper = v1._crossed_bootstrap(
        difference,
        seed=123,
        replicates=100,
    )
    assert lower == 0.25
    assert upper == 0.25


def test_v2_greedy_uses_primary_source_and_reports_repeat_sensitivity(
    tmp_path,
):
    control_primary = tmp_path / "control.json"
    repair_primary = tmp_path / "repair.json"
    _primary(control_primary, [0, 0])
    _primary(repair_primary, [1, 0])

    vectors, sensitivity = v2._paired_prompt_vectors_v2(
        _records(greedy=[0, 0], sampled=[0, 0]),
        _records(greedy=[0, 0], sampled=[0.25, 0]),
        control_primary=control_primary,
        repair_primary=repair_primary,
        label="synthetic",
    )

    assert vectors["greedy"].tolist() == [1.0, 0.0]
    assert vectors["mean8"].tolist() == [0.25, 0.0]
    assert vectors["pass8"].tolist() == [1.0, 0.0]
    assert vectors["distinct8"].tolist() == [2.0, 0.0]
    assert sensitivity["control"]["different_prompt_scores"] == 0
    assert sensitivity["repair"]["different_prompt_scores"] == 1
