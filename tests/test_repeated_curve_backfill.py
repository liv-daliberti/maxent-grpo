import importlib.util
import json
import statistics
from pathlib import Path

import pytest


_SCRIPT = (
    Path(__file__).parents[1]
    / "ops"
    / "exp_scaling"
    / "refresh_campaign_curves.py"
)
_SPEC = importlib.util.spec_from_file_location("refresh_campaign_curves", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
merge_repeated_terminal_evaluation = _MODULE.merge_repeated_terminal_evaluation


def _summary(path: Path, eval_seed: int, value: float) -> None:
    metrics = {
        "any_correct_at_k": value,
        "mean_at_k": value / 2,
        "mode_coverage_at_k": value / 4,
        "distinct_correct_modes_at_k": value * 2,
    }
    path.write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "alias": "grpo_s43",
                        "splits": {
                            "multi_answer": {
                                "attempts_path": f"attempts-{eval_seed}.json",
                                "checkpoint_path": "saved_models/step_00101",
                                "metrics": metrics,
                                "prompt_metrics_path": f"prompts-{eval_seed}.json",
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_merge_repeated_terminal_evaluation_retains_draws_and_trace_paths(tmp_path):
    curve_path = tmp_path / "curve.json"
    curve_path.write_text(
        json.dumps(
            [
                {
                    "arm": "grpo",
                    "seed": 43,
                    "step": 0,
                    "training_passes": 0.0,
                    "split": "multi_answer",
                    "pass8": 0.1,
                },
                {
                    "arm": "grpo",
                    "seed": 43,
                    "step": 100,
                    "training_passes": 5.0,
                    "split": "multi_answer",
                    "pass8": 0.2,
                },
            ]
        ),
        encoding="utf-8",
    )
    summary_paths = []
    values = [0.3, 0.4, 0.5, 0.6]
    for eval_seed, value in zip(range(1001, 1005), values):
        path = tmp_path / f"experiment_e{eval_seed}_coverage_summary.json"
        _summary(path, eval_seed, value)
        summary_paths.append(path)
    greedy_path = tmp_path / "experiment_greedy_coverage_summary.json"
    _summary(greedy_path, 0, 0.7)

    merged = merge_repeated_terminal_evaluation(
        curve_path,
        summary_paths,
        target_arm="grpo",
        greedy_summary_path=greedy_path,
    )

    rows = json.loads(curve_path.read_text(encoding="utf-8"))
    assert merged == 1
    assert rows[0]["pass8"] == 0.1
    terminal = rows[1]
    assert terminal["pass8"] == pytest.approx(0.45)
    assert terminal["pass8_draws"] == values
    assert terminal["pass8_draw_std"] == pytest.approx(statistics.stdev(values))
    assert terminal["pass8_draw_se"] == pytest.approx(
        statistics.stdev(values) / 2
    )
    assert terminal["pass8_draw_min"] == 0.3
    assert terminal["pass8_draw_max"] == 0.6
    assert terminal["repeated_eval_seeds"] == [1001, 1002, 1003, 1004]
    assert terminal["repeated_eval_trace_count"] == 4
    assert terminal["greedy"] == 0.7
    assert terminal["greedy_eval_source"] == "external_deterministic_terminal_backfill"
    assert [
        trace["attempts_path"] for trace in terminal["repeated_eval_traces"]
    ] == [
        "attempts-1001.json",
        "attempts-1002.json",
        "attempts-1003.json",
        "attempts-1004.json",
    ]


def test_merge_repeated_terminal_evaluation_rejects_incomplete_seed_set(tmp_path):
    curve_path = tmp_path / "curve.json"
    curve_path.write_text(
        json.dumps(
            [
                {
                    "arm": "grpo",
                    "seed": 43,
                    "step": 100,
                    "training_passes": 5.0,
                    "split": "multi_answer",
                }
            ]
        ),
        encoding="utf-8",
    )
    path = tmp_path / "experiment_e1001_coverage_summary.json"
    _summary(path, 1001, 0.3)

    with pytest.raises(ValueError, match="expected fixed seeds 1001--1004"):
        merge_repeated_terminal_evaluation(curve_path, [path], target_arm="grpo")
