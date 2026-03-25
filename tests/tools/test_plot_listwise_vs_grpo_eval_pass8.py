from __future__ import annotations

import json
from pathlib import Path

from tools import plot_listwise_vs_grpo_eval_pass8 as mod


def _write_payload(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_eval_pass8_plot_renders(tmp_path: Path) -> None:
    grpo_path = tmp_path / "grpo.json"
    listwise_path = tmp_path / "listwise.json"
    _write_payload(
        grpo_path,
        [
            {
                "task_name": "aime",
                "prompt_index": 0,
                "samples": [
                    {"reward": 1.0},
                    {"reward": 0.0},
                    {"reward": 0.0},
                    {"reward": 1.0},
                ],
            },
            {
                "task_name": "math",
                "prompt_index": 1,
                "samples": [
                    {"reward": 0.0},
                    {"reward": 0.0},
                    {"reward": 0.0},
                    {"reward": 0.0},
                ],
            },
        ],
    )
    _write_payload(
        listwise_path,
        [
            {
                "task_name": "aime",
                "prompt_index": 0,
                "samples": [
                    {"reward": 1.0, "token_count": 2, "logprob_sum": -0.2},
                    {"reward": 0.0, "token_count": 2, "logprob_sum": -0.8},
                    {"reward": 0.0, "token_count": 2, "logprob_sum": -0.9},
                    {"reward": 1.0, "token_count": 2, "logprob_sum": -0.3},
                ],
            },
            {
                "task_name": "math",
                "prompt_index": 1,
                "samples": [
                    {"reward": 1.0, "token_count": 2, "logprob_sum": -0.4},
                    {"reward": 0.0, "token_count": 2, "logprob_sum": -0.5},
                    {"reward": 0.0, "token_count": 2, "logprob_sum": -0.6},
                    {"reward": 0.0, "token_count": 2, "logprob_sum": -0.7},
                ],
            },
        ],
    )
    output = tmp_path / "eval.svg"
    summary = tmp_path / "eval.summary.json"
    groups_grpo = mod._load_groups(
        grpo_path,
        method="grpo",
        listwise_tau=0.5,
        listwise_beta=0.08,
        listwise_q_temperature=2.0,
        listwise_len_norm_ref=True,
    )
    groups_listwise = mod._load_groups(
        listwise_path,
        method="listwise",
        listwise_tau=0.5,
        listwise_beta=0.08,
        listwise_q_temperature=2.0,
        listwise_len_norm_ref=True,
    )
    payload = mod._plot_svg(
        grpo_groups=groups_grpo,
        listwise_groups=groups_listwise,
        output_path=output,
    )
    summary.write_text(json.dumps(payload), encoding="utf-8")
    assert output.exists()
    assert payload["grpo"]["total_groups"] == 2
    assert payload["grpo"]["informative_groups"] == 1
    assert payload["listwise"]["informative_groups"] == 2
