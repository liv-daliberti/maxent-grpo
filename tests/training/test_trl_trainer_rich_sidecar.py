from __future__ import annotations

import json

from maxent_grpo.training import trl_trainer as mod


def test_build_rich_rollout_rows_uses_q_mass_when_available() -> None:
    columns, rows = mod._build_rich_rollout_rows(
        step=3,
        group_size=2,
        prompt_texts=["p1", "p1", "p2", "p2"],
        completion_texts=["a", "b", "c", "d"],
        rewards=[1.0, 0.0, 0.0, 1.0],
        advantages=[0.5, -0.5, -0.5, 0.5],
        q_values=[0.7, 0.3, 0.4, 0.6],
    )
    col_idx = {name: idx for idx, name in enumerate(columns)}
    assert rows[0][col_idx["prompt_index"]] == 0
    assert rows[0][col_idx["completion_index"]] == 0
    assert rows[0][col_idx["reward_total"]] == 1.0
    assert rows[0][col_idx["q_mass"]] == 0.7
    assert rows[0][col_idx["update_mass_proxy"]] == 0.7
    assert rows[1][col_idx["reward_rank_desc"]] == 2


def test_write_rich_rollout_sidecar_persists_json(tmp_path) -> None:
    path = mod._write_rich_rollout_sidecar(
        output_dir=str(tmp_path),
        table_key="rich_completions",
        step=4,
        columns=["step", "prompt_index"],
        rows=[[4, 0], [4, 1]],
    )
    assert path is not None
    payload = json.loads(
        (tmp_path / "rich_completions" / "rich_completions_step_000004.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["columns"] == ["step", "prompt_index"]
    assert payload["data"] == [[4, 0], [4, 1]]
