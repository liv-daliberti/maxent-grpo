from __future__ import annotations

import json
from pathlib import Path

from tools import plot_listwise_vs_grpo_distribution as mod


def _write_run(
    root: Path,
    run_dir_name: str,
    run_name: str,
    rows: list[list[object]] | None,
    *,
    table_prefix: str = "completions",
    output_dir: Path | None = None,
) -> Path:
    run_dir = root / run_dir_name
    files_dir = run_dir / "files"
    table_dir = files_dir / "media" / "table"
    table_dir.mkdir(parents=True, exist_ok=True)
    args = ["python", "train.py", "--run_name", run_name]
    if output_dir is not None:
        args.extend(["--output_dir", str(output_dir)])
    metadata = {"args": args}
    (files_dir / "wandb-metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    if rows is not None:
        table = {
            "columns": [
                "step",
                "prompt_index",
                "completion_index",
                "group_size",
                "reward_rank_desc",
                "prompt",
                "completion",
                "reward_total",
                "advantage",
                "q_mass",
                "update_weight_raw",
                "update_mass_proxy",
                "reward/seed_paper_boxed_accuracy_reward_math",
            ],
            "data": rows,
        }
        (table_dir / f"{table_prefix}_25_fake.table.json").write_text(
            json.dumps(table), encoding="utf-8"
        )
    return run_dir


def test_distribution_plot_script_renders_from_local_tables(tmp_path: Path) -> None:
    wandb_root = tmp_path / "wandb"
    _write_run(
        wandb_root,
        "run-grpo",
        "test-grpo",
        [
            [25, 0, 0, 2, 1, "p1", "a", 1.0, 0.5, 0.7, 0.5, 1.0, 1.0],
            [25, 0, 1, 2, 2, "p1", "b", 0.0, -0.5, 0.3, -0.5, 0.0, 0.0],
            [25, 1, 0, 2, 1, "p2", "c", 1.0, 0.5, 0.7, 0.5, 1.0, 1.0],
            [25, 1, 1, 2, 2, "p2", "d", 0.0, -0.5, 0.3, -0.5, 0.0, 0.0],
        ],
    )
    _write_run(
        wandb_root,
        "run-listwise",
        "test-listwise",
        [
            [25, 0, 0, 2, 1, "p1", "a", 1.0, 0.5, 0.62, 0.62, 0.62, 1.0],
            [25, 0, 1, 2, 2, "p1", "b", 0.0, -0.5, 0.38, 0.38, 0.38, 0.0],
            [25, 1, 0, 2, 1, "p2", "c", 1.0, 0.5, 0.62, 0.62, 0.62, 1.0],
            [25, 1, 1, 2, 2, "p2", "d", 0.0, -0.5, 0.38, 0.38, 0.38, 0.0],
        ],
    )
    output_path = tmp_path / "dist.svg"
    summary_path = tmp_path / "summary.json"
    grpo_dir = mod._resolve_run("test-grpo", wandb_root)
    listwise_dir = mod._resolve_run("test-listwise", wandb_root)
    grpo_records, grpo_source = mod._load_records(
        grpo_dir,
        label="grpo",
        q_temperature=2.0,
        include_neutral_groups=False,
        max_groups=0,
    )
    listwise_records, listwise_source = mod._load_records(
        listwise_dir,
        label="listwise",
        q_temperature=2.0,
        include_neutral_groups=False,
        max_groups=0,
    )
    grpo_result = mod._analyze("grpo", grpo_records, grpo_records, grpo_source)
    listwise_result = mod._analyze(
        "listwise", listwise_records, listwise_records, listwise_source
    )
    mod._plot(grpo_result, listwise_result, output_path)
    summary = {
        "grpo": {
            "total_groups": grpo_result.total_groups,
            "groups_used": grpo_result.groups_used,
            "mass_source": grpo_result.mass_source,
        },
        "listwise": {
            "total_groups": listwise_result.total_groups,
            "groups_used": listwise_result.groups_used,
            "mass_source": listwise_result.mass_source,
        },
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert grpo_result.total_groups == 2
    assert grpo_result.groups_used == 2
    assert listwise_result.total_groups == 2
    assert listwise_result.groups_used == 2


def test_group_record_falls_back_to_q_targets_for_older_listwise_rows() -> None:
    rows = [
        {
            "step": 10,
            "prompt": "p",
            "reward_total": 1.0,
            "advantage": 0.5,
        },
        {
            "step": 10,
            "prompt": "p",
            "reward_total": 0.0,
            "advantage": -0.5,
        },
    ]
    parsed = mod._group_record_from_rows(
        rows,
        label="listwise",
        q_temperature=2.0,
        include_neutral_groups=False,
    )
    assert parsed is not None
    record, source = parsed
    assert source == "softmax_reward_fallback"
    assert len(record.mass) == 2
    assert record.mass[0] > record.mass[1]


def test_completion_tables_prefers_rich_completions_files(tmp_path: Path) -> None:
    wandb_root = tmp_path / "wandb"
    run_dir = _write_run(
        wandb_root,
        "run-grpo",
        "test-grpo-rich",
        [
            [25, 0, 0, 2, 1, "p1", "a", 1.0, 0.5, 0.7, 0.5, 1.0, 1.0],
            [25, 0, 1, 2, 2, "p1", "b", 0.0, -0.5, 0.3, -0.5, 0.0, 0.0],
        ],
        table_prefix="rich_completions",
    )
    tables = mod._completion_tables(run_dir)
    assert len(tables) == 1
    assert tables[0].name.startswith("rich_completions_")


def test_completion_tables_falls_back_to_local_sidecars(tmp_path: Path) -> None:
    wandb_root = tmp_path / "wandb"
    output_dir = tmp_path / "outputs" / "run-a"
    sidecar_dir = output_dir / "rich_completions"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    _write_run(
        wandb_root,
        "run-sidecar",
        "test-sidecar",
        None,
        output_dir=output_dir,
    )
    payload = {
        "columns": [
            "step",
            "prompt_index",
            "completion_index",
            "group_size",
            "reward_rank_desc",
            "prompt",
            "completion",
            "reward_total",
            "advantage",
            "q_mass",
            "update_weight_raw",
            "update_mass_proxy",
        ],
        "data": [
            [5, 0, 0, 2, 1, "p", "a", 1.0, 0.5, 0.7, 0.6, 0.6],
            [5, 0, 1, 2, 2, "p", "b", 0.0, -0.5, 0.3, 0.4, 0.4],
        ],
    }
    (sidecar_dir / "rich_completions_step_000005.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    run_dir = wandb_root / "run-sidecar"
    tables = mod._completion_tables(run_dir)
    assert len(tables) == 1
    assert tables[0].name == "rich_completions_step_000005.json"


def test_completion_tables_prefers_more_complete_sidecars(tmp_path: Path) -> None:
    wandb_root = tmp_path / "wandb"
    output_dir = tmp_path / "outputs" / "run-b"
    sidecar_dir = output_dir / "rich_completions"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _write_run(
        wandb_root,
        "run-sidecar-rich",
        "test-sidecar-rich",
        [
            [1, 0, 0, 2, 1, "p", "a", 1.0, 0.5, 0.7, 0.6, 0.6],
            [1, 0, 1, 2, 2, "p", "b", 0.0, -0.5, 0.3, 0.4, 0.4],
        ],
        table_prefix="rich_completions",
        output_dir=output_dir,
    )
    payload = {
        "columns": [
            "step",
            "prompt_index",
            "completion_index",
            "group_size",
            "reward_rank_desc",
            "prompt",
            "completion",
            "reward_total",
            "advantage",
            "q_mass",
            "update_weight_raw",
            "update_mass_proxy",
        ],
        "data": [
            [1, 0, 0, 2, 1, "p", "a", 1.0, 0.5, 0.7, 0.6, 0.6],
            [1, 0, 1, 2, 2, "p", "b", 0.0, -0.5, 0.3, 0.4, 0.4],
        ],
    }
    (sidecar_dir / "rich_completions_step_000001.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    (sidecar_dir / "rich_completions_step_000002.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    tables = mod._completion_tables(run_dir)
    assert len(tables) == 2
    assert tables[0].name == "rich_completions_step_000001.json"
