from __future__ import annotations

import json
from pathlib import Path

from tools import plot_listwise_vs_grpo_over_time as mod


def _write_sidecar_run(root: Path, run_dir_name: str, run_name: str, output_dir: Path) -> Path:
    run_dir = root / run_dir_name
    files_dir = run_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "args": [
            "python",
            "train.py",
            "--run_name",
            run_name,
            "--output_dir",
            str(output_dir),
        ]
    }
    (files_dir / "wandb-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return run_dir


def _write_sidecar(output_dir: Path, step: int, rewards: list[float], masses: list[float]) -> None:
    sidecar_dir = output_dir / "rich_completions"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
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
            [step, 0, idx, len(rewards), idx + 1, "p", f"c{idx}", reward, reward, None, None, masses[idx]]
            for idx, reward in enumerate(rewards)
        ],
    }
    (sidecar_dir / f"rich_completions_step_{step:06d}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_over_time_plot_renders(tmp_path: Path) -> None:
    wandb_root = tmp_path / "wandb"
    grpo_out = tmp_path / "outputs" / "grpo"
    listwise_out = tmp_path / "outputs" / "listwise"
    _write_sidecar_run(wandb_root, "run-grpo", "over-time-grpo", grpo_out)
    _write_sidecar_run(wandb_root, "run-listwise", "over-time-listwise", listwise_out)
    _write_sidecar(grpo_out, 1, [1.0, 0.0], [1.0, 0.0])
    _write_sidecar(grpo_out, 2, [0.0, 0.0], [float("nan"), float("nan")])
    _write_sidecar(listwise_out, 1, [1.0, 0.0], [0.7, 0.3])
    _write_sidecar(listwise_out, 2, [1.0, 1.0], [0.5, 0.5])

    grpo_dir = mod.dist._resolve_run("over-time-grpo", wandb_root)
    listwise_dir = mod.dist._resolve_run("over-time-listwise", wandb_root)
    grpo_records, _ = mod.dist._load_records(
        grpo_dir,
        label="grpo",
        q_temperature=2.0,
        include_neutral_groups=True,
        max_groups=0,
    )
    listwise_records, _ = mod.dist._load_records(
        listwise_dir,
        label="listwise",
        q_temperature=2.0,
        include_neutral_groups=True,
        max_groups=0,
    )
    grpo_steps = mod._aggregate_steps(grpo_records)
    listwise_steps = mod._aggregate_steps(listwise_records)
    output = tmp_path / "over_time.svg"
    mod._plot_svg(grpo_steps, listwise_steps, output)
    assert output.exists()
    assert output.stat().st_size > 0
    assert len(grpo_steps) == 2
    assert grpo_steps[1].neutral_groups == 1
    assert len(listwise_steps) == 2
