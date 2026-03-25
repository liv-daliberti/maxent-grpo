from __future__ import annotations

import json
from pathlib import Path

from tools import plot_listwise_vs_grpo_distribution as dist
from tools import plot_listwise_vs_grpo_prompt_dynamics as mod


def test_prompt_dynamics_plot_renders(tmp_path: Path) -> None:
    grpo_records = [
        dist.GroupRecord(step=1, prompt_index=0, rewards=[1.0, 1.0, 0.0, 0.0], mass=[0.5, 0.5, 0.0, 0.0]),
        dist.GroupRecord(step=1, prompt_index=1, rewards=[0.0, 0.0, 0.0, 0.0], mass=[0.25, 0.25, 0.25, 0.25]),
        dist.GroupRecord(step=2, prompt_index=0, rewards=[1.0, 0.0, 0.0, 0.0], mass=[1.0, 0.0, 0.0, 0.0]),
    ]
    listwise_records = [
        dist.GroupRecord(step=1, prompt_index=0, rewards=[1.0, 1.0, 0.0, 0.0], mass=[0.25, 0.25, 0.25, 0.25]),
        dist.GroupRecord(step=1, prompt_index=1, rewards=[0.0, 0.0, 0.0, 0.0], mass=[0.25, 0.25, 0.25, 0.25]),
        dist.GroupRecord(step=2, prompt_index=0, rewards=[1.0, 0.0, 0.0, 0.0], mass=[0.4, 0.2, 0.2, 0.2]),
    ]
    summary = mod._plot_svg(
        grpo_steps=mod._aggregate_dynamics(grpo_records),
        listwise_steps=mod._aggregate_dynamics(listwise_records),
        output_path=tmp_path / "prompt_dynamics.svg",
    )
    assert (tmp_path / "prompt_dynamics.svg").exists()
    encoded = json.dumps(summary)
    assert "informative_share" in encoded
    assert "mean_incorrect_mass" in encoded
