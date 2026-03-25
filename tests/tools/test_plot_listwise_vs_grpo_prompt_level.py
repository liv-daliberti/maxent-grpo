from __future__ import annotations

import json
from pathlib import Path

from tools import plot_listwise_vs_grpo_prompt_level as mod
from tools import plot_listwise_vs_grpo_distribution as dist


def test_prompt_level_plot_renders(tmp_path: Path) -> None:
    grpo_records = [
        dist.GroupRecord(step=1, prompt_index=0, rewards=[1.0, 1.0, 0.0, 0.0], mass=[0.6, 0.4, 0.0, 0.0]),
        dist.GroupRecord(step=1, prompt_index=1, rewards=[1.0, 0.0, 0.0, 0.0], mass=[1.0, 0.0, 0.0, 0.0]),
    ]
    listwise_records = [
        dist.GroupRecord(step=1, prompt_index=0, rewards=[1.0, 1.0, 0.0, 0.0], mass=[0.25, 0.25, 0.25, 0.25]),
        dist.GroupRecord(step=1, prompt_index=1, rewards=[1.0, 0.0, 0.0, 0.0], mass=[0.3, 0.2, 0.25, 0.25]),
    ]
    output = tmp_path / "prompt_level.svg"
    summary = mod._plot_svg(grpo_records=grpo_records, listwise_records=listwise_records, output_path=output)
    assert output.exists()
    assert output.stat().st_size > 0
    encoded = json.dumps(summary)
    assert summary["grpo"]["informative_groups"] == 2
    assert summary["listwise"]["informative_groups"] == 2
    assert "effective_rollouts_mean" in encoded
