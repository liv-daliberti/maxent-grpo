from __future__ import annotations

import json
from pathlib import Path

from tools import plot_full_run_interim_training_compare as mod


def test_load_series_and_render_dashboard(tmp_path: Path) -> None:
    grpo_log = tmp_path / "grpo.log"
    listwise_log = tmp_path / "listwise.log"
    grpo_log.write_text(
        "\n".join(
            [
                "ignored",
                "{'train/loss/total': 0.1, 'reward': 0.21, 'completions/mean_length': 1080.0, 'completions/clipped_ratio': 0.20, 'frac_reward_zero_std': 0.4}",
                "{'train/loss/total': 0.2, 'reward': 0.24, 'completions/mean_length': 1050.0, 'completions/clipped_ratio': 0.19, 'frac_reward_zero_std': 0.3}",
            ]
        )
    )
    listwise_log.write_text(
        "\n".join(
            [
                "{'train/loss/total': 1.3, 'reward': 0.23, 'completions/mean_length': 1090.0, 'completions/clipped_ratio': 0.21, 'frac_reward_zero_std': 0.25, 'weight_entropy': 1.95, 'maxent/listwise_active_group_frac': 0.75}",
                "{'train/loss/total': 1.2, 'reward': 0.26, 'completions/mean_length': 1070.0, 'completions/clipped_ratio': 0.20, 'frac_reward_zero_std': 0.15, 'weight_entropy': 1.90, 'maxent/listwise_active_group_frac': 0.85}",
            ]
        )
    )

    grpo_points = mod._load_series(grpo_log)
    listwise_points = mod._load_series(listwise_log)

    assert [point.step for point in grpo_points] == [1, 2]
    assert grpo_points[-1].informative_share == 0.7
    assert listwise_points[-1].weight_entropy == 1.9
    assert listwise_points[-1].active_group_frac == 0.85

    output = tmp_path / "dashboard.svg"
    args = type(
        "Args",
        (),
        {
            "grpo_log": grpo_log,
            "listwise_log": listwise_log,
            "grpo_label": "GRPO",
            "listwise_label": "Listwise",
        },
    )()
    mod._plot_svg("Test Dashboard", grpo_points, listwise_points, output, args)
    assert output.exists()
    assert output.stat().st_size > 0

    summary = mod._summary_payload("Test Dashboard", grpo_points, listwise_points, args)
    encoded = json.dumps(summary)
    assert summary["grpo"]["latest_step"] == 2
    assert summary["listwise"]["latest_reward"] == 0.26
    assert "Held-out SEED eval" in encoded
