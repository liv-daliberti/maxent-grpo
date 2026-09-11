#!/usr/bin/env python3
"""Five-seed adapter for the qualified PointMaze repair-v2 trainer."""

from __future__ import annotations

import json

import train_point_maze_algorithm_repair_v2_direct_r5 as qualified


qualified.direct.base.SEEDS = (76531, 76532, 76533, 76534, 76535)


if __name__ == "__main__":
    receipt_path = qualified.direct.output_receipt_path()
    qualified.direct.base.main()
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload.update(
        schema="point-maze-algorithm-repair-receipt-v2",
        evaluation_request_seed_schedule=(
            "arm_seed_row_draw_sample_decision__checkpoint_invariant"
        ),
        secondary_post_outcome_repair=True,
        k16_aligned_algorithmic_gate=True,
        generic_guard_decision_adapter=True,
        json_loads_api_compatible=True,
        final_seed_cohort=True,
        evaluation_split="previously_untouched_eval",
    )
    qualified.direct.base.atomic(receipt_path, payload)
