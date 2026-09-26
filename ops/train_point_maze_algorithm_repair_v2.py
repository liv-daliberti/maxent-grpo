#!/usr/bin/env python3
"""Train PointMaze repair-v2 with checkpoint-invariant evaluation draws."""

from __future__ import annotations

import json

import train_point_maze_algorithm_repair_v1 as v1


v1.base.SEEDS = (76521,)
v1.base.EXPECTED_FAMILIES = (
    "cross9_balanced",
    "upper_offset9_balanced",
    "block11_balanced",
    "bar11_balanced",
)
_evaluate = v1.base.evaluate


def evaluate(**kwargs):
    result = _evaluate(**kwargs)
    result["schema"] = "point-maze-algorithm-repair-evaluation-v2"
    return result


v1.base.evaluate = evaluate


if __name__ == "__main__":
    receipt_path = v1.output_receipt_path()
    v1.base.main()
    payload = json.loads(receipt_path.read_text())
    payload.update(
        schema="point-maze-algorithm-repair-receipt-v2",
        evaluation_request_seed_schedule=(
            "arm_seed_row_draw_sample_decision__checkpoint_invariant"
        ),
        secondary_post_outcome_repair=True,
        k16_aligned_algorithmic_gate=True,
    )
    v1.base.atomic(receipt_path, payload)
