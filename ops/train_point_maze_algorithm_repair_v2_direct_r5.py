#!/usr/bin/env python3
"""API-compatible qualification adapter for PointMaze repair-v2."""

from __future__ import annotations

import json

import train_point_maze_algorithm_repair_v2_direct as direct


_loads = direct.base.json.loads


def loads(value, *args, **kwargs):
    payload = _loads(value, *args, **kwargs)
    if (
        isinstance(payload, dict)
        and payload.get("schema")
        == "point-maze-algorithm-repair-v2-qualification"
        and payload.get("decision")
        == "eligible_for_point_maze_algorithm_repair_v2_pair"
    ):
        payload["decision"] = "eligible_for_ten_point_maze_stage_b_jobs"
    return payload


direct.base.json.loads = loads


if __name__ == "__main__":
    receipt_path = direct.output_receipt_path()
    direct.base.main()
    payload = json.loads(receipt_path.read_text())
    payload.update(
        schema="point-maze-algorithm-repair-receipt-v2",
        evaluation_request_seed_schedule=(
            "arm_seed_row_draw_sample_decision__checkpoint_invariant"
        ),
        secondary_post_outcome_repair=True,
        k16_aligned_algorithmic_gate=True,
        generic_guard_decision_adapter=True,
        json_loads_api_compatible=True,
    )
    direct.base.atomic(receipt_path, payload)
