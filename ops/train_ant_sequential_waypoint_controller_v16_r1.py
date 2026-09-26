#!/usr/bin/env python3
"""Prelaunch metric-semantics repair for sequential controller v16."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import train_ant_sequential_waypoint_controller_v16 as v16


_evaluate = v16._evaluate


def evaluate(*args, **kwargs):
    result = _evaluate(*args, **kwargs)
    successful = [
        row for row in result["episodes"] if row["success"]
    ]
    segment_steps = [
        step for row in successful for step in row["segment_steps"]
    ]
    total_steps = [row["steps"] for row in successful]
    result["summary"]["median_success_steps"] = (
        float(np.median(segment_steps)) if segment_steps else None
    )
    result["summary"]["median_success_total_steps"] = (
        float(np.median(total_steps)) if total_steps else None
    )
    return result


v16.controller._evaluate = evaluate
v16.controller.TRAINING_SOURCE_PATH = Path(__file__).resolve()


if __name__ == "__main__":
    v16.controller.main()
