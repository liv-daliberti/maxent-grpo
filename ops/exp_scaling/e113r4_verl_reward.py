#!/usr/bin/env python3
"""Exact ModeBench correctness adapter for verl's DAPORewardManager."""

from __future__ import annotations

import json
from typing import Any

from oat_drgrpo.math_grader import boxed_reward_fn
from oat_drgrpo.pantry_support_action import decode_pantry_support_mask


ALLOWED_SOURCES = {
    "modebench/graph_coloring",
    "modebench/countdown",
    "modebench/python_factors",
    "modebench/mathir",
    "modebench/pantry_plan",
}


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Any = None,
) -> dict[str, float]:
    """Return binary task accuracy separately from DAPO's shaped token reward."""

    del extra_info
    if str(data_source) not in ALLOWED_SOURCES:
        raise ValueError(f"unexpected E113-R4 data source: {data_source!r}")
    verifier_input = solution_str
    if str(data_source) == "modebench/pantry_plan":
        specification = json.loads(str(ground_truth))
        verifier_input = decode_pantry_support_mask(solution_str, specification)
    metadata, score = boxed_reward_fn(verifier_input, ground_truth, fast=True)
    accuracy = float(score)
    if accuracy not in (0.0, 1.0):
        raise ValueError(f"ModeBench verifier returned non-binary score {accuracy}")
    return {
        "score": accuracy,
        "acc": accuracy,
        "formatted": float(bool(metadata.get("formatted", False))),
    }

