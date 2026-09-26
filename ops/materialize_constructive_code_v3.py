#!/usr/bin/env python3
"""Materialize the frozen 12-task, problem-disjoint ConstructiveCode v3 slate."""

from __future__ import annotations

import os
from pathlib import Path

import materialize_constructive_code_v2 as base


ROOT = Path(
    os.environ.get("OAT_ZERO_REPO_ROOT", Path(__file__).resolve().parents[1])
).resolve()
PROTOCOL = Path(
    os.environ.get(
        "OAT_ZERO_CONSTRUCTIVE_V3_PROTOCOL",
        ROOT
        / "paper/preregistration/constructive_code_executable_slate_v3_20260729.md",
    )
).resolve()
V3_TASKS = {
    "327_B": ("ordered_sequence", "fixed_integer_sequence_v1"),
    "359_B": ("ordered_sequence", "fixed_integer_sequence_v1"),
    "361_B": ("ordered_sequence", "sentinel_integer_sequence_v1"),
    "659_C": ("unordered_set", "counted_integer_set_v1"),
    "988_A": ("unordered_set", "status_integer_set_v1"),
    "1294_C": ("unordered_set", "multi_case_status_integer_set_v1"),
    "1208_C": ("assignment", "matrix_assignment_v1"),
    "1283_C": ("assignment", "implicit_assignment_v1"),
    "1408_A": ("assignment", "multi_case_implicit_assignment_v1"),
    "1102_B": ("unordered_partition", "status_label_partition_v1"),
    "1399_D": ("unordered_partition", "multi_case_label_partition_v1"),
    "149_C": ("unordered_partition", "two_group_partition_v1"),
}
V3_SPLIT_ASSIGNMENT = {
    "327_B": "train",
    "659_C": "train",
    "1208_C": "train",
    "1102_B": "train",
    "359_B": "development",
    "988_A": "development",
    "1283_C": "development",
    "1399_D": "development",
    "361_B": "evaluation",
    "1294_C": "evaluation",
    "1408_A": "evaluation",
    "149_C": "evaluation",
}


base.DEFAULT_INDEX = Path(
    os.environ.get(
        "OAT_ZERO_CONSTRUCTIVE_V3_INDEX",
        ROOT / "var/artifacts/constructive_code_candidate_source_index.json",
    )
).resolve()
base.DEFAULT_V1 = Path(
    os.environ.get(
        "OAT_ZERO_CONSTRUCTIVE_V3_V1_ROOT",
        ROOT / "var/data/constructive_code_review_slate_v1",
    )
).resolve()
base.DEFAULT_OUTPUT = ROOT / "var/data/constructive_code_v3"
base.PROTOCOL = PROTOCOL
base.SLATE_SCHEMA = "constructive-code-slate-v3"
base.TASK_SCHEMA = "constructive-code-task-v3"
base.VERSION_LABEL = "constructive-v3"
base.LOGICAL_V1_ROOT = "var/data/constructive_code_review_slate_v1"
base.SPLIT_ASSIGNMENT = V3_SPLIT_ASSIGNMENT
base.V2_TASKS = V3_TASKS


if __name__ == "__main__":
    base.main()
