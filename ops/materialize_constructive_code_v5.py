#!/usr/bin/env python3
"""Materialize the prospective 12-task ConstructiveCode v5 source slate."""

from __future__ import annotations

import os
from pathlib import Path

import materialize_constructive_code_v4 as v4


ROOT = Path(os.environ.get("OAT_ZERO_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
base = v4.base
base.DEFAULT_INDEX = Path(os.environ.get(
    "OAT_ZERO_CONSTRUCTIVE_V5_INDEX",
    ROOT / "var/artifacts/constructive_code_candidate_source_index.json",
)).resolve()
base.DEFAULT_V1 = Path(os.environ.get(
    "OAT_ZERO_CONSTRUCTIVE_V5_V1_ROOT",
    ROOT / "var/data/constructive_code_review_slate_v1",
)).resolve()
base.DEFAULT_OUTPUT = ROOT / "var/data/constructive_code_v5"
base.PROTOCOL = Path(os.environ.get(
    "OAT_ZERO_CONSTRUCTIVE_V5_PROTOCOL",
    ROOT / "paper/preregistration/constructive_code_executable_slate_v5_20260730.md",
)).resolve()
base.SLATE_SCHEMA = "constructive-code-slate-v5"
base.TASK_SCHEMA = "constructive-code-task-v5"
base.VERSION_LABEL = "constructive-v5"
base.REPLAYS_PER_LABEL = 48
base.EXCLUDE_V1_HASHES = False
base.V1_LEDGER_COUNT_FIELD = "available_v1_ledger_hash_count"
base.SELECTION_DESCRIPTION = (
    "explicit Python-3 labels from full pinned source, v1 overlap allowed and "
    "reported, unique SHA-256 ascending, 48 per known label"
)

V5_TASKS = v4.V4_TASKS
V5_SPLIT_ASSIGNMENT = v4.V4_SPLIT_ASSIGNMENT


if __name__ == "__main__":
    base.main()
