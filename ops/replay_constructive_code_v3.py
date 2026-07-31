#!/usr/bin/env python3
"""Replay the frozen ConstructiveCode v3 slate under both official suites."""

from __future__ import annotations

from functools import partial
import os
from pathlib import Path

import audit_constructive_code_v2 as audit
import materialize_constructive_code_v3 as materialize_v3
import replay_constructive_code_v2 as base


ROOT = Path(
    os.environ.get("OAT_ZERO_REPO_ROOT", Path(__file__).resolve().parents[1])
).resolve()
PROTOCOL = materialize_v3.PROTOCOL

base.V2_TASKS = materialize_v3.V3_TASKS
base.DEFAULT_INDEX = materialize_v3.base.DEFAULT_INDEX
base.PROTOCOL = PROTOCOL
base.DEFAULT_SLATE = ROOT / "var/data/constructive_code_v3"
base.DEFAULT_REPLAYS = ROOT / "var/artifacts/constructive_code_v3_replays.jsonl"
base.DEFAULT_MANIFEST = (
    ROOT / "var/artifacts/constructive_code_v3_replay_manifest.json"
)
base.DEFAULT_EQUIVALENCE = (
    ROOT / "var/artifacts/constructive_code_v3_checker_equivalence.json"
)
base.DEFAULT_GATE_AUDIT = (
    ROOT / "var/artifacts/constructive_code_v3_gate_audit.json"
)
base.DEFAULT_RUN_AUDIT = (
    ROOT / "var/artifacts/constructive_code_v3_run_audit.json"
)
base.SOURCE_MANIFEST_SCHEMA = "constructive-code-slate-v3"
base.TASK_RECORD_SCHEMA = "constructive-code-task-v3"
base.RUN_STATUS = "v3_frozen_dual_suite_replay_with_task_level_fallback"
base.RUN_AUDIT_SCHEMA = "constructive-code-v3-run-audit-v1"
base.VERSION_LABEL = "constructive-code-v3-replay"
base.GATE_AUDIT_HASH_FIELD = "v3_gate_audit_sha256"
base.build_v2_gate_audit = partial(
    audit.build_v2_gate_audit,
    expected_tasks=materialize_v3.V3_TASKS,
    required_suite_ids=audit.REQUIRED_SUITE_IDS,
    suite_policy="at_least_one",
    schema_version="constructive-code-v3-gate-audit-v1",
    decision_boundary=(
        "Pass freezes the preregistered primary suite for all 12 tasks and "
        "admits only split construction plus development-only "
        "Qwen2.5-Coder-0.5B viability sampling."
    ),
)


if __name__ == "__main__":
    base.main()
