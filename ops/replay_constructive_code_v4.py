#!/usr/bin/env python3
"""Replay the frozen ConstructiveCode v4 slate under both official suites."""

from __future__ import annotations

from functools import partial
import os
from pathlib import Path

import audit_constructive_code_v2 as audit
import materialize_constructive_code_v4 as materialize_v4
import replay_constructive_code_v2 as base


ROOT = Path(os.environ.get("OAT_ZERO_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
base.V2_TASKS = materialize_v4.V4_TASKS
base.DEFAULT_INDEX = materialize_v4.base.DEFAULT_INDEX
base.PROTOCOL = materialize_v4.base.PROTOCOL
base.DEFAULT_SLATE = ROOT / "var/data/constructive_code_v4"
base.DEFAULT_REPLAYS = ROOT / "var/artifacts/constructive_code_v4_replays.jsonl"
base.DEFAULT_MANIFEST = ROOT / "var/artifacts/constructive_code_v4_replay_manifest.json"
base.DEFAULT_EQUIVALENCE = ROOT / "var/artifacts/constructive_code_v4_checker_equivalence.json"
base.DEFAULT_GATE_AUDIT = ROOT / "var/artifacts/constructive_code_v4_gate_audit.json"
base.DEFAULT_RUN_AUDIT = ROOT / "var/artifacts/constructive_code_v4_run_audit.json"
base.SOURCE_MANIFEST_SCHEMA = "constructive-code-slate-v4"
base.TASK_RECORD_SCHEMA = "constructive-code-task-v4"
base.RUN_STATUS = "v4_frozen_dual_suite_replay_with_task_level_fallback"
base.RUN_AUDIT_SCHEMA = "constructive-code-v4-run-audit-v1"
base.VERSION_LABEL = "constructive-code-v4-replay"
base.GATE_AUDIT_HASH_FIELD = "v4_gate_audit_sha256"
base.REQUIRED_PER_LABEL = 64
base.REQUIRE_V1_DISJOINT = False
base.V1_LEDGER_COUNT_FIELD = "available_v1_ledger_hash_count"
audit.REQUIRED_PER_LABEL = 64
base.build_v2_gate_audit = partial(
    audit.build_v2_gate_audit,
    expected_tasks=materialize_v4.V4_TASKS,
    required_suite_ids=audit.REQUIRED_SUITE_IDS,
    suite_policy="at_least_one",
    schema_version="constructive-code-v4-gate-audit-v1",
    decision_boundary=(
        "Pass freezes the preregistered primary suite for all 12 tasks and "
        "admits only split construction plus development-only "
        "Qwen2.5-Coder-0.5B viability sampling."
    ),
)


if __name__ == "__main__":
    base.main()
