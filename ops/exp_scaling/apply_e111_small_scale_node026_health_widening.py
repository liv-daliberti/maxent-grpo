#!/usr/bin/env python3
"""Move three pending E111 small-scale jobs off drained node026."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e111_verified_support_discovery_mechanism_gate_three_scale as e111  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / e111.LEDGER
PROTOCOL = ROOT / (
    "paper/preregistration/"
    "e111_small_scale_node026_health_widening_20260818.md"
)
RECORD = ROOT / "var/artifacts/e111_small_scale_node026_health_widening.json"
JOB_IDS = (30674729, 30674733, 30674754)
OLD_NODES = "node026"
NEW_NODES = "node[202-204,403]"
HEALTH_NODES = ("node202", "node203", "node204", "node403")
VARIANT = e111.VARIANT


def now() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat()


def command(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def scheduler(job_id: int) -> str:
    result = command("scontrol", "show", "job", "-dd", "-o", str(job_id))
    if result.returncode != 0 or f"JobId={job_id}" not in result.stdout:
        raise RuntimeError(f"cannot inspect E111 job {job_id}")
    return result.stdout.strip()


def node(name: str) -> str:
    result = command("scontrol", "show", "node", "-o", name)
    if result.returncode != 0 or f"NodeName={name}" not in result.stdout:
        raise RuntimeError(f"cannot inspect {name}")
    return result.stdout.strip()


def validate_job(record: str, run: dict[str, Any], *, nodes: str) -> None:
    required = (
        f"JobId={int(run['job_id'])}",
        "JobState=PENDING",
        "Partition=lowprio",
        f"ReqNodeList={nodes}",
        "TresPerNode=gres/gpu:1",
        "MinMemoryNode=64G",
        "TimeLimit=00:45:00",
        f"OAT_ZERO_VARIANT={VARIANT}",
        f"OAT_ZERO_SEED={int(run['seed'])}",
        "OAT_ZERO_MAX_TRAIN=8",
        "OAT_ZERO_NUM_PROMPT_EPOCH=8",
        "OAT_ZERO_EVAL_PROMPT_INTERVAL=64",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_VERIFIED_SUPPORT_ADVANTAGE=1",
        "OAT_ZERO_SEMANTIC_SHANNON_VERIFIED_SUPPORT_INCLUDE_REPLAY_BANK=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=verified_likelihood_per_rollout",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=1",
        "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_VISITS=0",
        "OAT_ZERO_SOURCE_ROOT=" + str(
            ROOT / "var/artifacts/source_snapshots/e76_tuned_scale_2febfdc12e36650d/src"
        ),
        "OAT_ZERO_OPS_SNAPSHOT_ROOT=" + str(
            ROOT / "var/artifacts/source_snapshots/e76_tuned_scale_2febfdc12e36650d/ops"
        ),
    )
    missing = [value for value in required if value not in record]
    if missing:
        raise RuntimeError(f"job {run['job_id']} lacks frozen fields: {missing}")


def update(job_id: int, nodes: str) -> subprocess.CompletedProcess[str]:
    return command(
        "scontrol",
        "update",
        f"JobId={job_id}",
        f"NodeList={nodes}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        raise SystemExit("pass --apply after reviewing the frozen protocol")
    if RECORD.exists():
        raise SystemExit(f"refusing duplicate amendment: {RECORD}")
    for path in (LEDGER, PROTOCOL):
        if not path.is_file():
            raise SystemExit(f"required frozen input is absent: {path}")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    runs = {int(run["job_id"]): run for run in ledger.get("runs", [])}
    if set(JOB_IDS) - set(runs):
        raise RuntimeError("health-widening jobs are absent from E111 ledger")

    before = {str(job_id): scheduler(job_id) for job_id in JOB_IDS}
    for job_id in JOB_IDS:
        validate_job(before[str(job_id)], runs[job_id], nodes=OLD_NODES)
    old_node_health = node(OLD_NODES)
    if "DRAIN" not in old_node_health or "overheated" not in old_node_health:
        raise RuntimeError("node026 no longer has the frozen health trigger")
    healthy_nodes = {name: node(name) for name in HEALTH_NODES}
    for name, record in healthy_nodes.items():
        if any(marker in record for marker in ("DRAIN", "DOWN", "FAIL")):
            raise RuntimeError(f"replacement node is not healthy: {name}")

    payload: dict[str, Any] = {
        "schema": "e111_small_scale_node026_health_widening_v1",
        "recorded_before_at": now(),
        "protocol": str(PROTOCOL),
        "protocol_sha256": e111.digest(PROTOCOL),
        "ledger": str(LEDGER),
        "ledger_sha256": e111.digest(LEDGER),
        "job_ids": list(JOB_IDS),
        "old_required_nodes": OLD_NODES,
        "new_required_nodes": NEW_NODES,
        "time_limit": "00:45:00",
        "gres": "gpu:1",
        "changed_fields": ["ReqNodeList"],
        "same_job_ids": True,
        "jobs_requeued": False,
        "jobs_reset": False,
        "replacement_jobs_submitted": False,
        "environment_changed": False,
        "treatment_changed": False,
        "e112_paired_hardware_changed": False,
        "outcomes_inspected": False,
        "pointmaze": "excluded",
        "node026_before": old_node_health,
        "replacement_nodes_before": healthy_nodes,
        "before_scheduler_records": before,
        "installed": False,
    }
    changed: list[int] = []
    results: dict[str, dict[str, Any]] = {}
    try:
        for job_id in JOB_IDS:
            result = update(job_id, NEW_NODES)
            results[str(job_id)] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            if result.returncode != 0:
                raise RuntimeError(
                    f"node health widening failed for {job_id}: {result.stderr.strip()}"
                )
            changed.append(job_id)
        after = {str(job_id): scheduler(job_id) for job_id in JOB_IDS}
        for job_id in JOB_IDS:
            validate_job(after[str(job_id)], runs[job_id], nodes=NEW_NODES)
        payload.update(
            {
                "recorded_after_at": now(),
                "update_results": results,
                "after_scheduler_records": after,
                "installed": True,
            }
        )
        e111.e81.atomic_json(RECORD, payload)
    except Exception as exc:
        restores: dict[str, dict[str, Any]] = {}
        for job_id in changed:
            result = update(job_id, OLD_NODES)
            restores[str(job_id)] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        payload.update(
            {
                "failed_at": now(),
                "error": str(exc),
                "update_results": results,
                "restore_results": restores,
                "installed": False,
            }
        )
        e111.e81.atomic_json(RECORD, payload)
        raise
    print(
        f"[e111-small-health] installed=True jobs={len(JOB_IDS)} "
        f"nodes={NEW_NODES}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
