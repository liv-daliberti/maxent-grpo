#!/usr/bin/env python3
"""Conditionally move untouched Qwen-3B E104 jobs to the proven A6000 pool."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e104_group_centered_semantic_repair_three_scale as e104  # noqa: E402
import launch_e104_qwen3_a6000_capacity_preflight as preflight  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / (
    "paper/preregistration/"
    "e104_qwen3_a6000_placement_amendment_20260817.md"
)
E104_LEDGER = ROOT / e104.LEDGER
PREFLIGHT_LEDGER = ROOT / preflight.LEDGER
PREFLIGHT_AUDIT = ROOT / preflight.AUDIT
OUT = ROOT / "var/artifacts/e104_qwen3_a6000_placement_amendment.json"
SCALE = "qwen3b"
TARGET_JOB_IDS = (30637795, 30637796, 30637797, 30637798, 30637799)
ORIGINAL_PARTITION = "mltheory"
ORIGINAL_NODE_LIST = "node302"
ORIGINAL_GRES = "gres/gpu:a100:1"
TARGET_PARTITION = preflight.PARTITION
TARGET_NODE_LIST = preflight.NODE_LIST
TARGET_GRES = "gres/gpu:a6000:1"
ACCOUNT = "mltheory"
TIME_LIMIT = "02:00:00"


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required artifact is absent: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def scheduler_record(job_id: int) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", str(job_id)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot inspect E104 job {job_id}")
    return result.stdout.strip()


def require_record(record: str, job_id: int, needles: tuple[str, ...]) -> None:
    missing = [needle for needle in needles if needle not in record]
    if missing:
        raise RuntimeError(f"E104 job {job_id} lacks scheduler fields {missing}")


def validate_preflight() -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = load(PREFLIGHT_LEDGER)
    audit = load(PREFLIGHT_AUDIT)
    if not audit.get("complete") or not audit.get("passed"):
        raise RuntimeError("Qwen-3B A6000 capacity preflight has not passed")
    if audit.get("outcome_metrics_inspected") is not False:
        raise RuntimeError("capacity preflight audit was not outcome-blind")
    if int(audit.get("job_id", -1)) != int(ledger.get("job_id", -2)):
        raise RuntimeError("capacity preflight audit/ledger job mismatch")
    if not str(audit.get("scheduler_state", "")).startswith("COMPLETED"):
        raise RuntimeError("capacity preflight did not complete normally")
    if audit.get("violations"):
        raise RuntimeError("capacity preflight audit contains violations")
    return ledger, audit


def target_runs(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    runs = [run for run in ledger.get("runs", []) if run.get("scale") == SCALE]
    ids = tuple(sorted(int(run["job_id"]) for run in runs))
    if ids != TARGET_JOB_IDS or len(runs) != 5:
        raise RuntimeError(f"unexpected Qwen-3B E104 job set: {ids}")
    if ledger.get("pointmaze") != "excluded":
        raise RuntimeError("E104 ledger does not exclude PointMaze")
    return sorted(runs, key=lambda run: int(run["job_id"]))


def validate_original(run: dict[str, Any], record: str) -> None:
    job_id = int(run["job_id"])
    require_record(
        record,
        job_id,
        (
            "JobState=PENDING",
            "RunTime=00:00:00",
            f"Partition={ORIGINAL_PARTITION}",
            f"Account={ACCOUNT}",
            f"ReqNodeList={ORIGINAL_NODE_LIST}",
            f"TimeLimit={TIME_LIMIT}",
            f"TresPerNode={ORIGINAL_GRES}",
            f"OAT_ZERO_VARIANT={e104.VARIANT}",
            "OAT_ZERO_MAX_TRAIN=64",
            "OAT_ZERO_SEED=70",
            "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE=1",
            "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        ),
    )


def validate_amended(run: dict[str, Any], record: str) -> None:
    job_id = int(run["job_id"])
    require_record(
        record,
        job_id,
        (
            "JobState=PENDING",
            "RunTime=00:00:00",
            f"Partition={TARGET_PARTITION}",
            f"Account={ACCOUNT}",
            f"ReqNodeList={TARGET_NODE_LIST}",
            f"TimeLimit={TIME_LIMIT}",
            f"TresPerNode={TARGET_GRES}",
            f"OAT_ZERO_VARIANT={e104.VARIANT}",
            "OAT_ZERO_MAX_TRAIN=64",
            "OAT_ZERO_SEED=70",
            "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE=1",
            "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        ),
    )


def update_command(job_id: int, amended: bool) -> list[str]:
    if amended:
        partition, nodes, gres = (
            TARGET_PARTITION,
            TARGET_NODE_LIST,
            "gpu:a6000:1",
        )
    else:
        partition, nodes, gres = (
            ORIGINAL_PARTITION,
            ORIGINAL_NODE_LIST,
            "gpu:a100:1",
        )
    return [
        "scontrol",
        "update",
        f"JobId={job_id}",
        f"Partition={partition}",
        f"Account={ACCOUNT}",
        f"NodeList={nodes}",
        f"Gres={gres}",
        f"TimeLimit={TIME_LIMIT}",
    ]


def execute(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"scheduler update failed: {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not PROTOCOL.is_file():
        raise SystemExit(f"placement protocol is absent: {PROTOCOL}")
    if args.apply and OUT.exists():
        raise SystemExit(f"refusing duplicate placement amendment: {OUT}")
    preflight_ledger, preflight_audit = validate_preflight()
    ledger = load(E104_LEDGER)
    if ledger.get("snapshot_root") != preflight_ledger.get("snapshot_root"):
        raise SystemExit("preflight and E104 snapshot roots differ")
    runs = target_runs(ledger)
    before = {int(run["job_id"]): scheduler_record(int(run["job_id"])) for run in runs}
    for run in runs:
        validate_original(run, before[int(run["job_id"])])
    commands = [update_command(int(run["job_id"]), amended=True) for run in runs]
    if not args.apply:
        for command in commands:
            print(" ".join(shlex.quote(token) for token in command))
        return 0

    changed: list[int] = []
    try:
        for run, command in zip(runs, commands, strict=True):
            execute(command)
            changed.append(int(run["job_id"]))
        after = {
            int(run["job_id"]): scheduler_record(int(run["job_id"]))
            for run in runs
        }
        for run in runs:
            validate_amended(run, after[int(run["job_id"])])
    except Exception:
        for job_id in changed:
            execute(update_command(job_id, amended=False))
        raise

    payload = {
        "schema": "e104_qwen3_a6000_placement_amendment_v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(PROTOCOL),
        "protocol_sha256": e104.digest(PROTOCOL),
        "script": str(Path(__file__)),
        "script_sha256": e104.digest(Path(__file__)),
        "e104_ledger": str(E104_LEDGER),
        "e104_ledger_sha256": e104.digest(E104_LEDGER),
        "preflight_ledger": str(PREFLIGHT_LEDGER),
        "preflight_ledger_sha256": e104.digest(PREFLIGHT_LEDGER),
        "preflight_audit": str(PREFLIGHT_AUDIT),
        "preflight_audit_sha256": e104.digest(PREFLIGHT_AUDIT),
        "preflight_job_id": int(preflight_audit["job_id"]),
        "preflight_passed": True,
        "outcome_metrics_inspected": False,
        "scientific_configuration_changed": False,
        "e105_placement_changed": False,
        "job_ids": list(TARGET_JOB_IDS),
        "before": {str(key): value for key, value in before.items()},
        "after": {str(key): value for key, value in after.items()},
    }
    e104.e81.atomic_json(OUT, payload)
    print(f"[e104-q3-a6000-amendment] amended={len(runs)} artifact={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
