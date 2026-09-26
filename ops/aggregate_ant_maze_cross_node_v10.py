#!/usr/bin/env python3
"""Aggregate the three frozen AntMaze v10 node receipts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile


VERSION = "v10"
REPLICA_SCHEMA = "ant-maze-v10-cross-node-replica-v1"
AUDIT_SCHEMA = "ant-maze-v10-cross-node-audit-v1"
PASS_DECISION = "eligible_for_frozen_05b_viability_gate_v10"
FAIL_DECISION = "ant_maze_v10_stopped"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicas", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--execution-hash", required=True)
    parser.add_argument("--job-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"fresh Ant v10 cross-node aggregate required: {args.output}")
    receipts = [json.loads(path.read_text(encoding="utf-8")) for path in args.replicas]
    expected_protocol = _sha256(args.protocol)
    expected_specs = _sha256(args.specs)
    violations = []
    if len({row.get("hostname") for row in receipts}) != 3:
        violations.append("replicas did not execute on three distinct nodes")
    if {row.get("replica") for row in receipts} != {0, 1, 2}:
        violations.append("replica indices differ from 0,1,2")
    for index, receipt in enumerate(receipts):
        if receipt.get("schema_version") != REPLICA_SCHEMA:
            violations.append(f"replica {index} schema mismatch")
        if receipt.get("status") != "pass":
            violations.append(f"replica {index} did not pass")
        if receipt.get("source_hash") != args.source_hash:
            violations.append(f"replica {index} source hash mismatch")
        if receipt.get("execution_hash") != args.execution_hash:
            violations.append(f"replica {index} execution hash mismatch")
        if receipt.get("protocol_sha256") != expected_protocol:
            violations.append(f"replica {index} protocol hash mismatch")
        if receipt.get("spec_export_sha256") != expected_specs:
            violations.append(f"replica {index} spec export hash mismatch")
        if receipt.get("summary", {}).get("execution_count") != 72:
            violations.append(f"replica {index} execution count differs from 72")
        if receipt.get("summary", {}).get("validated_count") != 72:
            violations.append(f"replica {index} validation count differs from 72")
    passed = not violations
    payload = {
        "schema_version": AUDIT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "decision": (
            PASS_DECISION if passed else FAIL_DECISION
        ),
        "job_id": int(args.job_id),
        "source_hash": args.source_hash,
        "execution_hash": args.execution_hash,
        "protocol_sha256": expected_protocol,
        "spec_export_sha256": expected_specs,
        "node_count": len({row.get("hostname") for row in receipts}),
        "hostnames": sorted(str(row.get("hostname")) for row in receipts),
        "replica_sha256": [_sha256(path) for path in args.replicas],
        "validated_count": sum(row["summary"]["validated_count"] for row in receipts),
        "expected_execution_count": 216,
        "violations": violations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, args.output)
    print(
        f"[ant-{VERSION}-cross-node-aggregate] status={payload['status']} "
        f"nodes={payload['node_count']} validated={payload['validated_count']}/216",
        flush=True,
    )


if __name__ == "__main__":
    main()
