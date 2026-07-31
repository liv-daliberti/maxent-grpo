#!/usr/bin/env python3
"""Aggregate the three frozen AntMaze v5 node receipts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicas", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--execution-hash", required=True)
    parser.add_argument("--job-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"fresh Ant aggregate required: {args.output}")
    receipts = [json.loads(path.read_text()) for path in args.replicas]
    protocol_sha = _sha256(args.protocol)
    violations = []
    if len({row.get("hostname") for row in receipts}) != 3:
        violations.append("replicas did not execute on three distinct nodes")
    if {row.get("replica") for row in receipts} != {0, 1, 2}:
        violations.append("replica indices differ from 0,1,2")
    for index, receipt in enumerate(receipts):
        if receipt.get("status") != "pass":
            violations.append(f"replica {index} did not pass")
        if receipt.get("source_hash") != args.source_hash:
            violations.append(f"replica {index} source hash mismatch")
        if receipt.get("execution_hash") != args.execution_hash:
            violations.append(f"replica {index} execution hash mismatch")
        if receipt.get("protocol_sha256") != protocol_sha:
            violations.append(f"replica {index} protocol hash mismatch")
        if receipt.get("summary", {}).get("execution_count") != 72:
            violations.append(f"replica {index} execution count differs from 72")

    passed = not violations
    payload = {
        "schema_version": "ant-maze-v5-cross-node-audit-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "decision": (
            "eligible_for_stepwise_lm_capability_design"
            if passed
            else "controller_v6_required_before_lm_sampling"
        ),
        "job_id": int(args.job_id),
        "source_hash": args.source_hash,
        "execution_hash": args.execution_hash,
        "protocol_sha256": protocol_sha,
        "node_count": len({row.get("hostname") for row in receipts}),
        "hostnames": sorted(str(row.get("hostname")) for row in receipts),
        "replica_sha256": [_sha256(path) for path in args.replicas],
        "validated_count": sum(
            row["summary"]["validated_count"] for row in receipts
        ),
        "margin_pass_count": sum(
            row["summary"]["margin_pass_count"] for row in receipts
        ),
        "expected_execution_count": 216,
        "violations": violations,
    }
    _atomic_json(args.output, payload)
    print(
        "[ant-cross-node-aggregate] "
        f"status={payload['status']} nodes={payload['node_count']} "
        f"validated={payload['validated_count']}/216 "
        f"margin={payload['margin_pass_count']}/216",
        flush=True,
    )


if __name__ == "__main__":
    main()
