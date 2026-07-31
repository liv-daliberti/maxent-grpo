#!/usr/bin/env python3
"""Replay the exact admitted AntMaze v9 slate on one cluster node."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import sys
import tempfile
from typing import Any


ROOT = Path(os.environ.get("OAT_ZERO_REPO_ROOT", Path(__file__).resolve().parents[1]))
SRC = Path(os.environ.get("OAT_ZERO_SOURCE_ROOT", ROOT / "src"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oat_drgrpo.ant_maze_worker_v9 import (  # noqa: E402
    controller_receipt_sha256,
    execute_ant_v9_raw,
)
from oat_drgrpo.maze_modebench import validate_maze_execution  # noqa: E402
from oat_drgrpo.maze_runtime_identity import maze_runtime_identity  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--replica", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--execution-hash", required=True)
    parser.add_argument("--job-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"fresh Ant v9 node receipt required: {args.output}")
    export = json.loads(args.specs.read_text(encoding="utf-8"))
    if export.get("schema_version") != "ant-maze-v9-cross-node-spec-export-v1":
        raise ValueError("Ant v9 cross-node spec schema mismatch")
    if export.get("controller_receipt_sha256") != controller_receipt_sha256():
        raise ValueError("Ant v9 cross-node export binds the wrong controller")
    records = []
    for row in export["rows"]:
        spec = row["spec"]
        for route in row["routes"]:
            candidate = str(route["program"])
            for repetition in range(args.repetitions):
                execution = validation = None
                error = None
                try:
                    execution = execute_ant_v9_raw(candidate, spec)
                    validation = validate_maze_execution(candidate, spec, execution)
                    if validation.canonical_key != route["canonical_key"]:
                        raise ValueError("canonical topology key changed")
                    if list(validation.directed_gates) != route["directed_gates"]:
                        raise ValueError("directed route gate changed")
                except Exception as caught:
                    error = f"{type(caught).__name__}: {caught}"
                records.append(
                    {
                        "map_id": spec["map_id"],
                        "spec_sha256": spec["spec_sha256"],
                        "route_name": route["route_name"],
                        "program_sha256": route["program_sha256"],
                        "repetition": repetition,
                        "validated": validation is not None and error is None,
                        "canonical_key": validation.canonical_key if validation else None,
                        "directed_gates": list(validation.directed_gates) if validation else [],
                        "error": error,
                        "execution": execution,
                    }
                )
    passed = len(records) == 72 and all(row["validated"] for row in records)
    payload = {
        "schema_version": "ant-maze-v9-cross-node-replica-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "job_id": int(args.job_id),
        "replica": args.replica,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "source_hash": args.source_hash,
        "execution_hash": args.execution_hash,
        "protocol_sha256": _sha256(args.protocol),
        "spec_export_sha256": _sha256(args.specs),
        "controller_receipt_sha256": controller_receipt_sha256(),
        "runtime_identity": maze_runtime_identity(),
        "map_count": 12,
        "route_count": 2,
        "repetitions": args.repetitions,
        "summary": {
            "execution_count": len(records),
            "validated_count": sum(row["validated"] for row in records),
        },
        "records": records,
    }
    _atomic_json(args.output, payload)
    print(
        f"[ant-v9-cross-node] replica={args.replica} host={payload['hostname']} "
        f"status={payload['status']} validated={payload['summary']['validated_count']}/72",
        flush=True,
    )


if __name__ == "__main__":
    main()
