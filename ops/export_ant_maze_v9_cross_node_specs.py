#!/usr/bin/env python3
"""Export the exact admitted v9 Arrow rows for networkless node replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from datasets import load_from_disk


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--route-audit", type=Path, required=True)
    parser.add_argument("--route-identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"fresh v9 cross-node export required: {args.output}")
    data_identity_path = args.data_root / "identity.json"
    data_identity = json.loads(data_identity_path.read_text(encoding="utf-8"))
    route_audit = json.loads(args.route_audit.read_text(encoding="utf-8"))
    route_identity = json.loads(args.route_identity.read_text(encoding="utf-8"))
    if data_identity.get("schema_version") != "ant-maze-modebench-data-v9":
        raise ValueError("v9 cross-node export received the wrong data schema")
    if (
        route_audit.get("status") != "pass"
        or route_audit.get("decision")
        != "admitted_to_v9_cross_node_route_determinism_gate"
    ):
        raise ValueError("v9 route audit is not admitted to cross-node replay")
    if route_identity.get("schema_version") != (
        "ant-maze-v9-route-generation-identity-v1"
    ):
        raise ValueError("v9 route generation identity schema mismatch")
    certification = {
        str(row["map_id"]): row for row in data_identity.get("certification", [])
    }
    if len(certification) != 12:
        raise ValueError("v9 data identity does not certify 12 maps")
    rows = []
    for split, dataset_split in (
        ("train", "train"),
        ("dev", "multi_answer"),
        ("eval", "multi_answer"),
    ):
        dataset = load_from_disk(str(args.data_root / split))
        if set(dataset) != {dataset_split}:
            raise ValueError(f"v9 {split} dataset split mismatch")
        for row in dataset[dataset_split].to_list():
            spec = json.loads(row["answer"])
            map_id = str(spec["map_id"])
            record = certification.get(map_id)
            if record is None or record.get("spec_sha256") != spec.get("spec_sha256"):
                raise ValueError(f"v9 certification mismatch for {map_id}")
            routes = [
                {
                    "route_name": route["route_name"],
                    "program": route["program"],
                    "program_sha256": route["program_sha256"],
                    "canonical_key": route["canonical_key"],
                    "directed_gates": route["directed_gates"],
                }
                for route in record["routes"]
            ]
            if len(routes) != 2 or len({item["canonical_key"] for item in routes}) != 2:
                raise ValueError(f"v9 route identity mismatch for {map_id}")
            rows.append({"split": split, "spec": spec, "routes": routes})
    if len(rows) != 12 or len({row["spec"]["spec_sha256"] for row in rows}) != 12:
        raise ValueError("v9 cross-node export rows are not 12 unique specs")
    rows.sort(key=lambda row: str(row["spec"]["map_id"]))
    payload = {
        "schema_version": "ant-maze-v9-cross-node-spec-export-v1",
        "data_identity_sha256": _sha256(data_identity_path),
        "route_audit_sha256": _sha256(args.route_audit),
        "route_identity_sha256": _sha256(args.route_identity),
        "controller_receipt_sha256": route_identity["controller_receipt_sha256"],
        "controller_model_sha256": route_identity["controller_model_sha256"],
        "map_count": 12,
        "routes_per_map": 2,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, args.output)
    print(f"[ant-v9-cross-node-export] maps=12 routes=24 output={args.output}")


if __name__ == "__main__":
    main()
