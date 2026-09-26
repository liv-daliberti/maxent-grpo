#!/usr/bin/env python3
"""Audit cross-version PantryPlan fingerprint disjointness."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

from datasets import load_from_disk


SPLITS = ("train", "dev", "eval")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprints(root: Path, split: str) -> set[str]:
    dataset = load_from_disk(str(root / split))
    name = "train" if split == "train" else "multi_answer"
    if set(dataset) != {name}:
        raise ValueError(f"{root.name}/{split}: dataset key drift")
    return {str(value) for value in dataset[name]["instance_fingerprint"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--new-admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"fresh disjointness audit required: {args.output}")
    old = {split: fingerprints(args.old_root, split) for split in SPLITS}
    new = {split: fingerprints(args.new_root, split) for split in SPLITS}
    intersections = {
        f"{old_split}/{new_split}": len(old[old_split] & new[new_split])
        for old_split in SPLITS
        for new_split in SPLITS
    }
    admission = json.loads(args.new_admission.read_text())
    passed = (
        admission.get("status") == "pass"
        and all(value == 0 for value in intersections.values())
    )
    payload = {
        "schema": "pantry-repair-split-disjointness-audit-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "decision": (
            "fresh_repair_split_admitted"
            if passed
            else "pantry_repair_sampling_stopped"
        ),
        "old_root": str(args.old_root.resolve()),
        "new_root": str(args.new_root.resolve()),
        "old_counts": {split: len(values) for split, values in old.items()},
        "new_counts": {split: len(values) for split, values in new.items()},
        "cross_version_intersections": intersections,
        "new_admission_sha256": sha(args.new_admission),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=args.output.parent, prefix=f".{args.output.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(args.output)
    print(json.dumps({"status": payload["status"], "intersections": intersections}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

