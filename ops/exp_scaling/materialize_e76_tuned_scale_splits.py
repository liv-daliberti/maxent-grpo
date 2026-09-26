#!/usr/bin/env python3
"""Materialize the sealed E76 320/64 train/validation splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_from_disk

SALT = "modebench-e76-tuned-scale-v1-2026-08-03"
VALIDATION_ROWS = 64
EXPECTED_ROWS = 384
DOMAINS = {
    "graph_coloring": "var/data/graph_coloring_modebench_v2",
    "pantry_plan": "var/data/pantry_plan_modebench_v2",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def canonical_row(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def row_digest(domain: str, row: dict[str, Any]) -> str:
    payload = f"{SALT}\n{domain}\n{canonical_row(row)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prompt_identity(row: dict[str, Any]) -> str:
    payload = json.dumps(
        [row.get("modebench_task"), row.get("problem")],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as sink:
            json.dump(payload, sink, indent=2, sort_keys=True)
            sink.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def split_domain(root: Path, domain: str, relative: str, force: bool) -> dict[str, Any]:
    source_root = root / relative
    source = load_from_disk(str(source_root / "train"))
    if not isinstance(source, DatasetDict) or "train" not in source:
        raise SystemExit(f"{source_root / 'train'} is not a DatasetDict with train")
    train = source["train"]
    if len(train) != EXPECTED_ROWS:
        raise SystemExit(f"{domain}: expected {EXPECTED_ROWS} source rows, found {len(train)}")

    ranked = sorted(
        ((row_digest(domain, train[index]), index) for index in range(len(train))),
        key=lambda item: (item[0], item[1]),
    )
    validation_indices = sorted(index for _, index in ranked[:VALIDATION_ROWS])
    training_indices = sorted(index for _, index in ranked[VALIDATION_ROWS:])
    if set(validation_indices) & set(training_indices):
        raise AssertionError(f"{domain}: train/validation index overlap")
    if sorted(validation_indices + training_indices) != list(range(EXPECTED_ROWS)):
        raise AssertionError(f"{domain}: split is not a lossless partition")

    test = load_from_disk(str(source_root / "eval"))
    if not isinstance(test, DatasetDict) or "multi_answer" not in test:
        raise SystemExit(f"{source_root / 'eval'} lacks multi_answer")
    train_ids = {prompt_identity(train[index]) for index in training_indices}
    val_ids = {prompt_identity(train[index]) for index in validation_indices}
    test_ids = {prompt_identity(row) for row in test["multi_answer"]}
    if train_ids & val_ids:
        raise SystemExit(f"{domain}: prompt identity overlaps train and validation")
    if (train_ids | val_ids) & test_ids:
        raise SystemExit(f"{domain}: original train pool overlaps reported test prompts")

    output_root = root / "var" / "data" / "e76_tuned_scale" / domain
    train_out = output_root / "train"
    validation_out = output_root / "validation"
    if (train_out.exists() or validation_out.exists()) and not force:
        raise SystemExit(f"{output_root} already exists; use --force only after auditing it")
    train_out.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": train.select(training_indices)}).save_to_disk(str(train_out))
    DatasetDict({"multi_answer": train.select(validation_indices)}).save_to_disk(
        str(validation_out)
    )

    all_hashes = [row_digest(domain, train[index]) for index in range(len(train))]
    return {
        "domain": domain,
        "source_train": str(source_root / "train"),
        "source_test": str(source_root / "eval"),
        "source_dataset_fingerprint": train._fingerprint,
        "source_rows": len(train),
        "train_rows": len(training_indices),
        "validation_rows": len(validation_indices),
        "train_indices": training_indices,
        "validation_indices": validation_indices,
        "source_row_sha256": hashlib.sha256("\n".join(all_hashes).encode()).hexdigest(),
        "train_prompt_identity_sha256": hashlib.sha256(
            "\n".join(sorted(train_ids)).encode()
        ).hexdigest(),
        "validation_prompt_identity_sha256": hashlib.sha256(
            "\n".join(sorted(val_ids)).encode()
        ).hexdigest(),
        "reported_test_prompt_identity_sha256": hashlib.sha256(
            "\n".join(sorted(test_ids)).encode()
        ).hexdigest(),
        "train_path": str(train_out),
        "validation_path": str(validation_out),
    }


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "var" / "artifacts" / "e76_tuned_scale_splits.json",
    )
    args = parser.parse_args()
    records = [split_domain(root, domain, relative, args.force) for domain, relative in DOMAINS.items()]
    payload = {
        "schema": "e76_tuned_scale_splits_v1",
        "frozen_at": "2026-08-03",
        "salt": SALT,
        "selection": "lowest SHA-256 ranks form validation",
        "domains": records,
    }
    atomic_json(args.manifest, payload)
    print(f"[e76 split] wrote {args.manifest} ({len(records)} domains)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
