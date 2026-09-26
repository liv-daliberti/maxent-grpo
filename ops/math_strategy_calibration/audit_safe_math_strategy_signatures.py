#!/usr/bin/env python3
"""Retrospective development audit for safe decisive-operation signatures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import tempfile
from typing import Any

from safe_math_strategy_signatures import safe_distinct_pair, signature_hits


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = pathlib.Path(__file__).resolve()
PACKET = (
    ROOT
    / "var/artifacts/e49r_combined_manual_audit_v1/"
    "manual_audit_packet.jsonl"
)
LABELS = (
    ROOT
    / "var/artifacts/e49r_combined_manual_audit_v1/"
    "manual_audit_labels.json"
)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    packets = {
        str(row["pair_id"]): row for row in _read_jsonl(PACKET)
    }
    labels = json.loads(LABELS.read_text(encoding="utf-8"))["labels"]
    results = []
    for label in labels:
        if (
            label["route_a_sound_and_self_contained"] is not True
            or label["route_b_sound_and_self_contained"] is not True
        ):
            continue
        packet = packets[str(label["pair_id"])]
        distinct, left, right = safe_distinct_pair(
            packet["route_a"], packet["route_b"]
        )
        expected = bool(label["genuinely_distinct_decisive_strategy"])
        results.append(
            {
                "pair_id": label["pair_id"],
                "expected_distinct": expected,
                "observed_distinct": distinct,
                "correct": distinct == expected,
                "route_a_signature": left,
                "route_a_hits": signature_hits(packet["route_a"]),
                "route_b_signature": right,
                "route_b_hits": signature_hits(packet["route_b"]),
            }
        )
    false_new = sum(
        not row["expected_distinct"] and row["observed_distinct"]
        for row in results
    )
    false_merge = sum(
        row["expected_distinct"] and not row["observed_distinct"]
        for row in results
    )
    payload = {
        "schema": "safe_math_strategy_signature_development_audit_v1",
        "claim_scope": (
            "retrospective development audit; does not authorize training"
        ),
        "counts": {
            "sound_pair_count": len(results),
            "same_pair_count": sum(
                not row["expected_distinct"] for row in results
            ),
            "distinct_pair_count": sum(
                row["expected_distinct"] for row in results
            ),
            "false_new_count": false_new,
            "false_merge_count": false_merge,
            "distinct_recall": (
                (
                    sum(
                        row["expected_distinct"]
                        and row["observed_distinct"]
                        for row in results
                    )
                    / sum(row["expected_distinct"] for row in results)
                )
                if any(row["expected_distinct"] for row in results)
                else 0.0
            ),
        },
        "checks": {
            "zero_false_new_on_development_pairs": false_new == 0,
            "all_same_pairs_rejected": false_new == 0,
        },
        "results": results,
        "identity": {
            "script_sha256": _sha256(SCRIPT),
            "signature_source_sha256": _sha256(
                SCRIPT.with_name("safe_math_strategy_signatures.py")
            ),
            "packet_sha256": _sha256(PACKET),
            "labels_sha256": _sha256(LABELS),
        },
    }
    _write_json(args.output.resolve(), payload)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
