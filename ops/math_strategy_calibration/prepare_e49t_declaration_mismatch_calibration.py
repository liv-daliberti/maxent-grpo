#!/usr/bin/env python3
"""Freeze matched and deliberately mismatched E49T route declarations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import tempfile
from typing import Any

from oat_drgrpo.math_strategy_menu import parse_strategy_menu


ROOT = pathlib.Path(__file__).resolve().parents[2]
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e49t_declaration_mismatch_calibration_20260726.md"
)
SCRIPT = pathlib.Path(__file__).resolve()
SOURCE = ROOT / "var/artifacts/e49t_route_confusion_calibration_v1"
SEED = 492727


def _bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(_bytes(row).decode("utf-8") + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare(output: pathlib.Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"fresh mismatch calibration required: {output}")
    if "FROZEN BEFORE 72B SCORING" not in PROTOCOL.read_text(encoding="utf-8"):
        raise RuntimeError("mismatch protocol is not frozen")
    source_identity = json.loads(
        (SOURCE / "frozen_identity.json").read_text(encoding="utf-8")
    )
    cohort_path = SOURCE / "cohort.jsonl"
    labels_path = SOURCE / "private/labels.jsonl"
    if (
        source_identity["cohort_sha256"] != _sha(cohort_path)
        or source_identity["private_labels_sha256"] != _sha(labels_path)
    ):
        raise RuntimeError("source calibration identity mismatch")
    public_source = {row["item_id"]: row for row in _read_jsonl(cohort_path)}
    labels = _read_jsonl(labels_path)
    selected = [
        row
        for row in labels
        if row["kind"] == "positive" and row["rendering_index"] == 0
    ]
    if len(selected) != 24:
        raise RuntimeError("expected one rendering of each of 24 routes")

    by_group: dict[str, list[dict[str, Any]]] = {}
    for label in selected:
        by_group.setdefault(label["group_id"], []).append(label)
    public_rows = []
    private_rows = []
    for group_index, group_id in enumerate(sorted(by_group)):
        routes = by_group[group_id]
        if sorted(row["expected_strategy_id"] for row in routes) != ["S1", "S2"]:
            raise RuntimeError("mismatch source group lacks two routes")
        prompt = public_source[routes[0]["item_id"]]["problem"]
        menu = parse_strategy_menu(prompt)
        if menu is None or len(menu.strategies) != 2:
            raise RuntimeError("mismatch prompt lacks its two-route menu")
        candidates = []
        for route in routes:
            expected = menu.strategy(route["expected_strategy_id"])
            assert expected is not None
            opposite = next(
                strategy
                for strategy in menu.strategies
                if strategy.strategy_id != expected.strategy_id
            )
            natural = public_source[route["item_id"]]["response"]
            for kind, declared in (
                ("matched_declaration", expected),
                ("mismatched_declaration", opposite),
            ):
                response = (
                    f"Chosen strategy ID: {declared.strategy_id}\n"
                    f"Chosen action combo: {declared.action_combo}\n\n"
                    f"{natural}"
                )
                candidates.append(
                    {
                        "kind": kind,
                        "expected_strategy_id": (
                            expected.strategy_id
                            if kind == "matched_declaration"
                            else None
                        ),
                        "executed_strategy_id": expected.strategy_id,
                        "declared_strategy_id": declared.strategy_id,
                        "response": response,
                    }
                )
        random.Random(SEED + group_index).shuffle(candidates)
        for position, candidate in enumerate(candidates):
            item_id = f"{group_id}_DECL_{position:02d}"
            public_rows.append(
                {
                    "group_id": group_id,
                    "group_index": group_index,
                    "item_id": item_id,
                    "problem": prompt,
                    "response": candidate["response"],
                }
            )
            private_rows.append(
                {
                    "group_id": group_id,
                    "group_index": group_index,
                    "item_id": item_id,
                    "kind": candidate["kind"],
                    "expected_strategy_id": candidate["expected_strategy_id"],
                    "executed_strategy_id": candidate["executed_strategy_id"],
                    "declared_strategy_id": candidate["declared_strategy_id"],
                    "response_sha256": hashlib.sha256(
                        candidate["response"].encode("utf-8")
                    ).hexdigest(),
                }
            )
    if len(public_rows) != 48 or len(private_rows) != 48:
        raise RuntimeError("mismatch calibration size mismatch")
    output.mkdir(parents=True)
    _write_jsonl(output / "cohort.jsonl", public_rows)
    _write_jsonl(output / "private/labels.jsonl", private_rows)
    identity = {
        "schema": "e49t_declaration_mismatch_identity_v1",
        "group_count": 12,
        "matched_count": 24,
        "mismatched_count": 24,
        "cohort_sha256": _sha(output / "cohort.jsonl"),
        "private_labels_sha256": _sha(output / "private/labels.jsonl"),
        "source_identity_sha256": _sha(SOURCE / "frozen_identity.json"),
        "protocol_sha256": _sha(PROTOCOL),
        "preparer_sha256": _sha(SCRIPT),
    }
    _write_json(output / "frozen_identity.json", identity)
    return identity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
