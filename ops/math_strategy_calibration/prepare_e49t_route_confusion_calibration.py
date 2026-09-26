#!/usr/bin/env python3
"""Freeze E49T's blinded finite-menu route-confusion calibration cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import tempfile
from typing import Any

from datasets import load_from_disk

from oat_drgrpo.math_strategy_menu import parse_strategy_menu


ROOT = pathlib.Path(__file__).resolve().parents[2]
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e49t_finite_menu_route_confusion_calibration_20260726.md"
)
SCRIPT = pathlib.Path(__file__).resolve()
E49R = ROOT / "var/artifacts/e49r_combined_manual_audit_v1"
E49S = ROOT / "var/artifacts/e49s_deterministic_mathir_repairs_v1"
DATA = ROOT / "var/data/e49s_deterministic_mathir_toy"
COHORT_SEED = 492726


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


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
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
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
                handle.write(_canonical_bytes(row).decode("utf-8") + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _dataset_rows() -> dict[str, list[dict[str, Any]]]:
    result = {}
    for split in ("train", "eval"):
        dataset = load_from_disk(str(DATA / split))
        name = next(iter(dataset))
        result[split] = [dict(row) for row in dataset[name]]
    return result


def _trace_text(trace: dict[str, Any], reference: str) -> str:
    paragraphs = []
    executions = trace.get("action_executions")
    if not isinstance(executions, list) or not executions:
        raise RuntimeError("route trace has no action executions")
    for execution in executions:
        calculation = str(execution.get("executed_calculation", "")).strip()
        output = str(execution.get("output_fact", "")).strip()
        if not calculation or not output or execution.get("status") != "valid":
            raise RuntimeError("route trace is not a valid visible execution")
        # Do not expose strategy IDs or action IDs to the semantic classifier.
        paragraphs.append(f"{calculation}\n{output}")
    paragraphs.append(f"Therefore the answer is \\boxed{{{reference}}}.")
    return "\n\n".join(paragraphs)


def prepare(output_dir: pathlib.Path) -> None:
    if output_dir.exists():
        raise RuntimeError(f"fresh E49T cohort directory required: {output_dir}")
    if (
        not PROTOCOL.is_file()
        or "FROZEN BEFORE 72B SCORING"
        not in PROTOCOL.read_text(encoding="utf-8")
    ):
        raise RuntimeError("E49T route-confusion protocol is not frozen")

    labels_payload = json.loads(
        (E49R / "manual_audit_labels.json").read_text(encoding="utf-8")
    )
    labels = {
        row["pair_id"]: row for row in labels_payload["labels"]
    }
    packets = {
        row["pair_id"]: row
        for row in _read_jsonl(E49R / "manual_audit_packet.jsonl")
    }
    private_key = {
        row["pair_id"]: row
        for row in _read_jsonl(E49R / "private/manual_audit_key.jsonl")
    }
    e49s_records = {
        row["row_id"]: row
        for row in _read_jsonl(E49S / "audited_records.jsonl")
    }
    datasets = _dataset_rows()

    eligible_pair_ids = sorted(
        pair_id
        for pair_id, label in labels.items()
        if label["route_a_sound_and_self_contained"]
        and label["route_b_sound_and_self_contained"]
        and label["genuinely_distinct_decisive_strategy"]
    )
    if len(eligible_pair_ids) != 12:
        raise RuntimeError("expected exactly twelve manually accepted pairs")

    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    rng = random.Random(COHORT_SEED)
    for group_index, pair_id in enumerate(eligible_pair_ids):
        packet = packets[pair_id]
        key = private_key[pair_id]
        row_id = key["row_id"]
        if row_id not in e49s_records:
            raise RuntimeError(f"E49T pair is absent from E49S: {pair_id}")
        split, raw_index, _ = row_id.split(":", maxsplit=2)
        dataset_row = datasets[split][int(raw_index)]
        if (
            dataset_row["original_problem"].strip() != packet["problem"].strip()
            or dataset_row["strategy_menu_sha256"] != key["menu_sha256"]
            or e49s_records[row_id]["menu_sha256"] != key["menu_sha256"]
        ):
            raise RuntimeError(f"E49T source identity mismatch: {pair_id}")
        menu = parse_strategy_menu(dataset_row["problem"])
        if menu is None or menu.sha256 != key["menu_sha256"]:
            raise RuntimeError(f"E49T menu parse mismatch: {pair_id}")

        candidates = []
        for side in ("route_a", "route_b"):
            route = packet[side]
            combo = route["action_combo"]
            strategy = next(
                (
                    option
                    for option in menu.strategies
                    if option.action_combo == combo
                ),
                None,
            )
            if strategy is None:
                raise RuntimeError(f"E49T route is absent from menu: {pair_id}")
            traces = route["independent_execution_traces"]
            if len(traces) != 2:
                raise RuntimeError("expected two independent route traces")
            for rendering_index, trace in enumerate(traces):
                if (
                    trace.get("status") != "sound"
                    or not trace.get("matches_reference_answer")
                    or not trace.get("self_contained_without_other_strategy")
                    or not trace.get("uses_only_declared_actions")
                    or trace.get("strategy_id") != strategy.strategy_id
                ):
                    raise RuntimeError(
                        f"E49T positive trace contract failed: {pair_id}"
                    )
                candidates.append(
                    {
                        "kind": "positive",
                        "expected_strategy_id": strategy.strategy_id,
                        "source_side": side,
                        "rendering_index": rendering_index,
                        "response": _trace_text(
                            trace, str(packet["reference_answer"])
                        ),
                    }
                )
        candidates.append(
            {
                "kind": "answer_only_negative",
                "expected_strategy_id": None,
                "source_side": None,
                "rendering_index": None,
                "response": (
                    "The requested result is "
                    f"\\boxed{{{packet['reference_answer']}}}."
                ),
            }
        )
        rng.shuffle(candidates)
        group_id = "GROUP_" + hashlib.sha256(
            f"{COHORT_SEED}:{pair_id}".encode("utf-8")
        ).hexdigest()[:16]
        for position, candidate in enumerate(candidates):
            item_id = f"{group_id}_ITEM_{position:02d}"
            public = {
                "group_id": group_id,
                "group_index": group_index,
                "item_id": item_id,
                "problem": dataset_row["problem"],
                "response": candidate["response"],
            }
            private = {
                "group_id": group_id,
                "group_index": group_index,
                "item_id": item_id,
                "pair_id": pair_id,
                "row_id": row_id,
                "kind": candidate["kind"],
                "expected_strategy_id": candidate["expected_strategy_id"],
                "source_side": candidate["source_side"],
                "rendering_index": candidate["rendering_index"],
                "menu_sha256": menu.sha256,
                "response_sha256": hashlib.sha256(
                    candidate["response"].encode("utf-8")
                ).hexdigest(),
            }
            public_rows.append(public)
            private_rows.append(private)

    if len(public_rows) != 60 or len(private_rows) != 60:
        raise RuntimeError("E49T cohort size mismatch")
    output_dir.mkdir(parents=True)
    _write_jsonl(output_dir / "cohort.jsonl", public_rows)
    private_dir = output_dir / "private"
    _write_jsonl(private_dir / "labels.jsonl", private_rows)
    identity = {
        "schema": "e49t_route_confusion_frozen_identity_v1",
        "cohort_seed": COHORT_SEED,
        "group_count": 12,
        "positive_count": 48,
        "answer_only_negative_count": 12,
        "cohort_sha256": _sha256(output_dir / "cohort.jsonl"),
        "private_labels_sha256": _sha256(private_dir / "labels.jsonl"),
        "protocol_sha256": _sha256(PROTOCOL),
        "prepare_script_sha256": _sha256(SCRIPT),
        "source_sha256s": {
            "e49r_manual_labels": _sha256(
                E49R / "manual_audit_labels.json"
            ),
            "e49r_manual_packet": _sha256(
                E49R / "manual_audit_packet.jsonl"
            ),
            "e49r_private_key": _sha256(
                E49R / "private/manual_audit_key.jsonl"
            ),
            "e49s_audited_records": _sha256(
                E49S / "audited_records.jsonl"
            ),
            "e49s_materialization_manifest": _sha256(
                DATA / "MATERIALIZATION_MANIFEST.json"
            ),
        },
    }
    _write_json(output_dir / "frozen_identity.json", identity)
    print(json.dumps(identity, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    prepare(args.output_dir.resolve())


if __name__ == "__main__":
    main()
