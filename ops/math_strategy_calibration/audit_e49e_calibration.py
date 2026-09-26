#!/usr/bin/env python3
"""Prepare and finalize E49E's blinded, fail-closed toy calibration audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import pathlib
import random
import sys
import tempfile
from typing import Any

from datasets import load_from_disk
from transformers import AutoTokenizer


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL = (
    ROOT
    / "var/cache/huggingface/transformers/"
    "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
    "7ae557604adf67be50417f59c2c2f167def9a775"
)
PACKET_SEED = 492131


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hash(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(
        candidate for candidate in path.rglob("*") if candidate.is_file()
    ):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_pipeline(identity: dict[str, Any]):
    snapshot = pathlib.Path(identity["snapshot_root"])
    if (
        not snapshot.is_dir()
        or _tree_hash(snapshot) != identity["snapshot_tree_sha256"]
    ):
        raise RuntimeError("E49E source snapshot identity changed")
    sys.path.insert(0, str(snapshot / "src"))
    path = (
        snapshot
        / "ops/math_strategy_calibration/"
        "materialize_e49e_trace_bank_data.py"
    )
    spec = importlib.util.spec_from_file_location(
        "frozen_e49e_materializer",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen E49E materializer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_rows(module, source: pathlib.Path):
    train = load_from_disk(str(source / "train"))
    evaluation = load_from_disk(str(source / "eval"))
    splits = {
        "train": train[next(iter(train))],
        "eval": evaluation[next(iter(evaluation))],
    }
    rows = []
    for split, dataset in splits.items():
        for index, row in enumerate(dataset):
            rows.append(
                {
                    "split": split,
                    "index": index,
                    "row_id": module.base._row_id(split, index, row),
                    "problem": str(row["problem"]),
                    "reference_answer": str(row["answer"]),
                }
            )
    return rows


def _route_payload(
    menu,
    strategy_id: str,
    sound_audits: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    strategy = menu.strategy(strategy_id)
    if strategy is None:
        raise RuntimeError("manual-audit strategy disappeared")
    action_by_id = {
        action.action_id: action.operation for action in menu.actions
    }
    return {
        "plan": strategy.plan,
        "action_combo": strategy.action_combo,
        "actions": [
            {
                "action_id": action_id,
                "operation": action_by_id[action_id],
            }
            for action_id in strategy.action_ids
        ],
        "independent_execution_traces": [
            audit["assessment"] for audit in sound_audits[strategy_id]
        ],
    }


def _validated_state(
    evidence: pathlib.Path,
    data: pathlib.Path,
    source: pathlib.Path,
):
    identity_path = evidence / "frozen_identity.json"
    summary_path = evidence / "generation_summary.json"
    records_path = evidence / "trace_bank_records.jsonl"
    manifest_path = data / "MATERIALIZATION_MANIFEST.json"
    for required in (
        identity_path,
        summary_path,
        records_path,
        manifest_path,
    ):
        if not required.is_file():
            raise RuntimeError(f"E49E calibration is incomplete: {required}")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        identity.get("schema") != "e49e_trace_bank_frozen_identity_v1"
        or summary.get("pass") is not True
        or manifest.get("schema") != "e49e_trace_bank_materialization_v1"
        or manifest.get("all_known_invalid_controls_rejected") is not True
        or manifest.get(
            "all_known_equivalent_controls_rejected_as_new"
        )
        is not True
        or manifest.get("trace_bank_records_sha256")
        != _sha256(records_path)
    ):
        raise RuntimeError("E49E automatic calibration gate failed")
    module = _load_pipeline(identity)
    input_path = evidence / "frozen_e49d_input/menu_records.jsonl"
    if (
        not input_path.is_file()
        or _sha256(input_path) != identity["input_evidence_sha256"]
    ):
        raise RuntimeError("E49E frozen input identity changed")
    input_records = module._load_latest(input_path)
    records = module._load_latest(records_path)
    rows = _source_rows(module, source)
    if set(records) != {row["row_id"] for row in rows}:
        raise RuntimeError("E49E trace records do not cover the source")
    for row in rows:
        record = records[row["row_id"]]
        if (
            record.get("problem_sha256")
            != hashlib.sha256(row["problem"].encode()).hexdigest()
            or record.get("reference_answer_sha256")
            != hashlib.sha256(
                row["reference_answer"].encode()
            ).hexdigest()
            or not module._record_passes_contract(
                record,
                reference_answer=row["reference_answer"],
                input_record=input_records[row["row_id"]],
            )
        ):
            raise RuntimeError(f"E49E record failed: {row['row_id']}")
    return identity, manifest, module, records, rows


def _prompt_length_report(
    data: pathlib.Path,
    module,
) -> dict[str, Any]:
    template_spec = importlib.util.find_spec("oat_drgrpo.templates")
    if template_spec is None:
        raise RuntimeError("frozen prompt template is missing")
    from oat_drgrpo.templates import apply_qwen_math_template

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL),
        local_files_only=True,
    )
    rows = []
    for tree in ("train", "eval"):
        dataset_dict = load_from_disk(str(data / tree))
        split = next(iter(dataset_dict))
        for index, row in enumerate(dataset_dict[split]):
            problem = str(row["problem"])
            menu = module.parse_strategy_menu(problem)
            if menu is None:
                raise RuntimeError(f"{tree}:{index} has no strategy menu")
            count = len(
                tokenizer(
                    apply_qwen_math_template(problem),
                    add_special_tokens=False,
                    truncation=False,
                )["input_ids"]
            )
            rows.append(
                {
                    "tree": tree,
                    "index": index,
                    "tokens": count,
                    "menu_sha256": menu.sha256,
                }
            )
    over = [row for row in rows if row["tokens"] > 2048]
    return {
        "schema": "e49e_prompt_length_audit_v1",
        "pass": not over,
        "row_count": len(rows),
        "limit": 2048,
        "observed_max": max(row["tokens"] for row in rows),
        "observed_mean": sum(row["tokens"] for row in rows) / len(rows),
        "over_limit": over,
        "longest": sorted(
            rows,
            key=lambda row: row["tokens"],
            reverse=True,
        )[:10],
    }


def prepare(args) -> None:
    identity, manifest, module, records, rows = _validated_state(
        args.evidence,
        args.data,
        args.source,
    )
    packet_path = args.evidence / "manual_audit_packet.jsonl"
    private_path = args.evidence / "private/manual_audit_key.jsonl"
    packet_manifest_path = args.evidence / "manual_audit_manifest.json"
    prompt_report_path = args.evidence / "prompt_length_audit.json"
    if any(
        path.exists()
        for path in (packet_path, private_path, packet_manifest_path)
    ):
        raise RuntimeError("E49E manual-audit packet already exists")

    row_by_id = {row["row_id"]: row for row in rows}
    packet = []
    private = []
    for row_id in sorted(records):
        record = records[row_id]
        certification = record["certification"]
        retained = certification["retained_original_strategy_ids"]
        if len(retained) < 2:
            continue
        candidate = module._menu_from_payload(
            certification["candidate_menu"]
        )
        sound = certification["sound_audits"]
        for left, right in itertools.combinations(retained, 2):
            digest = hashlib.sha256(
                f"{row_id}\0{left}\0{right}\0{record['menu_sha256']}".encode()
            ).hexdigest()
            pair_id = f"PAIR_{digest[:20]}"
            routes = [
                _route_payload(candidate, left, sound),
                _route_payload(candidate, right, sound),
            ]
            swapped = random.Random(
                PACKET_SEED + int(digest[:12], 16)
            ).randrange(2)
            if swapped:
                routes.reverse()
            packet.append(
                {
                    "schema": "e49e_blinded_manual_pair_v1",
                    "pair_id": pair_id,
                    "problem": row_by_id[row_id]["problem"],
                    "reference_answer": row_by_id[row_id][
                        "reference_answer"
                    ],
                    "route_a": routes[0],
                    "route_b": routes[1],
                    "questions": {
                        "route_a_sound_and_self_contained": "boolean",
                        "route_b_sound_and_self_contained": "boolean",
                        "genuinely_distinct_decisive_strategy": "boolean",
                        "rationale": "nonempty string",
                    },
                }
            )
            private.append(
                {
                    "pair_id": pair_id,
                    "row_id": row_id,
                    "candidate_menu_sha256": candidate.sha256,
                    "original_strategy_ids": [left, right],
                    "display_swapped": bool(swapped),
                    "pair_auditor_decisions": [
                        next(
                            item
                            for item in audit["assessment"][
                                "pair_assessments"
                            ]
                            if item["pair_id"] == f"{left}__{right}"
                        )
                        for audit in certification["pair_audits"]
                    ],
                }
            )
    packet.sort(key=lambda row: row["pair_id"])
    private.sort(key=lambda row: row["pair_id"])
    _write_jsonl(packet_path, packet)
    _write_jsonl(private_path, private)
    prompt_report = _prompt_length_report(args.data, module)
    _write_json(prompt_report_path, prompt_report)
    packet_manifest = {
        "schema": "e49e_manual_audit_manifest_v1",
        "identity_sha256": _sha256(
            args.evidence / "frozen_identity.json"
        ),
        "records_sha256": _sha256(
            args.evidence / "trace_bank_records.jsonl"
        ),
        "materialization_manifest_sha256": _sha256(
            args.data / "MATERIALIZATION_MANIFEST.json"
        ),
        "packet_seed": PACKET_SEED,
        "pair_count": len(packet),
        "packet_sha256": _sha256(packet_path),
        "private_key_sha256": _sha256(private_path),
        "prompt_length_audit_sha256": _sha256(prompt_report_path),
        "multi_strategy_menu_count": manifest[
            "multi_strategy_menu_count"
        ],
    }
    _write_json(packet_manifest_path, packet_manifest)
    print(json.dumps(packet_manifest, indent=2, sort_keys=True))


def finalize(args) -> None:
    identity, manifest, _, records, rows = _validated_state(
        args.evidence,
        args.data,
        args.source,
    )
    packet_path = args.evidence / "manual_audit_packet.jsonl"
    packet_manifest_path = args.evidence / "manual_audit_manifest.json"
    labels_path = args.evidence / "manual_audit_labels.json"
    prompt_path = args.evidence / "prompt_length_audit.json"
    for required in (
        packet_path,
        packet_manifest_path,
        labels_path,
        prompt_path,
    ):
        if not required.is_file():
            raise RuntimeError(f"missing E49E audit artifact: {required}")
    packet_manifest = json.loads(
        packet_manifest_path.read_text(encoding="utf-8")
    )
    if (
        packet_manifest.get("packet_sha256") != _sha256(packet_path)
        or packet_manifest.get("identity_sha256")
        != _sha256(args.evidence / "frozen_identity.json")
    ):
        raise RuntimeError("E49E blinded packet identity changed")
    labels_payload = json.loads(labels_path.read_text(encoding="utf-8"))
    if (
        not isinstance(labels_payload, dict)
        or labels_payload.get("schema")
        != "e49e_blinded_manual_labels_v1"
        or labels_payload.get("blinded_before_private_key") is not True
        or not isinstance(labels_payload.get("auditor"), str)
        or not labels_payload["auditor"].strip()
        or not isinstance(labels_payload.get("labels"), list)
    ):
        raise RuntimeError("invalid E49E manual labels")
    required_fields = {
        "pair_id",
        "route_a_sound_and_self_contained",
        "route_b_sound_and_self_contained",
        "genuinely_distinct_decisive_strategy",
        "rationale",
    }
    labels = {}
    for row in labels_payload["labels"]:
        if (
            not isinstance(row, dict)
            or set(row) != required_fields
            or not isinstance(row["pair_id"], str)
            or row["pair_id"] in labels
            or type(row["route_a_sound_and_self_contained"]) is not bool
            or type(row["route_b_sound_and_self_contained"]) is not bool
            or type(row["genuinely_distinct_decisive_strategy"]) is not bool
            or not isinstance(row["rationale"], str)
            or not row["rationale"].strip()
        ):
            raise RuntimeError("invalid E49E manual label row")
        labels[row["pair_id"]] = row
    expected = {
        row["pair_id"] for row in _read_jsonl(packet_path)
    }
    if set(labels) != expected:
        raise RuntimeError("manual labels do not exactly cover blinded packet")
    false_new = [
        pair_id
        for pair_id, row in labels.items()
        if not (
            row["route_a_sound_and_self_contained"]
            and row["route_b_sound_and_self_contained"]
            and row["genuinely_distinct_decisive_strategy"]
        )
    ]
    prompt_report = json.loads(prompt_path.read_text(encoding="utf-8"))
    counts = {
        "overall_multi": 0,
        "train_multi": 0,
        "eval_multi": 0,
    }
    split_by_id = {row["row_id"]: row["split"] for row in rows}
    for row_id, record in records.items():
        if len(record["menu"]["strategies"]) >= 2:
            counts["overall_multi"] += 1
            counts[f"{split_by_id[row_id]}_multi"] += 1
    checks = {
        "all_100_rows_certified": len(records) == 100,
        "overall_multi_at_least_20": counts["overall_multi"] >= 20,
        "eval_multi_at_least_10": counts["eval_multi"] >= 10,
        "known_invalid_controls_rejected": manifest.get(
            "all_known_invalid_controls_rejected"
        )
        is True,
        "known_equivalent_controls_rejected_as_new": manifest.get(
            "all_known_equivalent_controls_rejected_as_new"
        )
        is True,
        "every_retained_pair_manually_labeled": (
            len(labels) == packet_manifest["pair_count"]
        ),
        "manual_false_new_exactly_zero": not false_new,
        "all_prompts_at_most_2048_tokens": prompt_report.get("pass") is True,
    }
    passed = all(checks.values())
    decision = {
        "schema": "e49e_toy_calibration_decision_v1",
        "pass": passed,
        "advance_to_training": passed,
        "identity_sha256": _sha256(
            args.evidence / "frozen_identity.json"
        ),
        "manual_labels_sha256": _sha256(labels_path),
        "manual_pair_count": len(labels),
        "manual_false_new_count": len(false_new),
        "manual_false_new_pair_ids": sorted(false_new),
        "support_counts": counts,
        "checks": checks,
        "snapshot_tree_sha256": identity["snapshot_tree_sha256"],
    }
    output = args.evidence / "calibration_decision.json"
    if output.exists():
        raise RuntimeError("E49E calibration decision already exists")
    _write_json(output, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "finalize"))
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--data", type=pathlib.Path, required=True)
    parser.add_argument("--evidence", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
