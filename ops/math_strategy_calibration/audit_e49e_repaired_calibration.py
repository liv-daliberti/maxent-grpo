#!/usr/bin/env python3
"""Blindly audit the final repaired E49E toy bank before policy training."""

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

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL = (
    ROOT
    / "var/cache/huggingface/transformers/"
    "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
    "7ae557604adf67be50417f59c2c2f167def9a775"
)
PACKET_SEED = 492231
AUDIT_SCRIPT = pathlib.Path(__file__).resolve()
BASE_CALIBRATION_AMENDMENT = (
    ROOT
    / "paper/preregistration/"
    "e49e_repaired_blinded_calibration_amendment_20260724.md"
)
MANUAL_PRUNING_AMENDMENT = (
    ROOT
    / "paper/preregistration/"
    "e49e_manual_pruning_amendment_20260724.md"
)
V5_CALIBRATION_AMENDMENT = (
    ROOT
    / "paper/preregistration/"
    "e49f_v5_calibration_replay_amendment_20260724.md"
)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _tree_hash(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and "__pycache__" not in candidate.parts
        and candidate.suffix != ".pyc"
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


def _load_repair(identity: dict[str, Any], evidence: pathlib.Path):
    snapshot = pathlib.Path(identity["snapshot_root"])
    if (
        not snapshot.is_dir()
        or _tree_hash(snapshot) != identity["snapshot_tree_sha256"]
    ):
        raise RuntimeError("E49E repair source snapshot identity changed")
    sys.path.insert(0, str(snapshot / "src"))
    is_v5 = (
        identity.get("schema")
        == "e49e_singleton_repair_v5_frozen_identity_v1"
    )
    path = snapshot / "ops/math_strategy_calibration" / (
        "repair_e49e_singleton_gaps_v5_curated.py"
        if is_v5
        else "repair_e49e_singleton_gaps.py"
    )
    spec = importlib.util.spec_from_file_location(
        "frozen_e49e_repair_for_manual_audit",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen E49E repair")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if is_v5:
        v3_contracts = (
            snapshot
            / "ops/math_strategy_calibration/"
            "e49e_curated_singleton_contracts_toy.json"
        )
        v4_contracts = (
            snapshot
            / "ops/math_strategy_calibration/"
            "e49e_curated_singleton_contracts_toy_v4.json"
        )
        v5_contracts = (
            snapshot
            / "ops/math_strategy_calibration/"
            "e49e_curated_singleton_contracts_toy_v5.json"
        )
        v3_records = evidence / "frozen_v3_repair_records.jsonl"
        v4_records = evidence / "frozen_v4_repair_records.jsonl"
        for required in (
            v3_contracts,
            v4_contracts,
            v5_contracts,
            v3_records,
            v4_records,
        ):
            if not required.is_file():
                raise RuntimeError(
                    f"V5 calibration replay input is missing: {required}"
                )
        module._prior_records.update(
            module.impl.base.pipeline._load_latest(v4_records)
        )
        module._configure_v4_validator(
            v3_records=v3_records,
            v3_contracts=v3_contracts,
            v4_contracts=v4_contracts,
        )
        module._configure_impl(v5_contracts)
        module.pipeline = module.impl.base.pipeline
        module._augmentation_selected = (
            module.impl.base._augmentation_selected
        )
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
                    "row_id": module.pipeline.base._row_id(
                        split,
                        index,
                        row,
                    ),
                    "problem": str(row["problem"]),
                    "reference_answer": str(row["answer"]),
                }
            )
    return rows


def _route_payload(
    menu: Any,
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
    records_path = evidence / "final_records.jsonl"
    manifest_path = data / "MATERIALIZATION_MANIFEST.json"
    for required in (
        identity_path,
        summary_path,
        records_path,
        manifest_path,
    ):
        if not required.is_file():
            raise RuntimeError(
                f"E49E repaired calibration is incomplete: {required}"
            )
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        identity.get("schema")
        not in {
            "e49e_singleton_repair_frozen_identity_v1",
            "e49e_singleton_repair_v5_frozen_identity_v1",
        }
        or summary.get("schema") != "e49e_singleton_repair_summary_v1"
        or summary.get("pass") is not True
        or manifest.get("schema")
        != "e49e_repaired_trace_bank_materialization_v1"
        or manifest.get("all_controls_pass") is not True
        or manifest.get("final_records_sha256") != _sha256(records_path)
    ):
        raise RuntimeError("E49E repaired automatic gate failed")
    repair = _load_repair(identity, evidence)
    pipeline = repair.pipeline
    input_path = evidence / "frozen_e49d_input/menu_records.jsonl"
    raw_path = evidence / "frozen_e49e_input/trace_bank_records.jsonl"
    augmentation_path = (
        evidence
        / "frozen_kernel_augmentation/augmentation_records.jsonl"
    )
    for required in (input_path, raw_path, augmentation_path):
        if not required.is_file():
            raise RuntimeError(f"frozen E49E input is missing: {required}")
    if (
        manifest.get("e49d_input_sha256") != _sha256(input_path)
        or manifest.get("raw_trace_records_sha256") != _sha256(raw_path)
        or manifest.get("augmentation_records_sha256")
        != _sha256(augmentation_path)
    ):
        raise RuntimeError("E49E repaired input identity changed")
    if identity["schema"] == "e49e_singleton_repair_frozen_identity_v1":
        if (
            _sha256(input_path) != identity["e49d_input_sha256"]
            or _sha256(raw_path) != identity["raw_trace_records_sha256"]
            or _sha256(augmentation_path)
            != identity["kernel_augmentation_records_sha256"]
        ):
            raise RuntimeError("E49E repaired input identity changed")
    else:
        if (
            _tree_hash(evidence / "frozen_e49d_input")
            != identity["frozen_e49d_tree_sha256"]
            or _tree_hash(evidence / "frozen_e49e_input")
            != identity["frozen_e49e_tree_sha256"]
            or _tree_hash(evidence / "frozen_kernel_augmentation")
            != identity["frozen_augmentation_tree_sha256"]
        ):
            raise RuntimeError("E49E V5 input tree identity changed")

    input_records = pipeline._load_latest(input_path)
    raw_records = pipeline._load_latest(raw_path)
    augmentation_records = pipeline._load_latest(augmentation_path)
    final_records = pipeline._load_latest(records_path)
    repair_records = pipeline._load_latest(
        evidence / "repair_records.jsonl"
    )
    rows = _source_rows(repair, source)
    row_by_id = {row["row_id"]: row for row in rows}
    expected = set(row_by_id)
    if not (
        set(input_records)
        == set(raw_records)
        == set(augmentation_records)
        == set(final_records)
        == expected
    ):
        raise RuntimeError("E49E repaired records do not cover source")

    data_splits = {}
    for split in ("train", "eval"):
        dataset_dict = load_from_disk(str(data / split))
        data_splits[split] = dataset_dict[next(iter(dataset_dict))]

    for row_id, row in row_by_id.items():
        record = final_records[row_id]
        if (
            record.get("schema")
            != "e49e_repaired_trace_bank_record_v1"
            or record.get("repair_version") != repair.REPAIR_VERSION
            or record.get("pass") is not True
            or record.get("row_id") != row_id
            or record.get("split") != row["split"]
            or record.get("raw_record_sha256")
            != _canonical_sha256(raw_records[row_id])
        ):
            raise RuntimeError(f"E49E final row identity failed: {row_id}")
        menu = pipeline._menu_from_payload(record["menu"])
        if record.get("menu_sha256") != menu.sha256:
            raise RuntimeError(f"E49E final menu hash failed: {row_id}")
        rendered = data_splits[row["split"]][row["index"]]
        rendered_menu = pipeline.parse_strategy_menu(
            str(rendered["problem"])
        )
        if (
            rendered_menu is None
            or rendered_menu.sha256 != menu.sha256
            or str(rendered.get("original_problem") or "")
            != row["problem"]
        ):
            raise RuntimeError(f"E49E rendered menu failed: {row_id}")

        if "repair_record_sha256" in record:
            repaired = repair_records.get(row_id)
            if (
                repaired is None
                or record.get("repair_record_sha256")
                != _canonical_sha256(repaired)
                or not repair._repair_record_passes(
                    repaired,
                    row_id=row_id,
                    problem=row["problem"],
                    reference_answer=row["reference_answer"],
                )
                or repaired.get("menu_sha256") != menu.sha256
            ):
                raise RuntimeError(
                    f"E49E singleton repair replay failed: {row_id}"
                )
            continue

        augmentation = augmentation_records[row_id]
        if (
            record.get("augmentation_record_sha256")
            != _canonical_sha256(augmentation)
        ):
            raise RuntimeError(
                f"E49E augmentation binding failed: {row_id}"
            )
        selected = repair._augmentation_selected(
            augmentation,
            raw_record=raw_records[row_id],
            input_record=input_records[row_id],
            problem=row["problem"],
            reference_answer=row["reference_answer"],
        )
        if (
            selected is None
            or selected[0].sha256 != menu.sha256
            or not isinstance(record.get("certification"), dict)
        ):
            raise RuntimeError(
                f"E49E certification replay failed: {row_id}"
            )
    return (
        identity,
        manifest,
        repair,
        final_records,
        raw_records,
        input_records,
        rows,
    )


def _prompt_length_report(data: pathlib.Path, pipeline: Any) -> dict[str, Any]:
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
            menu = pipeline.parse_strategy_menu(problem)
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
        "schema": "e49e_repaired_prompt_length_audit_v1",
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


def _packet_row(
    *,
    pair_id: str,
    problem: str,
    reference_answer: str,
    routes: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "e49e_repaired_blinded_manual_pair_v1",
        "pair_id": pair_id,
        "problem": problem,
        "reference_answer": reference_answer,
        "route_a": routes[0],
        "route_b": routes[1],
        "questions": {
            "route_a_sound_and_self_contained": "boolean",
            "route_b_sound_and_self_contained": "boolean",
            "genuinely_distinct_decisive_strategy": "boolean",
            "rationale": "nonempty string",
        },
    }


def _maximum_manual_clique(
    retained_ids: list[str],
    sound_ids: set[str],
    distinct_edges: set[frozenset[str]],
) -> tuple[str, ...]:
    """Choose the deterministic largest bank certified by manual labels."""

    eligible = [
        strategy_id
        for strategy_id in retained_ids
        if strategy_id in sound_ids
    ]
    for size in range(len(eligible), 0, -1):
        for candidate in itertools.combinations(eligible, size):
            if all(
                frozenset((left, right)) in distinct_edges
                for left, right in itertools.combinations(candidate, 2)
            ):
                return candidate
    return ()


def prepare(args: Any) -> None:
    (
        identity,
        manifest,
        repair,
        records,
        raw_records,
        input_records,
        rows,
    ) = _validated_state(args.evidence, args.data, args.source)
    if not (
        BASE_CALIBRATION_AMENDMENT.is_file()
        and MANUAL_PRUNING_AMENDMENT.is_file()
        and V5_CALIBRATION_AMENDMENT.is_file()
    ):
        raise RuntimeError("E49E calibration amendment is missing")
    pipeline = repair.pipeline
    packet_path = args.evidence / "manual_audit_packet.jsonl"
    private_path = args.evidence / "private/manual_audit_key.jsonl"
    packet_manifest_path = args.evidence / "manual_audit_manifest.json"
    prompt_report_path = args.evidence / "prompt_length_audit.json"
    if any(
        path.exists()
        for path in (packet_path, private_path, packet_manifest_path)
    ):
        raise RuntimeError("E49E repaired manual packet already exists")
    row_by_id = {row["row_id"]: row for row in rows}
    packet = []
    private = []

    for row_id in sorted(records):
        record = records[row_id]
        certification = record.get("certification")
        if not isinstance(certification, dict):
            continue
        retained = certification["retained_original_strategy_ids"]
        if len(retained) < 2:
            continue
        candidate = pipeline._menu_from_payload(
            certification["candidate_menu"]
        )
        sound = certification["sound_audits"]
        for left, right in itertools.combinations(retained, 2):
            digest = hashlib.sha256(
                (
                    f"retained\0{row_id}\0{left}\0{right}\0"
                    f"{record['menu_sha256']}"
                ).encode()
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
                _packet_row(
                    pair_id=pair_id,
                    problem=row_by_id[row_id]["problem"],
                    reference_answer=row_by_id[row_id][
                        "reference_answer"
                    ],
                    routes=routes,
                )
            )
            private.append(
                {
                    "pair_id": pair_id,
                    "kind": "retained_distinct_claim",
                    "row_id": row_id,
                    "original_strategy_ids": [left, right],
                    "display_swapped": bool(swapped),
                    "expected_distinct": True,
                }
            )

    equivalent_control_path = (
        repair.HERE / "e49e_known_equivalent_controls_toy.json"
    )
    if not equivalent_control_path.is_file():
        raise RuntimeError("frozen equivalent-control manifest is missing")
    controls = pipeline._load_known_equivalent_controls(
        equivalent_control_path,
        input_records,
    )
    for control in controls:
        row_id = control["row_id"]
        raw = raw_records[row_id]
        certification = raw.get("certification")
        if isinstance(certification, dict):
            candidate_payload = certification["candidate_menu"]
            sound = certification["sound_audits"]
        else:
            candidate_payload = raw["candidate_menu"]
            sound = raw["sound_audits"]
        candidate = pipeline._menu_from_payload(candidate_payload)
        left = control["left_strategy_id"]
        right = control["right_strategy_id"]
        digest = hashlib.sha256(
            (
                f"equivalent\0{row_id}\0{left}\0{right}\0"
                f"{candidate.sha256}"
            ).encode()
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
            _packet_row(
                pair_id=pair_id,
                problem=row_by_id[row_id]["problem"],
                reference_answer=row_by_id[row_id]["reference_answer"],
                routes=routes,
            )
        )
        private.append(
            {
                "pair_id": pair_id,
                "kind": "blinded_equivalent_control",
                "row_id": row_id,
                "original_strategy_ids": [left, right],
                "display_swapped": bool(swapped),
                "expected_distinct": False,
            }
        )

    packet.sort(key=lambda row: row["pair_id"])
    private.sort(key=lambda row: row["pair_id"])
    if len({row["pair_id"] for row in packet}) != len(packet):
        raise RuntimeError("manual packet pair IDs collided")
    _write_jsonl(packet_path, packet)
    _write_jsonl(private_path, private)
    prompt_report = _prompt_length_report(args.data, pipeline)
    _write_json(prompt_report_path, prompt_report)
    retained_count = sum(
        row["kind"] == "retained_distinct_claim" for row in private
    )
    equivalent_count = sum(
        row["kind"] == "blinded_equivalent_control" for row in private
    )
    packet_manifest = {
        "schema": "e49e_repaired_manual_audit_manifest_v1",
        "identity_sha256": _sha256(
            args.evidence / "frozen_identity.json"
        ),
        "records_sha256": _sha256(
            args.evidence / "final_records.jsonl"
        ),
        "materialization_manifest_sha256": _sha256(
            args.data / "MATERIALIZATION_MANIFEST.json"
        ),
        "packet_seed": PACKET_SEED,
        "pair_count": len(packet),
        "retained_claim_count": retained_count,
        "blinded_equivalent_control_count": equivalent_count,
        "packet_sha256": _sha256(packet_path),
        "private_key_sha256": _sha256(private_path),
        "prompt_length_audit_sha256": _sha256(prompt_report_path),
        "multi_strategy_menu_count": manifest[
            "multi_strategy_menu_count"
        ],
        "snapshot_tree_sha256": identity["snapshot_tree_sha256"],
        "audit_script_sha256": _sha256(AUDIT_SCRIPT),
        "base_calibration_amendment_sha256": _sha256(
            BASE_CALIBRATION_AMENDMENT
        ),
        "manual_pruning_amendment_sha256": _sha256(
            MANUAL_PRUNING_AMENDMENT
        ),
        "v5_calibration_replay_amendment_sha256": _sha256(
            V5_CALIBRATION_AMENDMENT
        ),
    }
    _write_json(packet_manifest_path, packet_manifest)
    print(json.dumps(packet_manifest, indent=2, sort_keys=True))


def finalize(args: Any) -> None:
    if args.output is None:
        raise RuntimeError("--output is required for finalize")
    if args.output.exists():
        raise RuntimeError(f"audited output already exists: {args.output}")
    (
        identity,
        manifest,
        repair,
        records,
        _,
        _,
        rows,
    ) = _validated_state(args.evidence, args.data, args.source)
    pipeline = repair.pipeline
    packet_path = args.evidence / "manual_audit_packet.jsonl"
    private_path = args.evidence / "private/manual_audit_key.jsonl"
    packet_manifest_path = args.evidence / "manual_audit_manifest.json"
    labels_path = args.evidence / "manual_audit_labels.json"
    prompt_path = args.evidence / "prompt_length_audit.json"
    audited_records_path = args.evidence / "audited_records.jsonl"
    audited_prompt_path = (
        args.evidence / "audited_prompt_length_audit.json"
    )
    decision_path = args.evidence / "calibration_decision.json"
    for new_output in (
        audited_records_path,
        audited_prompt_path,
        decision_path,
    ):
        if new_output.exists():
            raise RuntimeError(
                f"E49E repaired audit output already exists: {new_output}"
            )
    for required in (
        packet_path,
        private_path,
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
        or packet_manifest.get("private_key_sha256")
        != _sha256(private_path)
        or packet_manifest.get("identity_sha256")
        != _sha256(args.evidence / "frozen_identity.json")
        or packet_manifest.get("audit_script_sha256")
        != _sha256(AUDIT_SCRIPT)
        or not BASE_CALIBRATION_AMENDMENT.is_file()
        or packet_manifest.get("base_calibration_amendment_sha256")
        != _sha256(BASE_CALIBRATION_AMENDMENT)
        or not MANUAL_PRUNING_AMENDMENT.is_file()
        or packet_manifest.get("manual_pruning_amendment_sha256")
        != _sha256(MANUAL_PRUNING_AMENDMENT)
        or not V5_CALIBRATION_AMENDMENT.is_file()
        or packet_manifest.get(
            "v5_calibration_replay_amendment_sha256"
        )
        != _sha256(V5_CALIBRATION_AMENDMENT)
    ):
        raise RuntimeError("E49E blinded packet identity changed")
    labels_payload = json.loads(labels_path.read_text(encoding="utf-8"))
    if (
        not isinstance(labels_payload, dict)
        or labels_payload.get("schema")
        != "e49e_repaired_blinded_manual_labels_v1"
        or labels_payload.get("blinded_before_private_key") is not True
        or not isinstance(labels_payload.get("auditor"), str)
        or not labels_payload["auditor"].strip()
        or not isinstance(labels_payload.get("labels"), list)
    ):
        raise RuntimeError("invalid E49E repaired manual labels")
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
            raise RuntimeError("invalid E49E repaired manual label row")
        labels[row["pair_id"]] = row
    expected = {row["pair_id"] for row in _read_jsonl(packet_path)}
    if set(labels) != expected:
        raise RuntimeError("manual labels do not cover blinded packet")
    private = {
        row["pair_id"]: row for row in _read_jsonl(private_path)
    }
    if set(private) != expected:
        raise RuntimeError("private key does not cover blinded packet")

    rejected_claims = []
    equivalent_control_errors = []
    unsound_by_row: dict[str, set[str]] = {}
    distinct_by_row: dict[str, set[frozenset[str]]] = {}
    for pair_id, key in private.items():
        label = labels[pair_id]
        both_sound = bool(
            label["route_a_sound_and_self_contained"]
            and label["route_b_sound_and_self_contained"]
        )
        distinct = label["genuinely_distinct_decisive_strategy"]
        if key["kind"] == "retained_distinct_claim":
            left, right = key["original_strategy_ids"]
            route_a, route_b = (
                (right, left)
                if key["display_swapped"]
                else (left, right)
            )
            row_id = key["row_id"]
            unsound = unsound_by_row.setdefault(row_id, set())
            if not label["route_a_sound_and_self_contained"]:
                unsound.add(route_a)
            if not label["route_b_sound_and_self_contained"]:
                unsound.add(route_b)
            if both_sound and distinct:
                distinct_by_row.setdefault(row_id, set()).add(
                    frozenset((left, right))
                )
            if not (both_sound and distinct):
                rejected_claims.append(pair_id)
        elif not (both_sound and not distinct):
            equivalent_control_errors.append(pair_id)

    # The prepared report remains identity-bound evidence for the unpruned
    # packet. The final advancement check below is recomputed on the pruned
    # dataset that would actually be shown to the policy.
    json.loads(prompt_path.read_text(encoding="utf-8"))
    split_by_id = {row["row_id"]: row["split"] for row in rows}
    audited_records: dict[str, dict[str, Any]] = {}
    zero_support_rows = []
    for row_id, source_record in records.items():
        source_menu = pipeline._menu_from_payload(source_record["menu"])
        certification = source_record.get("certification")
        selected_ids: tuple[str, ...] | None = None
        dropped_ids: list[str] = []
        if isinstance(certification, dict):
            retained = list(
                certification["retained_original_strategy_ids"]
            )
            if len(retained) >= 2:
                selected_ids = _maximum_manual_clique(
                    retained,
                    set(retained) - unsound_by_row.get(row_id, set()),
                    distinct_by_row.get(row_id, set()),
                )
            else:
                selected_ids = tuple(retained)
            dropped_ids = [
                strategy_id
                for strategy_id in retained
                if strategy_id not in selected_ids
            ]
            if selected_ids:
                candidate = pipeline._menu_from_payload(
                    certification["candidate_menu"]
                )
                audited_menu = pipeline._prune_menu_closed(
                    candidate,
                    selected_ids,
                )
            else:
                audited_menu = None
        else:
            # Reference-bound repair rows are singletons with two independent
            # execution audits and have no distinctness claim to prune.
            audited_menu = source_menu

        if audited_menu is None:
            zero_support_rows.append(row_id)
        audited_records[row_id] = {
            "schema": "e49e_manually_audited_trace_bank_record_v1",
            "row_id": row_id,
            "split": split_by_id[row_id],
            "origin": source_record["origin"],
            "source_record_sha256": _canonical_sha256(source_record),
            "manual_audit_applied": bool(
                isinstance(certification, dict)
                and len(
                    certification["retained_original_strategy_ids"]
                )
                >= 2
            ),
            "retained_original_strategy_ids": (
                list(selected_ids) if selected_ids is not None else None
            ),
            "manually_dropped_original_strategy_ids": dropped_ids,
            "menu": (
                json.loads(audited_menu.canonical_json)
                if audited_menu is not None
                else None
            ),
            "menu_sha256": (
                audited_menu.sha256 if audited_menu is not None else None
            ),
            "pass": audited_menu is not None,
        }

    _write_jsonl(
        audited_records_path,
        [audited_records[row["row_id"]] for row in rows],
    )
    final_retained_false_new = []
    for pair_id in rejected_claims:
        key = private[pair_id]
        retained_after_audit = set(
            audited_records[key["row_id"]][
                "retained_original_strategy_ids"
            ]
            or []
        )
        if set(key["original_strategy_ids"]) <= retained_after_audit:
            final_retained_false_new.append(pair_id)

    counts = {"overall_multi": 0, "train_multi": 0, "eval_multi": 0}
    for row_id, record in audited_records.items():
        if (
            isinstance(record["menu"], dict)
            and len(record["menu"]["strategies"]) >= 2
        ):
            counts["overall_multi"] += 1
            counts[f"{split_by_id[row_id]}_multi"] += 1

    audited_prompt_report: dict[str, Any]
    if not zero_support_rows:
        source_train = load_from_disk(str(args.source / "train"))
        source_eval = load_from_disk(str(args.source / "eval"))
        train_name = next(iter(source_train))
        eval_name = next(iter(source_eval))
        source_splits = {
            "train": source_train[train_name],
            "eval": source_eval[eval_name],
        }
        output_splits = {}
        for split, dataset in source_splits.items():
            data = dataset.to_dict()
            augmented = []
            menu_hashes = []
            origins = []
            for index, row in enumerate(dataset):
                row_id = pipeline.base._row_id(split, index, row)
                record = audited_records[row_id]
                augmented.append(
                    pipeline._embed(str(row["problem"]), record["menu"])
                )
                menu_hashes.append(record["menu_sha256"])
                origins.append(record["origin"])
            data["original_problem"] = list(data["problem"])
            data["problem"] = augmented
            data["strategy_menu_sha256"] = menu_hashes
            data["strategy_menu_origin"] = origins
            output_splits[split] = Dataset.from_dict(data)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        staging = pathlib.Path(
            tempfile.mkdtemp(
                prefix=f".{args.output.name}.",
                dir=args.output.parent,
            )
        )
        try:
            DatasetDict(
                {train_name: output_splits["train"]}
            ).save_to_disk(str(staging / "train"))
            DatasetDict(
                {eval_name: output_splits["eval"]}
            ).save_to_disk(str(staging / "eval"))
            audited_manifest = {
                "schema": "e49e_manually_audited_materialization_v1",
                "source_repaired_manifest_sha256": _sha256(
                    args.data / "MATERIALIZATION_MANIFEST.json"
                ),
                "manual_labels_sha256": _sha256(labels_path),
                "manual_packet_sha256": _sha256(packet_path),
                "manual_private_key_sha256": _sha256(private_path),
                "audited_records_sha256": _sha256(
                    audited_records_path
                ),
                "menu_count": len(audited_records),
                "multi_strategy_menu_count": counts["overall_multi"],
                "train_multi_strategy_menu_count": counts[
                    "train_multi"
                ],
                "eval_multi_strategy_menu_count": counts["eval_multi"],
                "manual_rejected_claim_count": len(rejected_claims),
                "train_tree_sha256": _tree_hash(staging / "train"),
                "eval_tree_sha256": _tree_hash(staging / "eval"),
            }
            _write_json(
                staging / "MATERIALIZATION_MANIFEST.json",
                audited_manifest,
            )
            os.replace(staging, args.output)
        finally:
            if staging.exists():
                for item in sorted(staging.rglob("*"), reverse=True):
                    if item.is_file():
                        item.unlink()
                    else:
                        item.rmdir()
                staging.rmdir()
        audited_prompt_report = _prompt_length_report(
            args.output,
            pipeline,
        )
    else:
        audited_prompt_report = {
            "schema": "e49e_repaired_prompt_length_audit_v1",
            "pass": False,
            "reason": "zero_support_rows_prevented_materialization",
            "zero_support_rows": sorted(zero_support_rows),
        }
    _write_json(audited_prompt_path, audited_prompt_report)

    invalid_controls = manifest.get("known_invalid_control_results") or []
    equivalent_controls = (
        manifest.get("known_equivalent_control_results") or []
    )
    checks = {
        "all_100_rows_certified": len(records) == 100,
        "no_zero_support_rows_after_manual_pruning": (
            not zero_support_rows
        ),
        "overall_multi_at_least_20": counts["overall_multi"] >= 20,
        "eval_multi_at_least_10": counts["eval_multi"] >= 10,
        "known_invalid_controls_rejected": (
            len(invalid_controls) == 5
            and all(
                row.get("rejected_by_soundness") is True
                for row in invalid_controls
            )
        ),
        "known_equivalent_controls_rejected_as_new": (
            len(equivalent_controls) == 3
            and all(
                row.get("rejected_as_new") is True
                for row in equivalent_controls
            )
        ),
        "every_blinded_pair_manually_labeled": (
            len(labels) == packet_manifest["pair_count"]
        ),
        "final_retained_pairs_all_manually_distinct": all(
            frozenset(edge)
            in distinct_by_row.get(row_id, set())
            for row_id, record in audited_records.items()
            for edge in itertools.combinations(
                record["retained_original_strategy_ids"] or [],
                2,
            )
        )
        and not final_retained_false_new,
        "manual_equivalent_controls_all_recognized": (
            not equivalent_control_errors
            and packet_manifest["blinded_equivalent_control_count"] >= 3
        ),
        "all_prompts_at_most_2048_tokens": (
            audited_prompt_report.get("pass") is True
        ),
    }
    passed = all(checks.values())
    decision = {
        "schema": "e49e_repaired_toy_calibration_decision_v1",
        "pass": passed,
        "advance_to_training": passed,
        "identity_sha256": _sha256(
            args.evidence / "frozen_identity.json"
        ),
        "manual_labels_sha256": _sha256(labels_path),
        "manual_pair_count": len(labels),
        "preprune_manual_false_new_count": len(rejected_claims),
        "preprune_manual_false_new_rate": (
            len(rejected_claims)
            / max(1, packet_manifest["retained_claim_count"])
        ),
        "manual_rejected_claim_count": len(rejected_claims),
        "manual_rejected_claim_pair_ids": sorted(rejected_claims),
        "manual_retained_false_new_count": len(
            final_retained_false_new
        ),
        "manual_retained_false_new_pair_ids": sorted(
            final_retained_false_new
        ),
        "manual_equivalent_control_error_count": len(
            equivalent_control_errors
        ),
        "manual_equivalent_control_error_pair_ids": sorted(
            equivalent_control_errors
        ),
        "zero_support_rows_after_manual_pruning": sorted(
            zero_support_rows
        ),
        "support_counts": counts,
        "checks": checks,
        "snapshot_tree_sha256": identity["snapshot_tree_sha256"],
        "audit_script_sha256": _sha256(AUDIT_SCRIPT),
        "base_calibration_amendment_sha256": _sha256(
            BASE_CALIBRATION_AMENDMENT
        ),
        "manual_pruning_amendment_sha256": _sha256(
            MANUAL_PRUNING_AMENDMENT
        ),
        "v5_calibration_replay_amendment_sha256": _sha256(
            V5_CALIBRATION_AMENDMENT
        ),
        "audited_records_sha256": _sha256(audited_records_path),
        "audited_prompt_length_audit_sha256": _sha256(
            audited_prompt_path
        ),
        "audited_data": str(args.output) if args.output.exists() else None,
    }
    _write_json(decision_path, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "finalize"))
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--data", type=pathlib.Path, required=True)
    parser.add_argument("--evidence", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
