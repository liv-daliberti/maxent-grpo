#!/usr/bin/env python3
"""Prepare, manually audit, and materialize E49R's combined toy bank."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import pathlib
import random
import tempfile
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk


ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
CERTIFIER_PATH = HERE / "certify_e49h_curated_routes.py"
AUDIT_BASE_PATH = HERE / "audit_e49e_repaired_calibration.py"
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e49r_combined_manual_audit_20260724.md"
)
PACKET_SEED = 492271
AUDIT_SCRIPT = pathlib.Path(__file__).resolve()


def _load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import E49R dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


certifier = _load_module("e49r_curated_certifier", CERTIFIER_PATH)
audit_base = _load_module("e49r_base_audit", AUDIT_BASE_PATH)
pipeline = certifier.e49e


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


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
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
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _source_rows(source: pathlib.Path) -> dict[str, dict[str, Any]]:
    return certifier._source_rows(source)


def _base_records(
    base_evidence: pathlib.Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    records_path = base_evidence / "audited_records.jsonl"
    decision_path = base_evidence / "calibration_decision.json"
    if not records_path.is_file() or not decision_path.is_file():
        raise RuntimeError("E49R base audited bank is incomplete")
    records = {
        row["row_id"]: row for row in _read_jsonl(records_path)
    }
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if (
        len(records) != 100
        or decision.get("checks", {}).get("all_100_rows_certified")
        is not True
        or decision.get("manual_retained_false_new_count") != 0
        or decision.get("support_counts")
        != {"overall_multi": 4, "train_multi": 2, "eval_multi": 2}
        or _sha256(records_path)
        != "b945211f0baa20800e322958ef9095af034c354cc123043fcaa2eaf63ef9e484"
    ):
        raise RuntimeError("E49R base audited bank identity failed")
    return records, decision


def _strict_survivors(
    evidence_paths: list[pathlib.Path],
) -> list[dict[str, Any]]:
    survivors = []
    for evidence in evidence_paths:
        summary_path = evidence / "run_summary.json"
        records_path = evidence / "candidate_records.jsonl"
        identity_path = evidence / "frozen_identity.json"
        for required in (summary_path, records_path, identity_path):
            if not required.is_file():
                raise RuntimeError(
                    f"E49R strict evidence incomplete: {required}"
                )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("all_requests_complete") is not True
            or summary.get("candidate_records_sha256")
            != _sha256(records_path)
        ):
            raise RuntimeError(
                f"E49R strict evidence identity failed: {evidence}"
            )
        for record in _read_jsonl(records_path):
            if record.get("pass") is True:
                survivors.append(
                    {
                        "source_kind": "strict_survivor",
                        "source_evidence": str(evidence),
                        "source_identity_sha256": _sha256(identity_path),
                        "source_record_sha256": _canonical_sha256(record),
                        "record": record,
                    }
                )
    return survivors


def _recovery_survivors(evidence: pathlib.Path) -> list[dict[str, Any]]:
    summary_path = evidence / "run_summary.json"
    records_path = evidence / "recovered_records.jsonl"
    identity_path = evidence / "frozen_identity.json"
    for required in (summary_path, records_path, identity_path):
        if not required.is_file():
            raise RuntimeError(f"E49R recovery evidence missing: {required}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("all_requests_complete") is not True
        or summary.get("recovered_records_sha256") != _sha256(records_path)
        or summary.get("passing_candidate_count") != 4
    ):
        raise RuntimeError("E49R recovery evidence identity failed")
    return [
        {
            "source_kind": "recovered_survivor",
            "source_evidence": str(evidence),
            "source_identity_sha256": _sha256(identity_path),
            "source_record_sha256": _canonical_sha256(record),
            "record": record,
        }
        for record in _read_jsonl(records_path)
        if record.get("pass") is True
    ]


def _packet_row(
    *,
    pair_id: str,
    problem: str,
    reference_answer: str,
    routes: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "e49r_blinded_manual_pair_v1",
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


def _equivalent_copy(route: dict[str, Any]) -> dict[str, Any]:
    cloned = copy.deepcopy(route)
    cloned["plan"] = (
        "[EQUIVALENT RENDERING] Execute the same declared actions in their "
        "listed order. " + str(cloned["plan"])
    )
    return cloned


def _singleton_payload(
    *,
    e49h_evidence: pathlib.Path,
    source_rows: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    row_id = "eval:0010:834bdfcea94c2ee5ca2f"
    records = {
        row["row_id"]: row
        for row in _read_jsonl(
            e49h_evidence / "candidate_records.jsonl"
        )
    }
    record = records[row_id]
    audits = record["sound_audits"]["S2"]
    if not (
        len(audits) == 2
        and all(audit.get("pass") is True for audit in audits)
    ):
        raise RuntimeError("E49R singleton repair lost its two exact traces")
    menu = pipeline._menu_from_payload(record["menu"])
    route = certifier._route_payload(menu, "S2", audits)
    return (
        {
            "problem": source_rows[row_id]["problem"],
            "reference_answer": source_rows[row_id]["reference_answer"],
            "routes": [route, _equivalent_copy(route)],
        },
        {
            "row_id": row_id,
            "menu": record["menu"],
            "menu_sha256": record["menu_sha256"],
            "strategy_id": "S2",
            "source_record_sha256": _canonical_sha256(record),
        },
    )


def prepare(args: argparse.Namespace) -> None:
    if not PROTOCOL.is_file() or "FROZEN BEFORE E49R PACKET" not in (
        PROTOCOL.read_text(encoding="utf-8")
    ):
        raise RuntimeError("E49R protocol is not frozen")
    packet_path = args.evidence / "manual_audit_packet.jsonl"
    private_path = args.evidence / "private/manual_audit_key.jsonl"
    manifest_path = args.evidence / "manual_audit_manifest.json"
    if any(path.exists() for path in (packet_path, private_path, manifest_path)):
        raise RuntimeError("fresh E49R packet directory required")
    base_records, _ = _base_records(args.base_evidence)
    rows = _source_rows(args.source)
    if set(base_records) != set(rows):
        raise RuntimeError("E49R source rows do not match base bank")
    claims = _strict_survivors(args.strict_evidence)
    claims.extend(_recovery_survivors(args.recovery_evidence))
    if (
        len(claims) != 20
        or len({claim["record"]["row_id"] for claim in claims}) != 20
        or sum(
            claim["record"]["split"] == "train" for claim in claims
        )
        != 11
        or sum(claim["record"]["split"] == "eval" for claim in claims)
        != 9
    ):
        raise RuntimeError("E49R frozen claim cohort changed")

    packet = []
    private = []
    claim_packet_rows = []
    for claim in claims:
        record = claim["record"]
        pair = record["pair_payload"]
        digest = hashlib.sha256(
            (
                f"claim\0{claim['source_identity_sha256']}\0"
                f"{record['row_id']}\0{record['menu_sha256']}"
            ).encode("utf-8")
        ).hexdigest()
        pair_id = f"PAIR_{digest[:20]}"
        routes = [pair["route_a"], pair["route_b"]]
        swapped = random.Random(
            PACKET_SEED + int(digest[:12], 16)
        ).randrange(2)
        if swapped:
            routes.reverse()
        packet_row = _packet_row(
            pair_id=pair_id,
            problem=pair["problem"],
            reference_answer=pair["reference_answer"],
            routes=routes,
        )
        packet.append(packet_row)
        claim_packet_rows.append(packet_row)
        private.append(
            {
                "pair_id": pair_id,
                "kind": "retained_distinct_claim",
                "row_id": record["row_id"],
                "split": record["split"],
                "menu": record["menu"],
                "menu_sha256": record["menu_sha256"],
                "display_swapped": bool(swapped),
                "source_kind": claim["source_kind"],
                "source_evidence": claim["source_evidence"],
                "source_identity_sha256": claim[
                    "source_identity_sha256"
                ],
                "source_record_sha256": claim[
                    "source_record_sha256"
                ],
            }
        )

    singleton, singleton_key = _singleton_payload(
        e49h_evidence=args.e49h_evidence,
        source_rows=rows,
    )
    singleton_digest = hashlib.sha256(
        (
            f"singleton\0{singleton_key['row_id']}\0"
            f"{singleton_key['menu_sha256']}"
        ).encode("utf-8")
    ).hexdigest()
    singleton_pair_id = f"PAIR_{singleton_digest[:20]}"
    singleton_routes = singleton["routes"]
    singleton_swapped = random.Random(
        PACKET_SEED + int(singleton_digest[:12], 16)
    ).randrange(2)
    if singleton_swapped:
        singleton_routes.reverse()
    packet.append(
        _packet_row(
            pair_id=singleton_pair_id,
            problem=singleton["problem"],
            reference_answer=singleton["reference_answer"],
            routes=singleton_routes,
        )
    )
    private.append(
        {
            "pair_id": singleton_pair_id,
            "kind": "singleton_repair_control",
            **singleton_key,
            "display_swapped": bool(singleton_swapped),
        }
    )

    control_sources = random.Random(PACKET_SEED).sample(
        sorted(claim_packet_rows, key=lambda row: row["pair_id"]),
        5,
    )
    for index, source_row in enumerate(control_sources):
        digest = hashlib.sha256(
            (
                f"control\0{index}\0{source_row['pair_id']}\0"
                f"{PACKET_SEED}"
            ).encode("utf-8")
        ).hexdigest()
        pair_id = f"PAIR_{digest[:20]}"
        route = source_row["route_a"]
        routes = [route, _equivalent_copy(route)]
        swapped = random.Random(
            PACKET_SEED + int(digest[:12], 16)
        ).randrange(2)
        if swapped:
            routes.reverse()
        packet.append(
            _packet_row(
                pair_id=pair_id,
                problem=source_row["problem"],
                reference_answer=source_row["reference_answer"],
                routes=routes,
            )
        )
        private.append(
            {
                "pair_id": pair_id,
                "kind": "blinded_equivalent_control",
                "source_pair_id": source_row["pair_id"],
                "display_swapped": bool(swapped),
            }
        )

    packet.sort(key=lambda row: row["pair_id"])
    private.sort(key=lambda row: row["pair_id"])
    if (
        len(packet) != 26
        or len(private) != 26
        or len({row["pair_id"] for row in packet}) != 26
    ):
        raise RuntimeError("E49R packet identity/count failed")
    _write_jsonl(packet_path, packet)
    _write_jsonl(private_path, private)
    manifest = {
        "schema": "e49r_manual_audit_manifest_v1",
        "packet_seed": PACKET_SEED,
        "pair_count": 26,
        "retained_claim_count": 20,
        "singleton_repair_control_count": 1,
        "blinded_equivalent_control_count": 5,
        "packet_sha256": _sha256(packet_path),
        "private_key_sha256": _sha256(private_path),
        "protocol_sha256": _sha256(PROTOCOL),
        "audit_script_sha256": _sha256(AUDIT_SCRIPT),
        "base_audited_records_sha256": _sha256(
            args.base_evidence / "audited_records.jsonl"
        ),
        "strict_source_identity_sha256s": sorted(
            {
                row["source_identity_sha256"]
                for row in private
                if row["kind"] == "retained_distinct_claim"
                and row["source_kind"] == "strict_survivor"
            }
        ),
        "recovery_source_identity_sha256s": sorted(
            {
                row["source_identity_sha256"]
                for row in private
                if row["kind"] == "retained_distinct_claim"
                and row["source_kind"] == "recovered_survivor"
            }
        ),
    }
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _validated_labels(
    labels_path: pathlib.Path,
    expected: set[str],
) -> dict[str, dict[str, Any]]:
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "e49r_blinded_manual_labels_v1"
        or payload.get("blinded_before_private_key") is not True
        or not isinstance(payload.get("auditor"), str)
        or not payload["auditor"].strip()
        or not isinstance(payload.get("labels"), list)
    ):
        raise RuntimeError("invalid E49R manual-label envelope")
    required = {
        "pair_id",
        "route_a_sound_and_self_contained",
        "route_b_sound_and_self_contained",
        "genuinely_distinct_decisive_strategy",
        "rationale",
    }
    labels = {}
    for row in payload["labels"]:
        if (
            not isinstance(row, dict)
            or set(row) != required
            or row["pair_id"] in labels
            or type(row["route_a_sound_and_self_contained"]) is not bool
            or type(row["route_b_sound_and_self_contained"]) is not bool
            or type(row["genuinely_distinct_decisive_strategy"]) is not bool
            or not isinstance(row["rationale"], str)
            or not row["rationale"].strip()
        ):
            raise RuntimeError("invalid E49R manual-label row")
        labels[row["pair_id"]] = row
    if set(labels) != expected:
        raise RuntimeError("E49R labels do not exactly cover packet")
    return labels


def _materialize(
    *,
    source: pathlib.Path,
    output: pathlib.Path,
    records: dict[str, dict[str, Any]],
    labels_path: pathlib.Path,
    packet_path: pathlib.Path,
    private_path: pathlib.Path,
) -> dict[str, Any]:
    source_train = load_from_disk(str(source / "train"))
    source_eval = load_from_disk(str(source / "eval"))
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
        hashes = []
        origins = []
        for index, row in enumerate(dataset):
            row_id = pipeline.base._row_id(split, index, row)
            record = records[row_id]
            augmented.append(
                pipeline._embed(str(row["problem"]), record["menu"])
            )
            hashes.append(record["menu_sha256"])
            origins.append(record["origin"])
        data["original_problem"] = list(data["problem"])
        data["problem"] = augmented
        data["strategy_menu_sha256"] = hashes
        data["strategy_menu_origin"] = origins
        output_splits[split] = Dataset.from_dict(data)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        DatasetDict({train_name: output_splits["train"]}).save_to_disk(
            str(staging / "train")
        )
        DatasetDict({eval_name: output_splits["eval"]}).save_to_disk(
            str(staging / "eval")
        )
        manifest = {
            "schema": "e49r_combined_audited_materialization_v1",
            "manual_labels_sha256": _sha256(labels_path),
            "manual_packet_sha256": _sha256(packet_path),
            "manual_private_key_sha256": _sha256(private_path),
            "menu_count": len(records),
            "train_tree_sha256": audit_base._tree_hash(
                staging / "train"
            ),
            "eval_tree_sha256": audit_base._tree_hash(staging / "eval"),
        }
        _write_json(staging / "MATERIALIZATION_MANIFEST.json", manifest)
        os.replace(staging, output)
    finally:
        if staging.exists():
            for item in sorted(staging.rglob("*"), reverse=True):
                if item.is_file():
                    item.unlink()
                else:
                    item.rmdir()
            staging.rmdir()
    return manifest


def finalize(args: argparse.Namespace) -> None:
    if args.output is None:
        raise RuntimeError("--output is required for finalize")
    if args.output.exists():
        raise RuntimeError(f"fresh E49R output required: {args.output}")
    packet_path = args.evidence / "manual_audit_packet.jsonl"
    private_path = args.evidence / "private/manual_audit_key.jsonl"
    manifest_path = args.evidence / "manual_audit_manifest.json"
    labels_path = args.evidence / "manual_audit_labels.json"
    records_path = args.evidence / "audited_records.jsonl"
    prompt_path = args.evidence / "prompt_length_audit.json"
    decision_path = args.evidence / "advancement_decision.json"
    for required in (
        packet_path,
        private_path,
        manifest_path,
        labels_path,
    ):
        if not required.is_file():
            raise RuntimeError(f"missing E49R audit artifact: {required}")
    if any(path.exists() for path in (records_path, prompt_path, decision_path)):
        raise RuntimeError("fresh E49R final evidence required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("packet_sha256") != _sha256(packet_path)
        or manifest.get("private_key_sha256") != _sha256(private_path)
        or manifest.get("protocol_sha256") != _sha256(PROTOCOL)
        or manifest.get("audit_script_sha256") != _sha256(AUDIT_SCRIPT)
    ):
        raise RuntimeError("E49R packet identity changed")
    expected = {row["pair_id"] for row in _read_jsonl(packet_path)}
    labels = _validated_labels(labels_path, expected)
    private = {
        row["pair_id"]: row for row in _read_jsonl(private_path)
    }
    if set(private) != expected:
        raise RuntimeError("E49R private key does not cover packet")

    accepted_claims = []
    false_new = []
    unsound_claims = []
    control_errors = []
    singleton_ok = False
    singleton_key = None
    for pair_id, key in private.items():
        label = labels[pair_id]
        both_sound = bool(
            label["route_a_sound_and_self_contained"]
            and label["route_b_sound_and_self_contained"]
        )
        distinct = label["genuinely_distinct_decisive_strategy"]
        if key["kind"] == "retained_distinct_claim":
            if not both_sound:
                unsound_claims.append(pair_id)
            elif not distinct:
                false_new.append(pair_id)
            else:
                accepted_claims.append(key)
        elif key["kind"] == "blinded_equivalent_control":
            if not (both_sound and not distinct):
                control_errors.append(pair_id)
        elif key["kind"] == "singleton_repair_control":
            singleton_ok = bool(both_sound and not distinct)
            singleton_key = key
            if not singleton_ok:
                control_errors.append(pair_id)

    base_records, _ = _base_records(args.base_evidence)
    final_records = copy.deepcopy(base_records)
    for key in accepted_claims:
        final_records[key["row_id"]] = {
            "schema": "e49r_combined_audited_record_v1",
            "row_id": key["row_id"],
            "split": key["split"],
            "origin": key["source_kind"],
            "source_record_sha256": key["source_record_sha256"],
            "menu": key["menu"],
            "menu_sha256": key["menu_sha256"],
            "pass": True,
            "manual_audit_applied": True,
        }
    if singleton_ok and singleton_key is not None:
        full_menu = pipeline._menu_from_payload(singleton_key["menu"])
        singleton_menu = pipeline._prune_menu_closed(
            full_menu,
            (singleton_key["strategy_id"],),
        )
        final_records[singleton_key["row_id"]] = {
            "schema": "e49r_combined_audited_record_v1",
            "row_id": singleton_key["row_id"],
            "split": "eval",
            "origin": "manually_audited_singleton_repair",
            "source_record_sha256": singleton_key[
                "source_record_sha256"
            ],
            "menu": json.loads(singleton_menu.canonical_json),
            "menu_sha256": singleton_menu.sha256,
            "pass": True,
            "manual_audit_applied": True,
        }

    zero_support = [
        row_id
        for row_id, record in final_records.items()
        if not isinstance(record.get("menu"), dict)
        or not record["menu"].get("strategies")
    ]
    counts = {"train_multi": 0, "eval_multi": 0, "overall_multi": 0}
    for record in final_records.values():
        if (
            isinstance(record.get("menu"), dict)
            and len(record["menu"]["strategies"]) >= 2
        ):
            counts[f"{record['split']}_multi"] += 1
            counts["overall_multi"] += 1
    _write_jsonl(
        records_path,
        [final_records[row_id] for row_id in sorted(final_records)],
    )

    materialization = None
    prompt_report = {
        "schema": "e49r_prompt_length_audit_v1",
        "pass": False,
        "reason": "zero_support_prevented_materialization",
        "zero_support_rows": zero_support,
    }
    if not zero_support:
        materialization = _materialize(
            source=args.source,
            output=args.output,
            records=final_records,
            labels_path=labels_path,
            packet_path=packet_path,
            private_path=private_path,
        )
        prompt_report = audit_base._prompt_length_report(
            args.output,
            pipeline,
        )
    _write_json(prompt_path, prompt_report)
    checks = {
        "all_26_pairs_labeled": len(labels) == 26,
        "manual_false_new_exactly_zero": not false_new,
        "manual_unsound_claims_exactly_zero": not unsound_claims,
        "all_hidden_equivalent_controls_recognized": (
            not control_errors
            and manifest["blinded_equivalent_control_count"] == 5
        ),
        "singleton_repair_sound_and_non_distinct": singleton_ok,
        "no_zero_support_rows": not zero_support,
        "train_multi_at_least_10": counts["train_multi"] >= 10,
        "eval_multi_at_least_10": counts["eval_multi"] >= 10,
        "all_prompts_at_most_2048_tokens": (
            prompt_report.get("pass") is True
        ),
    }
    passed = all(checks.values())
    decision = {
        "schema": "e49r_combined_toy_advancement_decision_v1",
        "pass": passed,
        "advance_to_training": passed,
        "checks": checks,
        "manual_pair_count": len(labels),
        "accepted_new_claim_count": len(accepted_claims),
        "manual_false_new_count": len(false_new),
        "manual_false_new_pair_ids": sorted(false_new),
        "manual_unsound_claim_count": len(unsound_claims),
        "manual_unsound_claim_pair_ids": sorted(unsound_claims),
        "control_error_count": len(control_errors),
        "control_error_pair_ids": sorted(control_errors),
        "zero_support_rows": sorted(zero_support),
        "support_counts": counts,
        "audited_records_sha256": _sha256(records_path),
        "manual_labels_sha256": _sha256(labels_path),
        "prompt_length_audit_sha256": _sha256(prompt_path),
        "materialized_data": (
            str(args.output) if materialization is not None else None
        ),
        "materialization_manifest": materialization,
    }
    _write_json(decision_path, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "finalize"))
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--base-evidence", type=pathlib.Path, required=True)
    parser.add_argument(
        "--strict-evidence",
        type=pathlib.Path,
        action="append",
        required=True,
    )
    parser.add_argument("--recovery-evidence", type=pathlib.Path, required=True)
    parser.add_argument("--e49h-evidence", type=pathlib.Path, required=True)
    parser.add_argument("--evidence", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
