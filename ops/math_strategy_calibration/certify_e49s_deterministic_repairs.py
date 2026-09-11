#!/usr/bin/env python3
"""Execute E49S's four deterministic toy MathIR repairs and materialize data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
import math
import os
import pathlib
import tempfile
from fractions import Fraction
from typing import Any, Callable

from datasets import Dataset, DatasetDict, load_from_disk


ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e49s_deterministic_mathir_repairs_20260724.md"
)
SCRIPT = pathlib.Path(__file__).resolve()
E49R_SCRIPT = HERE / "audit_e49r_combined_toy_bank.py"


def _load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e49r = _load_module("e49s_e49r", E49R_SCRIPT)
pipeline = e49r.pipeline
audit_base = e49r.audit_base


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


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


def _write_jsonl(path: pathlib.Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for value in values:
                handle.write(_canonical_bytes(value).decode("utf-8") + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _fraction(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _trace(combo: tuple[str, ...], states: list[Any], answer: str) -> dict[str, Any]:
    if len(combo) != len(states):
        raise RuntimeError("E49S interpreter returned wrong state count")
    return {
        "action_combo": ">".join(combo),
        "action_states": [
            {"action_id": action_id, "state": state}
            for action_id, state in zip(combo, states, strict=True)
        ],
        "answer": answer,
    }


def _divisibility_lcm() -> dict[str, Any]:
    lcm = math.lcm(3, 4, 5)
    multiples = list(range(lcm, 501, lcm))
    return _trace(
        ("A1", "A2"),
        [
            {"moduli": [3, 4, 5], "lcm": lcm},
            {"bound": 500, "multiples": multiples, "count": len(multiples)},
        ],
        str(len(multiples)),
    )


def _divisibility_scan() -> dict[str, Any]:
    candidates = list(range(1, 501))
    passing = [
        value
        for value in candidates
        if all(value % divisor == 0 for divisor in (3, 4, 5))
    ]
    return _trace(
        ("A3", "A4"),
        [
            {
                "candidate_count": len(candidates),
                "first_candidate": candidates[0],
                "last_candidate": candidates[-1],
                "candidate_sha256": _canonical_sha256(candidates),
            },
            {"passing_values": passing, "count": len(passing)},
        ],
        str(len(passing)),
    )


def _binomial_formula() -> dict[str, Any]:
    combinations = math.comb(7, 4)
    probability = Fraction(combinations) * Fraction(1, 5) ** 4 * Fraction(4, 5) ** 3
    return _trace(
        ("A1", "A2", "A3"),
        [
            {"trials": 7, "successes": 4, "p": "1/5"},
            {"combination": combinations},
            {"probability": _fraction(probability)},
        ],
        _fraction(probability),
    )


def _binomial_dp() -> dict[str, Any]:
    vector = [Fraction(1)]
    vectors = [[_fraction(value) for value in vector]]
    for _ in range(7):
        updated = [Fraction(0)] * (len(vector) + 1)
        for successes in range(len(updated)):
            if successes < len(vector):
                updated[successes] += Fraction(4, 5) * vector[successes]
            if successes:
                updated[successes] += Fraction(1, 5) * vector[successes - 1]
        vector = updated
        vectors.append([_fraction(value) for value in vector])
    probability = vector[4]
    return _trace(
        ("A4", "A5", "A6"),
        [
            {"initial_probability_vector": vectors[0]},
            {"vectors_after_each_island": vectors[1:]},
            {"four_success_component": _fraction(probability)},
        ],
        _fraction(probability),
    )


def _circular_block() -> dict[str, Any]:
    circular = math.factorial(5)
    internal = math.factorial(3)
    answer = circular * internal
    return _trace(
        ("A1", "A2", "A3", "A4"),
        [
            {
                "contracted_group": ["Pierre", "Rosa", "Thomas"],
                "object_count": 6,
            },
            {"circular_arrangements": circular},
            {"internal_arrangements": internal},
            {"product": answer},
        ],
        str(answer),
    )


def _three_consecutive(arrangement: tuple[str, ...]) -> bool:
    positions = {
        person: arrangement.index(person)
        for person in ("Pierre", "Rosa", "Thomas")
    }
    occupied = set(positions.values())
    return any(
        occupied == {start, (start + 1) % 8, (start + 2) % 8}
        for start in range(8)
    )


def _circular_scan() -> dict[str, Any]:
    remaining = ("Rosa", "Thomas", "O1", "O2", "O3", "O4", "O5")
    all_permutations = list(itertools.permutations(remaining))
    anchored = [("Pierre",) + permutation for permutation in all_permutations]
    passing = [
        list(arrangement)
        for arrangement in anchored
        if _three_consecutive(arrangement)
    ]
    return _trace(
        ("A5", "A6"),
        [
            {
                "fixed_person": "Pierre",
                "generated_count": len(anchored),
                "generated_sha256": _canonical_sha256(anchored),
            },
            {
                "cyclic_adjacency_passing_count": len(passing),
                "passing_arrangements": passing,
            },
        ],
        str(len(passing)),
    )


def _pentagon_exterior() -> dict[str, Any]:
    exterior = Fraction(360, 5)
    tip = Fraction(180) - 2 * exterior
    return _trace(
        ("A1", "A2"),
        [
            {"exterior_turn_degrees": _fraction(exterior)},
            {
                "extension_triangle_base_angles_degrees": [
                    _fraction(exterior),
                    _fraction(exterior),
                ],
                "tip_degrees": _fraction(tip),
            },
        ],
        str(tip.numerator),
    )


def _pentagon_central() -> dict[str, Any]:
    central = Fraction(360, 5)
    central_base = (Fraction(180) - central) / 2
    interior = 2 * central_base
    extension_base = Fraction(180) - interior
    tip = Fraction(180) - 2 * extension_base
    return _trace(
        ("A3", "A4", "A5"),
        [
            {
                "central_angle_degrees": _fraction(central),
                "central_isosceles_base_angle_degrees": _fraction(central_base),
            },
            {
                "pentagon_interior_degrees": _fraction(interior),
                "straight_line_extension_degrees": _fraction(extension_base),
            },
            {"star_tip_degrees": _fraction(tip)},
        ],
        str(tip.numerator),
    )


REPAIRS: dict[str, dict[str, Any]] = {
    "train:0031:test/prealgebra/937.json": {
        "split": "train",
        "pair_id": "PAIR_38f980165fc292bfbac0",
        "answer": "8",
        "programs": (_divisibility_lcm, _divisibility_scan),
        "menu": {
            "schema": "math_strategy_action_menu_v1",
            "actions": [
                {"action_id": "A1", "operation": "Construct lcm(3,4,5) from exact prime exponents."},
                {"action_id": "A2", "operation": "Enumerate its positive multiples through 500 and count them."},
                {"action_id": "A3", "operation": "Generate every integer in the closed range 1 through 500."},
                {"action_id": "A4", "operation": "Test all three exact remainder predicates for every generated integer and count the passing list."},
            ],
            "strategies": [
                {"strategy_id": "S1", "action_ids": ["A1", "A2"], "plan": "Reduce simultaneous divisibility to exact multiples of the least common multiple."},
                {"strategy_id": "S2", "action_ids": ["A3", "A4"], "plan": "Exhaust the requested finite range and retain exactly the simultaneous divisibility passes."},
            ],
        },
    },
    "eval:0027:cf127e68fa63742a3582": {
        "split": "eval",
        "pair_id": "PAIR_3fa95a0cdaa7f8f785dc",
        "answer": "448/15625",
        "programs": (_binomial_formula, _binomial_dp),
        "menu": {
            "schema": "math_strategy_action_menu_v1",
            "actions": [
                {"action_id": "A1", "operation": "Bind the exact Bernoulli parameters n=7, k=4, and p=1/5."},
                {"action_id": "A2", "operation": "Compute the exact binomial coefficient C(7,4)."},
                {"action_id": "A3", "operation": "Evaluate and reduce C(7,4)(1/5)^4(4/5)^3."},
                {"action_id": "A4", "operation": "Initialize the exact probability vector [1] for zero searched islands."},
                {"action_id": "A5", "operation": "Execute seven Bernoulli DP updates with exact rational arithmetic and display every resulting vector."},
                {"action_id": "A6", "operation": "Extract and reduce the component for exactly four treasures."},
            ],
            "strategies": [
                {"strategy_id": "S1", "action_ids": ["A1", "A2", "A3"], "plan": "Evaluate the closed-form binomial probability."},
                {"strategy_id": "S2", "action_ids": ["A4", "A5", "A6"], "plan": "Propagate the complete exact treasure-count distribution by dynamic programming."},
            ],
        },
    },
    "eval:0041:9a01eabaf416b53ac8ba": {
        "split": "eval",
        "pair_id": "PAIR_06940aba0cb3db5f4914",
        "answer": "720",
        "programs": (_circular_block, _circular_scan),
        "menu": {
            "schema": "math_strategy_action_menu_v1",
            "actions": [
                {"action_id": "A1", "operation": "Contract Pierre, Rosa, and Thomas into one object."},
                {"action_id": "A2", "operation": "Count the circular arrangements of the six contracted objects."},
                {"action_id": "A3", "operation": "Count all internal permutations of the contracted three-person block."},
                {"action_id": "A4", "operation": "Multiply the independent circular and internal counts."},
                {"action_id": "A5", "operation": "Fix Pierre at a reference seat and generate all 7! permutations of the remaining people."},
                {"action_id": "A6", "operation": "Test every generated circle for one cyclic run containing Pierre, Rosa, and Thomas; count the passing set."},
            ],
            "strategies": [
                {"strategy_id": "S1", "action_ids": ["A1", "A2", "A3", "A4"], "plan": "Use circular block contraction and internal permutations."},
                {"strategy_id": "S2", "action_ids": ["A5", "A6"], "plan": "Exhaust anchored circular arrangements and directly filter cyclic adjacency."},
            ],
        },
    },
    "eval:0047:5310db58fbda4a6559d0": {
        "split": "eval",
        "pair_id": "PAIR_b6fa4e172ff0db63cc44",
        "answer": "36",
        "programs": (_pentagon_exterior, _pentagon_central),
        "menu": {
            "schema": "math_strategy_action_menu_v1",
            "actions": [
                {"action_id": "A1", "operation": "Compute the regular pentagon exterior turn as 360/5 degrees."},
                {"action_id": "A2", "operation": "Use the two exterior turns as the extension-triangle base angles and subtract them from 180 degrees."},
                {"action_id": "A3", "operation": "Compute the 360/5 central angle and the base angle of its isosceles central triangle."},
                {"action_id": "A4", "operation": "Assemble two central base angles into a pentagon interior angle, then take its straight-line supplement."},
                {"action_id": "A5", "operation": "Subtract the two exact extension-triangle base angles from 180 degrees."},
            ],
            "strategies": [
                {"strategy_id": "S1", "action_ids": ["A1", "A2"], "plan": "Derive the star tip directly from regular-polygon exterior turns."},
                {"strategy_id": "S2", "action_ids": ["A3", "A4", "A5"], "plan": "Derive the interior and extension angles through central isosceles triangles."},
            ],
        },
    },
}


def _source_rows(source: pathlib.Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    train = load_from_disk(str(source / "train"))
    evaluation = load_from_disk(str(source / "eval"))
    train_name = next(iter(train))
    eval_name = next(iter(evaluation))
    datasets = {
        "train": train[train_name],
        "eval": evaluation[eval_name],
    }
    rows: dict[str, dict[str, Any]] = {}
    for split, dataset in datasets.items():
        for index, row in enumerate(dataset):
            row_id = pipeline.base._row_id(split, index, row)
            rows[row_id] = dict(row)
    return rows, {
        "train": (train_name, datasets["train"]),
        "eval": (eval_name, datasets["eval"]),
    }


def _validate_e49r_blinding(evidence: pathlib.Path) -> None:
    packet = {
        row["pair_id"]: row
        for row in _read_jsonl(evidence / "manual_audit_packet.jsonl")
    }
    labels_payload = json.loads(
        (evidence / "manual_audit_labels.json").read_text(encoding="utf-8")
    )
    labels = {row["pair_id"]: row for row in labels_payload["labels"]}
    private = {
        row["pair_id"]: row
        for row in _read_jsonl(evidence / "private/manual_audit_key.jsonl")
    }
    if labels_payload.get("blinded_before_private_key") is not True:
        raise RuntimeError("E49S requires blinded E49R labels")
    for row_id, repair in REPAIRS.items():
        pair_id = repair["pair_id"]
        if (
            pair_id not in packet
            or private[pair_id].get("row_id") != row_id
            or labels[pair_id]["genuinely_distinct_decisive_strategy"] is not True
        ):
            raise RuntimeError(f"E49S pairwise-distinction binding failed: {row_id}")


def _execute_repairs() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    certificates: list[dict[str, Any]] = []
    menus: dict[str, dict[str, Any]] = {}
    for row_id, repair in REPAIRS.items():
        menu = pipeline._menu_from_payload(repair["menu"])
        if len(menu.strategies) != 2:
            raise RuntimeError("E49S repair menu is not binary")
        traces = []
        for option, program in zip(menu.strategies, repair["programs"], strict=True):
            first = program()
            second = program()
            if _canonical_bytes(first) != _canonical_bytes(second):
                raise RuntimeError(f"E49S nondeterministic execution: {row_id}")
            if (
                first["action_combo"] != option.action_combo
                or first["answer"] != repair["answer"]
                or [
                    state["action_id"] for state in first["action_states"]
                ]
                != list(option.action_ids)
            ):
                raise RuntimeError(f"E49S exact execution failed: {row_id}")
            traces.append(first)
        certificate = {
            "schema": "e49s_deterministic_mathir_certificate_v1",
            "row_id": row_id,
            "split": repair["split"],
            "e49r_pair_id": repair["pair_id"],
            "reference_answer_normalized": repair["answer"],
            "menu": json.loads(menu.canonical_json),
            "menu_sha256": menu.sha256,
            "executions": traces,
            "all_programs_deterministic": True,
            "all_action_combos_exact": True,
            "all_answers_exact": True,
        }
        certificate["certificate_payload_sha256"] = _canonical_sha256(certificate)
        certificates.append(certificate)
        menus[row_id] = json.loads(menu.canonical_json)
    return certificates, menus


def _materialize(
    source: pathlib.Path,
    output: pathlib.Path,
    records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    _, source_splits = _source_rows(source)
    output_splits = {}
    for split, (name, dataset) in source_splits.items():
        data = dataset.to_dict()
        augmented, hashes, origins = [], [], []
        for index, row in enumerate(dataset):
            row_id = pipeline.base._row_id(split, index, row)
            record = records[row_id]
            augmented.append(pipeline._embed(str(row["problem"]), record["menu"]))
            hashes.append(record["menu_sha256"])
            origins.append(record["origin"])
        data["original_problem"] = list(data["problem"])
        data["problem"] = augmented
        data["strategy_menu_sha256"] = hashes
        data["strategy_menu_origin"] = origins
        output_splits[split] = (name, Dataset.from_dict(data))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        DatasetDict(
            {output_splits["train"][0]: output_splits["train"][1]}
        ).save_to_disk(str(staging / "train"))
        DatasetDict(
            {output_splits["eval"][0]: output_splits["eval"][1]}
        ).save_to_disk(str(staging / "eval"))
        manifest = {
            "schema": "e49s_deterministic_mathir_materialization_v1",
            "menu_count": len(records),
            "train_tree_sha256": audit_base._tree_hash(staging / "train"),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--e49r-evidence", type=pathlib.Path, required=True)
    parser.add_argument("--evidence", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if (
        not PROTOCOL.is_file()
        or "FROZEN BEFORE REPAIR EXECUTION OR TRAINING"
        not in PROTOCOL.read_text(encoding="utf-8")
    ):
        raise RuntimeError("E49S protocol is not frozen")
    if args.evidence.exists() or args.output.exists():
        raise RuntimeError("E49S requires fresh evidence and output paths")
    _validate_e49r_blinding(args.e49r_evidence)
    source_rows, _ = _source_rows(args.source)
    if len(source_rows) != 100 or set(REPAIRS) - set(source_rows):
        raise RuntimeError("E49S source identity/count failed")
    records_path = args.e49r_evidence / "audited_records.jsonl"
    records = {
        row["row_id"]: row for row in _read_jsonl(records_path)
    }
    if len(records) != 100:
        raise RuntimeError("E49S requires the complete E49R fallback bank")
    certificates, menus = _execute_repairs()
    final_records = copy.deepcopy(records)
    for certificate in certificates:
        row_id = certificate["row_id"]
        menu = pipeline._menu_from_payload(menus[row_id])
        final_records[row_id] = {
            "schema": "e49s_deterministically_repaired_record_v1",
            "row_id": row_id,
            "split": certificate["split"],
            "origin": "deterministic_mathir_repair",
            "source_record_sha256": _canonical_sha256(source_rows[row_id]),
            "e49r_pair_id": certificate["e49r_pair_id"],
            "certificate_payload_sha256": certificate[
                "certificate_payload_sha256"
            ],
            "menu": json.loads(menu.canonical_json),
            "menu_sha256": menu.sha256,
            "pass": True,
        }
    support = {"train_multi": 0, "eval_multi": 0, "overall_multi": 0}
    zero_support = []
    for row_id, record in final_records.items():
        strategies = record.get("menu", {}).get("strategies", [])
        if not strategies:
            zero_support.append(row_id)
        if len(strategies) >= 2:
            support[f"{record['split']}_multi"] += 1
            support["overall_multi"] += 1
    args.evidence.mkdir(parents=True)
    certificate_path = args.evidence / "deterministic_certificates.jsonl"
    final_records_path = args.evidence / "audited_records.jsonl"
    _write_jsonl(certificate_path, certificates)
    _write_jsonl(
        final_records_path,
        [final_records[row_id] for row_id in sorted(final_records)],
    )
    materialization = _materialize(args.source, args.output, final_records)
    prompt_report = audit_base._prompt_length_report(args.output, pipeline)
    prompt_path = args.evidence / "prompt_length_audit.json"
    _write_json(prompt_path, prompt_report)
    checks = {
        "all_eight_programs_deterministic": (
            len(certificates) == 4
            and all(
                certificate["all_programs_deterministic"]
                and certificate["all_action_combos_exact"]
                and certificate["all_answers_exact"]
                and len(certificate["executions"]) == 2
                for certificate in certificates
            )
        ),
        "all_four_blinded_distinctions_preserved": True,
        "no_zero_support_rows": not zero_support,
        "train_multi_at_least_10": support["train_multi"] >= 10,
        "eval_multi_at_least_10": support["eval_multi"] >= 10,
        "all_prompts_at_most_2048_tokens": prompt_report.get("pass") is True,
    }
    identity = {
        "schema": "e49s_deterministic_mathir_identity_v1",
        "protocol_sha256": _sha256(PROTOCOL),
        "certifier_sha256": _sha256(SCRIPT),
        "e49r_records_sha256": _sha256(records_path),
        "e49r_packet_sha256": _sha256(
            args.e49r_evidence / "manual_audit_packet.jsonl"
        ),
        "e49r_labels_sha256": _sha256(
            args.e49r_evidence / "manual_audit_labels.json"
        ),
        "e49r_private_key_sha256": _sha256(
            args.e49r_evidence / "private/manual_audit_key.jsonl"
        ),
        "certificates_sha256": _sha256(certificate_path),
        "audited_records_sha256": _sha256(final_records_path),
        "prompt_length_audit_sha256": _sha256(prompt_path),
        "train_tree_sha256": materialization["train_tree_sha256"],
        "eval_tree_sha256": materialization["eval_tree_sha256"],
    }
    identity_path = args.evidence / "frozen_identity.json"
    _write_json(identity_path, identity)
    passed = all(checks.values())
    decision = {
        "schema": "e49s_deterministic_mathir_advancement_v1",
        "pass": passed,
        "advance_to_matched_toy_training": passed,
        "checks": checks,
        "support_counts": support,
        "zero_support_rows": zero_support,
        "repair_row_ids": sorted(REPAIRS),
        "identity_sha256": _sha256(identity_path),
        "materialized_data": str(args.output),
        "materialization_manifest": materialization,
    }
    _write_json(args.evidence / "advancement_decision.json", decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
