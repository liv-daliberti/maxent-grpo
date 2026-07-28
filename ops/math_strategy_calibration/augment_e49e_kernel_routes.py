#!/usr/bin/env python3
"""Answer-bound, finite-kernel diversity augmentation for E49E.

The augmentation proposes one bounded bank per problem, audits every route
with the frozen E49E soundness checks, applies the stricter pair veto, and
keeps the better of the original trace-certified bank and the augmented bank.
It never turns a failed row into more than the independently certified support
actually observed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from datasets import load_from_disk


HERE = pathlib.Path(__file__).resolve().parent
REPAIR_PATH = HERE / "repair_e49e_singleton_gaps.py"
SPEC = importlib.util.spec_from_file_location(
    "e49e_repair_for_kernel_augmentation",
    REPAIR_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load E49E repair implementation")
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)
pipeline = repair.pipeline

AUGMENTATION_VERSION = "e49e_finite_kernel_augmentation_v1"
PROPOSAL_SEED = 492201
PROPOSAL_EXTRA_RULES = ""
KERNELS = (
    "direct_algebra",
    "substitution_change_variable",
    "factorization_roots",
    "inequality_bound",
    "modular_invariant",
    "recurrence_induction",
    "generating_function",
    "combinatorial_bijection",
    "inclusion_exclusion",
    "geometric_similarity",
    "coordinate_geometry",
    "complex_plane",
    "trigonometric_identity",
    "calculus_extremum",
    "symmetry_invariant",
    "exhaustive_casework",
)
OP_CODES = (
    "READ_GIVENS",
    "NORMALIZE",
    "SUBSTITUTE",
    "EXPAND",
    "FACTOR",
    "SOLVE_EQUATION",
    "ENUMERATE_CASES",
    "COUNT_OBJECTS",
    "APPLY_MODULAR_RULE",
    "APPLY_THEOREM",
    "CONSTRUCT_OBJECT",
    "TRANSFORM_REPRESENTATION",
    "ESTABLISH_BOUND",
    "CHECK_CASES",
    "CONCLUDE",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _proposal_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema", "routes"],
        "properties": {
            "schema": {
                "type": "string",
                "enum": ["math_finite_kernel_bank_v1"],
            },
            "routes": {
                "type": "array",
                "minItems": 2,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["kernel_id", "actions", "plan"],
                    "properties": {
                        "kernel_id": {
                            "type": "string",
                            "enum": list(KERNELS),
                        },
                        "actions": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 6,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["op_code", "operation"],
                                "properties": {
                                    "op_code": {
                                        "type": "string",
                                        "enum": list(OP_CODES),
                                    },
                                    "operation": {
                                        "type": "string",
                                        "minLength": 1,
                                        "maxLength": 260,
                                    },
                                },
                            },
                        },
                        "plan": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 560,
                        },
                    },
                },
            },
        },
    }


def _payload_to_menu(
    payload: dict[str, Any],
    reference_answer: str,
    *,
    problem: str,
):
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "routes"}
        or payload.get("schema") != "math_finite_kernel_bank_v1"
        or not isinstance(payload.get("routes"), list)
        or not 2 <= len(payload["routes"]) <= 3
        or any(not isinstance(route, dict) for route in payload["routes"])
    ):
        raise ValueError("invalid finite-kernel bank")
    kernels = [route.get("kernel_id") for route in payload["routes"]]
    if (
        any(kernel not in KERNELS for kernel in kernels)
        or len(set(kernels)) != len(kernels)
    ):
        raise ValueError("route kernels must be distinct and finite")
    if sum(len(route.get("actions") or []) for route in payload["routes"]) > 12:
        raise ValueError("finite-kernel bank exceeds twelve actions")

    actions = []
    strategies = []
    op_combos = []
    next_action = 1
    for strategy_index, route in enumerate(payload["routes"], start=1):
        route_actions = route.get("actions")
        if (
            set(route) != {"kernel_id", "actions", "plan"}
            or not isinstance(route_actions, list)
            or not 2 <= len(route_actions) <= 6
            or not isinstance(route.get("plan"), str)
            or not route["plan"].strip()
            or len(route["plan"]) > 560
        ):
            raise ValueError("invalid finite-kernel route")
        ids = []
        op_combo = []
        for action in route_actions:
            if (
                not isinstance(action, dict)
                or set(action) != {"op_code", "operation"}
                or action.get("op_code") not in OP_CODES
                or not isinstance(action.get("operation"), str)
                or not action["operation"].strip()
                or len(action["operation"]) > 260
                or pipeline.ACTION_REFERENCE_RE.search(action["operation"])
            ):
                raise ValueError("invalid finite operation")
            op_combo.append(action["op_code"])
            action_id = f"A{next_action}"
            ids.append(action_id)
            actions.append(
                {
                    "action_id": action_id,
                    "operation": (
                        f"[OP:{action['op_code']}] "
                        f"{action['operation'].strip()}"
                    ),
                }
            )
            next_action += 1
        op_combos.append(tuple(op_combo))
        if pipeline.ACTION_REFERENCE_RE.search(route["plan"]):
            raise ValueError("kernel plan must not predeclare action IDs")
        strategies.append(
            {
                "strategy_id": f"S{strategy_index}",
                "action_ids": ids,
                "plan": (
                    f"[KERNEL:{route['kernel_id']}] "
                    f"{route['plan'].strip()}"
                ),
            }
        )
    if len(set(op_combos)) != len(op_combos):
        raise ValueError("route operation-code combos must be distinct")
    menu = pipeline._menu_from_payload(
        {
            "schema": pipeline.MENU_SCHEMA,
            "actions": actions,
            "strategies": strategies,
        }
    )
    if not repair._proposal_is_nonleaking(
        menu,
        problem=problem,
        reference_answer=reference_answer,
    ):
        raise ValueError(
            "finite-kernel bank leaks an answer or derived numeric value"
        )
    return menu


def _proposal_request(
    *,
    endpoint: str,
    model: str,
    problem: str,
    reference_answer: str,
    gold_solution: str,
    timeout: int,
) -> dict[str, Any]:
    gold = (
        f"\nAUDITOR-ONLY GOLD DERIVATION:\n{gold_solution}\n"
        if gold_solution.strip()
        else ""
    )
    prompt = f"""Propose a finite bank of two or three genuinely different
solution routes for this problem.

Choose a different KERNEL_ID for each route from the provided finite enum, but
do not use different labels to disguise the same mathematics. Decimals versus
fractions, unit conversion before versus after, reordered equations, direct
expansion versus the identical formula, redundant verification, or extra
arithmetic are the same route and must not receive different kernels.
Different routes must establish genuinely route-exclusive intermediate facts
using different decisive theorems, invariants, constructions, counting
spaces, or proof structures. If only two legitimate kernels exist, return two.

Each route must contain two to six ordered actions from the finite OP_CODE
enum, and the complete bank may contain at most twelve actions. Every
operation must specify what theorem or transformation is executed and its
conditions. Do not mention action IDs; they are assigned later.

The reference material is auditor-only. Use it to avoid proposing an invalid
route, but do not reveal the reference answer, boxed answer, evaluated final
value, or worked numerical result in any operation or plan. Numeric literals
may appear only when they are already in the problem or are the structural
constants 0, 1, and 2.
{PROPOSAL_EXTRA_RULES}

PROBLEM:
{problem}

AUDITOR-ONLY REFERENCE ANSWER:
{reference_answer}
{gold}
"""
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": pipeline.SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 4096,
        "seed": PROPOSAL_SEED,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "math_finite_kernel_bank",
                "strict": True,
                "schema": _proposal_schema(),
            },
        },
    }
    response, content = pipeline.base._post(
        endpoint,
        request,
        timeout=timeout,
    )
    common = {
        "augmentation_version": AUGMENTATION_VERSION,
        "kind": "kernel_bank_proposal",
        "seed": PROPOSAL_SEED,
        "response_id": response.get("id"),
        "finish_reason": (
            ((response.get("choices") or [{}])[0]).get("finish_reason")
        ),
        "content_sha256": _sha256_bytes(content.encode("utf-8")),
    }
    try:
        payload = json.loads(content)
        menu = _payload_to_menu(
            payload,
            reference_answer,
            problem=problem,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return {
            **common,
            "proposal": None,
            "menu": None,
            "completed_invalid": True,
            "error": str(exc),
            "pass": False,
        }
    return {
        **common,
        "proposal": payload,
        "menu": json.loads(menu.canonical_json),
        "menu_sha256": menu.sha256,
        "completed_invalid": False,
        "pass": bool(
            common["finish_reason"] == "stop"
            and str(common["response_id"] or "")
        ),
    }


def _proposal_cache(
    path: pathlib.Path,
    **kwargs,
) -> dict[str, Any]:
    if path.is_file():
        record = json.loads(path.read_text(encoding="utf-8"))
        if not _proposal_record_passes(
            record,
            problem=kwargs["problem"],
            reference_answer=kwargs["reference_answer"],
        ):
            raise RuntimeError("kernel proposal cache contract changed")
        return record
    record = _proposal_request(**kwargs)
    pipeline._write_json(path, record)
    return record


def _proposal_record_passes(
    record: dict[str, Any],
    *,
    problem: str,
    reference_answer: str,
) -> bool:
    if (
        not isinstance(record, dict)
        or record.get("augmentation_version") != AUGMENTATION_VERSION
        or record.get("kind") != "kernel_bank_proposal"
        or record.get("seed") != PROPOSAL_SEED
        or not str(record.get("response_id") or "")
        or not str(record.get("finish_reason") or "")
        or not str(record.get("content_sha256") or "")
    ):
        return False
    if record.get("completed_invalid") is True:
        return bool(
            record.get("pass") is not True
            and record.get("proposal") is None
            and record.get("menu") is None
        )
    if (
        not isinstance(record.get("proposal"), dict)
        or not isinstance(record.get("menu"), dict)
    ):
        return False
    try:
        recomputed = _payload_to_menu(
            record["proposal"],
            reference_answer,
            problem=problem,
        )
        recorded = pipeline._menu_from_payload(record["menu"])
    except (TypeError, ValueError):
        return False
    return bool(
        recomputed.sha256 == record.get("menu_sha256")
        and recomputed.canonical_json == recorded.canonical_json
        and (record.get("pass") is True)
        == (record.get("finish_reason") == "stop")
    )


def _audit_augmented(
    *,
    endpoint: str,
    model: str,
    row_id: str,
    problem: str,
    reference_answer: str,
    menu,
    cache_root: pathlib.Path,
    timeout: int,
):
    sound: dict[str, list[dict[str, Any]]] = {}
    for strategy in menu.strategies:
        sound[strategy.strategy_id] = []
        for seed, role in zip(
            pipeline.SOUNDNESS_SEEDS,
            pipeline.SOUNDNESS_ROLES,
            strict=True,
        ):
            path = pipeline._cache_path(
                cache_root,
                row_id=row_id,
                kind="kernel-soundness",
                strategy_id=strategy.strategy_id,
                seed=seed,
                menu_sha256=menu.sha256,
            )
            sound[strategy.strategy_id].append(
                pipeline._cached_request(
                    path,
                    pipeline._sound_request,
                    endpoint=endpoint,
                    model=model,
                    problem=problem,
                    reference_answer=reference_answer,
                    menu=menu,
                    strategy_id=strategy.strategy_id,
                    role=role,
                    seed=seed,
                    timeout=timeout,
                )
            )
    eligible = [
        strategy.strategy_id
        for strategy in menu.strategies
        if all(
            pipeline._sound_record_passes(
                menu,
                strategy.strategy_id,
                audit,
                reference_answer=reference_answer,
            )
            for audit in sound[strategy.strategy_id]
        )
    ]
    pair_audits = []
    if len(eligible) >= 2:
        for seed, role in zip(
            pipeline.PAIR_SEEDS,
            pipeline.PAIR_ROLES,
            strict=True,
        ):
            path = pipeline._cache_path(
                cache_root,
                row_id=row_id,
                kind="kernel-pair",
                strategy_id="-".join(eligible),
                seed=seed,
                menu_sha256=menu.sha256,
            )
            pair_audits.append(
                pipeline._cached_request(
                    path,
                    pipeline._pair_request,
                    endpoint=endpoint,
                    model=model,
                    problem=problem,
                    reference_answer=reference_answer,
                    menu=menu,
                    sound_audits=sound,
                    eligible_ids=eligible,
                    role=role,
                    seed=seed,
                    timeout=timeout,
                )
            )
    return pipeline._maximal_trace_certified_subset(
        menu,
        sound,
        pair_audits,
        reference_answer=reference_answer,
    )


def _build_one(
    *,
    endpoint: str,
    model: str,
    row_id: str,
    split: str,
    problem: str,
    reference_answer: str,
    gold_solution: str,
    raw_record: dict[str, Any],
    input_record: dict[str, Any],
    cache_root: pathlib.Path,
    timeout: int,
) -> dict[str, Any]:
    common = {
        "schema": "e49e_kernel_augmentation_record_v1",
        "augmentation_version": AUGMENTATION_VERSION,
        "row_id": row_id,
        "split": split,
        "problem_sha256": _sha256_bytes(problem.encode("utf-8")),
        "reference_answer_sha256": _sha256_bytes(
            reference_answer.encode("utf-8")
        ),
        "raw_record_sha256": _canonical_sha256(raw_record),
        "input_record_sha256": _canonical_sha256(input_record),
    }
    original = repair._raw_selected(
        raw_record,
        input_record=input_record,
        reference_answer=reference_answer,
    )
    row_hash = _sha256_bytes(row_id.encode("utf-8"))[:20]
    proposal = _proposal_cache(
        cache_root / row_hash / f"kernel-proposal-{PROPOSAL_SEED}.json",
        endpoint=endpoint,
        model=model,
        problem=problem,
        reference_answer=reference_answer,
        gold_solution=gold_solution,
        timeout=timeout,
    )
    augmented = None
    if proposal.get("pass") is True:
        menu = pipeline._menu_from_payload(proposal["menu"])
        augmented = _audit_augmented(
            endpoint=endpoint,
            model=model,
            row_id=row_id,
            problem=problem,
            reference_answer=reference_answer,
            menu=menu,
            cache_root=cache_root,
            timeout=timeout,
        )
    choices = [
        (name, selected)
        for name, selected in (
            ("original_trace_bank", original),
            ("finite_kernel_augmentation", augmented),
        )
        if selected is not None
    ]
    if not choices:
        return {
            **common,
            "proposal": proposal,
            "error": "no trace-certified original or augmented route",
            "pass": False,
        }
    origin, selected = max(
        choices,
        key=lambda item: (
            len(item[1][0].strategies),
            item[0] == "original_trace_bank",
        ),
    )
    menu, certification = selected
    return {
        **common,
        "proposal": proposal,
        "origin": origin,
        "menu": json.loads(menu.canonical_json),
        "menu_sha256": menu.sha256,
        "certification": certification,
        "pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--input-e49d", type=pathlib.Path, required=True)
    parser.add_argument("--input-e49e", type=pathlib.Path, required=True)
    parser.add_argument("--evidence", type=pathlib.Path, required=True)
    parser.add_argument("--endpoint", type=pathlib.Path, required=True)
    parser.add_argument(
        "--known-invalid-controls",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument(
        "--known-equivalent-controls",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    raw_path = args.input_e49e / "trace_bank_records.jsonl"
    raw = pipeline._load_latest(raw_path)
    input_path = args.input_e49d / "menu_records.jsonl"
    input_records = pipeline._load_latest(input_path)
    train_dict = load_from_disk(str(args.source / "train"))
    eval_dict = load_from_disk(str(args.source / "eval"))
    splits = {
        "train": train_dict[next(iter(train_dict))],
        "eval": eval_dict[next(iter(eval_dict))],
    }
    work = []
    for split, dataset in splits.items():
        for index, row in enumerate(dataset):
            row_id = pipeline.base._row_id(split, index, row)
            work.append((split, row_id, row))
    expected = {row_id for _, row_id, _ in work}
    if set(raw) != expected or set(input_records) != expected:
        raise RuntimeError("kernel augmentation inputs are incomplete")

    references = {
        row_id: str(row["answer"]) for _, row_id, row in work
    }
    invalid = pipeline._load_known_invalid_controls(
        args.known_invalid_controls,
        input_records,
    )
    invalid_results = pipeline._known_invalid_control_results(
        raw,
        invalid,
        input_records,
        references,
    )
    equivalent = pipeline._load_known_equivalent_controls(
        args.known_equivalent_controls,
        input_records,
    )
    equivalent_results = pipeline._known_equivalent_control_results(
        raw,
        equivalent,
        input_records,
        references,
    )
    if not all(row["rejected_by_soundness"] for row in invalid_results):
        raise RuntimeError("known-invalid control gate failed")
    if not all(row["rejected_as_new"] for row in equivalent_results):
        raise RuntimeError("known-equivalent control gate failed")

    original_counts = []
    for _, row_id, row in work:
        selected = repair._raw_selected(
            raw[row_id],
            input_record=input_records[row_id],
            reference_answer=str(row["answer"]),
        )
        original_counts.append(
            0 if selected is None else len(selected[0].strategies)
        )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "schema": "e49e_kernel_augmentation_preflight_v1",
                    "augmentation_version": AUGMENTATION_VERSION,
                    "answer_normalization_version": (
                        pipeline.ANSWER_NORMALIZATION_VERSION
                    ),
                    "row_count": len(work),
                    "original_trace_support_histogram": {
                        str(count): original_counts.count(count)
                        for count in range(4)
                    },
                    "known_invalid_control_count": len(invalid_results),
                    "known_equivalent_control_count": len(
                        equivalent_results
                    ),
                    "all_controls_pass": True,
                    "raw_trace_records_sha256": _sha256_bytes(
                        raw_path.read_bytes()
                    ),
                    "e49d_input_sha256": _sha256_bytes(
                        input_path.read_bytes()
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    endpoint, model = pipeline.base._endpoint(args.endpoint)
    record_path = args.evidence / "augmentation_records.jsonl"
    existing = pipeline._load_latest(record_path)
    pending = [
        (split, row_id, row)
        for split, row_id, row in work
        if row_id not in existing
    ]
    print(
        f"kernel augmentation: total={len(work)} "
        f"complete={len(existing)} pending={len(pending)}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _build_one,
                endpoint=endpoint,
                model=model,
                row_id=row_id,
                split=split,
                problem=str(row["problem"]),
                reference_answer=str(row["answer"]),
                gold_solution=str(row.get("solution") or ""),
                raw_record=raw[row_id],
                input_record=input_records[row_id],
                cache_root=args.evidence / "request_cache",
                timeout=args.timeout,
            ): row_id
            for split, row_id, row in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            pipeline._append_jsonl(record_path, record)
            existing[record["row_id"]] = record
            print(
                f"[{index}/{len(pending)}] {record['row_id']} "
                f"pass={record['pass']} "
                f"support={len((record.get('menu') or {}).get('strategies') or [])}",
                flush=True,
            )
    if set(existing) != expected:
        raise RuntimeError("kernel augmentation did not cover every row")
    counts = [
        len((existing[row_id].get("menu") or {}).get("strategies") or [])
        for _, row_id, _ in work
    ]
    summary = {
        "schema": "e49e_kernel_augmentation_summary_v1",
        "augmentation_version": AUGMENTATION_VERSION,
        "complete": True,
        "row_count": len(work),
        "covered_row_count": sum(count >= 1 for count in counts),
        "multi_strategy_menu_count": sum(count >= 2 for count in counts),
        "train_multi_strategy_menu_count": sum(
            len(
                (existing[row_id].get("menu") or {}).get("strategies")
                or []
            )
            >= 2
            for split, row_id, _ in work
            if split == "train"
        ),
        "eval_multi_strategy_menu_count": sum(
            len(
                (existing[row_id].get("menu") or {}).get("strategies")
                or []
            )
            >= 2
            for split, row_id, _ in work
            if split == "eval"
        ),
        "support_histogram": {
            str(count): counts.count(count) for count in range(4)
        },
        "known_invalid_control_results": invalid_results,
        "known_equivalent_control_results": equivalent_results,
        "all_controls_pass": True,
        "records_sha256": _sha256_bytes(record_path.read_bytes()),
    }
    pipeline._write_json(args.evidence / "generation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
