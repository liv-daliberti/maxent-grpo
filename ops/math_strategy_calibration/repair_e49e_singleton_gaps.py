#!/usr/bin/env python3
"""Repair only zero-route E49E rows with answer-bound singleton contracts.

The repair cannot add diversity. Existing trace-certified banks are
recomputed from their frozen audits with reference-closed pruning. A row with
no surviving route receives a bounded sequence of fixed, answer-bound
singleton proposals; the first proposal that passes both original E49E
soundness auditors is retained.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk


HERE = pathlib.Path(__file__).resolve().parent
PIPELINE_PATH = HERE / "materialize_e49e_trace_bank_data.py"
SPEC = importlib.util.spec_from_file_location(
    "e49e_trace_bank_for_repair",
    PIPELINE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load E49E trace-bank implementation")
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)

REPAIR_VERSION = "e49e_reference_bound_singleton_repair_v1"
REPAIR_ORIGIN = "reference_bound_singleton_repair"
PROPOSAL_SEEDS = (492141, 492142)
PROPOSAL_ROLES = (
    "minimal_direct_route",
    "independent_checked_route",
)
NUMBER_RE = re.compile(
    r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
)
STRUCTURAL_NUMBERS = {"0", "1", "2", "+0", "+1", "+2", "-0", "-1", "-2"}


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


def _tree_hash(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def _singleton_schema() -> dict[str, Any]:
    action_ids = [f"A{index}" for index in range(1, 8)]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema", "actions", "strategies"],
        "properties": {
            "schema": {
                "type": "string",
                "enum": [pipeline.MENU_SCHEMA],
            },
            "actions": {
                "type": "array",
                "minItems": 2,
                "maxItems": 7,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["action_id", "operation"],
                    "properties": {
                        "action_id": {
                            "type": "string",
                            "enum": action_ids,
                        },
                        "operation": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 420,
                        },
                    },
                },
            },
            "strategies": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["strategy_id", "action_ids", "plan"],
                    "properties": {
                        "strategy_id": {
                            "type": "string",
                            "enum": ["S1"],
                        },
                        "action_ids": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 7,
                            "items": {
                                "type": "string",
                                "enum": action_ids,
                            },
                        },
                        "plan": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 1000,
                        },
                    },
                },
            },
        },
    }


def _answer_variants(reference_answer: str) -> set[str]:
    raw = reference_answer.casefold().strip()
    variants = {raw}
    simplified = raw
    for token in ("\\boxed", "\\$", "$", "\\", "{", "}", "(", ")"):
        simplified = simplified.replace(token, "")
    simplified = re.sub(r"\s+", "", simplified)
    if simplified:
        variants.add(simplified)
    return {value for value in variants if value}


def _leaks_answer(menu: Any, reference_answer: str) -> bool:
    text = "\n".join(
        [action.operation for action in menu.actions]
        + [strategy.plan for strategy in menu.strategies]
    ).casefold()
    if any(
        phrase in text
        for phrase in (
            "\\boxed",
            "reference answer",
            "correct answer",
            "final answer is",
            "answer equals",
            "answer is",
            "final result is",
        )
    ):
        return True
    compact = re.sub(r"\s+", "", text)
    for value in _answer_variants(reference_answer):
        target = re.sub(r"\s+", "", value)
        if target and re.search(
            rf"(?<![a-z0-9]){re.escape(target)}(?![a-z0-9])",
            compact,
        ):
            return True
    return False


def _numeric_literals(value: str) -> set[str]:
    return {match.group(0) for match in NUMBER_RE.finditer(str(value))}


def _proposal_is_nonleaking(
    menu: Any,
    *,
    problem: str,
    reference_answer: str,
) -> bool:
    if _leaks_answer(menu, reference_answer):
        return False
    allowed = _numeric_literals(problem) | STRUCTURAL_NUMBERS
    proposal_text = "\n".join(
        [action.operation for action in menu.actions]
        + [strategy.plan for strategy in menu.strategies]
    )
    return _numeric_literals(proposal_text) <= allowed


def _proposal_request(
    *,
    endpoint: str,
    model: str,
    problem: str,
    reference_answer: str,
    gold_solution: str,
    role: str,
    seed: int,
    timeout: int,
) -> dict[str, Any]:
    gold = (
        f"\nAUDITOR-ONLY GOLD DERIVATION:\n{gold_solution}\n"
        if gold_solution.strip()
        else ""
    )
    role_instruction = {
        "minimal_direct_route": (
            "Use the shortest complete route with explicit theorem "
            "conditions and the decisive calculation."
        ),
        "independent_checked_route": (
            "Construct a complete route and explicitly include the domain, "
            "case, or sufficiency check most likely to be omitted."
        ),
    }[role]
    prompt = f"""Construct one finite action contract for solving this problem.

{role_instruction}

The reference material is auditor-only: use it to choose a correct route, but
do not reveal the reference answer, evaluated final value, boxed answer, or
worked numerical result in the action definitions or plan. Numeric literals
may appear only when they are already in the problem or are the structural
constants 0, 1, and 2. Give two to seven
ordered, specific operations. Each operation must name the actual algebraic,
geometric, combinatorial, or logical transformation the policy must execute;
vague instructions such as "solve" or "calculate the answer" are invalid.
The single strategy must be S1 and must use every action exactly once in the
declared order. It must be self-contained and must not mention another route.

PROBLEM:
{problem}

AUDITOR-ONLY REFERENCE ANSWER:
{reference_answer}
{gold}
"""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": pipeline.SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 2048,
        "seed": seed,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "math_singleton_action_contract",
                "strict": True,
                "schema": _singleton_schema(),
            },
        },
    }
    response, content = pipeline.base._post(
        endpoint,
        payload,
        timeout=timeout,
    )
    common = {
        "repair_version": REPAIR_VERSION,
        "kind": "singleton_proposal",
        "role": role,
        "seed": seed,
        "response_id": response.get("id"),
        "finish_reason": (
            ((response.get("choices") or [{}])[0]).get("finish_reason")
        ),
        "content_sha256": _sha256_bytes(content.encode("utf-8")),
    }
    try:
        payload = json.loads(content)
        menu = pipeline._menu_from_payload(payload)
        closed = pipeline._extract_closed_candidate(menu, "S1")
        if (
            closed is None
            or len(menu.strategies) != 1
            or len(closed.strategies) != 1
            or list(menu.strategies[0].action_ids)
            != [action.action_id for action in menu.actions]
            or not _proposal_is_nonleaking(
                menu,
                problem=problem,
                reference_answer=reference_answer,
            )
        ):
            raise ValueError("singleton proposal failed closure or leakage gate")
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return {
            **common,
            "menu": None,
            "completed_invalid": True,
            "error": str(exc),
            "pass": False,
        }
    return {
        **common,
        "menu": json.loads(closed.canonical_json),
        "menu_sha256": closed.sha256,
        "completed_invalid": False,
        "pass": bool(
            common["finish_reason"] == "stop"
            and str(common["response_id"] or "")
        ),
    }


def _cache(
    path: pathlib.Path,
    request,
    **kwargs,
) -> dict[str, Any]:
    if path.is_file():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("repair_version") != REPAIR_VERSION:
            raise RuntimeError("repair proposal cache contract changed")
        return record
    record = request(**kwargs)
    pipeline._write_json(path, record)
    return record


def _raw_selected(
    raw: dict[str, Any],
    *,
    input_record: dict[str, Any],
    reference_answer: str,
) -> tuple[Any, dict[str, Any]] | None:
    if (
        raw.get("problem_sha256") is None
        or raw.get("reference_answer_sha256") is None
        or raw.get("input_record_sha256")
        != _canonical_sha256(input_record)
    ):
        raise RuntimeError("raw E49E record identity changed")
    certification = raw.get("certification")
    if not isinstance(certification, dict):
        candidate_payload = raw.get("candidate_menu")
        sound_audits = raw.get("sound_audits")
        pair_audits = raw.get("pair_audits")
    else:
        candidate_payload = certification.get("candidate_menu")
        sound_audits = certification.get("sound_audits")
        pair_audits = certification.get("pair_audits")
    if not isinstance(candidate_payload, dict):
        return None
    candidate = pipeline._menu_from_payload(candidate_payload)
    expected = pipeline._candidate_bank(input_record)
    if expected is None or candidate.sha256 != expected[0].sha256:
        raise RuntimeError("raw E49E candidate bank changed")
    if not isinstance(sound_audits, dict) or not isinstance(pair_audits, list):
        return None
    return pipeline._maximal_trace_certified_subset(
        candidate,
        sound_audits,
        pair_audits,
        reference_answer=reference_answer,
    )


def _augmentation_selected(
    record: dict[str, Any],
    *,
    raw_record: dict[str, Any],
    input_record: dict[str, Any],
    problem: str,
    reference_answer: str,
) -> tuple[Any, dict[str, Any]] | None:
    if record.get("pass") is not True:
        return _raw_selected(
            raw_record,
            input_record=input_record,
            reference_answer=reference_answer,
        )
    if (
        record.get("augmentation_version")
        not in {
            "e49e_finite_kernel_augmentation_v1",
            "e49e_finite_kernel_augmentation_v2",
        }
        or record.get("problem_sha256")
        != _sha256_bytes(problem.encode("utf-8"))
        or record.get("reference_answer_sha256")
        != _sha256_bytes(reference_answer.encode("utf-8"))
        or record.get("raw_record_sha256")
        != _canonical_sha256(raw_record)
        or record.get("input_record_sha256")
        != _canonical_sha256(input_record)
    ):
        raise RuntimeError("kernel augmentation record identity changed")
    if record.get("origin") == "original_trace_bank":
        selected = _raw_selected(
            raw_record,
            input_record=input_record,
            reference_answer=reference_answer,
        )
        if (
            selected is None
            or selected[0].sha256 != record.get("menu_sha256")
            or selected[0].canonical_json
            != pipeline._menu_from_payload(record["menu"]).canonical_json
            or _canonical_sha256(selected[1])
            != _canonical_sha256(record.get("certification"))
        ):
            raise RuntimeError(
                "original trace-bank augmentation replay failed"
            )
        return selected
    certification = record.get("certification")
    if not isinstance(certification, dict):
        raise RuntimeError("kernel augmentation certification disappeared")
    candidate = pipeline._menu_from_payload(
        certification["candidate_menu"]
    )
    if not _proposal_is_nonleaking(
        candidate,
        problem=problem,
        reference_answer=reference_answer,
    ):
        raise RuntimeError("kernel augmentation leaked a numeric answer hint")
    selected = pipeline._maximal_trace_certified_subset(
        candidate,
        certification["sound_audits"],
        certification["pair_audits"],
        reference_answer=reference_answer,
    )
    if (
        selected is None
        or selected[0].sha256 != record.get("menu_sha256")
        or selected[0].canonical_json
        != pipeline._menu_from_payload(record["menu"]).canonical_json
    ):
        raise RuntimeError("kernel augmentation record failed recomputation")
    return selected


def _repair_one(
    *,
    endpoint: str,
    model: str,
    row_id: str,
    problem: str,
    reference_answer: str,
    gold_solution: str,
    cache_root: pathlib.Path,
    timeout: int,
) -> dict[str, Any]:
    row_hash = _sha256_bytes(row_id.encode("utf-8"))[:20]
    attempts = []
    for role, seed in zip(PROPOSAL_ROLES, PROPOSAL_SEEDS, strict=True):
        proposal_path = cache_root / row_hash / f"proposal-{seed}.json"
        proposal = _cache(
            proposal_path,
            _proposal_request,
            endpoint=endpoint,
            model=model,
            problem=problem,
            reference_answer=reference_answer,
            gold_solution=gold_solution,
            role=role,
            seed=seed,
            timeout=timeout,
        )
        attempt: dict[str, Any] = {"proposal": proposal, "sound_audits": []}
        if proposal.get("pass") is not True:
            attempts.append(attempt)
            continue
        menu = pipeline._menu_from_payload(proposal["menu"])
        for audit_seed, audit_role in zip(
            pipeline.SOUNDNESS_SEEDS,
            pipeline.SOUNDNESS_ROLES,
            strict=True,
        ):
            audit_path = (
                cache_root
                / row_hash
                / f"soundness-S1-{audit_seed}-{menu.sha256[:16]}.json"
            )
            attempt["sound_audits"].append(
                pipeline._cached_request(
                    audit_path,
                    pipeline._sound_request,
                    endpoint=endpoint,
                    model=model,
                    problem=problem,
                    reference_answer=reference_answer,
                    menu=menu,
                    strategy_id="S1",
                    role=audit_role,
                    seed=audit_seed,
                    timeout=timeout,
                )
            )
        attempts.append(attempt)
        if all(
            pipeline._sound_record_passes(
                menu,
                "S1",
                audit,
                reference_answer=reference_answer,
            )
            for audit in attempt["sound_audits"]
        ):
            return {
                "schema": "e49e_singleton_repair_record_v1",
                "repair_version": REPAIR_VERSION,
                "row_id": row_id,
                "problem_sha256": _sha256_bytes(problem.encode("utf-8")),
                "reference_answer_sha256": _sha256_bytes(
                    reference_answer.encode("utf-8")
                ),
                "attempts": attempts,
                "menu": json.loads(menu.canonical_json),
                "menu_sha256": menu.sha256,
                "pass": True,
            }
    return {
        "schema": "e49e_singleton_repair_record_v1",
        "repair_version": REPAIR_VERSION,
        "row_id": row_id,
        "problem_sha256": _sha256_bytes(problem.encode("utf-8")),
        "reference_answer_sha256": _sha256_bytes(
            reference_answer.encode("utf-8")
        ),
        "attempts": attempts,
        "error": "no double-sound singleton repair",
        "pass": False,
    }


def _repair_record_passes(
    record: dict[str, Any],
    *,
    row_id: str,
    problem: str,
    reference_answer: str,
) -> bool:
    if (
        not isinstance(record, dict)
        or record.get("schema") != "e49e_singleton_repair_record_v1"
        or record.get("repair_version") != REPAIR_VERSION
        or record.get("row_id") != row_id
        or record.get("problem_sha256")
        != _sha256_bytes(problem.encode("utf-8"))
        or record.get("reference_answer_sha256")
        != _sha256_bytes(reference_answer.encode("utf-8"))
        or not isinstance(record.get("attempts"), list)
        or not 1 <= len(record["attempts"]) <= len(PROPOSAL_SEEDS)
    ):
        return False
    accepted_menu = None
    for index, attempt in enumerate(record["attempts"]):
        if not isinstance(attempt, dict):
            return False
        proposal = attempt.get("proposal")
        audits = attempt.get("sound_audits")
        expected_role = PROPOSAL_ROLES[index]
        expected_seed = PROPOSAL_SEEDS[index]
        if (
            not isinstance(proposal, dict)
            or proposal.get("repair_version") != REPAIR_VERSION
            or proposal.get("kind") != "singleton_proposal"
            or proposal.get("role") != expected_role
            or proposal.get("seed") != expected_seed
            or not isinstance(audits, list)
        ):
            return False
        if proposal.get("pass") is not True:
            if audits:
                return False
            continue
        if (
            proposal.get("finish_reason") != "stop"
            or not str(proposal.get("response_id") or "")
            or not str(proposal.get("content_sha256") or "")
            or not isinstance(proposal.get("menu"), dict)
        ):
            return False
        try:
            menu = pipeline._menu_from_payload(proposal["menu"])
            closed = pipeline._extract_closed_candidate(menu, "S1")
        except (TypeError, ValueError):
            return False
        if (
            closed is None
            or closed.sha256 != proposal.get("menu_sha256")
            or len(closed.strategies) != 1
            or list(closed.strategies[0].action_ids)
            != [action.action_id for action in closed.actions]
            or not _proposal_is_nonleaking(
                closed,
                problem=problem,
                reference_answer=reference_answer,
            )
            or len(audits) != 2
            or {
                (audit.get("seed"), audit.get("role"))
                for audit in audits
                if isinstance(audit, dict)
            }
            != set(
                zip(
                    pipeline.SOUNDNESS_SEEDS,
                    pipeline.SOUNDNESS_ROLES,
                    strict=True,
                )
            )
        ):
            return False
        double_sound = all(
            pipeline._sound_record_passes(
                closed,
                "S1",
                audit,
                reference_answer=reference_answer,
            )
            for audit in audits
        )
        if double_sound:
            accepted_menu = closed
            if index != len(record["attempts"]) - 1:
                return False
            break

    if record.get("pass") is not True:
        return accepted_menu is None and not isinstance(record.get("menu"), dict)
    if accepted_menu is None or not isinstance(record.get("menu"), dict):
        return False
    try:
        recorded_menu = pipeline._menu_from_payload(record["menu"])
    except (TypeError, ValueError):
        return False
    return bool(
        record.get("menu_sha256") == accepted_menu.sha256
        and recorded_menu.canonical_json == accepted_menu.canonical_json
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--input-e49d", type=pathlib.Path, required=True)
    parser.add_argument("--input-e49e", type=pathlib.Path, required=True)
    parser.add_argument("--input-augmentation", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
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
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--repair-workers", type=int, default=1)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    raw_path = args.input_e49e / "trace_bank_records.jsonl"
    raw = pipeline._load_latest(raw_path)
    augmentation_path = None
    augmentation: dict[str, dict[str, Any]] | None = None
    if args.input_augmentation is not None:
        augmentation_path = (
            args.input_augmentation / "augmentation_records.jsonl"
        )
        augmentation = pipeline._load_latest(augmentation_path)
    input_records = pipeline._load_latest(
        args.input_e49d / "menu_records.jsonl"
    )
    train_dict = load_from_disk(str(args.source / "train"))
    eval_dict = load_from_disk(str(args.source / "eval"))
    train_name = next(iter(train_dict))
    eval_name = next(iter(eval_dict))
    splits = {
        "train": train_dict[train_name],
        "eval": eval_dict[eval_name],
    }
    work = []
    for split, dataset in splits.items():
        for index, row in enumerate(dataset):
            row_id = pipeline.base._row_id(split, index, row)
            work.append((split, index, row_id, row))
    expected_ids = {row_id for _, _, row_id, _ in work}
    if set(raw) != expected_ids or set(input_records) != expected_ids:
        raise RuntimeError("repair inputs do not cover the exact source rows")
    if augmentation is not None and set(augmentation) != expected_ids:
        raise RuntimeError(
            "kernel augmentation does not cover the exact source rows"
        )

    references = {
        row_id: str(row["answer"]) for _, _, row_id, row in work
    }
    invalid_controls = pipeline._load_known_invalid_controls(
        args.known_invalid_controls,
        input_records,
    )
    invalid_results = pipeline._known_invalid_control_results(
        raw,
        invalid_controls,
        input_records,
        references,
    )
    equivalent_controls = pipeline._load_known_equivalent_controls(
        args.known_equivalent_controls,
        input_records,
    )
    equivalent_results = pipeline._known_equivalent_control_results(
        raw,
        equivalent_controls,
        input_records,
        references,
    )
    if not all(row["rejected_by_soundness"] for row in invalid_results):
        raise RuntimeError("known-invalid control gate failed before repair")
    if not all(row["rejected_as_new"] for row in equivalent_results):
        raise RuntimeError("known-equivalent control gate failed before repair")

    raw_selected_by_row = {}
    for split, _, row_id, row in work:
        problem = str(row["problem"])
        reference = str(row["answer"])
        if (
            raw[row_id].get("problem_sha256")
            != _sha256_bytes(problem.encode("utf-8"))
            or raw[row_id].get("reference_answer_sha256")
            != _sha256_bytes(reference.encode("utf-8"))
        ):
            raise RuntimeError(f"raw row binding changed: {row_id}")
        if augmentation is None:
            raw_selected_by_row[row_id] = _raw_selected(
                raw[row_id],
                input_record=input_records[row_id],
                reference_answer=reference,
            )
        else:
            raw_selected_by_row[row_id] = _augmentation_selected(
                augmentation[row_id],
                raw_record=raw[row_id],
                input_record=input_records[row_id],
                problem=problem,
                reference_answer=reference,
            )

    if args.preflight_only:
        gaps = [
            row_id
            for _, _, row_id, _ in work
            if raw_selected_by_row[row_id] is None
        ]
        print(
            json.dumps(
                {
                    "schema": "e49e_singleton_repair_preflight_v1",
                    "repair_version": REPAIR_VERSION,
                    "answer_normalization_version": (
                        pipeline.ANSWER_NORMALIZATION_VERSION
                    ),
                    "row_count": len(work),
                    "trace_certified_row_count": len(work) - len(gaps),
                    "singleton_repair_row_count": len(gaps),
                    "singleton_repair_rows": gaps,
                    "known_invalid_control_count": len(invalid_results),
                    "known_equivalent_control_count": len(
                        equivalent_results
                    ),
                    "all_controls_pass": True,
                    "raw_trace_records_sha256": _sha256_bytes(
                        raw_path.read_bytes()
                    ),
                    "augmentation_records_sha256": (
                        _sha256_bytes(augmentation_path.read_bytes())
                        if augmentation_path is not None
                        else None
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    endpoint, model = pipeline.base._endpoint(args.endpoint)
    repair_path = args.evidence / "repair_records.jsonl"
    repairs = pipeline._load_latest(repair_path)
    gap_work = [
        (split, row_id, row)
        for split, _, row_id, row in work
        if raw_selected_by_row[row_id] is None
    ]
    for _, row_id, row in gap_work:
        repair = repairs.get(row_id)
        if repair is not None and not _repair_record_passes(
            repair,
            row_id=row_id,
            problem=str(row["problem"]),
            reference_answer=str(row["answer"]),
        ):
            raise RuntimeError(f"cached singleton repair changed: {row_id}")
    pending_repairs = [
        (split, row_id, row)
        for split, row_id, row in gap_work
        if repairs.get(row_id, {}).get("pass") is not True
    ]

    def run_repair(item):
        _, row_id, row = item
        return _repair_one(
            endpoint=endpoint,
            model=model,
            row_id=row_id,
            problem=str(row["problem"]),
            reference_answer=str(row["answer"]),
            gold_solution=str(row.get("solution") or ""),
            cache_root=args.evidence / "request_cache",
            timeout=args.timeout,
        )

    worker_count = max(1, min(int(args.repair_workers), len(pending_repairs)))
    if worker_count == 1:
        completed_repairs = map(run_repair, pending_repairs)
        for repair in completed_repairs:
            pipeline._append_jsonl(repair_path, repair)
            repairs[repair["row_id"]] = repair
    elif pending_repairs:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {
                pool.submit(run_repair, item): item[1]
                for item in pending_repairs
            }
            for future in as_completed(futures):
                repair = future.result()
                pipeline._append_jsonl(repair_path, repair)
                repairs[repair["row_id"]] = repair

    final_records: dict[str, dict[str, Any]] = {}
    failures = []
    for split, _, row_id, row in work:
        problem = str(row["problem"])
        reference = str(row["answer"])
        selected = raw_selected_by_row[row_id]
        if selected is not None:
            menu, certification = selected
            selected_origin = "original_trace_certification"
            augmentation_record_sha256 = None
            if augmentation is not None:
                selected_origin = str(
                    augmentation[row_id].get("origin")
                    or "kernel_augmentation"
                )
                augmentation_record_sha256 = _canonical_sha256(
                    augmentation[row_id]
                )
            final_records[row_id] = {
                "schema": "e49e_repaired_trace_bank_record_v1",
                "repair_version": REPAIR_VERSION,
                "row_id": row_id,
                "split": split,
                "origin": selected_origin,
                "menu": json.loads(menu.canonical_json),
                "menu_sha256": menu.sha256,
                "certification": certification,
                "raw_record_sha256": _canonical_sha256(raw[row_id]),
                "augmentation_record_sha256": (
                    augmentation_record_sha256
                ),
                "pass": True,
            }
            continue

        repair = repairs.get(row_id)
        if (
            repair is None
            or repair.get("pass") is not True
            or not _repair_record_passes(
                repair,
                row_id=row_id,
                problem=problem,
                reference_answer=reference,
            )
        ):
            failures.append(row_id)
            continue
        menu = pipeline._menu_from_payload(repair["menu"])
        final_records[row_id] = {
            "schema": "e49e_repaired_trace_bank_record_v1",
            "repair_version": REPAIR_VERSION,
            "row_id": row_id,
            "split": split,
            "origin": REPAIR_ORIGIN,
            "menu": json.loads(menu.canonical_json),
            "menu_sha256": menu.sha256,
            "repair_record_sha256": _canonical_sha256(repair),
            "raw_record_sha256": _canonical_sha256(raw[row_id]),
            "pass": True,
        }

    if failures:
        pipeline._write_json(
            args.evidence / "generation_summary.json",
            {
                "schema": "e49e_singleton_repair_summary_v1",
                "pass": False,
                "failures": failures,
            },
        )
        raise RuntimeError(
            f"{len(failures)} singleton repairs failed: {failures[:10]}"
        )

    output_splits = {}
    for split, dataset in splits.items():
        data = dataset.to_dict()
        augmented = []
        menu_hashes = []
        origins = []
        for index, row in enumerate(dataset):
            row_id = pipeline.base._row_id(split, index, row)
            record = final_records[row_id]
            augmented.append(pipeline._embed(str(row["problem"]), record["menu"]))
            menu_hashes.append(record["menu_sha256"])
            origins.append(record["origin"])
        data["original_problem"] = list(data["problem"])
        data["problem"] = augmented
        data["strategy_menu_sha256"] = menu_hashes
        data["strategy_menu_origin"] = origins
        output_splits[split] = Dataset.from_dict(data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(
        tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    )
    try:
        DatasetDict({train_name: output_splits["train"]}).save_to_disk(
            str(staging / "train")
        )
        DatasetDict({eval_name: output_splits["eval"]}).save_to_disk(
            str(staging / "eval")
        )
        counts = [
            len(final_records[row_id]["menu"]["strategies"])
            for _, _, row_id, _ in work
        ]
        repair_count = sum(
            record["origin"] == REPAIR_ORIGIN
            for record in final_records.values()
        )
        final_path = args.evidence / "final_records.jsonl"
        with final_path.open("w", encoding="utf-8") as handle:
            for _, _, row_id, _ in work:
                handle.write(
                    json.dumps(
                        final_records[row_id],
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        manifest = {
            "schema": "e49e_repaired_trace_bank_materialization_v1",
            "repair_version": REPAIR_VERSION,
            "answer_normalization_version": (
                pipeline.ANSWER_NORMALIZATION_VERSION
            ),
            "source_train_tree_sha256": _tree_hash(args.source / "train"),
            "source_eval_tree_sha256": _tree_hash(args.source / "eval"),
            "raw_trace_records_sha256": _sha256_bytes(raw_path.read_bytes()),
            "augmentation_records_sha256": (
                _sha256_bytes(augmentation_path.read_bytes())
                if augmentation_path is not None
                else None
            ),
            "e49d_input_sha256": _sha256_bytes(
                (args.input_e49d / "menu_records.jsonl").read_bytes()
            ),
            "final_records_sha256": _sha256_bytes(final_path.read_bytes()),
            "menu_count": len(work),
            "singleton_repair_count": repair_count,
            "multi_strategy_menu_count": sum(count >= 2 for count in counts),
            "known_invalid_control_results": invalid_results,
            "known_equivalent_control_results": equivalent_results,
            "all_controls_pass": True,
            "train_tree_sha256": _tree_hash(staging / "train"),
            "eval_tree_sha256": _tree_hash(staging / "eval"),
        }
        pipeline._write_json(
            staging / "MATERIALIZATION_MANIFEST.json",
            manifest,
        )
        if args.output.exists():
            raise RuntimeError(f"output already exists: {args.output}")
        os.replace(staging, args.output)
    finally:
        if staging.exists():
            for item in sorted(staging.rglob("*"), reverse=True):
                if item.is_file():
                    item.unlink()
                else:
                    item.rmdir()
            staging.rmdir()

    pipeline._write_json(
        args.evidence / "generation_summary.json",
        {
            "schema": "e49e_singleton_repair_summary_v1",
            "pass": True,
            "menu_count": len(work),
            "singleton_repair_count": repair_count,
            "multi_strategy_menu_count": sum(count >= 2 for count in counts),
            "materialization_manifest": str(
                args.output / "MATERIALIZATION_MANIFEST.json"
            ),
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
