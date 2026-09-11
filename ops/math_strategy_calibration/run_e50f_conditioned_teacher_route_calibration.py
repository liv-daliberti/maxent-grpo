#!/usr/bin/env python3
"""Calibrate answer-blind conditioned 72B routes for natural 0.5B support."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = pathlib.Path(__file__).resolve()
BASE_PATH = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e50c_72b_teacher_route_calibration.py"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e50f_conditioned_teacher_route_calibration_20260726.md"
)
E50C_RESULT = (
    ROOT
    / "var/artifacts/e50c_72b_teacher_route_calibration_v1/result.json"
)
SCHEMA = "e50f_conditioned_teacher_route_calibration_v1"
PROPOSAL_SEED = 500731
RELATION_SEEDS = (470741, 470742)
EXECUTION_SAMPLES = 8
FORCED_SAMPLES = 16
UNFORCED_SAMPLES = 64
SEED = 500732
QUARANTINED_CORPUS_ONLY = True
SAFE_PAIR_OPTIONS = (
    (
        "Chinese remainder theorem (CRT) construction",
        "exhaustive finite constraint search",
    ),
    ("calculus using derivatives", "a sharp inequality such as AM-GM"),
    ("the Euclidean algorithm", "prime factorization"),
    ("the quadratic formula", "Vieta relations"),
    ("explicit polynomial expansion", "function value evaluation"),
    (
        "exhaustive finite constraint search",
        "solve symbolically for the interval",
    ),
    (
        "exhaustive finite constraint search",
        "a convexity identity",
    ),
    ("factor localization by square-root bounds", "the quadratic formula"),
    (
        "vector geometry using a cross product",
        "synthetic geometry using an isosceles altitude",
    ),
    ("dynamic programming", "a closed-form count"),
    ("a complementary count", "inclusion-exclusion"),
    ("incidence double counting", "unordered pair counting"),
)


def _load_base() -> Any:
    spec = importlib.util.spec_from_file_location("e50f_base_e50c", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E50C base: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
                    json.dumps(row, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _verify_activation() -> dict[str, Any]:
    if not E50C_RESULT.is_file():
        raise RuntimeError("E50F requires terminal E50C")
    result = json.loads(E50C_RESULT.read_text(encoding="utf-8"))
    if result.get("schema") != "e50c_72b_teacher_route_calibration_v1":
        raise RuntimeError("E50F E50C schema mismatch")
    # E50F originally activated only after an E50C failure.  The prospective
    # E50F2/E50F4 audits subsequently found false-new errors in the
    # open-ended relation judge before E50C became terminal.  E50C therefore
    # cannot authorize training even if its natural-support gates pass.
    # Always materialize this answer-blind conditioned corpus so E50G can
    # apply the fail-closed deterministic signature boundary.
    return result


def _proposal_schema() -> dict[str, Any]:
    method = {
        "type": "object",
        "additionalProperties": False,
        "required": ["label", "actions"],
        "properties": {
            "label": {"type": "string", "minLength": 1},
            "actions": {
                "type": "array",
                "minItems": 2,
                "maxItems": 6,
                "items": {"type": "string", "minLength": 1},
            },
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["methods", "decisive_difference"],
        "properties": {
            "methods": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": method,
            },
            "decisive_difference": {
                "type": "string",
                "minLength": 1,
            },
        },
    }


def _post(
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout: int,
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    error: Exception | None = None
    for attempt in range(3):
        try:
            with opener.open(request, timeout=timeout) as response:
                decoded = json.loads(response.read().decode("utf-8"))
            choices = decoded.get("choices") or []
            if not choices:
                raise RuntimeError("Qwen72 returned no choices")
            return decoded
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            RuntimeError,
        ) as exc:
            error = exc
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(f"Qwen72 request failed after retries: {error}")


def _proposal_payload(model: str, problem: str, problem_order: int) -> dict[str, Any]:
    options = "\n".join(
        f"- {left} VERSUS {right}" for left, right in SAFE_PAIR_OPTIONS
    )
    user = f"""Propose exactly two genuinely different mathematical methods
for solving the problem below. Choose one and only one pair from the frozen
decisive-engine list below for which BOTH methods are mathematically sound
and sufficient for this exact problem. Preserve each engine phrase verbatim
in that method's label or actions. Do not combine the two engines in one
method.

Each method must contain 2 to 6 concrete, ordered, problem-specific actions.
The methods must differ in their listed decisive engines, not notation,
wording, algebraic rearrangement, or step order. Do not solve the problem,
state the final answer, encode the final answer, or refer to another method's
hidden steps. Explain the decisive difference in one short field.

FROZEN ELIGIBLE ENGINE PAIRS:
{options}

PROBLEM:
{problem}
"""
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You design answer-blind, executable mathematical "
                    "methods. Return valid JSON only."
                ),
            },
            {"role": "user", "content": user},
        ],
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 1024,
        "seed": PROPOSAL_SEED + problem_order,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "e50f_method_pair",
                "strict": True,
                "schema": _proposal_schema(),
            },
        },
    }


def _execution_payload(
    *,
    model: str,
    problem: str,
    method: dict[str, Any],
    problem_order: int,
    route_index: int,
) -> dict[str, Any]:
    actions = "\n".join(
        f"{index}. {action}"
        for index, action in enumerate(method["actions"], start=1)
    )
    user = f"""Solve the mathematics problem by literally following every
action in the prescribed method, in order. Give a complete derivation,
introduce no substitute method or hidden decisive step, and finish with a
boxed final answer.

PROBLEM:
{problem}

PRESCRIBED METHOD ({method['label']}):
{actions}
"""
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a rigorous mathematical problem solver.",
            },
            {"role": "user", "content": user},
        ],
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 1024,
        "n": EXECUTION_SAMPLES,
        "seed": SEED + problem_order * 2 + route_index,
        "stream": False,
    }


def _generate_problem(
    *,
    endpoint: str,
    model: str,
    problem: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    problem_order = int(problem["problem_order"])
    try:
        proposal_payload = _proposal_payload(
            model, str(problem["problem"]), problem_order
        )
        proposal_response = _post(
            endpoint, proposal_payload, timeout=timeout
        )
        proposal_choice = (proposal_response["choices"] or [{}])[0]
        proposal_text = str(
            (proposal_choice.get("message") or {}).get("content") or ""
        )
        proposal = json.loads(proposal_text)
        methods = proposal.get("methods")
        if (
            not isinstance(methods, list)
            or len(methods) != 2
            or any(
                not isinstance(method.get("actions"), list)
                or not 2 <= len(method["actions"]) <= 6
                for method in methods
            )
        ):
            raise RuntimeError("conditioned method proposal contract failed")
        normalized = [
            tuple(
                " ".join(str(action).lower().split())
                for action in method["actions"]
            )
            for method in methods
        ]
        if normalized[0] == normalized[1]:
            raise RuntimeError("conditioned methods are textually identical")
        executions = []
        for route_index, method in enumerate(methods):
            payload = _execution_payload(
                model=model,
                problem=str(problem["problem"]),
                method=method,
                problem_order=problem_order,
                route_index=route_index,
            )
            response = _post(endpoint, payload, timeout=timeout)
            choices = response.get("choices") or []
            if len(choices) != EXECUTION_SAMPLES:
                raise RuntimeError("conditioned execution sample mismatch")
            for choice_index, choice in enumerate(choices):
                text = str(
                    (choice.get("message") or {}).get("content") or ""
                )
                executions.append(
                    {
                        "route_index": route_index,
                        "choice_index": choice_index,
                        "response_id": response.get("id"),
                        "finish_reason": choice.get("finish_reason"),
                        "response_sha256": _sha256_text(text),
                        "response": text,
                    }
                )
        return {
            "problem_order": problem_order,
            "source_index": int(problem["source_index"]),
            "row_id": str(problem["unique_id"]),
            "proposal": proposal,
            "proposal_response_id": proposal_response.get("id"),
            "proposal_finish_reason": proposal_choice.get("finish_reason"),
            "proposal_request_sha256": _sha256_text(
                json.dumps(
                    proposal_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            "answer_was_not_in_proposal_payload": True,
            "answer_was_not_in_execution_payloads": True,
            "executions": executions,
        }
    except Exception as exc:
        return {
            "problem_order": problem_order,
            "source_index": int(problem["source_index"]),
            "row_id": str(problem["unique_id"]),
            "error": str(exc),
            "executions": [],
        }


def _relation_record(
    *,
    pairwise_module: Any,
    endpoint: str,
    model: str,
    candidate: dict[str, Any],
    route_members: list[list[dict[str, Any]]],
    timeout: int,
) -> dict[str, Any]:
    base = {
        key: candidate[key]
        for key in (
            "problem_order",
            "source_index",
            "row_id",
            "subject",
            "level",
        )
    }
    try:
        route_positive_counts = [len(members) for members in route_members]
        selected = [members[:3] for members in route_members]
        if any(len(members) < 2 for members in selected):
            raise RuntimeError("each conditioned route requires two positives")
        items = []
        route_ids: list[list[str]] = []
        for route_index, members in enumerate(selected):
            ids = []
            for member_index, member in enumerate(members):
                item_id = f"R{route_index + 1}_{member_index + 1}"
                ids.append(item_id)
                items.append((item_id, str(member["text"])))
            route_ids.append(ids)
        pairs: list[tuple[str, str, str]] = []
        expected: dict[str, str] = {}
        pair_index = 0
        for ids in route_ids:
            for left_index in range(len(ids)):
                for right_index in range(left_index + 1, len(ids)):
                    pair_id = f"PAIR_{pair_index:04d}"
                    pairs.append((pair_id, ids[left_index], ids[right_index]))
                    expected[pair_id] = "same"
                    pair_index += 1
        for left in route_ids[0]:
            for right in route_ids[1]:
                pair_id = f"PAIR_{pair_index:04d}"
                pairs.append((pair_id, left, right))
                expected[pair_id] = "different"
                pair_index += 1
        audits = []
        for seed in RELATION_SEEDS:
            judge = pairwise_module.MathStrategyCanonicalizer(
                endpoint=endpoint,
                model=model,
                timeout_seconds=timeout,
                max_workers=1,
                permutation_seeds=(470721, 470722),
                max_item_chars=4000,
                missing_ids_are_ambiguous=True,
            )
            observed = judge._judge_relations(
                problem=str(candidate["problem"]),
                base_items=items,
                pairs=pairs,
                seed=seed,
            )
            audits.append({"seed": seed, "relations": observed})
        passed = all(
            audit["relations"] == expected for audit in audits
        )
        methods = candidate["proposal"]["methods"]
        clusters = []
        for route_index, members in enumerate(selected):
            method = methods[route_index]
            clusters.append(
                {
                    "cluster_id": f"C{route_index + 1}",
                    "strategy": (
                        f"{method['label']}: "
                        + " -> ".join(str(item) for item in method["actions"])
                    ),
                    "strategy_key": f"conditioned_route_{route_index + 1}",
                    "member_ids": [str(member["sample_id"]) for member in members],
                }
            )
        return {
            **base,
            "relation_audits": audits,
            "clusters": clusters,
            "cluster_eligible": passed,
            "minimum_cluster_size": min(
                len(row["member_ids"]) for row in clusters
            ),
            "combined_cluster_size": sum(
                len(row["member_ids"]) for row in clusters
            ),
            "route_positive_counts": route_positive_counts,
            "minimum_teacher_execution_count": min(route_positive_counts),
            "combined_teacher_execution_count": sum(route_positive_counts),
        }
    except Exception as exc:
        return {**base, "cluster_eligible": False, "error": str(exc)}


def _identity(e50c: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol_sha256": _sha256(PROTOCOL),
        "script_sha256": _sha256(SCRIPT),
        "base_e50c_script_sha256": _sha256(BASE_PATH),
        "e50c_result_sha256": _sha256(E50C_RESULT),
        "e50c_result_schema": e50c["schema"],
        "e47_manifest_sha256": _sha256(BASE.E47_MANIFEST),
        "e47_problems_sha256": _sha256(BASE.E47_PROBLEMS),
        "endpoint_record_sha256": _sha256(BASE.ENDPOINT_RECORD),
        "pairwise_source_sha256": _sha256(BASE.PAIRWISE_SOURCE),
        "e49t_identity_sha256": _sha256(BASE.E49T_IDENTITY),
        "proposal_seed": PROPOSAL_SEED,
        "relation_seeds": list(RELATION_SEEDS),
        "seed": SEED,
    }


def _failure(
    *,
    result_path: pathlib.Path,
    reason: str,
    e50c: dict[str, Any],
    generated_path: pathlib.Path,
    relation_path: pathlib.Path,
    menu_path: pathlib.Path,
    candidates: list[dict[str, Any]],
    eligible: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
) -> None:
    _write_json(
        result_path,
        {
            "schema": SCHEMA,
            "pass": False,
            "failure": reason,
            "candidate_count": len(candidates),
            "relation_eligible_count": len(eligible),
            "double_audited_menu_count": len(accepted),
            "bidirectionally_executable_count": 0,
            "selected_source_indices": [],
            "identity": _identity(e50c),
            "conditioned_teacher_records_sha256": _sha256(generated_path),
            "relation_records_sha256": _sha256(relation_path),
            "menu_records_sha256": _sha256(menu_path),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    output = args.output_root.resolve()
    result_path = output / "result.json"
    if result_path.exists():
        raise RuntimeError(f"fresh E50F result required: {result_path}")
    e50c = _verify_activation()

    manifest = json.loads(BASE.E47_MANIFEST.read_text(encoding="utf-8"))
    problems = BASE._load_jsonl(BASE.E47_PROBLEMS)
    expected_indices = manifest["selection"]["source_indices"]
    if (
        len(problems) != 50
        or [int(row["source_index"]) for row in problems] != expected_indices
        or any(int(row["level"]) != 5 for row in problems)
    ):
        raise RuntimeError("E50F frozen E47 problem cohort drifted")
    for problem_order, problem in enumerate(problems):
        problem["problem_order"] = problem_order

    from oat_drgrpo.math_grader import boxed_reward_fn
    from oat_drgrpo.templates import apply_qwen_math_template

    endpoint, judge_model = BASE._endpoint(BASE.ENDPOINT_RECORD)
    partial_path = (
        output / "private/conditioned_teacher_records.partial.jsonl"
    )
    checkpoint_identity_path = (
        output / "private/conditioned_teacher_checkpoint_identity.json"
    )
    checkpoint_identity = _identity(e50c)
    if checkpoint_identity_path.is_file():
        observed_checkpoint_identity = json.loads(
            checkpoint_identity_path.read_text(encoding="utf-8")
        )
        if observed_checkpoint_identity != checkpoint_identity:
            raise RuntimeError("E50F partial checkpoint identity drifted")
    else:
        _write_json(checkpoint_identity_path, checkpoint_identity)
    generated = _read_jsonl(partial_path) if partial_path.is_file() else []
    completed_orders = [int(row["problem_order"]) for row in generated]
    if (
        len(completed_orders) != len(set(completed_orders))
        or not set(completed_orders) <= set(range(50))
    ):
        raise RuntimeError("E50F partial generation checkpoint is invalid")
    remaining = [
        problem
        for problem in problems
        if int(problem["problem_order"]) not in set(completed_orders)
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                _generate_problem,
                endpoint=endpoint,
                model=judge_model,
                problem=problem,
                timeout=args.timeout,
            )
            for problem in remaining
        ]
        for future in as_completed(futures):
            record = future.result()
            generated.append(record)
            generated.sort(key=lambda row: int(row["problem_order"]))
            _write_jsonl(partial_path, generated)
            print(
                json.dumps(
                    {
                        "phase": "conditioned_teacher",
                        "problem_order": record["problem_order"],
                        "execution_count": len(record["executions"]),
                        "error": record.get("error"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    generated.sort(key=lambda row: int(row["problem_order"]))
    if (
        len(generated) != 50
        or [int(row["problem_order"]) for row in generated]
        != list(range(50))
    ):
        raise RuntimeError("E50F conditioned corpus is incomplete")
    generated_path = output / "private/conditioned_teacher_records.jsonl"
    _write_jsonl(generated_path, generated)
    if QUARANTINED_CORPUS_ONLY:
        _write_json(
            result_path,
            {
                "schema": SCHEMA,
                "pass": False,
                "failure": (
                    "open_relation_judge_quarantined_after_frozen_"
                    "conditioned_corpus_generation"
                ),
                "candidate_count": 0,
                "relation_eligible_count": 0,
                "double_audited_menu_count": 0,
                "bidirectionally_executable_count": 0,
                "selected_source_indices": [],
                "generation_problem_count": len(generated),
                "generation_error_count": sum(
                    bool(row.get("error")) for row in generated
                ),
                "conditioned_execution_count": sum(
                    len(row.get("executions") or ()) for row in generated
                ),
                "identity": _identity(e50c),
                "conditioned_teacher_records_sha256": _sha256(
                    generated_path
                ),
            },
        )
        print(result_path)
        return

    problem_by_order = {
        int(problem["problem_order"]): problem for problem in problems
    }
    candidates = []
    route_members_by_source: dict[int, list[list[dict[str, Any]]]] = {}
    for record in generated:
        if record.get("error"):
            continue
        problem = problem_by_order[int(record["problem_order"])]
        members = [[], []]
        exemplars = []
        for execution in record["executions"]:
            if len(str(execution["response"])) > 4000:
                continue
            _, reward = boxed_reward_fn(
                str(execution["response"]),
                str(problem["answer"]),
                fast=False,
            )
            if float(reward) <= 0:
                continue
            route_index = int(execution["route_index"])
            sample_id = (
                f"r{route_index + 1}s"
                f"{int(execution['choice_index']):02d}_"
                f"{str(execution['response_sha256'])[:12]}"
            )
            member = {
                "sample_id": sample_id,
                "sample_index": int(execution["choice_index"]),
                "text": str(execution["response"]),
                "response_sha256": str(execution["response_sha256"]),
            }
            members[route_index].append(member)
            exemplars.append(member)
        for route in members:
            route.sort(key=lambda row: row["response_sha256"])
        if any(len(route) < 2 for route in members):
            continue
        source_index = int(problem["source_index"])
        candidate = {
            "problem_order": int(problem["problem_order"]),
            "source_index": source_index,
            "row_id": str(problem["unique_id"]),
            "problem": str(problem["problem"]),
            "answer": str(problem["answer"]),
            "subject": str(problem["subject"]),
            "level": int(problem["level"]),
            "proposal": record["proposal"],
            "exemplars": exemplars,
        }
        candidates.append(candidate)
        route_members_by_source[source_index] = members

    pairwise_module = BASE._load_pairwise_module()
    relations = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                _relation_record,
                pairwise_module=pairwise_module,
                endpoint=endpoint,
                model=judge_model,
                candidate=candidate,
                route_members=route_members_by_source[
                    int(candidate["source_index"])
                ],
                timeout=args.timeout,
            )
            for candidate in candidates
        ]
        for future in as_completed(futures):
            record = future.result()
            relations.append(record)
            print(
                json.dumps(
                    {
                        "phase": "relation",
                        "source_index": record["source_index"],
                        "eligible": record["cluster_eligible"],
                        "error": record.get("error"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    relations.sort(key=lambda row: int(row["source_index"]))
    relation_path = output / "relation_records.jsonl"
    _write_jsonl(relation_path, relations)
    eligible = [row for row in relations if row["cluster_eligible"]]
    relation_by_source = {
        int(row["source_index"]): row for row in eligible
    }
    candidate_by_source = {
        int(row["source_index"]): row for row in candidates
    }

    menus = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                BASE._menu_candidate,
                endpoint=endpoint,
                model=judge_model,
                candidate=candidate_by_source[int(record["source_index"])],
                cluster_record=record,
                timeout=args.timeout,
            )
            for record in eligible
        ]
        for future in as_completed(futures):
            record = future.result()
            relation = relation_by_source[int(record["source_index"])]
            record["route_positive_counts"] = relation[
                "route_positive_counts"
            ]
            record["minimum_teacher_execution_count"] = relation[
                "minimum_teacher_execution_count"
            ]
            record["combined_teacher_execution_count"] = relation[
                "combined_teacher_execution_count"
            ]
            menus.append(record)
            print(
                json.dumps(
                    {
                        "phase": "menu",
                        "source_index": record["source_index"],
                        "double_audit_pass": record["double_audit_pass"],
                        "error": record.get("error"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    menus.sort(key=lambda row: int(row["source_index"]))
    menu_path = output / "menu_records.jsonl"
    _write_jsonl(menu_path, menus)
    accepted = [row for row in menus if row["double_audit_pass"]]
    accepted.sort(
        key=lambda row: (
            -int(row["minimum_teacher_execution_count"]),
            -int(row["combined_teacher_execution_count"]),
            int(row["source_index"]),
        )
    )
    if not accepted:
        _failure(
            result_path=result_path,
            reason="no_double_audited_conditioned_teacher_menu",
            e50c=e50c,
            generated_path=generated_path,
            relation_path=relation_path,
            menu_path=menu_path,
            candidates=candidates,
            eligible=eligible,
            accepted=accepted,
        )
        print(result_path)
        return

    import vllm

    forced_cases = []
    unforced_cases = []
    for record in accepted:
        candidate = candidate_by_source[int(record["source_index"])]
        menu = BASE._parse_menu(record["menu"])
        for strategy in menu.strategies:
            forced_cases.append(
                {
                    "source_index": int(record["source_index"]),
                    "row_id": str(record["row_id"]),
                    "problem": BASE._forced_problem(
                        candidate["problem"], menu, strategy
                    ),
                    "answer": candidate["answer"],
                    "menu": menu,
                    "strategy": strategy,
                }
            )
        unforced_cases.append(
            {
                "source_index": int(record["source_index"]),
                "row_id": str(record["row_id"]),
                "problem": BASE._neutral_problem(
                    candidate["problem"], menu
                ),
                "answer": candidate["answer"],
                "menu": menu,
            }
        )
    llm = vllm.LLM(
        model=str(BASE.MODEL),
        dtype="bfloat16",
        max_model_len=3072,
        gpu_memory_utilization=0.80,
        swap_space=8.0,
        enable_prefix_caching=True,
        seed=SEED,
    )
    forced_outputs = llm.generate(
        [apply_qwen_math_template(case["problem"]) for case in forced_cases],
        vllm.SamplingParams(
            n=FORCED_SAMPLES,
            temperature=1.0,
            top_p=1.0,
            max_tokens=1024,
            seed=SEED + 1,
        ),
    )
    unforced_outputs = llm.generate(
        [apply_qwen_math_template(case["problem"]) for case in unforced_cases],
        vllm.SamplingParams(
            n=UNFORCED_SAMPLES,
            temperature=1.0,
            top_p=1.0,
            max_tokens=1024,
            seed=SEED + 2,
        ),
    )

    private_05b = []
    forced_tokens: list[list[int]] = []
    forced_prompts: list[str] = []
    forced_responses: list[str] = []
    forced_positive: list[bool] = []
    for case_index, (case, request_output) in enumerate(
        zip(forced_cases, forced_outputs, strict=True)
    ):
        if len(request_output.outputs) != FORCED_SAMPLES:
            raise RuntimeError("E50F forced sample count mismatch")
        for sample_index, sample in enumerate(request_output.outputs):
            text = str(sample.text)
            _, reward = boxed_reward_fn(text, case["answer"], fast=False)
            positive = float(reward) > 0.0
            forced_tokens.append(list(request_output.prompt_token_ids))
            forced_prompts.append(case["problem"])
            forced_responses.append(text)
            forced_positive.append(positive)
            private_05b.append(
                {
                    "kind": "forced",
                    "case_index": case_index,
                    "source_index": case["source_index"],
                    "row_id": case["row_id"],
                    "strategy_id": case["strategy"].strategy_id,
                    "sample_index": sample_index,
                    "answer_positive": positive,
                    "response_sha256": _sha256_text(text),
                    "response": text,
                }
            )
    unforced_tokens: list[list[int]] = []
    unforced_prompts: list[str] = []
    unforced_responses: list[str] = []
    unforced_positive: list[bool] = []
    for case_index, (case, request_output) in enumerate(
        zip(unforced_cases, unforced_outputs, strict=True)
    ):
        if len(request_output.outputs) != UNFORCED_SAMPLES:
            raise RuntimeError("E50F unforced sample count mismatch")
        for sample_index, sample in enumerate(request_output.outputs):
            text = str(sample.text)
            _, reward = boxed_reward_fn(text, case["answer"], fast=False)
            positive = float(reward) > 0.0
            unforced_tokens.append(list(request_output.prompt_token_ids))
            unforced_prompts.append(case["problem"])
            unforced_responses.append(text)
            unforced_positive.append(positive)
            private_05b.append(
                {
                    "kind": "unforced",
                    "case_index": case_index,
                    "source_index": case["source_index"],
                    "row_id": case["row_id"],
                    "sample_index": sample_index,
                    "answer_positive": positive,
                    "response_sha256": _sha256_text(text),
                    "response": text,
                }
            )
    private_05b_path = output / "private/base_05b_responses.jsonl"
    _write_jsonl(private_05b_path, private_05b)

    canonicalizer = BASE._load_frozen_canonicalizer()
    forced_keys, forced_diagnostics = BASE._canonical_keys(
        canonicalizer=canonicalizer,
        endpoint=endpoint,
        model=judge_model,
        prompt_tokens=forced_tokens,
        prompt_texts=forced_prompts,
        responses=forced_responses,
        positives=forced_positive,
        samples_per_prompt=FORCED_SAMPLES,
        timeout=args.timeout,
        workers=args.workers,
    )
    unforced_keys, unforced_diagnostics = BASE._canonical_keys(
        canonicalizer=canonicalizer,
        endpoint=endpoint,
        model=judge_model,
        prompt_tokens=unforced_tokens,
        prompt_texts=unforced_prompts,
        responses=unforced_responses,
        positives=unforced_positive,
        samples_per_prompt=UNFORCED_SAMPLES,
        timeout=args.timeout,
        workers=args.workers,
    )

    forced_results = []
    for case_index, case in enumerate(forced_cases):
        start = case_index * FORCED_SAMPLES
        stop = start + FORCED_SAMPLES
        probe = canonicalizer(endpoint=endpoint, model=judge_model)
        expected_key = probe._menu_strategy_key(
            case["menu"], case["strategy"].strategy_id
        )
        forced_results.append(
            {
                "source_index": case["source_index"],
                "strategy_id": case["strategy"].strategy_id,
                "answer_success_count": sum(forced_positive[start:stop]),
                "forced_route_success_count": sum(
                    key == expected_key for key in forced_keys[start:stop]
                ),
                "wrong_route_success_count": sum(
                    key is not None and key != expected_key
                    for key in forced_keys[start:stop]
                ),
            }
        )
    menu_by_source = {
        int(record["source_index"]): record for record in accepted
    }
    problems_out = []
    for case_index, case in enumerate(unforced_cases):
        start = case_index * UNFORCED_SAMPLES
        stop = start + UNFORCED_SAMPLES
        probe = canonicalizer(endpoint=endpoint, model=judge_model)
        route_counts = {
            strategy.strategy_id: sum(
                key
                == probe._menu_strategy_key(
                    case["menu"], strategy.strategy_id
                )
                for key in unforced_keys[start:stop]
            )
            for strategy in case["menu"].strategies
        }
        forced_for_problem = [
            row
            for row in forced_results
            if row["source_index"] == case["source_index"]
        ]
        forced_ok = (
            len(forced_for_problem) == 2
            and all(
                row["forced_route_success_count"] >= 1
                for row in forced_for_problem
            )
        )
        unforced_ok = (
            min(route_counts.values()) >= 2
            and sum(route_counts.values()) >= 8
        )
        record = menu_by_source[case["source_index"]]
        problems_out.append(
            {
                "problem_order": int(
                    candidate_by_source[case["source_index"]][
                        "problem_order"
                    ]
                ),
                "source_index": case["source_index"],
                "row_id": case["row_id"],
                "menu": record["menu"],
                "menu_sha256": record["menu_sha256"],
                "route_sources": record["route_sources"],
                "minimum_cluster_size": record["minimum_cluster_size"],
                "combined_cluster_size": record["combined_cluster_size"],
                "route_positive_counts": record["route_positive_counts"],
                "minimum_teacher_execution_count": record[
                    "minimum_teacher_execution_count"
                ],
                "combined_teacher_execution_count": record[
                    "combined_teacher_execution_count"
                ],
                "forced_route_success_counts": {
                    row["strategy_id"]: row["forced_route_success_count"]
                    for row in forced_for_problem
                },
                "unforced_route_success_counts": route_counts,
                "unforced_accepted_count": sum(route_counts.values()),
                "bidirectionally_executable": forced_ok and unforced_ok,
            }
        )
    problems_out.sort(
        key=lambda row: (
            -min(row["unforced_route_success_counts"].values()),
            -int(row["unforced_accepted_count"]),
            -int(row["minimum_teacher_execution_count"]),
            int(row["problem_order"]),
        )
    )
    selected = [
        row["source_index"]
        for row in problems_out
        if row["bidirectionally_executable"]
    ][:10]
    payload = {
        "schema": SCHEMA,
        "pass": len(selected) == 10,
        "candidate_count": len(candidates),
        "relation_eligible_count": len(eligible),
        "double_audited_menu_count": len(accepted),
        "bidirectionally_executable_count": sum(
            row["bidirectionally_executable"] for row in problems_out
        ),
        "selected_source_indices": selected,
        "wrong_route_success_count": sum(
            row["wrong_route_success_count"] for row in forced_results
        ),
        "forced_canonicalizer_diagnostics": forced_diagnostics,
        "unforced_canonicalizer_diagnostics": unforced_diagnostics,
        "identity": _identity(e50c),
        "conditioned_teacher_records_sha256": _sha256(generated_path),
        "relation_records_sha256": _sha256(relation_path),
        "menu_records_sha256": _sha256(menu_path),
        "base_05b_responses_sha256": _sha256(private_05b_path),
        "forced_cases": forced_results,
        "problems": problems_out,
    }
    _write_json(result_path, payload)
    print(result_path)


if __name__ == "__main__":
    main()
