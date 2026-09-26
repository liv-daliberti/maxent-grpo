#!/usr/bin/env python3
"""Generate a second answer-blind concise safe-pair corpus for E50G."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = pathlib.Path(__file__).resolve()
BASE_PATH = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e50f_conditioned_teacher_route_calibration.py"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e50h_second_conditioned_teacher_corpus_20260726.md"
)
E50F_ROOT = (
    ROOT
    / "var/artifacts/e50f_conditioned_teacher_route_calibration_v1"
)
E50F_RESULT = E50F_ROOT / "result.json"
E50F_GENERATED = (
    E50F_ROOT / "private/conditioned_teacher_records.jsonl"
)
SCHEMA = "e50h_second_conditioned_teacher_corpus_v1"
PROPOSAL_SEED = 500831
EXECUTION_SEED = 500832
EXECUTION_SAMPLES = 8


def _load_base() -> Any:
    spec = importlib.util.spec_from_file_location("e50h_e50f_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E50F base: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
E50C = BASE.BASE


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
                    json.dumps(row, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _proposal_payload(
    model: str,
    problem: str,
    problem_order: int,
    excluded_labels: tuple[str, str],
) -> dict[str, Any]:
    options = "\n".join(
        f"- {left} VERSUS {right}"
        for left, right in BASE.SAFE_PAIR_OPTIONS
    )
    excluded = " VERSUS ".join(excluded_labels)
    user = f"""Propose exactly two genuinely different mathematical methods
for solving the problem below. Choose one and only one pair from the frozen
decisive-engine list for which BOTH methods are sound and sufficient. Do not
choose the prior attempted pair shown below. Preserve each selected engine
phrase verbatim in that method's label or actions, and do not combine the two
engines in one method.

Each method must have 2 to 6 concise, concrete, ordered, problem-specific
actions. Do not solve the problem, state or encode the final answer, or use a
hidden step. Explain the decisive difference in one short field.

PRIOR PAIR TO EXCLUDE:
{excluded}

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
                "name": "e50h_second_method_pair",
                "strict": True,
                "schema": BASE._proposal_schema(),
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
    user = f"""Solve the problem by literally following every prescribed
action in order. Use no substitute method or hidden decisive step. Give a
complete rigorous derivation in at most 450 words and finish with a boxed
final answer.

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
                "content": "You are a concise rigorous mathematical solver.",
            },
            {"role": "user", "content": user},
        ],
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 1024,
        "n": EXECUTION_SAMPLES,
        "seed": EXECUTION_SEED + problem_order * 2 + route_index,
        "stream": False,
    }


def _generate_problem(
    *,
    endpoint: str,
    model: str,
    problem: dict[str, Any],
    first_record: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    order = int(problem["problem_order"])
    base = {
        "problem_order": order,
        "source_index": int(problem["source_index"]),
        "row_id": str(problem["unique_id"]),
        "attempt_index": 1,
    }
    try:
        first_methods = first_record["proposal"]["methods"]
        excluded_labels = (
            str(first_methods[0]["label"]),
            str(first_methods[1]["label"]),
        )
        proposal_payload = _proposal_payload(
            model, str(problem["problem"]), order, excluded_labels
        )
        proposal_response = BASE._post(
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
            raise RuntimeError("E50H method proposal contract failed")
        repeated_excluded_label_pair = {
            " ".join(str(method["label"]).lower().split())
            for method in methods
        } == {
            " ".join(label.lower().split())
            for label in excluded_labels
        }

        executions = []
        for route_index, method in enumerate(methods):
            payload = _execution_payload(
                model=model,
                problem=str(problem["problem"]),
                method=method,
                problem_order=order,
                route_index=route_index,
            )
            response = BASE._post(endpoint, payload, timeout=timeout)
            choices = response.get("choices") or []
            if len(choices) != EXECUTION_SAMPLES:
                raise RuntimeError("E50H execution sample mismatch")
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
            **base,
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
            "excluded_first_pair_labels": list(excluded_labels),
            "repeated_excluded_label_pair": repeated_excluded_label_pair,
            "answer_was_not_in_proposal_payload": True,
            "answer_was_not_in_execution_payloads": True,
            "executions": executions,
        }
    except Exception as exc:
        return {**base, "error": str(exc), "executions": []}


def _identity(e50f: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol_sha256": _sha256(PROTOCOL),
        "script_sha256": _sha256(SCRIPT),
        "base_e50f_script_sha256": _sha256(BASE_PATH),
        "e50f_result_sha256": _sha256(E50F_RESULT),
        "e50f_result_schema": e50f["schema"],
        "e50f_generated_sha256": _sha256(E50F_GENERATED),
        "e47_manifest_sha256": _sha256(E50C.E47_MANIFEST),
        "e47_problems_sha256": _sha256(E50C.E47_PROBLEMS),
        "endpoint_record_sha256": _sha256(E50C.ENDPOINT_RECORD),
        "proposal_seed": PROPOSAL_SEED,
        "execution_seed": EXECUTION_SEED,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    output = args.output_root.resolve()
    result_path = output / "result.json"
    if result_path.exists():
        raise RuntimeError(f"fresh E50H result required: {result_path}")

    e50f = json.loads(E50F_RESULT.read_text(encoding="utf-8"))
    if (
        e50f.get("schema")
        != "e50f_conditioned_teacher_route_calibration_v1"
        or e50f.get("pass") is not False
        or e50f.get("failure")
        != (
            "open_relation_judge_quarantined_after_frozen_"
            "conditioned_corpus_generation"
        )
        or e50f.get("generation_problem_count") != 50
        or e50f.get("generation_error_count") != 0
        or e50f.get("conditioned_execution_count") != 800
        or e50f.get("selected_source_indices") != []
        or e50f.get("conditioned_teacher_records_sha256")
        != _sha256(E50F_GENERATED)
        or e50f.get("identity", {}).get("script_sha256")
        != _sha256(BASE_PATH)
        or e50f.get("identity", {}).get("protocol_sha256")
        != _sha256(BASE.PROTOCOL)
        or e50f.get("identity", {}).get("e50c_result_sha256")
        != _sha256(BASE.E50C_RESULT)
    ):
        raise RuntimeError("E50H requires the complete frozen E50F corpus")
    problems = E50C._load_jsonl(E50C.E47_PROBLEMS)
    first_records = _read_jsonl(E50F_GENERATED)
    if (
        len(problems) != 50
        or len(first_records) != 50
        or [int(row["problem_order"]) for row in first_records]
        != list(range(50))
    ):
        raise RuntimeError("E50H frozen cohort or first corpus drifted")
    for order, problem in enumerate(problems):
        problem["problem_order"] = order

    endpoint, model = E50C._endpoint(E50C.ENDPOINT_RECORD)
    partial_path = output / "private/second_conditioned_records.partial.jsonl"
    checkpoint_path = output / "private/checkpoint_identity.json"
    identity = _identity(e50f)
    if checkpoint_path.is_file():
        if json.loads(checkpoint_path.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("E50H partial checkpoint identity drifted")
    else:
        _write_json(checkpoint_path, identity)
    generated = _read_jsonl(partial_path) if partial_path.is_file() else []
    completed = [int(row["problem_order"]) for row in generated]
    if len(completed) != len(set(completed)) or not set(completed) <= set(
        range(50)
    ):
        raise RuntimeError("E50H partial checkpoint is invalid")
    remaining = [
        problem
        for problem in problems
        if int(problem["problem_order"]) not in set(completed)
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                _generate_problem,
                endpoint=endpoint,
                model=model,
                problem=problem,
                first_record=first_records[int(problem["problem_order"])],
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
                        "phase": "second_conditioned_teacher",
                        "problem_order": record["problem_order"],
                        "execution_count": len(record["executions"]),
                        "error": record.get("error"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if (
        len(generated) != 50
        or [int(row["problem_order"]) for row in generated]
        != list(range(50))
    ):
        raise RuntimeError("E50H second corpus is incomplete")
    generated_path = output / "private/second_conditioned_records.jsonl"
    _write_jsonl(generated_path, generated)
    result = {
        "schema": SCHEMA,
        "pass": False,
        "failure": "corpus_only_second_attempt_no_training_authority",
        "generation_problem_count": 50,
        "generation_error_count": sum(
            bool(row.get("error")) for row in generated
        ),
        "conditioned_execution_count": sum(
            len(row.get("executions") or ()) for row in generated
        ),
        "selected_source_indices": [],
        "identity": _identity(e50f),
        "second_conditioned_records_sha256": _sha256(generated_path),
    }
    _write_json(result_path, result)
    print(result_path)


if __name__ == "__main__":
    main()
