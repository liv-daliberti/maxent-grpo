#!/usr/bin/env python3
"""Conditionally generate a third answer-blind safe-pair teacher corpus."""

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
H_PATH = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e50h_second_conditioned_teacher_corpus.py"
)
SIGNATURE_PATH = (
    ROOT / "ops/math_strategy_calibration/safe_math_strategy_signatures.py"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e50i_third_conditioned_teacher_contingency_20260726.md"
)
F_ROOT = ROOT / "var/artifacts/e50f_conditioned_teacher_route_calibration_v1"
F_RESULT = F_ROOT / "result.json"
F_GENERATED = F_ROOT / "private/conditioned_teacher_records.jsonl"
H_ROOT = ROOT / "var/artifacts/e50h_second_conditioned_teacher_corpus_v1"
H_RESULT = H_ROOT / "result.json"
H_GENERATED = H_ROOT / "private/second_conditioned_records.jsonl"
SCHEMA = "e50i_third_conditioned_teacher_contingency_v1"
TRIGGER_THRESHOLD = 20
PROPOSAL_SEED = 500841
EXECUTION_SEED = 500842
EXECUTION_SAMPLES = 8


def _load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E50I helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


H = _load_module("e50i_second_base", H_PATH)
F = H.BASE
E50C = H.E50C
SIGNATURES = _load_module("e50i_safe_signatures", SIGNATURE_PATH)


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


def _method_route(method: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": str(method.get("label") or ""),
        "actions": [str(action) for action in method.get("actions") or ()],
    }


def _proposal_payload(
    model: str,
    problem: str,
    problem_order: int,
    excluded_pairs: tuple[tuple[str, str], tuple[str, str]],
) -> dict[str, Any]:
    options = "\n".join(
        f"- {left} VERSUS {right}" for left, right in F.SAFE_PAIR_OPTIONS
    )
    exclusions = "\n".join(
        f"- {left} VERSUS {right}" for left, right in excluded_pairs
    )
    user = f"""Propose exactly two genuinely different mathematical methods
for solving the problem below. Choose one and only one pair from the frozen
decisive-engine list for which BOTH methods are sound and sufficient. Do not
choose either prior attempted pair shown below. Preserve each selected engine
phrase verbatim in that method's label or actions, and do not combine the two
engines in one method.

Each method must have 2 to 6 concise, concrete, ordered, problem-specific
actions. Do not solve the problem, state or encode the final answer, or use a
hidden step. Explain the decisive difference in one short field.

PRIOR PAIRS TO EXCLUDE:
{exclusions}

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
                "name": "e50i_third_method_pair",
                "strict": True,
                "schema": F._proposal_schema(),
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


def _labels(record: dict[str, Any]) -> tuple[str, str]:
    methods = record["proposal"]["methods"]
    return str(methods[0]["label"]), str(methods[1]["label"])


def _eligible(
    *,
    record: dict[str, Any],
    problem: dict[str, Any],
    boxed_reward_fn: Any,
    prior_records: tuple[dict[str, Any], ...] = (),
) -> bool:
    if (
        record.get("error")
        or record.get("proposal_finish_reason") != "stop"
        or not isinstance(record.get("proposal"), dict)
    ):
        return False
    methods = record["proposal"].get("methods")
    if not isinstance(methods, list) or len(methods) != 2:
        return False
    pair = SIGNATURES.safe_distinct_pair(
        *[_method_route(method) for method in methods]
    )
    if not pair[0]:
        return False
    for prior in prior_records:
        prior_methods = (prior.get("proposal") or {}).get("methods")
        if not isinstance(prior_methods, list) or len(prior_methods) != 2:
            continue
        prior_pair = SIGNATURES.safe_distinct_pair(
            *[_method_route(method) for method in prior_methods]
        )
        if (
            prior_pair[0]
            and {pair[1], pair[2]} == {prior_pair[1], prior_pair[2]}
        ):
            return False
    counts = [0, 0]
    for execution in record.get("executions") or ():
        text = str(execution.get("response") or "")
        route_index = int(execution.get("route_index", -1))
        if (
            route_index not in (0, 1)
            or execution.get("finish_reason") != "stop"
            or len(text) > 4000
            or _sha256_text(text)
            != str(execution.get("response_sha256") or "")
        ):
            continue
        _, reward = boxed_reward_fn(
            text, str(problem["answer"]), fast=False
        )
        counts[route_index] += float(reward) > 0.0
    return min(counts) >= 2


def _generate_problem(
    *,
    endpoint: str,
    model: str,
    problem: dict[str, Any],
    first_record: dict[str, Any],
    second_record: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    order = int(problem["problem_order"])
    base = {
        "problem_order": order,
        "source_index": int(problem["source_index"]),
        "row_id": str(problem["unique_id"]),
        "attempt_index": 2,
    }
    try:
        excluded_pairs = (_labels(first_record), _labels(second_record))
        proposal_payload = _proposal_payload(
            model, str(problem["problem"]), order, excluded_pairs
        )
        proposal_response = F._post(
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
            raise RuntimeError("E50I method proposal contract failed")
        normalized_observed = {
            " ".join(str(method["label"]).lower().split())
            for method in methods
        }
        repeated_excluded_label_pair = any(
            normalized_observed
            == {" ".join(label.lower().split()) for label in pair}
            for pair in excluded_pairs
        )
        executions = []
        for route_index, method in enumerate(methods):
            payload = _execution_payload(
                model=model,
                problem=str(problem["problem"]),
                method=method,
                problem_order=order,
                route_index=route_index,
            )
            response = F._post(endpoint, payload, timeout=timeout)
            choices = response.get("choices") or []
            if len(choices) != EXECUTION_SAMPLES:
                raise RuntimeError("E50I execution sample mismatch")
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
            "excluded_first_two_pair_labels": [
                list(pair) for pair in excluded_pairs
            ],
            "repeated_excluded_label_pair": repeated_excluded_label_pair,
            "answer_was_not_in_proposal_payload": True,
            "answer_was_not_in_execution_payloads": True,
            "executions": executions,
        }
    except Exception as exc:
        return {**base, "error": str(exc), "executions": []}


def _identity(
    f_result: dict[str, Any],
    h_result: dict[str, Any],
    prethird_eligible: list[int],
) -> dict[str, Any]:
    return {
        "protocol_sha256": _sha256(PROTOCOL),
        "script_sha256": _sha256(SCRIPT),
        "e50h_script_sha256": _sha256(H_PATH),
        "signature_source_sha256": _sha256(SIGNATURE_PATH),
        "e50f_result_sha256": _sha256(F_RESULT),
        "e50f_result_schema": f_result["schema"],
        "e50f_generated_sha256": _sha256(F_GENERATED),
        "e50h_result_sha256": _sha256(H_RESULT),
        "e50h_result_schema": h_result["schema"],
        "e50h_generated_sha256": _sha256(H_GENERATED),
        "e47_manifest_sha256": _sha256(E50C.E47_MANIFEST),
        "e47_problems_sha256": _sha256(E50C.E47_PROBLEMS),
        "endpoint_record_sha256": _sha256(E50C.ENDPOINT_RECORD),
        "prethird_eligible_problem_orders": prethird_eligible,
        "trigger_threshold": TRIGGER_THRESHOLD,
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
        raise RuntimeError(f"fresh E50I result required: {result_path}")

    f_result = json.loads(F_RESULT.read_text(encoding="utf-8"))
    h_result = json.loads(H_RESULT.read_text(encoding="utf-8"))
    if (
        f_result.get("schema")
        != "e50f_conditioned_teacher_route_calibration_v1"
        or f_result.get("pass") is not False
        or f_result.get("generation_problem_count") != 50
        or f_result.get("generation_error_count") != 0
        or f_result.get("conditioned_execution_count") != 800
        or f_result.get("conditioned_teacher_records_sha256")
        != _sha256(F_GENERATED)
        or h_result.get("schema")
        != "e50h_second_conditioned_teacher_corpus_v1"
        or h_result.get("pass") is not False
        or h_result.get("failure")
        != "corpus_only_second_attempt_no_training_authority"
        or h_result.get("generation_problem_count") != 50
        or h_result.get("generation_error_count") != 0
        or h_result.get("conditioned_execution_count") != 800
        or h_result.get("selected_source_indices") != []
        or h_result.get("second_conditioned_records_sha256")
        != _sha256(H_GENERATED)
        or h_result.get("identity", {}).get("script_sha256")
        != _sha256(H_PATH)
        or h_result.get("identity", {}).get("e50f_result_sha256")
        != _sha256(F_RESULT)
    ):
        raise RuntimeError("E50I requires complete frozen E50F/E50H corpora")

    from oat_drgrpo.math_grader import boxed_reward_fn

    problems = E50C._load_jsonl(E50C.E47_PROBLEMS)
    first_rows = _read_jsonl(F_GENERATED)
    second_rows = _read_jsonl(H_GENERATED)
    if (
        len(problems) != 50
        or [int(row["problem_order"]) for row in first_rows]
        != list(range(50))
        or [int(row["problem_order"]) for row in second_rows]
        != list(range(50))
    ):
        raise RuntimeError("E50I frozen corpus order drifted")
    for order, problem in enumerate(problems):
        problem["problem_order"] = order
    prethird_eligible = []
    for order, problem in enumerate(problems):
        if _eligible(
            record=first_rows[order],
            problem=problem,
            boxed_reward_fn=boxed_reward_fn,
        ) or _eligible(
            record=second_rows[order],
            problem=problem,
            boxed_reward_fn=boxed_reward_fn,
            prior_records=(first_rows[order],),
        ):
            prethird_eligible.append(order)
    triggered = len(prethird_eligible) < TRIGGER_THRESHOLD
    attempted_orders = (
        [
            order
            for order in range(50)
            if order not in set(prethird_eligible)
        ]
        if triggered
        else []
    )
    identity = _identity(f_result, h_result, prethird_eligible)
    checkpoint_path = output / "private/checkpoint_identity.json"
    if checkpoint_path.is_file():
        if json.loads(checkpoint_path.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("E50I partial checkpoint identity drifted")
    else:
        _write_json(checkpoint_path, identity)
    partial_path = output / "private/third_conditioned_records.partial.jsonl"
    generated = _read_jsonl(partial_path) if partial_path.is_file() else []
    completed = [int(row["problem_order"]) for row in generated]
    if (
        len(completed) != len(set(completed))
        or not set(completed) <= set(attempted_orders)
    ):
        raise RuntimeError("E50I partial checkpoint is invalid")
    endpoint, model = E50C._endpoint(E50C.ENDPOINT_RECORD)
    remaining = [
        order
        for order in attempted_orders
        if order not in set(completed)
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                _generate_problem,
                endpoint=endpoint,
                model=model,
                problem=problems[order],
                first_record=first_rows[order],
                second_record=second_rows[order],
                timeout=args.timeout,
            )
            for order in remaining
        ]
        for future in as_completed(futures):
            record = future.result()
            generated.append(record)
            generated.sort(key=lambda row: int(row["problem_order"]))
            _write_jsonl(partial_path, generated)
            print(
                json.dumps(
                    {
                        "phase": "third_conditioned_teacher",
                        "problem_order": record["problem_order"],
                        "execution_count": len(record["executions"]),
                        "error": record.get("error"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if [int(row["problem_order"]) for row in generated] != attempted_orders:
        raise RuntimeError("E50I third corpus is incomplete")
    generated_path = output / "private/third_conditioned_records.jsonl"
    _write_jsonl(generated_path, generated)
    result = {
        "schema": SCHEMA,
        "pass": False,
        "failure": "corpus_only_third_attempt_no_training_authority",
        "triggered": triggered,
        "trigger_threshold": TRIGGER_THRESHOLD,
        "prethird_eligible_problem_count": len(prethird_eligible),
        "prethird_eligible_problem_orders": prethird_eligible,
        "attempted_problem_orders": attempted_orders,
        "generation_problem_count": len(generated),
        "generation_error_count": sum(
            bool(row.get("error")) for row in generated
        ),
        "conditioned_execution_count": sum(
            len(row.get("executions") or ()) for row in generated
        ),
        "selected_source_indices": [],
        "identity": _identity(f_result, h_result, prethird_eligible),
        "third_conditioned_records_sha256": _sha256(generated_path),
    }
    _write_json(result_path, result)
    print(result_path)


if __name__ == "__main__":
    main()
