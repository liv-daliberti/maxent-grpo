#!/usr/bin/env python3
"""Conditionally generate the final answer-blind teacher corpus."""

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
J_PATH = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e50j_fourth_conditioned_teacher_contingency.py"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e50k_fifth_conditioned_teacher_contingency_20260726.md"
)
J_ROOT = ROOT / "var/artifacts/e50j_fourth_conditioned_teacher_contingency_v1"
J_RESULT = J_ROOT / "result.json"
J_GENERATED = J_ROOT / "private/fourth_conditioned_records.jsonl"
SCHEMA = "e50k_fifth_conditioned_teacher_contingency_v1"
TRIGGER_THRESHOLD = 30
PROPOSAL_SEED = 500861
EXECUTION_SEED = 500862


def _load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E50K helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


J = _load_module("e50k_fourth_base", J_PATH)
I = J.I
H = J.H
F = J.F
E50C = J.E50C
F_RESULT = J.F_RESULT
F_GENERATED = J.F_GENERATED
H_RESULT = J.H_RESULT
H_GENERATED = J.H_GENERATED
I_RESULT = J.I_RESULT
I_GENERATED = J.I_GENERATED


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
    excluded_pairs: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    payload = J._proposal_payload(
        model, problem, problem_order, excluded_pairs
    )
    payload["seed"] = PROPOSAL_SEED + problem_order
    payload["response_format"]["json_schema"][
        "name"
    ] = "e50k_fifth_stage_method_pair"
    return payload


def _execution_payload(**kwargs: Any) -> dict[str, Any]:
    payload = J._execution_payload(**kwargs)
    payload["seed"] = (
        EXECUTION_SEED
        + int(kwargs["problem_order"]) * 2
        + int(kwargs["route_index"])
    )
    return payload


def _identity(
    f_result: dict[str, Any],
    h_result: dict[str, Any],
    i_result: dict[str, Any],
    j_result: dict[str, Any],
    prefifth_eligible: list[int],
) -> dict[str, Any]:
    return {
        "protocol_sha256": _sha256(PROTOCOL),
        "script_sha256": _sha256(SCRIPT),
        "e50j_script_sha256": _sha256(J_PATH),
        "signature_source_sha256": _sha256(I.SIGNATURE_PATH),
        "e50f_result_sha256": _sha256(F_RESULT),
        "e50f_result_schema": f_result["schema"],
        "e50f_generated_sha256": _sha256(F_GENERATED),
        "e50h_result_sha256": _sha256(H_RESULT),
        "e50h_result_schema": h_result["schema"],
        "e50h_generated_sha256": _sha256(H_GENERATED),
        "e50i_result_sha256": _sha256(I_RESULT),
        "e50i_result_schema": i_result["schema"],
        "e50i_generated_sha256": _sha256(I_GENERATED),
        "e50j_result_sha256": _sha256(J_RESULT),
        "e50j_result_schema": j_result["schema"],
        "e50j_generated_sha256": _sha256(J_GENERATED),
        "e47_manifest_sha256": _sha256(E50C.E47_MANIFEST),
        "e47_problems_sha256": _sha256(E50C.E47_PROBLEMS),
        "endpoint_record_sha256": _sha256(E50C.ENDPOINT_RECORD),
        "prefifth_eligible_problem_orders": prefifth_eligible,
        "trigger_threshold": TRIGGER_THRESHOLD,
        "proposal_seed": PROPOSAL_SEED,
        "execution_seed": EXECUTION_SEED,
    }


def _valid_upstream(
    f_result: dict[str, Any],
    h_result: dict[str, Any],
    i_result: dict[str, Any],
    j_result: dict[str, Any],
) -> bool:
    attempted_j = j_result.get("attempted_problem_orders")
    return bool(
        J._valid_upstream(f_result, h_result, i_result)
        and j_result.get("schema")
        == "e50j_fourth_conditioned_teacher_contingency_v1"
        and j_result.get("pass") is False
        and j_result.get("failure")
        == "corpus_only_fourth_attempt_no_training_authority"
        and isinstance(attempted_j, list)
        and j_result.get("generation_problem_count") == len(attempted_j)
        and j_result.get("generation_error_count") == 0
        and j_result.get("conditioned_execution_count")
        == len(attempted_j) * 16
        and j_result.get("selected_source_indices") == []
        and j_result.get("fourth_conditioned_records_sha256")
        == _sha256(J_GENERATED)
        and j_result.get("identity", {}).get("script_sha256")
        == _sha256(J_PATH)
        and j_result.get("identity", {}).get("e50i_result_sha256")
        == _sha256(I_RESULT)
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
        raise RuntimeError(f"fresh E50K result required: {result_path}")

    f_result = json.loads(F_RESULT.read_text(encoding="utf-8"))
    h_result = json.loads(H_RESULT.read_text(encoding="utf-8"))
    i_result = json.loads(I_RESULT.read_text(encoding="utf-8"))
    j_result = json.loads(J_RESULT.read_text(encoding="utf-8"))
    if not _valid_upstream(f_result, h_result, i_result, j_result):
        raise RuntimeError(
            "E50K requires complete frozen E50F/E50H/E50I/E50J"
        )

    from oat_drgrpo.math_grader import boxed_reward_fn

    problems = E50C._load_jsonl(E50C.E47_PROBLEMS)
    first_rows = _read_jsonl(F_GENERATED)
    second_rows = _read_jsonl(H_GENERATED)
    third_rows = _read_jsonl(I_GENERATED)
    fourth_rows = _read_jsonl(J_GENERATED)
    third_by_order = {
        int(record["problem_order"]): record for record in third_rows
    }
    fourth_by_order = {
        int(record["problem_order"]): record for record in fourth_rows
    }
    if (
        len(problems) != 50
        or [int(row["problem_order"]) for row in first_rows]
        != list(range(50))
        or [int(row["problem_order"]) for row in second_rows]
        != list(range(50))
        or sorted(third_by_order)
        != [int(order) for order in i_result["attempted_problem_orders"]]
        or sorted(fourth_by_order)
        != [int(order) for order in j_result["attempted_problem_orders"]]
    ):
        raise RuntimeError("E50K frozen corpus order drifted")
    for order, problem in enumerate(problems):
        problem["problem_order"] = order

    prefifth_eligible = []
    for order, problem in enumerate(problems):
        prior = (first_rows[order],)
        first_ok = I._eligible(
            record=first_rows[order],
            problem=problem,
            boxed_reward_fn=boxed_reward_fn,
        )
        second_ok = I._eligible(
            record=second_rows[order],
            problem=problem,
            boxed_reward_fn=boxed_reward_fn,
            prior_records=prior,
        )
        prior = prior + (second_rows[order],)
        third_ok = (
            order in third_by_order
            and I._eligible(
                record=third_by_order[order],
                problem=problem,
                boxed_reward_fn=boxed_reward_fn,
                prior_records=prior,
            )
        )
        if order in third_by_order:
            prior = prior + (third_by_order[order],)
        fourth_ok = (
            order in fourth_by_order
            and I._eligible(
                record=fourth_by_order[order],
                problem=problem,
                boxed_reward_fn=boxed_reward_fn,
                prior_records=prior,
            )
        )
        if first_ok or second_ok or third_ok or fourth_ok:
            prefifth_eligible.append(order)
    triggered = len(prefifth_eligible) < TRIGGER_THRESHOLD
    attempted_orders = (
        [
            order
            for order in range(50)
            if order not in set(prefifth_eligible)
        ]
        if triggered
        else []
    )
    identity = _identity(
        f_result, h_result, i_result, j_result, prefifth_eligible
    )
    checkpoint_path = output / "private/checkpoint_identity.json"
    if checkpoint_path.is_file():
        if json.loads(checkpoint_path.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("E50K partial checkpoint identity drifted")
    else:
        _write_json(checkpoint_path, identity)
    partial_path = output / "private/fifth_conditioned_records.partial.jsonl"
    generated = _read_jsonl(partial_path) if partial_path.is_file() else []
    completed = [int(row["problem_order"]) for row in generated]
    if (
        len(completed) != len(set(completed))
        or not set(completed) <= set(attempted_orders)
    ):
        raise RuntimeError("E50K partial checkpoint is invalid")
    endpoint, model = E50C._endpoint(E50C.ENDPOINT_RECORD)
    remaining = [
        order for order in attempted_orders if order not in set(completed)
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for order in remaining:
            prior_records = [first_rows[order], second_rows[order]]
            if order in third_by_order:
                prior_records.append(third_by_order[order])
            if order in fourth_by_order:
                prior_records.append(fourth_by_order[order])
            futures.append(
                pool.submit(
                    J._generate_problem,
                    endpoint=endpoint,
                    model=model,
                    problem=problems[order],
                    prior_records=tuple(prior_records),
                    timeout=args.timeout,
                    attempt_index=4,
                    proposal_payload_fn=_proposal_payload,
                    execution_payload_fn=_execution_payload,
                )
            )
        for future in as_completed(futures):
            record = future.result()
            generated.append(record)
            generated.sort(key=lambda row: int(row["problem_order"]))
            _write_jsonl(partial_path, generated)
            print(
                json.dumps(
                    {
                        "phase": "fifth_conditioned_teacher",
                        "problem_order": record["problem_order"],
                        "execution_count": len(record["executions"]),
                        "error": record.get("error"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if [int(row["problem_order"]) for row in generated] != attempted_orders:
        raise RuntimeError("E50K fifth corpus is incomplete")
    generated_path = output / "private/fifth_conditioned_records.jsonl"
    _write_jsonl(generated_path, generated)
    result = {
        "schema": SCHEMA,
        "pass": False,
        "failure": "corpus_only_fifth_attempt_no_training_authority",
        "triggered": triggered,
        "trigger_threshold": TRIGGER_THRESHOLD,
        "prefifth_eligible_problem_count": len(prefifth_eligible),
        "prefifth_eligible_problem_orders": prefifth_eligible,
        "attempted_problem_orders": attempted_orders,
        "generation_problem_count": len(generated),
        "generation_error_count": sum(
            bool(row.get("error")) for row in generated
        ),
        "conditioned_execution_count": sum(
            len(row.get("executions") or ()) for row in generated
        ),
        "selected_source_indices": [],
        "identity": _identity(
            f_result, h_result, i_result, j_result, prefifth_eligible
        ),
        "fifth_conditioned_records_sha256": _sha256(generated_path),
    }
    _write_json(result_path, result)
    print(result_path)


if __name__ == "__main__":
    main()
