#!/usr/bin/env python3
"""Calibrate 72B-discovered routes for natural execution by Qwen2.5-0.5B."""

from __future__ import annotations

import argparse
import hashlib
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
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e50c_72b_teacher_route_calibration_20260726.md"
)
E47_ROOT = ROOT / "var/artifacts/e47_math_strategy_calibration_v1"
E47_MANIFEST = E47_ROOT / "manifest.json"
E47_PROBLEMS = E47_ROOT / "problems.jsonl"
E49AA_RESULT = (
    ROOT
    / "var/artifacts/"
    "e49aa_calibrated_pairwise_observed_route_discovery_v1/result.json"
)
E49AB_RESULT = (
    ROOT
    / "var/artifacts/"
    "e49ab_all_observed_persistent_pairwise_route_discovery_v1/result.json"
)
E50A_RESULT = (
    ROOT
    / "var/artifacts/"
    "e50a_consensus_observed_route_calibration_v1/result.json"
)
E49AC_RESULT = (
    ROOT
    / "var/artifacts/"
    "e49ac_confirmed_singleton_observed_route_discovery_v1/result.json"
)
E49AA_HELPER = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e49aa_calibrated_pairwise_observed_route_discovery.py"
)
E49Y_HELPER = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e49y_observed_route_discovery.py"
)
E49W_HELPER = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e49w_bottom_up_route_calibration.py"
)
MODEL = (
    ROOT
    / "var/cache/huggingface/transformers/"
    "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
    "7ae557604adf67be50417f59c2c2f167def9a775"
)
TEACHER_SAMPLES = 16
TEACHER_BATCH = 8
FORCED_SAMPLES = 16
UNFORCED_SAMPLES = 64
SEED = 500721

sys.path.insert(0, str(ROOT / "ops/math_strategy_calibration"))
from run_e49aa_calibrated_pairwise_observed_route_discovery import (  # noqa: E402
    ENDPOINT_RECORD,
    E49T_IDENTITY,
    PAIRWISE_SOURCE,
    _cluster_candidate,
    _endpoint,
    _forced_problem,
    _load_pairwise_module,
    _menu_candidate,
    _parse_menu,
)


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


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _verify_activation() -> None:
    for path, schema in (
        (
            E49AA_RESULT,
            "e49aa_calibrated_pairwise_observed_route_discovery_v1",
        ),
        (
            E49AB_RESULT,
            "e49ab_all_observed_persistent_pairwise_route_discovery_v1",
        ),
        (E50A_RESULT, "e50a_consensus_observed_route_calibration_v1"),
        (
            E49AC_RESULT,
            "e49ac_confirmed_singleton_observed_route_discovery_v1",
        ),
    ):
        if not path.is_file():
            raise RuntimeError(f"E50C requires terminal predecessor: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("schema") != schema:
            raise RuntimeError(f"E50C predecessor schema mismatch: {path}")
        if (
            result.get("pass") is True
            and len(result.get("selected_source_indices") or []) == 10
        ):
            raise RuntimeError(
                "E50C is inactive because an observed-route source passed"
            )


def _identity() -> dict[str, Any]:
    return {
        "protocol_sha256": _sha256(PROTOCOL),
        "script_sha256": _sha256(SCRIPT),
        "e47_manifest_sha256": _sha256(E47_MANIFEST),
        "e47_problems_sha256": _sha256(E47_PROBLEMS),
        "endpoint_record_sha256": _sha256(ENDPOINT_RECORD),
        "pairwise_source_sha256": _sha256(PAIRWISE_SOURCE),
        "e49t_identity_sha256": _sha256(E49T_IDENTITY),
        "e49aa_helper_sha256": _sha256(E49AA_HELPER),
        "e49y_helper_sha256": _sha256(E49Y_HELPER),
        "e49w_helper_sha256": _sha256(E49W_HELPER),
        "e49aa_result_sha256": _sha256(E49AA_RESULT),
        "e49ab_result_sha256": _sha256(E49AB_RESULT),
        "e50a_result_sha256": _sha256(E50A_RESULT),
        "e49ac_result_sha256": _sha256(E49AC_RESULT),
        "seed": SEED,
    }


def _teacher_samples(
    *,
    endpoint: str,
    model: str,
    problem: dict[str, Any],
    timeout: int,
) -> list[dict[str, Any]]:
    prompt = (
        "Solve the following mathematics problem rigorously. Give a complete "
        "derivation and finish with a boxed final answer. Do not discuss this "
        "instruction or omit decisive mathematical steps.\n\n"
        + str(problem["problem"])
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    rows = []
    for batch_index in range(TEACHER_SAMPLES // TEACHER_BATCH):
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a rigorous mathematical problem solver.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 1024,
            "n": TEACHER_BATCH,
            "seed": SEED + 1000 * int(problem["problem_order"]) + batch_index,
            "stream": False,
        }
        request = urllib.request.Request(
            f"{endpoint.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with opener.open(request, timeout=timeout) as response:
                    decoded = json.loads(response.read().decode("utf-8"))
                choices = decoded.get("choices") or []
                if len(choices) != TEACHER_BATCH:
                    raise RuntimeError(
                        f"teacher returned {len(choices)} choices"
                    )
                for choice_index, choice in enumerate(choices):
                    text = str(
                        (choice.get("message") or {}).get("content") or ""
                    )
                    rows.append(
                        {
                            "problem_order": int(problem["problem_order"]),
                            "problem_id": str(problem["problem_id"]),
                            "source_index": int(problem["source_index"]),
                            "batch_index": batch_index,
                            "choice_index": choice_index,
                            "seed": payload["seed"],
                            "finish_reason": choice.get("finish_reason"),
                            "response_sha256": _sha256_text(text),
                            "response": text,
                        }
                    )
                break
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
                RuntimeError,
            ) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2**attempt)
        else:
            raise RuntimeError(
                f"teacher request failed after retries: {last_error}"
            )
    if len(rows) != TEACHER_SAMPLES:
        raise RuntimeError("teacher sample count mismatch")
    return rows


def _neutral_problem(problem: str, menu: Any) -> str:
    from oat_drgrpo.math_strategy_menu import (
        MENU_END,
        MENU_START,
        strategy_menu_natural_response_instructions,
    )

    return (
        problem.rstrip()
        + f"\n\n{MENU_START}\n"
        + menu.canonical_json
        + f"\n{MENU_END}"
        + "\n\nStrategy IDs and listing order are labels only; neither "
        "strategy is preferred. Choose independently on each attempt."
        + strategy_menu_natural_response_instructions(menu)
    )


def _load_frozen_canonicalizer() -> Any:
    identity = json.loads(E49T_IDENTITY.read_text(encoding="utf-8"))
    frozen_source = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e49t_natural_menu_{identity['source_hash']}"
        / "src"
    )
    import oat_drgrpo

    oat_drgrpo.__path__.insert(0, str(frozen_source / "oat_drgrpo"))
    from oat_drgrpo.math_strategy_canonicalizer import (
        MathStrategyCanonicalizer,
    )

    return MathStrategyCanonicalizer


def _canonical_keys(
    *,
    canonicalizer: Any,
    endpoint: str,
    model: str,
    prompt_tokens: list[list[int]],
    prompt_texts: list[str],
    responses: list[str],
    positives: list[bool],
    samples_per_prompt: int,
    timeout: int,
    workers: int,
) -> tuple[list[str | None], dict[str, float]]:
    judge = canonicalizer(
        endpoint=endpoint,
        model=model,
        timeout_seconds=timeout,
        max_workers=workers,
        permutation_seeds=(470721, 470722),
        max_item_chars=4000,
        missing_ids_are_ambiguous=True,
        allow_unstructured_menu_inference=True,
    )
    keys, diagnostics = judge.canonicalize(
        prompt_token_ids=prompt_tokens,
        prompt_texts=prompt_texts,
        response_texts=responses,
        task_reward_positive=positives,
        active_mask=[True] * len(responses),
        num_samples=samples_per_prompt,
    )
    return keys, {
        field: float(getattr(diagnostics, field))
        for field in diagnostics.__dataclass_fields__
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
        raise RuntimeError(f"fresh E50C result required: {result_path}")
    _verify_activation()

    manifest = json.loads(E47_MANIFEST.read_text(encoding="utf-8"))
    problems = _load_jsonl(E47_PROBLEMS)
    expected_indices = manifest["selection"]["source_indices"]
    if (
        len(problems) != 50
        or [int(row["source_index"]) for row in problems] != expected_indices
        or any(int(row["level"]) != 5 for row in problems)
    ):
        raise RuntimeError("E50C frozen E47 problem cohort drifted")
    for problem_order, problem in enumerate(problems):
        problem["problem_order"] = problem_order

    from oat_drgrpo.math_grader import boxed_reward_fn
    from oat_drgrpo.templates import apply_qwen_math_template

    endpoint, judge_model = _endpoint(ENDPOINT_RECORD)
    teacher_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _teacher_samples,
                endpoint=endpoint,
                model=judge_model,
                problem=problem,
                timeout=args.timeout,
            ): int(problem["problem_order"])
            for problem in problems
        }
        for future in as_completed(futures):
            rows = future.result()
            teacher_rows.extend(rows)
            print(
                json.dumps(
                    {
                        "phase": "teacher",
                        "problem_order": rows[0]["problem_order"],
                        "sample_count": len(rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    teacher_rows.sort(
        key=lambda row: (
            int(row["problem_order"]),
            int(row["batch_index"]),
            int(row["choice_index"]),
        )
    )
    problem_by_order = {
        int(problem["problem_order"]): problem for problem in problems
    }
    validated_rows = []
    for row in teacher_rows:
        problem = problem_by_order[int(row["problem_order"])]
        _, reward = boxed_reward_fn(
            str(row["response"]), str(problem["answer"]), fast=False
        )
        validated_rows.append(
            {**row, "answer_positive": float(reward) > 0.0}
        )
    teacher_path = output / "private/teacher_responses.jsonl"
    _write_jsonl(teacher_path, validated_rows)

    by_order: dict[int, list[dict[str, Any]]] = {}
    for row in validated_rows:
        if row["answer_positive"]:
            by_order.setdefault(int(row["problem_order"]), []).append(row)
    candidates = []
    for problem in problems:
        positives = by_order.get(int(problem["problem_order"]), [])
        if len(positives) < 4:
            continue
        candidates.append(
            {
                "problem_order": int(problem["problem_order"]),
                "source_index": int(problem["source_index"]),
                "row_id": str(problem["unique_id"]),
                "problem": str(problem["problem"]),
                "answer": str(problem["answer"]),
                "subject": str(problem["subject"]),
                "level": int(problem["level"]),
                "validator_positive_count": len(positives),
                "exemplars": [
                    {
                        "sample_id": str(row["response_sha256"])[:20],
                        "sample_index": (
                            int(row["batch_index"]) * TEACHER_BATCH
                            + int(row["choice_index"])
                        ),
                        "text": str(row["response"]),
                        "response_sha256": str(row["response_sha256"]),
                    }
                    for row in positives
                ],
            }
        )
    print(
        json.dumps(
            {
                "phase": "validation",
                "teacher_response_count": len(validated_rows),
                "validator_positive_count": sum(
                    row["answer_positive"] for row in validated_rows
                ),
                "candidate_count": len(candidates),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    pairwise_module = _load_pairwise_module()
    clusters = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                _cluster_candidate,
                pairwise_module,
                endpoint,
                judge_model,
                candidate,
                args.timeout,
            )
            for candidate in candidates
        ]
        for future in as_completed(futures):
            record = future.result()
            clusters.append(record)
            print(
                json.dumps(
                    {
                        "phase": "cluster",
                        "source_index": record["source_index"],
                        "eligible": record["cluster_eligible"],
                        "error": record.get("error"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    clusters.sort(key=lambda row: int(row["source_index"]))
    cluster_path = output / "cluster_records.jsonl"
    _write_jsonl(cluster_path, clusters)
    candidate_by_source = {
        int(row["source_index"]): row for row in candidates
    }
    cluster_eligible = [row for row in clusters if row["cluster_eligible"]]

    menus = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                _menu_candidate,
                endpoint=endpoint,
                model=judge_model,
                candidate=candidate_by_source[int(record["source_index"])],
                cluster_record=record,
                timeout=args.timeout,
            )
            for record in cluster_eligible
        ]
        for future in as_completed(futures):
            record = future.result()
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
            -int(row["minimum_cluster_size"]),
            -int(row["combined_cluster_size"]),
            int(row["source_index"]),
        )
    )
    if not accepted:
        _write_json(
            result_path,
            {
                "schema": "e50c_72b_teacher_route_calibration_v1",
                "pass": False,
                "failure": "no_double_audited_teacher_route_menu",
                "teacher_response_count": len(validated_rows),
                "teacher_validator_positive_count": sum(
                    row["answer_positive"] for row in validated_rows
                ),
                "candidate_count": len(candidates),
                "cluster_eligible_count": len(cluster_eligible),
                "double_audited_menu_count": 0,
                "bidirectionally_executable_count": 0,
                "selected_source_indices": [],
                "wrong_route_success_count": 0,
                "identity": _identity(),
                "teacher_responses_sha256": _sha256(teacher_path),
                "cluster_records_sha256": _sha256(cluster_path),
                "menu_records_sha256": _sha256(menu_path),
            },
        )
        print(result_path)
        return

    import vllm

    forced_cases = []
    unforced_cases = []
    for record in accepted:
        candidate = candidate_by_source[int(record["source_index"])]
        menu = _parse_menu(record["menu"])
        for strategy in menu.strategies:
            forced_cases.append(
                {
                    "source_index": int(record["source_index"]),
                    "row_id": str(record["row_id"]),
                    "problem": _forced_problem(
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
                "problem": _neutral_problem(candidate["problem"], menu),
                "answer": candidate["answer"],
                "menu": menu,
            }
        )
    llm = vllm.LLM(
        model=str(MODEL),
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

    canonicalizer = _load_frozen_canonicalizer()
    forced_keys, forced_diagnostics = _canonical_keys(
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
    unforced_keys, unforced_diagnostics = _canonical_keys(
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
                "problem_order": next(
                    int(problem["problem_order"])
                    for problem in problems
                    if int(problem["source_index"]) == case["source_index"]
                ),
                "source_index": case["source_index"],
                "row_id": case["row_id"],
                "menu": record["menu"],
                "menu_sha256": record["menu_sha256"],
                "route_sources": record["route_sources"],
                "minimum_cluster_size": record["minimum_cluster_size"],
                "combined_cluster_size": record["combined_cluster_size"],
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
            -int(row["minimum_cluster_size"]),
            int(row["problem_order"]),
        )
    )
    selected = [
        row["source_index"]
        for row in problems_out
        if row["bidirectionally_executable"]
    ][:10]
    payload = {
        "schema": "e50c_72b_teacher_route_calibration_v1",
        "pass": len(selected) == 10,
        "teacher_response_count": len(validated_rows),
        "teacher_validator_positive_count": sum(
            row["answer_positive"] for row in validated_rows
        ),
        "candidate_count": len(candidates),
        "cluster_eligible_count": len(cluster_eligible),
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
        "identity": _identity(),
        "teacher_responses_sha256": _sha256(teacher_path),
        "base_05b_responses_sha256": _sha256(private_05b_path),
        "cluster_records_sha256": _sha256(cluster_path),
        "menu_records_sha256": _sha256(menu_path),
        "forced_cases": forced_results,
        "problems": problems_out,
    }
    _write_json(result_path, payload)
    print(result_path)


if __name__ == "__main__":
    main()
