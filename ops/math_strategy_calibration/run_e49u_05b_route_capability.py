#!/usr/bin/env python3
"""Measure whether 0.5B can execute each frozen E49T dual-menu route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL = (
    ROOT
    / "var/cache/huggingface/transformers/"
    "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
    "7ae557604adf67be50417f59c2c2f167def9a775"
)
DATA = ROOT / "var/data/e49t_natural_menu_math_toy"
IDENTITY = (
    ROOT
    / "var/artifacts/"
    "e49t_natural_menu_math_toy_05b_3ep_v1_identity.json"
)
ENDPOINT_RECORD = (
    ROOT / "var/artifacts/e49t_qwen72_node302_v1/qwen72_endpoint.json"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e49u_05b_route_capability_calibration_20260726.md"
)
SAMPLE_COUNT = 8
SEED = 490761


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _forced_problem(problem: str, menu: Any, strategy: Any) -> str:
    marker = "</strategy_menu_json>"
    if problem.count(marker) != 1:
        raise RuntimeError("E49U source prompt lost its exact menu boundary")
    menu_end = problem.index(marker) + len(marker)
    return (
        problem[:menu_end]
        + "\n\nROUTE-CAPABILITY AUDIT: execute exactly "
        + f"{strategy.strategy_id}: {strategy.action_combo}. "
        + "State that strategy ID and combo, then materially execute every "
        + "listed action in order as a natural mathematical derivation. Do "
        + "not use, mention, mix, or switch to any other route. Finish with "
        + "the final answer inside \\boxed{}. A correct answer reached by a "
        + "different route does not pass this audit."
    )


def _endpoint(record_path: pathlib.Path) -> tuple[str, str]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    expected = {
        "model": "qwen2.5-72b",
        "node": "node302",
        "port": 8770,
        "tensor_parallel_size": 4,
        "max_model_len": 32768,
        "max_num_seqs": 8,
        "enforce_eager": True,
        "checkpoint_revision": "698703eae6604af048a3d2f509995dc302088217",
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError("E49U requires the frozen E49T Qwen72 endpoint")
    return (
        f"http://{record['node']}:{int(record['port'])}/v1",
        str(record["model"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=(
            ROOT / "var/artifacts/e49u_05b_route_capability_calibration_v1"
        ),
    )
    args = parser.parse_args()
    output = args.output_root.resolve()
    result_path = output / "result.json"
    if result_path.exists():
        raise RuntimeError(f"fresh E49U result required: {result_path}")

    identity = json.loads(IDENTITY.read_text(encoding="utf-8"))
    if (
        identity.get("schema")
        != "e49t_natural_menu_math_toy_05b_3ep_v1"
        or identity.get("support_counts")
        != {"train_multi": 10, "eval_multi": 10}
        or identity.get("task_reward_gate")
        != "answer_validator_plus_unanimous_finite_menu_route_inference"
    ):
        raise RuntimeError("E49U source identity is not frozen E49T")
    frozen_source = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e49t_natural_menu_{identity['source_hash']}"
        / "src"
    )
    if not frozen_source.is_dir():
        raise RuntimeError("E49U frozen E49T source snapshot is missing")
    sys.path.insert(0, str(ROOT / "src"))

    from datasets import load_from_disk
    import vllm

    from oat_drgrpo.math_grader import boxed_reward_fn
    import oat_drgrpo

    # Use the corrected successor verifier, but keep route admission bound to
    # the exact canonicalizer snapshot that passed E49T's blinded calibration.
    oat_drgrpo.__path__.insert(0, str(frozen_source / "oat_drgrpo"))
    from oat_drgrpo.math_strategy_canonicalizer import (
        MathStrategyCanonicalizer,
    )
    from oat_drgrpo.math_strategy_menu import parse_strategy_menu
    from oat_drgrpo.templates import apply_qwen_math_template

    endpoint, judge_model = _endpoint(ENDPOINT_RECORD)

    cases = []
    for split in ("train", "eval"):
        dataset_dict = load_from_disk(str(DATA / split))
        dataset = dataset_dict[next(iter(dataset_dict))]
        for row_index, row in enumerate(dataset):
            menu = parse_strategy_menu(str(row["problem"]))
            if menu is None:
                raise RuntimeError("E49U source row has no menu")
            if len(menu.strategies) < 2:
                continue
            if len(menu.strategies) != 2:
                raise RuntimeError("E49U calibration expects exactly two routes")
            for strategy in menu.strategies:
                problem = _forced_problem(str(row["problem"]), menu, strategy)
                cases.append(
                    {
                        "split": split,
                        "row_index": row_index,
                        "row_id": str(
                            row.get("unique_id")
                            or hashlib.sha256(
                                str(row["original_problem"]).encode("utf-8")
                            ).hexdigest()[:20]
                        ),
                        "answer": str(row["answer"]),
                        "problem": problem,
                        "menu": menu,
                        "strategy": strategy,
                    }
                )
    if len(cases) != 40:
        raise RuntimeError(f"E49U expected 40 problem-route cases, saw {len(cases)}")

    prompts = [apply_qwen_math_template(case["problem"]) for case in cases]
    llm = vllm.LLM(
        model=str(MODEL),
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.80,
        swap_space=8.0,
        enable_prefix_caching=True,
        seed=SEED,
    )
    outputs = llm.generate(
        prompts,
        vllm.SamplingParams(
            n=SAMPLE_COUNT,
            temperature=1.0,
            top_p=1.0,
            max_tokens=1024,
            seed=SEED,
        ),
    )
    if len(outputs) != len(cases):
        raise RuntimeError("E49U vLLM output count mismatch")

    flat_prompt_tokens = []
    flat_problem_texts = []
    flat_responses = []
    flat_answer_positive = []
    flat_active = []
    raw_rows = []
    for case_index, (case, request_output) in enumerate(
        zip(cases, outputs, strict=True)
    ):
        if len(request_output.outputs) != SAMPLE_COUNT:
            raise RuntimeError("E49U vLLM sample count mismatch")
        encoded_prompt = list(request_output.prompt_token_ids)
        for sample_index, sample in enumerate(request_output.outputs):
            text = str(sample.text)
            _, reward = boxed_reward_fn(text, case["answer"], fast=False)
            flat_prompt_tokens.append(encoded_prompt)
            flat_problem_texts.append(case["problem"])
            flat_responses.append(text)
            flat_answer_positive.append(float(reward) > 0.0)
            flat_active.append(True)
            raw_rows.append(
                {
                    "case_index": case_index,
                    "split": case["split"],
                    "row_index": case["row_index"],
                    "row_id": case["row_id"],
                    "strategy_id": case["strategy"].strategy_id,
                    "action_combo": case["strategy"].action_combo,
                    "sample_index": sample_index,
                    "answer_positive": float(reward) > 0.0,
                    "response": text,
                    "response_sha256": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                }
            )

    judge = MathStrategyCanonicalizer(
        endpoint=endpoint,
        model=judge_model,
        timeout_seconds=600,
        max_workers=4,
        permutation_seeds=(470721, 470722),
        max_item_chars=4000,
        missing_ids_are_ambiguous=True,
        allow_unstructured_menu_inference=True,
    )
    keys, diagnostics = judge.canonicalize(
        prompt_token_ids=flat_prompt_tokens,
        prompt_texts=flat_problem_texts,
        response_texts=flat_responses,
        task_reward_positive=flat_answer_positive,
        active_mask=flat_active,
        num_samples=SAMPLE_COUNT,
    )

    case_results = []
    for case_index, case in enumerate(cases):
        start = case_index * SAMPLE_COUNT
        stop = start + SAMPLE_COUNT
        expected_key = judge._menu_strategy_key(
            case["menu"], case["strategy"].strategy_id
        )
        answer_successes = sum(flat_answer_positive[start:stop])
        expected_route_successes = sum(
            keys[index] == expected_key for index in range(start, stop)
        )
        wrong_route_successes = sum(
            keys[index] is not None and keys[index] != expected_key
            for index in range(start, stop)
        )
        case_results.append(
            {
                "split": case["split"],
                "row_index": case["row_index"],
                "row_id": case["row_id"],
                "menu_sha256": case["menu"].sha256,
                "strategy_id": case["strategy"].strategy_id,
                "action_combo": case["strategy"].action_combo,
                "sample_count": SAMPLE_COUNT,
                "answer_success_count": answer_successes,
                "forced_route_success_count": expected_route_successes,
                "wrong_route_success_count": wrong_route_successes,
                "minimally_executable": expected_route_successes >= 1,
            }
        )

    problem_results = []
    for split in ("train", "eval"):
        row_ids = sorted(
            {
                row["row_id"]
                for row in case_results
                if row["split"] == split
            }
        )
        for row_id in row_ids:
            routes = [
                row
                for row in case_results
                if row["split"] == split and row["row_id"] == row_id
            ]
            if len(routes) != 2:
                raise RuntimeError("E49U problem did not retain exactly two routes")
            problem_results.append(
                {
                    "split": split,
                    "row_id": row_id,
                    "row_index": routes[0]["row_index"],
                    "bidirectionally_executable": all(
                        row["minimally_executable"] for row in routes
                    ),
                    "route_success_counts": {
                        row["strategy_id"]: row["forced_route_success_count"]
                        for row in routes
                    },
                }
            )

    raw_path = output / "private/responses.jsonl"
    _write_jsonl(raw_path, raw_rows)
    split_bidirectional = {
        split: sum(
            row["bidirectionally_executable"]
            for row in problem_results
            if row["split"] == split
        )
        for split in ("train", "eval")
    }
    payload = {
        "schema": "e49u_05b_route_capability_calibration_v1",
        "pass": split_bidirectional["train"] >= 10,
        "decision_rule": {
            "route_minimum_successes_of_8": 1,
            "required_bidirectional_train_problems": 10,
        },
        "identity": {
            "e49t_identity_sha256": _sha256(IDENTITY),
            "e49t_data_manifest_sha256": _sha256(
                DATA / "MATERIALIZATION_MANIFEST.json"
            ),
            "endpoint_record_sha256": _sha256(ENDPOINT_RECORD),
            "protocol_sha256": _sha256(PROTOCOL),
            "script_sha256": _sha256(pathlib.Path(__file__).resolve()),
            "canonicalizer_sha256": _sha256(
                frozen_source
                / "oat_drgrpo/math_strategy_canonicalizer.py"
            ),
            "math_grader_sha256": _sha256(
                ROOT / "src/oat_drgrpo/math_grader.py"
            ),
            "frozen_source_hash": identity["source_hash"],
            "model_revision": MODEL.name,
            "seed": SEED,
        },
        "sample_count_per_route": SAMPLE_COUNT,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 1024,
        "case_count": len(case_results),
        "problem_count": len(problem_results),
        "bidirectionally_executable": split_bidirectional,
        "wrong_route_success_count": sum(
            row["wrong_route_success_count"] for row in case_results
        ),
        "canonicalizer_diagnostics": {
            field: float(getattr(diagnostics, field))
            for field in diagnostics.__dataclass_fields__
        },
        "private_responses_sha256": _sha256(raw_path),
        "cases": case_results,
        "problems": problem_results,
    }
    _write_json(result_path, payload)
    print(result_path)


if __name__ == "__main__":
    main()
