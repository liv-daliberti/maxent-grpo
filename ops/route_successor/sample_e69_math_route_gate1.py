#!/usr/bin/env python3
"""Generate and audit the frozen E69 base-model MATH route archive."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MODEL = (
    ROOT / "var/cache/huggingface/transformers/"
    "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
    "7ae557604adf67be50417f59c2c2f167def9a775"
)
DATA_ROOT = ROOT / "var/data/math12k_384_route_dev128_v1"
PROTOCOL = (
    ROOT / "paper/preregistration/e69_verified_route_successor_protocol_20260728.md"
)
OUTPUT_ROOT = ROOT / "var/artifacts/e69_math_route_gate1_base_v1"
SAMPLE_COUNT = 8
PROMPT_COUNT = 128
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_TOKENS = 1024
MAX_MODEL_LEN = 2048
SEED = 690101
MIN_TASK_CORRECT = 50
MIN_ROUTE_COVERAGE = 0.80
MIN_MULTI_ROUTE_PROMPTS = 8
MIN_RECURRING_SIGNATURES = 3
MANUAL_REVIEW_SIZE = 50
IDENTITY_FILES = (
    Path("src/oat_drgrpo/math_route.py"),
    Path("src/oat_drgrpo/math_grader.py"),
    Path("src/oat_drgrpo/templates.py"),
    Path("ops/route_successor/sample_e69_math_route_gate1.py"),
    Path("paper/preregistration/e69_verified_route_successor_protocol_20260728.md"),
    Path("var/data/math12k_384_route_dev128_v1/MATERIALIZATION_MANIFEST.json"),
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_identity() -> tuple[str, dict[str, str]]:
    hashes = {str(relative): _sha256(ROOT / relative) for relative in IDENTITY_FILES}
    payload = (
        "\n".join(f"{digest}  {path}" for path, digest in sorted(hashes.items())) + "\n"
    )
    return _sha256_bytes(payload.encode("utf-8")), hashes


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _formatting_variant(response: str, block: str) -> str:
    parsed = json.loads(block)
    formatted = json.dumps(parsed, indent=2, sort_keys=True)
    return re.sub(
        r"<route>.*?</route>",
        f"<route>\n{formatted}\n</route>",
        response,
        count=1,
        flags=re.DOTALL,
    )


def _score_sample(
    record: dict[str, Any],
) -> dict[str, Any]:
    from oat_drgrpo.math_grader import (
        boxed_reward_fn,
        validated_math_route_signature,
    )
    from oat_drgrpo.math_route import (
        extract_math_route_block,
        validate_math_route_response,
    )

    response = str(record["response"])
    problem = str(record["problem"])
    answer = record["answer"]
    _info, reward = boxed_reward_fn(response, answer, fast=False)
    task_correct = float(reward) > 0.0
    validation = validate_math_route_response(response, problem)
    route_signature = validated_math_route_signature(
        response,
        problem,
        answer,
        fast=False,
        task_verified=task_correct,
    )
    route_accepted = route_signature is not None
    if route_accepted != (task_correct and validation is not None):
        raise RuntimeError(
            "route admission disagrees with task/trace gate intersection"
        )
    formatting_stable: bool | None = None
    if route_accepted:
        block = extract_math_route_block(response)
        if block is None:
            raise RuntimeError("accepted route has no unique route block")
        variant = _formatting_variant(response, block)
        formatting_stable = (
            validated_math_route_signature(
                variant,
                problem,
                answer,
                fast=False,
                task_verified=True,
            )
            == route_signature
        )
    return {
        **record,
        "task_correct": task_correct,
        "trace_parse_execute": validation is not None,
        "route_accepted": route_accepted,
        "route_signature": route_signature,
        "route_formatting_stable": formatting_stable,
        "route_terminal_value": (
            str(validation.terminal_value) if validation is not None else None
        ),
        "route_operations": (
            list(validation.operations) if validation is not None else []
        ),
        "route_step_count": (
            int(validation.step_count) if validation is not None else 0
        ),
        "route_source_count": (
            int(validation.source_count) if validation is not None else 0
        ),
    }


def automatic_gate_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(rows) != PROMPT_COUNT * SAMPLE_COUNT:
        raise ValueError("fixed sample archive has the wrong row count")
    task_correct = [row for row in rows if bool(row["task_correct"])]
    accepted = [row for row in rows if bool(row["route_accepted"])]
    if any(not bool(row["task_correct"]) for row in accepted):
        raise ValueError("route archive admitted a task-incorrect row")
    coverage = len(accepted) / len(task_correct) if task_correct else 0.0
    prompt_signatures: dict[str, set[str]] = defaultdict(set)
    signature_prompts: dict[str, set[str]] = defaultdict(set)
    for row in accepted:
        prompt_id = str(row["unique_id"])
        signature = str(row["route_signature"])
        prompt_signatures[prompt_id].add(signature)
        signature_prompts[signature].add(prompt_id)
    multi_route_prompts = sum(
        len(signatures) >= 2 for signatures in prompt_signatures.values()
    )
    recurring_signatures = sum(
        len(prompt_ids) >= 2 for prompt_ids in signature_prompts.values()
    )
    unstable_rows = sum(row["route_formatting_stable"] is not True for row in accepted)
    violations: list[str] = []
    if len(task_correct) < MIN_TASK_CORRECT:
        violations.append(
            f"task-correct samples {len(task_correct)} < {MIN_TASK_CORRECT}"
        )
    if coverage < MIN_ROUTE_COVERAGE:
        violations.append(f"route coverage {coverage:.6f} < {MIN_ROUTE_COVERAGE:.6f}")
    if unstable_rows:
        violations.append(
            f"{unstable_rows} accepted routes changed under JSON formatting"
        )
    if multi_route_prompts < MIN_MULTI_ROUTE_PROMPTS:
        violations.append(
            f"multi-route prompts {multi_route_prompts} < {MIN_MULTI_ROUTE_PROMPTS}"
        )
    if recurring_signatures < MIN_RECURRING_SIGNATURES:
        violations.append(
            "cross-prompt recurring signatures "
            f"{recurring_signatures} < {MIN_RECURRING_SIGNATURES}"
        )
    if len(accepted) < MANUAL_REVIEW_SIZE:
        violations.append(
            f"accepted routes {len(accepted)} < manual review size {MANUAL_REVIEW_SIZE}"
        )
    return {
        "status": "pass" if not violations else "fail",
        "violations": violations,
        "thresholds": {
            "minimum_task_correct_samples": MIN_TASK_CORRECT,
            "minimum_route_coverage_given_task_correct": (MIN_ROUTE_COVERAGE),
            "minimum_multi_route_prompts": MIN_MULTI_ROUTE_PROMPTS,
            "minimum_cross_prompt_recurring_signatures": (MIN_RECURRING_SIGNATURES),
            "formatting_stability_required": 1.0,
            "manual_review_size": MANUAL_REVIEW_SIZE,
        },
        "counts": {
            "prompts": len({str(row["unique_id"]) for row in rows}),
            "samples": len(rows),
            "task_correct_samples": len(task_correct),
            "trace_parse_execute_samples": sum(
                bool(row["trace_parse_execute"]) for row in rows
            ),
            "route_accepted_samples": len(accepted),
            "route_accepted_distinct_signatures": len(signature_prompts),
            "multi_route_prompts": multi_route_prompts,
            "cross_prompt_recurring_signatures": recurring_signatures,
            "formatting_unstable_accepted_rows": unstable_rows,
        },
        "rates": {
            "task_correct": len(task_correct) / len(rows),
            "trace_parse_execute": sum(bool(row["trace_parse_execute"]) for row in rows)
            / len(rows),
            "route_accepted": len(accepted) / len(rows),
            "route_coverage_given_task_correct": coverage,
        },
        "prompt_route_support_histogram": {
            str(support): sum(
                len(signatures) == support for signatures in prompt_signatures.values()
            )
            for support in sorted({len(value) for value in prompt_signatures.values()})
        },
        "recurring_signature_prompt_counts": sorted(
            (
                {
                    "route_signature": signature,
                    "distinct_prompts": len(prompt_ids),
                }
                for signature, prompt_ids in signature_prompts.items()
                if len(prompt_ids) >= 2
            ),
            key=lambda row: (
                -int(row["distinct_prompts"]),
                str(row["route_signature"]),
            ),
        ),
    }


def _manual_review_queue(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    accepted = [row for row in rows if bool(row["route_accepted"])]
    ranked = sorted(
        accepted,
        key=lambda row: _sha256_bytes(
            (
                f"{SEED}|{row['unique_id']}|{row['sample_index']}|"
                f"{row['response_sha256']}"
            ).encode("utf-8")
        ),
    )
    queue: list[dict[str, Any]] = []
    for review_index, row in enumerate(
        ranked[:MANUAL_REVIEW_SIZE],
        start=1,
    ):
        queue.append(
            {
                "review_id": f"e69-g1-{review_index:03d}",
                "prompt_index": row["prompt_index"],
                "sample_index": row["sample_index"],
                "unique_id": row["unique_id"],
                "subject": row["subject"],
                "level": row["level"],
                "problem": row["problem"],
                "gold_answer": row["answer"],
                "response": row["response"],
                "response_sha256": row["response_sha256"],
                "route_signature": row["route_signature"],
                "route_terminal_value": row["route_terminal_value"],
                "route_operations": row["route_operations"],
                "review_decision": None,
                "review_notes": None,
            }
        )
    return queue


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--expected-input-identity",
        default=os.environ.get("OAT_E69_GATE1_INPUT_IDENTITY", ""),
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise RuntimeError(
            f"fresh fixed archive required; output already exists: {output_root}"
        )
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    for path in (MODEL, DATA_ROOT, PROTOCOL):
        if not path.exists():
            raise FileNotFoundError(path)
    identity, input_hashes = _input_identity()
    if not args.expected_input_identity or args.expected_input_identity != identity:
        raise RuntimeError("E69 Gate 1 input identity is absent or does not match")
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()

    from datasets import load_from_disk
    import vllm

    from oat_drgrpo.templates import apply_qwen_math_route_template

    datasets = load_from_disk(str(DATA_ROOT / "eval"))
    if set(datasets) != {"math_dev"}:
        raise RuntimeError("sealed E69 eval root is not exactly math_dev")
    dataset = datasets["math_dev"]
    if len(dataset) != PROMPT_COUNT:
        raise RuntimeError("sealed E69 route-dev split is not 128 rows")
    source_rows = [dict(dataset[index]) for index in range(len(dataset))]
    prompts = [
        apply_qwen_math_route_template(str(row["problem"])) for row in source_rows
    ]
    llm = vllm.LLM(
        model=str(MODEL),
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.80,
        swap_space=8.0,
        enable_prefix_caching=True,
        seed=SEED,
    )
    outputs = llm.generate(
        prompts,
        vllm.SamplingParams(
            n=SAMPLE_COUNT,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            seed=SEED,
        ),
    )
    if len(outputs) != PROMPT_COUNT:
        raise RuntimeError("vLLM returned the wrong prompt count")

    raw_rows: list[dict[str, Any]] = []
    for prompt_index, (source, output) in enumerate(
        zip(source_rows, outputs, strict=True)
    ):
        if len(output.outputs) != SAMPLE_COUNT:
            raise RuntimeError("vLLM returned the wrong sample width")
        for sample_index, sample in enumerate(output.outputs):
            response = str(sample.text)
            raw_rows.append(
                {
                    "prompt_index": prompt_index,
                    "sample_index": sample_index,
                    "unique_id": str(source["unique_id"]),
                    "subject": str(source["subject"]),
                    "level": int(source["level"]),
                    "problem": str(source["problem"]),
                    "answer": source["answer"],
                    "prompt_sha256": _sha256_bytes(
                        prompts[prompt_index].encode("utf-8")
                    ),
                    "response": response,
                    "response_sha256": _sha256_bytes(response.encode("utf-8")),
                    "finish_reason": str(sample.finish_reason),
                    "generated_tokens": len(sample.token_ids),
                }
            )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        scored_rows = list(pool.map(_score_sample, raw_rows))
    automatic = automatic_gate_summary(scored_rows)
    queue = _manual_review_queue(scored_rows)
    output_root.mkdir(parents=True, exist_ok=False)
    archive_path = output_root / "sample_archive.jsonl"
    queue_path = output_root / "manual_review_queue.jsonl"
    _atomic_jsonl(archive_path, scored_rows)
    _atomic_jsonl(queue_path, queue)
    summary = {
        "schema": "e69_math_route_gate1_base_archive_v1",
        "created_at": "2026-07-28",
        "automatic_gate": automatic,
        "manual_gate": {
            "status": "pending",
            "required_reviews": MANUAL_REVIEW_SIZE,
            "false_admission_tolerance": 0,
            "queue_rows": len(queue),
        },
        "sampling_contract": {
            "model": str(MODEL.relative_to(ROOT)),
            "model_revision": MODEL.name,
            "dataset": str(DATA_ROOT.relative_to(ROOT)),
            "split": "math_dev",
            "prompt_template": "qwen_math_route",
            "prompt_count": PROMPT_COUNT,
            "samples_per_prompt": SAMPLE_COUNT,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
            "max_model_len": MAX_MODEL_LEN,
            "seed": SEED,
        },
        "provenance": {
            "git_commit": git_commit,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "input_identity_sha256": identity,
            "input_sha256": input_hashes,
        },
        "artifacts": {
            "sample_archive": str(archive_path.relative_to(ROOT)),
            "sample_archive_sha256": _sha256(archive_path),
            "manual_review_queue": str(queue_path.relative_to(ROOT)),
            "manual_review_queue_sha256": _sha256(queue_path),
        },
    }
    _atomic_json(output_root / "automatic_summary.json", summary)
    print(
        json.dumps(
            {
                "status": automatic["status"],
                "task_correct": automatic["counts"]["task_correct_samples"],
                "route_accepted": automatic["counts"]["route_accepted_samples"],
                "coverage_given_correct": automatic["rates"][
                    "route_coverage_given_task_correct"
                ],
                "manual_queue": len(queue),
                "violations": automatic["violations"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
