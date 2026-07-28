#!/usr/bin/env python3
"""Run one frozen E69 checkpoint on MATH-500 exactly once."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


EXPECTED_ROWS = 500
EXPECTED_ORDERED_ROW_SHA256 = (
    "1576fd11df21dc705a7c85000f232031212225cd9c00520faa26f6bdfc751166"
)
EXPECTED_ARROW_SHA256 = (
    "2104f8f8eef09ce0bfc929e255f0f04c59311f1f3395bd293f1d03051c482cf7"
)
SAMPLED_SEED = 690401
MAX_TOKENS = 1024
MAX_MODEL_LEN = 2048


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_data(data_root: Path) -> Any:
    from datasets import load_from_disk

    manifest_path = data_root / "MATERIALIZATION_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = manifest.get("output", {})
    audit = manifest.get("audit", {})
    if (
        manifest.get("schema") != "e39_math12k_384_math500_materialization_v1"
        or output.get("eval_rows") != EXPECTED_ROWS
        or output.get("eval_ordered_row_sha256")
        != EXPECTED_ORDERED_ROW_SHA256
        or audit.get("normalized_problem_overlap") != 0
    ):
        raise RuntimeError("frozen MATH-500 materialization identity mismatch")
    arrow = data_root / "eval/math/data-00000-of-00001.arrow"
    if _sha256(arrow) != EXPECTED_ARROW_SHA256:
        raise RuntimeError("frozen MATH-500 Arrow hash mismatch")
    loaded = load_from_disk(str(data_root / "eval"))
    dataset = loaded["math"]
    if len(dataset) != EXPECTED_ROWS:
        raise RuntimeError(f"expected 500 MATH-500 rows, found {len(dataset)}")
    return dataset


def _validate_identity(
    *,
    identity_path: Path,
    alias: str,
    arm: str,
    seed: int,
    checkpoint: Path,
    output: Path,
) -> tuple[dict[str, Any], str]:
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("schema") != "e69_gate4_math500_one_time_transfer_v1":
        raise RuntimeError("unexpected E69 Gate 4 identity schema")
    matches = [
        row
        for row in identity.get("evaluations", [])
        if row.get("alias") == alias
        and row.get("arm") == arm
        and int(row.get("seed", -1)) == seed
    ]
    if len(matches) != 1:
        raise RuntimeError("checkpoint is absent or duplicated in Gate 4 identity")
    row = matches[0]
    if Path(row["checkpoint"]).resolve() != checkpoint.resolve():
        raise RuntimeError("checkpoint path differs from Gate 4 identity")
    if Path(row["output"]).resolve() != output.resolve():
        raise RuntimeError("output path differs from Gate 4 identity")
    checkpoint_hash = _tree_hash(checkpoint)
    if checkpoint_hash != row["checkpoint_tree_sha256"]:
        raise RuntimeError("checkpoint tree differs from Gate 4 identity")
    return identity, checkpoint_hash


def _grade(
    responses: list[str],
    references: list[Any],
    *,
    workers: int = 4,
) -> tuple[list[float], list[dict[str, Any]]]:
    from oat_drgrpo.math_grader_process import FullMathVerifierProcess

    if len(responses) != len(references):
        raise ValueError("responses and references must have the same length")
    assignments: list[list[int]] = [[] for _ in range(workers)]
    for index in range(len(responses)):
        assignments[index % workers].append(index)

    def work(indices: list[int]) -> list[tuple[int, float, dict[str, Any]]]:
        verifier = FullMathVerifierProcess(
            reward_kind="boxed",
            timeout_seconds=5.0,
        )
        try:
            rewards, infos = verifier.grade_batch(
                [responses[index] for index in indices],
                [references[index] for index in indices],
            )
            return [
                (index, float(reward), dict(info))
                for index, reward, info in zip(indices, rewards, infos)
            ]
        finally:
            verifier.close()

    rewards = [0.0] * len(responses)
    infos: list[dict[str, Any]] = [{} for _ in responses]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for completed in executor.map(work, assignments):
            for index, reward, info in completed:
                rewards[index] = reward
                infos[index] = info
    return rewards, infos


def _load_model_and_prompts(
    *,
    checkpoint: Path,
    problems: list[str],
) -> tuple[Any, list[Any], list[int]]:
    import vllm
    from transformers import AutoTokenizer
    from vllm.inputs import TokensPrompt

    from oat_drgrpo.templates import apply_qwen_math_template

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        trust_remote_code=True,
        use_fast=False,
    )
    prompt_ids: list[list[int]] = []
    prompt_lengths: list[int] = []
    bos_token = getattr(tokenizer, "bos_token", None)
    for problem in problems:
        formatted = apply_qwen_math_template(problem)
        if bos_token:
            formatted = formatted.removeprefix(bos_token)
        encoded = list(tokenizer.encode(formatted))
        if len(encoded) > 1024:
            raise RuntimeError("MATH-500 prompt exceeds frozen 1,024-token limit")
        prompt_ids.append(encoded)
        prompt_lengths.append(len(encoded))

    llm = vllm.LLM(
        model=str(checkpoint),
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.80,
        swap_space=8,
        enable_prefix_caching=True,
    )
    prompts = [
        TokensPrompt(prompt_token_ids=encoded) for encoded in prompt_ids
    ]
    return llm, prompts, prompt_lengths


def _generate(
    *,
    llm: Any,
    prompts: list[Any],
    sample_count: int,
    temperature: float,
    seed: int,
) -> list[list[dict[str, Any]]]:
    import vllm

    params = vllm.SamplingParams(
        n=sample_count,
        temperature=temperature,
        top_p=1.0,
        max_tokens=MAX_TOKENS,
        seed=seed,
    )
    outputs = llm.generate(prompts, params)
    if len(outputs) != len(prompts):
        raise RuntimeError("vLLM returned an incomplete MATH-500 request grid")
    rows: list[list[dict[str, Any]]] = []
    for output in outputs:
        if len(output.outputs) != sample_count:
            raise RuntimeError("vLLM returned an incomplete response group")
        rows.append(
            [
                {
                    "response": sample.text.strip(),
                    "response_token_count": len(sample.token_ids),
                    "finish_reason": sample.finish_reason,
                }
                for sample in output.outputs
            ]
        )
    return rows


def _attach_rewards(
    generated: list[list[dict[str, Any]]],
    references: list[Any],
) -> None:
    width = len(generated[0])
    responses = [
        sample["response"]
        for prompt_samples in generated
        for sample in prompt_samples
    ]
    flat_references = [
        reference for reference in references for _ in range(width)
    ]
    rewards, infos = _grade(responses, flat_references)
    offset = 0
    for prompt_samples in generated:
        for sample in prompt_samples:
            sample["reward"] = rewards[offset]
            sample["verifier_info"] = infos[offset]
            offset += 1


def _metrics(rows: list[list[dict[str, Any]]]) -> dict[str, float]:
    prompt_means = [
        sum(float(sample["reward"]) for sample in samples) / len(samples)
        for samples in rows
    ]
    prompt_passes = [
        float(any(float(sample["reward"]) > 0 for sample in samples))
        for samples in rows
    ]
    flat = [sample for samples in rows for sample in samples]
    return {
        "mean_at_k": sum(prompt_means) / len(prompt_means),
        "any_correct_at_k": sum(prompt_passes) / len(prompt_passes),
        "mean_response_tokens": sum(
            int(sample["response_token_count"]) for sample in flat
        )
        / len(flat),
        "verifier_timeout_rate": sum(
            bool(sample["verifier_info"].get("verifier_timeout"))
            for sample in flat
        )
        / len(flat),
        "verifier_worker_error_rate": sum(
            bool(sample["verifier_info"].get("verifier_worker_error"))
            for sample in flat
        )
        / len(flat),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"immutable Gate 4 output already exists: {output}")
    if checkpoint.name != "step_02305":
        raise SystemExit(f"expected terminal step_02305 checkpoint: {checkpoint}")
    identity, checkpoint_hash = _validate_identity(
        identity_path=args.identity.resolve(),
        alias=args.alias,
        arm=args.arm,
        seed=args.seed,
        checkpoint=checkpoint,
        output=output,
    )
    dataset = _validate_data(args.data_root.resolve())
    problems = [str(dataset[index]["problem"]) for index in range(len(dataset))]
    references = [dataset[index]["answer"] for index in range(len(dataset))]

    llm, tokenized_prompts, prompt_lengths = _load_model_and_prompts(
        checkpoint=checkpoint,
        problems=problems,
    )
    greedy = _generate(
        llm=llm,
        prompts=tokenized_prompts,
        sample_count=1,
        temperature=0.0,
        seed=0,
    )
    _attach_rewards(greedy, references)
    sampled = _generate(
        llm=llm,
        prompts=tokenized_prompts,
        sample_count=8,
        temperature=1.0,
        seed=SAMPLED_SEED,
    )
    _attach_rewards(sampled, references)
    del llm

    prompts = []
    for index, (problem, prompt_length, greedy_rows, sampled_rows) in enumerate(
        zip(problems, prompt_lengths, greedy, sampled)
    ):
        prompts.append(
            {
                "index": index,
                "problem_sha256": hashlib.sha256(
                    problem.encode("utf-8")
                ).hexdigest(),
                "prompt_token_count": prompt_length,
                "greedy": greedy_rows[0],
                "sampled": sampled_rows,
            }
        )
    payload = {
        "schema": "e69_gate4_math500_checkpoint_result_v1",
        "identity_sha256": _sha256(args.identity.resolve()),
        "identity_schema": identity["schema"],
        "alias": args.alias,
        "arm": args.arm,
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_tree_sha256": checkpoint_hash,
        "data": {
            "rows": EXPECTED_ROWS,
            "ordered_row_sha256": EXPECTED_ORDERED_ROW_SHA256,
            "arrow_sha256": EXPECTED_ARROW_SHA256,
        },
        "requests": {
            "template": "qwen_math",
            "max_tokens": MAX_TOKENS,
            "max_model_len": MAX_MODEL_LEN,
            "greedy": {"n": 1, "temperature": 0.0, "top_p": 1.0, "seed": 0},
            "sampled": {
                "n": 8,
                "temperature": 1.0,
                "top_p": 1.0,
                "seed": SAMPLED_SEED,
            },
            "verifier": "math_verify",
        },
        "metrics": {
            "greedy": _metrics(greedy)["mean_at_k"],
            "mean8": _metrics(sampled)["mean_at_k"],
            "pass8": _metrics(sampled)["any_correct_at_k"],
            "greedy_diagnostics": _metrics(greedy),
            "sampled_diagnostics": _metrics(sampled),
        },
        "prompts": prompts,
    }
    _atomic_json(output, payload)
    print(
        f"[e69-gate4-eval] alias={args.alias} "
        f"greedy={payload['metrics']['greedy']:.4f} "
        f"mean8={payload['metrics']['mean8']:.4f} "
        f"pass8={payload['metrics']['pass8']:.4f}"
    )


if __name__ == "__main__":
    main()
