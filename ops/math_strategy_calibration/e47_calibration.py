#!/usr/bin/env python3
"""E47 hard-MATH calibration for a validator-gated 72B strategy canonicalizer."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "var/data/math12k_384_math500/train"
MODEL_05B = (
    ROOT
    / "var/cache/huggingface/transformers/"
    "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
    "7ae557604adf67be50417f59c2c2f167def9a775"
)
QWEN72_SNAPSHOT = (
    Path("/n/fs/similarity/tokenomics/.hf_cache/hub/")
    / "models--Qwen--Qwen2.5-72B-Instruct-AWQ/snapshots/"
    "698703eae6604af048a3d2f509995dc302088217"
)
DEFAULT_OUTPUT = ROOT / "var/artifacts/e47_math_strategy_calibration_v1"
ENDPOINT_RECORD = Path(
    "/n/fs/similarity/tokenomics/runs/qwen72/endpoint.json"
)
SELECTION_SALT = "e47-cal-selection-v1"
BLINDING_SALT = "e47-cal-blinding-v1"
SAMPLE_SEED = 470064
JUDGE_SEEDS = (470721, 470722)
JUDGE_CONTEXT_LENGTH = 32768
JUDGE_MAX_PROMPT_TOKENS = 24000
JUDGE_MAX_OUTPUT_TOKENS = 8192
QUOTAS = {
    "Intermediate Algebra": 15,
    "Geometry": 8,
    "Algebra": 7,
    "Number Theory": 6,
    "Prealgebra": 5,
    "Precalculus": 5,
    "Counting & Probability": 4,
}
EXPECTED_HASHES = {
    DATA_ROOT / "train/data-00000-of-00001.arrow": (
        "359defbf82b6e05a1fdddb3479ed689f8a607dc727814e73ebfe69b2ffdff8b8"
    ),
    MODEL_05B / "config.json": (
        "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45"
    ),
    MODEL_05B / "tokenizer.json": (
        "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
    ),
    MODEL_05B / "model.safetensors": (
        "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
    ),
    QWEN72_SNAPSHOT / "config.json": (
        "ec4813b56d971f7a9c4490c7c148f997c468e97e9e8ae39e80f380d51a6dda1f"
    ),
    QWEN72_SNAPSHOT / "tokenizer.json": (
        "22bfef58b74d2fa40f402ac0c8d638d9e54d12137da2e2c7e9f32a1e7534e42f"
    ),
    QWEN72_SNAPSHOT / "model.safetensors.index.json": (
        "9f9e509271b4a2da37444f8a7ac3b3758279f6044c9c2f6eb40edc171329a65a"
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(*parts: object) -> str:
    text = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _blind_id(problem_id: str, source: str, index: object) -> str:
    return "s_" + _stable_hash(BLINDING_SALT, problem_id, source, index)[:16]


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _write_json(path: Path, value: Any) -> None:
    _atomic_text(path, _json_dumps(value) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_text(path, "".join(_json_dumps(row) + "\n" for row in rows))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _verify_frozen_inputs(include_judge: bool = True) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path, expected in EXPECTED_HASHES.items():
        if not include_judge and path.is_relative_to(QWEN72_SNAPSHOT):
            continue
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen input: {path}")
        observed = _sha256_file(path)
        if observed != expected:
            raise RuntimeError(
                f"frozen input hash mismatch: {path}: {observed} != {expected}"
            )
        hashes[str(path)] = observed
    return hashes


def _validate(text: str, answer: Any) -> tuple[dict[str, Any], float]:
    sys.path.insert(0, str(ROOT / "src"))
    from oat_drgrpo.math_grader import boxed_reward_fn

    info, reward = boxed_reward_fn(text, answer, fast=False)
    return dict(info), float(reward)


def _ensure_valid_gold(solution: str, answer: Any) -> str:
    _, reward = _validate(solution, answer)
    if reward == 1.0:
        return solution
    repaired = solution.rstrip() + f"\nTherefore the final answer is $\\boxed{{{answer}}}$."
    _, reward = _validate(repaired, answer)
    if reward != 1.0:
        raise RuntimeError("official solution could not pass the frozen validator")
    return repaired


def _format_variant(text: str) -> str:
    # Preserve line boundaries (notably Asymptote // comments) while changing
    # only horizontal/blank-line presentation.
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    result = "\n".join(lines)
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return "\n" + result + "\n"


_LEXICAL_REPLACEMENTS = (
    (r"\bTherefore\b", "Consequently"),
    (r"\btherefore\b", "consequently"),
    (r"\bThus\b", "Hence"),
    (r"\bthus\b", "hence"),
    (r"\bHence\b", "Therefore"),
    (r"\bhence\b", "therefore"),
    (r"\bWe have\b", "We obtain"),
    (r"\bwe have\b", "we obtain"),
    (r"\bIt follows that\b", "This implies that"),
    (r"\bit follows that\b", "this implies that"),
    (r"\bthe answer is\b", "the result is"),
)


def _lexical_variant(text: str) -> str:
    result = text
    for pattern, replacement in _LEXICAL_REPLACEMENTS:
        result = re.sub(pattern, replacement, result)
    if result == text:
        result = "We proceed with the same calculation. " + result
    return result


def prepare(output: Path) -> None:
    from datasets import load_from_disk

    hashes = _verify_frozen_inputs(include_judge=True)
    dataset = load_from_disk(str(DATA_ROOT))["train"]
    eligible: dict[str, list[tuple[str, int, dict[str, Any]]]] = defaultdict(list)
    for index, raw_row in enumerate(dataset):
        row = dict(raw_row)
        if str(row["level"]) != "5":
            continue
        subject = str(row["subject"])
        if subject not in QUOTAS:
            continue
        key = _stable_hash(SELECTION_SALT, row["unique_id"])
        eligible[subject].append((key, index, row))

    selected: list[tuple[str, int, dict[str, Any]]] = []
    for subject, quota in QUOTAS.items():
        subject_rows = sorted(eligible[subject], key=lambda item: item[0])
        if len(subject_rows) < quota:
            raise RuntimeError(f"not enough level-5 rows for {subject}")
        selected.extend(subject_rows[:quota])
    selected.sort(key=lambda item: item[0])
    if len(selected) != 50:
        raise AssertionError("E47 selection must contain exactly 50 rows")

    problems: list[dict[str, Any]] = []
    injections: list[dict[str, Any]] = []
    injection_key: list[dict[str, Any]] = []
    for ordinal, (_, source_index, row) in enumerate(selected):
        problem_id = f"p{ordinal:03d}"
        anchor = _ensure_valid_gold(str(row["solution"]), row["answer"])
        variants = {
            "anchor": anchor,
            "exact_duplicate": anchor,
            "format_variant": _format_variant(anchor),
            "lexical_paraphrase": _lexical_variant(anchor),
        }
        for kind, text in variants.items():
            info, reward = _validate(text, row["answer"])
            if reward != 1.0:
                raise RuntimeError(
                    f"{problem_id} {kind} failed frozen validator: {info}"
                )
            sample_id = _blind_id(problem_id, "injection", kind)
            injections.append(
                {
                    "problem_id": problem_id,
                    "sample_id": sample_id,
                    "text": text,
                }
            )
            injection_key.append(
                {
                    "problem_id": problem_id,
                    "sample_id": sample_id,
                    "injection_kind": kind,
                    "same_strategy_family": f"{problem_id}:official",
                    "validator_reward": reward,
                    "validator_info": info,
                }
            )
        problems.append(
            {
                "problem_id": problem_id,
                "source_index": source_index,
                "unique_id": row["unique_id"],
                "subject": row["subject"],
                "level": str(row["level"]),
                "problem": row["problem"],
                "answer": row["answer"],
            }
        )

    _write_jsonl(output / "problems.jsonl", problems)
    _write_jsonl(output / "injections.blinded.jsonl", injections)
    _write_jsonl(output / "private/injection_key.jsonl", injection_key)
    selection_hash = _sha256_file(output / "problems.jsonl")
    manifest = {
        "schema": "e47_math_strategy_calibration_v1",
        "status": "frozen_before_launch",
        "protocol": "paper/preregistration/e47_math_strategy_canonicalizer_calibration.md",
        "selection": {
            "rule": "level5_subject_quota_sha256",
            "salt": SELECTION_SALT,
            "quotas": QUOTAS,
            "rows": len(problems),
            "ordered_problems_sha256": selection_hash,
            "source_indices": [row["source_index"] for row in problems],
            "subject_counts": dict(Counter(row["subject"] for row in problems)),
        },
        "sampling": {
            "model": str(MODEL_05B),
            "samples_per_problem": 64,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 1024,
            "seed": SAMPLE_SEED,
            "prompt": "neutral_qwen_math",
        },
        "validator": "boxed_reward_fn_fast_false",
        "judge": {
            "model": "Qwen/Qwen2.5-72B-Instruct-AWQ",
            "snapshot": str(QWEN72_SNAPSHOT),
            "revision": QWEN72_SNAPSHOT.name,
            "temperature": 0.0,
            "permutation_seeds": list(JUDGE_SEEDS),
            "new_strategy_on_disagreement": False,
        },
        "e46_replay": {
            "sensor": "verified_bank_entropy_over_log_support_v1",
            "group_size": 16,
            "pseudocount": 1.0,
            "novelty_beta": 0.5,
            "surprisal_clip": 5.0,
            "alpha_initial": 0.1,
            "alpha_min": 0.1,
            "alpha_max": 0.5,
            "target_ratio": 0.8,
            "alpha_lr": 0.003,
            "ema_decay": 0.9,
        },
        "frozen_input_sha256": hashes,
    }
    _write_json(output / "manifest.json", manifest)
    print(f"prepared {len(problems)} hard MATH problems in {output}")


def _qwen_math_prompt(question: str) -> str:
    return (
        "<|im_start|>system\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
        "<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def generate(output: Path, batch_size: int, gpu_memory: float) -> None:
    import vllm

    _verify_frozen_inputs(include_judge=False)
    problems = _read_jsonl(output / "problems.jsonl")
    raw_root = output / "raw_policy"
    raw_root.mkdir(parents=True, exist_ok=True)
    pending = [
        row for row in problems if not (raw_root / f"{row['problem_id']}.json").is_file()
    ]
    if not pending:
        print("all 50 policy-generation shards already exist")
        return
    llm = vllm.LLM(
        model=str(MODEL_05B),
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=gpu_memory,
        enable_prefix_caching=True,
    )
    params = vllm.SamplingParams(
        n=64,
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        seed=SAMPLE_SEED,
    )
    for start in range(0, len(pending), max(batch_size, 1)):
        batch = pending[start : start + max(batch_size, 1)]
        outputs = llm.generate(
            [_qwen_math_prompt(str(row["problem"])) for row in batch],
            params,
        )
        for problem, request_output in zip(batch, outputs, strict=True):
            samples = []
            if len(request_output.outputs) != 64:
                raise RuntimeError(
                    f"{problem['problem_id']} returned "
                    f"{len(request_output.outputs)} samples, expected 64"
                )
            for sample_index, completion in enumerate(request_output.outputs):
                samples.append(
                    {
                        "problem_id": problem["problem_id"],
                        "sample_id": _blind_id(
                            problem["problem_id"], "policy", sample_index
                        ),
                        "sample_index": sample_index,
                        "text": completion.text,
                        "token_length": len(completion.token_ids),
                        "finish_reason": (
                            None
                            if getattr(completion, "finish_reason", None) is None
                            else str(completion.finish_reason)
                        ),
                    }
                )
            _write_json(
                raw_root / f"{problem['problem_id']}.json",
                {"problem_id": problem["problem_id"], "samples": samples},
            )
    print(f"generated 64 samples for {len(problems)} problems")


def grade(output: Path) -> None:
    problems = {
        row["problem_id"]: row for row in _read_jsonl(output / "problems.jsonl")
    }
    grades: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for problem_id in sorted(problems):
        raw_path = output / "raw_policy" / f"{problem_id}.json"
        if not raw_path.is_file():
            raise FileNotFoundError(f"missing generation shard: {raw_path}")
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        if len(payload["samples"]) != 64:
            raise RuntimeError(f"{raw_path} does not contain 64 samples")
        for sample in payload["samples"]:
            info, reward = _validate(sample["text"], problems[problem_id]["answer"])
            record = {
                **sample,
                "validator_reward": reward,
                "validator_info": info,
            }
            grades.append(record)
            if reward == 1.0:
                valid.append(
                    {
                        "problem_id": problem_id,
                        "sample_id": sample["sample_id"],
                        "sample_index": sample["sample_index"],
                        "text": sample["text"],
                    }
                )
    _write_jsonl(output / "policy_grades.jsonl", grades)
    _write_jsonl(output / "validated_policy.blinded.jsonl", valid)
    summary = {
        "attempts": len(grades),
        "validator_positive": len(valid),
        "validator_positive_rate": len(valid) / len(grades),
        "per_problem": dict(Counter(row["problem_id"] for row in valid)),
    }
    _write_json(output / "validation_summary.json", summary)
    print(_json_dumps(summary))


def _compact_solution(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars * 4 // 5
    tail = max_chars - head
    return text[:head] + "\n[...middle elided for context bound...]\n" + text[-tail:]


def _judge_prompt(
    problem: dict[str, Any],
    items: list[dict[str, Any]],
    per_item_chars: int,
) -> tuple[str, list[str]]:
    truncated_ids = [
        item["sample_id"] for item in items if len(item["text"]) > per_item_chars
    ]
    rendered = "\n\n".join(
        f"### ID {item['sample_id']}\n"
        f"{_compact_solution(item['text'], per_item_chars)}"
        for item in items
    )
    prompt = f"""Partition the externally validated correct solutions below by
their ESSENTIAL MATHEMATICAL STRATEGY for solving the stated problem.

Same cluster: rewording, formatting or notation changes, reordered routine
algebra, or expansion/omission of routine steps while retaining the same
central mathematical route.

Different cluster: a genuinely different central construction, theorem,
substitution, invariant, case decomposition, counting argument, or proof
route. Do not split merely because prose, LaTeX, variable names, or detail
differs. Do not judge final-answer correctness; an independent symbolic
validator already admitted every item. However, if the written path is
internally incoherent or does not actually establish that validated answer,
or if the text is insufficient to decide, put its ID in ambiguous_ids instead
of inventing a strategy.

Return one JSON object only:
{{"clusters":[{{"cluster_id":"c1","strategy":"short description",
"member_ids":["opaque id"]}}],"ambiguous_ids":[]}}
Every non-ambiguous ID must occur exactly once.

PROBLEM:
{problem['problem']}

SOLUTIONS (opaque IDs, randomly permuted):
{rendered}
"""
    return prompt, truncated_ids


def _post_slurm_json(
    endpoint: dict[str, Any], payload: dict[str, Any], timeout: int
) -> dict[str, Any]:
    job_id = str(endpoint["job_id"])
    port = int(endpoint["port"])
    if not job_id.isdigit() or not 1 <= port <= 65535:
        raise ValueError("invalid Slurm endpoint record")
    command = [
        "srun",
        f"--jobid={job_id}",
        "--overlap",
        "--ntasks=1",
        "--nodes=1",
        "curl",
        "-sS",
        "--fail-with-body",
        "--max-time",
        str(timeout),
        f"http://127.0.0.1:{port}/v1/chat/completions",
        "-H",
        "Content-Type: application/json",
        "--data-binary",
        "@-",
    ]
    result = subprocess.run(
        command,
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=timeout + 30,
        check=False,
    )
    if result.returncode:
        detail = (result.stdout or result.stderr).strip()
        raise RuntimeError(f"Qwen72 relay failed ({result.returncode}): {detail}")
    return json.loads(result.stdout)


def _extract_judge_partition(
    content: str, expected_ids: set[str]
) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    payload = json.loads(text)
    clusters = payload.get("clusters")
    ambiguous = payload.get("ambiguous_ids", [])
    if not isinstance(clusters, list) or not isinstance(ambiguous, list):
        raise ValueError("judge JSON lacks clusters/ambiguous_ids lists")
    assigned: list[str] = []
    normalized_clusters = []
    for index, cluster in enumerate(clusters):
        if not isinstance(cluster, dict) or not isinstance(
            cluster.get("member_ids"), list
        ):
            raise ValueError("judge cluster lacks member_ids list")
        members = [str(value) for value in cluster["member_ids"]]
        assigned.extend(members)
        normalized_clusters.append(
            {
                "cluster_id": str(cluster.get("cluster_id", f"c{index + 1}")),
                "strategy": str(cluster.get("strategy", "")),
                "member_ids": members,
            }
        )
    ambiguous_ids = [str(value) for value in ambiguous]
    observed = assigned + ambiguous_ids
    if len(observed) != len(set(observed)):
        raise ValueError("judge assigned an ID more than once")
    if set(observed) != expected_ids:
        missing = sorted(expected_ids - set(observed))
        extra = sorted(set(observed) - expected_ids)
        raise ValueError(f"judge assignment mismatch missing={missing} extra={extra}")
    return {"clusters": normalized_clusters, "ambiguous_ids": ambiguous_ids}


def judge(
    output: Path,
    endpoint_path: Path,
    workers: int,
    timeout: int,
) -> None:
    from transformers import AutoTokenizer

    _verify_frozen_inputs(include_judge=True)
    endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
    if endpoint.get("model") != "qwen2.5-72b":
        raise RuntimeError(f"unexpected served model: {endpoint}")
    problems = {
        row["problem_id"]: row for row in _read_jsonl(output / "problems.jsonl")
    }
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(output / "validated_policy.blinded.jsonl"):
        by_problem[row["problem_id"]].append(row)
    for row in _read_jsonl(output / "injections.blinded.jsonl"):
        by_problem[row["problem_id"]].append(row)
    judge_root = output / "judge"
    judge_root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(QWEN72_SNAPSHOT),
        local_files_only=True,
    )

    tasks = []
    for problem_id in sorted(problems):
        for pass_index, seed in enumerate(JUDGE_SEEDS):
            path = judge_root / f"{problem_id}.pass{pass_index}.json"
            if not path.is_file():
                items = list(by_problem[problem_id])
                random.Random(seed + int(problem_id[1:])).shuffle(items)
                per_item_chars = 6000
                while True:
                    prompt, truncated_ids = _judge_prompt(
                        problems[problem_id],
                        items,
                        per_item_chars,
                    )
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are a conservative mathematical-strategy "
                                "equivalence auditor. Output valid JSON only."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ]
                    prompt_tokens = len(
                        tokenizer.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                        )
                    )
                    if prompt_tokens <= JUDGE_MAX_PROMPT_TOKENS:
                        break
                    if per_item_chars <= 600:
                        raise RuntimeError(
                            f"{problem_id} cannot fit the frozen judge context"
                        )
                    scaled = int(
                        per_item_chars
                        * (JUDGE_MAX_PROMPT_TOKENS - 512)
                        / prompt_tokens
                    )
                    per_item_chars = max(
                        600,
                        min(per_item_chars - 100, scaled),
                    )
                tasks.append(
                    {
                        "problem_id": problem_id,
                        "pass_index": pass_index,
                        "seed": seed,
                        "path": path,
                        "items": items,
                        "messages": messages,
                        "prompt_tokens": prompt_tokens,
                        "per_item_chars": per_item_chars,
                        "truncated_ids": truncated_ids,
                    }
                )

    def run_one(task: dict[str, Any]) -> tuple[str, int]:
        problem_id = task["problem_id"]
        pass_index = task["pass_index"]
        seed = task["seed"]
        path = task["path"]
        items = task["items"]
        request = {
            "model": "qwen2.5-72b",
            "messages": task["messages"],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": JUDGE_MAX_OUTPUT_TOKENS,
            "seed": seed,
            "stream": False,
        }
        response = _post_slurm_json(endpoint, request, timeout)
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("judge returned no choices")
        choice = choices[0]
        content = str((choice.get("message") or {}).get("content") or "")
        partition = _extract_judge_partition(
            content, {item["sample_id"] for item in items}
        )
        record = {
            "schema": "e47_qwen72_strategy_partition_v1",
            "problem_id": problem_id,
            "pass_index": pass_index,
            "permutation_seed": seed,
            "endpoint": endpoint,
            "request_item_order": [item["sample_id"] for item in items],
            "prompt_tokens_preflight": task["prompt_tokens"],
            "per_item_character_cap": task["per_item_chars"],
            "truncated_ids": task["truncated_ids"],
            "partition": partition,
            "finish_reason": choice.get("finish_reason"),
            "usage": response.get("usage", {}),
            "raw_content": content,
        }
        _write_json(path, record)
        return problem_id, pass_index

    errors = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        future_tasks = {pool.submit(run_one, task): task for task in tasks}
        for future in as_completed(future_tasks):
            task = future_tasks[future]
            try:
                problem_id, pass_index = future.result()
                print(f"judged {problem_id} pass {pass_index}", flush=True)
            except Exception as exc:
                errors.append((task, repr(exc)))
                print(
                    "ERROR "
                    f"{task['problem_id']} pass {task['pass_index']}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
    if errors:
        raise RuntimeError(f"{len(errors)} judge calls failed: {errors[:3]}")
    print(f"completed {len(tasks)} pending judge calls")


def _assignment(partition: dict[str, Any]) -> tuple[dict[str, str], set[str]]:
    assignment = {}
    for cluster in partition["clusters"]:
        for sample_id in cluster["member_ids"]:
            assignment[sample_id] = cluster["cluster_id"]
    return assignment, set(partition["ambiguous_ids"])


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [math.nan, math.nan]
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    radius = (
        z
        * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
        / denom
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _partition_pair_map(
    assignment: dict[str, str], ambiguous: set[str], ids: list[str]
) -> dict[tuple[str, str], bool]:
    result = {}
    for left_index, left in enumerate(ids):
        if left in ambiguous:
            continue
        for right in ids[left_index + 1 :]:
            if right in ambiguous:
                continue
            result[(left, right)] = assignment[left] == assignment[right]
    return result


def _entropy_ratio(counts: Counter[str]) -> float | None:
    if len(counts) < 2:
        return None
    total = float(sum(counts.values()) + len(counts))
    probabilities = [(count + 1.0) / total for count in counts.values()]
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    return min(1.0, max(0.0, entropy / math.log(len(counts))))


def _replay_e46(
    valid_rows: list[dict[str, Any]],
    assignments: dict[str, dict[str, str]],
) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "src"))
    from oat_drgrpo.online_canonical_controller import (
        OnlineCanonicalDualController,
    )

    controller = OnlineCanonicalDualController(
        base_alpha=0.10,
        min_alpha=0.10,
        max_alpha=0.50,
        target_ratio=0.80,
        alpha_lr=0.003,
        ema_decay=0.90,
    )
    rows_by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid_rows:
        rows_by_problem[row["problem_id"]].append(row)
    observations = []
    counts_by_problem: dict[str, Counter[str]] = defaultdict(Counter)
    for group_start in range(0, 64, 16):
        eligible_ratios = []
        for problem_id in sorted(assignments):
            group = [
                row
                for row in rows_by_problem[problem_id]
                if group_start <= int(row["sample_index"]) < group_start + 16
            ]
            for row in group:
                cluster = assignments[problem_id].get(row["sample_id"])
                if cluster is not None:
                    counts_by_problem[problem_id][cluster] += 1
            ratio = _entropy_ratio(counts_by_problem[problem_id])
            if ratio is not None:
                eligible_ratios.append(ratio)
        if eligible_ratios:
            observed = sum(eligible_ratios) / len(eligible_ratios)
            diagnostics = controller.observe(observed)
        else:
            observed = None
            diagnostics = controller.idle_diagnostics()
        observations.append(
            {
                "group_end_sample_index": group_start + 15,
                "eligible_prompts": len(eligible_ratios),
                "mean_normalized_entropy": observed,
                "alpha_next": controller.current_alpha,
                "controller": diagnostics,
            }
        )
    eligible = [
        row["mean_normalized_entropy"]
        for row in observations
        if row["mean_normalized_entropy"] is not None
    ]
    return {
        "observations": observations,
        "terminal_alpha": controller.current_alpha,
        "mean_eligible_normalized_entropy": (
            sum(eligible) / len(eligible) if eligible else None
        ),
        "controller_state": controller.state_dict(),
    }


def analyze(output: Path) -> None:
    problems = _read_jsonl(output / "problems.jsonl")
    injections = _read_jsonl(output / "private/injection_key.jsonl")
    injection_by_problem: dict[str, dict[str, str]] = defaultdict(dict)
    for row in injections:
        injection_by_problem[row["problem_id"]][row["injection_kind"]] = row[
            "sample_id"
        ]
    valid_rows = _read_jsonl(output / "validated_policy.blinded.jsonl")
    all_items: dict[str, list[str]] = defaultdict(list)
    for row in valid_rows:
        all_items[row["problem_id"]].append(row["sample_id"])
    for row in injections:
        all_items[row["problem_id"]].append(row["sample_id"])

    pass_assignments: list[dict[str, dict[str, str]]] = []
    pass_ambiguous: list[dict[str, set[str]]] = []
    pass_metrics = []
    for pass_index in range(2):
        assignments: dict[str, dict[str, str]] = {}
        ambiguities: dict[str, set[str]] = {}
        by_kind: dict[str, list[bool]] = defaultdict(list)
        any_false = 0
        ambiguous_count = 0
        truncated_count = 0
        for problem in problems:
            problem_id = problem["problem_id"]
            path = output / "judge" / f"{problem_id}.pass{pass_index}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            truncated_count += len(record.get("truncated_ids", []))
            assignment, ambiguous = _assignment(record["partition"])
            assignments[problem_id] = assignment
            ambiguities[problem_id] = ambiguous
            ambiguous_count += len(ambiguous)
            anchor = injection_by_problem[problem_id]["anchor"]
            problem_false = False
            for kind in ("exact_duplicate", "format_variant", "lexical_paraphrase"):
                control = injection_by_problem[problem_id][kind]
                false_new = (
                    anchor in ambiguous
                    or control in ambiguous
                    or assignment.get(anchor) != assignment.get(control)
                )
                by_kind[kind].append(false_new)
                problem_false |= false_new
            any_false += int(problem_false)
        pass_assignments.append(assignments)
        pass_ambiguous.append(ambiguities)
        flat = [value for values in by_kind.values() for value in values]
        failures = sum(flat)
        pass_metrics.append(
            {
                "pass_index": pass_index,
                "false_new": failures,
                "comparisons": len(flat),
                "false_new_rate": failures / len(flat),
                "false_new_wilson95": _wilson(failures, len(flat)),
                "by_injection_kind": {
                    kind: {
                        "false_new": sum(values),
                        "comparisons": len(values),
                        "rate": sum(values) / len(values),
                        "wilson95": _wilson(sum(values), len(values)),
                    }
                    for kind, values in sorted(by_kind.items())
                },
                "problems_with_any_false_new": any_false,
                "problem_any_false_new_rate": any_false / len(problems),
                "ambiguous_items": ambiguous_count,
                "structural_failures": 0,
                "truncated_items": truncated_count,
                "items": sum(len(ids) for ids in all_items.values()),
            }
        )

    disagreement_numerator = 0
    disagreement_denominator = 0
    per_problem_disagreement = {}
    for problem in problems:
        problem_id = problem["problem_id"]
        ids = sorted(all_items[problem_id])
        first = _partition_pair_map(
            pass_assignments[0][problem_id], pass_ambiguous[0][problem_id], ids
        )
        second = _partition_pair_map(
            pass_assignments[1][problem_id], pass_ambiguous[1][problem_id], ids
        )
        shared = set(first) & set(second)
        differences = sum(first[pair] != second[pair] for pair in shared)
        disagreement_numerator += differences
        disagreement_denominator += len(shared)
        per_problem_disagreement[problem_id] = {
            "disagreements": differences,
            "comparable_pairs": len(shared),
            "rate": differences / len(shared) if shared else None,
        }

    replays = [
        _replay_e46(valid_rows, assignments) for assignments in pass_assignments
    ]
    terminal_alpha_difference = abs(
        replays[0]["terminal_alpha"] - replays[1]["terminal_alpha"]
    )
    entropy_means = [
        replay["mean_eligible_normalized_entropy"] for replay in replays
    ]
    entropy_difference = (
        abs(entropy_means[0] - entropy_means[1])
        if all(value is not None for value in entropy_means)
        else None
    )
    report = {
        "schema": "e47_math_strategy_calibration_report_v1",
        "primary_injection_false_new": pass_metrics,
        "partition_permutation_stability": {
            "pair_disagreements": disagreement_numerator,
            "comparable_pairs": disagreement_denominator,
            "rate": (
                disagreement_numerator / disagreement_denominator
                if disagreement_denominator
                else None
            ),
            "per_problem": per_problem_disagreement,
        },
        "policy_validation": json.loads(
            (output / "validation_summary.json").read_text(encoding="utf-8")
        ),
        "e46_replay": {
            "passes": replays,
            "terminal_alpha_absolute_difference": terminal_alpha_difference,
            "mean_normalized_entropy_absolute_difference": entropy_difference,
        },
        "manual_audit": {
            "status": "pending",
            "note": "Run audit-packet, complete labels, then run audit-score.",
        },
        "gate_status": "pending_manual_audit",
    }
    _write_json(output / "analysis.json", report)
    print(_json_dumps(report))


def audit_packet(output: Path) -> None:
    problems = {
        row["problem_id"]: row for row in _read_jsonl(output / "problems.jsonl")
    }
    injections = _read_jsonl(output / "injections.blinded.jsonl")
    injection_key = _read_jsonl(output / "private/injection_key.jsonl")
    text_by_id = {row["sample_id"]: row["text"] for row in injections}
    valid_path = output / "validated_policy.blinded.jsonl"
    valid_rows = _read_jsonl(valid_path) if valid_path.is_file() else []
    text_by_id.update({row["sample_id"]: row["text"] for row in valid_rows})
    injection_ids: dict[str, dict[str, str]] = defaultdict(dict)
    for row in injection_key:
        injection_ids[row["problem_id"]][row["injection_kind"]] = row["sample_id"]
    packet = []
    hidden = []
    rng = random.Random(470880)
    kinds = ("exact_duplicate", "format_variant", "lexical_paraphrase")
    for problem_id in sorted(problems):
        kind = kinds[rng.randrange(len(kinds))]
        left = injection_ids[problem_id]["anchor"]
        right = injection_ids[problem_id][kind]
        if rng.random() < 0.5:
            left, right = right, left
        pair_id = "a_" + _stable_hash("audit", problem_id, left, right)[:16]
        packet.append(
            {
                "pair_id": pair_id,
                "problem_id": problem_id,
                "problem": problems[problem_id]["problem"],
                "solution_a": text_by_id[left],
                "solution_b": text_by_id[right],
                "human_label": "",
                "human_reason": "",
            }
        )
        hidden.append(
            {
                "pair_id": pair_id,
                "pair_source": "injected_same",
                "expected_label": "same",
                "left_id": left,
                "right_id": right,
            }
        )
    complete_judgments = all(
        (output / "judge" / f"{problem_id}.pass{pass_index}.json").is_file()
        for problem_id in sorted(problems)
        for pass_index in range(2)
    )
    if complete_judgments and valid_rows:
        valid_ids: dict[str, list[str]] = defaultdict(list)
        for row in valid_rows:
            valid_ids[row["problem_id"]].append(row["sample_id"])
        candidate_pairs: dict[str, list[tuple[str, str, str]]] = {
            "judge_same": [],
            "judge_different": [],
        }
        for problem_id in sorted(problems):
            pass_data = []
            for pass_index in range(2):
                record = json.loads(
                    (
                        output / "judge" / f"{problem_id}.pass{pass_index}.json"
                    ).read_text(encoding="utf-8")
                )
                pass_data.append(_assignment(record["partition"]))
            first_assignment, first_ambiguous = pass_data[0]
            second_assignment, second_ambiguous = pass_data[1]
            ids = sorted(valid_ids[problem_id])
            for left_index, left in enumerate(ids):
                for right in ids[left_index + 1 :]:
                    if (
                        left in first_ambiguous
                        or right in first_ambiguous
                        or left in second_ambiguous
                        or right in second_ambiguous
                    ):
                        continue
                    first_same = first_assignment[left] == first_assignment[right]
                    second_same = second_assignment[left] == second_assignment[right]
                    if first_same != second_same:
                        continue
                    label = "judge_same" if first_same else "judge_different"
                    candidate_pairs[label].append((problem_id, left, right))
        for label, pairs in candidate_pairs.items():
            pairs.sort(key=lambda pair: _stable_hash("audit-policy", *pair))
            # The total manual packet is at most 100 pairs: 50 injected plus
            # 25 stable judged-same and 25 stable judged-different.
            for problem_id, left, right in pairs[:25]:
                if rng.random() < 0.5:
                    left, right = right, left
                pair_id = "a_" + _stable_hash(
                    "audit-policy", problem_id, left, right
                )[:16]
                packet.append(
                    {
                        "pair_id": pair_id,
                        "problem_id": problem_id,
                        "problem": problems[problem_id]["problem"],
                        "solution_a": text_by_id[left],
                        "solution_b": text_by_id[right],
                        "human_label": "",
                        "human_reason": "",
                    }
                )
                hidden.append(
                    {
                        "pair_id": pair_id,
                        "pair_source": label,
                        "expected_label": None,
                        "left_id": left,
                        "right_id": right,
                    }
                )
    rng.shuffle(packet)
    _write_jsonl(output / "manual_audit_packet.jsonl", packet)
    _write_jsonl(output / "private/manual_audit_key.jsonl", hidden)
    print(f"wrote {len(packet)} blinded manual-audit pairs")


def audit_score(output: Path) -> None:
    packet = _read_jsonl(output / "manual_audit_packet.jsonl")
    hidden = {
        row["pair_id"]: row
        for row in _read_jsonl(output / "private/manual_audit_key.jsonl")
    }
    allowed = {"same", "different", "uncertain"}
    missing = [
        row["pair_id"]
        for row in packet
        if str(row.get("human_label", "")).strip().lower() not in allowed
    ]
    if missing:
        raise RuntimeError(
            f"manual audit has {len(missing)} unlabeled/invalid pairs: {missing[:5]}"
        )

    judge_data: dict[tuple[str, int], tuple[dict[str, str], set[str]]] = {}
    for problem_id in sorted({row["problem_id"] for row in packet}):
        for pass_index in range(2):
            record = json.loads(
                (
                    output / "judge" / f"{problem_id}.pass{pass_index}.json"
                ).read_text(encoding="utf-8")
            )
            judge_data[(problem_id, pass_index)] = _assignment(
                record["partition"]
            )

    rows = []
    for row in packet:
        key = hidden[row["pair_id"]]
        left = key["left_id"]
        right = key["right_id"]
        decisions = []
        for pass_index in range(2):
            assignment, ambiguous = judge_data[(row["problem_id"], pass_index)]
            if left in ambiguous or right in ambiguous:
                decisions.append("ambiguous")
            elif assignment[left] == assignment[right]:
                decisions.append("same")
            else:
                decisions.append("different")
        if decisions == ["different", "different"]:
            consensus = "different"
        elif decisions == ["same", "same"]:
            consensus = "same"
        else:
            # E46 admission is conservative: judge disagreement/ambiguity
            # never earns a new-strategy reward.
            consensus = "not_new_fail_closed"
        label = str(row["human_label"]).strip().lower()
        rows.append(
            {
                "pair_id": row["pair_id"],
                "pair_source": key["pair_source"],
                "human_label": label,
                "judge_pass_decisions": decisions,
                "prospective_consensus": consensus,
                "false_new": label == "same" and consensus == "different",
                "false_merge": label == "different" and consensus == "same",
            }
        )

    binary = [row for row in rows if row["human_label"] != "uncertain"]
    same = [row for row in binary if row["human_label"] == "same"]
    different = [row for row in binary if row["human_label"] == "different"]
    false_new = sum(row["false_new"] for row in same)
    false_merge = sum(row["false_merge"] for row in different)
    manual = {
        "status": "complete",
        "packet_sha256": _sha256_file(output / "manual_audit_packet.jsonl"),
        "labeled_pairs": len(packet),
        "uncertain_pairs": len(packet) - len(binary),
        "same_pairs": len(same),
        "same_pair_false_new": false_new,
        "same_pair_false_new_rate": false_new / len(same) if same else None,
        "same_pair_false_new_wilson95": _wilson(false_new, len(same)),
        "different_pairs": len(different),
        "different_pair_false_merge": false_merge,
        "different_pair_false_merge_rate": (
            false_merge / len(different) if different else None
        ),
        "different_pair_false_merge_wilson95": _wilson(
            false_merge, len(different)
        ),
        "scored_pairs": rows,
    }
    report_path = output / "analysis.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["manual_audit"] = manual

    injection_passes = report["primary_injection_false_new"]
    stability = report["partition_permutation_stability"]["rate"]
    replay = report["e46_replay"]
    structural_rates = [
        (
            metrics["ambiguous_items"] + metrics.get("structural_failures", 0)
        )
        / metrics["items"]
        for metrics in injection_passes
        if metrics["items"]
    ]
    gate_checks = {
        "all_50_injection_families_valid": len(
            _read_jsonl(output / "private/injection_key.jsonl")
        )
        == 200,
        "exact_duplicate_false_new_zero_both_passes": all(
            metrics["by_injection_kind"]["exact_duplicate"]["false_new"] == 0
            for metrics in injection_passes
        ),
        "overall_false_new_rate_at_most_0_05_both_passes": all(
            metrics["false_new_rate"] <= 0.05 for metrics in injection_passes
        ),
        "overall_false_new_wilson_upper_at_most_0_10_both_passes": all(
            metrics["false_new_wilson95"][1] <= 0.10
            for metrics in injection_passes
        ),
        "lexical_false_new_rate_at_most_0_10_both_passes": all(
            metrics["by_injection_kind"]["lexical_paraphrase"]["rate"] <= 0.10
            for metrics in injection_passes
        ),
        "pairwise_partition_disagreement_at_most_0_05": (
            stability is not None and stability <= 0.05
        ),
        "judge_ambiguity_at_most_0_05_both_passes": (
            len(structural_rates) == 2
            and all(rate <= 0.05 for rate in structural_rates)
        ),
        "manual_same_false_new_at_most_0_05": (
            manual["same_pair_false_new_rate"] is not None
            and manual["same_pair_false_new_rate"] <= 0.05
        ),
        "manual_different_false_merge_at_most_0_20": (
            manual["different_pair_false_merge_rate"] is not None
            and manual["different_pair_false_merge_rate"] <= 0.20
        ),
        "e46_terminal_alpha_difference_at_most_0_03": (
            replay["terminal_alpha_absolute_difference"] <= 0.03
        ),
        "e46_mean_entropy_difference_at_most_0_05": (
            replay["mean_normalized_entropy_absolute_difference"] is not None
            and replay["mean_normalized_entropy_absolute_difference"] <= 0.05
        ),
    }
    report["gate_checks"] = gate_checks
    report["gate_status"] = (
        "pass" if all(gate_checks.values()) else "fail"
    )
    _write_json(report_path, report)
    _write_json(output / "manual_audit_score.json", manual)
    print(_json_dumps({"gate_status": report["gate_status"], **gate_checks}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "generate",
            "grade",
            "judge",
            "analyze",
            "audit-packet",
            "audit-score",
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gpu-memory", type=float, default=0.80)
    parser.add_argument("--endpoint", type=Path, default=ENDPOINT_RECORD)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args.output)
    elif args.command == "generate":
        generate(args.output, args.batch_size, args.gpu_memory)
    elif args.command == "grade":
        grade(args.output)
    elif args.command == "judge":
        judge(args.output, args.endpoint, args.workers, args.timeout)
    elif args.command == "analyze":
        analyze(args.output)
    elif args.command == "audit-packet":
        audit_packet(args.output)
    elif args.command == "audit-score":
        audit_score(args.output)


if __name__ == "__main__":
    main()
