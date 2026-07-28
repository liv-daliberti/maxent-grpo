#!/usr/bin/env python3
"""Audit terminal E49C eval responses under the exact execution contract."""

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
sys.path.insert(0, str(ROOT / "src"))

from oat_drgrpo.math_strategy_canonicalizer import MathStrategyCanonicalizer


PREFIX = {
    "toy": "e49c_finite_action_math_toy_05b_v1",
    "full": "e49c_finite_action_math_full_05b_v1",
}
ARMS = ("grpo", "online_canonical_haarnoja")
KINDS = (
    "deterministic_greedy_trace_neutral",
    "fixed_seed_sampled_k_neutral",
)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_name, path)


def _run_dir(prefix: str, arm: str) -> pathlib.Path:
    pattern = (
        "xdr_qwen25_0p5b_instruct_"
        f"{arm}_{prefix}_{arm}_s45"
    )
    matches = sorted((ROOT / "var/data").glob(pattern))
    complete = [
        path for path in matches if (path / "TRAINING_COMPLETE.json").is_file()
    ]
    if len(complete) != 1:
        raise RuntimeError(
            f"expected one complete E49C {arm} run, found {len(complete)}"
        )
    return complete[0]


def _terminal_records(run_dir: pathlib.Path) -> tuple[pathlib.Path, dict[str, dict]]:
    paths = sorted(run_dir.glob("debug_*/eval_mode_coverage_draws.jsonl"))
    if not paths:
        raise RuntimeError(f"missing eval traces in {run_dir}")
    path = paths[-1]
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    terminal = {}
    for kind in KINDS:
        candidates = [
            record
            for record in records
            if record.get("evaluation_kind") == kind
            and record.get("benchmark") == "math"
        ]
        if not candidates:
            raise RuntimeError(f"missing {kind} in {path}")
        terminal[kind] = max(candidates, key=lambda row: int(row["step"]))
    steps = {int(record["step"]) for record in terminal.values()}
    if len(steps) != 1:
        raise RuntimeError("terminal greedy and sampled traces differ in step")
    return path, terminal


def _endpoint(
    record_path: pathlib.Path,
    host_override: str,
) -> tuple[str, str, dict]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    expected = {
        "model": "qwen2.5-72b",
        "node": "node105",
        "tensor_parallel_size": 4,
        "max_model_len": 32768,
        "max_num_seqs": 8,
        "enforce_eager": True,
        "checkpoint_revision": "698703eae6604af048a3d2f509995dc302088217",
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError("unexpected E49C judge endpoint")
    host = host_override or str(record["node"])
    return (
        f"http://{host}:{int(record['port'])}/v1",
        str(record["model"]),
        record,
    )


def _audit_record(
    canonicalizer: MathStrategyCanonicalizer,
    record: dict,
    *,
    token_namespace: int,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, float]]:
    prompts = record.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise RuntimeError("evaluation record has no prompt traces")
    sample_count = int(record["sample_count"])
    prompt_token_ids = []
    prompt_texts = []
    response_texts = []
    positives = []
    active = []
    row_meta = []
    for prompt_position, prompt in enumerate(prompts):
        responses = prompt.get("responses")
        rewards = prompt.get("rewards")
        if (
            not isinstance(responses, list)
            or not isinstance(rewards, list)
            or len(responses) != sample_count
            or len(rewards) != sample_count
        ):
            raise RuntimeError("evaluation trace has incomplete response group")
        raw_problem = str(prompt["prompt"])
        for sample_position, (response, reward) in enumerate(
            zip(responses, rewards, strict=True)
        ):
            prompt_token_ids.append(
                [token_namespace, prompt_position + 1]
            )
            prompt_texts.append(raw_problem)
            response_texts.append(str(response))
            positives.append(float(reward) > 0)
            active.append(True)
            row_meta.append((prompt_position, sample_position))

    keys, diagnostics = canonicalizer.canonicalize(
        prompt_token_ids=prompt_token_ids,
        prompt_texts=prompt_texts,
        response_texts=response_texts,
        task_reward_positive=positives,
        active_mask=active,
        num_samples=sample_count,
    )
    grouped_keys = [
        keys[start : start + sample_count]
        for start in range(0, len(keys), sample_count)
    ]
    grouped_positive = [
        positives[start : start + sample_count]
        for start in range(0, len(positives), sample_count)
    ]
    contract_correct = sum(key is not None for key in keys)
    raw_correct = sum(positives)
    prompt_count = len(prompts)
    metrics = {
        "step": float(record["step"]),
        "sample_count": float(sample_count),
        "prompt_count": float(prompt_count),
        "raw_mean_correct": raw_correct / len(keys),
        "raw_pass_at_k": (
            sum(any(group) for group in grouped_positive) / prompt_count
        ),
        "execution_gated_mean_correct": contract_correct / len(keys),
        "execution_gated_pass_at_k": (
            sum(any(key is not None for key in group) for group in grouped_keys)
            / prompt_count
        ),
        "audited_distinct_correct_strategies_mean": (
            sum(len({key for key in group if key is not None}) for group in grouped_keys)
            / prompt_count
        ),
        "audited_support_at_least_two_prompt_fraction": (
            sum(
                len({key for key in group if key is not None}) >= 2
                for group in grouped_keys
            )
            / prompt_count
        ),
        "answer_positive_contract_acceptance_fraction": (
            contract_correct / raw_correct if raw_correct else 0.0
        ),
    }
    evidence = []
    for prompt_position, (prompt, group_keys, group_positive) in enumerate(
        zip(prompts, grouped_keys, grouped_positive, strict=True)
    ):
        responses = prompt["responses"]
        evidence.append(
            {
                "prompt_index": int(prompt["prompt_index"]),
                "raw_positive": [bool(value) for value in group_positive],
                "outcome_keys": group_keys,
                "response_sha256": [
                    hashlib.sha256(str(response).encode("utf-8")).hexdigest()
                    for response in responses
                ],
            }
        )
    diagnostic_payload = {
        field: float(getattr(diagnostics, field))
        for field in diagnostics.__dataclass_fields__
    }
    return metrics, evidence, diagnostic_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(PREFIX), required=True)
    parser.add_argument("--endpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--host-override", default="")
    args = parser.parse_args()

    prefix = PREFIX[args.stage]
    traces = {}
    input_hashes = {}
    for arm in ARMS:
        run_dir = _run_dir(prefix, arm)
        trace_path, records = _terminal_records(run_dir)
        traces[arm] = records
        input_hashes[arm] = {
            "run_dir": str(run_dir.relative_to(ROOT)),
            "trace_path": str(trace_path.relative_to(ROOT)),
            "trace_sha256": _sha256(trace_path),
        }
    endpoint_url, model, endpoint_record = _endpoint(
        args.endpoint, args.host_override
    )
    identity = {
        "stage": args.stage,
        "inputs": input_hashes,
        "endpoint_record_sha256": _sha256(args.endpoint),
        "canonicalizer_sha256": _sha256(
            ROOT / "src/oat_drgrpo/math_strategy_canonicalizer.py"
        ),
    }
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("identity") != identity:
            raise RuntimeError("existing E49C eval audit identity drift")
        print(args.output)
        return

    results = {}
    for arm_index, arm in enumerate(ARMS, start=1):
        canonicalizer = MathStrategyCanonicalizer(
            endpoint=endpoint_url,
            model=model,
            timeout_seconds=900,
            max_workers=4,
            max_item_chars=4000,
        )
        arm_result = {}
        for kind_index, kind in enumerate(KINDS, start=1):
            metrics, evidence, diagnostics = _audit_record(
                canonicalizer,
                traces[arm][kind],
                token_namespace=100 * arm_index + kind_index,
            )
            arm_result[kind] = {
                "metrics": metrics,
                "diagnostics": diagnostics,
                "prompts": evidence,
            }
        results[arm] = arm_result

    report = {
        "schema": "e49c_terminal_execution_gated_eval_v1",
        "identity": identity,
        "endpoint_job_id": endpoint_record["job_id"],
        "arms": results,
    }
    _write_json(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
