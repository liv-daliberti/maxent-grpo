#!/usr/bin/env python3
"""Fixed-sample evaluation of the real-domain pilots, with durable raw receipts.

Adapters supply complete rendered prompts and verifier decisions. This runner
never exposes labels, checker code, reference programs, or verification inputs
to the model. It records every requested sample, including invalid answers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

SCHEMA = "real-domains-fixed-evaluation-20260921-v1"


class MaskUnrenderableTokens:
    """Match the training policy's tokenizer-supported categorical action space."""
    def __init__(self, upper: int):
        if upper < 1:
            raise ValueError("tokenizer vocabulary must be nonempty")
        self.upper = upper

    def __call__(self, generated_token_ids, logits):
        logits[self.upper:] = float("-inf")
        return logits


def configure_vllm_cachetools_compatibility(vllm_version: str) -> dict:
    """Bridge the private LRU method rename used by installed vLLM 0.8.4.

    The alias is confined to this evaluator process. It delegates to the exact
    installed cache implementation and changes neither weights nor sampling.
    """
    import cachetools
    cache_class = cachetools.LRUCache
    action = "not_required"
    if vllm_version == "0.8.4" and not hasattr(cache_class, "_LRUCache__update"):
        if not hasattr(cache_class, "_LRUCache__touch"):
            raise RuntimeError("unsupported cachetools LRU API for vLLM 0.8.4")
        cache_class._LRUCache__update = cache_class._LRUCache__touch
        action = "process_local_private_update_alias_to_touch"
    return {"vllm_version": vllm_version, "cachetools_version": cachetools.__version__, "action": action}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def append_rows(handle: Any, rows: list[dict]) -> None:
    for row in rows:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def discovery_probability(n: int, count: int, k: int) -> float:
    if not 0 <= count <= n or not 1 <= k <= n:
        raise ValueError("invalid without-replacement sampling counts")
    return 1.0 if n - count < k else 1.0 - math.comb(n - count, k) / math.comb(n, k)


def mode_metrics(rows: list[dict], ks: tuple[int, ...] = (1, 8, 32), threshold: int = 30) -> dict:
    n = len(rows)
    if n < 1:
        raise ValueError("cannot summarize zero requested samples")
    for row in rows:
        if type(row.get("accepted")) is not bool:
            raise ValueError("acceptance must be a boolean")
        if row["accepted"] and (not isinstance(row.get("canonical_key"), str) or not row["canonical_key"]):
            raise ValueError("accepted sample lacks canonical mode")
        if not row["accepted"] and row.get("canonical_key") is not None:
            raise ValueError("invalid samples cannot contribute modes")
    counts = Counter(row["canonical_key"] for row in rows if row["accepted"])
    m = sum(counts.values())
    eligible = m >= threshold
    return {
        "samples": n, "accepted": m, "accuracy": m / n,
        "distinct_valid_modes": len(counts), "mode_counts": dict(sorted(counts.items())),
        "pcmd_eligible": eligible, "pcmd_accepted_threshold": threshold,
        "pcmd": 1 - sum(c * (c - 1) for c in counts.values()) / (m * (m - 1)) if eligible else None,
        "pass_at_k": {str(k): discovery_probability(n, m, k) for k in ks if k <= n},
        "expected_distinct_valid_modes_at_k": {str(k): sum(discovery_probability(n, c, k) for c in counts.values()) for k in ks if k <= n},
        "hard_violation_count": sum(len(row.get("hard_violations", [])) for row in rows),
    }


def validate_decision(value: Any) -> dict:
    if not isinstance(value, dict) or type(value.get("accepted")) is not bool:
        raise ValueError("adapter must return a boolean accepted decision")
    if not isinstance(value.get("hard_violations"), list):
        raise ValueError("adapter must explicitly return hard_violations")
    if value["hard_violations"] and value["accepted"]:
        raise ValueError("hard verifier violations cannot receive positive reward")
    key = value.get("canonical_key")
    if value["accepted"] and (not isinstance(key, str) or not key):
        raise ValueError("accepted response has no canonical key")
    if not value["accepted"] and key is not None:
        raise ValueError("rejected response has a canonical key")
    if not isinstance(value.get("receipt"), dict):
        raise ValueError("adapter must return a receipt object")
    json.dumps(value, allow_nan=False)
    return value


def verify_one(task: Any, response: dict) -> dict:
    identity = {k: response[k] for k in ("task_id", "sample_index", "request_seed", "prompt_sha256", "text_sha256", "token_count", "finish_reason")}
    try:
        result = validate_decision(task.verify(response["text"]))
    except Exception as exc:
        result = {"accepted": False, "canonical_key": None, "hard_violations": [f"verifier_exception:{type(exc).__name__}:{exc}"], "receipt": {}}
    return {**identity, **result}


def summarize(tasks: list[Any], attempts: list[dict], samples: int) -> dict:
    expected = {(task.task_id, i) for task in tasks for i in range(samples)}
    observed = [(row["task_id"], row["sample_index"]) for row in attempts]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("missing, duplicate or unexpected verification request identities")
    results = []
    for task in tasks:
        rows = [row for row in attempts if row["task_id"] == task.task_id]
        metric = mode_metrics(rows)
        metadata = getattr(task, "metadata", {})
        if "known_mode_count" in metadata:
            support = int(metadata["known_mode_count"])
            if support < 1 or metric["distinct_valid_modes"] > support:
                raise ValueError("observed modes exceed the known labeled support")
            metric["known_mode_count"] = support
            metric["observed_support_fraction"] = metric["distinct_valid_modes"] / support
        results.append({"task_id": task.task_id, "family": task.family, "split": task.split, **metric})
    eligible = [r["pcmd"] for r in results if r["pcmd_eligible"]]
    hard = sum(r["hard_violation_count"] for r in results)
    return {
        "status": "complete" if hard == 0 else "audit_fail",
        "task_results": results,
        "summary": {
            "tasks": len(tasks), "samples": len(attempts),
            "accepted": sum(r["accepted"] for r in results),
            "macro_accuracy": sum(r["accuracy"] for r in results) / len(results),
            "tasks_with_multiple_modes": sum(r["distinct_valid_modes"] >= 2 for r in results),
            "families_with_multiple_modes": sorted({r["family"] for r in results if r["distinct_valid_modes"] >= 2}),
            "pcmd_eligible_tasks": len(eligible), "pcmd_total_tasks": len(results),
            "macro_pcmd_over_eligible_tasks": sum(eligible) / len(eligible) if eligible else None,
            "hard_violation_count": hard,
            "macro_pass_at_k": {k: sum(r["pass_at_k"][k] for r in results) / len(results) for k in results[0]["pass_at_k"]},
            "macro_expected_distinct_valid_modes_at_k": {k: sum(r["expected_distinct_valid_modes_at_k"][k] for r in results) / len(results) for k in results[0]["expected_distinct_valid_modes_at_k"]},
        },
    }


def load_task_selection(config: dict) -> tuple[Any, list[Any], dict]:
    adapter = importlib.import_module(config["adapter_module"])
    task_ids = config["task_ids"]
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("freeze a nonempty unique task ID list before evaluation")
    adapter_config = config["adapter_config"]
    tasks = adapter.load_tasks(adapter_config)
    mapping = {task.task_id: task for task in tasks}
    if len(mapping) != len(tasks):
        raise ValueError("adapter returned duplicate task IDs")
    selected = [mapping[task_id] for task_id in task_ids]
    for task in selected:
        if not isinstance(task.prompt, str) or not task.prompt.strip():
            raise ValueError("task prompt is empty")
    identity = adapter.dataset_identity(adapter_config)
    return adapter, selected, identity


def verify_lora_checkpoint(config: dict, model: Path) -> dict:
    adapter = Path(config["lora_path"]).resolve()
    checkpoint = adapter.parent
    seal = json.loads((checkpoint / "complete.json").read_text())
    for field, config_field in (("arm", "lora_arm"), ("completed_updates", "lora_completed_updates"), ("config_sha256", "lora_checkpoint_config_sha256")):
        if seal[field] != config[config_field]:
            raise ValueError(f"checkpoint seal differs from selected {field}")
    for filename, digest_field in (("bank.json", "bank_sha256"), ("training.pt", "training_state_sha256")):
        if sha256(checkpoint / filename) != seal[digest_field]:
            raise ValueError("checkpoint training/bank seal mismatch")
    actual_files = {p.relative_to(adapter).as_posix(): sha256(p) for p in adapter.rglob("*") if p.is_file()}
    if actual_files != seal["adapter_files"]:
        raise ValueError("checkpoint adapter seal mismatch")
    adapter_config = json.loads((adapter / "adapter_config.json").read_text())
    if Path(adapter_config["base_model_name_or_path"]).resolve() != model.resolve():
        raise ValueError("LoRA checkpoint belongs to a different base model")
    return {"seal": seal, "seal_sha256": sha256(checkpoint / "complete.json"), "adapter_config": adapter_config}


def run(config_path: Path, output: Path, preflight: bool = False) -> dict:
    config = json.loads(config_path.read_text())
    samples = config["samples_per_task"]
    if type(samples) is not int or not 1 <= samples < 10_000:
        raise ValueError("samples_per_task must be a positive integer under 10000")
    paths = {key: output.with_name(output.stem + "." + key + suffix) for key, suffix in (("responses", ".jsonl"), ("attempts", ".jsonl"), ("progress", ".json"))}
    if output.exists() or any(path.exists() for path in paths.values()):
        raise FileExistsError("evaluation artifacts already exist; choose a fresh output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    def progress(stage: str, **extra: Any) -> None:
        atomic_json(paths["progress"], {"stage": stage, "updated_at": datetime.now(timezone.utc).isoformat(), **extra})
    progress("adapter_preflight")
    adapter, tasks, dataset_identity = load_task_selection(config)
    model = Path(config["model"]).resolve()
    if model.name != config["model_revision"] or len(config["model_revision"]) != 40:
        raise ValueError("model path must be the pinned revision snapshot")
    if not (model / "config.json").is_file():
        raise FileNotFoundError(model / "config.json")
    identity = {
        "schema": SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config, "config_sha256": sha256(config_path),
        "dataset_identity": dataset_identity, "dataset_identity_sha256": object_sha(dataset_identity),
        "runner_sha256": sha256(Path(__file__)), "adapter_path": str(Path(adapter.__file__).resolve()), "adapter_sha256": sha256(Path(adapter.__file__)),
        "model": str(model), "model_config_sha256": sha256(model / "config.json"),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "task_prompts": [{"task_id": t.task_id, "family": t.family, "split": t.split, "prompt_sha256": hashlib.sha256(t.prompt.encode()).hexdigest(), "prompt": t.prompt} for t in tasks],
    }
    if config.get("lora_path"):
        identity["lora_checkpoint"] = verify_lora_checkpoint(config, model)
    if preflight:
        result = {**identity, "status": "preflight_pass", "model_sampling": False}
        atomic_json(output, result)
        progress("complete", status=result["status"])
        return result
    import torch
    import vllm
    identity["runtime_compatibility"] = configure_vllm_cachetools_compatibility(vllm.__version__)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("single-GPU pilot requires exactly one visible CUDA device")
    identity["gpu"] = {"name": torch.cuda.get_device_name(0), "visible_count": 1}
    identity["software"] = {"torch": torch.__version__, "vllm": vllm.__version__}
    load_started = time.monotonic()
    progress("model_loading")
    lora_path = config.get("lora_path")
    llm_kwargs = dict(model=str(model), dtype="bfloat16", max_model_len=int(config.get("max_model_len", 8192)), gpu_memory_utilization=float(config.get("gpu_memory_utilization", .82)), swap_space=16.0, enable_prefix_caching=True)
    lora_request = None
    if lora_path:
        from vllm.lora.request import LoRARequest
        llm_kwargs.update(enable_lora=True, max_lora_rank=int(config.get("max_lora_rank", 16)))
        lora_request = LoRARequest("pilot_adapter", 1, str(Path(lora_path).resolve()))
        identity["lora_files"] = {p.name: sha256(p) for p in sorted(Path(lora_path).iterdir()) if p.is_file()}
    llm = vllm.LLM(**llm_kwargs)
    model_load_seconds = time.monotonic() - load_started
    tokenizer = llm.get_tokenizer()
    vocab_upper = min(len(tokenizer), int(json.loads((model / "config.json").read_text())["vocab_size"]))
    identity["sampling_vocab_upper_bound"] = vocab_upper
    identity["sampling_action_space"] = "tokenizer-supported tokens, matching the HF training policy"
    max_tokens = int(config["max_tokens"])
    max_len = int(config.get("max_model_len", 8192))
    lengths = {t.task_id: len(tokenizer.encode(t.prompt, add_special_tokens=False)) for t in tasks}
    if any(length + max_tokens > max_len for length in lengths.values()):
        raise ValueError("a complete prompt plus response budget exceeds the frozen context limit")
    identity["prompt_token_counts"] = lengths
    requests = [{"task_id": task.task_id, "sample_index": i, "request_seed": int(config["seed"]) + 10_000 * ti + i, "prompt": task.prompt} for ti, task in enumerate(tasks) for i in range(samples)]
    responses = []
    generation_started = time.monotonic()
    batch_size = int(config.get("generation_batch_size", 32))
    if batch_size < 1:
        raise ValueError("generation batch size must be positive")
    with paths["responses"].open("x") as handle:
        for offset in range(0, len(requests), batch_size):
            batch = requests[offset:offset + batch_size]
            params = [vllm.SamplingParams(n=1, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=max_tokens, seed=r["request_seed"], logits_processors=[MaskUnrenderableTokens(vocab_upper)]) for r in batch]
            outputs = llm.generate([r["prompt"] for r in batch], params, lora_request=lora_request, use_tqdm=False)
            if len(outputs) != len(batch):
                raise RuntimeError("generation returned the wrong request count")
            rows = []
            for request, generated in zip(batch, outputs):
                if generated.prompt != request["prompt"]:
                    raise RuntimeError("generation response is bound to the wrong prompt")
                if len(generated.outputs) != 1:
                    raise RuntimeError("generation returned multiple completions for a one-sample request")
                answer = generated.outputs[0]
                if any(int(token) >= vocab_upper or int(token) < 0 for token in answer.token_ids):
                    raise RuntimeError("generated an unrenderable token outside the registered policy action space")
                text = str(answer.text)
                rows.append({k: request[k] for k in ("task_id", "sample_index", "request_seed")} | {
                    "prompt_sha256": hashlib.sha256(request["prompt"].encode()).hexdigest(),
                    "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "token_ids": list(answer.token_ids), "token_count": len(answer.token_ids),
                    "finish_reason": str(answer.finish_reason), "stop_reason": answer.stop_reason,
                })
            append_rows(handle, rows)
            responses.extend(rows)
            progress("generating", completed=len(responses), expected=len(requests))
            print(f"generated {len(responses)}/{len(requests)}", flush=True)
    generation_seconds = time.monotonic() - generation_started
    # Keep the allocation receipt inclusive of verifier time; avoid claiming
    # GPU savings until generation and verification actually use separate jobs.
    verification_started = time.monotonic()
    mapping = {task.task_id: task for task in tasks}
    attempts = []
    workers = int(config.get("execution_workers", 8))
    if not 1 <= workers <= 32:
        raise ValueError("execution workers must be in [1,32]")
    with paths["attempts"].open("x") as handle, ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(verify_one, mapping[row["task_id"]], row) for row in responses]
        for future in as_completed(futures):
            row = future.result()
            append_rows(handle, [row])
            attempts.append(row)
            if len(attempts) % 16 == 0 or len(attempts) == len(requests):
                progress("verifying", completed=len(attempts), expected=len(requests))
                print(f"verified {len(attempts)}/{len(requests)}", flush=True)
    outcome = summarize(tasks, attempts, samples)
    result = {**identity, **outcome, "timing": {
        "model_load_seconds": model_load_seconds, "generation_seconds": generation_seconds,
        "verification_seconds": time.monotonic() - verification_started,
        "evaluator_seconds": time.monotonic() - started,
        "generated_tokens": sum(row["token_count"] for row in responses),
        "estimated_evaluator_gpu_hours": (time.monotonic() - started) / 3600,
        "scheduler_accounting_required": True,
    }, "artifacts": {key: {"path": str(paths[key].resolve()), "sha256": sha256(paths[key])} for key in ("responses", "attempts")}}
    atomic_json(output, result)
    progress("complete", status=result["status"], summary=result["summary"])
    print(json.dumps(result["summary"], sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.output, args.preflight)
    raise SystemExit(0 if result["status"] in ("complete", "preflight_pass") else 1)
