#!/usr/bin/env python3
"""Teacher-force fixed pilot continuations with vLLM; never evaluate new tasks.

prompt_logprobs are normalized over the full model vocabulary, before the
generation-time vocabulary mask. Compare with HF *raw*, not masked, logprobs.
The one generated token required by vLLM is discarded and is not an evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

from evaluate_real_domains_20260921 import (
    atomic_json, configure_vllm_cachetools_compatibility, object_sha, sha256,
)

SCHEMA = "real-domains-vllm-teacher-forcing-20260921-v1"


def validate_config(config: dict) -> None:
    if not isinstance(config.get("model"), str):
        raise ValueError("model path required")
    rows, checkpoints = config.get("rows"), config.get("checkpoints")
    if not isinstance(rows, list) or not rows or not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("nonempty fixed rows and checkpoints required")
    if len({r["row_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate row IDs")
    if len({c["name"] for c in checkpoints}) != len(checkpoints):
        raise ValueError("duplicate checkpoint names")
    for row in rows:
        if not isinstance(row["row_id"], str) or not row["row_id"] or not isinstance(row.get("task_id"), str):
            raise ValueError("named task and row required")
        for field in ("prompt_token_ids", "response_token_ids"):
            if not isinstance(row.get(field), list) or not row[field] or any(type(t) is not int or t < 0 for t in row[field]):
                raise ValueError("nonempty original integer token sequences required")
    for checkpoint in checkpoints:
        if not isinstance(checkpoint.get("name"), str) or not checkpoint["name"]:
            raise ValueError("checkpoint name required")
        if checkpoint.get("adapter") is not None and not isinstance(checkpoint["adapter"], str):
            raise ValueError("adapter must be a path or null")


def extract_score(row: dict, generated: object, checkpoint: str) -> dict:
    tokens = row["prompt_token_ids"] + row["response_token_ids"]
    if list(generated.prompt_token_ids) != tokens:
        raise ValueError("vLLM prompt token echo differs from the frozen full sequence")
    logprobs = generated.prompt_logprobs
    if logprobs is None or len(logprobs) != len(tokens):
        raise ValueError("missing or truncated teacher-forced prompt logprobs")
    values = []
    for position in range(len(row["prompt_token_ids"]), len(tokens)):
        item = logprobs[position]
        token = tokens[position]
        if item is None or token not in item:
            raise ValueError("selected response token is absent from prompt logprobs")
        value = float(item[token].logprob)
        if not math.isfinite(value) or value > 1e-5:
            raise ValueError("invalid response-token log probability")
        values.append(value)
    return {
        "checkpoint": checkpoint, "row_id": row["row_id"], "task_id": row["task_id"],
        "canonical_key": row.get("canonical_key"), "origins": row.get("origins"),
        "token_logprobs": values, "sum_logprob": sum(values),
        "mean_logprob": sum(values) / len(values),
        "response_tokens": len(values), "full_token_echo_verified": True,
        "row_sha256": object_sha(row),
    }


def adapter_identity(adapter: str | None, model: Path) -> dict | None:
    if adapter is None:
        return None
    path = Path(adapter).resolve()
    seal_path = path.parent / "complete.json"
    seal = json.loads(seal_path.read_text())
    files = {p.relative_to(path).as_posix(): sha256(p) for p in path.rglob("*") if p.is_file()}
    if files != seal["adapter_files"]:
        raise ValueError("adapter bytes differ from the checkpoint seal")
    adapter_config = json.loads((path / "adapter_config.json").read_text())
    if Path(adapter_config["base_model_name_or_path"]).resolve() != model:
        raise ValueError("adapter base-model path mismatch")
    return {"path": str(path), "seal_sha256": sha256(seal_path), "adapter_files": files,
            "adapter_config": adapter_config}


def run(config_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("diagnostics require a new output path")
    config = json.loads(config_path.read_text())
    validate_config(config)
    model = Path(config["model"]).resolve()
    if len(model.name) != 40 or not (model / "config.json").is_file():
        raise ValueError("pinned local model revision required")
    identities = {c["name"]: adapter_identity(c.get("adapter"), model) for c in config["checkpoints"]}
    import torch
    import vllm
    from vllm.lora.request import LoRARequest
    compatibility = configure_vllm_cachetools_compatibility(vllm.__version__)
    if torch.cuda.device_count() != 1:
        raise ValueError("one allocated GPU required")
    started = time.monotonic()
    use_lora = any(c.get("adapter") for c in config["checkpoints"])
    kwargs = dict(model=str(model), dtype="bfloat16", generation_config="vllm",
                  max_model_len=int(config.get("max_model_len", 8192)),
                  gpu_memory_utilization=float(config.get("gpu_memory_utilization", .75)),
                  enable_prefix_caching=False, swap_space=16.0,
                  enable_lora=use_lora)
    if use_lora:
        kwargs.update(max_lora_rank=16, lora_dtype="bfloat16")
    llm = vllm.LLM(**kwargs)
    tokenizer = llm.get_tokenizer()
    model_vocab = int(json.loads((model / "config.json").read_text())["vocab_size"])
    upper = min(len(tokenizer), model_vocab)
    for row in config["rows"]:
        tokens = row["prompt_token_ids"] + row["response_token_ids"]
        if any(token >= upper for token in tokens) or len(tokens) + 1 > kwargs["max_model_len"]:
            raise ValueError("tokens exceed tokenizer action space or frozen context")
    scores = []
    batch_size = int(config.get("batch_size", 4))
    if batch_size < 1:
        raise ValueError("positive diagnostic batch size required")
    for index, checkpoint in enumerate(config["checkpoints"]):
        adapter = checkpoint.get("adapter")
        request = LoRARequest(checkpoint["name"], index + 1, str(Path(adapter).resolve())) if adapter else None
        for offset in range(0, len(config["rows"]), batch_size):
            rows = config["rows"][offset:offset + batch_size]
            prompts = [{"prompt_token_ids": row["prompt_token_ids"] + row["response_token_ids"]} for row in rows]
            params = vllm.SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1,
                                         ignore_eos=True, seed=0)
            results = llm.generate(prompts, params, lora_request=request, use_tqdm=False)
            if len(results) != len(rows):
                raise ValueError("incorrect teacher-forcing result count")
            scores.extend(extract_score(row, result, checkpoint["name"]) for row, result in zip(rows, results))
            print(f"scored {checkpoint['name']} {min(offset + len(rows), len(config['rows']))}/{len(config['rows'])}", flush=True)
    result = {
        "schema": SCHEMA, "status": "complete", "scores": scores,
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(), "config": config,
            "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__)),
            "evaluator_helper_sha256": sha256(Path(__file__).with_name("evaluate_real_domains_20260921.py")),
            "model_config_sha256": sha256(model / "config.json"), "checkpoints": identities,
            "normalization": "full model vocabulary before sampling logits processors; compare to HF raw token logprobs",
            "generation_mask_equivalent": False, "vocab_size": model_vocab,
            "sampling_action_space_upper": upper, "lora_dtype": "bfloat16",
            "prefix_caching": False, "extra_generated_tokens_discarded": len(scores),
            "runtime_compatibility": compatibility, "torch": torch.__version__, "vllm": vllm.__version__,
            "job_id": os.environ.get("SLURM_JOB_ID"), "gpu": torch.cuda.get_device_name(0),
            "elapsed_seconds": time.monotonic() - started,
        },
    }
    atomic_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.output)
    print(json.dumps({"status": result["status"], "scores": len(result["scores"]), "output": str(args.output)}))
