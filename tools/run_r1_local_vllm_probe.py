#!/usr/bin/env python
"""Run a tiny direct-vLLM r1 probe on selected checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from datasets import load_from_disk
from transformers import AutoConfig, AutoTokenizer
import vllm

from maxent_grpo.prompt_templates import apply_r1_template


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_CHECKPOINT_ROOT = (
    "/n/fs/similarity/maxent-grpo/var/data/"
    "drgrpo_1p5b_oat_parity_trl_r1_20260331_144832"
)
DEFAULT_DATASET_DIR = (
    "/n/fs/similarity/maxent-grpo/var/seed_paper_eval/external/"
    "SEED-GRPO/datasets/evaluation_suite"
)
DEFAULT_STOP_SEQUENCES = ["</answer>", "</answer>\n"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--steps", default="200,300")
    parser.add_argument("--checkpoint-root", type=Path, default=Path(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dataset-dir", type=Path, default=Path(DEFAULT_DATASET_DIR))
    parser.add_argument("--task", default="aime")
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def build_sampling_params(
    *,
    tokenizer_limit: int,
    model_limit: int,
    max_tokens: int,
    stop_sequences: list[str] | None,
) -> Any:
    params = vllm.SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        stop=stop_sequences,
        allowed_token_ids=list(range(tokenizer_limit))
        if model_limit > tokenizer_limit
        else None,
    )
    if model_limit > tokenizer_limit:
        blocked = [[token_id] for token_id in range(tokenizer_limit, model_limit)]
        try:
            setattr(params, "_bad_words_token_ids", blocked)
        except Exception:
            pass
        kwargs = getattr(params, "kwargs", None)
        if isinstance(kwargs, dict):
            kwargs["_bad_words_token_ids"] = blocked
    return params


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    config = AutoConfig.from_pretrained(args.base_model, local_files_only=True)
    tokenizer_limit = max(int(getattr(tokenizer, "vocab_size", 0) or 0), int(len(tokenizer)))
    model_limit = int(getattr(config, "vocab_size", 0) or 0)

    ds = load_from_disk(str(args.dataset_dir))[args.task]
    prompt = apply_r1_template(str(ds["problem"][args.prompt_index]))
    gt = str(ds["answer"][args.prompt_index])

    try:
        root = Path("/n/fs/similarity/maxent-grpo/var/seed_paper_eval/external/SEED-GRPO")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from understand_r1_zero.math_grader import answer_tag_reward_fn
    except Exception:
        answer_tag_reward_fn = None

    overview: dict[str, Any] = {
        "task": args.task,
        "prompt_index": int(args.prompt_index),
        "tokenizer_limit": tokenizer_limit,
        "model_limit": model_limit,
        "steps": {},
    }
    print(
        json.dumps(
            {
                "task": args.task,
                "prompt_index": int(args.prompt_index),
                "tokenizer_limit": tokenizer_limit,
                "model_limit": model_limit,
                "steps": [int(part.strip()) for part in str(args.steps).split(",") if part.strip()],
            },
            indent=2,
        ),
        flush=True,
    )

    for raw_step in str(args.steps).split(","):
        step = int(raw_step.strip())
        model_path = args.checkpoint_root / f"checkpoint-{step}"
        step_dir = results_dir / f"step{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        print(f"[probe] loading step {step} from {model_path}", flush=True)
        llm = vllm.LLM(
            str(model_path),
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            enable_prefix_caching=True,
            trust_remote_code=False,
        )
        step_result: dict[str, Any] = {}
        for variant, stop_sequences in (
            ("guarded", None),
            ("guarded_stop", DEFAULT_STOP_SEQUENCES),
        ):
            print(
                f"[probe] generating step={step} variant={variant} "
                f"stop={stop_sequences}",
                flush=True,
            )
            params = build_sampling_params(
                tokenizer_limit=tokenizer_limit,
                model_limit=model_limit,
                max_tokens=args.max_tokens,
                stop_sequences=stop_sequences,
            )
            outputs = llm.generate([prompt], params, use_tqdm=False)
            output = outputs[0].outputs[0]
            text = str(output.text)
            token_ids = list(output.token_ids)
            reward = None
            info = {}
            if answer_tag_reward_fn is not None:
                info, reward = answer_tag_reward_fn(text, gt, fast=False)
            payload = {
                "reward": float(reward) if reward is not None else None,
                "formatted": bool(info.get("formatted", False)) if isinstance(info, dict) else None,
                "token_count": len(token_ids),
                "text_head": text[:300],
                "text_tail": text[-300:] if text else "",
            }
            (step_dir / f"{variant}.json").write_text(json.dumps(payload, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "step": step,
                        "variant": variant,
                        "reward": payload["reward"],
                        "formatted": payload["formatted"],
                        "token_count": payload["token_count"],
                        "text_head": payload["text_head"],
                    },
                    indent=2,
                ),
                flush=True,
            )
            step_result[variant] = payload
        overview["steps"][str(step)] = step_result
        del llm

    overview_path = results_dir / "probe_overview.json"
    overview_path.write_text(json.dumps(overview, indent=2) + "\n")
    print(json.dumps({"overview_path": str(overview_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
