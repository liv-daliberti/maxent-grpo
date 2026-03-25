"""Small helper to mimic SEED's training-time eval path on one task.

This reproduces the important trainer-side differences relative to
`evaluate_model.py`:
- batched generation (`eval_batch_size`)
- optional stripped outputs
- optional fast grader
"""

from __future__ import annotations

import argparse
import json
from statistics import mean

from datasets import load_from_disk
import vllm

from understand_r1_zero.math_grader import answer_tag_reward_fn, boxed_reward_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--template", choices=["no", "qwen_math", "r1"], default="no")
    parser.add_argument("--eval-batch-size", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--fast-grader", action="store_true")
    parser.add_argument("--strip-output", action="store_true")
    parser.add_argument("--swap-space", type=float, default=0.0)
    return parser.parse_args()


def apply_template(template: str, question: str) -> str:
    if template == "no":
        return question
    if template == "qwen_math":
        return (
            "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
            "<|im_start|>user\n"
            + question
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
    if template == "r1":
        return (
            "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
            "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
            "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, "
            "respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: "
            + question
            + "\nAssistant: <think>"
        )
    raise ValueError(template)


def get_reward_fn(template: str):
    if template == "r1":
        return answer_tag_reward_fn
    return boxed_reward_fn


def main() -> int:
    args = parse_args()
    dataset = load_from_disk(args.dataset_name)[args.task]
    prompts = [apply_template(args.template, q) for q in dataset["problem"]]
    targets = dataset["answer"]
    reward_fn = get_reward_fn(args.template)
    sampling_params = vllm.SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
    )
    llm = vllm.LLM(
        args.model_name,
        swap_space=args.swap_space,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        enable_prefix_caching=True,
    )

    rewards = []
    lengths = []
    formatted = []
    for start in range(0, len(prompts), args.eval_batch_size):
        batch_prompts = prompts[start : start + args.eval_batch_size]
        batch_targets = targets[start : start + args.eval_batch_size]
        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
        for output, gt in zip(outputs, batch_targets):
            text = output.outputs[0].text
            if args.strip_output:
                text = text.strip()
            info, reward = reward_fn(text, gt, fast=args.fast_grader)
            rewards.append(float(reward))
            lengths.append(len(output.outputs[0].token_ids))
            formatted.append(bool(info.get("formatted", False)))

    result = {
        args.task: mean(rewards),
        "avg_len": mean(lengths),
        "max_len": max(lengths),
        "formatted": mean(formatted),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
