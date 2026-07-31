#!/usr/bin/env python3
"""Diagnose immutable PointMaze v2 sampling/scoring batch drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from diagnose_ant_maze_paired_smoke_v13r1_logprob import selected_scores, summarize
from oat_drgrpo.point_maze_interactive_policy import POINT_TERMINAL_PADDING_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--state-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def update_slots(
    *, row: dict[str, Any], tokenizer: Any, support: Sequence[int]
) -> tuple[list[Sequence[int]], list[int], list[float | None], list[int]]:
    episodes = row["episodes"]
    padding_prompt = tokenizer.encode(
        POINT_TERMINAL_PADDING_PROMPT, add_special_tokens=False
    )
    prompts: list[Sequence[int]] = []
    selected: list[int] = []
    behavior: list[float | None] = []
    group_lengths: list[int] = []
    for decision_index in range(96):
        active = 0
        for episode in episodes:
            decisions = episode["decisions"]
            if decision_index < len(decisions):
                decision = decisions[decision_index]
                prompts.append(decision["prompt_token_ids"])
                selected_token = int(decision["selected_token_id"])
                selected.append(selected_token)
                position = decision["allowed_token_ids"].index(selected_token)
                behavior.append(float(decision["behavior_logprobs"][position]))
                active += 1
            else:
                prompts.append(padding_prompt)
                selected.append(int(support[0]))
                behavior.append(None)
        group_lengths.append(active)
    if len(prompts) != 1536:
        raise ValueError("PointMaze fixed policy-slot count changed")
    return prompts, selected, behavior, group_lengths


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("fresh PointMaze diagnostic output required")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    identity = json.loads(args.identity.read_text())
    if (
        identity.get("schema") != "point-maze-interactive-paired-smoke-identity-v2"
        or identity.get("policy_microbatch_size") != 4
        or identity.get("jobs", {}).get("grpo") != 30202913
        or identity.get("seed") != 75302
    ):
        raise ValueError("immutable PointMaze v2 identity changed")
    metrics = [json.loads(line) for line in args.metrics.read_text().splitlines() if line]
    rows = [json.loads(line) for line in args.state_replay.read_text().splitlines() if line]
    if [item.get("update") for item in metrics] != [1, 2, 3, 4]:
        raise ValueError("expected four immutable metrics")
    if [item.get("update") for item in rows] != [1, 2, 3, 4]:
        raise ValueError("expected four immutable replay rows")
    if any(
        item.get("arm") != "grpo"
        or item.get("grad_norm") != 0
        or item.get("loss") != 0
        for item in metrics
    ):
        raise ValueError("GRPO checkpoint was not unchanged across the failed smoke")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    first_decision = rows[0]["episodes"][0]["decisions"][0]
    support = tuple(int(token) for token in first_decision["allowed_token_ids"])
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    torch.backends.cuda.matmul.allow_tf32 = True

    diagnostics: dict[str, Any] = {
        "schema": "point-maze-paired-smoke-v2-logprob-batch-diagnostic-v1",
        "identity": str(args.identity),
        "metrics": str(args.metrics),
        "state_replay": str(args.state_replay),
        "checkpoint": "unchanged_grpo_initial_model",
        "settings": {},
    }
    for row in rows:
        update = int(row["update"])
        prompts, selected, behavior, group_lengths = update_slots(
            row=row, tokenizer=tokenizer, support=support
        )
        update_result: dict[str, Any] = {
            "active_decisions": sum(value is not None for value in behavior),
            "decision_group_lengths": group_lengths,
            "recorded_microbatch4_max_abs_difference": metrics[update - 1][
                "behavior_live_logprob_abs_diff_max"
            ],
        }
        for batch_size in (4, 8, 16):
            observed = selected_scores(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                selected=selected,
                support=support,
                batch_size=batch_size,
            )
            label = f"batch{batch_size}_train_implicit_positions"
            update_result[label] = summarize(observed, behavior)
            print(f"update={update} {label} {update_result[label]}", flush=True)
        diagnostics["settings"][f"update_{update}"] = update_result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(diagnostics, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
