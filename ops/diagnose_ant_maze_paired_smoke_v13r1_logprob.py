#!/usr/bin/env python3
"""Diagnose immutable AntMaze v13r1 sampling/scoring batch drift."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from oat_drgrpo.ant_maze_interactive_policy import ANT_TERMINAL_PADDING_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--state-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summarize(observed: Sequence[float], expected: Sequence[float | None]) -> dict[str, float]:
    differences = [
        abs(left - right)
        for left, right in zip(observed, expected)
        if right is not None
    ]
    return {
        "count": float(len(differences)),
        "max_abs_difference": max(differences, default=0.0),
        "mean_abs_difference": sum(differences) / max(1, len(differences)),
        "rms_difference": math.sqrt(
            sum(value * value for value in differences) / max(1, len(differences))
        ),
    }


def selected_scores(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[Sequence[int]],
    selected: Sequence[int],
    support: Sequence[int],
    batch_size: int,
) -> list[float]:
    import torch

    model.train()
    output: list[float] = []
    support_tensor = torch.tensor(support, dtype=torch.long, device="cuda")
    support_index = {token: index for index, token in enumerate(support)}
    for start in range(0, len(prompts), batch_size):
        rows = prompts[start : start + batch_size]
        maximum = max(len(row) for row in rows)
        input_ids = torch.full(
            (len(rows), maximum),
            int(tokenizer.pad_token_id),
            dtype=torch.long,
            device="cuda",
        )
        attention = torch.zeros_like(input_ids)
        for index, row in enumerate(rows):
            input_ids[index, maximum - len(row) :] = torch.tensor(
                row, dtype=torch.long, device="cuda"
            )
            attention[index, maximum - len(row) :] = 1
        logits = model(
            input_ids=input_ids,
            attention_mask=attention,
            logits_to_keep=1,
        ).logits[:, -1, :].float()
        restricted = torch.log_softmax(
            logits.index_select(1, support_tensor), dim=-1
        )
        indices = torch.tensor(
            [support_index[int(token)] for token in selected[start : start + batch_size]],
            dtype=torch.long,
            device="cuda",
        )
        values = restricted[torch.arange(len(rows), device="cuda"), indices]
        output.extend(float(value) for value in values.detach().cpu())
        del logits, restricted, values
    return output


def update_slots(
    *,
    row: dict[str, Any],
    tokenizer: Any,
    support: Sequence[int],
) -> tuple[list[Sequence[int]], list[int], list[float | None], list[int]]:
    episodes = row["episodes"]
    padding_prompt = tokenizer.encode(
        ANT_TERMINAL_PADDING_PROMPT, add_special_tokens=False
    )
    prompts: list[Sequence[int]] = []
    selected: list[int] = []
    behavior: list[float | None] = []
    group_lengths: list[int] = []
    for decision_index in range(16):
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
    if len(prompts) != 256:
        raise ValueError("AntMaze fixed policy-slot count changed")
    return prompts, selected, behavior, group_lengths


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("fresh AntMaze diagnostic output required")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    identity = json.loads(args.identity.read_text())
    if (
        identity.get("schema") != "ant-maze-interactive-paired-smoke-identity-v13"
        or identity.get("artifact_cohort") != "v13r1"
        or identity.get("policy_microbatch_size") != 4
        or identity.get("jobs", {}).get("grpo") != 30202916
    ):
        raise ValueError("immutable AntMaze v13r1 identity changed")
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
        "schema": "ant-maze-paired-smoke-v13r1-logprob-batch-diagnostic-v1",
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
        for batch_size in (1, 2, 4, 8, 16):
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
