#!/usr/bin/env python3
"""Diagnose immutable PointMaze v1 sampling/scoring log-probability drift."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

from oat_drgrpo.point_maze_interactive_policy import POINT_TERMINAL_PADDING_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--state-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def selected_scores(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[Sequence[int]],
    selected: Sequence[int],
    support: Sequence[int],
    batch_size: int,
    training: bool,
    explicit_positions: bool,
) -> list[float]:
    import torch

    model.train(training)
    output: list[float] = []
    support_tensor = torch.tensor(support, dtype=torch.long, device="cuda")
    support_index = {token: index for index, token in enumerate(support)}
    with torch.no_grad():
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
            kwargs = {}
            if explicit_positions:
                kwargs["position_ids"] = (attention.cumsum(dim=-1) - 1).clamp_min(0)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention,
                logits_to_keep=1,
                **kwargs,
            ).logits[:, -1, :].float()
            restricted = torch.log_softmax(
                logits.index_select(1, support_tensor), dim=-1
            )
            indices = torch.tensor(
                [support_index[int(token)] for token in selected[start : start + batch_size]],
                dtype=torch.long,
                device="cuda",
            )
            values = restricted[
                torch.arange(len(rows), device="cuda"), indices
            ]
            output.extend(float(value) for value in values.cpu())
    return output


def summarize(
    observed: Sequence[float], expected: Sequence[float | None]
) -> dict[str, float]:
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


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("fresh diagnostic output required")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    all_rows = [
        json.loads(line)
        for line in args.state_replay.read_text().splitlines()
        if line
    ]
    if [row.get("update") for row in all_rows] != [1, 2, 3, 4]:
        raise ValueError("expected the immutable four-update v1 replay")
    # Only update 1 was generated from the independently hash-bound initial
    # checkpoint loaded below.  Later rows legitimately follow optimizer steps.
    rows = all_rows[:1]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
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

    prompts: list[Sequence[int]] = []
    selected: list[int] = []
    behavior: list[float | None] = []
    group_lengths: list[int] = []
    padding_prompt = tokenizer.encode(
        POINT_TERMINAL_PADDING_PROMPT, add_special_tokens=False
    )
    for row in rows:
        episodes = row["episodes"]
        maximum = max(len(episode["decisions"]) for episode in episodes)
        for decision_index in range(maximum):
            active = 0
            for episode in episodes:
                if decision_index < len(episode["decisions"]):
                    decision = episode["decisions"][decision_index]
                    prompts.append(decision["prompt_token_ids"])
                    selected.append(int(decision["selected_token_id"]))
                    position = decision["allowed_token_ids"].index(
                        decision["selected_token_id"]
                    )
                    behavior.append(float(decision["behavior_logprobs"][position]))
                    active += 1
                else:
                    prompts.append(padding_prompt)
                    selected.append(support[0])
                    behavior.append(None)
            group_lengths.append(active)

    diagnostics: dict[str, Any] = {
        "schema": "point-maze-paired-smoke-v1-grouped-logprob-diagnostic-v1",
        "state_replay": str(args.state_replay),
        "update": 1,
        "checkpoint": "initial_model",
        "active_decisions": sum(value is not None for value in behavior),
        "fixed_policy_slots": len(prompts),
        "decision_group_lengths": group_lengths,
        "settings": {},
    }
    for batch_size in (8,):
        for training in (False, True):
            for explicit_positions in (False,):
                label = (
                    f"batch{batch_size}_"
                    f"{'train' if training else 'eval'}_"
                    f"{'explicit_positions' if explicit_positions else 'implicit_positions'}"
                )
                observed = selected_scores(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    selected=selected,
                    support=support,
                    batch_size=batch_size,
                    training=training,
                    explicit_positions=explicit_positions,
                )
                diagnostics["settings"][label] = summarize(observed, behavior)
                print(label, diagnostics["settings"][label], flush=True)

    identity = json.loads(args.identity.read_text())
    trainer_path = (
        Path(identity["execution_root"])
        / "train_point_maze_interactive_paired_smoke_v1.py"
    )
    spec = importlib.util.spec_from_file_location(
        "point_v1_snapshot_trainer", trainer_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load immutable v1 trainer")
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    slots = [
        {
            "kind": "on_policy",
            "prompt_token_ids": tuple(prompt),
            "selected_token_id": token,
            "behavior_logprob": 0.0 if old is None else old,
            "episode": index % 16,
            "active": old is not None,
            "weight": 0.0,
        }
        for index, (prompt, token, old) in enumerate(
            zip(prompts, selected, behavior)
        )
    ]
    model.zero_grad(set_to_none=True)
    exact = trainer._backward_fixed_slots(
        model=model,
        tokenizer=tokenizer,
        slots=slots,
        action_token_ids=support,
        microbatch_size=8,
    )
    diagnostics["settings"]["snapshot_backward_fixed_slots_batch8"] = exact
    print("snapshot_backward_fixed_slots_batch8", exact, flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(diagnostics, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
