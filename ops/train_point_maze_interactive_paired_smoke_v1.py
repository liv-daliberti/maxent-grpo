#!/usr/bin/env python3
"""Train one arm of the frozen PointMaze interactive paired smoke."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = Path(os.environ.get("OAT_ZERO_SOURCE_ROOT", ROOT / "src")).resolve()
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oat_drgrpo.canonical_replay import (  # noqa: E402
    canonical_replay_split_mass_balance_loss,
)
from oat_drgrpo.interactive_episode_objective import (  # noqa: E402
    add_verified_advantage_outside_centering,
    drgrpo_task_advantages,
)
from oat_drgrpo.interactive_episode_replay import (  # noqa: E402
    InteractiveDecisionRecord,
    InteractiveEpisodeRecord,
    VerifiedInteractiveReplayBank,
    fixed_replay_slots,
    interactive_transition_sha256,
)
from oat_drgrpo.maze_modebench import parse_maze_action_spec  # noqa: E402
from oat_drgrpo.online_canonical_bank import OnlineCanonicalBank  # noqa: E402
from oat_drgrpo.point_maze_interactive_policy import (  # noqa: E402
    POINT_POLICY_LABELS,
    POINT_TERMINAL_PADDING_PROMPT,
    render_point_policy_prompt_v3,
)
from oat_drgrpo.point_maze_interactive_process import (  # noqa: E402
    PointMazeInteractiveProcess,
)
from oat_drgrpo.semantic_shannon import SemanticShannonTracker  # noqa: E402


CONTROL = "grpo"
TREATMENT = "verified_first_global_replay_canonical"
ARMS = (CONTROL, TREATMENT)
ROW_INDICES = (0, 2, 4, 6)
ROW_INDICES_V2 = (1, 3, 5, 7)
EXPECTED_FAMILIES = ("bar7", "block9", "bar9", "asymmetric_block9")
GEOMETRY_SHIFT_FAMILIES = (
    "wide_block9_shift",
    "cross9_shift",
    "upper_offset9_shift",
    "lower_offset9_shift",
)
SAMPLES = 16
HORIZON = 96
REPLAY_CAPACITY = 16
ARM_SEED_OFFSET = {CONTROL: 0, TREATMENT: 100_000_000}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-split-root", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--viability-receipt", type=Path, required=True)
    parser.add_argument("--admission-audit", type=Path, required=True)
    parser.add_argument("--warmstart-receipt", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--output-replay", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=75301)
    parser.add_argument("--learning-rate", type=float, default=2e-7)
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--execution-hash", required=True)
    return parser.parse_args()


def _restricted_sample(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    action_token_ids: Sequence[int],
    request_seeds: Sequence[int],
    max_length: int,
) -> tuple[list[list[int]], list[int], list[tuple[float, ...]]]:
    import torch

    if len(prompts) != len(request_seeds):
        raise ValueError("one request seed is required per policy slot")
    encoded = [tokenizer.encode(text, add_special_tokens=False) for text in prompts]
    if any(not row or len(row) > max_length for row in encoded):
        raise ValueError("PointMaze policy prompt length left the frozen bound")
    maximum = max(len(row) for row in encoded)
    input_ids = torch.full(
        (len(encoded), maximum),
        int(tokenizer.pad_token_id),
        dtype=torch.long,
        device="cuda",
    )
    attention = torch.zeros_like(input_ids)
    for index, row in enumerate(encoded):
        input_ids[index, maximum - len(row) :] = torch.tensor(row, device="cuda")
        attention[index, maximum - len(row) :] = 1
    with torch.no_grad():
        decision = model(
            input_ids=input_ids,
            attention_mask=attention,
            logits_to_keep=1,
        ).logits[:, -1, :].float()
        restricted = decision.index_select(
            1,
            torch.tensor(action_token_ids, dtype=torch.long, device="cuda"),
        )
        logprobs = torch.log_softmax(restricted, dim=-1).cpu()
    selected = []
    rows = []
    for row, seed in zip(logprobs.tolist(), request_seeds):
        probabilities = [math.exp(value) for value in row]
        draw = random.Random(int(seed)).random()
        cumulative = 0.0
        index = len(probabilities) - 1
        for candidate, probability in enumerate(probabilities):
            cumulative += probability
            if draw <= cumulative:
                index = candidate
                break
        selected.append(int(action_token_ids[index]))
        rows.append(tuple(float(value) for value in row))
    return encoded, selected, rows


def _collate_selected_logprobs(
    *,
    model: Any,
    tokenizer: Any,
    slots: Sequence[Mapping[str, Any]],
    action_token_ids: Sequence[int],
):
    import torch

    maximum = max(len(slot["prompt_token_ids"]) for slot in slots)
    input_ids = torch.full(
        (len(slots), maximum),
        int(tokenizer.pad_token_id),
        dtype=torch.long,
        device="cuda",
    )
    attention = torch.zeros_like(input_ids)
    for index, slot in enumerate(slots):
        row = slot["prompt_token_ids"]
        input_ids[index, maximum - len(row) :] = torch.tensor(row, device="cuda")
        attention[index, maximum - len(row) :] = 1
    decision = model(
        input_ids=input_ids,
        attention_mask=attention,
        logits_to_keep=1,
    ).logits[:, -1, :].float()
    support = torch.tensor(action_token_ids, dtype=torch.long, device="cuda")
    restricted = torch.log_softmax(decision.index_select(1, support), dim=-1)
    support_index = {int(token): index for index, token in enumerate(action_token_ids)}
    selected_indices = torch.tensor(
        [support_index[int(slot["selected_token_id"])] for slot in slots],
        dtype=torch.long,
        device="cuda",
    )
    return restricted[
        torch.arange(len(slots), device="cuda"),
        selected_indices,
    ]


def _backward_fixed_slots(
    *,
    model: Any,
    tokenizer: Any,
    slots: Sequence[Mapping[str, Any]],
    action_token_ids: Sequence[int],
    microbatch_size: int,
) -> dict[str, float]:
    import torch

    total_loss = 0.0
    ratio_differences = []
    for start in range(0, len(slots), microbatch_size):
        batch = slots[start : start + microbatch_size]
        live = _collate_selected_logprobs(
            model=model,
            tokenizer=tokenizer,
            slots=batch,
            action_token_ids=action_token_ids,
        )
        losses = []
        for index, slot in enumerate(batch):
            weight = float(slot["weight"])
            if slot["kind"] == "on_policy":
                old = torch.tensor(
                    float(slot["behavior_logprob"]),
                    dtype=live.dtype,
                    device=live.device,
                )
                ratio = torch.exp(live[index] - old)
                advantage_weight = torch.tensor(
                    weight,
                    dtype=live.dtype,
                    device=live.device,
                )
                unclipped = ratio * advantage_weight
                clipped = torch.clamp(ratio, 0.8, 1.2) * advantage_weight
                losses.append(-torch.minimum(unclipped, clipped))
                if bool(slot["active"]):
                    ratio_differences.append(
                        abs(float((live[index] - old).detach().item()))
                    )
            elif slot["kind"] == "replay":
                losses.append(live[index] * weight)
            else:
                raise ValueError("unknown PointMaze loss-slot kind")
        loss = torch.stack(losses).sum()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite PointMaze policy loss")
        loss.backward()
        total_loss += float(loss.detach().item())
    return {
        "loss": total_loss,
        "behavior_live_logprob_abs_diff_max": max(ratio_differences, default=0.0),
    }


def _rollout_group(
    *,
    model: Any,
    tokenizer: Any,
    worker: PointMazeInteractiveProcess,
    row: Mapping[str, Any],
    row_index: int,
    update_index: int,
    arm: str,
    seed: int,
    action_token_ids: Sequence[int],
    padding_token_ids: Sequence[int],
    max_length: int,
) -> tuple[list[InteractiveEpisodeRecord], list[dict[str, Any]], dict[str, Any]]:
    raw_spec = row["answer"]
    if isinstance(raw_spec, str):
        raw_spec = json.loads(raw_spec)
    spec = parse_maze_action_spec(raw_spec)
    if tuple(spec.action_tokens) != (
        "N", "NE", "E", "SE", "S", "SW", "W", "NW", "COAST"
    ) or spec.max_actions != HORIZON or spec.action_repeat != 5:
        raise ValueError("PointMaze action contract drift")
    group_prompt_ids = tuple(
        tokenizer.encode(str(row["problem"]), add_special_tokens=False)
    )
    sessions = [
        {
            "session_id": f"{arm}-u{update_index}-e{episode}",
            "spec": raw_spec,
        }
        for episode in range(SAMPLES)
    ]
    reset = worker.reset_batch(sessions)
    observations = {item["session_id"]: item for item in reset}
    active = [True] * SAMPLES
    histories: list[list[str]] = [[] for _ in range(SAMPLES)]
    decisions: list[list[InteractiveDecisionRecord]] = [[] for _ in range(SAMPLES)]
    final_transitions: list[dict[str, Any] | None] = [None] * SAMPLES
    fixed_slots: list[dict[str, Any]] = []
    simulator_step_calls = 0

    model.eval()
    for decision_round in range(1, HORIZON + 1):
        prompts = []
        request_seeds = []
        for episode in range(SAMPLES):
            if active[episode]:
                session_id = sessions[episode]["session_id"]
                prompts.append(
                    render_point_policy_prompt_v3(
                        str(row["problem"]),
                        observations[session_id],
                        tuple(spec.action_tokens),
                        histories[episode],
                    )
                )
            else:
                prompts.append(POINT_TERMINAL_PADDING_PROMPT)
            request_seeds.append(
                int(seed)
                + ARM_SEED_OFFSET[arm]
                + update_index * 1_000_000
                + episode * 1_000
                + decision_round
            )
        prompt_ids, selected_ids, behavior_rows = _restricted_sample(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            action_token_ids=action_token_ids,
            request_seeds=request_seeds,
            max_length=max_length,
        )
        choices = []
        active_episodes = []
        for episode in range(SAMPLES):
            if active[episode]:
                action_position = list(action_token_ids).index(selected_ids[episode])
                action = spec.action_tokens[action_position]
                choices.append(
                    {"session_id": sessions[episode]["session_id"], "action": action}
                )
                active_episodes.append((episode, action))
            fixed_slots.append(
                {
                    "kind": "on_policy",
                    "prompt_token_ids": tuple(prompt_ids[episode]),
                    "selected_token_id": int(selected_ids[episode]),
                    "behavior_logprob": float(
                        behavior_rows[episode][
                            list(action_token_ids).index(selected_ids[episode])
                        ]
                    ),
                    "episode": episode,
                    "active": bool(active[episode]),
                    "weight": 0.0,
                }
            )
        transitions = worker.step_batch(choices) if choices else []
        if len(transitions) != len(active_episodes):
            raise RuntimeError("PointMaze worker returned the wrong step batch")
        simulator_step_calls += len(transitions)
        for (episode, action), transition in zip(active_episodes, transitions):
            session_id = sessions[episode]["session_id"]
            if transition["session_id"] != session_id:
                raise RuntimeError("PointMaze worker reordered interactive sessions")
            before = observations[session_id]
            transition_hash = interactive_transition_sha256(
                before=before,
                action=action,
                after=transition,
            )
            position = list(action_token_ids).index(selected_ids[episode])
            decisions[episode].append(
                InteractiveDecisionRecord(
                    prompt_token_ids=tuple(prompt_ids[episode]),
                    allowed_token_ids=tuple(action_token_ids),
                    selected_token_id=int(selected_ids[episode]),
                    behavior_logprobs=tuple(behavior_rows[episode]),
                    transition_sha256=transition_hash,
                )
            )
            histories[episode].append(action)
            observations[session_id] = transition
            if transition["done"]:
                active[episode] = False
                final_transitions[episode] = transition
    if any(active) or any(value is None for value in final_transitions):
        raise RuntimeError("not every PointMaze episode reached a terminal state")
    if len(fixed_slots) != SAMPLES * HORIZON:
        raise RuntimeError("PointMaze fixed policy-slot budget changed")
    episodes = []
    for episode, transition in enumerate(final_transitions):
        assert transition is not None
        key = transition.get("canonical_key")
        reward = float(key is not None)
        episodes.append(
            InteractiveEpisodeRecord(
                group_prompt_token_ids=group_prompt_ids,
                outcome_key=None if key is None else str(key),
                task_reward=reward,
                decisions=tuple(decisions[episode]),
            )
        )
    return episodes, fixed_slots, {
        "row_index": row_index,
        "family": str(row["answer_mode_family"]),
        "verified_episodes": sum(episode.task_reward > 0 for episode in episodes),
        "distinct_verified_keys": len(
            {episode.outcome_key for episode in episodes if episode.outcome_key is not None}
        ),
        "active_decisions": sum(len(episode.decisions) for episode in episodes),
        "fixed_policy_slots": len(fixed_slots),
        "simulator_step_calls": simulator_step_calls,
    }


def _replay_slots(
    *,
    group: Any,
    padding_token_ids: Sequence[int],
    action_token_ids: Sequence[int],
    compute_only: bool,
    horizon: int = HORIZON,
    replay_capacity: int = REPLAY_CAPACITY,
    samples: int = SAMPLES,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    import torch

    padded, active = fixed_replay_slots(group, slot_count=replay_capacity)
    active_episodes = [episode for episode in padded if episode is not None]
    if active_episodes:
        behavior_scores = torch.tensor(
            [
                sum(
                    decision.selected_behavior_logprob
                    for decision in episode.decisions
                )
                / len(episode.decisions)
                for episode in active_episodes
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        split = canonical_replay_split_mass_balance_loss(
            behavior_scores,
            [len(active_episodes)],
        )
        raw = 0.10 * split.mass_score_gradients + 0.10 * split.balance_score_gradients
        raw_l2 = float(torch.linalg.vector_norm(raw).item())
        applied = torch.zeros_like(raw) if compute_only else raw
        applied_l2 = float(torch.linalg.vector_norm(applied).item())
        balance_eligible = int(split.balance_eligible_groups)
    else:
        raw = torch.zeros(0, dtype=torch.float64)
        applied = raw
        raw_l2 = applied_l2 = 0.0
        balance_eligible = 0

    slots: list[dict[str, Any]] = []
    active_index = 0
    for episode, is_active in zip(padded, active):
        if is_active:
            assert episode is not None
            gradient = float(applied[active_index].item())
            per_decision = gradient / len(episode.decisions) / samples
            for decision in episode.decisions:
                slots.append(
                    {
                        "kind": "replay",
                        "prompt_token_ids": decision.prompt_token_ids,
                        "selected_token_id": decision.selected_token_id,
                        "behavior_logprob": decision.selected_behavior_logprob,
                        "active": True,
                        "weight": per_decision,
                    }
                )
            for _ in range(horizon - len(episode.decisions)):
                slots.append(
                    {
                        "kind": "replay",
                        "prompt_token_ids": tuple(padding_token_ids),
                        "selected_token_id": int(action_token_ids[0]),
                        "behavior_logprob": 0.0,
                        "active": False,
                        "weight": 0.0,
                    }
                )
            active_index += 1
        else:
            for _ in range(horizon):
                slots.append(
                    {
                        "kind": "replay",
                        "prompt_token_ids": tuple(padding_token_ids),
                        "selected_token_id": int(action_token_ids[0]),
                        "behavior_logprob": 0.0,
                        "active": False,
                        "weight": 0.0,
                    }
                )
    if len(slots) != replay_capacity * horizon:
        raise RuntimeError("PointMaze fixed replay-decision budget changed")
    return slots, {
        "replay_mode_slots": float(len(padded)),
        "replay_active_modes": float(len(active_episodes)),
        "replay_balance_eligible_groups": float(balance_eligible),
        "replay_raw_score_gradient_l2": raw_l2,
        "replay_applied_score_gradient_l2": applied_l2,
        "replay_compute_only": float(compute_only),
        "replay_decision_forward_slots": float(len(slots)),
    }


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    if (
        args.seed not in (75301, 75302, 75303, 75304)
        or args.learning_rate != 2e-7
        or args.max_length != 1536
        or args.microbatch_size <= 0
    ):
        raise ValueError("PointMaze paired-smoke frozen optimizer contract drift")
    for fresh in (args.output_receipt, args.output_metrics, args.output_replay):
        if fresh.exists():
            raise FileExistsError(f"fresh PointMaze output required: {fresh}")
    for required in (
        args.model / "config.json",
        args.data_root / "identity.json",
        args.data_split_root / "dataset_dict.json",
        args.worker_python,
        args.protocol,
        args.viability_receipt,
        args.admission_audit,
        args.warmstart_receipt,
        args.identity,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("PointMaze paired smoke requires a BF16 GPU")
    viability = json.loads(args.viability_receipt.read_text())
    admission = json.loads(args.admission_audit.read_text())
    identity = json.loads(args.identity.read_text())
    identity_schema = identity.get("schema")
    if identity_schema == "point-maze-interactive-paired-smoke-identity-v1":
        version = 1
        expected_seed = 75301
        row_indices = ROW_INDICES
    elif identity_schema == "point-maze-interactive-paired-smoke-identity-v2":
        version = 2
        expected_seed = 75302
        row_indices = ROW_INDICES_V2
    elif identity_schema == "point-maze-interactive-paired-smoke-identity-v3":
        version = 3
        expected_seed = 75303
        row_indices = ROW_INDICES_V2
        expected_families = EXPECTED_FAMILIES
        expected_verified = 32
    elif identity_schema == "point-maze-interactive-paired-smoke-identity-v4":
        version = 4
        expected_seed = 75304
        row_indices = ROW_INDICES
        expected_families = GEOMETRY_SHIFT_FAMILIES
        expected_verified = 10
    else:
        raise ValueError("PointMaze paired-smoke identity schema drift")
    if version in {1, 2}:
        expected_families = EXPECTED_FAMILIES
        expected_verified = 32
    if (
        viability.get("status") != "pass"
        or viability.get("summary", {}).get("verified_completions") != expected_verified
        or admission.get("status") != "pass"
        or args.seed != expected_seed
        or identity.get("seed") != expected_seed
        or identity.get("row_indices") != list(row_indices)
        or identity.get("policy_microbatch_size") != args.microbatch_size
        or identity.get("jobs", {}).get(args.arm) != int(args.job_id)
    ):
        raise ValueError("PointMaze paired-smoke antecedent or job identity drift")

    dataset = load_from_disk(str(args.data_split_root))
    if set(dataset) != {"train"}:
        raise ValueError("PointMaze frozen train split key changed")
    all_rows = dataset["train"].to_list()
    rows = [all_rows[index] for index in row_indices]
    if tuple(str(row["answer_mode_family"]) for row in rows) != expected_families:
        raise ValueError("PointMaze frozen paired-smoke row order changed")

    random.seed(args.seed + ARM_SEED_OFFSET[args.arm])
    torch.manual_seed(args.seed + ARM_SEED_OFFSET[args.arm])
    torch.cuda.manual_seed_all(args.seed + ARM_SEED_OFFSET[args.arm])
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    action_token_ids = []
    for label in POINT_POLICY_LABELS:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1 or tokenizer.decode(token_ids) != label:
            raise RuntimeError(f"PointMaze label is not one exact token: {label}")
        action_token_ids.append(int(token_ids[0]))
    padding_token_ids = tokenizer.encode(
        POINT_TERMINAL_PADDING_PROMPT,
        add_special_tokens=False,
    )
    if not padding_token_ids or len(padding_token_ids) > args.max_length:
        raise ValueError("PointMaze padding prompt left its token bound")

    initial_model_hash = tree_sha256(args.model)
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    semantic = SemanticShannonTracker(
        coefficient=0.10,
        surprisal_clip=5.0,
        pseudocount=1.0,
        success_conditioned_signed_advantage=True,
        success_conditioned_signed_cap=0.05,
        open_set_inverse_adaptation=True,
        open_set_warmup_steps=64,
        open_set_ema_decay=0.90,
    )
    canonical = OnlineCanonicalBank(
        entropy_alpha=0.0,
        novelty_beta=0.50,
        pseudocount=1.0,
        surprisal_clip=5.0,
        retain_exemplars=False,
        replay_capacity=16,
        global_replay_groups_per_step=0,
    )
    replay_bank = VerifiedInteractiveReplayBank(capacity=REPLAY_CAPACITY)
    metrics = []
    replay_rows = []
    total_fixed_policy_slots = 0
    total_simulator_calls = 0
    total_replay_mode_slots = 0
    total_optimizer_steps = 0

    with PointMazeInteractiveProcess(worker_python=args.worker_python) as worker:
        for update_index, (source_row_index, row) in enumerate(
            zip(row_indices, rows),
            start=1,
        ):
            episodes, policy_slots, rollout = _rollout_group(
                model=model,
                tokenizer=tokenizer,
                worker=worker,
                row=row,
                row_index=source_row_index,
                update_index=update_index,
                arm=args.arm,
                seed=args.seed,
                action_token_ids=action_token_ids,
                padding_token_ids=padding_token_ids,
                max_length=args.max_length,
            )
            rewards = torch.tensor(
                [[episode.task_reward for episode in episodes]],
                dtype=torch.float32,
            )
            task_advantages = drgrpo_task_advantages(rewards).flatten()
            group_prompts = [episode.group_prompt_token_ids for episode in episodes]
            keys = [episode.outcome_key for episode in episodes]
            reward_values = [episode.task_reward for episode in episodes]
            active_mask = [True] * SAMPLES
            semantic_raw, semantic_diag = (
                semantic.score_success_conditioned_signed_advantages_and_update(
                    prompt_token_ids=group_prompts,
                    answer_keys=keys,
                    task_rewards=reward_values,
                    active_mask=active_mask,
                    num_samples=SAMPLES,
                )
            )
            canonical_raw, canonical_diag = canonical.score_and_update(
                prompt_token_ids=group_prompts,
                outcome_keys=keys,
                task_rewards=reward_values,
                active_mask=active_mask,
                num_samples=SAMPLES,
            )
            raw_exploration = torch.tensor(
                [
                    (float(semantic_value) + float(canonical_value)) * 15.0 / 16.0
                    for semantic_value, canonical_value in zip(
                        semantic_raw,
                        canonical_raw,
                    )
                ],
                dtype=torch.float32,
            )
            applied_exploration = (
                raw_exploration
                if args.arm == TREATMENT
                else torch.zeros_like(raw_exploration)
            )
            advantages = add_verified_advantage_outside_centering(
                task_advantages,
                applied_exploration,
            )
            decision_counts = [len(episode.decisions) for episode in episodes]
            for slot in policy_slots:
                if slot["active"]:
                    episode = int(slot["episode"])
                    slot["weight"] = (
                        float(advantages[episode].item())
                        / decision_counts[episode]
                        / SAMPLES
                    )

            replay_bank.observe_group(episodes)
            replay_group = replay_bank.schedule_one_global_round_robin()
            replay_slots, replay_diag = _replay_slots(
                group=replay_group,
                padding_token_ids=padding_token_ids,
                action_token_ids=action_token_ids,
                compute_only=args.arm == CONTROL,
            )
            optimizer.zero_grad(set_to_none=True)
            model.train()
            policy_diagnostics = _backward_fixed_slots(
                model=model,
                tokenizer=tokenizer,
                slots=[*policy_slots, *replay_slots],
                action_token_ids=action_token_ids,
                microbatch_size=args.microbatch_size,
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not math.isfinite(float(grad_norm.detach().item())):
                raise RuntimeError("nonfinite PointMaze gradient norm")
            optimizer.step()
            total_optimizer_steps += 1

            metric = {
                "schema": f"point-maze-interactive-paired-smoke-metric-v{version}",
                "arm": args.arm,
                "seed": args.seed,
                "update": update_index,
                "policy_microbatch_size": args.microbatch_size,
                **rollout,
                **policy_diagnostics,
                **replay_diag,
                "task_advantage_rms": float(torch.sqrt(torch.mean(task_advantages.square())).item()),
                "raw_exploration_advantage_rms": float(torch.sqrt(torch.mean(raw_exploration.square())).item()),
                "applied_exploration_advantage_rms": float(torch.sqrt(torch.mean(applied_exploration.square())).item()),
                "semantic_effective_advantage_rms": float(semantic_diag.effective_advantage_rms),
                "canonical_novelty_advantage_rms": float(canonical_diag.novelty_advantage_rms),
                "canonical_tracked_prompts": float(canonical.tracked_prompt_count),
                "canonical_tracked_outcomes": float(canonical.tracked_outcome_count),
                "canonical_support_at_least_two_prompt_fraction": float(canonical.support_at_least_two_prompt_fraction),
                "replay_bank_tracked_prompts": float(replay_bank.tracked_prompt_count),
                "replay_bank_tracked_outcomes": float(replay_bank.tracked_outcome_count),
                "grad_norm": float(grad_norm.detach().item()),
                "optimizer_step": total_optimizer_steps,
                "action_support_escapes": 0,
                "transition_hash_count": sum(len(episode.decisions) for episode in episodes),
            }
            if any(
                isinstance(value, float) and not math.isfinite(value)
                for value in metric.values()
            ):
                raise RuntimeError("nonfinite PointMaze metric")
            metrics.append(metric)
            _append_jsonl(args.output_metrics, [metric])
            replay_payload = {
                "schema": f"point-maze-interactive-state-replay-v{version}",
                "arm": args.arm,
                "seed": args.seed,
                "update": update_index,
                "source_row_index": source_row_index,
                "family": str(row["answer_mode_family"]),
                "instance_fingerprint": str(row["instance_fingerprint"]),
                "episodes": [episode.state_dict() for episode in episodes],
            }
            replay_rows.append(replay_payload)
            _append_jsonl(args.output_replay, [replay_payload])
            total_fixed_policy_slots += int(rollout["fixed_policy_slots"])
            total_simulator_calls += int(rollout["simulator_step_calls"])
            total_replay_mode_slots += int(replay_diag["replay_mode_slots"])
            print(
                "[point-paired-smoke] "
                f"arm={args.arm} update={update_index}/4 "
                f"verified={rollout['verified_episodes']}/16 "
                f"modes={rollout['distinct_verified_keys']} "
                f"loss={policy_diagnostics['loss']:.6f}",
                flush=True,
            )

    verified_total = sum(int(row["verified_episodes"]) for row in metrics)
    multimode_updates = sum(int(row["distinct_verified_keys"] >= 2) for row in metrics)
    payload = {
        "schema": f"point-maze-interactive-paired-smoke-receipt-v{version}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "arm": args.arm,
        "seed": args.seed,
        "job_id": int(args.job_id),
        "source_hash": args.source_hash,
        "execution_hash": args.execution_hash,
        "initial_model_tree_sha256": initial_model_hash,
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "protocol_sha256": sha256_file(args.protocol),
        "identity_sha256": sha256_file(args.identity),
        "viability_receipt_sha256": sha256_file(args.viability_receipt),
        "admission_audit_sha256": sha256_file(args.admission_audit),
        "warmstart_receipt_sha256": sha256_file(args.warmstart_receipt),
        "data_identity_sha256": sha256_file(args.data_root / "identity.json"),
        "training_rows_sha256": canonical_sha256(rows),
        "metrics_sha256": sha256_file(args.output_metrics),
        "state_replay_sha256": sha256_file(args.output_replay),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0.0,
            "clip_epsilon": 0.2,
            "max_grad_norm": 1.0,
            "updates": total_optimizer_steps,
        },
        "counts": {
            "prompts": len(rows),
            "rollouts": len(rows) * SAMPLES,
            "fixed_policy_slots": total_fixed_policy_slots,
            "simulator_step_calls": total_simulator_calls,
            "replay_group_slots": len(rows),
            "replay_mode_slots": total_replay_mode_slots,
            "optimizer_steps": total_optimizer_steps,
            "verified_episodes": verified_total,
            "multimode_updates": multimode_updates,
        },
        "mechanism": {
            "compute_only_control": args.arm == CONTROL,
            "semantic_coefficient": 0.10,
            "novelty_beta": 0.50,
            "replay_mass_alpha": 0.10,
            "replay_balance_alpha": 0.10,
            "warmup_steps": 64,
            "ema_decay": 0.90,
            "reward_estimator_factor": 15.0 / 16.0,
            "per_rollout_replay_factor": 1.0 / 16.0,
            "gold_support_feedback": False,
            "coefficient_projection": False,
        },
        "information_boundary": {
            "evaluation_rows_loaded": False,
            "development_rows_loaded": False,
            "certified_routes_in_context": False,
            "canonical_keys_in_context": False,
            "verifier_feedback_before_terminal": False,
            "fixed_public_action_mask": True,
        },
        "metrics": metrics,
    }
    atomic_json(args.output_receipt, payload)
    print(
        f"[point-paired-smoke] complete arm={args.arm} "
        f"verified={verified_total}/64 multimode_updates={multimode_updates}/4 "
        f"receipt={args.output_receipt}",
        flush=True,
    )


if __name__ == "__main__":
    main()
