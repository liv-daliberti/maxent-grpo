#!/usr/bin/env python3
"""Train one frozen PointMaze Stage-B arm/seed cell for 12 prompt passes."""

from __future__ import annotations

import argparse
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
OPS = Path(os.environ.get("OAT_ZERO_EXECUTION_ROOT", ROOT / "ops")).resolve()
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

import train_point_maze_interactive_paired_smoke_v1 as base  # noqa: E402
from oat_drgrpo.interactive_episode_objective import (  # noqa: E402
    add_verified_advantage_outside_centering,
    drgrpo_task_advantages,
)
from oat_drgrpo.interactive_episode_replay import (  # noqa: E402
    VerifiedInteractiveReplayBank,
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


CONTROL = base.CONTROL
TREATMENT = base.TREATMENT
ARMS = base.ARMS
SEEDS = (43, 44, 45, 46, 47)
TRAIN_PROMPTS = 8
PASSES = 12
UPDATES = TRAIN_PROMPTS * PASSES
EVAL_INTERVAL = 2
EVAL_DRAWS = 4
EVAL_K = 8
EVAL_PROMPTS = 4
EVAL_TRAJECTORIES = EVAL_PROMPTS * (1 + EVAL_DRAWS * EVAL_K)
EXPECTED_FAMILIES = ("bar7", "block9", "bar9", "asymmetric_block9")
GEOMETRY_SHIFT_FAMILIES = (
    "wide_block9_shift",
    "cross9_shift",
    "upper_offset9_shift",
    "lower_offset9_shift",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def atomic(path: Path, payload: Any) -> None:
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


def append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--qualification-audit", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--output-replay", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=2e-7)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--execution-hash", required=True)
    return parser.parse_args()


def _restricted_mixed_actions(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    action_token_ids: Sequence[int],
    greedy: Sequence[bool],
    request_seeds: Sequence[int],
    max_length: int,
    batch_size: int,
) -> list[int]:
    import torch

    if not (len(prompts) == len(greedy) == len(request_seeds)):
        raise ValueError("PointMaze evaluation selection vectors differ")
    encoded = [tokenizer.encode(text, add_special_tokens=False) for text in prompts]
    if any(not row or len(row) > max_length for row in encoded):
        raise ValueError("PointMaze evaluation prompt left its frozen token bound")
    selected: list[int] = []
    support = torch.tensor(action_token_ids, dtype=torch.long, device="cuda")
    model.eval()
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            batch = encoded[start : start + batch_size]
            maximum = max(len(row) for row in batch)
            input_ids = torch.full(
                (len(batch), maximum), int(tokenizer.pad_token_id),
                dtype=torch.long, device="cuda",
            )
            attention = torch.zeros_like(input_ids)
            for index, row in enumerate(batch):
                input_ids[index, maximum - len(row) :] = torch.tensor(row, device="cuda")
                attention[index, maximum - len(row) :] = 1
            logits = model(
                input_ids=input_ids, attention_mask=attention, logits_to_keep=1
            ).logits[:, -1, :].float().index_select(1, support)
            probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
            for offset, row in enumerate(probabilities):
                global_index = start + offset
                if greedy[global_index]:
                    position = max(range(len(row)), key=row.__getitem__)
                else:
                    draw = random.Random(int(request_seeds[global_index])).random()
                    cumulative = 0.0
                    position = len(row) - 1
                    for candidate, probability in enumerate(row):
                        cumulative += float(probability)
                        if draw <= cumulative:
                            position = candidate
                            break
                selected.append(int(action_token_ids[position]))
    return selected


def evaluate(
    *,
    model: Any,
    tokenizer: Any,
    worker: PointMazeInteractiveProcess,
    rows: Sequence[Mapping[str, Any]],
    arm: str,
    seed: int,
    update: int,
    action_token_ids: Sequence[int],
    max_length: int,
    batch_size: int,
) -> dict[str, Any]:
    specs = []
    for row in rows:
        raw = row["answer"]
        if isinstance(raw, str):
            raw = json.loads(raw)
        spec = parse_maze_action_spec(raw)
        if tuple(spec.action_tokens) != ("N", "NE", "E", "SE", "S", "SW", "W", "NW", "COAST"):
            raise ValueError("PointMaze evaluation action alphabet drift")
        specs.append((raw, spec))

    metadata: list[dict[str, Any]] = []
    sessions = []
    for row_index, (raw, _spec) in enumerate(specs):
        candidates = [(-1, 0)] + [
            (draw, sample) for draw in range(EVAL_DRAWS) for sample in range(EVAL_K)
        ]
        for draw, sample in candidates:
            session_id = f"eval-{arm}-s{seed}-u{update}-r{row_index}-d{draw}-k{sample}"
            sessions.append({"session_id": session_id, "spec": raw})
            metadata.append({
                "session_id": session_id, "row": row_index, "draw": draw,
                "sample": sample, "greedy": draw == -1,
            })
    if len(sessions) != EVAL_TRAJECTORIES:
        raise RuntimeError("PointMaze evaluation trajectory count changed")
    reset = worker.reset_batch(sessions)
    observations = {item["session_id"]: item for item in reset}
    active = [True] * len(sessions)
    outcomes: list[str | None] = [None] * len(sessions)
    simulator_calls = 0
    action_positions = {int(token): index for index, token in enumerate(action_token_ids)}
    for decision_round in range(1, base.HORIZON + 1):
        prompts = []
        request_seeds = []
        greedy_flags = []
        for index, item in enumerate(metadata):
            if active[index]:
                row_index = int(item["row"])
                prompts.append(render_point_policy_prompt_v3(
                    str(rows[row_index]["problem"]), observations[item["session_id"]],
                    tuple(specs[row_index][1].action_tokens), (),
                ))
            else:
                prompts.append(POINT_TERMINAL_PADDING_PROMPT)
            greedy_flags.append(bool(item["greedy"]))
            request_seeds.append(
                1_500_000_000 + seed * 10_000_000 + update * 100_000
                + int(item["row"]) * 10_000 + max(int(item["draw"]), 0) * 1_000
                + int(item["sample"]) * 100 + decision_round
            )
        selected = _restricted_mixed_actions(
            model=model, tokenizer=tokenizer, prompts=prompts,
            action_token_ids=action_token_ids, greedy=greedy_flags,
            request_seeds=request_seeds, max_length=max_length, batch_size=batch_size,
        )
        requests = []
        active_indices = []
        for index, item in enumerate(metadata):
            if not active[index]:
                continue
            row_index = int(item["row"])
            action = specs[row_index][1].action_tokens[action_positions[selected[index]]]
            requests.append({"session_id": item["session_id"], "action": action})
            active_indices.append(index)
        transitions = worker.step_batch(requests) if requests else []
        if len(transitions) != len(active_indices):
            raise RuntimeError("PointMaze evaluation worker batch changed")
        simulator_calls += len(transitions)
        for index, transition in zip(active_indices, transitions):
            session_id = metadata[index]["session_id"]
            if transition["session_id"] != session_id:
                raise RuntimeError("PointMaze evaluation worker reordered sessions")
            observations[session_id] = transition
            if transition["done"]:
                active[index] = False
                key = transition.get("canonical_key")
                outcomes[index] = None if key is None else str(key)
    if any(active):
        raise RuntimeError("PointMaze evaluation exceeded the frozen horizon")

    by_row: dict[int, dict[int, list[str | None]]] = {
        row: {-1: [], **{draw: [] for draw in range(EVAL_DRAWS)}}
        for row in range(EVAL_PROMPTS)
    }
    for item, outcome in zip(metadata, outcomes):
        by_row[int(item["row"])][int(item["draw"])].append(outcome)
    result: dict[str, Any] = {
        "schema": "point-maze-stage-b-evaluation-v1", "split": "multi_answer",
        "arm": arm, "seed": seed, "learning_round": update,
        "training_passes": update / TRAIN_PROMPTS,
        "evaluation_draw_count": EVAL_DRAWS,
        "evaluation_trajectory_count": EVAL_TRAJECTORIES,
        "evaluation_simulator_step_calls": simulator_calls,
    }
    greedy_values = []
    mean_values = []
    pass_values = []
    distinct_values = []
    for row_index, family in enumerate(
        str(row["answer_mode_family"]) for row in rows
    ):
        greedy_rows = by_row[row_index][-1]
        if len(greedy_rows) != 1:
            raise RuntimeError("PointMaze greedy evaluation count changed")
        greedy_value = float(greedy_rows[0] is not None)
        greedy_values.append(greedy_value)
        result[f"eval/{family}/greedy"] = greedy_value
        for draw in range(EVAL_DRAWS):
            draw_rows = by_row[row_index][draw]
            if len(draw_rows) != EVAL_K:
                raise RuntimeError("PointMaze K=8 evaluation count changed")
            mean_value = sum(value is not None for value in draw_rows) / EVAL_K
            pass_value = float(any(value is not None for value in draw_rows))
            distinct_value = float(len({value for value in draw_rows if value is not None}))
            mean_values.append(mean_value)
            pass_values.append(pass_value)
            distinct_values.append(distinct_value)
            result[f"eval/{family}/mean8_draw_{draw}"] = mean_value
            result[f"eval/{family}/pass8_draw_{draw}"] = pass_value
            result[f"eval/{family}/distinct8_draw_{draw}"] = distinct_value
    result["greedy"] = sum(greedy_values) / len(greedy_values)
    result["mean8"] = sum(mean_values) / len(mean_values)
    result["pass8"] = sum(pass_values) / len(pass_values)
    result["distinct8"] = sum(distinct_values) / len(distinct_values)
    for draw in range(EVAL_DRAWS):
        selection = [row * EVAL_DRAWS + draw for row in range(EVAL_PROMPTS)]
        result[f"mean8_draw_{draw}"] = sum(mean_values[index] for index in selection) / EVAL_PROMPTS
        result[f"pass8_draw_{draw}"] = sum(pass_values[index] for index in selection) / EVAL_PROMPTS
        result[f"distinct8_draw_{draw}"] = sum(distinct_values[index] for index in selection) / EVAL_PROMPTS
    if any(isinstance(value, float) and not math.isfinite(value) for value in result.values()):
        raise RuntimeError("nonfinite PointMaze evaluation metric")
    return result


def main() -> None:
    args = parse_args()
    if args.seed not in SEEDS or args.learning_rate != 2e-7 or args.max_length != 1536:
        raise ValueError("PointMaze Stage-B frozen cell contract drift")
    if args.microbatch_size != 16 or args.evaluation_batch_size != 16:
        raise ValueError("PointMaze Stage-B batch contract drift")
    for fresh in (args.output_receipt, args.output_metrics, args.output_replay):
        if fresh.exists():
            raise FileExistsError(f"fresh PointMaze Stage-B output required: {fresh}")
    for required in (
        args.model / "config.json", args.data_root / "identity.json",
        args.train_root / "dataset_dict.json", args.evaluation_root / "dataset_dict.json",
        args.worker_python, args.protocol, args.qualification_audit, args.identity,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("PointMaze Stage B requires a BF16 GPU")
    qualification = json.loads(args.qualification_audit.read_text())
    identity = json.loads(args.identity.read_text())
    geometry_shift = (
        identity.get("schema")
        == "point-maze-geometry-shift-stage-b-05b-12pass-identity-v2"
    )
    expected_decision = (
        "eligible_for_ten_point_maze_geometry_shift_replacement_jobs"
        if geometry_shift
        else "eligible_for_ten_point_maze_stage_b_jobs"
    )
    expected_families = GEOMETRY_SHIFT_FAMILIES if geometry_shift else EXPECTED_FAMILIES
    label = f"{args.arm}/s{args.seed}"
    if (
        qualification.get("status") != "pass"
        or qualification.get("decision") != expected_decision
        or identity.get("jobs", {}).get(label) != int(args.job_id)
    ):
        raise ValueError("PointMaze Stage-B qualification or job identity drift")

    train_dict = load_from_disk(str(args.train_root))
    evaluation_dict = load_from_disk(str(args.evaluation_root))
    if set(train_dict) != {"train"} or set(evaluation_dict) != {"multi_answer"}:
        raise ValueError("PointMaze Stage-B split keys changed")
    train_rows = train_dict["train"].to_list()
    evaluation_rows = evaluation_dict["multi_answer"].to_list()
    if len(train_rows) != TRAIN_PROMPTS or len(evaluation_rows) != EVAL_PROMPTS:
        raise ValueError("PointMaze Stage-B split cardinality changed")
    if tuple(str(row["answer_mode_family"]) for row in evaluation_rows) != expected_families:
        raise ValueError("PointMaze Stage-B evaluation family order changed")

    random.seed(args.seed + base.ARM_SEED_OFFSET[args.arm])
    torch.manual_seed(args.seed + base.ARM_SEED_OFFSET[args.arm])
    torch.cuda.manual_seed_all(args.seed + base.ARM_SEED_OFFSET[args.arm])
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    action_token_ids = []
    for label_token in POINT_POLICY_LABELS:
        token_ids = tokenizer.encode(label_token, add_special_tokens=False)
        if len(token_ids) != 1 or tokenizer.decode(token_ids) != label_token:
            raise RuntimeError(f"PointMaze label is not one exact token: {label_token}")
        action_token_ids.append(int(token_ids[0]))
    padding_token_ids = tokenizer.encode(POINT_TERMINAL_PADDING_PROMPT, add_special_tokens=False)
    initial_model_hash = base.tree_sha256(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), betas=(0.9, 0.999),
        eps=1e-8, weight_decay=0.0,
    )
    semantic = SemanticShannonTracker(
        coefficient=0.10, surprisal_clip=5.0, pseudocount=1.0,
        success_conditioned_signed_advantage=True, success_conditioned_signed_cap=0.05,
        open_set_inverse_adaptation=True, open_set_warmup_steps=64, open_set_ema_decay=0.90,
    )
    canonical = OnlineCanonicalBank(
        entropy_alpha=0.0, novelty_beta=0.50, pseudocount=1.0, surprisal_clip=5.0,
        retain_exemplars=False, replay_capacity=16, global_replay_groups_per_step=0,
    )
    replay_bank = VerifiedInteractiveReplayBank(capacity=base.REPLAY_CAPACITY)
    training_metrics: list[dict[str, Any]] = []
    evaluation_metrics: list[dict[str, Any]] = []
    total_simulator_calls = 0

    with PointMazeInteractiveProcess(worker_python=args.worker_python) as worker:
        initial_eval = evaluate(
            model=model, tokenizer=tokenizer, worker=worker, rows=evaluation_rows,
            arm=args.arm, seed=args.seed, update=0, action_token_ids=action_token_ids,
            max_length=args.max_length, batch_size=args.evaluation_batch_size,
        )
        evaluation_metrics.append(initial_eval)
        append_jsonl(args.output_metrics, [initial_eval])
        for update_index in range(1, UPDATES + 1):
            row_index = (update_index - 1) % TRAIN_PROMPTS
            row = train_rows[row_index]
            episodes, policy_slots, rollout = base._rollout_group(
                model=model, tokenizer=tokenizer, worker=worker, row=row,
                row_index=row_index, update_index=update_index, arm=args.arm,
                seed=args.seed, action_token_ids=action_token_ids,
                padding_token_ids=padding_token_ids, max_length=args.max_length,
            )
            rewards = torch.tensor([[episode.task_reward for episode in episodes]], dtype=torch.float32)
            task_advantages = drgrpo_task_advantages(rewards).flatten()
            group_prompts = [episode.group_prompt_token_ids for episode in episodes]
            keys = [episode.outcome_key for episode in episodes]
            reward_values = [episode.task_reward for episode in episodes]
            active_mask = [True] * base.SAMPLES
            semantic_raw, semantic_diag = semantic.score_success_conditioned_signed_advantages_and_update(
                prompt_token_ids=group_prompts, answer_keys=keys, task_rewards=reward_values,
                active_mask=active_mask, num_samples=base.SAMPLES,
            )
            canonical_raw, canonical_diag = canonical.score_and_update(
                prompt_token_ids=group_prompts, outcome_keys=keys, task_rewards=reward_values,
                active_mask=active_mask, num_samples=base.SAMPLES,
            )
            raw_exploration = torch.tensor([
                (float(semantic_value) + float(canonical_value)) * 15.0 / 16.0
                for semantic_value, canonical_value in zip(semantic_raw, canonical_raw)
            ], dtype=torch.float32)
            applied_exploration = raw_exploration if args.arm == TREATMENT else torch.zeros_like(raw_exploration)
            advantages = add_verified_advantage_outside_centering(task_advantages, applied_exploration)
            decision_counts = [len(episode.decisions) for episode in episodes]
            for slot in policy_slots:
                if slot["active"]:
                    episode_index = int(slot["episode"])
                    slot["weight"] = float(advantages[episode_index].item()) / decision_counts[episode_index] / base.SAMPLES
            replay_bank.observe_group(episodes)
            replay_group = replay_bank.schedule_one_global_round_robin()
            replay_slots, replay_diag = base._replay_slots(
                group=replay_group, padding_token_ids=padding_token_ids,
                action_token_ids=action_token_ids, compute_only=args.arm == CONTROL,
            )
            optimizer.zero_grad(set_to_none=True)
            model.train()
            policy_diag = base._backward_fixed_slots(
                model=model, tokenizer=tokenizer, slots=[*policy_slots, *replay_slots],
                action_token_ids=action_token_ids, microbatch_size=args.microbatch_size,
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not math.isfinite(float(grad_norm.detach().item())):
                raise RuntimeError("nonfinite PointMaze Stage-B gradient")
            optimizer.step()
            metric = {
                "schema": "point-maze-stage-b-training-metric-v1", "split": "train",
                "arm": args.arm, "seed": args.seed, "learning_round": update_index,
                "training_passes": update_index / TRAIN_PROMPTS,
                "policy_microbatch_size": args.microbatch_size,
                **rollout, **policy_diag, **replay_diag,
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
                "grad_norm": float(grad_norm.detach().item()), "optimizer_step": update_index,
                "action_support_escapes": 0,
            }
            if any(isinstance(value, float) and not math.isfinite(value) for value in metric.values()):
                raise RuntimeError("nonfinite PointMaze Stage-B training metric")
            training_metrics.append(metric)
            append_jsonl(args.output_metrics, [metric])
            append_jsonl(args.output_replay, [{
                "schema": "point-maze-stage-b-state-replay-v1", "arm": args.arm,
                "seed": args.seed, "update": update_index, "source_row_index": row_index,
                "family": str(row["answer_mode_family"]),
                "instance_fingerprint": str(row["instance_fingerprint"]),
                "episodes": [episode.state_dict() for episode in episodes],
            }])
            total_simulator_calls += int(rollout["simulator_step_calls"])
            if update_index % EVAL_INTERVAL == 0:
                evaluation = evaluate(
                    model=model, tokenizer=tokenizer, worker=worker, rows=evaluation_rows,
                    arm=args.arm, seed=args.seed, update=update_index,
                    action_token_ids=action_token_ids, max_length=args.max_length,
                    batch_size=args.evaluation_batch_size,
                )
                evaluation_metrics.append(evaluation)
                append_jsonl(args.output_metrics, [evaluation])
            print(
                f"[point-stage-b] arm={args.arm} seed={args.seed} "
                f"update={update_index}/{UPDATES} verified={rollout['verified_episodes']}/16 "
                f"modes={rollout['distinct_verified_keys']}", flush=True,
            )

    payload = {
        "schema": "point-maze-stage-b-05b-12pass-receipt-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(), "status": "complete",
        "arm": args.arm, "seed": args.seed, "job_id": int(args.job_id),
        "source_hash": args.source_hash, "execution_hash": args.execution_hash,
        "initial_model_tree_sha256": initial_model_hash,
        "model_config_sha256": sha(args.model / "config.json"),
        "protocol_sha256": sha(args.protocol), "identity_sha256": sha(args.identity),
        "qualification_audit_sha256": sha(args.qualification_audit),
        "data_identity_sha256": sha(args.data_root / "identity.json"),
        "train_rows_sha256": canonical_sha(train_rows),
        "evaluation_rows_sha256": canonical_sha(evaluation_rows),
        "metrics_sha256": sha(args.output_metrics), "state_replay_sha256": sha(args.output_replay),
        "counts": {
            "prompt_passes": PASSES, "optimizer_updates": len(training_metrics),
            "training_rollouts": UPDATES * base.SAMPLES,
            "training_fixed_policy_slots": UPDATES * base.SAMPLES * base.HORIZON,
            "training_fixed_replay_decision_slots": UPDATES * base.REPLAY_CAPACITY * base.HORIZON,
            "training_simulator_step_calls": total_simulator_calls,
            "evaluation_coordinates": len(evaluation_metrics),
            "evaluation_trajectories_per_coordinate": EVAL_TRAJECTORIES,
        },
        "mechanism": {
            "compute_only_control": args.arm == CONTROL, "semantic_coefficient": 0.10,
            "novelty_beta": 0.50, "replay_mass_alpha": 0.10,
            "replay_balance_alpha": 0.10, "warmup_steps": 64,
            "gold_support_feedback": False, "coefficient_projection": False,
        },
        "information_boundary": {
            "certified_routes_in_context": False, "canonical_keys_in_context": False,
            "evaluation_feedback_to_training": False, "development_rows_loaded": False,
            "fixed_public_action_mask": True,
        },
    }
    atomic(args.output_receipt, payload)
    print(f"[point-stage-b] complete arm={args.arm} seed={args.seed}", flush=True)


if __name__ == "__main__":
    main()
