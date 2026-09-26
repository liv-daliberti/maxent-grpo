#!/usr/bin/env python3
"""Run a configurable Dr.GRPO/xDr pilot on sequential PointMaze waypoints."""

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


ROOT = Path(
    os.environ.get("OAT_ZERO_REPO_ROOT", Path(__file__).resolve().parents[1])
).resolve()
SRC = Path(os.environ.get("OAT_ZERO_SOURCE_ROOT", ROOT / "src")).resolve()
OPS = Path(os.environ.get("OAT_ZERO_EXECUTION_ROOT", ROOT / "ops")).resolve()
for path in (SRC, OPS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_point_maze_interactive_paired_smoke_v1 as base  # noqa: E402
from oat_drgrpo.interactive_episode_objective import (  # noqa: E402
    add_verified_advantage_outside_centering,
    drgrpo_task_advantages,
)
from oat_drgrpo.interactive_episode_replay import (  # noqa: E402
    InteractiveDecisionRecord,
    InteractiveEpisodeRecord,
    VerifiedInteractiveReplayBank,
)
from oat_drgrpo.online_canonical_bank import OnlineCanonicalBank  # noqa: E402
from oat_drgrpo.point_maze_waypoint import (  # noqa: E402
    parse_point_waypoint_spec,
)
from oat_drgrpo.point_maze_waypoint_policy import (  # noqa: E402
    POINT_WAYPOINT_LABEL_TO_ACTION,
    POINT_WAYPOINT_LABELS,
    POINT_WAYPOINT_TERMINAL_PADDING_PROMPT,
    point_waypoint_allowed_labels,
    point_waypoint_transition_sha256,
    render_point_waypoint_prompt,
)
from oat_drgrpo.point_maze_waypoint_process import (  # noqa: E402
    PointMazeWaypointProcess,
)
from oat_drgrpo.semantic_shannon import SemanticShannonTracker  # noqa: E402


CONTROL = "grpo"
CURRENT_XDR = "verified_first_global_replay_canonical"
DELAYED_SINGLETON_XDR = "verified_first_delayed_singleton_replay_canonical"
ARMS = (CONTROL, CURRENT_XDR, DELAYED_SINGLETON_XDR)
ARM_SEED_OFFSET = {
    CONTROL: 0,
    CURRENT_XDR: 0,
    DELAYED_SINGLETON_XDR: 0,
}
SAMPLES = 16
FIXED_HORIZON = 64
REPLAY_CAPACITY = 16
EVAL_K = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
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
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--output-replay", type=Path, required=True)
    parser.add_argument("--output-model", type=Path)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--evaluation-interval", type=int, default=64)
    parser.add_argument("--evaluation-prompts", type=int, default=32)
    parser.add_argument(
        "--evaluation-split",
        choices=("dev", "eval"),
        default="dev",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-7)
    parser.add_argument("--microbatch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1536)
    return parser.parse_args()


def _raw_spec(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row["answer"]
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("waypoint row answer must be a JSON object")
    return value


def _run_episodes(
    *,
    model: Any,
    tokenizer: Any,
    worker: PointMazeWaypointProcess,
    row: Mapping[str, Any],
    sample_count: int,
    session_prefix: str,
    seed_base: int,
    label_token_ids: Mapping[str, int],
    padding_token_ids: Sequence[int],
    max_length: int,
    record_decisions: bool,
    prompt_format: str = "qwen_chatml",
) -> tuple[
    list[InteractiveEpisodeRecord],
    list[dict[str, Any]],
    list[str | None],
    dict[str, Any],
]:
    raw_spec = _raw_spec(row)
    spec = parse_point_waypoint_spec(raw_spec)
    if spec.max_actions > FIXED_HORIZON:
        raise ValueError("waypoint spec exceeds the fixed policy horizon")
    global_action_ids = tuple(
        int(label_token_ids[label]) for label in POINT_WAYPOINT_LABELS
    )
    token_to_label = {int(token): label for label, token in label_token_ids.items()}
    group_prompt_ids = tuple(
        tokenizer.encode(str(row["problem"]), add_special_tokens=False)
    )
    sessions = [
        {
            "session_id": f"{session_prefix}-e{episode}",
            "spec": raw_spec,
        }
        for episode in range(sample_count)
    ]
    reset = worker.reset_batch(sessions)
    observations = {item["session_id"]: item for item in reset}
    active = [True] * sample_count
    decisions: list[list[InteractiveDecisionRecord]] = [[] for _ in range(sample_count)]
    final: list[dict[str, Any] | None] = [None] * sample_count
    fixed_slots: list[dict[str, Any]] = []
    simulator_calls = 0
    model.eval()

    for decision_round in range(1, FIXED_HORIZON + 1):
        prompts = []
        supports = []
        request_seeds = []
        for episode, session in enumerate(sessions):
            if active[episode]:
                state = observations[session["session_id"]]
                prompts.append(
                    render_point_waypoint_prompt(
                        str(row["problem"]),
                        state,
                        prompt_format=prompt_format,
                    )
                )
                labels = point_waypoint_allowed_labels(state["allowed_actions"])
                supports.append(tuple(int(label_token_ids[label]) for label in labels))
            else:
                prompts.append(POINT_WAYPOINT_TERMINAL_PADDING_PROMPT)
                supports.append(global_action_ids)
            request_seeds.append(int(seed_base) + episode * 1_000 + decision_round)
        prompt_ids, selected_ids, behavior_rows = base._restricted_sample(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            action_token_ids=global_action_ids,
            request_seeds=request_seeds,
            max_length=max_length,
            allowed_token_ids_by_prompt=supports,
        )
        requests = []
        active_indices = []
        selected_actions: dict[int, str] = {}
        for episode, session in enumerate(sessions):
            if active[episode]:
                label = token_to_label[int(selected_ids[episode])]
                action = POINT_WAYPOINT_LABEL_TO_ACTION[label]
                if action not in observations[session["session_id"]]["allowed_actions"]:
                    raise RuntimeError(
                        "sampled waypoint action escaped its legal support"
                    )
                requests.append({"session_id": session["session_id"], "action": action})
                active_indices.append(episode)
                selected_actions[episode] = action
            if record_decisions:
                support = tuple(int(token) for token in supports[episode])
                position = support.index(int(selected_ids[episode]))
                fixed_slots.append(
                    {
                        "kind": "on_policy",
                        "prompt_token_ids": tuple(prompt_ids[episode]),
                        "allowed_token_ids": support,
                        "selected_token_id": int(selected_ids[episode]),
                        "behavior_logprob": float(behavior_rows[episode][position]),
                        "episode": episode,
                        "active": bool(active[episode]),
                        "weight": 0.0,
                    }
                )
        transitions = worker.step_batch(requests) if requests else []
        if len(transitions) != len(active_indices):
            raise RuntimeError("waypoint worker returned the wrong batch size")
        simulator_calls += len(transitions)
        for episode, transition in zip(active_indices, transitions):
            session_id = sessions[episode]["session_id"]
            if transition["session_id"] != session_id:
                raise RuntimeError("waypoint worker reordered sessions")
            before = observations[session_id]
            if record_decisions:
                support = tuple(int(token) for token in supports[episode])
                decisions[episode].append(
                    InteractiveDecisionRecord(
                        prompt_token_ids=tuple(prompt_ids[episode]),
                        allowed_token_ids=support,
                        selected_token_id=int(selected_ids[episode]),
                        behavior_logprobs=tuple(behavior_rows[episode]),
                        transition_sha256=point_waypoint_transition_sha256(
                            before=before,
                            action=selected_actions[episode],
                            after=transition,
                        ),
                    )
                )
            observations[session_id] = transition
            if transition["done"]:
                active[episode] = False
                final[episode] = transition
    if any(active) or any(item is None for item in final):
        raise RuntimeError("waypoint episodes did not terminate inside fixed horizon")
    if record_decisions and len(fixed_slots) != sample_count * FIXED_HORIZON:
        raise RuntimeError("waypoint fixed policy-slot budget changed")

    outcomes = [
        (
            None
            if transition is None or transition.get("canonical_key") is None
            else str(transition["canonical_key"])
        )
        for transition in final
    ]
    episodes = []
    if record_decisions:
        for episode, outcome in enumerate(outcomes):
            episodes.append(
                InteractiveEpisodeRecord(
                    group_prompt_token_ids=group_prompt_ids,
                    outcome_key=outcome,
                    task_reward=float(outcome is not None),
                    decisions=tuple(decisions[episode]),
                )
            )
    return (
        episodes,
        fixed_slots,
        outcomes,
        {
            "family": str(row["answer_mode_family"]),
            "map_id": spec.base_spec.map_id,
            "verified_episodes": sum(outcome is not None for outcome in outcomes),
            "distinct_verified_keys": len(
                {outcome for outcome in outcomes if outcome is not None}
            ),
            "active_decisions": sum(len(record) for record in decisions)
            if record_decisions
            else 0,
            "simulator_step_calls": simulator_calls,
        },
    )


def _evaluate(
    *,
    model: Any,
    tokenizer: Any,
    worker: PointMazeWaypointProcess,
    rows: Sequence[Mapping[str, Any]],
    split: str,
    arm: str,
    seed: int,
    update: int,
    label_token_ids: Mapping[str, int],
    padding_token_ids: Sequence[int],
    max_length: int,
    prompt_format: str = "qwen_chatml",
) -> dict[str, Any]:
    per_map = []
    total_calls = 0
    for row_index, row in enumerate(rows):
        _episodes, _slots, outcomes, diagnostics = _run_episodes(
            model=model,
            tokenizer=tokenizer,
            worker=worker,
            row=row,
            sample_count=EVAL_K,
            session_prefix=(f"eval-{arm}-s{seed}-u{update}-r{row_index}"),
            seed_base=(
                1_500_000_000 + seed * 10_000_000 + update * 100_000 + row_index * 1_000
            ),
            label_token_ids=label_token_ids,
            padding_token_ids=padding_token_ids,
            max_length=max_length,
            record_decisions=False,
            prompt_format=prompt_format,
        )
        successes = sum(outcome is not None for outcome in outcomes)
        distinct = len({outcome for outcome in outcomes if outcome is not None})
        total_calls += int(diagnostics["simulator_step_calls"])
        per_map.append(
            {
                "map_id": diagnostics["map_id"],
                "family": diagnostics["family"],
                "mean8": successes / EVAL_K,
                "pass8": float(successes > 0),
                "distinct8": float(distinct),
                "modes_per_success": (distinct / successes if successes else 0.0),
            }
        )
    return {
        "schema": "point-maze-waypoint-pilot-evaluation-v1",
        "split": split,
        "arm": arm,
        "seed": seed,
        "learning_round": update,
        "evaluation_prompt_count": len(rows),
        "evaluation_trajectory_count": len(rows) * EVAL_K,
        "evaluation_simulator_step_calls": total_calls,
        "mean8": sum(row["mean8"] for row in per_map) / len(per_map),
        "pass8": sum(row["pass8"] for row in per_map) / len(per_map),
        "distinct8": sum(row["distinct8"] for row in per_map) / len(per_map),
        "modes_per_success": (
            sum(row["modes_per_success"] for row in per_map) / len(per_map)
        ),
        "per_map": per_map,
    }


def main() -> None:
    args = parse_args()
    if (
        (args.passes <= 0 and not args.evaluation_only)
        or args.evaluation_interval <= 0
        or args.evaluation_prompts <= 0
        or args.microbatch_size <= 0
    ):
        raise ValueError("waypoint pilot counts must be positive")
    if args.evaluation_only and args.output_model is not None:
        raise ValueError("evaluation-only runs do not write a model")
    if not args.evaluation_only and args.output_model is None:
        raise ValueError("training runs require --output-model")
    outputs = [
        args.output_receipt,
        args.output_metrics,
        args.output_replay,
    ]
    if args.output_model is not None:
        outputs.append(args.output_model)
    for output in outputs:
        if output.exists():
            raise FileExistsError(f"fresh waypoint pilot output required: {output}")
    args.output_replay.parent.mkdir(parents=True, exist_ok=True)
    args.output_replay.touch()

    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("waypoint pilot requires a BF16-capable GPU")
    dataset = load_from_disk(str(args.data_root))
    if set(dataset) != {"train", "dev", "eval"}:
        raise ValueError("waypoint pilot data splits changed")
    train_rows = dataset["train"].to_list()
    evaluation_dataset = dataset[args.evaluation_split]
    evaluation_rows = evaluation_dataset.select(
        range(min(args.evaluation_prompts, len(evaluation_dataset)))
    ).to_list()
    if not train_rows or not evaluation_rows:
        raise ValueError("waypoint pilot needs train and evaluation rows")
    for row in [*train_rows, *evaluation_rows]:
        parse_point_waypoint_spec(_raw_spec(row))

    random.seed(args.seed + ARM_SEED_OFFSET[args.arm])
    torch.manual_seed(args.seed + ARM_SEED_OFFSET[args.arm])
    torch.cuda.manual_seed_all(args.seed + ARM_SEED_OFFSET[args.arm])
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    label_token_ids = {}
    for label in POINT_WAYPOINT_LABELS:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1 or tokenizer.decode(token_ids) != label:
            raise RuntimeError(f"waypoint label {label!r} is not one token")
        label_token_ids[label] = int(token_ids[0])
    if len(set(label_token_ids.values())) != len(label_token_ids):
        raise RuntimeError("waypoint label token IDs are not unique")
    global_action_ids = tuple(label_token_ids[label] for label in POINT_WAYPOINT_LABELS)
    padding_token_ids = tokenizer.encode(
        POINT_WAYPOINT_TERMINAL_PADDING_PROMPT,
        add_special_tokens=False,
    )

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
    )
    canonical = OnlineCanonicalBank(
        entropy_alpha=0.0,
        pseudocount=1.0,
        surprisal_clip=5.0,
        retain_exemplars=False,
        replay_capacity=REPLAY_CAPACITY,
        global_replay_groups_per_step=0,
    )
    replay_bank = VerifiedInteractiveReplayBank(capacity=REPLAY_CAPACITY)
    updates = 0 if args.evaluation_only else len(train_rows) * args.passes
    evaluation_metrics = []
    training_metrics = []

    with PointMazeWaypointProcess(worker_python=args.worker_python) as worker:
        initial = _evaluate(
            model=model,
            tokenizer=tokenizer,
            worker=worker,
            split=args.evaluation_split,
            rows=evaluation_rows,
            arm=args.arm,
            seed=args.seed,
            update=0,
            label_token_ids=label_token_ids,
            padding_token_ids=padding_token_ids,
            max_length=args.max_length,
        )
        evaluation_metrics.append(initial)
        _append_jsonl(args.output_metrics, [initial])
        for update in range(1, updates + 1):
            row_index = (update - 1) % len(train_rows)
            row = train_rows[row_index]
            episodes, policy_slots, _outcomes, rollout = _run_episodes(
                model=model,
                tokenizer=tokenizer,
                worker=worker,
                row=row,
                sample_count=SAMPLES,
                session_prefix=f"train-{args.arm}-s{args.seed}-u{update}",
                seed_base=(args.seed + ARM_SEED_OFFSET[args.arm] + update * 1_000_000),
                label_token_ids=label_token_ids,
                padding_token_ids=padding_token_ids,
                max_length=args.max_length,
                record_decisions=True,
            )
            rewards = torch.tensor(
                [[episode.task_reward for episode in episodes]],
                dtype=torch.float32,
            )
            task_advantages = drgrpo_task_advantages(rewards).flatten()
            prompts = [episode.group_prompt_token_ids for episode in episodes]
            keys = [episode.outcome_key for episode in episodes]
            reward_values = [episode.task_reward for episode in episodes]
            active_mask = [True] * SAMPLES
            semantic_raw, semantic_diag = (
                semantic.score_success_conditioned_signed_advantages_and_update(
                    prompt_token_ids=prompts,
                    answer_keys=keys,
                    task_rewards=reward_values,
                    active_mask=active_mask,
                    num_samples=SAMPLES,
                )
            )
            canonical_raw, canonical_diag = canonical.score_and_update(
                prompt_token_ids=prompts,
                outcome_keys=keys,
                task_rewards=reward_values,
                active_mask=active_mask,
                num_samples=SAMPLES,
            )
            raw_exploration = torch.tensor(
                [
                    (float(first) + float(second)) * 15.0 / 16.0
                    for first, second in zip(semantic_raw, canonical_raw)
                ],
                dtype=torch.float32,
            )
            applied_exploration = (
                torch.zeros_like(raw_exploration)
                if args.arm == CONTROL
                else raw_exploration
            )
            advantages = add_verified_advantage_outside_centering(
                task_advantages, applied_exploration
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
            replay_slots, replay_diag = base._replay_slots(
                group=replay_group,
                padding_token_ids=padding_token_ids,
                action_token_ids=global_action_ids,
                compute_only=args.arm == CONTROL,
                allow_singleton_mass=(args.arm != DELAYED_SINGLETON_XDR),
                horizon=FIXED_HORIZON,
                replay_capacity=REPLAY_CAPACITY,
                samples=SAMPLES,
            )
            optimizer.zero_grad(set_to_none=True)
            model.train()
            policy_diag = base._backward_fixed_slots(
                model=model,
                tokenizer=tokenizer,
                slots=[*policy_slots, *replay_slots],
                action_token_ids=global_action_ids,
                microbatch_size=args.microbatch_size,
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not math.isfinite(float(grad_norm.detach().item())):
                raise RuntimeError("nonfinite waypoint pilot gradient")
            optimizer.step()
            metric = {
                "schema": "point-maze-waypoint-pilot-training-v1",
                "split": "train",
                "arm": args.arm,
                "seed": args.seed,
                "learning_round": update,
                "training_passes": update / len(train_rows),
                "row_index": row_index,
                **rollout,
                **policy_diag,
                **replay_diag,
                "task_advantage_rms": float(
                    torch.sqrt(torch.mean(task_advantages.square())).item()
                ),
                "raw_exploration_advantage_rms": float(
                    torch.sqrt(torch.mean(raw_exploration.square())).item()
                ),
                "applied_exploration_advantage_rms": float(
                    torch.sqrt(torch.mean(applied_exploration.square())).item()
                ),
                "semantic_effective_advantage_rms": float(
                    semantic_diag.effective_advantage_rms
                ),
                "canonical_tracked_prompts": float(canonical.tracked_prompt_count),
                "canonical_tracked_outcomes": float(canonical.tracked_outcome_count),
                "replay_bank_tracked_prompts": float(replay_bank.tracked_prompt_count),
                "replay_bank_tracked_outcomes": float(
                    replay_bank.tracked_outcome_count
                ),
                "grad_norm": float(grad_norm.detach().item()),
                "action_support_escapes": 0,
            }
            training_metrics.append(metric)
            _append_jsonl(args.output_metrics, [metric])
            _append_jsonl(
                args.output_replay,
                [
                    {
                        "schema": "point-maze-waypoint-state-replay-v1",
                        "arm": args.arm,
                        "seed": args.seed,
                        "update": update,
                        "source_row_index": row_index,
                        "map_id": rollout["map_id"],
                        "episodes": [episode.state_dict() for episode in episodes],
                    }
                ],
            )
            if update % args.evaluation_interval == 0 or update == updates:
                evaluation = _evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    split=args.evaluation_split,
                    worker=worker,
                    rows=evaluation_rows,
                    arm=args.arm,
                    seed=args.seed,
                    update=update,
                    label_token_ids=label_token_ids,
                    padding_token_ids=padding_token_ids,
                    max_length=args.max_length,
                )
                evaluation_metrics.append(evaluation)
                _append_jsonl(args.output_metrics, [evaluation])
            print(
                f"[waypoint-pilot] arm={args.arm} update={update}/{updates} "
                f"verified={rollout['verified_episodes']}/{SAMPLES} "
                f"modes={rollout['distinct_verified_keys']}",
                flush=True,
            )

    output_model_sha256 = None
    if args.output_model is not None:
        args.output_model.parent.mkdir(parents=True, exist_ok=True)
        model.eval()
        model.save_pretrained(
            args.output_model, safe_serialization=True, max_shard_size="2GB"
        )
        tokenizer.save_pretrained(args.output_model)
        output_model_sha256 = _tree_sha256(args.output_model)
    receipt = {
        "schema": "point-maze-waypoint-pilot-receipt-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "arm": args.arm,
        "seed": args.seed,
        "model": str(args.model.resolve()),
        "output_model": (
            None if args.output_model is None else str(args.output_model.resolve())
        ),
        "output_model_tree_sha256": output_model_sha256,
        "data_root": str(args.data_root.resolve()),
        "data_identity_sha256": _sha256(args.data_root / "identity.json"),
        "metrics_sha256": _sha256(args.output_metrics),
        "state_replay_sha256": _sha256(args.output_replay),
        "evaluation_only": args.evaluation_only,
        "passes": args.passes,
        "optimizer_updates": len(training_metrics),
        "evaluation_coordinates": len(evaluation_metrics),
        "evaluation_split": args.evaluation_split,
        "evaluation_prompt_count": len(evaluation_rows),
        "samples_per_update": SAMPLES,
        "fixed_horizon": FIXED_HORIZON,
        "replay_capacity": REPLAY_CAPACITY,
        "mechanism": {
            "task_objective": "group-centered binary terminal Dr.GRPO",
            "semantic_maxent": args.arm != CONTROL,
            "verified_replay": args.arm != CONTROL,
            "singleton_mass_enabled": args.arm != DELAYED_SINGLETON_XDR,
            "dynamic_legal_support": "adjacent free cells only",
            "gold_support_feedback": False,
            "training_common_random_numbers_across_arms": True,
        },
        "label_token_ids": label_token_ids,
        "source_sha256": {
            "runner": _sha256(Path(__file__).resolve()),
            "shared_interactive_trainer": _sha256(
                OPS / "train_point_maze_interactive_paired_smoke_v1.py"
            ),
            "waypoint_contract": _sha256(SRC / "oat_drgrpo/point_maze_waypoint.py"),
            "waypoint_policy": _sha256(
                SRC / "oat_drgrpo/point_maze_waypoint_policy.py"
            ),
            "waypoint_worker": _sha256(
                SRC / "oat_drgrpo/point_maze_waypoint_worker.py"
            ),
        },
    }
    _atomic(args.output_receipt, receipt)
    print(
        f"[waypoint-pilot] complete arm={args.arm} seed={args.seed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
