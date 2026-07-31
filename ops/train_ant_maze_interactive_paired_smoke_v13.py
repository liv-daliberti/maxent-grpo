#!/usr/bin/env python3
"""Train one arm of the frozen AntMaze v13 paired online smoke."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = Path(os.environ.get("OAT_ZERO_SOURCE_ROOT", ROOT / "src")).resolve()
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
OPS = Path(os.environ.get("OAT_ZERO_EXECUTION_ROOT", ROOT / "ops")).resolve()
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

import train_point_maze_interactive_paired_smoke_v1 as base  # noqa: E402
from oat_drgrpo.ant_maze_interactive_policy import (  # noqa: E402
    ANT_POLICY_ACTIONS,
    ANT_TERMINAL_PADDING_PROMPT,
    render_ant_policy_prompt,
)
from oat_drgrpo.ant_maze_interactive_process import AntMazeInteractiveProcess  # noqa: E402
from oat_drgrpo.interactive_episode_objective import (  # noqa: E402
    add_verified_advantage_outside_centering,
    drgrpo_task_advantages,
)
from oat_drgrpo.interactive_episode_replay import (  # noqa: E402
    InteractiveDecisionRecord,
    InteractiveEpisodeRecord,
    VerifiedInteractiveReplayBank,
    interactive_transition_sha256,
)
from oat_drgrpo.maze_modebench import ANT_MAZE_VERIFIER, parse_maze_action_spec  # noqa: E402
from oat_drgrpo.online_canonical_bank import OnlineCanonicalBank  # noqa: E402
from oat_drgrpo.semantic_shannon import SemanticShannonTracker  # noqa: E402


CONTROL = base.CONTROL
TREATMENT = base.TREATMENT
ARMS = base.ARMS
SEED = 76313
SAMPLES = 16
HORIZON = 16
REPLAY_CAPACITY = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--viability-receipt", type=Path, required=True)
    parser.add_argument("--admission-audit", type=Path, required=True)
    parser.add_argument("--warmstart-receipt", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--output-replay", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--learning-rate", type=float, default=2e-7)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--execution-hash", required=True)
    return parser.parse_args()


def rollout_group(
    *,
    model: Any,
    tokenizer: Any,
    worker: AntMazeInteractiveProcess,
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
    if (
        spec.verifier != ANT_MAZE_VERIFIER
        or tuple(spec.action_tokens) != ANT_POLICY_ACTIONS
        or spec.max_actions != HORIZON
        or spec.action_repeat != 400
    ):
        raise ValueError("AntMaze paired-smoke action contract drift")
    group_prompt_ids = tuple(tokenizer.encode(str(row["problem"]), add_special_tokens=False))
    sessions = [
        {"session_id": f"{arm}-u{update_index}-e{episode}", "spec": raw_spec}
        for episode in range(SAMPLES)
    ]
    reset = worker.reset_batch(sessions)
    observations = {item["session_id"]: item for item in reset}
    active = [True] * SAMPLES
    decisions: list[list[InteractiveDecisionRecord]] = [[] for _ in range(SAMPLES)]
    finals: list[dict[str, Any] | None] = [None] * SAMPLES
    fixed_slots: list[dict[str, Any]] = []
    simulator_calls = 0
    model.eval()
    for decision_round in range(1, HORIZON + 1):
        prompts = []
        request_seeds = []
        for episode in range(SAMPLES):
            prompts.append(
                render_ant_policy_prompt(
                    str(row["problem"]), observations[sessions[episode]["session_id"]],
                    ANT_POLICY_ACTIONS, (),
                )
                if active[episode]
                else ANT_TERMINAL_PADDING_PROMPT
            )
            request_seeds.append(
                seed + base.ARM_SEED_OFFSET[arm] + update_index * 1_000_000
                + episode * 1_000 + decision_round
            )
        prompt_ids, selected_ids, behavior_rows = base._restricted_sample(
            model=model, tokenizer=tokenizer, prompts=prompts,
            action_token_ids=action_token_ids, request_seeds=request_seeds,
            max_length=max_length,
        )
        requests = []
        metadata = []
        for episode in range(SAMPLES):
            position = list(action_token_ids).index(selected_ids[episode])
            if active[episode]:
                action = ANT_POLICY_ACTIONS[position]
                requests.append({"session_id": sessions[episode]["session_id"], "action": action})
                metadata.append((episode, action))
            fixed_slots.append({
                "kind": "on_policy", "prompt_token_ids": tuple(prompt_ids[episode]),
                "selected_token_id": int(selected_ids[episode]),
                "behavior_logprob": float(behavior_rows[episode][position]),
                "episode": episode, "active": bool(active[episode]), "weight": 0.0,
            })
        transitions = worker.step_batch(requests) if requests else []
        if len(transitions) != len(metadata):
            raise RuntimeError("AntMaze worker returned the wrong step batch")
        simulator_calls += len(transitions)
        for (episode, action), transition in zip(metadata, transitions):
            session_id = sessions[episode]["session_id"]
            if transition["session_id"] != session_id:
                raise RuntimeError("AntMaze worker reordered sessions")
            before = observations[session_id]
            transition_hash = interactive_transition_sha256(before=before, action=action, after=transition)
            decisions[episode].append(InteractiveDecisionRecord(
                prompt_token_ids=tuple(prompt_ids[episode]), allowed_token_ids=tuple(action_token_ids),
                selected_token_id=int(selected_ids[episode]),
                behavior_logprobs=tuple(behavior_rows[episode]), transition_sha256=transition_hash,
            ))
            observations[session_id] = transition
            if transition["done"]:
                active[episode] = False
                finals[episode] = transition
    if any(active) or any(value is None for value in finals):
        raise RuntimeError("not every AntMaze episode reached a terminal state")
    if len(fixed_slots) != SAMPLES * HORIZON:
        raise RuntimeError("AntMaze fixed policy-slot budget changed")
    episodes = []
    for episode, transition in enumerate(finals):
        assert transition is not None
        key = transition.get("canonical_key")
        episodes.append(InteractiveEpisodeRecord(
            group_prompt_token_ids=group_prompt_ids,
            outcome_key=None if key is None else str(key),
            task_reward=float(key is not None), decisions=tuple(decisions[episode]),
        ))
    return episodes, fixed_slots, {
        "row_index": row_index, "family": str(row["answer_mode_family"]),
        "verified_episodes": sum(episode.task_reward > 0 for episode in episodes),
        "distinct_verified_keys": len({episode.outcome_key for episode in episodes if episode.outcome_key is not None}),
        "active_decisions": sum(len(episode.decisions) for episode in episodes),
        "fixed_policy_slots": len(fixed_slots), "simulator_step_calls": simulator_calls,
    }


def main() -> None:
    args = parse_args()
    if (
        args.seed != SEED or args.learning_rate != 2e-7
        or args.microbatch_size not in (4, 16) or args.max_length != 1536
    ):
        raise ValueError("AntMaze paired-smoke optimizer contract drift")
    for fresh in (args.output_receipt, args.output_metrics, args.output_replay):
        if fresh.exists():
            raise FileExistsError(f"fresh AntMaze paired output required: {fresh}")
    for required in (
        args.model / "config.json", args.data_root / "identity.json",
        args.train_root / "dataset_dict.json", args.worker_python, args.protocol,
        args.viability_receipt, args.admission_audit, args.warmstart_receipt, args.identity,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    viability = json.loads(args.viability_receipt.read_text())
    admission = json.loads(args.admission_audit.read_text())
    warmstart = json.loads(args.warmstart_receipt.read_text())
    identity = json.loads(args.identity.read_text())
    summary = viability.get("summary", {})
    if (
        viability.get("status") != "pass"
        or viability.get("decision") != "eligible_for_ant_v13_paired_online_smoke"
        or summary.get("prompt_count") != 4
        or int(summary.get("prefix_success_prompts", 0)) < 2
        or int(summary.get("multimode_prompts", 0)) < 1
        or admission.get("status") != "pass"
        or warmstart.get("status") != "pass"
        or identity.get("policy_microbatch_size") != args.microbatch_size
        or identity.get("jobs", {}).get(args.arm) != int(args.job_id)
    ):
        raise ValueError("AntMaze paired-smoke antecedent or identity drift")
    dataset = load_from_disk(str(args.train_root))
    if set(dataset) != {"train"} or len(dataset["train"]) != 4:
        raise ValueError("AntMaze paired-smoke train split changed")
    rows = dataset["train"].to_list()

    random.seed(args.seed + base.ARM_SEED_OFFSET[args.arm])
    torch.manual_seed(args.seed + base.ARM_SEED_OFFSET[args.arm])
    torch.cuda.manual_seed_all(args.seed + base.ARM_SEED_OFFSET[args.arm])
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    action_token_ids = []
    for action in ANT_POLICY_ACTIONS:
        ids = tokenizer.encode(action, add_special_tokens=False)
        if len(ids) != 1 or tokenizer.decode(ids) != action:
            raise RuntimeError(f"AntMaze action is not one exact token: {action}")
        action_token_ids.append(int(ids[0]))
    padding_token_ids = tokenizer.encode(ANT_TERMINAL_PADDING_PROMPT, add_special_tokens=False)
    initial_model_hash = base.tree_sha256(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
    )
    semantic = SemanticShannonTracker(
        coefficient=0.10, surprisal_clip=5.0, pseudocount=1.0,
        success_conditioned_signed_advantage=True, success_conditioned_signed_cap=0.05,
        open_set_inverse_adaptation=True, open_set_warmup_steps=64, open_set_ema_decay=0.90,
    )
    canonical = OnlineCanonicalBank(
        entropy_alpha=0.0, novelty_beta=0.50, pseudocount=1.0, surprisal_clip=5.0,
        retain_exemplars=False, replay_capacity=REPLAY_CAPACITY, global_replay_groups_per_step=0,
    )
    replay_bank = VerifiedInteractiveReplayBank(capacity=REPLAY_CAPACITY)
    metrics = []
    with AntMazeInteractiveProcess(worker_python=args.worker_python) as worker:
        for update_index, row in enumerate(rows, start=1):
            episodes, policy_slots, rollout = rollout_group(
                model=model, tokenizer=tokenizer, worker=worker, row=row,
                row_index=update_index - 1, update_index=update_index, arm=args.arm,
                seed=args.seed, action_token_ids=action_token_ids,
                padding_token_ids=padding_token_ids, max_length=args.max_length,
            )
            rewards = torch.tensor([[episode.task_reward for episode in episodes]], dtype=torch.float32)
            task_advantages = drgrpo_task_advantages(rewards).flatten()
            prompts = [episode.group_prompt_token_ids for episode in episodes]
            keys = [episode.outcome_key for episode in episodes]
            reward_values = [episode.task_reward for episode in episodes]
            active_mask = [True] * SAMPLES
            semantic_raw, semantic_diag = semantic.score_success_conditioned_signed_advantages_and_update(
                prompt_token_ids=prompts, answer_keys=keys, task_rewards=reward_values,
                active_mask=active_mask, num_samples=SAMPLES,
            )
            canonical_raw, canonical_diag = canonical.score_and_update(
                prompt_token_ids=prompts, outcome_keys=keys, task_rewards=reward_values,
                active_mask=active_mask, num_samples=SAMPLES,
            )
            raw_exploration = torch.tensor([
                (float(a) + float(b)) * 15.0 / 16.0 for a, b in zip(semantic_raw, canonical_raw)
            ], dtype=torch.float32)
            applied_exploration = raw_exploration if args.arm == TREATMENT else torch.zeros_like(raw_exploration)
            advantages = add_verified_advantage_outside_centering(task_advantages, applied_exploration)
            decision_counts = [len(episode.decisions) for episode in episodes]
            for slot in policy_slots:
                if slot["active"]:
                    index = int(slot["episode"])
                    slot["weight"] = float(advantages[index]) / decision_counts[index] / SAMPLES
            replay_bank.observe_group(episodes)
            replay_slots, replay_diag = base._replay_slots(
                group=replay_bank.schedule_one_global_round_robin(),
                padding_token_ids=padding_token_ids, action_token_ids=action_token_ids,
                compute_only=args.arm == CONTROL,
                horizon=HORIZON, replay_capacity=REPLAY_CAPACITY, samples=SAMPLES,
            )
            optimizer.zero_grad(set_to_none=True)
            model.train()
            policy_diag = base._backward_fixed_slots(
                model=model, tokenizer=tokenizer, slots=[*policy_slots, *replay_slots],
                action_token_ids=action_token_ids, microbatch_size=args.microbatch_size,
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not math.isfinite(float(grad_norm)):
                raise RuntimeError("nonfinite AntMaze gradient")
            optimizer.step()
            metric = {
                "schema": "ant-maze-interactive-paired-smoke-metric-v13",
                "arm": args.arm, "seed": args.seed, "update": update_index,
                "policy_microbatch_size": args.microbatch_size,
                **rollout, **policy_diag, **replay_diag,
                "task_advantage_rms": float(torch.sqrt(torch.mean(task_advantages.square()))),
                "raw_exploration_advantage_rms": float(torch.sqrt(torch.mean(raw_exploration.square()))),
                "applied_exploration_advantage_rms": float(torch.sqrt(torch.mean(applied_exploration.square()))),
                "semantic_effective_advantage_rms": float(semantic_diag.effective_advantage_rms),
                "canonical_novelty_advantage_rms": float(canonical_diag.novelty_advantage_rms),
                "canonical_tracked_prompts": float(canonical.tracked_prompt_count),
                "canonical_tracked_outcomes": float(canonical.tracked_outcome_count),
                "replay_bank_tracked_prompts": float(replay_bank.tracked_prompt_count),
                "replay_bank_tracked_outcomes": float(replay_bank.tracked_outcome_count),
                "grad_norm": float(grad_norm), "optimizer_step": update_index,
                "action_support_escapes": 0,
            }
            if any(isinstance(value, float) and not math.isfinite(value) for value in metric.values()):
                raise RuntimeError("nonfinite AntMaze metric")
            metrics.append(metric)
            base._append_jsonl(args.output_metrics, [metric])
            base._append_jsonl(args.output_replay, [{
                "schema": "ant-maze-interactive-state-replay-v13", "arm": args.arm,
                "seed": args.seed, "update": update_index, "source_row_index": update_index - 1,
                "instance_fingerprint": str(row["instance_fingerprint"]),
                "episodes": [episode.state_dict() for episode in episodes],
            }])
            print(
                f"[ant-paired-v13] arm={args.arm} update={update_index}/4 "
                f"verified={rollout['verified_episodes']}/16 modes={rollout['distinct_verified_keys']}",
                flush=True,
            )
    verified = sum(int(item["verified_episodes"]) for item in metrics)
    multimode = sum(int(item["distinct_verified_keys"] >= 2) for item in metrics)
    base.atomic_json(args.output_receipt, {
        "schema": "ant-maze-interactive-paired-smoke-receipt-v13",
        "generated_at": datetime.now(timezone.utc).isoformat(), "status": "complete",
        "arm": args.arm, "seed": args.seed, "job_id": int(args.job_id),
        "source_hash": args.source_hash, "execution_hash": args.execution_hash,
        "initial_model_tree_sha256": initial_model_hash,
        "model_config_sha256": base.sha256_file(args.model / "config.json"),
        "protocol_sha256": base.sha256_file(args.protocol),
        "identity_sha256": base.sha256_file(args.identity),
        "viability_receipt_sha256": base.sha256_file(args.viability_receipt),
        "admission_audit_sha256": base.sha256_file(args.admission_audit),
        "warmstart_receipt_sha256": base.sha256_file(args.warmstart_receipt),
        "data_identity_sha256": base.sha256_file(args.data_root / "identity.json"),
        "metrics_sha256": base.sha256_file(args.output_metrics),
        "state_replay_sha256": base.sha256_file(args.output_replay),
        "counts": {
            "prompts": 4, "rollouts": 64, "optimizer_steps": 4,
            "fixed_policy_slots": 4 * SAMPLES * HORIZON,
            "fixed_replay_decision_slots": 4 * REPLAY_CAPACITY * HORIZON,
            "verified_episodes": verified, "multimode_updates": multimode,
        },
        "mechanism": {
            "compute_only_control": args.arm == CONTROL, "semantic_coefficient": 0.10,
            "novelty_beta": 0.50, "replay_mass_alpha": 0.10,
            "replay_balance_alpha": 0.10, "warmup_steps": 64,
        },
        "information_boundary": {
            "development_rows_loaded": False, "evaluation_rows_loaded": False,
            "certified_routes_in_context": False, "controller_feedback_in_prompt": False,
            "canonical_keys_in_context": False, "fixed_public_action_mask": True,
        },
    })


if __name__ == "__main__":
    main()
