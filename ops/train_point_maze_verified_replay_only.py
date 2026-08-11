#!/usr/bin/env python3
"""Train PointMaze with Dr.GRPO plus optional uniform verified replay only.

This runner deliberately excludes semantic novelty and known-bank balancing.
Both arms build and score the same verified replay batch.  The control applies
an exact-zero replay derivative; the replay arm minimizes uniform
length-normalized negative log likelihood over one stored successful episode
per observed route.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
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

import train_point_maze_interactive_paired_smoke_v1 as interactive  # noqa: E402
import train_point_maze_waypoint_pilot_v1 as waypoint  # noqa: E402
from oat_drgrpo.interactive_episode_objective import (  # noqa: E402
    drgrpo_task_advantages,
)
from oat_drgrpo.interactive_episode_replay import (  # noqa: E402
    VerifiedInteractiveReplayBank,
    fixed_replay_slots,
)
from oat_drgrpo.point_maze_waypoint_policy import (  # noqa: E402
    POINT_WAYPOINT_LABELS,
    POINT_WAYPOINT_PROMPT_FORMATS,
    render_point_waypoint_terminal_padding_prompt,
)
from oat_drgrpo.point_maze_waypoint_process import (  # noqa: E402
    PointMazeWaypointProcess,
)


ARMS = ("control", "replay")
SAMPLES = 16
FIXED_HORIZON = 64
REPLAY_CAPACITY = 16
EVAL_K = 8
DEFAULT_REPLAY_WEIGHT = 0.10
TRAIN_ROWS = 384
DEV_ROWS = 64
EVAL_ROWS = 128
CHECKPOINT_INTERVAL = TRAIN_ROWS // 2


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


def _atomic_json(path: Path, payload: Any) -> None:
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
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--passes", type=int, default=8)
    parser.add_argument(
        "--evaluation-interval", type=int, default=CHECKPOINT_INTERVAL
    )
    parser.add_argument("--evaluation-prompts", type=int, default=EVAL_ROWS)
    parser.add_argument("--learning-rate", type=float, default=2e-7)
    parser.add_argument("--replay-weight", type=float, default=DEFAULT_REPLAY_WEIGHT)
    parser.add_argument("--microbatch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument(
        "--prompt-format",
        choices=POINT_WAYPOINT_PROMPT_FORMATS,
        default="qwen_chatml",
    )
    parser.add_argument(
        "--experiment-id",
        choices=("e78pm", "e79pm"),
        default="e78pm",
    )
    parser.add_argument("--auto-resume", action="store_true")
    return parser.parse_args()


def uniform_verified_replay_slots(
    *,
    group: Any,
    padding_token_ids: Sequence[int],
    action_token_ids: Sequence[int],
    compute_only: bool,
    replay_weight: float,
    horizon: int = FIXED_HORIZON,
    replay_capacity: int = REPLAY_CAPACITY,
    samples: int = SAMPLES,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Materialize fixed-compute slots for E78's verified-likelihood loss.

    For one scheduled prompt bank with ``m`` retained routes, the applied loss
    is ``alpha * (N-1)/N * 1/N * mean_k[-score(k)]``.  ``score`` is the mean
    selected-action log probability along that interactive episode.  The two
    ``N`` factors exactly mirror E78's Dr.GRPO reward-estimator and
    per-rollout objective normalization.
    """

    if not math.isfinite(float(replay_weight)) or float(replay_weight) < 0.0:
        raise ValueError("replay_weight must be finite and nonnegative")
    if samples <= 1:
        raise ValueError("verified replay requires at least two rollout samples")
    padded, active = fixed_replay_slots(group, slot_count=replay_capacity)
    active_episodes = [episode for episode in padded if episode is not None]
    modes = len(active_episodes)
    raw_score_gradient = (-float(replay_weight) / modes) if modes else 0.0
    applied_score_gradient = 0.0 if compute_only else raw_score_gradient
    objective_scale = ((samples - 1.0) / samples) / samples

    slots: list[dict[str, Any]] = []
    for episode, is_active in zip(padded, active):
        if is_active:
            assert episode is not None
            per_decision = (
                applied_score_gradient * objective_scale / len(episode.decisions)
            )
            for decision in episode.decisions:
                slots.append(
                    {
                        "kind": "replay",
                        "prompt_token_ids": decision.prompt_token_ids,
                        "allowed_token_ids": decision.allowed_token_ids,
                        "selected_token_id": decision.selected_token_id,
                        "behavior_logprob": decision.selected_behavior_logprob,
                        "active": True,
                        "weight": per_decision,
                    }
                )
            padding = horizon - len(episode.decisions)
            if padding < 0:
                raise ValueError("replay episode exceeds the fixed horizon")
            for _ in range(padding):
                slots.append(
                    {
                        "kind": "replay",
                        "prompt_token_ids": tuple(padding_token_ids),
                        "allowed_token_ids": tuple(action_token_ids),
                        "selected_token_id": int(action_token_ids[0]),
                        "behavior_logprob": 0.0,
                        "active": False,
                        "weight": 0.0,
                    }
                )
        else:
            for _ in range(horizon):
                slots.append(
                    {
                        "kind": "replay",
                        "prompt_token_ids": tuple(padding_token_ids),
                        "allowed_token_ids": tuple(action_token_ids),
                        "selected_token_id": int(action_token_ids[0]),
                        "behavior_logprob": 0.0,
                        "active": False,
                        "weight": 0.0,
                    }
                )
    if len(slots) != replay_capacity * horizon:
        raise RuntimeError("fixed PointMaze replay-decision budget changed")
    raw_l2 = abs(raw_score_gradient) * math.sqrt(modes)
    applied_l2 = abs(applied_score_gradient) * math.sqrt(modes)
    return slots, {
        "replay_mode_slots": float(len(padded)),
        "replay_active_modes": float(modes),
        "replay_raw_score_gradient_l2": raw_l2,
        "replay_applied_score_gradient_l2": applied_l2,
        "replay_applied_objective_gradient_l2": applied_l2 * objective_scale,
        "replay_compute_only": float(compute_only),
        "replay_objective_scale": float(objective_scale),
        "replay_weight": float(replay_weight),
        "replay_singleton_mass_enabled": 1.0,
        "replay_balance_eligible_groups": 0.0,
    }


def _recover_checkpoint(path: Path) -> None:
    backup = path.with_name(path.name + ".previous")
    if not path.exists() and backup.exists():
        os.replace(backup, path)


def _checkpoint_metadata(path: Path) -> dict[str, Any] | None:
    _recover_checkpoint(path)
    metadata = path / "COMPLETE.json"
    if not metadata.is_file():
        return None
    return json.loads(metadata.read_text(encoding="utf-8"))


def _save_checkpoint(
    *,
    path: Path,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    replay_bank: VerifiedInteractiveReplayBank,
    update: int,
    args: argparse.Namespace,
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{path.name}.staging.", dir=path.parent))
    backup = path.with_name(path.name + ".previous")
    try:
        model_dir = staging / "model"
        model.eval()
        model.save_pretrained(model_dir, safe_serialization=True, max_shard_size="2GB")
        tokenizer.save_pretrained(model_dir)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "python_random_state": random.getstate(),
                "torch_random_state": torch.get_rng_state(),
                "cuda_random_states": torch.cuda.get_rng_state_all(),
                "replay_bank": replay_bank.state_dict(),
            },
            staging / "training_state.pt",
        )
        _atomic_json(
            staging / "COMPLETE.json",
            {
                "schema": f"{args.experiment_id}-point-maze-rolling-checkpoint-v1",
                "arm": args.arm,
                "seed": args.seed,
                "update": update,
                "passes": update / TRAIN_ROWS,
                "data_identity_sha256": _sha256(args.data_root / "identity.json"),
                "prompt_format": args.prompt_format,
                "metrics_bytes": args.output_metrics.stat().st_size,
                "replay_bytes": args.output_replay.stat().st_size,
            },
        )
        if backup.exists():
            shutil.rmtree(backup)
        if path.exists():
            os.replace(path, backup)
        os.replace(staging, path)
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _truncate_to_checkpoint(args: argparse.Namespace, metadata: Mapping[str, Any]) -> None:
    for path, key in (
        (args.output_metrics, "metrics_bytes"),
        (args.output_replay, "replay_bytes"),
    ):
        expected = int(metadata[key])
        if not path.is_file() or path.stat().st_size < expected:
            raise RuntimeError(f"checkpoint cannot recover truncated artifact: {path}")
        with path.open("r+b") as handle:
            handle.truncate(expected)


def main() -> None:
    args = parse_args()
    if (
        args.passes != 8
        or args.evaluation_interval != CHECKPOINT_INTERVAL
        or args.evaluation_prompts != EVAL_ROWS
        or args.microbatch_size <= 0
        or not math.isclose(args.replay_weight, DEFAULT_REPLAY_WEIGHT)
    ):
        raise ValueError(
            "E78-PM requires 8 passes, 192-update evaluation, 128 eval maps, "
            "and replay weight 0.10"
        )
    if args.output_receipt.exists() or args.output_model.exists():
        raise FileExistsError("fresh terminal PointMaze outputs are required")

    checkpoint = _checkpoint_metadata(args.checkpoint_dir) if args.auto_resume else None
    if checkpoint is None:
        for path in (args.output_metrics, args.output_replay, args.checkpoint_dir):
            if path.exists():
                raise FileExistsError(f"fresh PointMaze output required: {path}")
        args.output_replay.parent.mkdir(parents=True, exist_ok=True)
        args.output_replay.touch()
        args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
        args.output_metrics.touch()
        start_update = 0
        model_source = args.model
    else:
        if (
            checkpoint.get("schema")
            != f"{args.experiment_id}-point-maze-rolling-checkpoint-v1"
            or checkpoint.get("arm") != args.arm
            or int(checkpoint.get("seed", -1)) != args.seed
            or checkpoint.get("data_identity_sha256")
            != _sha256(args.data_root / "identity.json")
            or checkpoint.get("prompt_format") != args.prompt_format
        ):
            raise RuntimeError("PointMaze checkpoint identity mismatch")
        _truncate_to_checkpoint(args, checkpoint)
        start_update = int(checkpoint["update"])
        model_source = args.checkpoint_dir / "model"

    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("PointMaze verified replay requires a BF16-capable GPU")
    dataset = load_from_disk(str(args.data_root))
    if set(dataset) != {"train", "dev", "eval"}:
        raise ValueError("PointMaze data splits changed")
    train_rows = dataset["train"].to_list()
    evaluation_rows = dataset["eval"].to_list()
    if (
        len(train_rows) != TRAIN_ROWS
        or len(dataset["dev"]) != DEV_ROWS
        or len(evaluation_rows) != EVAL_ROWS
    ):
        raise ValueError("E78-PM requires 384 train, 64 dev, and 128 eval maps")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(
        model_source, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    label_token_ids: dict[str, int] = {}
    for label in POINT_WAYPOINT_LABELS:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1 or tokenizer.decode(token_ids) != label:
            raise RuntimeError(f"PointMaze label {label!r} is not one exact token")
        label_token_ids[label] = int(token_ids[0])
    global_action_ids = tuple(label_token_ids[label] for label in POINT_WAYPOINT_LABELS)
    padding_token_ids = tokenizer.encode(
        render_point_waypoint_terminal_padding_prompt(
            prompt_format=args.prompt_format
        ),
        add_special_tokens=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
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
    replay_bank = VerifiedInteractiveReplayBank(capacity=REPLAY_CAPACITY)
    if checkpoint is not None:
        state = torch.load(
            args.checkpoint_dir / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        optimizer.load_state_dict(state["optimizer"])
        replay_bank.load_state_dict(state["replay_bank"])
        random.setstate(state["python_random_state"])
        torch.set_rng_state(state["torch_random_state"].cpu())
        torch.cuda.set_rng_state_all(state["cuda_random_states"])

    updates = len(train_rows) * args.passes
    evaluation_coordinates = 0
    training_updates = start_update
    with PointMazeWaypointProcess(worker_python=args.worker_python) as worker:
        if start_update == 0:
            initial = waypoint._evaluate(
                model=model,
                tokenizer=tokenizer,
                worker=worker,
                rows=evaluation_rows,
                split="eval",
                arm=args.arm,
                seed=args.seed,
                update=0,
                label_token_ids=label_token_ids,
                padding_token_ids=padding_token_ids,
                max_length=args.max_length,
                prompt_format=args.prompt_format,
            )
            _append_jsonl(args.output_metrics, [initial])
            evaluation_coordinates += 1

        for update in range(start_update + 1, updates + 1):
            row_index = (update - 1) % len(train_rows)
            row = train_rows[row_index]
            episodes, policy_slots, _outcomes, rollout = waypoint._run_episodes(
                model=model,
                tokenizer=tokenizer,
                worker=worker,
                row=row,
                sample_count=SAMPLES,
                session_prefix=f"e78pm-{args.arm}-s{args.seed}-u{update}",
                seed_base=args.seed + update * 1_000_000,
                label_token_ids=label_token_ids,
                padding_token_ids=padding_token_ids,
                max_length=args.max_length,
                record_decisions=True,
                prompt_format=args.prompt_format,
            )
            rewards = torch.tensor(
                [[episode.task_reward for episode in episodes]], dtype=torch.float32
            )
            task_advantages = drgrpo_task_advantages(rewards).flatten()
            decision_counts = [len(episode.decisions) for episode in episodes]
            for slot in policy_slots:
                if slot["active"]:
                    episode = int(slot["episode"])
                    slot["weight"] = (
                        float(task_advantages[episode].item())
                        / decision_counts[episode]
                        / SAMPLES
                    )

            replay_bank.observe_group(episodes)
            replay_group = replay_bank.schedule_one_global_round_robin()
            replay_slots, replay_diag = uniform_verified_replay_slots(
                group=replay_group,
                padding_token_ids=padding_token_ids,
                action_token_ids=global_action_ids,
                compute_only=args.arm == "control",
                replay_weight=args.replay_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            model.train()
            policy_diag = interactive._backward_fixed_slots(
                model=model,
                tokenizer=tokenizer,
                slots=[*policy_slots, *replay_slots],
                action_token_ids=global_action_ids,
                microbatch_size=args.microbatch_size,
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not math.isfinite(float(grad_norm.detach().item())):
                raise RuntimeError("nonfinite E78-PM gradient")
            optimizer.step()
            metric = {
                "schema": "e78pm-point-maze-training-v1",
                "split": "train",
                "arm": args.arm,
                "seed": args.seed,
                "learning_round": update,
                "optimizer_step": update,
                "training_passes": update / len(train_rows),
                "row_index": row_index,
                **rollout,
                **policy_diag,
                **replay_diag,
                "task_advantage_rms": float(
                    torch.sqrt(torch.mean(task_advantages.square())).item()
                ),
                "semantic_effective_advantage_rms": 0.0,
                "applied_exploration_advantage_rms": 0.0,
                "canonical_balance_applied_gradient_l2": 0.0,
                "replay_bank_tracked_prompts": float(replay_bank.tracked_prompt_count),
                "replay_bank_tracked_outcomes": float(replay_bank.tracked_outcome_count),
                "grad_norm": float(grad_norm.detach().item()),
                "action_support_escapes": 0,
            }
            _append_jsonl(args.output_metrics, [metric])
            _append_jsonl(
                args.output_replay,
                [
                    {
                        "schema": "e78pm-point-maze-state-replay-v1",
                        "arm": args.arm,
                        "seed": args.seed,
                        "update": update,
                        "source_row_index": row_index,
                        "map_id": rollout["map_id"],
                        "episodes": [episode.state_dict() for episode in episodes],
                    }
                ],
            )
            training_updates = update
            if update % args.evaluation_interval == 0:
                evaluation = waypoint._evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    worker=worker,
                    rows=evaluation_rows,
                    split="eval",
                    arm=args.arm,
                    seed=args.seed,
                    update=update,
                    label_token_ids=label_token_ids,
                    padding_token_ids=padding_token_ids,
                    max_length=args.max_length,
                    prompt_format=args.prompt_format,
                )
                _append_jsonl(args.output_metrics, [evaluation])
                evaluation_coordinates += 1
                _save_checkpoint(
                    path=args.checkpoint_dir,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    replay_bank=replay_bank,
                    update=update,
                    args=args,
                )
            print(
                f"[e78pm] arm={args.arm} seed={args.seed} update={update}/{updates} "
                f"verified={rollout['verified_episodes']}/{SAMPLES} "
                f"modes={rollout['distinct_verified_keys']}",
                flush=True,
            )

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.save_pretrained(
        args.output_model, safe_serialization=True, max_shard_size="2GB"
    )
    tokenizer.save_pretrained(args.output_model)
    receipt = {
        "schema": (
            f"{args.experiment_id}-point-maze-verified-replay-only-receipt-v1"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "arm": args.arm,
        "seed": args.seed,
        "model": str(args.model.resolve()),
        "output_model": str(args.output_model.resolve()),
        "output_model_tree_sha256": _tree_sha256(args.output_model),
        "data_root": str(args.data_root.resolve()),
        "data_identity_sha256": _sha256(args.data_root / "identity.json"),
        "metrics_sha256": _sha256(args.output_metrics),
        "state_replay_sha256": _sha256(args.output_replay),
        "passes": args.passes,
        "optimizer_updates": training_updates,
        "evaluation_coordinates": 17,
        "evaluation_split": "eval",
        "evaluation_prompt_count": EVAL_ROWS,
        "prompt_format": args.prompt_format,
        "samples_per_update": SAMPLES,
        "fixed_horizon": FIXED_HORIZON,
        "replay_capacity": REPLAY_CAPACITY,
        "mechanism": {
            "task_objective": "group-centered binary terminal Dr.GRPO",
            "verified_replay": True,
            "verified_replay_compute_only": args.arm == "control",
            "verified_replay_objective": "uniform_verified_likelihood_per_rollout",
            "verified_replay_weight": args.replay_weight,
            "singleton_replay": True,
            "semantic_maxent": False,
            "canonical_balance": False,
            "adaptive_coefficients": False,
            "dynamic_legal_support": "adjacent prompt-visible free cells only",
            "goal_support_filtering": False,
            "training_common_random_numbers_across_arms": True,
        },
        "label_token_ids": label_token_ids,
        "source_sha256": {
            "runner": _sha256(Path(__file__).resolve()),
            "interactive_loss": _sha256(
                OPS / "train_point_maze_interactive_paired_smoke_v1.py"
            ),
            "waypoint_runner": _sha256(
                OPS / "train_point_maze_waypoint_pilot_v1.py"
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
    _atomic_json(args.output_receipt, receipt)
    print(f"[e78pm] complete arm={args.arm} seed={args.seed}", flush=True)


if __name__ == "__main__":
    main()
