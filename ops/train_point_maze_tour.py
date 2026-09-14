#!/usr/bin/env python3
"""Train PointMaze Tour with Dr.GRPO plus optional uniform verified replay.

The policy names one unvisited landmark per decision, so an episode is exactly
``K`` scored decisions and every one of them emits an element of the canonical
key.  Both arms build and score the same verified replay batch; the control
applies an exact-zero replay derivative while the replay arm minimizes uniform
length-normalized negative log likelihood over one stored successful tour per
observed order.

Because ``K`` is constant inside every prompt group, an episode-level advantage
divides evenly across decisions and cannot reward or punish episode length.
That is the property the v1 waypoint runner lacked, and it is what makes this
domain eligible for the semantic-MaxEnt arms.
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
import time
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
from oat_drgrpo.interactive_episode_objective import (  # noqa: E402
    drgrpo_task_advantages,
)
from oat_drgrpo.interactive_episode_replay import (  # noqa: E402
    InteractiveDecisionRecord,
    InteractiveEpisodeRecord,
    VerifiedInteractiveReplayBank,
    fixed_replay_slots,
)
from oat_drgrpo.point_maze_tour import parse_point_tour_spec  # noqa: E402
from oat_drgrpo.semantic_shannon import SemanticShannonTracker  # noqa: E402
from oat_drgrpo.point_maze_tour_policy import (  # noqa: E402
    POINT_TOUR_LABELS,
    POINT_TOUR_PROMPT_FORMATS,
    point_tour_allowed_labels,
    point_tour_label_map,
    point_tour_transition_sha256,
    render_point_tour_prompt,
    render_point_tour_terminal_padding_prompt,
)
from oat_drgrpo.point_maze_tour_process import PointMazeTourProcess  # noqa: E402


# "semantic" is E81's objective on this domain: verified replay *plus* the
# fixed open-set semantic MaxEnt term. PointMaze was excluded from E81/E82/E83
# because distributing an episode-level semantic advantage across a variable
# number of decisions creates a length incentive. Here K is constant inside a
# prompt group, so the term divides evenly and the exclusion no longer applies.
ARMS = ("control", "replay", "semantic")
SEMANTIC_COEFFICIENT = 0.10
SEMANTIC_SURPRISAL_CLIP = 5.0
SEMANTIC_PSEUDOCOUNT = 1.0
SAMPLES = 16
# The decision horizon is read from the dataset, not hardcoded: it is the map's
# landmark count. Pinning it to a constant meant the runner could only ever
# serve one K, so a K=4 run and a K=5 run could not coexist.
MAX_HORIZON = 6
REPLAY_CAPACITY = 16
EVAL_K = 8
DEFAULT_REPLAY_WEIGHT = 0.10
TRAIN_ROWS = 384
DEV_ROWS = 64
EVAL_ROWS = 128
CHECKPOINT_INTERVAL = TRAIN_ROWS // 2
PROGRESS_INTERVAL = 25


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("ascii"))
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="ascii", dir=path.parent, delete=False
    )
    try:
        json.dump(payload, handle, allow_nan=False, indent=1, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    os.replace(handle.name, path)


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="ascii") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _raw_spec(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row["answer"]
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("tour row answer must be a JSON object")
    return value


def run_episodes(
    *,
    model: Any,
    tokenizer: Any,
    worker: PointMazeTourProcess,
    row: Mapping[str, Any],
    sample_count: int,
    session_prefix: str,
    seed_base: int,
    label_token_ids: Mapping[str, int],
    padding_token_ids: Sequence[int],
    max_length: int,
    record_decisions: bool,
    prompt_format: str = "qwen_chatml",
    expected_horizon: int | None = None,
) -> tuple[
    list[InteractiveEpisodeRecord],
    list[dict[str, Any]],
    list[str | None],
    dict[str, Any],
]:
    raw_spec = _raw_spec(row)
    spec = parse_point_tour_spec(raw_spec)
    horizon = spec.landmark_count
    if expected_horizon is not None and horizon != expected_horizon:
        raise ValueError(
            f"tour spec has {horizon} landmarks but this run's horizon is "
            f"{expected_horizon}; a cohort must not mix K"
        )
    labels = point_tour_label_map(spec.landmark_ids)
    label_to_landmark = dict(labels)
    global_action_ids = tuple(
        int(label_token_ids[label])
        for label in POINT_TOUR_LABELS[: spec.landmark_count]
    )
    token_to_label = {int(token): label for label, token in label_token_ids.items()}
    group_prompt_ids = tuple(
        tokenizer.encode(str(row["problem"]), add_special_tokens=False)
    )
    sessions = [
        {"session_id": f"{session_prefix}-e{episode}", "spec": raw_spec}
        for episode in range(sample_count)
    ]
    reset = worker.reset_batch(sessions)
    observations = {item["session_id"]: item for item in reset}
    decisions: list[list[InteractiveDecisionRecord]] = [
        [] for _ in range(sample_count)
    ]
    final: list[dict[str, Any] | None] = [None] * sample_count
    fixed_slots: list[dict[str, Any]] = []
    simulator_calls = 0
    model.eval()

    for decision_round in range(1, horizon + 1):
        prompts = []
        supports = []
        request_seeds = []
        for episode, session in enumerate(sessions):
            state = observations[session["session_id"]]
            if state["done"]:
                raise RuntimeError("tour episode terminated before its fixed horizon")
            prompts.append(
                render_point_tour_prompt(
                    str(row["problem"]), state, prompt_format=prompt_format
                )
            )
            option_labels = point_tour_allowed_labels(
                state["landmark_ids"], state["allowed_landmarks"]
            )
            supports.append(
                tuple(int(label_token_ids[label]) for label in option_labels)
            )
            request_seeds.append(int(seed_base) + episode * 1_000 + decision_round)
        prompt_ids, selected_ids, behavior_rows = interactive._restricted_sample(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            action_token_ids=global_action_ids,
            request_seeds=request_seeds,
            max_length=max_length,
            allowed_token_ids_by_prompt=supports,
        )
        requests = []
        selected_landmarks: dict[int, str] = {}
        for episode, session in enumerate(sessions):
            label = token_to_label[int(selected_ids[episode])]
            landmark = label_to_landmark[label]
            state = observations[session["session_id"]]
            if landmark not in state["allowed_landmarks"]:
                raise RuntimeError("sampled tour landmark escaped its legal support")
            requests.append(
                {"session_id": session["session_id"], "landmark": landmark}
            )
            selected_landmarks[episode] = landmark
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
                        "active": True,
                        "weight": 0.0,
                    }
                )
        transitions = worker.step_batch(requests)
        if len(transitions) != sample_count:
            raise RuntimeError("tour worker returned the wrong batch size")
        simulator_calls += len(transitions)
        for episode, transition in enumerate(transitions):
            session_id = sessions[episode]["session_id"]
            if transition["session_id"] != session_id:
                raise RuntimeError("tour worker reordered sessions")
            before = observations[session_id]
            if record_decisions:
                support = tuple(int(token) for token in supports[episode])
                decisions[episode].append(
                    InteractiveDecisionRecord(
                        prompt_token_ids=tuple(prompt_ids[episode]),
                        allowed_token_ids=support,
                        selected_token_id=int(selected_ids[episode]),
                        behavior_logprobs=tuple(behavior_rows[episode]),
                        transition_sha256=point_tour_transition_sha256(
                            before=before,
                            landmark=selected_landmarks[episode],
                            after=transition,
                        ),
                    )
                )
            observations[session_id] = transition
            if transition["done"]:
                final[episode] = transition
    if any(item is None for item in final):
        raise RuntimeError("tour episodes did not terminate at the fixed horizon")
    if record_decisions:
        if len(fixed_slots) != sample_count * horizon:
            raise RuntimeError("fixed tour policy-slot budget changed")
        if any(len(record) != horizon for record in decisions):
            raise RuntimeError("tour episode decision count is not constant")

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
            "certified_tour_count": int(row.get("certified_tour_count", 0)),
            "horizon": horizon,
            "verified_episodes": sum(outcome is not None for outcome in outcomes),
            "distinct_verified_keys": len(
                {outcome for outcome in outcomes if outcome is not None}
            ),
            "simulator_step_calls": simulator_calls,
        },
    )


def evaluate(
    *,
    model: Any,
    tokenizer: Any,
    worker: PointMazeTourProcess,
    rows: Sequence[Mapping[str, Any]],
    split: str,
    arm: str,
    seed: int,
    update: int,
    label_token_ids: Mapping[str, int],
    padding_token_ids: Sequence[int],
    max_length: int,
    prompt_format: str,
    expected_horizon: int | None = None,
) -> dict[str, Any]:
    per_map = []
    total_calls = 0
    for index, row in enumerate(rows):
        _episodes, _slots, outcomes, diagnostics = run_episodes(
            model=model,
            tokenizer=tokenizer,
            worker=worker,
            row=row,
            sample_count=EVAL_K,
            session_prefix=f"eval-{arm}-s{seed}-u{update}-m{index}",
            seed_base=seed + update * 1_000_000 + index * 10_000,
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
                "certified_tour_count": diagnostics["certified_tour_count"],
                "mean8": successes / EVAL_K,
                "pass8": float(successes > 0),
                "distinct8": float(distinct),
                "modes_per_success": (distinct / successes if successes else 0.0),
            }
        )
    return {
        "schema": "point-maze-tour-evaluation-v1",
        "split": split,
        "arm": arm,
        "seed": seed,
        "learning_round": update,
        "evaluation_prompt_count": len(rows),
        "evaluation_trajectory_count": len(rows) * EVAL_K,
        "evaluation_decision_calls": total_calls,
        "mean8": sum(row["mean8"] for row in per_map) / len(per_map),
        "pass8": sum(row["pass8"] for row in per_map) / len(per_map),
        "distinct8": sum(row["distinct8"] for row in per_map) / len(per_map),
        "modes_per_success": sum(row["modes_per_success"] for row in per_map)
        / len(per_map),
        "certified_tour_mean": sum(row["certified_tour_count"] for row in per_map)
        / len(per_map),
        "per_map": per_map,
    }


def uniform_verified_replay_slots(
    *,
    group: Any,
    padding_token_ids: Sequence[int],
    action_token_ids: Sequence[int],
    compute_only: bool,
    replay_weight: float,
    horizon: int,
    replay_capacity: int = REPLAY_CAPACITY,
    samples: int = SAMPLES,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Materialize fixed-compute slots for the verified-likelihood loss.

    Identical in form to the v1 waypoint runner: for one scheduled prompt bank
    with ``m`` retained tours the applied loss is
    ``alpha * (N-1)/N * 1/N * mean_k[-score(k)]``, where ``score`` is the mean
    selected-action log probability along that tour.
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
            if len(episode.decisions) != horizon:
                raise ValueError("replay tour has a nonconstant decision count")
            per_decision = applied_score_gradient * objective_scale / horizon
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
        raise RuntimeError("fixed tour replay-decision budget changed")
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
    semantic: SemanticShannonTracker,
    update: int,
    args: argparse.Namespace,
) -> None:
    """Write a resumable checkpoint, surviving a kill at any instant.

    The new state is staged beside the target and swapped in by rename, with the
    previous checkpoint retained until the swap lands, so a job killed mid-save
    resumes from the older checkpoint rather than from a half-written one.
    """

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
                "semantic": semantic.state_dict(),
            },
            staging / "training_state.pt",
        )
        _atomic_json(
            staging / "COMPLETE.json",
            {
                "schema": "tour-point-maze-rolling-checkpoint-v1",
                "arm": args.arm,
                "seed": args.seed,
                "update": update,
                "passes": update / TRAIN_ROWS,
                "data_identity_sha256": _sha256(args.data_root / "identity.json"),
                "prompt_format": args.prompt_format,
                "metrics_bytes": args.output_metrics.stat().st_size,
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


def _truncate_to_checkpoint(
    args: argparse.Namespace, metadata: Mapping[str, Any]
) -> None:
    """Drop metrics appended after the checkpoint so a resume cannot double-count."""

    expected = int(metadata["metrics_bytes"])
    path = args.output_metrics
    if not path.is_file() or path.stat().st_size < expected:
        raise RuntimeError(f"checkpoint cannot recover truncated artifact: {path}")
    with path.open("r+b") as handle:
        handle.truncate(expected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--output-model", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--passes", type=int, default=8)
    parser.add_argument("--evaluation-interval", type=int, default=CHECKPOINT_INTERVAL)
    parser.add_argument("--evaluation-prompts", type=int, default=EVAL_ROWS)
    parser.add_argument(
        "--evaluation-split", choices=("dev", "eval"), default="eval"
    )
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=2e-7)
    parser.add_argument("--replay-weight", type=float, default=DEFAULT_REPLAY_WEIGHT)
    parser.add_argument("--microbatch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument(
        "--prompt-format", choices=POINT_TOUR_PROMPT_FORMATS, default="qwen_chatml"
    )
    parser.add_argument("--experiment-id", default="e85pt")
    parser.add_argument("--strict-protocol", action="store_true")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument(
        "--checkpoint-every", type=int, default=192,
        help="save a resumable checkpoint every N updates",
    )
    parser.add_argument(
        "--max-updates-per-run", type=int, default=0,
        help=(
            "stop cleanly after N updates in this invocation and checkpoint. "
            "0 runs to completion. Used to fit the one-hour partition lane."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.strict_protocol and (
        args.passes != 8
        or args.evaluation_interval != CHECKPOINT_INTERVAL
        or args.evaluation_prompts != EVAL_ROWS
        or not math.isclose(args.replay_weight, DEFAULT_REPLAY_WEIGHT)
    ):
        raise ValueError(
            "the registered tour protocol is 8 passes, 192-update evaluation, "
            "128 evaluation maps, and replay weight 0.10"
        )
    if args.output_receipt.exists():
        raise FileExistsError("a fresh terminal tour receipt is required")

    checkpoint = (
        _checkpoint_metadata(args.checkpoint_dir)
        if args.auto_resume and args.checkpoint_dir is not None
        else None
    )

    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("PointMaze Tour requires a BF16-capable GPU")
    device_name = torch.cuda.get_device_name(0)
    device_capability = torch.cuda.get_device_capability(0)
    # Turing reports bf16 support but emulates it, which costs roughly 4x. A
    # paired cohort must not straddle architectures, so record what we ran on.
    print(f"device: {device_name} sm_{device_capability[0]}{device_capability[1]}", flush=True)
    dataset = load_from_disk(str(args.data_root))
    if set(dataset) != {"train", "dev", "eval"}:
        raise ValueError("PointMaze Tour data splits changed")
    train_rows = dataset["train"].to_list()
    evaluation_rows = dataset[args.evaluation_split].to_list()[
        : args.evaluation_prompts
    ]

    if checkpoint is None:
        start_update = 0
        model_source = args.model
    else:
        if (
            checkpoint.get("schema") != "tour-point-maze-rolling-checkpoint-v1"
            or checkpoint.get("arm") != args.arm
            or int(checkpoint.get("seed", -1)) != args.seed
            or checkpoint.get("prompt_format") != args.prompt_format
            or checkpoint.get("data_identity_sha256")
            != _sha256(args.data_root / "identity.json")
        ):
            raise RuntimeError("tour checkpoint identity mismatch")
        _truncate_to_checkpoint(args, checkpoint)
        start_update = int(checkpoint["update"])
        model_source = args.checkpoint_dir / "model"
        print(f"resuming from update {start_update}", flush=True)

    # The horizon is the dataset's landmark count; assert the split is uniform
    # so a cohort cannot silently mix K.
    horizons = {
        len(_raw_spec(row)["landmarks"]) for row in train_rows + evaluation_rows
    }
    if len(horizons) != 1:
        raise ValueError(f"tour dataset mixes landmark counts {sorted(horizons)}")
    horizon = horizons.pop()
    if not 2 <= horizon <= MAX_HORIZON:
        raise ValueError(f"tour horizon {horizon} is outside 2..{MAX_HORIZON}")
    print(f"horizon: {horizon} landmarks per map", flush=True)

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
    for label in POINT_TOUR_LABELS[:horizon]:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1 or tokenizer.decode(token_ids) != label:
            raise RuntimeError(f"tour label {label!r} is not one exact token")
        label_token_ids[label] = int(token_ids[0])
    global_action_ids = tuple(
        label_token_ids[label] for label in POINT_TOUR_LABELS[:horizon]
    )
    padding_token_ids = tokenizer.encode(
        render_point_tour_terminal_padding_prompt(prompt_format=args.prompt_format),
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
    # Built for every arm so the predictor's compute is identical; only the
    # "semantic" arm adds its advantage to the objective.
    semantic = SemanticShannonTracker(
        coefficient=SEMANTIC_COEFFICIENT,
        surprisal_clip=SEMANTIC_SURPRISAL_CLIP,
        pseudocount=SEMANTIC_PSEUDOCOUNT,
    )
    if checkpoint is not None:
        state = torch.load(
            args.checkpoint_dir / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        optimizer.load_state_dict(state["optimizer"])
        replay_bank.load_state_dict(state["replay_bank"])
        if "semantic" in state:
            semantic.load_state_dict(state["semantic"])
        random.setstate(state["python_random_state"])
        torch.set_rng_state(state["torch_random_state"].cpu())
        torch.cuda.set_rng_state_all(state["cuda_random_states"])

    started = datetime.now(timezone.utc).isoformat()
    updates = 0 if args.evaluation_only else len(train_rows) * args.passes
    stopped_early = False
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    if not args.output_metrics.exists():
        args.output_metrics.touch()

    with PointMazeTourProcess(worker_python=args.worker_python) as worker:
        if start_update == 0:
            initial = evaluate(
                model=model,
                tokenizer=tokenizer,
                worker=worker,
                rows=evaluation_rows,
                split=args.evaluation_split,
                arm=args.arm,
                seed=args.seed,
                update=0,
                label_token_ids=label_token_ids,
                padding_token_ids=padding_token_ids,
                max_length=args.max_length,
                prompt_format=args.prompt_format,
                expected_horizon=horizon,
            )
            _append_jsonl(args.output_metrics, [initial])
            print(
                f"pass 0: pass8={initial['pass8']:.3f} "
                f"mean8={initial['mean8']:.3f} "
                f"distinct8={initial['distinct8']:.3f}",
                flush=True,
            )

        loop_started = time.monotonic()
        stopped_early = False
        for update in range(start_update + 1, updates + 1):
            if update % PROGRESS_INTERVAL == 1 and update > start_update + 1:
                done = update - 1 - start_update
                rate = (time.monotonic() - loop_started) / done
                print(
                    f"  progress {done}/{updates} updates, {rate:.2f}s/update, "
                    f"projected {rate * updates / 3600:.2f}h of training",
                    flush=True,
                )
            row = train_rows[(update - 1) % len(train_rows)]
            episodes, policy_slots, _outcomes, rollout = run_episodes(
                model=model,
                tokenizer=tokenizer,
                worker=worker,
                row=row,
                sample_count=SAMPLES,
                session_prefix=f"{args.experiment_id}-{args.arm}-s{args.seed}-u{update}",
                seed_base=args.seed + update * 1_000_000,
                label_token_ids=label_token_ids,
                padding_token_ids=padding_token_ids,
                max_length=args.max_length,
                record_decisions=True,
                prompt_format=args.prompt_format,
                expected_horizon=horizon,
            )
            rewards = torch.tensor(
                [[episode.task_reward for episode in episodes]], dtype=torch.float32
            )
            task_advantages = drgrpo_task_advantages(rewards).flatten()
            semantic_values, _sem_diag, sem_adv = semantic.score_separate_advantages_and_update(
                prompt_token_ids=[episode.group_prompt_token_ids for episode in episodes],
                answer_keys=[episode.outcome_key for episode in episodes],
                num_samples=SAMPLES,
            )
            if args.arm == "semantic":
                task_advantages = task_advantages + torch.tensor(
                    semantic_values, dtype=task_advantages.dtype
                )
            # Constant by construction; asserted in run_episodes.
            for slot in policy_slots:
                episode = int(slot["episode"])
                slot["weight"] = (
                    float(task_advantages[episode].item()) / horizon / SAMPLES
                )

            replay_bank.observe_group(episodes)
            replay_group = replay_bank.schedule_one_global_round_robin()
            replay_slots, replay_diag = uniform_verified_replay_slots(
                group=replay_group,
                padding_token_ids=padding_token_ids,
                action_token_ids=global_action_ids,
                compute_only=args.arm == "control",  # replay applies for "replay" and "semantic"
                replay_weight=args.replay_weight,
                horizon=horizon,
            )
            optimizer.zero_grad(set_to_none=True)
            model.train()
            policy_diag = interactive._backward_fixed_slots(
                model=model,
                tokenizer=tokenizer,
                slots=policy_slots,
                action_token_ids=global_action_ids,
                microbatch_size=args.microbatch_size,
            )
            replay_backward = interactive._backward_fixed_slots(
                model=model,
                tokenizer=tokenizer,
                slots=replay_slots,
                action_token_ids=global_action_ids,
                microbatch_size=args.microbatch_size,
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            _append_jsonl(
                args.output_metrics,
                [
                    {
                        "schema": "tour-point-maze-training-v1",
                        "arm": args.arm,
                        "seed": args.seed,
                        "optimizer_step": update,
                        "map_id": rollout["map_id"],
                        "verified_episodes": rollout["verified_episodes"],
                        "distinct_verified_keys": rollout["distinct_verified_keys"],
                        "semantic_advantage_rms": float(
                            (sum(v * v for v in semantic_values) / len(semantic_values))
                            ** 0.5
                        ),
                        "semantic_applied": float(args.arm == "semantic"),
                        "task_advantage_rms": float(
                            torch.sqrt(torch.mean(task_advantages.square())).item()
                        ),
                        "replay_active_modes": replay_diag["replay_active_modes"],
                        "replay_bank_tracked_outcomes": float(
                            replay_bank.tracked_outcome_count
                        ),
                    }
                ],
            )

            if update % args.evaluation_interval == 0 or update == updates:
                measured = evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    worker=worker,
                    rows=evaluation_rows,
                    split=args.evaluation_split,
                    arm=args.arm,
                    seed=args.seed,
                    update=update,
                    label_token_ids=label_token_ids,
                    padding_token_ids=padding_token_ids,
                    max_length=args.max_length,
                    prompt_format=args.prompt_format,
                    expected_horizon=horizon,
                )
                measured.update(
                    {
                        "semantic_advantage_rms": float(
                            (sum(v * v for v in semantic_values) / len(semantic_values))
                            ** 0.5
                        ),
                        "semantic_applied": float(args.arm == "semantic"),
                        "task_advantage_rms": float(
                            torch.sqrt(torch.mean(task_advantages.square())).item()
                        ),
                        "policy_loss": float(policy_diag.get("loss", 0.0)),
                        "replay_loss": float(replay_backward.get("loss", 0.0)),
                        "replay_bank_tracked_prompts": float(
                            replay_bank.tracked_prompt_count
                        ),
                        "replay_bank_tracked_outcomes": float(
                            replay_bank.tracked_outcome_count
                        ),
                        **replay_diag,
                        **{f"rollout_{k}": v for k, v in rollout.items() if k != "family"},
                    }
                )
                _append_jsonl(args.output_metrics, [measured])
                print(
                    f"update {update}/{updates}: pass8={measured['pass8']:.3f} "
                    f"mean8={measured['mean8']:.3f} "
                    f"distinct8={measured['distinct8']:.3f}",
                    flush=True,
                )

            due = args.checkpoint_dir is not None and (
                update % args.checkpoint_every == 0 or update == updates
            )
            budget_spent = (
                args.max_updates_per_run > 0
                and update - start_update >= args.max_updates_per_run
            )
            if due or (budget_spent and args.checkpoint_dir is not None):
                _save_checkpoint(
                    path=args.checkpoint_dir,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    replay_bank=replay_bank,
                    semantic=semantic,
                    update=update,
                    args=args,
                )
            if budget_spent and update < updates:
                # A clean stop inside this invocation's update budget, so the
                # chunk always ends on a checkpoint rather than on a walltime
                # kill. The next job resumes from exactly here.
                stopped_early = True
                print(
                    f"stopping cleanly at update {update}/{updates} "
                    f"(--max-updates-per-run {args.max_updates_per_run}); "
                    "resume with --auto-resume",
                    flush=True,
                )
                break

        if args.output_model is not None and not args.evaluation_only and not stopped_early:
            args.output_model.parent.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(args.output_model))
            tokenizer.save_pretrained(str(args.output_model))

    if stopped_early:
        print(
            "chunk complete; terminal receipt is written by the final chunk",
            flush=True,
        )
        return

    _atomic_json(
        args.output_receipt,
        {
            "schema": "point-maze-tour-run-v1",
            "status": "complete",
            "experiment_id": args.experiment_id,
            "arm": args.arm,
            "seed": args.seed,
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "model": str(args.model),
            "data_root": str(args.data_root),
            "data_identity_sha256": _sha256(args.data_root / "identity.json"),
            "source_sha256": {
                name: _sha256(SRC / "oat_drgrpo" / name)
                for name in (
                    "point_maze_tour.py",
                    "point_maze_tour_data.py",
                    "point_maze_tour_policy.py",
                    "point_maze_tour_worker.py",
                    "point_maze_tour_process.py",
                )
            },
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "passes": args.passes,
            "updates": updates,
            "samples_per_update": SAMPLES,
            "landmarks": horizon,
            "decisions_per_episode": horizon,
            "replay_capacity": REPLAY_CAPACITY,
            "replay_weight": args.replay_weight,
            "learning_rate": args.learning_rate,
            "prompt_format": args.prompt_format,
            "device_name": device_name,
            "device_capability": list(device_capability),
            "evaluation_split": args.evaluation_split,
            "evaluation_prompts": len(evaluation_rows),
            "evaluation_only": bool(args.evaluation_only),
        },
    )
    print(f"wrote {args.output_receipt}")


if __name__ == "__main__":
    main()
