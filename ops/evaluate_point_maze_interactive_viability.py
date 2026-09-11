#!/usr/bin/env python3
"""Development-only stepwise 0.5B viability for PointMaze."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = Path(os.environ.get("OAT_ZERO_SOURCE_ROOT", ROOT / "src")).resolve()
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oat_drgrpo.maze_modebench import (  # noqa: E402
    POINT_MAZE_VERIFIER,
    parse_maze_action_spec,
)
from oat_drgrpo.point_maze_interactive_process import (  # noqa: E402
    PointMazeInteractiveProcess,
)
from oat_drgrpo.point_maze_interactive_policy import (  # noqa: E402
    render_point_policy_prompt_v2,
    render_point_policy_prompt_v3,
)


LABELS = tuple("ABCDEFGHI")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _spec(row: dict[str, Any]) -> dict[str, Any]:
    value = row["answer"]
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("PointMaze answer must contain a JSON specification")
    parsed = parse_maze_action_spec(value)
    if parsed.verifier != POINT_MAZE_VERIFIER:
        raise ValueError("interactive viability received a non-PointMaze row")
    return value


def _prompt(
    problem: str,
    observation: dict[str, Any],
    actions: tuple[str, ...],
    history: list[str],
) -> str:
    if len(actions) != len(LABELS):
        raise ValueError("PointMaze alphabet differs from the label contract")
    options = "\n".join(
        f"{LABELS[index]}: {action}" for index, action in enumerate(actions)
    )
    recent = " ".join(history[-32:]) if history else "(none)"
    return (
        "<|im_start|>system\n"
        "Act as a closed-loop PointMaze policy. Choose exactly one listed "
        "force pulse. Return only its single capital option letter."
        "<|im_end|>\n<|im_start|>user\n"
        + problem
        + "\n\nCURRENT SIMULATOR OBSERVATION\n"
        + f"position_xy={observation['achieved_goal']}\n"
        + f"goal_xy={observation['desired_goal']}\n"
        + f"remaining_actions={observation['remaining_actions']}\n"
        + f"recent_actions={recent}\n"
        + "N is upward in the printed map; E is rightward. "
        + "Choose the next short force pulse from:\n"
        + options
        + "\nChoose one option letter."
        "<|im_end|>\n<|im_start|>assistant\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-split-root", type=Path, required=True)
    parser.add_argument("--split", default="multi_answer")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--admission-audit", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--prefix-count", type=int, default=16)
    parser.add_argument("--minimum-prefix-success-prompts", type=int, default=2)
    parser.add_argument("--minimum-multimode-prompts", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-model-len", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--execution-hash", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument(
        "--policy-interface",
        choices=("history_v1", "compact_state_v2", "velocity_state_v3"),
        default="history_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.prefix_count <= args.sample_count:
        raise ValueError("prefix count must be in [1, sample count]")
    if args.output.exists():
        raise FileExistsError(f"fresh viability receipt required: {args.output}")

    import vllm
    from datasets import load_from_disk

    admission = json.loads(args.admission_audit.read_text(encoding="utf-8"))
    if admission.get("status") != "pass":
        raise ValueError("deterministic PointMaze admission audit is not passing")
    identity = json.loads(
        (args.data_root / "identity.json").read_text(encoding="utf-8")
    )
    dataset_dict = load_from_disk(str(args.data_split_root))
    if set(dataset_dict) != {args.split}:
        raise ValueError("development split names differ from the frozen contract")
    rows = dataset_dict[args.split].to_list()
    if not rows:
        raise ValueError("development split is empty")
    specs = [_spec(row) for row in rows]

    llm = vllm.LLM(
        model=str(args.model.resolve()),
        dtype="bfloat16",
        max_model_len=int(args.max_model_len),
        gpu_memory_utilization=0.82,
        swap_space=16.0,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()
    label_token_ids: dict[str, int] = {}
    for label in LABELS:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1 or tokenizer.decode(token_ids) != label:
            raise RuntimeError(f"option label {label!r} is not one exact token")
        label_token_ids[label] = int(token_ids[0])

    rollouts = [
        {
            "session_id": f"r{row_index:02d}-s{sample_index:03d}",
            "row_index": row_index,
            "sample_index": sample_index,
            "history": [],
        }
        for row_index in range(len(rows))
        for sample_index in range(1, args.sample_count + 1)
    ]
    worker = PointMazeInteractiveProcess(worker_python=args.worker_python)
    try:
        observations = worker.reset_batch(
            [
                {
                    "session_id": rollout["session_id"],
                    "spec": specs[rollout["row_index"]],
                }
                for rollout in rollouts
            ]
        )
        by_session = {
            observation["session_id"]: observation
            for observation in observations
        }
        active = rollouts
        completed: list[dict[str, Any]] = []
        decision_round = 0
        hard_horizon = max(
            parse_maze_action_spec(spec).max_actions for spec in specs
        )
        while active:
            decision_round += 1
            if decision_round > hard_horizon:
                raise RuntimeError("PointMaze state machine exceeded its horizon")
            requests = []
            for rollout in active:
                spec = parse_maze_action_spec(specs[rollout["row_index"]])
                actions = tuple(spec.action_tokens)
                prompt_renderer = {
                    "history_v1": _prompt,
                    "compact_state_v2": render_point_policy_prompt_v2,
                    "velocity_state_v3": render_point_policy_prompt_v3,
                }[args.policy_interface]
                prompt = prompt_renderer(
                    str(rows[rollout["row_index"]]["problem"]),
                    by_session[rollout["session_id"]],
                    actions,
                    rollout["history"],
                )
                params = vllm.SamplingParams(
                    n=1,
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    max_tokens=1,
                    min_tokens=1,
                    ignore_eos=True,
                    allowed_token_ids=[
                        label_token_ids[label] for label in LABELS
                    ],
                    seed=(
                        int(args.seed)
                        + 10_000 * int(rollout["row_index"])
                        + 100 * int(rollout["sample_index"])
                        + decision_round
                    ),
                )
                requests.append((rollout, actions, prompt, params))

            choices: list[tuple[dict[str, Any], str]] = []
            for start in range(0, len(requests), max(int(args.batch_size), 1)):
                batch = requests[start : start + max(int(args.batch_size), 1)]
                outputs = llm.generate(
                    [request[2] for request in batch],
                    [request[3] for request in batch],
                    use_tqdm=False,
                )
                if len(outputs) != len(batch):
                    raise RuntimeError("vLLM returned the wrong constrained batch")
                for (rollout, actions, _prompt_text, _params), output in zip(
                    batch, outputs
                ):
                    if (
                        len(output.outputs) != 1
                        or len(output.outputs[0].token_ids) != 1
                    ):
                        raise RuntimeError(
                            "constrained action emitted other than one token"
                        )
                    token_id = int(output.outputs[0].token_ids[0])
                    allowed = [
                        label_token_ids[label] for label in LABELS
                    ]
                    if token_id not in allowed:
                        raise RuntimeError("vLLM escaped the PointMaze mask")
                    action = actions[allowed.index(token_id)]
                    choices.append((rollout, action))

            transitions = worker.step_batch(
                [
                    {
                        "session_id": rollout["session_id"],
                        "action": action,
                    }
                    for rollout, action in choices
                ]
            )
            if len(transitions) != len(choices):
                raise RuntimeError("PointMaze worker returned the wrong batch")
            next_active = []
            for (rollout, action), transition in zip(choices, transitions):
                if transition["session_id"] != rollout["session_id"]:
                    raise RuntimeError("PointMaze worker reordered sessions")
                updated = {
                    **rollout,
                    "history": [*rollout["history"], action],
                }
                by_session[rollout["session_id"]] = transition
                if transition["done"]:
                    completed.append(
                        {
                            **updated,
                            "verified": transition["canonical_key"] is not None,
                            "canonical_key": transition["canonical_key"],
                            "directed_gates": transition["directed_gates"],
                            "success": transition["success"],
                            "final_goal_distance": transition[
                                "final_goal_distance"
                            ],
                            "simulator_steps": transition["simulator_steps"],
                            "validation_error": transition["validation_error"],
                        }
                    )
                else:
                    next_active.append(updated)
            active = next_active
    finally:
        worker.close()

    expected = len(rows) * args.sample_count
    if len(completed) != expected:
        raise RuntimeError("not every PointMaze rollout reached a terminal state")
    completed.sort(key=lambda row: (row["row_index"], row["sample_index"]))

    prompt_results = []
    attempts = []
    for row_index, row in enumerate(rows):
        row_rollouts = [
            rollout
            for rollout in completed
            if rollout["row_index"] == row_index
        ]
        keys = [rollout["canonical_key"] for rollout in row_rollouts]
        prefix_keys = [key for key in keys[: args.prefix_count] if key is not None]
        full_keys = [key for key in keys if key is not None]
        counts = Counter(full_keys)
        prompt_results.append(
            {
                "row_index": row_index,
                "family": str(row.get("answer_mode_family", "")),
                "instance_fingerprint": str(
                    row.get("instance_fingerprint", "")
                ),
                "verified_in_prefix": len(prefix_keys),
                "verified_in_full_sample": len(full_keys),
                "distinct_keys_in_prefix": len(set(prefix_keys)),
                "distinct_keys_in_full_sample": len(counts),
                "canonical_key_counts": dict(sorted(counts.items())),
            }
        )
        for rollout in row_rollouts:
            attempts.append(
                {
                    "row_index": row_index,
                    "sample_index": rollout["sample_index"],
                    "actions": rollout["history"],
                    "verified": rollout["verified"],
                    "canonical_key": rollout["canonical_key"],
                    "directed_gates": rollout["directed_gates"],
                    "success": rollout["success"],
                    "final_goal_distance": rollout["final_goal_distance"],
                    "simulator_steps": rollout["simulator_steps"],
                    "validation_error": rollout["validation_error"],
                }
            )

    prefix_success_prompts = sum(
        row["verified_in_prefix"] > 0 for row in prompt_results
    )
    multimode_prompts = sum(
        row["distinct_keys_in_full_sample"] >= 2 for row in prompt_results
    )
    passed = (
        prefix_success_prompts >= args.minimum_prefix_success_prompts
        and multimode_prompts >= args.minimum_multimode_prompts
    )
    if not math.isfinite(float(prefix_success_prompts + multimode_prompts)):
        raise RuntimeError("nonfinite viability counts")
    payload = {
        "schema_version": "point-maze-interactive-viability-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "decision": (
            "eligible_for_shared_warmstart_design"
            if passed
            else (
                "advance_to_train_only_warmstart"
                if args.policy_interface == "history_v1"
                else (
                    "point_maze_compact_warmstart_v2_ineligible"
                    if args.policy_interface == "compact_state_v2"
                    else "point_maze_velocity_warmstart_v3_ineligible"
                )
            )
        ),
        "domain": "point_maze",
        "job_id": int(args.job_id),
        "model": str(args.model.resolve()),
        "model_config_sha256": _sha256_file(args.model / "config.json"),
        "source_hash": args.source_hash,
        "execution_hash": args.execution_hash,
        "protocol_sha256": _sha256_file(args.protocol),
        "admission_audit_sha256": _sha256_file(args.admission_audit),
        "dataset_identity_sha256": _canonical_sha256(identity),
        "data_split_root": str(args.data_split_root.resolve()),
        "data_split_sha256": _canonical_sha256(rows),
        "sampling": {
            "seed": args.seed,
            "sample_count": args.sample_count,
            "prefix_count": args.prefix_count,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_model_len": args.max_model_len,
            "policy_interface": (
                "closed_loop_finite_action_one_token_v1"
                if args.policy_interface == "history_v1"
                else (
                    "closed_loop_compact_public_state_one_token_v2"
                    if args.policy_interface == "compact_state_v2"
                    else "closed_loop_public_markov_state_one_token_v3"
                )
            ),
            "label_token_ids": label_token_ids,
            "action_repeat": 5,
        },
        "information_boundary": {
            "development_only": True,
            "evaluation_prompts_loaded": False,
            "certified_route_programs_in_context": False,
            "planner_outputs_in_context": False,
            "action_mask_is_fixed_public_alphabet": True,
            "endpoint_verifier_feedback_before_terminal": False,
        },
        "criteria": {
            "minimum_prefix_success_prompts": (
                args.minimum_prefix_success_prompts
            ),
            "minimum_multimode_prompts": args.minimum_multimode_prompts,
        },
        "summary": {
            "prompt_count": len(rows),
            "verified_completions": sum(
                row["verified_in_full_sample"] for row in prompt_results
            ),
            "prefix_success_prompts": prefix_success_prompts,
            "multimode_prompts": multimode_prompts,
            "maximum_decision_rounds": decision_round,
        },
        "prompt_results": prompt_results,
        "attempts": attempts,
    }
    _atomic_json(args.output, payload)
    print(
        "[point-interactive] "
        f"status={payload['status']} prompts={len(rows)} "
        f"verified={payload['summary']['verified_completions']} "
        f"prefix_success={prefix_success_prompts} multimode={multimode_prompts}",
        flush=True,
    )


if __name__ == "__main__":
    main()
