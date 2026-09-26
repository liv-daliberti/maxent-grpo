#!/usr/bin/env python3
"""Development-only constrained-action viability for PantryPlan."""

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

from oat_drgrpo.pantry_plan_interactive import (  # noqa: E402
    PantryInteractiveState,
    legal_pantry_actions,
    render_pantry_policy_state,
    step_pantry_plan,
)


LABELS = tuple("ABCDEFGH")


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
        raise ValueError("PantryPlan answer must contain a JSON specification")
    return value


def _prompt(
    problem: str,
    state: PantryInteractiveState,
    spec: dict[str, Any],
    actions: tuple[str, ...],
) -> str:
    if not 1 <= len(actions) <= len(LABELS):
        raise ValueError("finite PantryPlan menu exceeds the label alphabet")
    options = "\n".join(
        f"{LABELS[index]}: {action}" for index, action in enumerate(actions)
    )
    return (
        "<|im_start|>system\n"
        "Act as a PantryPlan policy. Choose exactly one listed option. "
        "Return only its single capital letter; do not explain."
        "<|im_end|>\n<|im_start|>user\n"
        + problem
        + "\n\nCURRENT STATE\n"
        + render_pantry_policy_state(state, spec)
        + "\n\nLEGAL OPTIONS\n"
        + options
        + "\n\nChoose one option letter."
        "<|im_end|>\n<|im_start|>assistant\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-split-root", type=Path, required=True)
    parser.add_argument("--split", default="multi_answer")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--admission-audit", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--prefix-count", type=int, default=16)
    parser.add_argument("--minimum-prefix-success-prompts", type=int, default=32)
    parser.add_argument("--minimum-multimode-prompts", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--execution-hash", required=True)
    parser.add_argument("--job-id", required=True)
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
        raise ValueError("deterministic PantryPlan admission audit is not passing")
    identity = json.loads(
        (args.data_root / "identity.json").read_text(encoding="utf-8")
    )
    dataset_dict = load_from_disk(str(args.data_split_root))
    if set(dataset_dict) != {args.split}:
        raise ValueError("development split names differ from the frozen contract")
    rows = dataset_dict[args.split].to_list()
    if not rows:
        raise ValueError("development split is empty")

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

    active: list[dict[str, Any]] = [
        {
            "row_index": row_index,
            "sample_index": sample_index,
            "state": PantryInteractiveState(),
            "trace": [],
        }
        for row_index in range(len(rows))
        for sample_index in range(1, args.sample_count + 1)
    ]
    completed: list[dict[str, Any]] = []
    decision_round = 0
    while active:
        decision_round += 1
        if decision_round > 12:
            raise RuntimeError("PantryPlan state machine exceeded its hard horizon")
        requests: list[tuple[dict[str, Any], tuple[str, ...], str, Any]] = []
        for rollout in active:
            row = rows[rollout["row_index"]]
            spec = _spec(row)
            actions = legal_pantry_actions(rollout["state"], spec)
            prompt = _prompt(
                str(row["problem"]),
                rollout["state"],
                spec,
                actions,
            )
            params = vllm.SamplingParams(
                n=1,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                max_tokens=1,
                min_tokens=1,
                ignore_eos=True,
                allowed_token_ids=[
                    label_token_ids[LABELS[index]]
                    for index in range(len(actions))
                ],
                seed=(
                    int(args.seed)
                    + 10_000 * int(rollout["row_index"])
                    + 100 * int(rollout["sample_index"])
                    + decision_round
                ),
            )
            requests.append((rollout, actions, prompt, params))

        next_active: list[dict[str, Any]] = []
        for start in range(0, len(requests), max(int(args.batch_size), 1)):
            batch = requests[start : start + max(int(args.batch_size), 1)]
            outputs = llm.generate(
                [request[2] for request in batch],
                [request[3] for request in batch],
                use_tqdm=False,
            )
            if len(outputs) != len(batch):
                raise RuntimeError("vLLM returned the wrong constrained batch size")
            for (rollout, actions, _prompt_text, _params), output in zip(
                batch, outputs
            ):
                if len(output.outputs) != 1 or len(output.outputs[0].token_ids) != 1:
                    raise RuntimeError("constrained choice emitted other than one token")
                token_id = int(output.outputs[0].token_ids[0])
                allowed = [
                    label_token_ids[LABELS[index]]
                    for index in range(len(actions))
                ]
                if token_id not in allowed:
                    raise RuntimeError("vLLM escaped the finite-action mask")
                option_index = allowed.index(token_id)
                label = LABELS[option_index]
                action = actions[option_index]
                row = rows[rollout["row_index"]]
                transition = step_pantry_plan(
                    rollout["state"],
                    action,
                    _spec(row),
                )
                updated = {
                    **rollout,
                    "state": transition.state,
                    "trace": [
                        *rollout["trace"],
                        {
                            "round": decision_round,
                            "label": label,
                            "action": action,
                            "menu_size": len(actions),
                        },
                    ],
                }
                if transition.terminal:
                    completed.append(
                        {
                            **updated,
                            "verified": transition.validation is not None,
                            "canonical_key": (
                                transition.validation.canonical_key
                                if transition.validation is not None
                                else None
                            ),
                            "candidate": ";".join(
                                f"{name}={grams}"
                                for name, grams in transition.state.allocations_g
                            ),
                        }
                    )
                else:
                    next_active.append(updated)
        active = next_active

    expected = len(rows) * args.sample_count
    if len(completed) != expected:
        raise RuntimeError("not every constrained rollout reached STOP")
    completed.sort(key=lambda row: (row["row_index"], row["sample_index"]))

    prompt_results = []
    attempts = []
    for row_index, row in enumerate(rows):
        rollouts = [row for row in completed if row["row_index"] == row_index]
        keys = [row["canonical_key"] for row in rollouts]
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
        for rollout in rollouts:
            attempts.append(
                {
                    "row_index": row_index,
                    "sample_index": rollout["sample_index"],
                    "candidate": rollout["candidate"],
                    "verified": rollout["verified"],
                    "canonical_key": rollout["canonical_key"],
                    "trace": rollout["trace"],
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
        "schema_version": "pantry-plan-interactive-viability-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "decision": (
            "eligible_for_shared_warmstart_design"
            if passed
            else "advance_to_train_only_warmstart"
        ),
        "domain": "pantry_plan",
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
            "policy_interface": "hierarchical_finite_action_one_token_v1",
            "label_token_ids": label_token_ids,
        },
        "information_boundary": {
            "development_only": True,
            "evaluation_prompts_loaded": False,
            "certified_supports_in_context": False,
            "solver_outputs_in_context": False,
            "action_mask_uses_only_public_local_constraints": True,
            "endpoint_verifier_feedback_before_stop": False,
            "running_totals_are_public_spec_arithmetic": True,
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
            "decision_rounds": decision_round,
        },
        "prompt_results": prompt_results,
        "attempts": attempts,
    }
    _atomic_json(args.output, payload)
    print(
        "[pantry-interactive] "
        f"status={payload['status']} prompts={len(rows)} "
        f"verified={payload['summary']['verified_completions']} "
        f"prefix_success={prefix_success_prompts} multimode={multimode_prompts}",
        flush=True,
    )


if __name__ == "__main__":
    main()
