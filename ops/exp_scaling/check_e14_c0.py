#!/usr/bin/env python3
"""Fail-closed validation gate for E14's canonical-action C0 run.

This checker deliberately consumes the exact endpoint audit as a separate
artifact.  The intended workflow is:

1. run ``audit_e14_checkpoint.py`` on ``saved_models/step_00128``;
2. pass that JSON to this checker together with the C0 run and identity; and
3. use the resulting approval artifact only if every runtime and exact-policy
   condition passes.

The approval output is removed before validation, so a failed re-check cannot
leave a stale positive artifact behind.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from check_e14_preflight import (  # noqa: E402
    ANSI_ESCAPE,
    BEHAVIOR_ESS_FRACTION_MIN,
    BEHAVIOR_KL_MAX,
    BEHAVIOR_Q_NORM_ERROR_MAX,
    BEHAVIOR_RATIO_MAX,
    BEHAVIOR_RATIO_MIN,
    BEHAVIOR_TV_MAX,
    CONFIG_FIELD,
    EXPECTED_CONFIG as PREFLIGHT_CONFIG,
    EXPECTED_DATASET_IDENTITY,
    EXPECTED_RUNTIME_IDENTITY,
    FATAL_LOG_PATTERNS,
    GateError,
    _as_integral,
    _finite,
    _matches_config,
    _read_metrics,
    _sha256_file,
    _strict_json,
    parse_slurm_status,
    query_slurm,
    read_identity,
    source_tree_hash,
    verify_approval_for_source,
)


GATE_NAME = "e14_canonical_c0"
EXPECTED_UPDATES = 128
EXPECTED_GROUP_SIZE = 16
EXPECTED_QUERY_BUDGET = (EXPECTED_UPDATES - 1) * EXPECTED_GROUP_SIZE
EXPECTED_BEHAVIOR_Q_ROWS = EXPECTED_GROUP_SIZE * 3
EXPECTED_EVAL_ROWS = 96
EXPECTED_LEAF_COUNT = 27
MAX_ACTION_ENTROPY = math.log(27.0)
MIN_ENDPOINT_P_VALID = 0.05
# KL is non-negative in exact arithmetic, but the float64 reduction of two
# identical three-way rows can leave a signed residue around zero.  This is a
# numerical-boundary tolerance only; the preregistered 0.01-nat upper gate is
# unchanged.
KL_NONNEGATIVE_ROUNDOFF_ABS = 1e-12
EXPECTED_ACTIONS = tuple(
    "".join(parts) for parts in itertools.product(("1", "2", "3"), repeat=3)
)
EXPECTED_ACTION_TOKEN_IDS = (16, 17, 18)
EXPECTED_CONFIG: dict[str, Any] = {
    **PREFLIGHT_CONFIG,
    "learning_rate": 2e-7,
    "max_queries": EXPECTED_QUERY_BUDGET,
    "max_train": EXPECTED_QUERY_BUDGET,
    "save_ckpt": True,
    "save_from": 32,
    "save_steps": 32,
    "train_batch_size": 16,
    "train_batch_size_per_device": 4,
    "rollout_batch_size": 1,
    "rollout_batch_size_per_device": 1,
    "cliprange": 0.2,
    "eval_batch_size": 64,
    "eval_generate_max_length": 192,
    "eval_steps": 32,
    "generate_max_length": 192,
    "max_save_mem": 2000,
    "max_save_num": 5,
    "num_prompt_epoch": 1,
    "prompt_max_length": 256,
    "zero_stage": 2,
}
FORBIDDEN_TELEMETRY_PREFIXES = (
    "train/maxent_",
    "train/policy_entropy_",
    "train/seed_entropy_",
    "train/xdr_sac_",
    "train/xdr_tau_",
)


def _mean(values: Sequence[float], *, label: str) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise GateError(f"cannot compute {label} from empty or nonfinite values")
    return math.fsum(values) / len(values)


def _require_close(
    observed: float,
    expected: float,
    *,
    label: str,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-10,
) -> None:
    if not math.isfinite(observed) or not math.isclose(
        observed, expected, rel_tol=rel_tol, abs_tol=abs_tol
    ):
        raise GateError(f"{label}={observed!r}; expected {expected!r}")


def _check_forbidden_telemetry(row: dict[str, Any], *, step: int) -> None:
    for key, raw_value in row.items():
        if key.startswith(FORBIDDEN_TELEMETRY_PREFIXES):
            raise GateError(f"step {step} contains forbidden treatment telemetry {key}")
        if key.endswith(("_nan", "_inf")):
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as error:
                raise GateError(f"step {step} has nonnumeric numerical marker {key}") from error
            if not math.isfinite(value) or value != 0:
                raise GateError(f"step {step} reports numerical failure in {key}")


def _inspect_update_row(row: dict[str, Any], *, step: int) -> dict[str, float]:
    required_finite = (
        "actor/canonical_action_count",
        "actor/canonical_action_support_size",
        "actor/canonical_behavior_q_norm_error_max",
        "actor/canonical_behavior_q_row_count",
        "actor/canonical_behavior_q_support_max",
        "actor/canonical_behavior_q_support_min",
        "actor/canonical_finish_length_count",
        "actor/canonical_finish_unexpected_count",
        "actor/canonical_graph_actions",
        "actor/canonical_invalid_count",
        "actor/canonical_sampler_fixed_shape",
        "actor/canonical_sampler_learner",
        "actor/formatted",
        "actor/generate_avg_str_len",
        "actor/no_eos_count",
        "actor/num_data",
        "actor/response_tok_len",
        "actor/rewards",
        "actor/sampling_max_tokens",
        "actor/sampling_temperature",
        "misc/lr",
        "misc/prompt_consumed",
        "misc/query_step",
        "train/canonical_action_count",
        "train/canonical_action_vocab_size",
        "train/canonical_behavior_denominator_actor",
        "train/canonical_behavior_kl_actor_learner_max",
        "train/canonical_behavior_kl_actor_learner_mean",
        "train/canonical_behavior_kl_learner_actor_max",
        "train/canonical_behavior_kl_learner_actor_mean",
        "train/canonical_behavior_prefix_ess_fraction_min",
        "train/canonical_behavior_q_norm_error_max",
        "train/canonical_behavior_q_row_count",
        "train/canonical_behavior_q_support_max",
        "train/canonical_behavior_q_support_min",
        "train/canonical_behavior_ratio_max",
        "train/canonical_behavior_ratio_min",
        "train/canonical_behavior_sequence_ess_fraction",
        "train/canonical_behavior_tv_max",
        "train/canonical_behavior_tv_mean",
        "train/canonical_sampled_prefix_entropy_ratio",
        "train/canonical_sampled_prefix_entropy_sum",
        "train/canonical_token_entropy_mean",
        "train/entropy",
        "train/learning_round",
        "train/pg_loss",
        "train/policy_grad_norm",
    )
    for key in required_finite:
        _finite(row, key, step=step)
    _check_forbidden_telemetry(row, step=step)

    exact_values = {
        "actor/canonical_action_count": 3,
        "actor/canonical_action_support_size": 3,
        "actor/canonical_behavior_q_row_count": EXPECTED_BEHAVIOR_Q_ROWS,
        "actor/canonical_behavior_q_support_max": 3,
        "actor/canonical_behavior_q_support_min": 3,
        "actor/canonical_finish_length_count": EXPECTED_GROUP_SIZE,
        "actor/canonical_finish_unexpected_count": 0,
        "actor/canonical_graph_actions": 1,
        "actor/canonical_invalid_count": 0,
        "actor/canonical_sampler_fixed_shape": 1,
        "actor/canonical_sampler_learner": 1,
        "actor/formatted": 1,
        "actor/generate_avg_str_len": 3,
        "actor/no_eos_count": 0,
        "actor/num_data": EXPECTED_GROUP_SIZE,
        "actor/response_tok_len": 3,
        "actor/sampling_max_tokens": 3,
        "actor/sampling_temperature": 1,
        "train/canonical_action_count": 3,
        "train/canonical_action_vocab_size": 3,
        "train/canonical_behavior_denominator_actor": 1,
        "train/canonical_behavior_q_row_count": EXPECTED_BEHAVIOR_Q_ROWS,
        "train/canonical_behavior_q_support_max": 3,
        "train/canonical_behavior_q_support_min": 3,
    }
    for key, expected in exact_values.items():
        observed = _finite(row, key, step=step)
        if observed != expected:
            raise GateError(f"step {step} has {key}={observed}; expected {expected}")
    learning_round = _finite(row, "train/learning_round", step=step)
    if learning_round != step:
        raise GateError(
            f"step {step} has train/learning_round={learning_round}; expected {step}"
        )

    _require_close(
        _finite(row, "misc/lr", step=step),
        2e-7,
        label=f"step {step} learning rate",
        rel_tol=1e-6,
        abs_tol=1e-12,
    )
    for key in ("misc/query_step", "misc/prompt_consumed"):
        observed = _finite(row, key, step=step)
        expected = step * EXPECTED_GROUP_SIZE
        if observed != expected:
            raise GateError(f"step {step} has {key}={observed}; expected {expected}")

    actor_norm = _finite(
        row, "actor/canonical_behavior_q_norm_error_max", step=step
    )
    learner_norm = _finite(
        row, "train/canonical_behavior_q_norm_error_max", step=step
    )
    for label, value in (("actor", actor_norm), ("learner", learner_norm)):
        if not 0 <= value <= BEHAVIOR_Q_NORM_ERROR_MAX:
            raise GateError(
                f"step {step} {label} behavior-q normalization error {value} "
                f"exceeds {BEHAVIOR_Q_NORM_ERROR_MAX}"
            )

    ratio_min = _finite(row, "train/canonical_behavior_ratio_min", step=step)
    ratio_max = _finite(row, "train/canonical_behavior_ratio_max", step=step)
    if not BEHAVIOR_RATIO_MIN <= ratio_min <= ratio_max <= BEHAVIOR_RATIO_MAX:
        raise GateError(
            f"step {step} full behavior-policy ratio range "
            f"[{ratio_min}, {ratio_max}] is outside "
            f"[{BEHAVIOR_RATIO_MIN}, {BEHAVIOR_RATIO_MAX}]"
        )

    tv_mean = _finite(row, "train/canonical_behavior_tv_mean", step=step)
    tv_max = _finite(row, "train/canonical_behavior_tv_max", step=step)
    if not 0 <= tv_mean <= tv_max <= BEHAVIOR_TV_MAX:
        raise GateError(
            f"step {step} behavior-policy TV {tv_mean}/{tv_max} exceeds "
            f"{BEHAVIOR_TV_MAX}"
        )

    kl_actor_mean = _finite(
        row, "train/canonical_behavior_kl_actor_learner_mean", step=step
    )
    kl_actor_max = _finite(
        row, "train/canonical_behavior_kl_actor_learner_max", step=step
    )
    kl_learner_mean = _finite(
        row, "train/canonical_behavior_kl_learner_actor_mean", step=step
    )
    kl_learner_max = _finite(
        row, "train/canonical_behavior_kl_learner_actor_max", step=step
    )
    for direction, mean, maximum in (
        ("behavior||learner", kl_actor_mean, kl_actor_max),
        ("learner||behavior", kl_learner_mean, kl_learner_max),
    ):
        if not (
            -KL_NONNEGATIVE_ROUNDOFF_ABS <= mean
            and mean <= maximum + KL_NONNEGATIVE_ROUNDOFF_ABS
            and -KL_NONNEGATIVE_ROUNDOFF_ABS <= maximum <= BEHAVIOR_KL_MAX
        ):
            raise GateError(
                f"step {step} KL({direction}) mean/max {mean}/{maximum} is "
                f"outside the roundoff-aware range "
                f"[-{KL_NONNEGATIVE_ROUNDOFF_ABS}, {BEHAVIOR_KL_MAX}]"
            )

    sequence_ess = _finite(
        row, "train/canonical_behavior_sequence_ess_fraction", step=step
    )
    prefix_ess = _finite(
        row, "train/canonical_behavior_prefix_ess_fraction_min", step=step
    )
    for label, value in (("sequence", sequence_ess), ("prefix", prefix_ess)):
        if not BEHAVIOR_ESS_FRACTION_MIN <= value <= 1.0 + 1e-6:
            raise GateError(
                f"step {step} behavior-policy {label} ESS fraction {value} is "
                f"outside [{BEHAVIOR_ESS_FRACTION_MIN}, 1]"
            )

    entropy_sum = _finite(
        row, "train/canonical_sampled_prefix_entropy_sum", step=step
    )
    entropy_ratio = _finite(
        row, "train/canonical_sampled_prefix_entropy_ratio", step=step
    )
    if not 0 <= entropy_sum <= MAX_ACTION_ENTROPY + 1e-5:
        raise GateError(f"step {step} canonical entropy is outside [0, log(27)]")
    _require_close(
        entropy_ratio,
        entropy_sum / MAX_ACTION_ENTROPY,
        label=f"step {step} canonical entropy ratio",
        rel_tol=1e-5,
        abs_tol=1e-7,
    )
    token_entropy = _finite(row, "train/canonical_token_entropy_mean", step=step)
    _require_close(
        token_entropy,
        entropy_sum / 3.0,
        label=f"step {step} canonical token entropy",
        rel_tol=1e-5,
        abs_tol=1e-7,
    )
    _require_close(
        _finite(row, "train/entropy", step=step),
        token_entropy,
        label=f"step {step} restricted entropy alias",
        rel_tol=1e-6,
        abs_tol=1e-7,
    )
    reward = _finite(row, "actor/rewards", step=step)
    if not 0 <= reward <= 1:
        raise GateError(f"step {step} rollout reward {reward} is outside [0, 1]")
    if not math.isclose(reward * EXPECTED_GROUP_SIZE, round(reward * EXPECTED_GROUP_SIZE), abs_tol=1e-7):
        raise GateError(f"step {step} rollout reward {reward} is not a 16-rollout mean")
    return {
        "actor_norm": actor_norm,
        "learner_norm": learner_norm,
        "ratio_min": ratio_min,
        "ratio_max": ratio_max,
        "tv_max": tv_max,
        "kl_actor_max": kl_actor_max,
        "kl_learner_max": kl_learner_max,
        "sequence_ess": sequence_ess,
        "prefix_ess": prefix_ess,
        "reward": reward,
    }


def inspect_metrics(
    run_dir: Path,
    *,
    archived_removed_steps: Sequence[str] | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    candidates = sorted(run_dir.glob("debug_*/train_metrics.jsonl"))
    if len(candidates) != 1:
        raise GateError(
            "restart-invalid C0 requires exactly one metrics stream; "
            f"found {len(candidates)}"
        )
    metrics_path = candidates[0]
    rows = _read_metrics(metrics_path)
    if len(rows) != EXPECTED_UPDATES + 2:
        raise GateError(
            "C0 must contain one initial row, 128 update rows, and one terminal "
            f"row; found {len(rows)} rows"
        )
    trainer_steps = [
        _as_integral(row.get("trainer/step"), label="trainer/step") for row in rows
    ]
    expected_trainer_steps = list(range(EXPECTED_UPDATES + 2))
    if trainer_steps != expected_trainer_steps:
        raise GateError(
            "C0 trainer steps are not the exact contiguous sequence 0--129"
        )
    global_steps = [
        _as_integral(row.get("trainer/global_step"), label="trainer/global_step")
        for row in rows
    ]
    expected_global_steps = [0, *range(1, EXPECTED_UPDATES + 1), EXPECTED_UPDATES]
    if global_steps != expected_global_steps:
        raise GateError(
            "C0 global steps must be updates 0--128 plus one terminal alias of 128"
        )

    initial = rows[0]
    _check_forbidden_telemetry(initial, step=0)
    if _finite(initial, "misc/prompt_dataset_len", step=0) != 192:
        raise GateError("C0 did not retain the frozen 192-prompt training pool")
    if _finite(initial, "misc/query_step", step=0) != 0:
        raise GateError("C0 initial query counter is not zero")
    if _finite(initial, "misc/prompt_consumed", step=0) != 0:
        raise GateError("C0 initial prompt counter is not zero")
    if _finite(initial, "eval/multi_answer/eval_count", step=0) != EXPECTED_EVAL_ROWS:
        raise GateError("C0 initial evaluation did not cover all 96 prompts")
    if _finite(initial, "eval/multi_answer/response_tok_len", step=0) != 3:
        raise GateError("C0 initial canonical evaluation response length is not three")

    eval_steps = [
        trainer_step
        for trainer_step, row in zip(trainer_steps, rows, strict=True)
        if "eval/multi_answer/eval_count" in row
    ]
    expected_eval_steps = [0, 32, 64, 96, 128, 129]
    if eval_steps != expected_eval_steps:
        raise GateError(
            f"C0 evaluation-bearing rows are {eval_steps}; expected {expected_eval_steps}"
        )
    for row, step in zip(rows, trainer_steps, strict=True):
        if step not in expected_eval_steps:
            continue
        if _finite(row, "eval/multi_answer/eval_count", step=step) != EXPECTED_EVAL_ROWS:
            raise GateError(f"C0 evaluation step {step} does not cover all 96 prompts")
        if _finite(row, "eval/multi_answer/response_tok_len", step=step) != 3:
            raise GateError(f"C0 evaluation step {step} response length is not three")

    update_rows = rows[1 : EXPECTED_UPDATES + 1]
    diagnostics = [
        _inspect_update_row(row, step=step)
        for step, row in enumerate(update_rows, start=1)
    ]
    policy_steps = [
        _finite(row, "trainer/policy_sgd_step", step=step)
        for step, row in enumerate(update_rows, start=1)
    ]
    if policy_steps != [float(step) for step in range(1, EXPECTED_UPDATES + 1)]:
        raise GateError("C0 policy-SGD steps are not contiguous updates 1--128")
    all_policy_steps = [
        _finite(row, "trainer/policy_sgd_step", step=step)
        for row, step in zip(rows, trainer_steps, strict=True)
    ]
    expected_policy_steps = [0.0, *map(float, range(1, 129)), 128.0]
    if all_policy_steps != expected_policy_steps:
        raise GateError("C0 terminal policy-SGD alias does not preserve update 128")

    # Explicitly name the preregistered final window; the exact full sequence
    # above ensures these are updates 97--128 rather than an arbitrary tail.
    tail = diagnostics[-32:]
    final_32_mean_reward = _mean(
        [record["reward"] for record in tail], label="final-32 rollout reward"
    )
    if final_32_mean_reward <= 0:
        raise GateError(
            f"C0 final-32 mean rollout reward {final_32_mean_reward} is not positive"
        )

    terminal = rows[-1]
    _check_forbidden_telemetry(terminal, step=EXPECTED_UPDATES + 1)
    if (
        _finite(
            terminal,
            "eval/multi_answer/eval_count",
            step=EXPECTED_UPDATES + 1,
        )
        != EXPECTED_EVAL_ROWS
    ):
        raise GateError("C0 terminal evaluation did not cover all 96 prompts")
    if (
        _finite(
            terminal,
            "eval/multi_answer/response_tok_len",
            step=EXPECTED_UPDATES + 1,
        )
        != 3
    ):
        raise GateError("C0 terminal canonical evaluation response length is not three")

    checkpoint = metrics_path.parent / "saved_models" / "step_00128"
    if not checkpoint.is_dir():
        raise GateError(f"C0 lacks the preregistered endpoint checkpoint: {checkpoint}")
    saved_models = metrics_path.parent / "saved_models"
    observed_checkpoint_tags = sorted(
        path.name for path in saved_models.iterdir() if path.is_dir()
    )
    expected_checkpoint_tags = [
        "step_00032",
        "step_00064",
        "step_00096",
        "step_00128",
        "step_00129",
    ]
    archived_tags = ["step_00128", "step_00129"]
    archive_authorized = (
        archived_removed_steps is not None
        and list(archived_removed_steps)
        == ["step_00032", "step_00064", "step_00096"]
        and observed_checkpoint_tags == archived_tags
    )
    if observed_checkpoint_tags != expected_checkpoint_tags and not archive_authorized:
        raise GateError(
            "C0 checkpoint schedule drifted; "
            f"observed={observed_checkpoint_tags} expected={expected_checkpoint_tags}"
        )
    endpoint_alias = saved_models / "step_00129"
    endpoint_hash, endpoint_files = _weight_identity(checkpoint)
    alias_hash, alias_files = _weight_identity(endpoint_alias)
    if endpoint_hash != alias_hash or endpoint_files != alias_files:
        raise GateError("C0 step_00129 is not a byte-identical weight alias of step_00128")

    eval_results = metrics_path.parent / "eval_results"
    observed_eval_files = sorted(path.name for path in eval_results.glob("*.json"))
    expected_eval_files = sorted(
        f"{step}_multi_answer.json" for step in (0, 32, 64, 96, 128, 129)
    )
    if observed_eval_files != expected_eval_files:
        raise GateError(
            "C0 evaluation artifact schedule drifted; "
            f"observed={observed_eval_files} expected={expected_eval_files}"
        )
    if _finite(terminal, "misc/query_step", step=129) != 2048:
        raise GateError("C0 terminal query counter is not the update-128 alias")
    if _finite(terminal, "misc/prompt_consumed", step=129) != 2048:
        raise GateError("C0 terminal prompt counter is not the update-128 alias")
    return (
        {
            "metrics_rows": len(rows),
            "optimizer_updates": EXPECTED_UPDATES,
            "contiguous_finite_tail": [97, 128],
            "final_32_mean_rollout_reward": final_32_mean_reward,
            "behavior_q_norm_error_max": max(
                max(record["actor_norm"], record["learner_norm"])
                for record in diagnostics
            ),
            "behavior_ratio_min": min(record["ratio_min"] for record in diagnostics),
            "behavior_ratio_max": max(record["ratio_max"] for record in diagnostics),
            "behavior_tv_max": max(record["tv_max"] for record in diagnostics),
            "behavior_kl_actor_learner_max": max(
                record["kl_actor_max"] for record in diagnostics
            ),
            "behavior_kl_learner_actor_max": max(
                record["kl_learner_max"] for record in diagnostics
            ),
            "behavior_sequence_ess_fraction_min": min(
                record["sequence_ess"] for record in diagnostics
            ),
            "behavior_prefix_ess_fraction_min": min(
                record["prefix_ess"] for record in diagnostics
            ),
            "step_128_weights_manifest_sha256": endpoint_hash,
            "step_129_byte_identical_alias": True,
        },
        metrics_path,
        checkpoint,
    )


def _parse_config(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for match in CONFIG_FIELD.finditer(text):
        key = match.group("key")
        value = match.group("value").strip()
        if key in EXPECTED_CONFIG and key in values and values[key] != value:
            raise GateError(
                f"runtime log reports conflicting values for {key}: "
                f"{values[key]!r} and {value!r}"
            )
        values[key] = value
    return values


def _exact_logged_steps(text: str, pattern: str, *, label: str) -> None:
    observed = [int(value) for value in re.findall(pattern, text)]
    expected = list(range(1, EXPECTED_UPDATES + 1))
    if observed != expected:
        raise GateError(f"runtime log lacks exact contiguous {label} steps 1--128")


def inspect_logs(
    stdout_path: Path,
    stderr_path: Path | None,
    *,
    run_dir: Path,
    source_root: Path,
    job_id: str,
) -> dict[str, Any]:
    if not stdout_path.is_file():
        raise GateError(f"Slurm stdout log is missing: {stdout_path}")
    stdout = ANSI_ESCAPE.sub(
        "", stdout_path.read_text(encoding="utf-8", errors="replace")
    )
    stderr = ""
    if stderr_path is not None:
        if not stderr_path.is_file():
            raise GateError(f"Slurm stderr log is missing: {stderr_path}")
        stderr = ANSI_ESCAPE.sub(
            "", stderr_path.read_text(encoding="utf-8", errors="replace")
        )
    combined = stdout + "\n" + stderr
    for pattern in FATAL_LOG_PATTERNS:
        if pattern in combined:
            raise GateError(f"runtime log contains fatal signature {pattern!r}")
    bad_exit = re.search(r"\[watchdog\] training exited status=(?!0(?:\D|$))\d+", combined)
    if bad_exit:
        raise GateError(f"runtime log contains {bad_exit.group(0)!r}")

    required_patterns = {
        "canonical_prompt_materialization": (
            r"canonical prompt materialization verified:\s*rows=192\s+"
            r"template=qwen_graph_digits\s+dataset_map_cache=disabled"
        ),
        "canonical_actor": (
            r"canonical graph actor configured:.*token_ids=\(16, 17, 18\).*"
            r"action_count=3.*vllm_engine=v0"
        ),
        "canonical_learner": (
            r"canonical graph policy: action_token_ids=\(16, 17, 18\) horizon=3"
        ),
        "learner_side_training_sampler": (
            r"\[train\] canonical_graph_actions=1 action_count=3 "
            r"learner_sampling=1 fixed_shape_sampling=1(?:\s|$)"
        ),
        "restart_disabled": (
            r"\[watchdog\].*requeue=0 restart_count=0/\d+"
        ),
        "vllm_v0": r"\[slurm\] vllm_use_v1=0",
        "step_128_checkpoint": (
            r"Checkpoint boundary at step 128: saving checkpoint before evaluation\."
        ),
    }
    for label, pattern in required_patterns.items():
        if re.search(pattern, stdout) is None:
            raise GateError(f"runtime log lacks {label} evidence")

    sampler_rows = re.findall(
        r"canonical learner sampler finished data_len=16 seed=\d+ "
        r"normalization_error_max=.*fixed_shape=1",
        stdout,
    )
    if len(sampler_rows) != EXPECTED_UPDATES:
        raise GateError(
            "runtime log must contain exactly 128 completed canonical learner samples; "
            f"found {len(sampler_rows)}"
        )
    _exact_logged_steps(
        stdout, r"post-learning done step=(\d+)", label="post-learning"
    )
    _exact_logged_steps(stdout, r"eval/log done step=(\d+)", label="eval/log")

    if EXPECTED_RUNTIME_IDENTITY["tokenizer_revision"] not in stdout:
        raise GateError("runtime log does not show the frozen model/tokenizer revision")
    if f"[slurm] oat_zero_source_root={source_root.resolve()}" not in stdout:
        raise GateError("runtime log source snapshot does not match the audited source")
    if f"[slurm] job_id={job_id}" not in stdout:
        raise GateError("runtime log does not belong to the audited Slurm allocation")
    if f"[experiment] save_path={run_dir.resolve()}" not in stdout:
        raise GateError("runtime log save path does not match the audited run directory")

    config = _parse_config(stdout)
    for key, expected in EXPECTED_CONFIG.items():
        raw = config.get(key)
        if raw is None:
            raise GateError(f"runtime log does not record configuration field {key}")
        if not _matches_config(raw, expected):
            raise GateError(
                f"runtime configuration {key}={raw!r}; expected {expected!r}"
            )
    return {
        "stdout": str(stdout_path.resolve()),
        "stderr": str(stderr_path.resolve()) if stderr_path is not None else None,
        "canonical_training_samples": len(sampler_rows),
        "post_learning_steps": [1, EXPECTED_UPDATES],
        "eval_log_steps": [1, EXPECTED_UPDATES],
        "vllm_engine": "v0",
        "recovery": "disabled",
    }


def _weight_identity(checkpoint: Path) -> tuple[str, list[dict[str, str]]]:
    paths = sorted(checkpoint.glob("*.safetensors"), key=lambda path: path.name)
    if not paths:
        raise GateError("C0 endpoint checkpoint contains no safetensors weights")
    digest = hashlib.sha256()
    records: list[dict[str, str]] = []
    for path in paths:
        file_hash = _sha256_file(path)
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(file_hash))
        records.append({"name": path.name, "sha256": file_hash})
    return digest.hexdigest(), records


def _number(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise GateError(f"endpoint audit {label} is not numeric") from error
    if not math.isfinite(number):
        raise GateError(f"endpoint audit {label} is nonfinite")
    return number


def _validate_prompt_record(record: Any, *, index: int) -> dict[str, float | int]:
    if not isinstance(record, dict):
        raise GateError(f"endpoint audit prompt {index} is not an object")
    if _as_integral(record.get("prompt_index"), label="prompt_index") != index:
        raise GateError(f"endpoint audit prompt index {index} is out of order")
    problem_hash = record.get("problem_sha256")
    if not isinstance(problem_hash, str) or re.fullmatch(r"[0-9a-f]{64}", problem_hash) is None:
        raise GateError(f"endpoint audit prompt {index} has invalid problem hash")
    if _as_integral(record.get("prompt_token_count"), label="prompt_token_count") <= 0:
        raise GateError(f"endpoint audit prompt {index} has no prompt tokens")

    leaves = record.get("leaves")
    if not isinstance(leaves, list) or len(leaves) != EXPECTED_LEAF_COUNT:
        raise GateError(f"endpoint audit prompt {index} does not contain 27 leaves")
    probabilities: list[float] = []
    rewards: list[int] = []
    q_plus_values: list[float] = []
    leaf_entropies: list[float] = []
    for leaf_index, (leaf, expected_action) in enumerate(
        zip(leaves, EXPECTED_ACTIONS, strict=True)
    ):
        if not isinstance(leaf, dict) or leaf.get("action") != expected_action:
            raise GateError(
                f"endpoint audit prompt {index} leaf {leaf_index} action order drifted"
            )
        expected_ids = [EXPECTED_ACTION_TOKEN_IDS[int(digit) - 1] for digit in expected_action]
        if leaf.get("action_token_ids") != expected_ids:
            raise GateError(
                f"endpoint audit prompt {index} leaf {expected_action} token IDs drifted"
            )
        reward = _as_integral(leaf.get("grader_reward"), label="grader_reward")
        if reward not in (0, 1):
            raise GateError(
                f"endpoint audit prompt {index} leaf {expected_action} has nonbinary reward"
            )
        log_probability = _number(
            leaf.get("leaf_log_probability"), label="leaf_log_probability"
        )
        probability = _number(
            leaf.get("leaf_probability"), label="leaf_probability"
        )
        if probability < 0 or probability > 1:
            raise GateError(
                f"endpoint audit prompt {index} leaf {expected_action} probability is invalid"
            )
        _require_close(
            probability,
            math.exp(log_probability),
            label=f"prompt {index} leaf {expected_action} probability/log-probability",
            rel_tol=1e-8,
            abs_tol=1e-12,
        )
        q_plus = _number(leaf.get("q_plus"), label="q_plus")
        if q_plus < 0 or q_plus > 1:
            raise GateError(
                f"endpoint audit prompt {index} leaf {expected_action} q_plus is invalid"
            )
        selected = leaf.get("teacher_forced_selected_log_probabilities")
        if not isinstance(selected, list) or len(selected) != 3:
            raise GateError(
                f"endpoint audit prompt {index} leaf {expected_action} lacks three teacher-forced log probabilities"
            )
        selected_values = [
            _number(value, label="teacher_forced_selected_log_probability")
            for value in selected
        ]
        _require_close(
            _number(
                leaf.get("teacher_forced_sequence_log_probability"),
                label="teacher_forced_sequence_log_probability",
            ),
            math.fsum(selected_values),
            label=f"prompt {index} leaf {expected_action} teacher-forced sequence sum",
        )
        probabilities.append(probability)
        rewards.append(reward)
        q_plus_values.append(q_plus)
        leaf_entropies.append(-probability * log_probability)

    probability_sum = math.fsum(probabilities)
    p_valid = math.fsum(
        probability * reward
        for probability, reward in zip(probabilities, rewards, strict=True)
    )
    valid_count = sum(rewards)
    if valid_count <= 0 or p_valid <= 0:
        raise GateError(f"endpoint audit prompt {index} has no valid policy mass")
    expected_q_plus = [
        probability / p_valid if reward else 0.0
        for probability, reward in zip(probabilities, rewards, strict=True)
    ]
    for leaf_index, (observed, expected) in enumerate(
        zip(q_plus_values, expected_q_plus, strict=True)
    ):
        _require_close(
            observed,
            expected,
            label=f"prompt {index} leaf {leaf_index} q_plus",
            rel_tol=1e-8,
            abs_tol=1e-10,
        )
    h_valid = -math.fsum(
        value * math.log(value) for value in expected_q_plus if value > 0
    )
    exact_entropy = math.fsum(leaf_entropies)
    declared_count = _as_integral(
        record.get("declared_valid_action_count"), label="declared_valid_action_count"
    )
    audited_count = _as_integral(
        record.get("valid_action_count"), label="valid_action_count"
    )
    if declared_count != valid_count or audited_count != valid_count:
        raise GateError(f"endpoint audit prompt {index} valid-action count drifted")
    _require_close(
        _number(record.get("probability_sum"), label="probability_sum"),
        probability_sum,
        label=f"prompt {index} probability sum",
    )
    _require_close(
        _number(record.get("probability_sum_abs_error"), label="probability_sum_abs_error"),
        abs(probability_sum - 1.0),
        label=f"prompt {index} probability-sum error",
    )
    _require_close(
        _number(record.get("exact_action_entropy"), label="exact_action_entropy"),
        exact_entropy,
        label=f"prompt {index} exact action entropy",
        rel_tol=1e-8,
    )
    conditional_entropy = _number(
        record.get("conditional_entropy"), label="conditional_entropy"
    )
    entropy_error = abs(exact_entropy - conditional_entropy)
    _require_close(
        _number(record.get("entropy_identity_abs_error"), label="entropy_identity_abs_error"),
        entropy_error,
        label=f"prompt {index} entropy identity error",
    )
    _require_close(
        _number(record.get("p_valid"), label="p_valid"),
        p_valid,
        label=f"prompt {index} P_valid",
        rel_tol=1e-8,
    )
    _require_close(
        _number(record.get("log_p_valid"), label="log_p_valid"),
        math.log(p_valid),
        label=f"prompt {index} log P_valid",
        rel_tol=1e-8,
    )
    _require_close(
        _number(record.get("h_valid"), label="h_valid"),
        h_valid,
        label=f"prompt {index} H_valid",
        rel_tol=1e-8,
    )
    _require_close(
        _number(record.get("n_eff_valid"), label="n_eff_valid"),
        math.exp(h_valid),
        label=f"prompt {index} N_eff_valid",
        rel_tol=1e-8,
    )
    token_error = _number(
        record.get("teacher_forced_vs_prefix_tree_per_token_max_abs_error"),
        label="teacher-forced per-token error",
    )
    sequence_error = _number(
        record.get("teacher_forced_vs_prefix_tree_sequence_max_abs_error"),
        label="teacher-forced sequence error",
    )
    return {
        "p_valid": p_valid,
        "h_valid": h_valid,
        "n_eff_valid": math.exp(h_valid),
        "exact_entropy": exact_entropy,
        "conditional_entropy": conditional_entropy,
        "valid_count": valid_count,
        "probability_error": abs(probability_sum - 1.0),
        "entropy_error": entropy_error,
        "token_error": token_error,
        "sequence_error": sequence_error,
    }


def inspect_endpoint_audit(
    audit_path: Path,
    *,
    checkpoint: Path,
    expected_source_root: Path | None = None,
    expected_source_hash: str | None = None,
) -> dict[str, Any]:
    if not audit_path.is_file():
        raise GateError(f"E14 endpoint audit is missing: {audit_path}")
    payload = _strict_json(
        audit_path.read_text(encoding="utf-8"), context=str(audit_path)
    )
    if not isinstance(payload, dict):
        raise GateError("E14 endpoint audit is not an object")
    if payload.get("schema") != "e14_exact_endpoint_audit_v1":
        raise GateError("E14 endpoint audit has the wrong schema")
    if payload.get("status") != "pass":
        raise GateError("E14 endpoint audit does not have status=pass")
    if payload.get("formulation") != "canonical_graph_actions":
        raise GateError("E14 endpoint audit has the wrong formulation")
    source = payload.get("source")
    if expected_source_root is not None or expected_source_hash is not None:
        if expected_source_root is None or expected_source_hash is None:
            raise GateError("both expected audit source root and hash are required")
        if not isinstance(source, dict):
            raise GateError("E14 endpoint audit does not record its source snapshot")
        if Path(str(source.get("root", ""))).resolve() != expected_source_root.resolve():
            raise GateError("E14 endpoint audit did not use the C0 source snapshot")
        if source.get("python_source_sha256") != expected_source_hash:
            raise GateError("E14 endpoint audit source hash differs from C0")

    checkpoint_record = payload.get("checkpoint")
    if not isinstance(checkpoint_record, dict):
        raise GateError("E14 endpoint audit has no checkpoint record")
    if Path(str(checkpoint_record.get("path", ""))).resolve() != checkpoint.resolve():
        raise GateError("E14 endpoint audit is not bound to C0 step_00128")
    if checkpoint_record.get("oat_step_tag") != EXPECTED_UPDATES:
        raise GateError("E14 endpoint audit did not use OAT step_00128")
    if checkpoint_record.get("optimizer_updates") != EXPECTED_UPDATES:
        raise GateError("E14 endpoint audit has the wrong optimizer-update count")
    if checkpoint_record.get("role") != "scheduled_update_boundary":
        raise GateError("E14 endpoint audit used a terminal alias instead of step_00128")
    observed_manifest_hash, observed_weight_files = _weight_identity(checkpoint)
    if checkpoint_record.get("weights_manifest_sha256") != observed_manifest_hash:
        raise GateError("E14 endpoint checkpoint weights changed after exact audit")
    if checkpoint_record.get("weight_files") != observed_weight_files:
        raise GateError("E14 endpoint audit weight-file manifest disagrees with checkpoint")

    data = payload.get("data")
    expected_data = {
        "combined_content_hash": EXPECTED_DATASET_IDENTITY["combined_content_hash"],
        "eval_rows": EXPECTED_EVAL_ROWS,
        "split": "multi_answer",
    }
    if not isinstance(data, dict) or any(
        data.get(key) != value for key, value in expected_data.items()
    ):
        raise GateError("E14 endpoint audit used the wrong frozen evaluation data")

    policy = payload.get("policy")
    expected_policy = {
        "actions": ["1", "2", "3"],
        "action_token_ids": list(EXPECTED_ACTION_TOKEN_IDS),
        "horizon": 3,
        "leaf_count": EXPECTED_LEAF_COUNT,
        "max_action_entropy_nats": MAX_ACTION_ENTROPY,
        "tokenizer_files_hash": EXPECTED_RUNTIME_IDENTITY["tokenizer_files_hash"],
        "tokenizer_revision": EXPECTED_RUNTIME_IDENTITY["tokenizer_revision"],
        "tokenizer_vocab_hash": EXPECTED_RUNTIME_IDENTITY["tokenizer_vocab_hash"],
    }
    if not isinstance(policy, dict) or any(
        policy.get(key) != value for key, value in expected_policy.items()
    ):
        raise GateError("E14 endpoint audit canonical-policy identity drifted")

    tolerances = payload.get("tolerances")
    expected_tolerances = {
        "probability_sum_abs": 1e-5,
        "entropy_identity_abs_nats": 1e-5,
        "teacher_forced_log_probability_abs": 5e-3,
    }
    if tolerances != expected_tolerances:
        raise GateError("E14 endpoint audit tolerances drifted")

    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != EXPECTED_EVAL_ROWS:
        raise GateError("E14 endpoint audit does not contain all 96 prompts")
    records = [
        _validate_prompt_record(record, index=index)
        for index, record in enumerate(prompts)
    ]
    probability_error_max = max(float(row["probability_error"]) for row in records)
    entropy_error_max = max(float(row["entropy_error"]) for row in records)
    token_error_max = max(float(row["token_error"]) for row in records)
    sequence_error_max = max(float(row["sequence_error"]) for row in records)
    if probability_error_max > expected_tolerances["probability_sum_abs"]:
        raise GateError("E14 endpoint audit probability normalization failed")
    if entropy_error_max > expected_tolerances["entropy_identity_abs_nats"]:
        raise GateError("E14 endpoint audit entropy identity failed")
    if token_error_max > expected_tolerances["teacher_forced_log_probability_abs"]:
        raise GateError("E14 endpoint audit teacher-forced token crosscheck failed")
    if sequence_error_max > expected_tolerances["teacher_forced_log_probability_abs"]:
        raise GateError("E14 endpoint audit teacher-forced sequence crosscheck failed")

    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, dict):
        raise GateError("E14 endpoint audit has no aggregate record")
    p_valid_values = [float(row["p_valid"]) for row in records]
    valid_counts = [int(row["valid_count"]) for row in records]
    recomputed = {
        "prompt_count": EXPECTED_EVAL_ROWS,
        "leaf_count": EXPECTED_EVAL_ROWS * EXPECTED_LEAF_COUNT,
        "exact_action_entropy_mean": _mean(
            [float(row["exact_entropy"]) for row in records],
            label="exact action entropy",
        ),
        "conditional_entropy_mean": _mean(
            [float(row["conditional_entropy"]) for row in records],
            label="conditional entropy",
        ),
        "p_valid_mean": _mean(p_valid_values, label="P_valid"),
        "p_valid_min": min(p_valid_values),
        "h_valid_mean": _mean(
            [float(row["h_valid"]) for row in records], label="H_valid"
        ),
        "n_eff_valid_mean": _mean(
            [float(row["n_eff_valid"]) for row in records], label="N_eff_valid"
        ),
        "valid_action_count_mean": _mean(
            [float(value) for value in valid_counts], label="valid action count"
        ),
        "probability_sum_max_abs_error": probability_error_max,
        "entropy_identity_max_abs_error": entropy_error_max,
        "teacher_forced_vs_prefix_tree_per_token_max_abs_error": token_error_max,
        "teacher_forced_vs_prefix_tree_sequence_max_abs_error": sequence_error_max,
    }
    for key, expected in recomputed.items():
        if isinstance(expected, int):
            if aggregate.get(key) != expected:
                raise GateError(f"E14 endpoint aggregate {key} disagrees with prompts")
        else:
            _require_close(
                _number(aggregate.get(key), label=key),
                expected,
                label=f"endpoint aggregate {key}",
                rel_tol=1e-8,
                abs_tol=1e-10,
            )
    expected_histogram = {
        str(key): value for key, value in sorted(Counter(valid_counts).items())
    }
    if aggregate.get("valid_action_count_histogram") != expected_histogram:
        raise GateError("E14 endpoint valid-action histogram disagrees with prompts")
    p_valid_mean = recomputed["p_valid_mean"]
    if not p_valid_mean > MIN_ENDPOINT_P_VALID:
        raise GateError(
            f"C0 endpoint mean exact P_valid {p_valid_mean} does not exceed "
            f"{MIN_ENDPOINT_P_VALID}"
        )
    return {
        "audit": str(audit_path.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_weights_manifest_sha256": observed_manifest_hash,
        **recomputed,
        "valid_action_count_histogram": expected_histogram,
    }


def _discover_job_id(repo_root: Path, *, stamp: str, run_dir: Path) -> str:
    manifest = repo_root / "var" / "artifacts" / f"{stamp}_comparative_jobs.tsv"
    if not manifest.is_file():
        raise GateError(f"E14 C0 submission manifest is missing: {manifest}")
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected_run_stamp = f"{stamp}_grpo_s9005"
    matches = [
        row
        for row in rows
        if row.get("arm") == "grpo"
        and row.get("seed") == "9005"
        and row.get("run_stamp") == expected_run_stamp
    ]
    if len(matches) != 1 or not matches[0].get("job_id"):
        raise GateError("submission manifest does not identify one E14 C0 job")
    if not run_dir.name.endswith(f"_{expected_run_stamp}"):
        raise GateError("C0 run directory does not match its stamp/arm/seed")
    return str(matches[0]["job_id"])


def check_c0(
    *,
    run_dir: Path,
    identity_path: Path,
    preflight_approval_path: Path,
    endpoint_audit_path: Path,
    source_root: Path,
    stdout_path: Path,
    stderr_path: Path | None,
    slurm_state: str,
    slurm_exit_code: str,
    job_id: str,
    logical_repo_root: Path,
    archived_removed_steps: Sequence[str] | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise GateError(f"C0 run directory is missing: {run_dir}")
    identity, dataset, runtime = read_identity(identity_path)
    if identity["phase"] != "c0":
        raise GateError(f"identity phase is {identity['phase']!r}, not 'c0'")
    stamp = identity["stamp"]
    if identity_path.name != f"{stamp}_e14_identity.tsv":
        raise GateError("C0 identity filename does not match its stamp")
    if not run_dir.name.endswith(f"_{stamp}_grpo_s9005"):
        raise GateError("C0 run directory does not match its identity stamp/arm/seed")
    expected_identity = {
        "target_optimizer_updates": str(EXPECTED_UPDATES),
        "trajectory_query_budget": str(EXPECTED_QUERY_BUDGET),
        "group_size": str(EXPECTED_GROUP_SIZE),
    }
    for key, expected in expected_identity.items():
        if identity[key] != expected:
            raise GateError(f"identity {key}={identity[key]!r}; expected {expected!r}")
    if re.fullmatch(r"[0-9a-f]{64}", identity["source_hash"]) is None:
        raise GateError("C0 identity source_hash is not a lowercase SHA-256 digest")
    observed_source_hash = source_tree_hash(
        source_root, logical_repo_root=logical_repo_root
    )
    if identity["source_hash"] != observed_source_hash:
        raise GateError("C0 immutable source snapshot does not match its source hash")
    preflight_summary = verify_approval_for_source(
        preflight_approval_path,
        expected_source_hash=observed_source_hash,
        logical_repo_root=logical_repo_root,
    )
    if slurm_state.strip() != "COMPLETED" or slurm_exit_code.strip() != "0:0":
        raise GateError(
            f"Slurm allocation {job_id} is not a clean terminal success: "
            f"state={slurm_state!r} exit_code={slurm_exit_code!r}"
        )

    log_summary = inspect_logs(
        stdout_path,
        stderr_path,
        run_dir=run_dir,
        source_root=source_root,
        job_id=job_id,
    )
    metrics_summary, metrics_path, checkpoint = inspect_metrics(
        run_dir, archived_removed_steps=archived_removed_steps
    )
    endpoint_summary = inspect_endpoint_audit(
        endpoint_audit_path,
        checkpoint=checkpoint,
        expected_source_root=source_root,
        expected_source_hash=observed_source_hash,
    )
    evidence_paths = {
        "identity": identity_path,
        "preflight_approval": preflight_approval_path,
        "metrics": metrics_path,
        "source_snapshot_marker": source_root / "oat_drgrpo" / "__init__.py",
        "stdout": stdout_path,
        "endpoint_audit": endpoint_audit_path,
        "checkpoint_config": checkpoint / "config.json",
    }
    if stderr_path is not None:
        evidence_paths["stderr"] = stderr_path
    for label, path in evidence_paths.items():
        if not path.is_file():
            raise GateError(f"C0 evidence {label!r} is missing: {path}")
    evidence = {
        key: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
        for key, path in evidence_paths.items()
    }
    return {
        "approved": True,
        "gate": GATE_NAME,
        "protocol": "E14",
        "arm": "C0",
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": job_id,
        "run_dir": str(run_dir),
        "slurm": {"state": "COMPLETED", "exit_code": "0:0"},
        "identity": {
            "stamp": stamp,
            "source_hash": observed_source_hash,
            "dataset": dataset,
            "runtime": runtime,
        },
        "checks": {
            "frozen_identity": True,
            "immutable_source": True,
            "terminal_success": True,
            "preflight_authorization_revalidated": True,
            "updates_97_128_contiguous_finite": True,
            "canonical_rollouts_valid": True,
            "behavior_policy_overlap": True,
            "final_32_mean_reward_positive": True,
            "forbidden_treatments_inactive": True,
            "exact_step_00128_audit": True,
            "endpoint_mean_p_valid_above_0p05": True,
        },
        "log_summary": log_summary,
        "preflight_summary": preflight_summary,
        "metrics_summary": metrics_summary,
        "endpoint_summary": endpoint_summary,
        "evidence": evidence,
    }


def write_approval(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Approve E14 C0 after its exact step_00128 checkpoint audit and all "
            "preregistered runtime gates pass."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--preflight-approval", type=Path, required=True)
    parser.add_argument("--endpoint-audit", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--stdout-log", type=Path)
    parser.add_argument("--stderr-log", type=Path)
    parser.add_argument("--slurm-status", type=Path)
    parser.add_argument("--slurm-state")
    parser.add_argument("--slurm-exit-code")
    parser.add_argument("--job-id")
    parser.add_argument("--approval-out", type=Path)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    identity, _, _ = read_identity(args.identity)
    stamp = identity["stamp"]
    approval_out = args.approval_out or (
        repo_root / "var" / "artifacts" / f"{stamp}_e14_c0_approval.json"
    )
    approval_out.unlink(missing_ok=True)

    source_root = args.source_root or (
        repo_root / "var" / "artifacts" / "source_snapshots" / stamp / "src"
    )
    job_id = args.job_id or _discover_job_id(
        repo_root, stamp=stamp, run_dir=args.run_dir
    )
    stdout_path = args.stdout_log or (
        repo_root / "var" / "artifacts" / "logs" / f"xdr_train-{job_id}.out"
    )
    stderr_path = args.stderr_log
    if stderr_path is None:
        candidate = (
            repo_root / "var" / "artifacts" / "logs" / f"xdr_train-{job_id}.err"
        )
        stderr_path = candidate if candidate.exists() else None

    if args.slurm_status is not None:
        state, exit_code = parse_slurm_status(
            args.slurm_status, expected_job_id=job_id
        )
    elif args.slurm_state is not None or args.slurm_exit_code is not None:
        if args.slurm_state is None or args.slurm_exit_code is None:
            raise GateError("both --slurm-state and --slurm-exit-code are required")
        state, exit_code = args.slurm_state, args.slurm_exit_code
    else:
        state, exit_code = query_slurm(job_id)

    payload = check_c0(
        run_dir=args.run_dir,
        identity_path=args.identity,
        preflight_approval_path=args.preflight_approval,
        endpoint_audit_path=args.endpoint_audit,
        source_root=source_root,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        slurm_state=state,
        slurm_exit_code=exit_code,
        job_id=job_id,
        logical_repo_root=repo_root,
    )
    write_approval(approval_out, payload)
    print(f"E14 C0 approved: {approval_out.resolve()}")


if __name__ == "__main__":
    main()
