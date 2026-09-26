#!/usr/bin/env python3
"""Fail-closed approval gate for E14's one-update canonical preflight.

The approval JSON is deliberately separate from the training outputs.  It is
written atomically, and only after the frozen identities, runtime log,
telemetry, and terminal Slurm status all pass.  A failed re-check removes the
requested approval path first so stale approval cannot authorize C0.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GATE_NAME = "e14_canonical_zero_learning_preflight"
REQUIRED_APPROVAL_CHECKS = {
    "frozen_identity",
    "immutable_source",
    "terminal_success",
    "canonical_prompt_materialization",
    "canonical_action_runtime",
    "exactly_one_optimizer_update",
    "zero_learning",
    "behavior_policy_overlap",
    "learner_side_behavior_sampling",
    "fixed_shape_behavior_sampling",
    "forbidden_treatments_inactive",
}
EXPECTED_BEHAVIOR_Q_ROWS = 16 * 3
BEHAVIOR_Q_NORM_ERROR_MAX = 1e-6
BEHAVIOR_RATIO_MIN = 0.8
BEHAVIOR_RATIO_MAX = 1.2
BEHAVIOR_TV_MAX = 0.05
BEHAVIOR_KL_MAX = 0.01
BEHAVIOR_ESS_FRACTION_MIN = 0.90
EXPECTED_SOURCE_KEYS = {
    "phase",
    "stamp",
    "source_hash",
    "dataset_identity",
    "runtime_identity",
    "target_optimizer_updates",
    "trajectory_query_budget",
    "group_size",
}
EXPECTED_DATASET_IDENTITY = {
    "combined_content_hash": (
        "8e8bd8d4986920784cb067f99f4dc1b401f553b0d40a378dd4ada5dab29b48d6"
    ),
    "eval_content_hash": (
        "f2c5888515e10204329a276302ddfaefba286436790a58582bc52b9a976af3eb"
    ),
    "eval_rows": 96,
    "hidden_nodes_per_row": 3,
    "train_content_hash": (
        "55d1f49cbf150b3685fbc73d0862226cf02e91a24fbc35fbcb5dffa6142cbd00"
    ),
    "train_rows": 192,
}
EXPECTED_RUNTIME_IDENTITY = {
    "action_count": 3,
    "action_token_ids": [16, 17, 18],
    "canonical_training_sampler": (
        "learner_hf_restricted_inverse_cdf_fixed_shape_causal_placeholder"
    ),
    "engine": "v0",
    "max_action_entropy_nats": math.log(27),
    "model_weights_hash": (
        "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
    ),
    "sampler_module": "vllm.model_executor.layers.sampler",
    "tokenizer": "Qwen/Qwen2.5-0.5B-Instruct",
    "tokenizer_class": "Qwen2TokenizerFast",
    "tokenizer_files_hash": (
        "caa4fecabf4ddfe3d6678b909ca31e73337cfbfd9aa6befc935a9d1d90ca089d"
    ),
    "tokenizer_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
    "tokenizer_vocab_hash": (
        "698c955b0b438a535083d1771ef8b41069afba4b3d51a482558bdb68ea55e800"
    ),
    "tokenizer_vocab_size": 151665,
    "vllm_version": "0.8.4",
    "vllm_role": "evaluation_only",
}
EXPECTED_CONFIG: dict[str, Any] = {
    "beta": 0.0,
    "canonical_graph_action_count": 3,
    "canonical_graph_actions": True,
    "canonical_graph_fixed_shape_sampling": True,
    "canonical_graph_learner_sampling": True,
    "ignore_no_eos": False,
    "kl_penalty_coef": 0.0,
    "learning_rate": 0.0,
    "max_queries": 1,
    "max_train": 192,
    "maxent_alpha": 0.0,
    "maxent_control_target_ratio": 0.0,
    "maxent_dual_target_ratio": 0.0,
    "maxent_length_target": 0.0,
    "non_stop_fixed_reward": None,
    "non_stop_penalty": 0.0,
    "num_ppo_epochs": 1,
    "num_samples": 16,
    "policy_entropy_coef": 0.0,
    "prompt_template": "qwen_graph_digits",
    "seed": 9005,
    "seed_entropy_alpha": 0.0,
    "temperature": 1.0,
    "test_split": "multi_answer",
    "top_k": -1,
    "top_p": 1.0,
    "xdr_mode_adaptive": False,
    "xdr_sac_dual_target_ratio": 0.0,
    "xdr_tau": math.inf,
    "xdr_tau_control_target_ratio": 0.0,
}
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
CONFIG_FIELD = re.compile(
    r"(?m)^[^\n]*?│\s+['\"](?P<key>[A-Za-z0-9_]+)['\"]\s*:\s*"
    r"(?P<value>[^,\n}]+)"
)
FATAL_LOG_PATTERNS = (
    "Traceback (most recent call last):",
    "RuntimeError:",
    "ValueError:",
    "OUT_OF_MEMORY",
    "CANCELLED AT",
    "DUE TO TIME LIMIT",
    "[watchdog] fatal:",
)


class GateError(RuntimeError):
    """An E14 approval condition was not established."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_hash(source_root: Path, *, logical_repo_root: Path) -> str:
    """Reproduce the launcher's nested GNU sha256sum over ``src/**/*.py``.

    The launcher hashes the checkout before copying it to the immutable source
    snapshot.  GNU ``sha256sum`` includes the original absolute filename in the
    outer digest, so snapshot files are mapped back to ``<repo>/src/<relative>``.
    """

    source_root = source_root.resolve()
    files = sorted(
        (path for path in source_root.rglob("*.py") if path.is_file()),
        key=lambda path: path.relative_to(source_root).as_posix(),
    )
    if not files:
        raise GateError(f"source snapshot has no Python files: {source_root}")
    outer = hashlib.sha256()
    logical_src = logical_repo_root.resolve() / "src"
    for path in files:
        logical_path = logical_src / path.relative_to(source_root)
        outer.update(f"{_sha256_file(path)}  {logical_path}\n".encode())
    return outer.hexdigest()


def _strict_json(text: str, *, context: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value}")

    try:
        return json.loads(text, parse_constant=reject_constant)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise GateError(f"invalid JSON in {context}: {error}") from error


def read_identity(path: Path) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    if not path.is_file():
        raise GateError(f"identity artifact is missing: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if not rows or rows[0] != ["key", "value"]:
        raise GateError("identity artifact must begin with the exact key/value header")
    identity: dict[str, str] = {}
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) != 2 or not row[0]:
            raise GateError(f"malformed identity row {line_number}")
        if row[0] in identity:
            raise GateError(f"duplicate identity key {row[0]!r}")
        identity[row[0]] = row[1]
    if set(identity) != EXPECTED_SOURCE_KEYS:
        missing = sorted(EXPECTED_SOURCE_KEYS - set(identity))
        extra = sorted(set(identity) - EXPECTED_SOURCE_KEYS)
        raise GateError(f"identity fields drifted; missing={missing} extra={extra}")
    dataset = _strict_json(identity["dataset_identity"], context="dataset_identity")
    runtime = _strict_json(identity["runtime_identity"], context="runtime_identity")
    if dataset != EXPECTED_DATASET_IDENTITY:
        raise GateError("frozen E14 dataset identity does not match")
    if runtime != EXPECTED_RUNTIME_IDENTITY:
        raise GateError("frozen E14 runtime identity does not match")
    return identity, dataset, runtime


def _as_integral(value: Any, *, label: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise GateError(f"{label} is not numeric") from error
    if not math.isfinite(number) or not number.is_integer():
        raise GateError(f"{label} is not a finite integer: {value!r}")
    return int(number)


def _finite(row: dict[str, Any], key: str, *, step: int) -> float:
    if key not in row:
        raise GateError(f"step {step} is missing {key}")
    try:
        value = float(row[key])
    except (TypeError, ValueError) as error:
        raise GateError(f"step {step} has nonnumeric {key}") from error
    if not math.isfinite(value):
        raise GateError(f"step {step} has nonfinite {key}")
    return value


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise GateError(f"cannot read metrics {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        row = _strict_json(line, context=f"{path}:{line_number}")
        if not isinstance(row, dict):
            raise GateError(f"metrics row {line_number} is not an object")
        rows.append(row)
    return rows


def inspect_metrics(run_dir: Path) -> tuple[dict[str, Any], Path]:
    candidates = sorted(run_dir.glob("debug_*/train_metrics.jsonl"))
    if len(candidates) != 1:
        raise GateError(
            "restart-invalid preflight requires exactly one metrics stream; "
            f"found {len(candidates)}"
        )
    metrics_path = candidates[0]
    rows = _read_metrics(metrics_path)
    if len(rows) != 3:
        raise GateError(
            "successful one-update preflight must have initial, update, and "
            f"terminal telemetry rows; found {len(rows)}"
        )
    steps = [_as_integral(row.get("trainer/step"), label="trainer/step") for row in rows]
    if steps != [0, 1, 2]:
        raise GateError(f"expected exact OAT step sequence [0, 1, 2], got {steps}")
    global_steps = [
        _as_integral(row.get("trainer/global_step"), label="trainer/global_step")
        for row in rows
    ]
    if global_steps != [0, 1, 1]:
        raise GateError(
            "preflight must complete exactly one learning update; expected global "
            f"steps [0, 1, 1], got {global_steps}"
        )
    policy_steps = [
        _finite(row, "trainer/policy_sgd_step", step=step)
        for row, step in zip(rows, steps, strict=True)
    ]
    if not math.isclose(policy_steps[0], 0.0, abs_tol=1e-12) or not (
        policy_steps[1] > 0 and math.isclose(policy_steps[1], policy_steps[2], abs_tol=1e-12)
    ):
        raise GateError(f"policy-SGD progress is inconsistent with one update: {policy_steps}")

    initial = rows[0]
    if _finite(initial, "misc/prompt_dataset_len", step=0) != 192:
        raise GateError("canonical preflight did not retain all 192 training prompts")
    if _finite(initial, "misc/query_step", step=0) != 0:
        raise GateError("initial query counter is not zero")
    if _finite(initial, "misc/prompt_consumed", step=0) != 0:
        raise GateError("initial prompt counter is not zero")
    if _finite(initial, "eval/multi_answer/eval_count", step=0) != 96:
        raise GateError("initial evaluation did not cover all 96 frozen prompts")
    if _finite(initial, "eval/multi_answer/response_tok_len", step=0) != 3:
        raise GateError("initial canonical evaluation response length is not three")

    required_train_finite = (
        "actor/canonical_action_count",
        "actor/canonical_action_support_size",
        "actor/canonical_finish_length_count",
        "actor/canonical_finish_unexpected_count",
        "actor/canonical_graph_actions",
        "actor/canonical_invalid_count",
        "actor/canonical_sampler_learner",
        "actor/canonical_sampler_fixed_shape",
        "actor/canonical_behavior_q_norm_error_max",
        "actor/canonical_behavior_q_row_count",
        "actor/canonical_behavior_q_support_max",
        "actor/canonical_behavior_q_support_min",
        "actor/no_eos_count",
        "actor/num_data",
        "actor/response_tok_len",
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
    forbidden_prefixes = (
        "train/maxent_",
        "train/policy_entropy_",
        "train/seed_entropy_",
        "train/xdr_sac_",
        "train/xdr_tau_",
    )
    behavior_extrema = {
        "behavior_q_norm_error_max": 0.0,
        "behavior_ratio_min": math.inf,
        "behavior_ratio_max": -math.inf,
        "behavior_tv_max": 0.0,
        "behavior_kl_actor_learner_max": 0.0,
        "behavior_kl_learner_actor_max": 0.0,
        "behavior_sequence_ess_fraction_min": math.inf,
        "behavior_prefix_ess_fraction_min": math.inf,
    }
    for row, step in zip(rows[1:], steps[1:], strict=True):
        for key in required_train_finite:
            _finite(row, key, step=step)
        if _finite(row, "misc/lr", step=step) != 0:
            raise GateError(f"step {step} was not zero-learning")
        if _finite(row, "misc/query_step", step=step) != 16:
            raise GateError(f"step {step} does not represent one 16-action group")
        if _finite(row, "misc/prompt_consumed", step=step) != 16:
            raise GateError(f"step {step} has the wrong consumed-action count")
        exact_values = {
            "actor/canonical_action_count": 3,
            "actor/canonical_action_support_size": 3,
            "actor/canonical_finish_length_count": 16,
            "actor/canonical_finish_unexpected_count": 0,
            "actor/canonical_graph_actions": 1,
            "actor/canonical_invalid_count": 0,
            "actor/canonical_sampler_learner": 1,
            "actor/canonical_sampler_fixed_shape": 1,
            "actor/canonical_behavior_q_row_count": EXPECTED_BEHAVIOR_Q_ROWS,
            "actor/canonical_behavior_q_support_max": 3,
            "actor/canonical_behavior_q_support_min": 3,
            "actor/no_eos_count": 0,
            "actor/num_data": 16,
            "actor/response_tok_len": 3,
            "actor/sampling_max_tokens": 3,
            "actor/sampling_temperature": 1,
            "train/canonical_action_count": 3,
            "train/canonical_action_vocab_size": 3,
            "train/canonical_behavior_denominator_actor": 1,
            "train/canonical_behavior_q_row_count": EXPECTED_BEHAVIOR_Q_ROWS,
            "train/canonical_behavior_q_support_max": 3,
            "train/canonical_behavior_q_support_min": 3,
            "train/learning_round": 1,
        }
        for key, expected in exact_values.items():
            observed = _finite(row, key, step=step)
            if observed != expected:
                raise GateError(
                    f"step {step} has {key}={observed}; expected {expected}"
                )
        actor_norm_error = _finite(
            row, "actor/canonical_behavior_q_norm_error_max", step=step
        )
        learner_norm_error = _finite(
            row, "train/canonical_behavior_q_norm_error_max", step=step
        )
        if not 0 <= actor_norm_error <= BEHAVIOR_Q_NORM_ERROR_MAX:
            raise GateError(
                f"step {step} actor behavior-q normalization error "
                f"{actor_norm_error} exceeds {BEHAVIOR_Q_NORM_ERROR_MAX}"
            )
        if not 0 <= learner_norm_error <= BEHAVIOR_Q_NORM_ERROR_MAX:
            raise GateError(
                f"step {step} learner-received behavior-q normalization error "
                f"{learner_norm_error} exceeds {BEHAVIOR_Q_NORM_ERROR_MAX}"
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
                f"step {step} behavior-policy TV mean/max "
                f"{tv_mean}/{tv_max} violates maximum {BEHAVIOR_TV_MAX}"
            )

        kl_actor_learner_mean = _finite(
            row, "train/canonical_behavior_kl_actor_learner_mean", step=step
        )
        kl_actor_learner_max = _finite(
            row, "train/canonical_behavior_kl_actor_learner_max", step=step
        )
        kl_learner_actor_mean = _finite(
            row, "train/canonical_behavior_kl_learner_actor_mean", step=step
        )
        kl_learner_actor_max = _finite(
            row, "train/canonical_behavior_kl_learner_actor_max", step=step
        )
        for direction, mean, maximum in (
            ("actor||learner", kl_actor_learner_mean, kl_actor_learner_max),
            ("learner||actor", kl_learner_actor_mean, kl_learner_actor_max),
        ):
            if not 0 <= mean <= maximum <= BEHAVIOR_KL_MAX:
                raise GateError(
                    f"step {step} behavior-policy KL({direction}) mean/max "
                    f"{mean}/{maximum} violates maximum {BEHAVIOR_KL_MAX}"
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
                    f"step {step} behavior-policy {label} ESS fraction {value} "
                    f"is outside [{BEHAVIOR_ESS_FRACTION_MIN}, 1]"
                )

        behavior_extrema["behavior_q_norm_error_max"] = max(
            behavior_extrema["behavior_q_norm_error_max"],
            actor_norm_error,
            learner_norm_error,
        )
        behavior_extrema["behavior_ratio_min"] = min(
            behavior_extrema["behavior_ratio_min"], ratio_min
        )
        behavior_extrema["behavior_ratio_max"] = max(
            behavior_extrema["behavior_ratio_max"], ratio_max
        )
        behavior_extrema["behavior_tv_max"] = max(
            behavior_extrema["behavior_tv_max"], tv_max
        )
        behavior_extrema["behavior_kl_actor_learner_max"] = max(
            behavior_extrema["behavior_kl_actor_learner_max"],
            kl_actor_learner_max,
        )
        behavior_extrema["behavior_kl_learner_actor_max"] = max(
            behavior_extrema["behavior_kl_learner_actor_max"],
            kl_learner_actor_max,
        )
        behavior_extrema["behavior_sequence_ess_fraction_min"] = min(
            behavior_extrema["behavior_sequence_ess_fraction_min"], sequence_ess
        )
        behavior_extrema["behavior_prefix_ess_fraction_min"] = min(
            behavior_extrema["behavior_prefix_ess_fraction_min"], prefix_ess
        )
        entropy_sum = _finite(
            row, "train/canonical_sampled_prefix_entropy_sum", step=step
        )
        entropy_ratio = _finite(
            row, "train/canonical_sampled_prefix_entropy_ratio", step=step
        )
        if not 0 <= entropy_sum <= math.log(27) + 1e-5:
            raise GateError(f"step {step} canonical entropy is outside [0, log(27)]")
        if not math.isclose(
            entropy_ratio, entropy_sum / math.log(27), rel_tol=1e-5, abs_tol=1e-7
        ):
            raise GateError(f"step {step} canonical entropy ratio has wrong units")
        for key, value in row.items():
            if (key.endswith("_nan") or key.endswith("_inf")) and float(value) != 0:
                raise GateError(f"step {step} reports numerical failure in {key}")
            if key.startswith(forbidden_prefixes):
                raise GateError(f"step {step} contains forbidden treatment telemetry {key}")

    terminal = rows[-1]
    if _finite(terminal, "eval/multi_answer/eval_count", step=2) != 96:
        raise GateError("terminal evaluation did not cover all 96 frozen prompts")
    if _finite(terminal, "eval/multi_answer/response_tok_len", step=2) != 3:
        raise GateError("terminal canonical evaluation response length is not three")
    return (
        {
            "metrics_rows": len(rows),
            "oat_steps": steps,
            "optimizer_updates": 1,
            "terminal_global_step": global_steps[-1],
            **behavior_extrema,
        },
        metrics_path,
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


def _matches_config(raw: str, expected: Any) -> bool:
    if expected is None:
        return raw == "None"
    if isinstance(expected, bool):
        return raw == str(expected)
    if isinstance(expected, str):
        return raw.strip("'\"") == expected
    try:
        observed = float(raw)
    except ValueError:
        return False
    if isinstance(expected, float) and math.isinf(expected):
        return math.isinf(observed) and observed > 0
    return math.isclose(observed, float(expected), rel_tol=0, abs_tol=1e-12)


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
    stdout = ANSI_ESCAPE.sub("", stdout_path.read_text(encoding="utf-8", errors="replace"))
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
        "learner_side_sampling_completed": (
            r"canonical learner sampler finished data_len=16 seed=\d+ "
            r"normalization_error_max=.*fixed_shape=1"
        ),
        "post_learning": r"post-learning done step=1",
        "update_log": r"eval/log done step=1",
        "vllm_v0": r"\[slurm\] vllm_use_v1=0",
    }
    for label, pattern in required_patterns.items():
        if re.search(pattern, stdout) is None:
            raise GateError(f"runtime log lacks {label} evidence")
    if EXPECTED_RUNTIME_IDENTITY["tokenizer_revision"] not in stdout:
        raise GateError("runtime log does not show the frozen model/tokenizer revision")
    source_line = f"[slurm] oat_zero_source_root={source_root.resolve()}"
    if source_line not in stdout:
        raise GateError("runtime log source snapshot does not match the audited source")
    if f"[slurm] job_id={job_id}" not in stdout:
        raise GateError("runtime log does not belong to the audited Slurm allocation")
    save_line = f"[experiment] save_path={run_dir.resolve()}"
    if save_line not in stdout:
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
        "prompt_rows_rendered": 192,
        "dataset_map_cache": "disabled",
        "vllm_engine": "v0",
        "canonical_training_sampler": "learner_hf_fixed_shape",
        "zero_learning_rate": True,
    }


def parse_slurm_status(path: Path, *, expected_job_id: str | None) -> tuple[str, str]:
    if not path.is_file():
        raise GateError(f"Slurm status artifact is missing: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise GateError("Slurm status artifact is empty")
    if text.startswith("{"):
        payload = _strict_json(text, context=str(path))
        if not isinstance(payload, dict):
            raise GateError("Slurm status JSON is not an object")
        if expected_job_id and str(payload.get("job_id")) != expected_job_id:
            raise GateError("Slurm status JSON has the wrong job ID")
        return str(payload.get("state", "")), str(payload.get("exit_code", ""))
    records = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) >= 3:
            records.append((parts[0], parts[1], parts[2]))
        elif len(parts) == 2:
            records.append((expected_job_id or "", parts[0], parts[1]))
    if expected_job_id:
        records = [record for record in records if record[0] == expected_job_id]
    if len(records) != 1:
        raise GateError("Slurm status artifact does not identify one allocation record")
    _, state, exit_code = records[0]
    return state, exit_code


def query_slurm(job_id: str) -> tuple[str, str]:
    command = [
        "sacct",
        "-X",
        "-n",
        "-P",
        "-j",
        job_id,
        "-o",
        "JobIDRaw,State,ExitCode",
    ]
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise GateError(f"could not obtain terminal Slurm status for {job_id}: {error}") from error
    records = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) >= 3 and parts[0] == job_id:
            records.append((parts[1], parts[2]))
    if len(records) != 1:
        raise GateError(f"sacct did not return one allocation record for {job_id}")
    return records[0]


def _discover_job_id(repo_root: Path, *, stamp: str, run_dir: Path) -> str:
    manifest = repo_root / "var" / "artifacts" / f"{stamp}_comparative_jobs.tsv"
    if not manifest.is_file():
        raise GateError(f"E14 submission manifest is missing: {manifest}")
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
        raise GateError("submission manifest does not identify one E14 preflight job")
    if not run_dir.name.endswith(f"_{expected_run_stamp}"):
        raise GateError("run directory does not match the identity stamp/arm/seed")
    return str(matches[0]["job_id"])


def check_preflight(
    *,
    run_dir: Path,
    identity_path: Path,
    source_root: Path,
    stdout_path: Path,
    stderr_path: Path | None,
    slurm_state: str,
    slurm_exit_code: str,
    job_id: str,
    logical_repo_root: Path,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise GateError(f"run directory is missing: {run_dir}")
    identity, dataset, runtime = read_identity(identity_path)
    if identity["phase"] != "preflight":
        raise GateError(f"identity phase is {identity['phase']!r}, not 'preflight'")
    stamp = identity["stamp"]
    if identity_path.name != f"{stamp}_e14_identity.tsv":
        raise GateError("identity filename does not match its stamp")
    if not run_dir.name.endswith(f"_{stamp}_grpo_s9005"):
        raise GateError("run directory does not match the identity stamp/arm/seed")
    expected_scalar_identity = {
        "target_optimizer_updates": "1",
        "trajectory_query_budget": "1",
        "group_size": "16",
    }
    for key, expected in expected_scalar_identity.items():
        if identity[key] != expected:
            raise GateError(f"identity {key}={identity[key]!r}; expected {expected!r}")
    if re.fullmatch(r"[0-9a-f]{64}", identity["source_hash"]) is None:
        raise GateError("identity source_hash is not a lowercase SHA-256 digest")
    observed_source_hash = source_tree_hash(
        source_root, logical_repo_root=logical_repo_root
    )
    if identity["source_hash"] != observed_source_hash:
        raise GateError(
            "immutable source snapshot does not match the preregistered source hash"
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
    metrics_summary, metrics_path = inspect_metrics(run_dir)
    evidence_paths = {
        "identity": identity_path,
        "metrics": metrics_path,
        "source_snapshot_marker": source_root / "oat_drgrpo" / "__init__.py",
        "stdout": stdout_path,
    }
    if stderr_path is not None:
        evidence_paths["stderr"] = stderr_path
    evidence_hashes = {
        key: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
        for key, path in evidence_paths.items()
    }
    return {
        "approved": True,
        "gate": GATE_NAME,
        "protocol": "E14",
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
            "canonical_prompt_materialization": True,
            "canonical_action_runtime": True,
            "exactly_one_optimizer_update": True,
            "zero_learning": True,
            "behavior_policy_overlap": True,
            "learner_side_behavior_sampling": True,
            "fixed_shape_behavior_sampling": True,
            "forbidden_treatments_inactive": True,
        },
        "log_summary": log_summary,
        "metrics_summary": metrics_summary,
        "evidence": evidence_hashes,
    }


def write_approval(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def verify_approval_for_source(
    approval_path: Path,
    *,
    expected_source_hash: str,
    logical_repo_root: Path,
) -> dict[str, Any]:
    """Revalidate an approval and bind it to the prospective C0 source.

    This deliberately replays the preflight's semantic checks in addition to
    hashing its evidence.  The JSON is an auditable gate artifact, not a
    signature, so trusting its booleans alone would recreate the old manual
    bypass under a different name.
    """

    if re.fullmatch(r"[0-9a-f]{64}", expected_source_hash) is None:
        raise GateError("expected C0 source hash is not a lowercase SHA-256 digest")
    if not approval_path.is_file():
        raise GateError(f"E14 preflight approval is missing: {approval_path}")
    payload = _strict_json(
        approval_path.read_text(encoding="utf-8"), context=str(approval_path)
    )
    if not isinstance(payload, dict):
        raise GateError("E14 preflight approval is not a JSON object")
    if payload.get("approved") is not True:
        raise GateError("E14 preflight approval does not have approved=true")
    if payload.get("gate") != GATE_NAME:
        raise GateError("E14 preflight approval has the wrong gate identifier")
    if payload.get("protocol") != "E14":
        raise GateError("E14 preflight approval has the wrong protocol")
    slurm = payload.get("slurm")
    if not isinstance(slurm, dict) or slurm.get("state") != "COMPLETED" or slurm.get(
        "exit_code"
    ) != "0:0":
        raise GateError("E14 preflight approval lacks clean Slurm terminal success")
    job_id = str(payload.get("job_id", ""))
    if re.fullmatch(r"[0-9]+", job_id) is None:
        raise GateError("E14 preflight approval has an invalid Slurm job ID")

    approved_at = payload.get("approved_at_utc")
    try:
        approved_time = datetime.fromisoformat(str(approved_at))
    except ValueError as error:
        raise GateError("E14 preflight approval has an invalid UTC timestamp") from error
    if approved_time.tzinfo is None or approved_time.utcoffset() is None:
        raise GateError("E14 preflight approval timestamp is not timezone-aware")

    checks = payload.get("checks")
    if not isinstance(checks, dict):
        raise GateError("E14 preflight approval has no checks object")
    missing_checks = sorted(REQUIRED_APPROVAL_CHECKS - set(checks))
    failed_checks = sorted(
        key for key in REQUIRED_APPROVAL_CHECKS if checks.get(key) is not True
    )
    if missing_checks or failed_checks:
        raise GateError(
            "E14 preflight approval checks are incomplete; "
            f"missing={missing_checks} failed={failed_checks}"
        )

    identity_payload = payload.get("identity")
    if not isinstance(identity_payload, dict):
        raise GateError("E14 preflight approval has no identity object")
    approved_source_hash = identity_payload.get("source_hash")
    if approved_source_hash != expected_source_hash:
        raise GateError(
            "C0 Python source differs from the source that passed E14 preflight"
        )
    if identity_payload.get("dataset") != EXPECTED_DATASET_IDENTITY:
        raise GateError("approval embeds the wrong E14 dataset identity")
    if identity_payload.get("runtime") != EXPECTED_RUNTIME_IDENTITY:
        raise GateError("approval embeds the wrong E14 runtime identity")
    stamp = identity_payload.get("stamp")
    if not isinstance(stamp, str) or not stamp:
        raise GateError("approval has no preflight stamp")

    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        raise GateError("E14 preflight approval has no evidence object")
    required_evidence = {"identity", "metrics", "source_snapshot_marker", "stdout"}
    missing_evidence = sorted(required_evidence - set(evidence))
    if missing_evidence:
        raise GateError(f"E14 preflight approval lacks evidence {missing_evidence}")
    evidence_paths: dict[str, Path] = {}
    for label, record in evidence.items():
        if not isinstance(label, str) or not isinstance(record, dict):
            raise GateError("E14 preflight evidence records are malformed")
        if set(record) != {"path", "sha256"}:
            raise GateError(f"E14 preflight evidence {label!r} has malformed fields")
        raw_path = record.get("path")
        expected_digest = record.get("sha256")
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise GateError(f"E14 preflight evidence {label!r} path is not absolute")
        if not isinstance(expected_digest, str) or re.fullmatch(
            r"[0-9a-f]{64}", expected_digest
        ) is None:
            raise GateError(f"E14 preflight evidence {label!r} has an invalid hash")
        evidence_path = Path(raw_path)
        if not evidence_path.is_file():
            raise GateError(f"E14 preflight evidence {label!r} is missing")
        observed_digest = _sha256_file(evidence_path)
        if observed_digest != expected_digest:
            raise GateError(f"E14 preflight evidence {label!r} changed after approval")
        evidence_paths[label] = evidence_path

    identity, dataset, runtime = read_identity(evidence_paths["identity"])
    if identity["phase"] != "preflight" or identity["stamp"] != stamp:
        raise GateError("approval and frozen identity artifact disagree on provenance")
    if identity["source_hash"] != expected_source_hash:
        raise GateError("frozen identity source hash differs from prospective C0")
    if dataset != identity_payload["dataset"] or runtime != identity_payload["runtime"]:
        raise GateError("approval and frozen identity artifact disagree")

    source_marker = evidence_paths["source_snapshot_marker"].resolve()
    if source_marker.name != "__init__.py" or source_marker.parent.name != "oat_drgrpo":
        raise GateError("approval source snapshot marker has the wrong location")
    source_root = source_marker.parents[1]
    observed_snapshot_hash = source_tree_hash(
        source_root, logical_repo_root=logical_repo_root
    )
    if observed_snapshot_hash != expected_source_hash:
        raise GateError("approved preflight source snapshot changed after approval")

    run_dir_raw = payload.get("run_dir")
    if not isinstance(run_dir_raw, str) or not Path(run_dir_raw).is_absolute():
        raise GateError("approval run directory is not an absolute path")
    run_dir = Path(run_dir_raw)
    metrics_summary, observed_metrics_path = inspect_metrics(run_dir)
    if observed_metrics_path.resolve() != evidence_paths["metrics"].resolve():
        raise GateError("approval metrics evidence is not the audited run's stream")
    if metrics_summary != payload.get("metrics_summary"):
        raise GateError("approval metrics summary does not match its evidence")

    stderr_path = evidence_paths.get("stderr")
    log_summary = inspect_logs(
        evidence_paths["stdout"],
        stderr_path,
        run_dir=run_dir,
        source_root=source_root,
        job_id=job_id,
    )
    if log_summary != payload.get("log_summary"):
        raise GateError("approval log summary does not match its evidence")
    overlap_bounds = {
        "behavior_q_norm_error_max": (0.0, BEHAVIOR_Q_NORM_ERROR_MAX),
        "behavior_ratio_min": (BEHAVIOR_RATIO_MIN, BEHAVIOR_RATIO_MAX),
        "behavior_ratio_max": (BEHAVIOR_RATIO_MIN, BEHAVIOR_RATIO_MAX),
        "behavior_tv_max": (0.0, BEHAVIOR_TV_MAX),
        "behavior_kl_actor_learner_max": (0.0, BEHAVIOR_KL_MAX),
        "behavior_kl_learner_actor_max": (0.0, BEHAVIOR_KL_MAX),
        "behavior_sequence_ess_fraction_min": (
            BEHAVIOR_ESS_FRACTION_MIN,
            1.0 + 1e-6,
        ),
        "behavior_prefix_ess_fraction_min": (
            BEHAVIOR_ESS_FRACTION_MIN,
            1.0 + 1e-6,
        ),
    }
    for key, (lower, upper) in overlap_bounds.items():
        try:
            value = float(metrics_summary[key])
        except (KeyError, TypeError, ValueError) as error:
            raise GateError(f"approval lacks finite behavior overlap metric {key}") from error
        if not math.isfinite(value) or not lower <= value <= upper:
            raise GateError(
                f"approval has invalid behavior overlap metric {key}={value}"
            )
    if metrics_summary["behavior_ratio_min"] > metrics_summary["behavior_ratio_max"]:
        raise GateError("approval has an inverted behavior-policy ratio range")
    if metrics_summary.get("optimizer_updates") != 1:
        raise GateError("approval does not establish exactly one optimizer update")

    return {
        "approval": str(approval_path.resolve()),
        "approval_sha256": _sha256_file(approval_path),
        "job_id": job_id,
        "preflight_stamp": stamp,
        "source_hash": expected_source_hash,
        "evidence_files_verified": len(evidence_paths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Approve E14's zero-learning canonical runtime probe."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
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
        repo_root / "var" / "artifacts" / f"{stamp}_e14_preflight_approval.json"
    )
    # Never leave a stale positive artifact at the requested path after a
    # failed re-check of mutated or incomplete evidence.
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

    payload = check_preflight(
        run_dir=args.run_dir,
        identity_path=args.identity,
        source_root=source_root,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        slurm_state=state,
        slurm_exit_code=exit_code,
        job_id=job_id,
        logical_repo_root=repo_root,
    )
    write_approval(approval_out, payload)
    print(f"E14 zero-learning preflight approved: {approval_out.resolve()}")


if __name__ == "__main__":
    main()
