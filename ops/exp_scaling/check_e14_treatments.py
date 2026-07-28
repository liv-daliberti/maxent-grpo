#!/usr/bin/env python3
"""Fail-closed runtime gate and frozen comparison for E14 M01/M05.

Run ``validate`` once per completed treatment after its exact step-128 audit,
then run ``compare`` on the two resulting artifacts.  Runtime/identity failures
raise and leave no stale result.  A scientifically non-viable but runtime-valid
arm is retained as a negative result, as required by the preregistration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:  # package import in tests
    from . import check_e14_c0 as c0
    from .check_e14_preflight import (
        ANSI_ESCAPE,
        CONFIG_FIELD,
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
        source_tree_hash,
    )
    from .verify_e14_c0_approval import verify_c0_approval_for_source
    from .e14_archival import (
        ArchiveReceiptError,
        replay_archive_authorization_if_needed,
    )
except ImportError:  # direct script execution
    import check_e14_c0 as c0
    from check_e14_preflight import (
        ANSI_ESCAPE,
        CONFIG_FIELD,
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
        source_tree_hash,
    )
    from verify_e14_c0_approval import verify_c0_approval_for_source
    from e14_archival import (
        ArchiveReceiptError,
        replay_archive_authorization_if_needed,
    )

# check_e14_c0 is also executable as a direct script and deliberately imports
# its preflight helpers through its script directory.  Use the exact exception
# class it raises so package-imported treatment tests and callers see one gate
# failure type rather than two module-qualified copies.
GateError = c0.GateError


GATE_NAME = "e14_canonical_fixed_treatment"
RESULT_SCHEMA = "e14_canonical_treatment_result_v1"
COMPARISON_SCHEMA = "e14_canonical_treatment_comparison_v1"
T_MAX = 192.0
REWARD_ESTIMATOR_SCALE = 15.0 / 16.0
ENTROPY_GAIN_MIN = math.log(1.25)
VALID_SUPPORT_GAIN_MIN = 1.25
P_VALID_RETENTION_MIN = 0.80
P_VALID_ABSOLUTE_MIN = 0.05
TIE_RELATIVE_WIDTH = 0.05
ARM_SPECS = {
    "M01": {"phase": "m01", "alpha": 0.01},
    "M05": {"phase": "m05", "alpha": 0.05},
}
BASE_IDENTITY_KEYS = {
    "phase",
    "stamp",
    "source_hash",
    "dataset_identity",
    "runtime_identity",
    "target_optimizer_updates",
    "trajectory_query_budget",
    "group_size",
}
TREATMENT_IDENTITY_KEYS = BASE_IDENTITY_KEYS | {
    "protocol_arm",
    "arm",
    "maxent_alpha",
    "c0_approval",
    "c0_approval_sha256",
}
MAXENT_BASE_KEYS = {
    "train/maxent_alpha_used",
    "train/maxent_sequence_entropy",
    "train/maxent_sequence_entropy_per_tmax",
    "train/maxent_entropy_surrogate",
    "train/maxent_sampled_prefix_entropy",
    "train/maxent_sampled_prefix_entropy_per_tmax",
    "train/maxent_prefix_ratio_mean",
    "train/maxent_prefix_ratio_max",
    "train/maxent_prefix_ratio_clipfrac",
    "train/maxent_entropy_loss",
    "train/maxent_reward_estimator_scale",
    "train/maxent_valid_row_fraction",
}
FORBIDDEN_TREATMENT_PREFIXES = (
    "train/maxent_control_",
    "train/maxent_dual_",
    "train/maxent_length_",
    "train/policy_entropy_",
    "train/seed_entropy_",
    "train/xdr_sac_",
    "train/xdr_tau_",
)


def _mean(values: Sequence[float], *, label: str) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise GateError(f"cannot compute {label} from empty or nonfinite values")
    return math.fsum(values) / len(values)


def _close(observed: float, expected: float, *, label: str) -> None:
    if not math.isfinite(observed) or not math.isclose(
        observed, expected, rel_tol=1e-5, abs_tol=1e-8
    ):
        raise GateError(f"{label}={observed!r}; expected {expected!r}")


def read_treatment_identity(
    path: Path, *, arm: str, c0_approval: Path
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    if arm not in ARM_SPECS:
        raise GateError(f"unknown E14 treatment arm {arm!r}")
    if not path.is_file():
        raise GateError(f"treatment identity is missing: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if not rows or rows[0] != ["key", "value"]:
        raise GateError("treatment identity must start with exact key/value header")
    identity: dict[str, str] = {}
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) != 2 or not row[0] or row[0] in identity:
            raise GateError(f"malformed or duplicate treatment identity row {line_number}")
        identity[row[0]] = row[1]
    if set(identity) != TREATMENT_IDENTITY_KEYS:
        raise GateError(
            "treatment identity fields drifted; "
            f"missing={sorted(TREATMENT_IDENTITY_KEYS - set(identity))} "
            f"extra={sorted(set(identity) - TREATMENT_IDENTITY_KEYS)}"
        )
    spec = ARM_SPECS[arm]
    expected = {
        "phase": spec["phase"],
        "protocol_arm": arm,
        "arm": "maxent",
        "target_optimizer_updates": str(c0.EXPECTED_UPDATES),
        "trajectory_query_budget": str(c0.EXPECTED_QUERY_BUDGET),
        "group_size": str(c0.EXPECTED_GROUP_SIZE),
    }
    for key, value in expected.items():
        if identity[key] != value:
            raise GateError(f"treatment identity {key}={identity[key]!r}; expected {value!r}")
    _close(float(identity["maxent_alpha"]), float(spec["alpha"]), label="identity alpha")
    if Path(identity["c0_approval"]).resolve() != c0_approval.resolve():
        raise GateError("treatment identity names a different C0 approval")
    if identity["c0_approval_sha256"] != _sha256_file(c0_approval):
        raise GateError("treatment identity C0 approval hash changed")
    dataset = _strict_json(identity["dataset_identity"], context="dataset_identity")
    runtime = _strict_json(identity["runtime_identity"], context="runtime_identity")
    if dataset != EXPECTED_DATASET_IDENTITY or runtime != EXPECTED_RUNTIME_IDENTITY:
        raise GateError("treatment frozen dataset/runtime identity drifted")
    if re.fullmatch(r"[0-9a-f]{64}", identity["source_hash"]) is None:
        raise GateError("treatment source hash is not a lowercase SHA-256")
    return identity, dataset, runtime


def _check_treatment_telemetry(row: dict[str, Any], *, step: int, alpha: float) -> dict[str, float]:
    for key in row:
        if key.startswith(FORBIDDEN_TREATMENT_PREFIXES):
            raise GateError(f"step {step} contains prohibited controller telemetry {key}")
        if key.startswith("train/maxent_"):
            base = key[:-4] if key.endswith(("_nan", "_inf")) else key
            if base not in MAXENT_BASE_KEYS:
                raise GateError(f"step {step} contains unexpected MaxEnt telemetry {key}")
    values = {key: _finite(row, key, step=step) for key in MAXENT_BASE_KEYS}
    for key, raw in row.items():
        if key.startswith("train/maxent_") and key.endswith(("_nan", "_inf")):
            if float(raw) != 0:
                raise GateError(f"step {step} reports numerical failure in {key}")
    _close(values["train/maxent_alpha_used"], alpha, label=f"step {step} alpha")
    raw_entropy = values["train/maxent_sequence_entropy"]
    surrogate = values["train/maxent_entropy_surrogate"]
    sampled = values["train/maxent_sampled_prefix_entropy"]
    if not 0 < raw_entropy <= c0.MAX_ACTION_ENTROPY + 1e-5:
        raise GateError(f"step {step} MaxEnt sequence entropy is outside (0, log(27)]")
    for label, value in (("entropy surrogate", surrogate), ("sampled-prefix entropy", sampled)):
        if not 0 < value <= c0.MAX_ACTION_ENTROPY + 1e-5:
            raise GateError(f"step {step} {label} is outside (0, log(27)]")
    _close(surrogate, raw_entropy, label=f"step {step} unclipped canonical surrogate")
    _close(
        sampled,
        raw_entropy,
        label=f"step {step} on-policy sampled-prefix entropy",
    )
    _close(
        values["train/maxent_sequence_entropy_per_tmax"],
        raw_entropy / T_MAX,
        label=f"step {step} sequence entropy telemetry units",
    )
    _close(
        values["train/maxent_sampled_prefix_entropy_per_tmax"],
        sampled / T_MAX,
        label=f"step {step} sampled entropy telemetry units",
    )
    # train/canonical_sampled_prefix_entropy_sum is the last backward
    # microbatch's behavior-prefix diagnostic, while train/maxent_* is the
    # mean over all four backward microbatches.  Both are independently
    # bounded above; equality between those differently reduced values is not
    # an invariant.
    prefix_mean = values["train/maxent_prefix_ratio_mean"]
    prefix_max = values["train/maxent_prefix_ratio_max"]
    if not 0 < prefix_mean <= prefix_max:
        raise GateError(f"step {step} has invalid exclusive-prefix ratio diagnostics")
    _close(values["train/maxent_prefix_ratio_clipfrac"], 0.0, label=f"step {step} unclipped prefix fraction")
    _close(
        values["train/maxent_reward_estimator_scale"],
        REWARD_ESTIMATOR_SCALE,
        label=f"step {step} reward-estimator scale",
    )
    _close(values["train/maxent_valid_row_fraction"], 1.0, label=f"step {step} valid-row fraction")
    expected_loss = -alpha * REWARD_ESTIMATOR_SCALE * surrogate / T_MAX
    _close(values["train/maxent_entropy_loss"], expected_loss, label=f"step {step} entropy loss")
    if values["train/maxent_entropy_loss"] >= 0:
        raise GateError(f"step {step} canonical entropy objective is not active")
    return {
        "entropy": raw_entropy,
        "sampled_entropy": sampled,
        "prefix_ratio_max": prefix_max,
        "entropy_loss": values["train/maxent_entropy_loss"],
    }


def _inspect_update(row: dict[str, Any], *, step: int, alpha: float) -> dict[str, float]:
    treatment = _check_treatment_telemetry(row, step=step, alpha=alpha)
    common = {key: value for key, value in row.items() if not key.startswith("train/maxent_")}
    diagnostics = c0._inspect_update_row(common, step=step)
    return {**diagnostics, **treatment}


def inspect_treatment_metrics(
    run_dir: Path,
    *,
    arm: str,
    alpha: float,
    archived_removed_steps: Sequence[str] | None = None,
    require_positive_final_reward: bool = True,
) -> tuple[dict[str, Any], Path, Path]:
    candidates = sorted(run_dir.glob("debug_*/train_metrics.jsonl"))
    if len(candidates) != 1:
        raise GateError(f"restart-invalid {arm} requires exactly one metrics stream; found {len(candidates)}")
    metrics_path = candidates[0]
    rows = _read_metrics(metrics_path)
    if len(rows) != c0.EXPECTED_UPDATES + 2:
        raise GateError(f"{arm} must contain initial, 128 updates, terminal alias; found {len(rows)}")
    steps = [_as_integral(row.get("trainer/step"), label="trainer/step") for row in rows]
    if steps != list(range(130)):
        raise GateError(f"{arm} trainer steps are not exact contiguous sequence 0--129")
    globals_ = [_as_integral(row.get("trainer/global_step"), label="trainer/global_step") for row in rows]
    if globals_ != [0, *range(1, 129), 128]:
        raise GateError(f"{arm} global steps do not end in the step-128 terminal alias")
    initial = rows[0]
    if any(key.startswith("train/maxent_") for key in initial):
        raise GateError(f"{arm} initial row unexpectedly contains treatment telemetry")
    for key, expected in (
        ("misc/prompt_dataset_len", 192),
        ("misc/query_step", 0),
        ("misc/prompt_consumed", 0),
        ("eval/multi_answer/eval_count", 96),
        ("eval/multi_answer/response_tok_len", 3),
    ):
        if _finite(initial, key, step=0) != expected:
            raise GateError(f"{arm} initial {key} differs from {expected}")
    expected_eval_steps = [0, 32, 64, 96, 128, 129]
    eval_steps = [step for step, row in zip(steps, rows, strict=True) if "eval/multi_answer/eval_count" in row]
    if eval_steps != expected_eval_steps:
        raise GateError(f"{arm} evaluation-bearing rows {eval_steps}; expected {expected_eval_steps}")
    for step in expected_eval_steps:
        if _finite(rows[step], "eval/multi_answer/eval_count", step=step) != 96:
            raise GateError(f"{arm} evaluation step {step} does not cover 96 prompts")
        if _finite(rows[step], "eval/multi_answer/response_tok_len", step=step) != 3:
            raise GateError(f"{arm} evaluation step {step} response length is not three")
    updates = rows[1:129]
    diagnostics = [_inspect_update(row, step=step, alpha=alpha) for step, row in enumerate(updates, 1)]
    policy_steps = [_finite(row, "trainer/policy_sgd_step", step=step) for step, row in enumerate(updates, 1)]
    if policy_steps != [float(step) for step in range(1, 129)]:
        raise GateError(f"{arm} policy-SGD steps are not contiguous 1--128")
    terminal = rows[-1]
    if _finite(terminal, "trainer/policy_sgd_step", step=129) != 128:
        raise GateError(f"{arm} terminal row is not an update-128 policy alias")
    for key in MAXENT_BASE_KEYS:
        _close(float(terminal[key]), float(updates[-1][key]), label=f"{arm} terminal alias {key}")
    for key in ("misc/query_step", "misc/prompt_consumed"):
        if _finite(terminal, key, step=129) != 2048:
            raise GateError(f"{arm} terminal {key} is not 2048")
    tail = diagnostics[-32:]
    final_reward = _mean([row["reward"] for row in tail], label=f"{arm} final-32 reward")
    if require_positive_final_reward and final_reward <= 0:
        raise GateError(f"{arm} final-32 mean rollout reward is not positive")
    saved_models = metrics_path.parent / "saved_models"
    checkpoint = saved_models / "step_00128"
    expected_tags = [f"step_{step:05d}" for step in (32, 64, 96, 128, 129)]
    observed_tags = sorted(path.name for path in saved_models.iterdir() if path.is_dir()) if saved_models.is_dir() else []
    archive_authorized = (
        archived_removed_steps is not None
        and list(archived_removed_steps)
        == ["step_00032", "step_00064", "step_00096"]
        and observed_tags == ["step_00128", "step_00129"]
    )
    if observed_tags != expected_tags and not archive_authorized:
        raise GateError(f"{arm} checkpoint schedule drifted: {observed_tags}")
    endpoint_hash, endpoint_files = c0._weight_identity(checkpoint)
    alias_hash, alias_files = c0._weight_identity(saved_models / "step_00129")
    if (endpoint_hash, endpoint_files) != (alias_hash, alias_files):
        raise GateError(f"{arm} step_00129 is not a byte-identical endpoint alias")
    eval_names = sorted(path.name for path in (metrics_path.parent / "eval_results").glob("*.json"))
    expected_eval_names = sorted(f"{step}_multi_answer.json" for step in expected_eval_steps)
    if eval_names != expected_eval_names:
        raise GateError(f"{arm} evaluation artifact schedule drifted: {eval_names}")
    return (
        {
            "metrics_rows": len(rows),
            "optimizer_updates": 128,
            "contiguous_finite_tail": [97, 128],
            "final_32_mean_rollout_reward": final_reward,
            "final_32_mean_sequence_entropy": _mean([row["entropy"] for row in tail], label="tail entropy"),
            "final_32_mean_entropy_loss": _mean([row["entropy_loss"] for row in tail], label="tail entropy loss"),
            "max_prefix_ratio": max(row["prefix_ratio_max"] for row in diagnostics),
            "behavior_ratio_min": min(row["ratio_min"] for row in diagnostics),
            "behavior_ratio_max": max(row["ratio_max"] for row in diagnostics),
            "step_128_weights_manifest_sha256": endpoint_hash,
            "step_129_byte_identical_alias": True,
        },
        metrics_path,
        checkpoint,
    )


def _parse_config(text: str, expected_config: dict[str, Any]) -> dict[str, str]:
    values: dict[str, str] = {}
    for match in CONFIG_FIELD.finditer(text):
        key, value = match.group("key"), match.group("value").strip()
        if key in expected_config and key in values and values[key] != value:
            raise GateError(f"runtime log reports conflicting {key} values")
        values[key] = value
    return values


def inspect_treatment_logs(
    stdout_path: Path,
    stderr_path: Path | None,
    *,
    run_dir: Path,
    source_root: Path,
    job_id: str,
    arm: str,
    alpha: float,
) -> dict[str, Any]:
    if not stdout_path.is_file():
        raise GateError(f"{arm} Slurm stdout is missing: {stdout_path}")
    stdout = ANSI_ESCAPE.sub("", stdout_path.read_text(encoding="utf-8", errors="replace"))
    stderr = ""
    if stderr_path is not None:
        if not stderr_path.is_file():
            raise GateError(f"{arm} Slurm stderr is missing: {stderr_path}")
        stderr = ANSI_ESCAPE.sub("", stderr_path.read_text(encoding="utf-8", errors="replace"))
    combined = stdout + "\n" + stderr
    for pattern in FATAL_LOG_PATTERNS:
        if pattern in combined:
            raise GateError(f"{arm} runtime log contains fatal signature {pattern!r}")
    required = {
        "canonical prompt": r"canonical prompt materialization verified:\s*rows=192\s+template=qwen_graph_digits\s+dataset_map_cache=disabled",
        "canonical actor": r"canonical graph actor configured:.*token_ids=\(16, 17, 18\).*action_count=3.*vllm_engine=v0",
        "canonical learner": r"canonical graph policy: action_token_ids=\(16, 17, 18\) horizon=3",
        "fixed sampler": r"\[train\] canonical_graph_actions=1 action_count=3 learner_sampling=1 fixed_shape_sampling=1(?:\s|$)",
        "restart disabled": r"\[watchdog\].*requeue=0 restart_count=0/\d+",
        "vLLM V0": r"\[slurm\] vllm_use_v1=0",
        "step-128 checkpoint": r"Checkpoint boundary at step 128: saving checkpoint before evaluation\.",
    }
    for label, pattern in required.items():
        if re.search(pattern, stdout) is None:
            raise GateError(f"{arm} runtime log lacks {label} evidence")
    samples = re.findall(r"canonical learner sampler finished data_len=16 seed=\d+ normalization_error_max=.*fixed_shape=1", stdout)
    if len(samples) != 128:
        raise GateError(f"{arm} runtime log has {len(samples)} canonical samples, expected 128")
    c0._exact_logged_steps(stdout, r"post-learning done step=(\d+)", label="post-learning")
    c0._exact_logged_steps(stdout, r"eval/log done step=(\d+)", label="eval/log")
    for evidence, label in (
        (EXPECTED_RUNTIME_IDENTITY["tokenizer_revision"], "tokenizer revision"),
        (f"[slurm] oat_zero_source_root={source_root.resolve()}", "source snapshot"),
        (f"[slurm] job_id={job_id}", "job ID"),
        (f"[experiment] save_path={run_dir.resolve()}", "run directory"),
    ):
        if evidence not in stdout:
            raise GateError(f"{arm} runtime log lacks matching {label}")
    expected_config = {**c0.EXPECTED_CONFIG, "maxent_alpha": alpha}
    config = _parse_config(stdout, expected_config)
    for key, expected in expected_config.items():
        if key not in config or not _matches_config(config[key], expected):
            raise GateError(f"{arm} runtime configuration {key}={config.get(key)!r}; expected {expected!r}")
    alpha_match = re.search(r"\[train\] maxent_alpha=([^\s]+) control_ratio=([^\s]+).*dual_ratio=([^\s]+)", stdout)
    if alpha_match is None:
        raise GateError(f"{arm} runtime log lacks fixed-MaxEnt launch evidence")
    _close(float(alpha_match.group(1)), alpha, label=f"{arm} logged alpha")
    _close(float(alpha_match.group(2)), 0.0, label=f"{arm} proportional controller ratio")
    _close(float(alpha_match.group(3)), 0.0, label=f"{arm} dual controller ratio")
    return {
        "stdout": str(stdout_path.resolve()),
        "stderr": str(stderr_path.resolve()) if stderr_path is not None else None,
        "canonical_training_samples": 128,
        "vllm_engine": "v0",
        "recovery": "disabled",
        "fixed_alpha": alpha,
    }


def _audit_semantic_fingerprint(path: Path) -> list[tuple[str, tuple[int, ...]]]:
    payload = _strict_json(path.read_text(encoding="utf-8"), context=str(path))
    prompts = payload.get("prompts") if isinstance(payload, dict) else None
    if not isinstance(prompts, list):
        raise GateError(f"endpoint audit {path} has no prompt records")
    result = []
    for prompt in prompts:
        leaves = prompt.get("leaves") if isinstance(prompt, dict) else None
        if not isinstance(leaves, list):
            raise GateError("endpoint audit prompt lacks leaves")
        result.append((str(prompt.get("problem_sha256")), tuple(int(leaf["grader_reward"]) for leaf in leaves)))
    return result


def classify_treatment(
    *, arm: str, c0_endpoint: dict[str, Any], endpoint: dict[str, Any], final_reward: float
) -> dict[str, Any]:
    p_valid = float(endpoint["p_valid_mean"])
    c0_p_valid = float(c0_endpoint["p_valid_mean"])
    entropy_delta = float(endpoint["exact_action_entropy_mean"]) - float(c0_endpoint["exact_action_entropy_mean"])
    support_ratio = float(endpoint["n_eff_valid_mean"]) / float(c0_endpoint["n_eff_valid_mean"])
    retention = p_valid / c0_p_valid
    # The thresholds are exact protocol quantities; the tiny subtraction
    # tolerance only absorbs binary64 roundoff in (treatment - C0), not a
    # scientific relaxation.
    threshold_tol = 1e-12
    safe = (
        p_valid > P_VALID_ABSOLUTE_MIN
        and retention >= P_VALID_RETENTION_MIN - threshold_tol
        and final_reward > 0
    )
    effective = (
        entropy_delta >= ENTROPY_GAIN_MIN - threshold_tol
        and support_ratio >= VALID_SUPPORT_GAIN_MIN - threshold_tol
    )
    failures: list[str] = []
    if not p_valid > P_VALID_ABSOLUTE_MIN:
        failures.append("endpoint mean P_valid <= 0.05")
    if retention < P_VALID_RETENTION_MIN - threshold_tol:
        failures.append("endpoint mean P_valid retained < 80% of C0")
    if not final_reward > 0:
        failures.append("final-32 mean rollout reward is not positive")
    if entropy_delta < ENTROPY_GAIN_MIN - threshold_tol:
        failures.append("exact action-entropy gain < log(1.25)")
    if support_ratio < VALID_SUPPORT_GAIN_MIN - threshold_tol:
        failures.append("exact valid-mode effective support gain < 25%")
    return {
        "arm": arm,
        "runtime_valid": True,
        "behaviorally_safe": safe,
        "diversity_effective": effective,
        "viable": safe and effective,
        "p_valid_mean": p_valid,
        "p_valid_retention_vs_c0": retention,
        "exact_action_entropy_mean": float(endpoint["exact_action_entropy_mean"]),
        "exact_action_entropy_gain_vs_c0": entropy_delta,
        "n_eff_valid_mean": float(endpoint["n_eff_valid_mean"]),
        "n_eff_valid_ratio_vs_c0": support_ratio,
        "final_32_mean_rollout_reward": final_reward,
        "failures": failures,
    }


def check_treatment(
    *,
    arm: str,
    run_dir: Path,
    identity_path: Path,
    c0_approval_path: Path,
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
    if arm not in ARM_SPECS:
        raise GateError(f"unknown E14 arm {arm}")
    if re.fullmatch(r"[0-9]+", str(job_id)) is None:
        raise GateError(f"{arm} has invalid Slurm job ID {job_id!r}")
    alpha = float(ARM_SPECS[arm]["alpha"])
    identity, dataset, runtime = read_treatment_identity(identity_path, arm=arm, c0_approval=c0_approval_path)
    stamp = identity["stamp"]
    run_dir = run_dir.resolve()
    if not run_dir.is_dir() or not run_dir.name.endswith(f"_{stamp}_maxent_s9005"):
        raise GateError(f"{arm} run directory does not match identity stamp/arm/seed")
    observed_source_hash = source_tree_hash(source_root, logical_repo_root=logical_repo_root)
    if identity["source_hash"] != observed_source_hash:
        raise GateError(f"{arm} immutable source snapshot hash drifted")
    c0_summary = verify_c0_approval_for_source(
        c0_approval_path,
        expected_source_hash=observed_source_hash,
        logical_repo_root=logical_repo_root,
    )
    if slurm_state.strip() != "COMPLETED" or slurm_exit_code.strip() != "0:0":
        raise GateError(f"{arm} job {job_id} is not clean terminal success: {slurm_state}/{slurm_exit_code}")
    log_summary = inspect_treatment_logs(
        stdout_path, stderr_path, run_dir=run_dir, source_root=source_root,
        job_id=job_id, arm=arm, alpha=alpha,
    )
    metrics_summary, metrics_path, checkpoint = inspect_treatment_metrics(
        run_dir,
        arm=arm,
        alpha=alpha,
        archived_removed_steps=archived_removed_steps,
    )
    endpoint_summary = c0.inspect_endpoint_audit(
        endpoint_audit_path,
        checkpoint=checkpoint,
        expected_source_root=source_root,
        expected_source_hash=observed_source_hash,
    )
    c0_payload = _strict_json(c0_approval_path.read_text(encoding="utf-8"), context=str(c0_approval_path))
    c0_audit_path = Path(c0_payload["endpoint_summary"]["audit"])
    if _audit_semantic_fingerprint(endpoint_audit_path) != _audit_semantic_fingerprint(c0_audit_path):
        raise GateError(f"{arm} endpoint audit does not use C0's exact prompt/reward alignment")
    classification = classify_treatment(
        arm=arm,
        c0_endpoint=c0_payload["endpoint_summary"],
        endpoint=endpoint_summary,
        final_reward=float(metrics_summary["final_32_mean_rollout_reward"]),
    )
    evidence_paths = {
        "identity": identity_path,
        "c0_approval": c0_approval_path,
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
            raise GateError(f"{arm} evidence {label!r} is missing: {path}")
    evidence = {key: {"path": str(path.resolve()), "sha256": _sha256_file(path)} for key, path in evidence_paths.items()}
    return {
        "schema": RESULT_SCHEMA,
        "gate": GATE_NAME,
        "protocol": "E14",
        "arm": arm,
        "alpha": alpha,
        "runtime_valid": True,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": job_id,
        "run_dir": str(run_dir),
        "slurm": {"state": "COMPLETED", "exit_code": "0:0"},
        "identity": {"stamp": stamp, "source_hash": observed_source_hash, "dataset": dataset, "runtime": runtime},
        "c0_summary": c0_summary,
        "checks": {
            "c0_approval_fully_replayed": True,
            "immutable_source_and_identity": True,
            "terminal_success": True,
            "updates_97_128_contiguous_finite": True,
            "canonical_rollouts_and_behavior_overlap": True,
            "fixed_alpha_on_every_update": True,
            "single_shared_outer_normalization": True,
            "adaptive_and_length_controllers_absent": True,
            "exact_step_00128_audit": True,
            "same_prompt_reward_alignment_as_c0": True,
        },
        "log_summary": log_summary,
        "metrics_summary": metrics_summary,
        "endpoint_summary": endpoint_summary,
        "classification": classification,
        "evidence": evidence,
    }


def choose_arm(classifications: Sequence[dict[str, Any]]) -> tuple[str | None, str]:
    viable = [row for row in classifications if row.get("viable") is True]
    if not viable:
        return None, "no runtime-valid arm passed both safety and diversity gates"
    if len(viable) == 1:
        return str(viable[0]["arm"]), "only viable fixed coefficient"
    by_arm = {str(row["arm"]): row for row in viable}
    m01, m05 = by_arm["M01"], by_arm["M05"]
    low = min(float(m01["n_eff_valid_mean"]), float(m05["n_eff_valid_mean"]))
    high = max(float(m01["n_eff_valid_mean"]), float(m05["n_eff_valid_mean"]))
    if low / high >= 1.0 - TIE_RELATIVE_WIDTH:
        return "M01", "valid-mode supports are within 5%; preregistered smaller coefficient wins"
    selected = max(viable, key=lambda row: float(row["n_eff_valid_mean"]))
    return str(selected["arm"]), "larger exact mean valid-mode effective support"


def compare_result_payloads(
    *, c0_approval_path: Path, m01_path: Path, m05_path: Path
) -> dict[str, Any]:
    results = []
    c0_hash = _sha256_file(c0_approval_path)
    for expected_arm, path in (("M01", m01_path), ("M05", m05_path)):
        payload = _strict_json(path.read_text(encoding="utf-8"), context=str(path))
        if not isinstance(payload, dict) or payload.get("schema") != RESULT_SCHEMA:
            raise GateError(f"{expected_arm} treatment result has wrong schema")
        if payload.get("arm") != expected_arm or payload.get("runtime_valid") is not True:
            raise GateError(f"{expected_arm} result has wrong arm or is not runtime-valid")
        evidence = payload.get("evidence")
        if not isinstance(evidence, dict) or "c0_approval" not in evidence:
            raise GateError(f"{expected_arm} result lacks C0 evidence")
        if evidence["c0_approval"].get("sha256") != c0_hash:
            raise GateError(f"{expected_arm} result is bound to a different C0 approval")
        for label, record in evidence.items():
            if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
                raise GateError(f"{expected_arm} evidence {label} is malformed")
            evidence_path = Path(str(record["path"]))
            if not evidence_path.is_file() or _sha256_file(evidence_path) != record["sha256"]:
                raise GateError(f"{expected_arm} evidence {label} is missing or changed")
        source_marker = Path(evidence["source_snapshot_marker"]["path"]).resolve()
        if source_marker.name != "__init__.py" or source_marker.parent.name != "oat_drgrpo":
            raise GateError(f"{expected_arm} source marker has the wrong location")
        # Approval JSON is an audit record rather than a signature.  Replay
        # the complete treatment gate so mutually consistent edits to a
        # result's summaries/classification cannot affect the comparison.
        try:
            archive_authorization = replay_archive_authorization_if_needed(path)
        except ArchiveReceiptError as error:
            raise GateError(
                f"{expected_arm} archival replay rejected: {error}"
            ) from error
        replayed = check_treatment(
            arm=expected_arm,
            run_dir=Path(payload["run_dir"]),
            identity_path=Path(evidence["identity"]["path"]),
            c0_approval_path=c0_approval_path,
            endpoint_audit_path=Path(evidence["endpoint_audit"]["path"]),
            source_root=source_marker.parents[1],
            stdout_path=Path(evidence["stdout"]["path"]),
            stderr_path=(
                Path(evidence["stderr"]["path"])
                if "stderr" in evidence
                else None
            ),
            slurm_state="COMPLETED",
            slurm_exit_code="0:0",
            job_id=str(payload["job_id"]),
            logical_repo_root=Path(__file__).resolve().parents[2],
            archived_removed_steps=(
                archive_authorization["removed_steps"]
                if archive_authorization is not None
                else None
            ),
        )
        comparable_payload = dict(payload)
        comparable_replayed = dict(replayed)
        comparable_payload.pop("validated_at_utc", None)
        comparable_replayed.pop("validated_at_utc", None)
        if comparable_payload != comparable_replayed:
            raise GateError(f"{expected_arm} result does not match replayed treatment evidence")
        classification = replayed["classification"]
        results.append(classification)
    selected, rationale = choose_arm(results)
    return {
        "schema": COMPARISON_SCHEMA,
        "protocol": "E14",
        "status": "selected" if selected is not None else "no_viable_dose",
        "selected_arm": selected,
        "selection_rationale": rationale,
        "single_seed_engineering_calibration": True,
        "does_not_authorize_scale_or_domain_expansion": True,
        "thresholds": {
            "p_valid_absolute_strict_min": P_VALID_ABSOLUTE_MIN,
            "p_valid_retention_min": P_VALID_RETENTION_MIN,
            "exact_entropy_gain_min_nats": ENTROPY_GAIN_MIN,
            "n_eff_valid_ratio_min": VALID_SUPPORT_GAIN_MIN,
            "tie_relative_width": TIE_RELATIVE_WIDTH,
        },
        "arms": results,
        "evidence": {
            "c0_approval": {"path": str(c0_approval_path.resolve()), "sha256": c0_hash},
            "m01_result": {"path": str(m01_path.resolve()), "sha256": _sha256_file(m01_path)},
            "m05_result": {"path": str(m05_path.resolve()), "sha256": _sha256_file(m05_path)},
        },
        "compared_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _job_status(args: argparse.Namespace) -> tuple[str, str]:
    if args.slurm_status is not None:
        return parse_slurm_status(args.slurm_status, expected_job_id=args.job_id)
    if args.slurm_state is not None or args.slurm_exit_code is not None:
        if args.slurm_state is None or args.slurm_exit_code is None:
            raise GateError("both --slurm-state and --slurm-exit-code are required")
        return args.slurm_state, args.slurm_exit_code
    return query_slurm(args.job_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and compare E14 fixed-alpha canonical MaxEnt treatments.")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--arm", choices=tuple(ARM_SPECS), required=True)
    validate.add_argument("--run-dir", type=Path, required=True)
    validate.add_argument("--identity", type=Path, required=True)
    validate.add_argument("--c0-approval", type=Path, required=True)
    validate.add_argument("--endpoint-audit", type=Path, required=True)
    validate.add_argument("--source-root", type=Path, required=True)
    validate.add_argument("--stdout-log", type=Path, required=True)
    validate.add_argument("--stderr-log", type=Path)
    validate.add_argument("--job-id", required=True)
    validate.add_argument("--slurm-status", type=Path)
    validate.add_argument("--slurm-state")
    validate.add_argument("--slurm-exit-code")
    validate.add_argument("--result-out", type=Path, required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--c0-approval", type=Path, required=True)
    compare.add_argument("--m01-result", type=Path, required=True)
    compare.add_argument("--m05-result", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output = args.result_out if args.command == "validate" else args.out
    output.unlink(missing_ok=True)
    if args.command == "validate":
        state, exit_code = _job_status(args)
        payload = check_treatment(
            arm=args.arm,
            run_dir=args.run_dir,
            identity_path=args.identity,
            c0_approval_path=args.c0_approval,
            endpoint_audit_path=args.endpoint_audit,
            source_root=args.source_root,
            stdout_path=args.stdout_log,
            stderr_path=args.stderr_log,
            slurm_state=state,
            slurm_exit_code=exit_code,
            job_id=args.job_id,
            logical_repo_root=repo_root,
        )
    else:
        payload = compare_result_payloads(
            c0_approval_path=args.c0_approval,
            m01_path=args.m01_result,
            m05_path=args.m05_result,
        )
    _write(output, payload)
    print(json.dumps({"output": str(output.resolve()), "status": payload.get("status", "runtime_valid"), "selected_arm": payload.get("selected_arm")}, sort_keys=True))


if __name__ == "__main__":
    main()
