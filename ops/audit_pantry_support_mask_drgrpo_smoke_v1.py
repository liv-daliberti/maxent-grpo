#!/usr/bin/env python3
"""Fail-closed audit for PantryPlan's six-bit Dr.GRPO plumbing smoke."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

UPDATES = 32
SEED = 76201
PREFIX = "ppsmoke_support_mask_drgrpo_v1"
SUBMISSION_SCHEMAS = {"pantry-support-mask-drgrpo-smoke-submission-v1"}
FATAL = ("Traceback (most recent call last):", "OUT_OF_MEMORY", "DUE TO TIME LIMIT", "[watchdog] fatal:")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_tree(root: Path) -> str:
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        lines.append(f"{_sha(path)}  ./{path.relative_to(root).as_posix()}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _finite(row: dict[str, Any], key: str, step: int) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"step {step} lacks finite {key}") from error
    if not math.isfinite(value):
        raise ValueError(f"step {step} has nonfinite {key}")
    return value


def _close(value: float, expected: float, label: str, tolerance: float = 2e-6) -> None:
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{label}: expected {expected}, observed {value}")


def _atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("fresh Pantry smoke audit output is required")
    root = args.repo_root.resolve()
    identity = _load(args.identity)
    submission = _load(args.submission)
    errors: list[str] = []
    checks: dict[str, bool] = {}

    schedule = subprocess.run(
        ["sacct", "-j", str(args.job_id), "--format=State,ExitCode", "-n", "-X", "-P"],
        check=True, text=True, capture_output=True,
    ).stdout.strip().splitlines()
    terminal = schedule[0].split("|") if schedule else []
    checks["slurm_completed"] = terminal[:2] == ["COMPLETED", "0:0"]
    checks["identity_contract"] = (
        identity.get("schema") == "pantry-support-mask-drgrpo-smoke-identity-v1"
        and identity.get("arm") == "grpo"
        and identity.get("seed") == SEED
        and identity.get("optimizer_updates") == UPDATES
        and identity.get("rollouts_per_prompt") == 16
        and identity.get("canonical_action_task") == "pantry_support_mask"
        and identity.get("horizon") == 6
        and identity.get("sequence_count") == 64
        and identity.get("maxent_actuators_enabled") is False
        and identity.get("recovery_enabled") is False
    )
    checks["submission_contract"] = (
        submission.get("schema") in SUBMISSION_SCHEMAS
        and submission.get("identity_sha256") == _sha(args.identity)
        and submission.get("manifest_sha256") == _sha(args.manifest)
        and submission.get("job_id") == args.job_id
        and submission.get("held_job_audit") == "pass"
        and submission.get("released") is True
    )
    source_root = Path(str(identity.get("source_root", "")))
    ops_root = Path(str(identity.get("ops_root", "")))
    checks["source_snapshot_matches"] = source_root.is_dir() and _hash_tree(source_root) == identity.get("source_hash")
    checks["execution_snapshot_matches"] = ops_root.is_dir() and _hash_tree(ops_root) == identity.get("execution_hash")
    checks["data_tree_matches"] = _hash_tree(root / "var/data/pantry_plan_modebench_v2") == identity.get("data_tree_sha256")
    checks["antecedents_match"] = all((root / path).is_file() and _sha(root / path) == identity.get(field) for field, path in {
        "viability_receipt_sha256": "var/artifacts/pantry_support_mask_dev_v1.json",
        "viability_identity_sha256": "var/artifacts/pantry_support_mask_dev_v1_identity.json",
        "viability_audit_sha256": "var/artifacts/pantry_support_mask_dev_v1_audit.json",
    }.items())

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle, delimiter="\t"))
    checks["manifest_has_exact_cell"] = manifest == [{
        "arm": "grpo", "seed": str(SEED), "job_id": str(args.job_id),
        "run_stamp": f"{PREFIX}_grpo_s{SEED}",
    }]
    run_candidates = list(root.glob(f"var/data/xdr_*_grpo_{PREFIX}_grpo_s{SEED}"))
    checks["unique_run_directory"] = len(run_candidates) == 1
    debug_dirs: list[Path] = []
    if len(run_candidates) == 1:
        debug_dirs = [p.parent for p in run_candidates[0].glob("debug_*/train_metrics.jsonl")]
    checks["unique_metrics_stream"] = len(debug_dirs) == 1

    positive_reward_updates = 0
    sampled_multimode_prompts = 0
    update_rows: list[dict[str, Any]] = []
    metrics_path = None
    draws_path = None
    if len(debug_dirs) == 1:
        debug = debug_dirs[0]
        metrics_path = debug / "train_metrics.jsonl"
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        checks["metrics_have_exact_order"] = (
            len(rows) == UPDATES + 2
            and [int(row["trainer/step"]) for row in rows] == list(range(UPDATES + 2))
            and [int(row["trainer/global_step"]) for row in rows] == [0, *range(1, UPDATES + 1), UPDATES]
            and [int(row["trainer/policy_sgd_step"]) for row in rows] == [0, *range(1, UPDATES + 1), UPDATES]
        )
        update_rows = rows[1 : UPDATES + 1]
        checks["learning_rounds_exact"] = [int(row["train/learning_round"]) for row in update_rows] == list(range(1, UPDATES + 1))
        try:
            for step, row in enumerate(update_rows, 1):
                reward = _finite(row, "actor/rewards", step)
                if not 0 <= reward <= 1 or not math.isclose(reward * 16, round(reward * 16), abs_tol=1e-7):
                    raise ValueError(f"step {step} reward is not a 16-rollout mean")
                positive_reward_updates += reward > 0
                for key, expected in {
                    "actor/canonical_pantry_support_mask_actions": 1,
                    "actor/canonical_action_count": 6,
                    "actor/canonical_action_support_size": 2,
                    "actor/canonical_sequence_support_size": 64,
                    "actor/canonical_invalid_count": 0,
                    "actor/canonical_finish_length_count": 16,
                    "actor/canonical_finish_unexpected_count": 0,
                    "actor/canonical_behavior_q_row_count": 96,
                    "actor/canonical_behavior_q_support_min": 2,
                    "actor/canonical_behavior_q_support_max": 2,
                    "actor/canonical_sampler_learner": 1,
                    "actor/canonical_sampler_fixed_shape": 1,
                    "train/canonical_action_count": 6,
                    "train/canonical_behavior_q_row_count": 96,
                    "train/canonical_behavior_q_support_min": 2,
                    "train/canonical_behavior_q_support_max": 2,
                    "train/canonical_exact_leaf_count": 64,
                    "train/canonical_exact_prefix_row_count": 63,
                    "train/canonical_exact_post_update": 1,
                }.items():
                    _close(_finite(row, key, step), expected, f"step {step} {key}")
                for key in (
                    "actor/canonical_behavior_q_norm_error_max",
                    "train/canonical_behavior_q_norm_error_max",
                    "train/canonical_actor_logp_diff_max",
                    "train/canonical_behavior_kl_actor_learner_max",
                    "train/canonical_behavior_kl_learner_actor_max",
                    "train/canonical_behavior_tv_max",
                    "train/canonical_behavior_selected_echo_diff_max",
                ):
                    if abs(_finite(row, key, step)) > 1e-5:
                        raise ValueError(f"step {step} parity tolerance exceeded for {key}")
                for key in (
                    "train/canonical_behavior_ratio_min",
                    "train/canonical_behavior_sequence_ess_fraction",
                    "train/canonical_behavior_prefix_ess_fraction_min",
                ):
                    if _finite(row, key, step) <= 0:
                        raise ValueError(f"step {step} nonpositive {key}")
                _close(_finite(row, "train/canonical_exact_leaf_mass", step), 1, f"step {step} leaf mass", 1e-9)
                entropy = _finite(row, "train/canonical_exact_sequence_entropy", step)
                if not 0 <= entropy <= math.log(64) + 1e-8:
                    raise ValueError(f"step {step} entropy outside log(64)")
                for key in ("train/pg_loss", "train/policy_grad_norm", "train/entropy"):
                    _finite(row, key, step)
                for key in (
                    "train/zero_pg_loss_count_inf",
                    "train/zero_pg_loss_count_nan",
                    "train/policy_grad_norm_inf",
                    "train/policy_grad_norm_nan",
                ):
                    _close(_finite(row, key, step), 0, f"step {step} {key}")
            checks["all_update_telemetry_passes"] = True
        except (KeyError, TypeError, ValueError) as error:
            checks["all_update_telemetry_passes"] = False
            errors.append(str(error))
        checks["positive_verified_reward"] = positive_reward_updates > 0

        draws_path = debug / "eval_mode_coverage_draws.jsonl"
        draw_rows = [json.loads(line) for line in draws_path.read_text().splitlines() if line.strip()] if draws_path.is_file() else []
        sampled = [row for row in draw_rows if row.get("evaluation_kind") == "fixed_seed_sampled_k_neutral" and row.get("seed") == 76299 and row.get("sample_count") == 8]
        for row in sampled:
            for prompt in row.get("prompts", []):
                keys = {key for key, reward in zip(prompt.get("answer_keys", []), prompt.get("rewards", [])) if key is not None and float(reward) > 0}
                sampled_multimode_prompts += len(keys) >= 2
        checks["sampled_eval_multimode"] = bool(sampled) and sampled_multimode_prompts > 0

    stdout = root / f"var/artifacts/logs/xdr_train-{args.job_id}.out"
    text = stdout.read_text(errors="replace") if stdout.is_file() else ""
    checks["stdout_identity"] = (
        f"[train] canonical_action_task=pantry_support_mask action_count=6" in text
        and "[train] target_optimizer_updates=32" in text
        and f"[slurm] job_id={args.job_id}" in text
        and not any(marker in text for marker in FATAL)
    )
    for name, passed in checks.items():
        if not passed:
            errors.append(f"failed check: {name}")
    payload = {
        "schema": "pantry-support-mask-drgrpo-smoke-audit-v1",
        "status": "pass" if not errors else "fail",
        "job_id": args.job_id,
        "checks": checks,
        "positive_reward_updates": positive_reward_updates,
        "sampled_multimode_prompts": sampled_multimode_prompts,
        "evidence": {
            "identity_sha256": _sha(args.identity),
            "submission_sha256": _sha(args.submission),
            "manifest_sha256": _sha(args.manifest),
            "metrics_sha256": _sha(metrics_path) if metrics_path else None,
            "draws_sha256": _sha(draws_path) if draws_path and draws_path.is_file() else None,
            "stdout_sha256": _sha(stdout) if stdout.is_file() else None,
        },
        "errors": errors,
        "decision": "eligible_for_paired_mechanism_smoke" if not errors else "pantry_training_plumbing_ineligible",
    }
    _atomic(args.output, payload)
    print(json.dumps({"status": payload["status"], "positive_reward_updates": positive_reward_updates, "sampled_multimode_prompts": sampled_multimode_prompts}))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
