#!/usr/bin/env python3
"""Fail-closed live audit for E64's six matched MATH realism runs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
IDENTITY = ROOT / "var/artifacts/e64_math500_realism_matched_identity.json"
PROTOCOL = ROOT / "paper/preregistration/e64_math500_realism_transfer_05b.md"
LAUNCHER = ROOT / "ops/exp_scaling/launch_e64_math500_realism_matched.sh"
MANIFEST = (
    ROOT / "var/artifacts/mte64_math500_realism_05b_12ep_comparative_jobs.tsv"
)
DATA_IDENTITY = (
    ROOT / "var/data/math12k_384_math500/MATERIALIZATION_MANIFEST.json"
)
SMOKE_IDENTITY = ROOT / "var/artifacts/e64_math500_realism_smoke_identity.json"
SMOKE_AUDIT = (
    ROOT / "var/artifacts/e64_math500_realism_smoke_audit_latest.json"
)
SMOKE_CHECKPOINT_AUDIT = (
    ROOT
    / "var/artifacts/e64_math500_realism_smoke_checkpoint_audit_latest.json"
)
OUT = ROOT / "var/artifacts/e64_math500_realism_matched_audit_latest.json"
CONTROL = "grpo"
TREATMENT = "verified_first_global_replay_canonical"
EXPECTED_STEP = 384 * 12
CRASH = re.compile(
    r"Traceback \(most recent call last\)|CUDA out of memory|"
    r"torch\.OutOfMemoryError|ChildFailedError|RayActorError|"
    r"worker unexpectedly died|RuntimeError:[^\n]*non-finite|"
    r"segmentation fault",
    re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _zero(value: Any, *, tolerance: float = 1e-9) -> bool:
    return _finite(value) and math.isclose(
        float(value),
        0.0,
        rel_tol=0.0,
        abs_tol=tolerance,
    )


def _run_dir(run_stamp: str, job_id: int) -> Path | None:
    candidates = sorted(
        (ROOT / "var/data").glob(f"*_{run_stamp}/debug_job{job_id}")
    )
    return candidates[0] if len(candidates) == 1 else None


def _scan_metrics(
    path: Path,
    *,
    arm: str,
    label: str,
) -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    latest_step = -1
    train_records = 0
    eval_steps: set[int] = set()
    first_discovery_step: int | None = None
    maximum_support = 0.0
    maximum_mass_observations = 0
    maximum_balance_observations = 0
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            violations.append(f"{label}: invalid JSON at metrics line {line_number}")
            continue
        step_value = row.get("trainer/global_step", -1)
        step = int(step_value) if _finite(step_value) else -1
        latest_step = max(latest_step, step)
        if any(key.startswith("eval/") for key in row):
            count = row.get("eval/math/eval_count")
            if not _finite(count) or int(count) != 500:
                violations.append(
                    f"{label}: step {step} MATH-500 eval count is not 500"
                )
            eval_steps.add(step)
        if not any(key.startswith("train/") for key in row):
            continue
        train_records += 1
        for key, value in row.items():
            if key.endswith(("_nan", "_inf")) and _finite(value) and value > 0:
                violations.append(
                    f"{label}: step {step} nonfinite diagnostic {key}={value}"
                )
            if (
                ("next_alpha" in key or "next_coefficient" in key)
                and value is not None
                and not _finite(value)
            ):
                violations.append(
                    f"{label}: step {step} nonfinite coefficient {key}"
                )

        support = row.get("train/verified_discovery_mean_support_per_prompt")
        if _finite(support):
            maximum_support = max(maximum_support, float(support))
            if float(support) > 1.0 + 1e-12:
                violations.append(
                    f"{label}: step {step} verified support exceeds one"
                )
        support_two = row.get(
            "train/online_canonical_support_at_least_two_prompt_fraction"
        )
        if support_two is not None and not _zero(support_two):
            violations.append(
                f"{label}: step {step} singleton support contract failed"
            )
        tracked = row.get("train/verified_discovery_tracked_prompts")
        if (
            first_discovery_step is None
            and _finite(tracked)
            and float(tracked) > 0
        ):
            first_discovery_step = step

        if arm == CONTROL:
            scheduler = row.get(
                "train/canonical_replay_global_scheduler_active"
            )
            if scheduler is not None and not _zero(scheduler):
                violations.append(
                    f"{label}: step {step} baseline replay is active"
                )
            continue

        modes = row.get("train/canonical_replay_available_modes")
        if _finite(modes) and float(modes) > 1.0 + 1e-12:
            violations.append(
                f"{label}: step {step} scheduled replay has multiple modes"
            )
        mass_observations = row.get(
            "train/canonical_replay_mass_observations"
        )
        if _finite(mass_observations):
            maximum_mass_observations = max(
                maximum_mass_observations,
                int(mass_observations),
            )
        balance_observations = row.get("train/canonical_replay_observations")
        if _finite(balance_observations):
            maximum_balance_observations = max(
                maximum_balance_observations,
                int(balance_observations),
            )
        for key in (
            "train/canonical_replay_observations",
            "train/canonical_replay_eligible_groups",
            "train/canonical_replay_retained_modes",
            "train/canonical_replay_balance_loss",
            "train/canonical_replay_balance_score_gradient_sum",
            "train/canonical_replay_balance_score_gradient_l2",
        ):
            value = row.get(key)
            if value is not None and not _zero(value):
                violations.append(
                    f"{label}: step {step} singleton balance metric {key} "
                    "is not zero"
                )
        for key in (
            "train/canonical_replay_alpha_projection_active",
            "train/canonical_replay_projection_active",
            "train/canonical_replay_mass_projection_active",
            "train/canonical_replay_gold_support_feedback",
            "train/"
            "semantic_shannon_success_conditioned_signed_"
            "open_set_projection_active",
        ):
            value = row.get(key)
            if value is not None and not _zero(value):
                violations.append(
                    f"{label}: step {step} forbidden projection/feedback {key}"
                )

    terminal = latest_step >= EXPECTED_STEP
    if terminal:
        if len(eval_steps) < 7:
            violations.append(
                f"{label}: terminal run has only {len(eval_steps)} "
                "MATH-500 boundaries"
            )
        if first_discovery_step is None:
            violations.append(f"{label}: no verified discovery")
        if arm == TREATMENT:
            if maximum_mass_observations <= 0:
                violations.append(f"{label}: verified-mass replay never activated")
            if maximum_balance_observations != 0:
                violations.append(f"{label}: balance controller was not idle")
    return (
        {
            "latest_step": latest_step,
            "expected_step": EXPECTED_STEP,
            "training_passes": (
                latest_step / 384 if latest_step >= 0 else None
            ),
            "terminal": terminal,
            "train_records": train_records,
            "evaluation_boundaries": len(eval_steps),
            "first_discovery_step": first_discovery_step,
            "maximum_verified_support": maximum_support,
            "controller_observations": {
                "mass": maximum_mass_observations,
                "balance": maximum_balance_observations,
            },
        },
        violations,
    )


def main() -> None:
    violations: list[str] = []
    try:
        identity = _load(IDENTITY)
    except Exception as error:
        identity = {}
        violations.append(f"cannot load identity: {error}")

    if identity.get("schema") != "e64_math500_realism_matched_v1":
        violations.append("identity schema mismatch")
    expected_hashes = {
        "protocol_sha256": PROTOCOL,
        "launcher_sha256": LAUNCHER,
        "auditor_sha256": Path(__file__).resolve(),
        "manifest_sha256": MANIFEST,
        "data_manifest_sha256": DATA_IDENTITY,
        "smoke_identity_sha256": SMOKE_IDENTITY,
        "smoke_audit_sha256": SMOKE_AUDIT,
        "smoke_checkpoint_audit_sha256": SMOKE_CHECKPOINT_AUDIT,
    }
    for field, path in expected_hashes.items():
        try:
            if identity.get(field) != _sha256(path):
                violations.append(f"{field} mismatch")
        except OSError as error:
            violations.append(f"cannot hash {path}: {error}")
    if (
        identity.get("arms") != [CONTROL, TREATMENT]
        or identity.get("seeds") != [43, 44, 45]
        or identity.get("prompt_epochs") != 12
        or identity.get("optimizer_updates_per_run") != EXPECTED_STEP
        or identity.get("canonical_key_mode") != "math_verified_answer"
        or identity.get("track")
        != "external_validity_generalization_not_modebench_mode_coverage"
    ):
        violations.append("identity matched-design contract mismatch")

    rows: list[dict[str, Any]] = []
    try:
        with MANIFEST.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    except OSError as error:
        violations.append(f"cannot read manifest: {error}")
    expected_pairs = {
        (arm, seed)
        for arm in (CONTROL, TREATMENT)
        for seed in (43, 44, 45)
    }
    actual_pairs = {
        (row.get("arm"), int(row.get("seed", -1))) for row in rows
    }
    actual_job_ids = [int(row.get("job_id", -1)) for row in rows]
    if (
        len(rows) != 6
        or actual_pairs != expected_pairs
        or actual_job_ids != identity.get("job_ids")
    ):
        violations.append("manifest/identity six-run pairing mismatch")

    runs: list[dict[str, Any]] = []
    terminal_runs = 0
    materialized_runs = 0
    metric_runs = 0
    for row in rows:
        arm = str(row["arm"])
        seed = int(row["seed"])
        job_id = int(row["job_id"])
        run_stamp = str(row["run_stamp"])
        label = f"{arm}/s{seed}/j{job_id}"
        run_dir = _run_dir(run_stamp, job_id)
        metrics_summary = {
            "latest_step": -1,
            "expected_step": EXPECTED_STEP,
            "training_passes": None,
            "terminal": False,
        }
        if run_dir is not None:
            materialized_runs += 1
            metrics_path = run_dir / "train_metrics.jsonl"
            if metrics_path.is_file():
                metric_runs += 1
                summary, metric_violations = _scan_metrics(
                    metrics_path,
                    arm=arm,
                    label=label,
                )
                metrics_summary.update(summary)
                violations.extend(metric_violations)
        if metrics_summary["terminal"]:
            terminal_runs += 1

        crash_signature = None
        for suffix in ("out", "err"):
            log = ROOT / f"var/artifacts/logs/xdr_train-{job_id}.{suffix}"
            if log.is_file():
                match = CRASH.search(
                    log.read_text(encoding="utf-8", errors="replace")
                )
                if match:
                    crash_signature = match.group(0)
                    violations.append(
                        f"{label}: crash signature {crash_signature!r}"
                    )
        runs.append(
            {
                "arm": arm,
                "seed": seed,
                "job_id": job_id,
                "run_stamp": run_stamp,
                "run_dir": (
                    str(run_dir.relative_to(ROOT))
                    if run_dir is not None
                    else None
                ),
                "crash_signature": crash_signature,
                **metrics_summary,
            }
        )

    all_terminal = len(runs) == 6 and terminal_runs == 6
    status = "fail" if violations else "pass" if all_terminal else "in_progress"
    payload = {
        "schema": "e64_math500_realism_matched_audit_v1",
        "status": status,
        "summary": {
            "expected_runs": 6,
            "identity_runs": len(rows),
            "materialized_runs": materialized_runs,
            "metric_runs": metric_runs,
            "terminal_runs": terminal_runs,
        },
        "runs": runs,
        "violations": sorted(set(violations)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{OUT.name}.", dir=OUT.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, OUT)
    print(
        f"[e64-matched-audit] status={status} "
        f"materialized={materialized_runs}/6 metrics={metric_runs}/6 "
        f"terminal={terminal_runs}/6 violations={len(payload['violations'])} "
        f"out={OUT}"
    )


if __name__ == "__main__":
    main()
