#!/usr/bin/env python3
"""Fail-closed live audit for the E60 Python causal pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
IDENTITY = ROOT / "var/artifacts/e60_bootstrap_local_python_pilot_identity.json"
OUT = ROOT / "var/artifacts/e60_python_pilot_audit_latest.json"
PROTOCOL = (
    ROOT
    / "paper/preregistration/e60_bootstrap_then_local_canonical_05b.md"
)
LAUNCHER = ROOT / "ops/exp_scaling/launch_e60_python_pilot.sh"
PREFIX = "pye60_bootstrap_local_05b_5ep_pilot"
ARM = "verified_first_bootstrap_local_canonical"
SEED = 9010
TERMINAL_STEP = 1920
BOOTSTRAP_STEPS = 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_tree(root: Path) -> str:
    lines = [
        f"{_sha256(path)}  ./{path.relative_to(root).as_posix()}\n"
        for path in sorted(
            (item for item in root.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(root).as_posix(),
        )
    ]
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        step = int(value.get("trainer/global_step", -1))
        previous = records.get(step)
        if previous is not None:
            conflicts = [
                key
                for key in set(previous) | set(value)
                if key.startswith(("train/", "actor/"))
                and previous.get(key) != value.get(key)
            ]
            if conflicts:
                raise ValueError(
                    f"{path}:{line_number} conflicts at step {step}"
                )
            if len(value) > len(previous):
                records[step] = value
        else:
            records[step] = value
    return [records[step] for step in sorted(records)]


def _evaluations(path: Path) -> list[dict[str, float | int]]:
    by_step: dict[int, list[dict[str, Any]]] = {}
    if not path.is_file():
        return []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        value = json.loads(raw)
        if isinstance(value, dict) and isinstance(value.get("metrics"), dict):
            by_step.setdefault(int(value["step"]), []).append(value["metrics"])
    result: list[dict[str, float | int]] = []
    for step in sorted(by_step):
        rows = by_step[step]
        if len(rows) != 4:
            continue
        result.append(
            {
                "step": step,
                "distinct8": sum(
                    float(row["distinct_correct_modes_at_k"])
                    for row in rows
                )
                / 4,
                "pass8": sum(float(row["any_correct_at_k"]) for row in rows)
                / 4,
                "mean8": sum(float(row["mean_at_k"]) for row in rows) / 4,
            }
        )
    return result


def audit(identity_path: Path) -> dict[str, Any]:
    violations: list[str] = []
    try:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if not isinstance(identity, dict):
            raise ValueError("identity is not an object")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {
            "schema": "e60_python_pilot_audit_v1",
            "status": "in_progress",
            "violations": [f"identity unavailable: {error}"],
        }

    manifest = ROOT / str(identity.get("manifest_relative_path", ""))
    expected = {
        "protocol_sha256": PROTOCOL,
        "launcher_sha256": LAUNCHER,
        "auditor_sha256": Path(__file__).resolve(),
        "manifest_sha256": manifest,
    }
    if identity.get("schema") != "e60_bootstrap_local_python_pilot_v1":
        violations.append("identity schema mismatch")
    for key, path in expected.items():
        if not path.is_file() or identity.get(key) != _sha256(path):
            violations.append(f"{key} binding mismatch")

    source_hash = str(identity.get("source_hash", ""))
    ops_hash = str(identity.get("execution_surface_hash", ""))
    source_root = (
        ROOT
        / f"var/artifacts/source_snapshots/e60_bootstrap_local_{source_hash}/src"
    )
    ops_root = (
        ROOT
        / f"var/artifacts/source_snapshots/e60_bootstrap_local_ops_{ops_hash}/ops"
    )
    try:
        if _hash_tree(source_root) != source_hash:
            violations.append("source snapshot binding mismatch")
        if _hash_tree(ops_root) != ops_hash:
            violations.append("execution snapshot binding mismatch")
    except OSError as error:
        violations.append(f"cannot inspect snapshots: {error}")

    job_id = int(identity.get("job_id", -1))
    try:
        with manifest.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        exact = [
            row
            for row in rows
            if row.get("arm") == ARM
            and row.get("seed") == str(SEED)
            and row.get("job_id") == str(job_id)
            and row.get("run_stamp") == f"{PREFIX}_{ARM}_s{SEED}"
        ]
        if len(exact) != 1:
            violations.append("manifest exact-job binding mismatch")
    except (OSError, csv.Error) as error:
        violations.append(f"cannot inspect manifest: {error}")

    run_candidates = sorted(
        (ROOT / "var/data").glob(
            f"*{PREFIX}_{ARM}_s{SEED}/debug_job{job_id}"
        )
    )
    run_root = run_candidates[0] if len(run_candidates) == 1 else None
    records: list[dict[str, Any]] = []
    if run_root is not None:
        metrics_path = run_root / "train_metrics.jsonl"
        try:
            records = _load_records(metrics_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            violations.append(f"cannot inspect metrics: {error}")
    elif run_candidates:
        violations.append("multiple exact-job run directories")

    latest_step = -1
    first_discovery_step: int | None = None
    first_local_step: int | None = None
    last_updates = 0
    previous_updates = 0
    local_records = 0
    global_records = 0
    negative_pressure_steps = 0
    positive_pressure_steps = 0
    for record in records:
        step = int(record.get("trainer/global_step", -1))
        latest_step = max(latest_step, step)
        for key, value in record.items():
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and not math.isfinite(float(value))
            ):
                violations.append(f"step {step}: nonfinite {key}")
            if key.startswith("train/maxent_"):
                violations.append(f"step {step}: direct MaxEnt telemetry")
        for key in (
            "train/canonical_replay_projection_active",
            "train/canonical_replay_mass_projection_active",
            "train/canonical_replay_alpha_projection_active",
            "train/semantic_shannon_success_conditioned_signed_open_set_projection_active",
            "train/canonical_replay_gold_support_feedback",
        ):
            value = record.get(key, 0)
            if _finite(value) and not math.isclose(float(value), 0.0):
                violations.append(f"step {step}: forbidden nonzero {key}")

        tracked = record.get("train/online_canonical_tracked_outcomes", 0)
        if _finite(tracked) and float(tracked) > 0 and first_discovery_step is None:
            first_discovery_step = step
        steps = record.get("train/canonical_replay_global_bootstrap_steps")
        updates = record.get("train/canonical_replay_global_bootstrap_updates")
        active = record.get("train/canonical_replay_global_bootstrap_active")
        local = record.get("train/canonical_replay_prompt_local_phase_active")
        used_global = record.get(
            "train/canonical_replay_schedule_used_global"
        )
        used_local = record.get(
            "train/canonical_replay_schedule_used_prompt_local"
        )
        if _finite(steps) and int(steps) != BOOTSTRAP_STEPS:
            violations.append(f"step {step}: bootstrap budget mismatch")
        if _finite(updates):
            update_count = int(updates)
            if not last_updates <= update_count <= BOOTSTRAP_STEPS:
                violations.append(f"step {step}: bootstrap progress invalid")
            advanced = update_count > previous_updates
            if _finite(used_global):
                if float(used_global) not in {0.0, 1.0}:
                    violations.append(
                        f"step {step}: invalid used-global flag"
                    )
                elif advanced and float(used_global) != 1.0:
                    violations.append(
                        f"step {step}: bootstrap advanced without global use"
                    )
                global_records += int(float(used_global) == 1.0)
            if _finite(used_local):
                if float(used_local) not in {0.0, 1.0}:
                    violations.append(
                        f"step {step}: invalid used-local flag"
                    )
                elif (
                    update_count < BOOTSTRAP_STEPS
                    and float(used_local) != 0.0
                ):
                    violations.append(
                        f"step {step}: local scheduler used before retirement"
                    )
                if float(used_local) == 1.0:
                    local_records += 1
                    if first_local_step is None:
                        first_local_step = step
            previous_updates = update_count
            last_updates = update_count
            expected_active = float(update_count < BOOTSTRAP_STEPS)
            if _finite(active) and float(active) != expected_active:
                violations.append(f"step {step}: global phase flag mismatch")
            expected_local = float(update_count == BOOTSTRAP_STEPS)
            if _finite(local) and float(local) != expected_local:
                violations.append(f"step {step}: local phase flag mismatch")

        negative = record.get(
            "train/semantic_shannon_success_conditioned_signed_"
            "effective_advantage_negative_fraction",
            0,
        )
        positive = record.get(
            "train/semantic_shannon_success_conditioned_signed_"
            "effective_advantage_positive_fraction",
            0,
        )
        negative_pressure_steps += int(_finite(negative) and float(negative) > 0)
        positive_pressure_steps += int(_finite(positive) and float(positive) > 0)

    evals = (
        _evaluations(run_root / "eval_mode_coverage_draws.jsonl")
        if run_root is not None
        else []
    )
    last_eight = evals[-8:]
    multiplicity_points = sum(
        float(row["distinct8"]) > float(row["pass8"]) + 1e-12
        for row in last_eight
    )
    final8 = None
    if len(last_eight) == 8:
        final8 = {
            "mean_distinct8": sum(float(row["distinct8"]) for row in last_eight)
            / 8,
            "mean_pass8": sum(float(row["pass8"]) for row in last_eight) / 8,
            "mean8": sum(float(row["mean8"]) for row in last_eight) / 8,
            "mean_excess_multiplicity": sum(
                float(row["distinct8"]) - float(row["pass8"])
                for row in last_eight
            )
            / 8,
            "positive_multiplicity_points": multiplicity_points,
        }

    terminal = latest_step >= TERMINAL_STEP
    if terminal:
        if last_updates != BOOTSTRAP_STEPS:
            violations.append("terminal run did not complete bootstrap")
        if local_records == 0:
            violations.append("terminal run never entered prompt-local phase")
        if final8 is None or multiplicity_points < 6:
            violations.append("terminal final-eight multiplicity gate failed")
        elif float(final8["mean_excess_multiplicity"]) <= 0:
            violations.append("terminal mean excess multiplicity is not positive")

    status = "fail" if violations else ("pass" if terminal else "in_progress")
    return {
        "schema": "e60_python_pilot_audit_v1",
        "status": status,
        "violations": violations,
        "identity_path": str(identity_path),
        "job_id": job_id,
        "run_root": str(run_root) if run_root is not None else None,
        "latest_step": latest_step,
        "terminal_step": TERMINAL_STEP,
        "training_passes": (
            latest_step / 384 if latest_step >= 0 else 0
        ),
        "first_discovery_step": first_discovery_step,
        "bootstrap": {
            "budget": BOOTSTRAP_STEPS,
            "updates": last_updates,
            "first_local_step": first_local_step,
            "local_records": local_records,
            "global_records": global_records,
        },
        "semantic_pressure": {
            "negative_steps": negative_pressure_steps,
            "positive_steps": positive_pressure_steps,
        },
        "evaluation_points": len(evals),
        "evaluations": evals,
        "final_eight": final8,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", type=Path, default=IDENTITY)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    payload = audit(args.identity)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out)
    print(
        f"[e60-python-audit] status={payload['status']} "
        f"step={payload['latest_step']} "
        f"violations={len(payload['violations'])} out={args.out}"
    )
    if payload["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
