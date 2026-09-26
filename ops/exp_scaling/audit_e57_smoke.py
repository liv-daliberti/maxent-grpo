#!/usr/bin/env python3
"""Fail-closed audit for E57's exact-seed Python cold-start smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IDENTITY = (
    ROOT
    / "var"
    / "artifacts"
    / "e57_verified_first_split_python_smoke_identity.json"
)
DEFAULT_OUT = ROOT / "var" / "artifacts" / "e57_python_smoke_audit_latest.json"
PREFIX = "e57_verified_first_split_smoke_python_seed9010"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an object")
    return payload


def _find_metrics(job_id: int) -> Path:
    candidates = sorted(
        (ROOT / "var" / "data").glob(
            f"*{PREFIX}_verified_first_split_canonical_s9010/"
            f"debug_job{job_id}/train_metrics.jsonl"
        )
    )
    if len(candidates) != 1:
        raise ValueError(
            f"expected one exact-job E57 metrics file, found {len(candidates)}"
        )
    return candidates[0]


def _read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        record = json.loads(raw)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        records.append(record)
    records.sort(key=lambda row: int(row.get("trainer/global_step", -1)))
    return records


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def audit(identity_path: Path) -> dict[str, Any]:
    violations: list[str] = []
    identity = _load_json(identity_path)
    expected_files = {
        "protocol_sha256": (
            ROOT
            / "paper"
            / "preregistration"
            / "e57_verified_first_split_canonical_05b.md"
        ),
        "launcher_sha256": (
            ROOT / "ops" / "exp_scaling" / "launch_e57_verified_first_smoke.sh"
        ),
        "auditor_sha256": Path(__file__).resolve(),
        "manifest_sha256": (
            ROOT
            / "var"
            / "artifacts"
            / f"{PREFIX}_comparative_jobs.tsv"
        ),
    }
    if identity.get("schema") != "e57_verified_first_split_python_smoke_v1":
        violations.append("identity schema mismatch")
    for field, path in expected_files.items():
        if not path.is_file() or identity.get(field) != _sha256(path):
            violations.append(f"{field} binding mismatch")
    if identity.get("seed") != 9010 or identity.get("max_updates") != 128:
        violations.append("identity smoke budget mismatch")
    direct = identity.get("direct_token_entropy")
    if direct != {"coefficient": 0.0, "controller": None}:
        violations.append("direct token entropy is not disabled")
    firewall = identity.get("information_firewall")
    if firewall != {
        "desired_entropy": None,
        "desired_mode_count": None,
        "evaluation_feedback": False,
        "gold_support_feedback": False,
    }:
        violations.append("information firewall mismatch")

    try:
        job_id = int(identity["job_id"])
        metrics_path = _find_metrics(job_id)
        records = _read_records(metrics_path)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        violations.append(str(exc))
        records = []
        metrics_path = None

    first_discovery_step: int | None = None
    replay_activations = 0
    semantic_observations = 0
    mass_observations = 0
    balance_observations = 0
    latest_step = -1
    for index, record in enumerate(records):
        step = int(record.get("trainer/global_step", -1))
        latest_step = max(latest_step, step)
        for key, value in record.items():
            if key.startswith("train/maxent_"):
                violations.append(f"step {step}: direct MaxEnt telemetry present: {key}")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(float(value)):
                    violations.append(f"step {step}: nonfinite {key}")

        groups = record.get("train/canonical_replay_available_groups", 0)
        groups = int(groups) if _finite(groups) else 0
        if groups > 0:
            replay_activations += 1
            if first_discovery_step is None:
                first_discovery_step = step
        if first_discovery_step is None:
            cold_checks = {
                "task_reward": record.get(
                    "train/online_canonical_task_reward_mean", 0
                ),
                "semantic_reward": record.get(
                    "train/semantic_shannon_augmented_reward_mean", 0
                ),
                "replay_groups": groups,
                "policy_grad_norm": record.get("train/policy_grad_norm", 0),
            }
            for name, value in cold_checks.items():
                if not _finite(value) or not math.isclose(
                    float(value), 0.0, rel_tol=0.0, abs_tol=1e-12
                ):
                    violations.append(
                        f"step {step}: pre-discovery {name} is not zero"
                    )

        semantic_observations = max(
            semantic_observations,
            int(
                record.get(
                    "train/"
                    "semantic_shannon_success_conditioned_signed_open_set_observations",
                    0,
                )
            ),
        )
        mass_observations = max(
            mass_observations,
            int(record.get("train/canonical_replay_mass_observations", 0)),
        )
        balance_observations = max(
            balance_observations,
            int(record.get("train/canonical_replay_observations", 0)),
        )
        for key in (
            "train/canonical_replay_alpha_projection_active",
            "train/canonical_replay_mass_projection_active",
            "train/canonical_replay_gold_support_feedback",
            "train/"
            "semantic_shannon_success_conditioned_signed_open_set_projection_active",
        ):
            value = record.get(key)
            if value is not None and (
                not _finite(value)
                or not math.isclose(float(value), 0.0, abs_tol=1e-12)
            ):
                violations.append(f"step {step}: forbidden projection/feedback {key}")

        if index > 0 and step < int(records[index - 1]["trainer/global_step"]):
            violations.append("metric steps are not monotonic")

    terminal = latest_step >= 128
    if terminal:
        if first_discovery_step is None:
            violations.append("no verifier-positive model discovery by step 128")
        if replay_activations <= 0 or mass_observations <= 0:
            violations.append("verified discovery did not activate mass replay")

    status = "fail" if violations else "pass" if terminal else "in_progress"
    return {
        "schema": "e57_verified_first_split_python_smoke_audit_v1",
        "status": status,
        "violations": sorted(set(violations)),
        "identity": str(identity_path),
        "metrics": None if metrics_path is None else str(metrics_path),
        "latest_step": latest_step,
        "terminal_step": 128,
        "first_discovery_step": first_discovery_step,
        "replay_activations": replay_activations,
        "controller_observations": {
            "semantic": semantic_observations,
            "mass": mass_observations,
            "balance": balance_observations,
        },
        "objective_isolation": {
            "direct_token_entropy": "absent",
            "zero_gradient_cold_start": not any(
                "pre-discovery" in violation for violation in violations
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    try:
        payload = audit(args.identity)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        payload = {
            "schema": "e57_verified_first_split_python_smoke_audit_v1",
            "status": "fail",
            "violations": [str(exc)],
        }
    _write(args.out, payload)
    print(
        f"[e57-smoke-audit] status={payload['status']} "
        f"violations={len(payload.get('violations', []))} out={args.out}"
    )


if __name__ == "__main__":
    main()
