#!/usr/bin/env python3
"""Fail-closed audit for E57's Graph multi-mode mechanism smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PREFIX = "e57_verified_first_split_smoke_graph_seed9057"
DEFAULT_IDENTITY = (
    ROOT
    / "var"
    / "artifacts"
    / "e57_verified_first_split_graph_smoke_identity.json"
)
DEFAULT_OUT = ROOT / "var" / "artifacts" / "e57_graph_smoke_audit_latest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def audit(identity_path: Path) -> dict[str, Any]:
    identity = _load(identity_path)
    violations: list[str] = []
    bindings = {
        "protocol_sha256": (
            ROOT
            / "paper"
            / "preregistration"
            / "e57_verified_first_split_canonical_05b.md"
        ),
        "launcher_sha256": (
            ROOT / "ops" / "exp_scaling" / "launch_e57_graph_smoke.sh"
        ),
        "auditor_sha256": Path(__file__).resolve(),
        "manifest_sha256": (
            ROOT / "var" / "artifacts" / f"{PREFIX}_comparative_jobs.tsv"
        ),
        "python_smoke_audit_sha256": (
            ROOT / "var" / "artifacts" / "e57_python_smoke_audit_latest.json"
        ),
    }
    if identity.get("schema") != "e57_verified_first_split_graph_smoke_v1":
        violations.append("identity schema mismatch")
    for field, path in bindings.items():
        if not path.is_file() or identity.get(field) != _sha256(path):
            violations.append(f"{field} binding mismatch")

    job_id = int(identity.get("job_id", -1))
    matches = sorted(
        (ROOT / "var" / "data").glob(
            f"*{PREFIX}_verified_first_split_canonical_s9057/"
            f"debug_job{job_id}/train_metrics.jsonl"
        )
    )
    records: list[dict[str, Any]] = []
    metrics_path: Path | None = None
    if len(matches) != 1:
        violations.append(
            f"expected one exact-job E57 graph metrics file, found {len(matches)}"
        )
    else:
        metrics_path = matches[0]
        for line_number, raw in enumerate(
            metrics_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw.strip():
                continue
            record = json.loads(raw)
            if not isinstance(record, dict):
                violations.append(f"metrics line {line_number} is not an object")
                continue
            records.append(record)
        records.sort(key=lambda row: int(row.get("trainer/global_step", -1)))

    latest_step = max(
        (int(row.get("trainer/global_step", -1)) for row in records),
        default=-1,
    )
    replay_activations = 0
    max_modes = 0
    semantic_observations = 0
    mass_observations = 0
    balance_observations = 0
    positive_pressure_steps = 0
    negative_pressure_steps = 0
    for record in records:
        step = int(record.get("trainer/global_step", -1))
        for key, value in record.items():
            if key.startswith("train/maxent_"):
                violations.append(f"step {step}: direct MaxEnt telemetry present: {key}")
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and not math.isfinite(float(value))
            ):
                violations.append(f"step {step}: nonfinite {key}")
        groups = float(record.get("train/canonical_replay_available_groups", 0))
        modes = float(record.get("train/canonical_replay_available_modes", 0))
        replay_activations += int(groups > 0)
        max_modes = max(max_modes, int(modes))
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
        positive_pressure_steps += int(
            float(
                record.get(
                    "train/"
                    "semantic_shannon_success_conditioned_signed_"
                    "effective_advantage_positive_fraction",
                    0,
                )
            )
            > 0
        )
        negative_pressure_steps += int(
            float(
                record.get(
                    "train/"
                    "semantic_shannon_success_conditioned_signed_"
                    "effective_advantage_negative_fraction",
                    0,
                )
            )
            > 0
        )
        for key in (
            "train/canonical_replay_alpha_projection_active",
            "train/canonical_replay_mass_projection_active",
            "train/canonical_replay_gold_support_feedback",
            "train/"
            "semantic_shannon_success_conditioned_signed_open_set_projection_active",
        ):
            if key in record and not math.isclose(
                float(record[key]), 0.0, rel_tol=0.0, abs_tol=1e-12
            ):
                violations.append(f"step {step}: forbidden projection/feedback {key}")

    terminal = latest_step >= 32
    if terminal:
        requirements = {
            "no replay activation": replay_activations > 0,
            "no multi-mode bank": max_modes >= 2,
            "semantic controller idle": semantic_observations > 0,
            "mass controller idle": mass_observations > 0,
            "balance controller idle": balance_observations > 0,
            "no positive semantic pressure": positive_pressure_steps > 0,
            "no negative semantic pressure": negative_pressure_steps > 0,
        }
        violations.extend(name for name, passed in requirements.items() if not passed)

    status = "fail" if violations else "pass" if terminal else "in_progress"
    return {
        "schema": "e57_verified_first_split_graph_smoke_audit_v1",
        "status": status,
        "violations": sorted(set(violations)),
        "identity": str(identity_path),
        "metrics": None if metrics_path is None else str(metrics_path),
        "latest_step": latest_step,
        "terminal_step": 32,
        "replay_activations": replay_activations,
        "max_available_modes": max_modes,
        "controller_observations": {
            "semantic": semantic_observations,
            "mass": mass_observations,
            "balance": balance_observations,
        },
        "semantic_pressure_steps": {
            "positive": positive_pressure_steps,
            "negative": negative_pressure_steps,
        },
        "objective_isolation": {"direct_token_entropy": "absent"},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    try:
        payload = audit(args.identity)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        payload = {
            "schema": "e57_verified_first_split_graph_smoke_audit_v1",
            "status": "fail",
            "violations": [str(exc)],
        }
    _write(args.out, payload)
    print(
        f"[e57-graph-smoke-audit] status={payload['status']} "
        f"violations={len(payload.get('violations', []))} out={args.out}"
    )


if __name__ == "__main__":
    main()
