#!/usr/bin/env python3
"""Supplemental model-tensor finiteness gate for E64's frozen smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
IDENTITY = ROOT / "var/artifacts/e64_math500_realism_smoke_identity.json"
OUT = (
    ROOT
    / "var/artifacts/e64_math500_realism_smoke_checkpoint_audit_latest.json"
)
PREFIX = "mte64_math500_realism_smoke_e58_s43"
ARM = "verified_first_global_replay_canonical"
SINGLE_KEY = "math_verified_answer:correct"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint(job_id: int) -> Path | None:
    candidates = sorted(
        (ROOT / "var/data").glob(
            f"*{PREFIX}_{ARM}_s43/debug_job{job_id}/checkpoints/"
            "step_00096/mp_rank_00_model_states.pt"
        )
    )
    return candidates[0] if len(candidates) == 1 else None


def _finite_numbers(
    value: Any,
    *,
    prefix: str,
) -> list[str]:
    violations: list[str] = []
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return violations
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            violations.append(f"nonfinite checkpoint scalar {prefix}")
        return violations
    if isinstance(value, dict):
        for key, item in value.items():
            violations.extend(
                _finite_numbers(item, prefix=f"{prefix}.{key}")
            )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            violations.extend(
                _finite_numbers(item, prefix=f"{prefix}[{index}]")
            )
    return violations


def audit(identity_path: Path) -> dict[str, Any]:
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    violations: list[str] = []
    if identity.get("schema") != "e64_math500_realism_smoke_v1":
        violations.append("identity schema mismatch")
    if (
        identity.get("arm") != ARM
        or identity.get("seed") != 43
        or identity.get("max_updates") != 96
    ):
        violations.append("identity job contract mismatch")
    try:
        job_id = int(identity["job_id"])
    except (KeyError, TypeError, ValueError):
        job_id = -1
        violations.append("identity job id is invalid")

    checkpoint = _checkpoint(job_id) if job_id >= 0 else None
    if checkpoint is None:
        return {
            "schema": "e64_math500_realism_smoke_checkpoint_audit_v1",
            "status": "fail" if violations else "in_progress",
            "violations": violations,
            "identity": str(identity_path),
            "identity_sha256": _sha256(identity_path),
            "checkpoint": None,
            "job_id": job_id,
        }

    try:
        import torch

        state = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except Exception as error:  # pragma: no cover - backend diagnostic
        violations.append(f"cannot inspect checkpoint: {error}")
        state = {}

    if state.get("global_steps") != 96 and state.get("global_step") != 96:
        violations.append("checkpoint global step is not 96")
    module = state.get("module")
    tensor_count = 0
    parameter_count = 0
    nonfinite_tensors: list[str] = []
    if not isinstance(module, dict) or not module:
        violations.append("checkpoint has no model tensor map")
    else:
        for name, tensor in module.items():
            if not hasattr(tensor, "numel"):
                continue
            tensor_count += 1
            parameter_count += int(tensor.numel())
            if (
                getattr(tensor, "is_floating_point", lambda: False)()
                and not bool(torch.isfinite(tensor).all())
            ):
                nonfinite_tensors.append(str(name))
        if tensor_count == 0 or parameter_count == 0:
            violations.append("checkpoint model tensor map is empty")
        if nonfinite_tensors:
            violations.append(
                "checkpoint contains nonfinite model tensors: "
                + ", ".join(nonfinite_tensors[:8])
            )

    for field in (
        "canonical_replay_controller_state",
        "canonical_replay_mass_controller_state",
        "semantic_shannon_tracker_state",
    ):
        violations.extend(
            _finite_numbers(state.get(field), prefix=field)
        )
    bank = state.get("online_canonical_bank_state")
    counts = bank.get("counts") if isinstance(bank, dict) else None
    if not isinstance(counts, dict) or not counts:
        violations.append("checkpoint has no verified-answer bank")
    elif any(
        not isinstance(outcomes, dict)
        or set(outcomes) != {SINGLE_KEY}
        for outcomes in counts.values()
    ):
        violations.append("checkpoint bank violates singleton correct-key support")

    return {
        "schema": "e64_math500_realism_smoke_checkpoint_audit_v1",
        "status": "pass" if not violations else "fail",
        "violations": sorted(set(violations)),
        "identity": str(identity_path),
        "identity_sha256": _sha256(identity_path),
        "checkpoint": str(checkpoint),
        "job_id": job_id,
        "model_tensor_count": tensor_count,
        "model_parameter_count": parameter_count,
        "nonfinite_model_tensors": nonfinite_tensors,
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", type=Path, default=IDENTITY)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    try:
        payload = audit(args.identity)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        payload = {
            "schema": "e64_math500_realism_smoke_checkpoint_audit_v1",
            "status": "fail",
            "violations": [str(error)],
        }
    _write(args.out, payload)
    print(
        f"[e64-checkpoint-audit] status={payload['status']} "
        f"violations={len(payload.get('violations', []))} out={args.out}"
    )


if __name__ == "__main__":
    main()
