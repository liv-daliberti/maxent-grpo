#!/usr/bin/env python3
"""Audit the six frozen E69 MATH checkpoints before MATH-500 unsealing."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from ops.route_successor.audit_e69_gate2_screen import _run_dir


ROOT = Path(__file__).resolve().parents[2]
IDENTITY = ROOT / "var/artifacts/e69_gate3_confirmatory_identity.json"
GATE3_AUDIT = ROOT / "var/artifacts/e69_gate3_confirmatory_audit_latest.json"
OUT = ROOT / "var/artifacts/e69_gate4_checkpoint_audit.json"
ARMS = ("grpo", "verified_first_global_replay_canonical")
SEEDS = (43, 44, 45)
CHECKPOINT_NAME = "step_02305"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def audit_checkpoints(
    identity_path: Path = IDENTITY,
    gate3_audit_path: Path = GATE3_AUDIT,
) -> dict[str, Any]:
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    gate3 = json.loads(gate3_audit_path.read_text(encoding="utf-8"))
    violations: list[str] = []
    pending: list[str] = []
    if identity.get("schema") != "e69_gate3_confirmatory_v1":
        violations.append("Gate 3 identity schema mismatch")
    if (
        gate3.get("schema") != "e69_gate3_confirmatory_audit_v1"
        or gate3.get("status") != "complete"
        or gate3.get("summary", {}).get("terminal_physical_runs") != 30
        or gate3.get("summary", {}).get("integrity_violations") != 0
    ):
        violations.append("Gate 3 is not complete with clean integrity")
    if gate3.get("math500_sealed") is not True:
        violations.append("Gate 3 did not preserve the MATH-500 seal")

    jobs = [
        row
        for row in identity.get("jobs", {}).get("math_dev", [])
        if row.get("arm") in ARMS and int(row.get("seed", -1)) in SEEDS
    ]
    observed = {(str(row["arm"]), int(row["seed"])) for row in jobs}
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if observed != expected or len(jobs) != 6:
        violations.append("Gate 3 MATH checkpoint grid is not exactly 2 x 3")

    checkpoints: list[dict[str, Any]] = []
    for job in sorted(
        jobs,
        key=lambda row: (str(row["arm"]), int(row["seed"])),
    ):
        arm = str(job["arm"])
        seed = int(job["seed"])
        job_id = int(job["job_id"])
        alias = f"{arm}_s{seed}"
        run_dir = _run_dir(str(job["run_stamp"]), job_id)
        checkpoint = (
            run_dir / "saved_models" / CHECKPOINT_NAME
            if run_dir is not None
            else None
        )
        if checkpoint is None or not checkpoint.is_dir():
            pending.append(f"{alias}: terminal exported checkpoint absent")
            continue
        model = checkpoint / "model.safetensors"
        config = checkpoint / "config.json"
        tokenizer = checkpoint / "tokenizer.json"
        for required in (model, config, tokenizer):
            if not required.is_file():
                violations.append(f"{alias}: missing {required.name}")
        tensor_count = 0
        parameter_count = 0
        nonfinite: list[str] = []
        if model.is_file():
            try:
                import torch
                from safetensors import safe_open

                with safe_open(model, framework="pt", device="cpu") as handle:
                    for name in handle.keys():
                        tensor = handle.get_tensor(name)
                        tensor_count += 1
                        parameter_count += int(tensor.numel())
                        if tensor.is_floating_point() and not bool(
                            torch.isfinite(tensor).all()
                        ):
                            nonfinite.append(name)
            except Exception as error:  # pragma: no cover - backend diagnostic
                violations.append(f"{alias}: cannot inspect model: {error}")
        if tensor_count <= 0 or parameter_count <= 0:
            violations.append(f"{alias}: exported model tensor map is empty")
        if nonfinite:
            violations.append(
                f"{alias}: nonfinite exported tensors: "
                + ", ".join(nonfinite[:8])
            )
        checkpoints.append(
            {
                "alias": alias,
                "arm": arm,
                "seed": seed,
                "job_id": job_id,
                "run_stamp": job["run_stamp"],
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_tree_sha256": _tree_hash(checkpoint),
                "model_tensor_count": tensor_count,
                "model_parameter_count": parameter_count,
                "nonfinite_model_tensors": nonfinite,
            }
        )

    status = (
        "fail"
        if violations
        else "pass"
        if len(checkpoints) == 6 and not pending
        else "in_progress"
    )
    return {
        "schema": "e69_gate4_checkpoint_audit_v1",
        "status": status,
        "identity": str(identity_path.resolve()),
        "identity_sha256": _sha256(identity_path),
        "gate3_audit": str(gate3_audit_path.resolve()),
        "gate3_audit_sha256": _sha256(gate3_audit_path),
        "math500_sealed": True,
        "expected_checkpoints": 6,
        "observed_checkpoints": len(checkpoints),
        "checkpoint_name": CHECKPOINT_NAME,
        "checkpoints": checkpoints,
        "pending": sorted(set(pending)),
        "violations": sorted(set(violations)),
    }


def main() -> None:
    payload = audit_checkpoints()
    _atomic_json(OUT, payload)
    print(
        f"[e69-gate4-checkpoints] status={payload['status']} "
        f"checkpoints={payload['observed_checkpoints']}/6 "
        f"violations={len(payload['violations'])}"
    )


if __name__ == "__main__":
    main()
