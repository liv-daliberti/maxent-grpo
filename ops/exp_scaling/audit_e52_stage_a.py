#!/usr/bin/env python3
"""Audit E52 Stage A across every run, seed, domain, and seed mean."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SENTINEL_AUDITOR_PATH = (
    ROOT / "ops/exp_scaling/audit_e52_sentinel.py"
)
APPROVAL_VERIFIER_PATH = (
    ROOT / "ops/exp_scaling/verify_e52_sentinel_approval.py"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SENTINEL = _load_module("audit_e52_sentinel_for_stage_a", SENTINEL_AUDITOR_PATH)
APPROVAL = _load_module(
    "verify_e52_sentinel_approval_for_stage_a",
    APPROVAL_VERIFIER_PATH,
)

ARMS = SENTINEL.ARMS
SEEDS = (43, 44, 45)
DOMAINS = {
    "countdown": (
        "cde52_direct_inverse_entropy_canonical_05b_50ep_stage_a_v1_allcs",
        384,
    ),
    "graph_coloring": (
        "gce52_direct_inverse_entropy_canonical_05b_50ep_stage_a_v1",
        192,
    ),
    "python_factor": (
        "pye52_direct_inverse_entropy_canonical_05b_50ep_stage_a_v1_allcs",
        384,
    ),
}
IDENTITY_PATH = (
    ROOT
    / "var/artifacts/"
    "e52_direct_inverse_entropy_canonical_05b_stage_a_v1_identity.json"
)
SENTINEL_IDENTITY_PATH = SENTINEL.IDENTITY_PATH
APPROVAL_PATH = SENTINEL.APPROVAL_PATH
STABILITY_PROTOCOL_PATH = SENTINEL.STABILITY_AMENDMENT_PATH
STAGE_PROTOCOL_PATH = (
    ROOT
    / "paper/preregistration/"
    "e52_stage_a_execution_20260726.md"
)
STAGE_LAUNCHER_PATH = ROOT / "ops/exp_scaling/launch_e52_stage_a.sh"
STAGE_WATCHER_PATH = ROOT / "ops/exp_scaling/watch_e52_stage_a.sh"
STAGE_WATCHER_SLURM_PATH = ROOT / "ops/slurm/watch_e52_stage_a.slurm"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is unavailable or invalid: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return payload


def stage_binding() -> tuple[dict[str, Any], list[str]]:
    try:
        identity = _json_object(IDENTITY_PATH, "Stage-A identity")
    except RuntimeError as error:
        return {}, [str(error)]
    if (
        identity.get("schema")
        != "e52_direct_inverse_entropy_canonical_05b_stage_a_v1"
    ):
        return {}, ["Stage-A identity schema is incompatible"]

    approval_sha256 = str(identity.get("sentinel_approval_sha256", ""))
    try:
        approval_summary = APPROVAL.verify_approval(
            approval_path=APPROVAL_PATH,
            expected_approval_sha256=approval_sha256,
        )
        observed = {
            "stage_protocol_sha256": _sha256_file(STAGE_PROTOCOL_PATH),
            "stability_protocol_sha256": _sha256_file(
                STABILITY_PROTOCOL_PATH
            ),
            "launcher_sha256": _sha256_file(STAGE_LAUNCHER_PATH),
            "approval_verifier_sha256": _sha256_file(
                APPROVAL_VERIFIER_PATH
            ),
            "stage_a_auditor_sha256": _sha256_file(
                Path(__file__).resolve()
            ),
            "stage_a_watcher_sha256": _sha256_file(
                STAGE_WATCHER_PATH
            ),
            "stage_a_watcher_slurm_sha256": _sha256_file(
                STAGE_WATCHER_SLURM_PATH
            ),
            "sentinel_identity_sha256": _sha256_file(
                SENTINEL_IDENTITY_PATH
            ),
            "sentinel_approval_sha256": _sha256_file(APPROVAL_PATH),
            "source_hash": APPROVAL._hash_tree(
                Path(approval_summary["source_root"])
            ),
            "execution_surface_hash": APPROVAL._hash_tree(
                Path(approval_summary["execution_root"])
            ),
        }
    except (OSError, RuntimeError, APPROVAL.ApprovalError) as error:
        return {}, [f"Stage-A approval binding failed: {error}"]

    violations = []
    for key, value in observed.items():
        if identity.get(key) != value:
            violations.append(
                f"Stage-A identity mismatch for {key}: "
                f"expected={identity.get(key)!r} observed={value!r}"
            )
    return {
        "identity_path": str(IDENTITY_PATH.resolve()),
        "identity_sha256": _sha256_file(IDENTITY_PATH),
        "sentinel_approval": str(APPROVAL_PATH.resolve()),
        **observed,
    }, violations


def _finite_evaluations(run: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(row["step"]): row
        for row in run["evaluations"]
        if (
            int(row["step"]) > 0
            and row["distinct8"] is not None
            and row["pass8"] is not None
            and row["mean8"] is not None
        )
    }


def seed_mean_behavioral_gate(
    seeds: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    control_by_seed = {
        seed: _finite_evaluations(payload["runs"]["grpo"])
        for seed, payload in seeds.items()
    }
    hybrid_by_seed = {
        seed: _finite_evaluations(
            payload["runs"]["maxent_inverse_canonical"]
        )
        for seed, payload in seeds.items()
    }
    paired_steps = sorted(
        set.intersection(
            *(
                set(control_by_seed[seed]) & set(hybrid_by_seed[seed])
                for seed in SEEDS
            )
        )
    )
    if len(paired_steps) < 8:
        return {
            "status": "pending",
            "paired_seed_mean_boundaries": len(paired_steps),
        }

    def mean_row(
        rows_by_seed: dict[int, dict[int, dict[str, Any]]],
        step: int,
    ) -> dict[str, Any]:
        row = {
            key: sum(rows_by_seed[seed][step][key] for seed in SEEDS)
            / len(SEEDS)
            for key in ("distinct8", "pass8", "mean8")
        }
        row["step"] = step
        return row

    all_complete = all(
        payload["runs"][arm]["status"] == "complete"
        for payload in seeds.values()
        for arm in ("grpo", "maxent_inverse_canonical")
    )
    control = {
        "status": "complete" if all_complete else "running",
        "evaluations": [
            mean_row(control_by_seed, step) for step in paired_steps
        ],
    }
    hybrid = {
        "status": "complete" if all_complete else "running",
        "evaluations": [
            mean_row(hybrid_by_seed, step) for step in paired_steps
        ],
    }
    result = SENTINEL.behavioral_gate(control, hybrid)
    result["paired_seed_mean_boundaries"] = len(paired_steps)
    return result


def audit(data_root: Path) -> dict[str, Any]:
    binding, binding_violations = stage_binding()
    domains: dict[str, Any] = {}
    for domain, (prefix, prompt_pool_size) in DOMAINS.items():
        seed_payloads: dict[int, dict[str, Any]] = {}
        for seed in SEEDS:
            runs = {
                arm: SENTINEL.audit_run(
                    SENTINEL._load_records(
                        data_root,
                        f"{prefix}_{arm}_s{seed}",
                    ),
                    arm=arm,
                    prompt_pool_size=prompt_pool_size,
                )
                for arm in ARMS
            }
            seed_payloads[seed] = {
                "runs": runs,
                "safety_gate": SENTINEL.safety_gate(runs),
                "behavioral_gate": SENTINEL.behavioral_gate(
                    runs["grpo"],
                    runs["maxent_inverse_canonical"],
                ),
            }
        domains[domain] = {
            "seeds": seed_payloads,
            "seed_mean_behavioral_gate": seed_mean_behavioral_gate(
                seed_payloads
            ),
        }

    violations = list(binding_violations) + [
        f"{domain}/seed{seed}/{arm}: {violation}"
        for domain, domain_payload in domains.items()
        for seed, seed_payload in domain_payload["seeds"].items()
        for arm, run in seed_payload["runs"].items()
        for violation in run["violations"]
    ]
    all_complete = all(
        run["status"] == "complete"
        for domain_payload in domains.values()
        for seed_payload in domain_payload["seeds"].values()
        for run in seed_payload["runs"].values()
    )
    gate_statuses = [
        gate["status"]
        for domain_payload in domains.values()
        for seed_payload in domain_payload["seeds"].values()
        for gate in (
            seed_payload["safety_gate"],
            seed_payload["behavioral_gate"],
        )
    ] + [
        domain_payload["seed_mean_behavioral_gate"]["status"]
        for domain_payload in domains.values()
    ]
    status = (
        "fail"
        if violations or "fail" in gate_statuses
        else "pass"
        if all_complete and all(value == "pass" for value in gate_statuses)
        else "in_progress"
    )
    return {
        "schema": "e52_stage_a_audit_v1",
        "status": status,
        "violations": violations,
        "approval_binding": binding,
        "domains": domains,
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "var/data")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "var/artifacts/e52_stage_a_audit_latest.json",
    )
    args = parser.parse_args()
    payload = audit(args.data_root)
    _atomic_write(args.out, payload)
    print(
        f"[e52-stage-a-audit] status={payload['status']} "
        f"violations={len(payload['violations'])} out={args.out}"
    )
    for domain, domain_payload in payload["domains"].items():
        positions = []
        for seed, seed_payload in domain_payload["seeds"].items():
            hybrid = seed_payload["runs"]["maxent_inverse_canonical"]
            positions.append(
                f"s{seed}={hybrid['status']}@{hybrid['training_passes']:.2f}"
            )
        print(
            f"[e52-stage-a-audit] {domain}: {', '.join(positions)}; "
            "seed_mean="
            f"{domain_payload['seed_mean_behavioral_gate']['status']}"
        )
    return 1 if payload["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
