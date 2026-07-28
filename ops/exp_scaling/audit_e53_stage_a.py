#!/usr/bin/env python3
"""Audit all E53 Stage-A runs, per-seed gates, and seed-mean gates."""

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


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SENTINEL_PATH = ROOT / "ops/exp_scaling/audit_e53_sentinel_v2.py"
VERIFIER_PATH = ROOT / "ops/exp_scaling/verify_e53_sentinel_approval.py"
SENTINEL = _load_module("audit_e53_sentinel_for_stage_a", SENTINEL_PATH)
VERIFIER = _load_module("verify_e53_approval_for_stage_a", VERIFIER_PATH)
ARMS = SENTINEL.ARMS
SEEDS = (43, 44, 45)
DOMAINS = {
    "countdown": ("cde53_verified_replay_05b_50ep_stage_a_allcs", 384),
    "graph_coloring": ("gce53_verified_replay_05b_50ep_stage_a", 192),
    "python_factor": ("pye53_verified_replay_05b_50ep_stage_a_allcs", 384),
}
IDENTITY_PATH = (
    ROOT / "var/artifacts/e53_verified_replay_05b_stage_a_identity.json"
)
APPROVAL_PATH = SENTINEL.APPROVAL_PATH
SENTINEL_IDENTITY_PATH = SENTINEL.IDENTITY_PATH
PROTOCOL_PATH = (
    ROOT / "paper/preregistration/e53_stage_a_execution_20260726.md"
)
RUNTIME_REPAIR_PROTOCOL_PATH = (
    ROOT
    / "paper/preregistration/e53_runtime_audit_scaling_repair_20260726.md"
)
LAUNCHER_PATH = ROOT / "ops/exp_scaling/launch_e53_stage_a.sh"
WATCHER_PATH = ROOT / "ops/exp_scaling/watch_e53_stage_a.sh"
WATCHER_SLURM_PATH = ROOT / "ops/slurm/watch_e53_stage_a.slurm"


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
        identity = _json_object(IDENTITY_PATH, "E53 Stage-A identity")
    except RuntimeError as error:
        return {}, [str(error)]
    if identity.get("schema") != "e53_verified_replay_05b_stage_a_v1":
        return {}, ["E53 Stage-A identity schema is incompatible"]
    try:
        approval = VERIFIER.verify_approval(
            approval_path=APPROVAL_PATH,
            expected_approval_sha256=identity.get("sentinel_approval_sha256"),
        )
        observed = {
            "stage_protocol_sha256": _sha256_file(PROTOCOL_PATH),
            "runtime_repair_protocol_sha256": _sha256_file(
                RUNTIME_REPAIR_PROTOCOL_PATH
            ),
            "sentinel_runtime_auditor_sha256": _sha256_file(
                SENTINEL_PATH
            ),
            "launcher_sha256": _sha256_file(LAUNCHER_PATH),
            "approval_verifier_sha256": _sha256_file(VERIFIER_PATH),
            "stage_a_auditor_sha256": _sha256_file(Path(__file__).resolve()),
            "stage_a_watcher_sha256": _sha256_file(WATCHER_PATH),
            "stage_a_watcher_slurm_sha256": _sha256_file(WATCHER_SLURM_PATH),
            "sentinel_identity_sha256": _sha256_file(SENTINEL_IDENTITY_PATH),
            "sentinel_approval_sha256": _sha256_file(APPROVAL_PATH),
            "source_hash": approval["source_hash"],
            "execution_surface_hash": approval["execution_surface_hash"],
        }
    except (OSError, RuntimeError, VERIFIER.ApprovalError) as error:
        return {}, [f"E53 Stage-A binding failed: {error}"]
    violations = [
        (
            f"E53 Stage-A identity mismatch for {key}: "
            f"expected={identity.get(key)!r} observed={value!r}"
        )
        for key, value in observed.items()
        if identity.get(key) != value
    ]
    return {
        "identity_path": str(IDENTITY_PATH.resolve()),
        "identity_sha256": _sha256_file(IDENTITY_PATH),
        **observed,
    }, violations


def _finite_evaluations(run: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(row["step"]): row
        for row in run["evaluations"]
        if int(row["step"]) > 0
        and all(row[key] is not None for key in ("distinct8", "pass8", "mean8"))
    }


def seed_mean_behavioral_gate(
    seeds: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    control_by_seed = {
        seed: _finite_evaluations(payload["runs"]["grpo"])
        for seed, payload in seeds.items()
    }
    replay_by_seed = {
        seed: _finite_evaluations(payload["runs"][SENTINEL.REPLAY_ARM])
        for seed, payload in seeds.items()
    }
    paired_steps = sorted(
        set.intersection(
            *(
                set(control_by_seed[seed]) & set(replay_by_seed[seed])
                for seed in SEEDS
            )
        )
    )
    if len(paired_steps) < 8:
        return {
            "status": "pending",
            "paired_seed_mean_boundaries": len(paired_steps),
        }

    def mean_rows(
        rows: dict[int, dict[int, dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "step": step,
                **{
                    key: sum(rows[seed][step][key] for seed in SEEDS)
                    / len(SEEDS)
                    for key in ("distinct8", "pass8", "mean8")
                },
            }
            for step in paired_steps
        ]

    complete = all(
        payload["runs"][arm]["status"] == "complete"
        for payload in seeds.values()
        for arm in ("grpo", SENTINEL.REPLAY_ARM)
    )
    status = "complete" if complete else "running"
    result = SENTINEL.behavioral_gate(
        {"status": status, "evaluations": mean_rows(control_by_seed)},
        {"status": status, "evaluations": mean_rows(replay_by_seed)},
    )
    result["paired_seed_mean_boundaries"] = len(paired_steps)
    return result


def audit(data_root: Path) -> dict[str, Any]:
    binding, binding_violations = stage_binding()
    domains: dict[str, Any] = {}
    for domain, (prefix, prompt_pool_size) in DOMAINS.items():
        seeds: dict[int, dict[str, Any]] = {}
        for seed in SEEDS:
            runs: dict[str, Any] = {}
            for arm in ARMS:
                run_stamp = f"{prefix}_{arm}_s{seed}"
                run = SENTINEL.audit_run(
                    SENTINEL._load_records(data_root, run_stamp),
                    arm=arm,
                    prompt_pool_size=prompt_pool_size,
                )
                if run["status"] == "complete":
                    checkpoint = SENTINEL.checkpoint_gate(
                        data_root,
                        run_stamp=run_stamp,
                        arm=arm,
                        terminal_step=run["terminal_step"],
                    )
                    run["checkpoint_gate"] = checkpoint
                    run["violations"].extend(checkpoint["violations"])
                else:
                    run["checkpoint_gate"] = {"status": "pending"}
                runs[arm] = run
            seeds[seed] = {
                "runs": runs,
                "safety_gate": SENTINEL.safety_gate(runs),
                "behavioral_gate": SENTINEL.behavioral_gate(
                    runs["grpo"], runs[SENTINEL.REPLAY_ARM]
                ),
            }
        domains[domain] = {
            "seeds": seeds,
            "seed_mean_behavioral_gate": seed_mean_behavioral_gate(seeds),
        }
    violations = list(binding_violations) + [
        f"{domain}/seed{seed}/{arm}: {violation}"
        for domain, payload in domains.items()
        for seed, seed_payload in payload["seeds"].items()
        for arm, run in seed_payload["runs"].items()
        for violation in run["violations"]
    ]
    complete = all(
        run["status"] == "complete"
        for payload in domains.values()
        for seed_payload in payload["seeds"].values()
        for run in seed_payload["runs"].values()
    )
    gate_statuses = [
        gate["status"]
        for payload in domains.values()
        for seed_payload in payload["seeds"].values()
        for gate in (seed_payload["safety_gate"], seed_payload["behavioral_gate"])
    ] + [
        payload["seed_mean_behavioral_gate"]["status"]
        for payload in domains.values()
    ]
    status = (
        "fail"
        if violations or "fail" in gate_statuses
        else "pass"
        if complete and all(value == "pass" for value in gate_statuses)
        else "in_progress"
    )
    return {
        "schema": "e53_stage_a_audit_v1",
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
        default=ROOT / "var/artifacts/e53_stage_a_audit_latest.json",
    )
    args = parser.parse_args()
    payload = audit(args.data_root)
    _atomic_write(args.out, payload)
    print(
        f"[e53-stage-a-audit] status={payload['status']} "
        f"violations={len(payload['violations'])} out={args.out}"
    )
    for domain, domain_payload in payload["domains"].items():
        positions = []
        for seed, seed_payload in domain_payload["seeds"].items():
            replay = seed_payload["runs"][SENTINEL.REPLAY_ARM]
            positions.append(
                f"s{seed}={replay['status']}@{replay['training_passes']:.2f}"
            )
        print(
            f"[e53-stage-a-audit] {domain}: {', '.join(positions)}; "
            f"seed_mean={domain_payload['seed_mean_behavioral_gate']['status']}"
        )
    return 1 if payload["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
