#!/usr/bin/env python3
"""Fail-closed audit for E58's global verified-replay sentinel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "ops/exp_scaling/audit_e57_sentinel.py"
SPEC = importlib.util.spec_from_file_location("e58_bound_e57_audit", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import sentinel helpers from {HELPER_PATH}")
E57 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E57)
BASE = E57.BASE

ARM = "verified_first_global_replay_canonical"
SEED = 9010
CAPACITY = 16
GLOBAL_GROUPS = 1
BALANCE_BASE = 0.10
MASS_BASE = 0.10
SEMANTIC_BASE = 0.10
WARMUP = 64
EMA_DECAY = 0.90
DOMAINS = {
    "countdown": {
        "pool": 384,
        "control_prefix": "cde53_verified_replay_05b_50ep_sentinel_allcs",
        "treatment_prefix": (
            "cde58_global_verified_replay_canonical_05b_50ep_sentinel_allcs"
        ),
    },
    "graph_coloring": {
        "pool": 192,
        "control_prefix": "gce53_verified_replay_05b_50ep_sentinel",
        "treatment_prefix": (
            "gce58_global_verified_replay_canonical_05b_50ep_sentinel"
        ),
    },
    "python_factor": {
        "pool": 384,
        "control_prefix": "pye53_verified_replay_05b_50ep_sentinel_allcs",
        "treatment_prefix": (
            "pye58_global_verified_replay_canonical_05b_50ep_sentinel_allcs"
        ),
    },
}
IDENTITY_PATH = (
    ROOT
    / "var/artifacts/e58_global_verified_replay_canonical_05b_sentinel_identity.json"
)
PROTOCOL_PATH = (
    ROOT / "paper/preregistration/e58_global_verified_replay_canonical_05b.md"
)
LAUNCHER_PATH = ROOT / "ops/exp_scaling/launch_e58_global_replay_sentinel.sh"
SMOKE_IDENTITY = (
    ROOT
    / "var/artifacts/e58_global_verified_replay_python_smoke_attempt3_identity.json"
)
SMOKE_AUDIT = ROOT / "var/artifacts/e58_global_replay_smoke_audit_latest.json"
E53_IDENTITY = (
    ROOT / "var/artifacts/e53_verified_replay_05b_sentinel_identity.json"
)
APPROVAL_PATH = ROOT / "var/artifacts/e58_sentinel_stage_a_approval.json"
TREATMENT_MANIFESTS = {
    domain: ROOT / f"var/artifacts/{config['treatment_prefix']}_comparative_jobs.tsv"
    for domain, config in DOMAINS.items()
}
CONTROL_MANIFESTS = {
    domain: ROOT / f"var/artifacts/{config['control_prefix']}_comparative_jobs.tsv"
    for domain, config in DOMAINS.items()
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_tree(root: Path) -> str:
    lines = [
        f"{_sha256_file(path)}  ./{path.relative_to(root).as_posix()}\n"
        for path in sorted(
            (item for item in root.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(root).as_posix(),
        )
    ]
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def _manifest_job(
    path: Path, *, arm: str, run_stamp: str
) -> tuple[int | None, list[str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    except (OSError, csv.Error) as error:
        return None, [f"cannot read job manifest {path}: {error}"]
    matching = [
        row
        for row in rows
        if row.get("arm") == arm
        and row.get("seed") == str(SEED)
        and row.get("run_stamp") == run_stamp
        and str(row.get("job_id", "")).isdigit()
    ]
    if len(matching) != 1:
        return None, [
            f"{path.name} has {len(matching)} exact rows for {run_stamp}"
        ]
    return int(matching[0]["job_id"]), []


def approval_binding() -> tuple[dict[str, Any], list[str]]:
    try:
        identity = json.loads(IDENTITY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {}, [f"E58 identity is unavailable or invalid: {error}"]
    if (
        not isinstance(identity, dict)
        or identity.get("schema")
        != "e58_global_verified_replay_canonical_05b_sentinel_v1"
    ):
        return {}, ["E58 identity has an incompatible schema"]

    source_hash = str(identity.get("source_hash", ""))
    execution_hash = str(identity.get("execution_surface_hash", ""))
    source_root = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e58_global_verified_replay_{source_hash}"
        / "src"
    )
    ops_root = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e58_global_verified_replay_ops_{execution_hash}"
        / "ops"
    )
    evidence_paths = {
        "protocol_sha256": PROTOCOL_PATH,
        "launcher_sha256": LAUNCHER_PATH,
        "auditor_sha256": Path(__file__).resolve(),
        "helper_auditor_sha256": HELPER_PATH,
        "e53_control_identity_sha256": E53_IDENTITY,
        "smoke_identity_sha256": SMOKE_IDENTITY,
        "smoke_audit_sha256": SMOKE_AUDIT,
    }
    try:
        observed: dict[str, Any] = {
            key: _sha256_file(path) for key, path in evidence_paths.items()
        }
        observed["source_hash"] = _hash_tree(source_root)
        observed["execution_surface_hash"] = _hash_tree(ops_root)
        observed["job_manifest_sha256"] = {
            domain: _sha256_file(path)
            for domain, path in TREATMENT_MANIFESTS.items()
        }
        observed["e53_control_manifest_sha256"] = {
            domain: _sha256_file(path)
            for domain, path in CONTROL_MANIFESTS.items()
        }
    except OSError as error:
        return {}, [f"E58 approval evidence is unavailable: {error}"]

    checked = tuple(evidence_paths) + (
        "source_hash",
        "execution_surface_hash",
        "job_manifest_sha256",
        "e53_control_manifest_sha256",
    )
    violations = [
        (
            f"approval binding mismatch for {key}: "
            f"expected={identity.get(key)!r} observed={observed[key]!r}"
        )
        for key in checked
        if identity.get(key) != observed[key]
    ]
    treatment_jobs: dict[str, int] = {}
    control_jobs: dict[str, int] = {}
    for domain, config in DOMAINS.items():
        treatment_stamp = f"{config['treatment_prefix']}_{ARM}_s{SEED}"
        control_stamp = f"{config['control_prefix']}_grpo_s{SEED}"
        treatment_job, errors = _manifest_job(
            TREATMENT_MANIFESTS[domain],
            arm=ARM,
            run_stamp=treatment_stamp,
        )
        violations.extend(errors)
        control_job, errors = _manifest_job(
            CONTROL_MANIFESTS[domain],
            arm="grpo",
            run_stamp=control_stamp,
        )
        violations.extend(errors)
        if treatment_job is not None:
            treatment_jobs[domain] = treatment_job
        if control_job is not None:
            control_jobs[domain] = control_job
    if treatment_jobs != identity.get("jobs"):
        violations.append(
            "approval binding mismatch for jobs: "
            f"expected={identity.get('jobs')!r} observed={treatment_jobs!r}"
        )
    return {
        "identity_path": str(IDENTITY_PATH.resolve()),
        "source_snapshot_root": str(source_root.resolve()),
        "execution_snapshot_root": str(ops_root.resolve()),
        "treatment_jobs": treatment_jobs,
        "control_jobs": control_jobs,
        **observed,
    }, violations


def _checkpoint_gate(
    data_root: Path,
    *,
    run_stamp: str,
    arm: str,
    terminal_step: int,
    job_id: int | None = None,
) -> dict[str, Any]:
    if arm == "grpo":
        return E57.E56.E53_CHECKPOINT_GATE(
            data_root,
            run_stamp=run_stamp,
            arm=arm,
            terminal_step=terminal_step,
        )
    expected_tag = f"step_{terminal_step:05d}"
    attempt = f"debug_job{job_id}" if job_id is not None else "debug_*"
    candidates = sorted(
        data_root.glob(
            f"*_{run_stamp}/{attempt}/checkpoints/{expected_tag}/"
            "mp_rank_00_model_states.pt"
        )
    )
    if len(candidates) != 1:
        return {
            "status": "fail",
            "violations": [
                f"expected one terminal checkpoint {expected_tag}, "
                f"found {len(candidates)}"
            ],
        }
    path = candidates[0]
    try:
        import torch

        state = torch.load(
            path, map_location="cpu", weights_only=False, mmap=True
        )
    except Exception as error:  # pragma: no cover - backend diagnostic
        return {
            "status": "fail",
            "path": str(path),
            "violations": [f"cannot inspect terminal checkpoint: {error}"],
        }
    violations: list[str] = []
    if state.get("maxent_alpha_controller_state") is not None:
        violations.append(
            "checkpoint unexpectedly contains a direct MaxEnt controller"
        )
    expected_states = {
        "canonical_replay_controller_state": (
            "canonical_replay_inverse",
            "unprojected_warmup_inverse_observed_bank_entropy_v1",
            BALANCE_BASE,
        ),
        "canonical_replay_mass_controller_state": (
            "canonical_replay_likelihood",
            "unprojected_warmup_verified_surprisal_ratio_v1",
            MASS_BASE,
        ),
    }
    for state_key, (kind, rule, base) in expected_states.items():
        payload = state.get(state_key)
        if not isinstance(payload, dict):
            violations.append(f"checkpoint lacks {state_key}")
        elif (
            payload.get("controller_kind") != kind
            or payload.get("controller_rule") != rule
            or payload.get("base_alpha") != base
            or payload.get("warmup_steps") != WARMUP
            or payload.get("ema_decay") != EMA_DECAY
        ):
            violations.append(f"checkpoint {state_key} configuration mismatch")
    semantic = state.get("semantic_shannon_tracker_state")
    controller = (
        semantic.get("open_set_controller")
        if isinstance(semantic, dict)
        else None
    )
    if (
        not isinstance(controller, dict)
        or controller.get("controller_kind") != "semantic_open_set_inverse"
        or controller.get("controller_rule")
        != "unprojected_warmup_inverse_open_set_entropy_v1"
        or controller.get("base_coefficient") != SEMANTIC_BASE
        or controller.get("warmup_steps") != WARMUP
        or controller.get("ema_decay") != EMA_DECAY
    ):
        violations.append("checkpoint semantic controller mismatch")
    bank = state.get("online_canonical_bank_state")
    cursor = bank.get("global_replay_cursor") if isinstance(bank, dict) else None
    if (
        not isinstance(bank, dict)
        or bank.get("retain_exemplars") is not True
        or bank.get("replay_capacity") != CAPACITY
        or bank.get("global_replay_groups_per_step") != GLOBAL_GROUPS
        or isinstance(cursor, bool)
        or not isinstance(cursor, int)
        or cursor < 0
    ):
        violations.append("checkpoint global replay bank is incompatible")
    return {
        "status": "pass" if not violations else "fail",
        "path": str(path),
        "tag": expected_tag,
        "scheduler_cursor": cursor,
        "violations": violations,
    }


def _scheduler_gate(
    records: list[dict[str, Any]], *, complete: bool
) -> dict[str, Any]:
    rows = BASE._latest_by_step(
        records, "train/canonical_replay_global_scheduler_active"
    )
    violations: list[str] = []
    first_discovery: int | None = None
    post_discovery = 0
    activations = 0
    previous_mass_observations = 0
    for row in rows:
        step_value = E57._finite(row, "trainer/global_step")
        step = int(step_value) if step_value is not None else -1
        tracked = E57._finite(row, "train/online_canonical_tracked_outcomes")
        groups = E57._finite(row, "train/canonical_replay_available_groups")
        mass_observations = E57._finite(
            row, "train/canonical_replay_mass_observations"
        )
        if tracked is not None and tracked > 0 and first_discovery is None:
            first_discovery = step
        if first_discovery is None:
            if groups is None or not math.isclose(
                groups, 0.0, rel_tol=0.0, abs_tol=1e-12
            ):
                violations.append(
                    f"step {step}: global replay active before discovery"
                )
        else:
            post_discovery += 1
            expected_one = (
                "train/canonical_replay_global_scheduler_active",
                "train/canonical_replay_global_groups_per_step",
                "train/canonical_replay_available_groups",
                "train/canonical_replay_actuator_groups",
            )
            for key in expected_one:
                value = E57._finite(row, key)
                if value is None or not math.isclose(
                    value, 1.0, rel_tol=0.0, abs_tol=1e-12
                ):
                    violations.append(
                        f"step {step}: {key}={value!r}, expected one"
                    )
            if groups is not None and groups > 0:
                activations += 1
            expected_mass = previous_mass_observations + 1
            if (
                mass_observations is None
                or int(mass_observations) != expected_mass
            ):
                violations.append(
                    f"step {step}: mass observations "
                    f"{mass_observations!r}, expected {expected_mass}"
                )
        if mass_observations is not None:
            previous_mass_observations = int(mass_observations)
    if complete:
        if first_discovery is None:
            violations.append("terminal run never discovered a verified bank")
        if post_discovery < 32:
            violations.append("fewer than 32 post-discovery replay updates")
        if activations != post_discovery:
            violations.append("global replay skipped a post-discovery update")
    status = (
        "fail"
        if violations
        else "pass"
        if complete
        else "running"
        if rows
        else "pending"
    )
    return {
        "status": status,
        "first_discovery_step": first_discovery,
        "post_discovery_records": post_discovery,
        "replay_activations": activations,
        "mass_observations": previous_mass_observations,
        "violations": violations,
    }


def _configure_bound_helper() -> None:
    E57.ARM = ARM
    E57.SEED = SEED
    E57.CAPACITY = CAPACITY
    E57.BALANCE_BASE = BALANCE_BASE
    E57.MASS_BASE = MASS_BASE
    E57.SEMANTIC_BASE = SEMANTIC_BASE
    E57.WARMUP = WARMUP
    E57.EMA_DECAY = EMA_DECAY
    E57.DOMAINS = DOMAINS
    E57.IDENTITY_PATH = IDENTITY_PATH
    E57.PROTOCOL_PATH = PROTOCOL_PATH
    E57.LAUNCHER_PATH = LAUNCHER_PATH
    E57.PYTHON_SMOKE_IDENTITY = SMOKE_IDENTITY
    E57.GRAPH_SMOKE_IDENTITY = SMOKE_IDENTITY
    E57.PYTHON_SMOKE_AUDIT = SMOKE_AUDIT
    E57.GRAPH_SMOKE_AUDIT = SMOKE_AUDIT
    E57.E53_IDENTITY = E53_IDENTITY
    E57.TREATMENT_MANIFESTS = TREATMENT_MANIFESTS
    E57.CONTROL_MANIFESTS = CONTROL_MANIFESTS
    E57.approval_binding = approval_binding
    E57._checkpoint_gate = _checkpoint_gate
    BASE.REPLAY_ARM = ARM
    BASE.checkpoint_gate = _checkpoint_gate


def audit(data_root: Path) -> dict[str, Any]:
    _configure_bound_helper()
    payload = E57.audit(data_root)
    binding = payload.get("approval_binding", {})
    treatment_jobs = binding.get("treatment_jobs", {})
    scheduler_states: list[str] = []
    for domain, config in DOMAINS.items():
        domain_payload = payload["domains"][domain]
        treatment = domain_payload["runs"].pop("e57")
        domain_payload["runs"]["e58"] = treatment
        treatment_stamp = f"{config['treatment_prefix']}_{ARM}_s{SEED}"
        records, _ = E57.E56._load_bound_records(
            data_root,
            run_stamp=treatment_stamp,
            job_id=treatment_jobs.get(domain),
        )
        scheduler = _scheduler_gate(
            records,
            complete=treatment["status"] == "complete",
        )
        domain_payload["global_scheduler_gate"] = scheduler
        scheduler_states.append(scheduler["status"])
        payload["violations"].extend(
            f"{domain}/global_scheduler_gate: {item}"
            for item in scheduler["violations"]
        )

    base_gate_states = [
        domain_payload[key]["status"]
        for domain_payload in payload["domains"].values()
        for key in (
            "behavioral_gate",
            "safety_gate",
            "semantic_gate",
            "cold_start_gate",
        )
    ]
    all_complete = all(
        run["status"] == "complete"
        for domain_payload in payload["domains"].values()
        for run in domain_payload["runs"].values()
    )
    all_states = base_gate_states + scheduler_states
    status = (
        "fail"
        if payload["violations"] or "fail" in all_states
        else "pass"
        if all_complete and all(item == "pass" for item in all_states)
        else "in_progress"
    )
    payload.update(
        {
            "schema": "e58_global_verified_replay_sentinel_audit_v1",
            "status": status,
            "authorizes_stage_a": status == "pass",
        }
    )
    return payload


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
        default=ROOT / "var/artifacts/e58_sentinel_audit_latest.json",
    )
    args = parser.parse_args()
    try:
        payload = audit(args.data_root)
    except Exception as error:  # pragma: no cover - fail-closed CLI
        payload = {
            "schema": "e58_global_verified_replay_sentinel_audit_v1",
            "status": "fail",
            "authorizes_stage_a": False,
            "violations": [f"uncaught audit failure: {error}"],
        }
    _atomic_write(args.out, payload)
    if payload["status"] == "pass":
        _atomic_write(APPROVAL_PATH, payload)
    print(
        f"[e58-sentinel-audit] status={payload['status']} "
        f"violations={len(payload.get('violations', []))} out={args.out}"
    )
    return 1 if payload["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
