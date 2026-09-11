#!/usr/bin/env python3
"""Fail-closed audit for E57's verified-first three-domain sentinel."""

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
HELPER_PATH = ROOT / "ops/exp_scaling/audit_e56_sentinel.py"
SPEC = importlib.util.spec_from_file_location("e57_bound_e56_audit", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import sentinel helpers from {HELPER_PATH}")
E56 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E56)
BASE = E56.BASE

ARM = "verified_first_split_canonical"
SEED = 9010
CAPACITY = 16
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
            "cde57_verified_first_split_canonical_05b_50ep_sentinel_allcs"
        ),
    },
    "graph_coloring": {
        "pool": 192,
        "control_prefix": "gce53_verified_replay_05b_50ep_sentinel",
        "treatment_prefix": (
            "gce57_verified_first_split_canonical_05b_50ep_sentinel"
        ),
    },
    "python_factor": {
        "pool": 384,
        "control_prefix": "pye53_verified_replay_05b_50ep_sentinel_allcs",
        "treatment_prefix": (
            "pye57_verified_first_split_canonical_05b_50ep_sentinel_allcs"
        ),
    },
}
IDENTITY_PATH = (
    ROOT
    / "var/artifacts/e57_verified_first_split_canonical_05b_sentinel_identity.json"
)
PROTOCOL_PATH = (
    ROOT / "paper/preregistration/e57_verified_first_split_canonical_05b.md"
)
LAUNCHER_PATH = ROOT / "ops/exp_scaling/launch_e57_verified_first_sentinel.sh"
PYTHON_SMOKE_IDENTITY = (
    ROOT / "var/artifacts/e57_verified_first_split_python_smoke_identity.json"
)
GRAPH_SMOKE_IDENTITY = (
    ROOT / "var/artifacts/e57_verified_first_split_graph_smoke_identity.json"
)
PYTHON_SMOKE_AUDIT = (
    ROOT / "var/artifacts/e57_python_smoke_audit_latest.json"
)
GRAPH_SMOKE_AUDIT = ROOT / "var/artifacts/e57_graph_smoke_audit_latest.json"
E53_IDENTITY = (
    ROOT / "var/artifacts/e53_verified_replay_05b_sentinel_identity.json"
)
APPROVAL_PATH = ROOT / "var/artifacts/e57_sentinel_stage_a_approval.json"
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
        return {}, [f"E57 identity is unavailable or invalid: {error}"]
    if (
        not isinstance(identity, dict)
        or identity.get("schema")
        != "e57_verified_first_split_canonical_05b_sentinel_v1"
    ):
        return {}, ["E57 identity has an incompatible schema"]

    source_hash = str(identity.get("source_hash", ""))
    execution_hash = str(identity.get("execution_surface_hash", ""))
    source_root = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e57_verified_first_{source_hash}"
        / "src"
    )
    ops_root = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e57_verified_first_ops_{execution_hash}"
        / "ops"
    )
    evidence_paths = {
        "protocol_sha256": PROTOCOL_PATH,
        "launcher_sha256": LAUNCHER_PATH,
        "auditor_sha256": Path(__file__).resolve(),
        "helper_auditor_sha256": HELPER_PATH,
        "e53_control_identity_sha256": E53_IDENTITY,
        "python_smoke_identity_sha256": PYTHON_SMOKE_IDENTITY,
        "graph_smoke_identity_sha256": GRAPH_SMOKE_IDENTITY,
        "python_smoke_audit_sha256": PYTHON_SMOKE_AUDIT,
        "graph_smoke_audit_sha256": GRAPH_SMOKE_AUDIT,
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
        return {}, [f"E57 approval evidence is unavailable: {error}"]

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


def _finite(record: dict[str, Any], key: str) -> float | None:
    return BASE._finite(record, key)


def _check_no_direct_controller(
    record: dict[str, Any],
    *,
    step: int,
    reference: Any,
) -> tuple[Any, list[str], dict[str, float]]:
    present = sorted(key for key in record if key.startswith("train/maxent_"))
    violations = (
        [f"step {step}: direct token-MaxEnt telemetry is present: {present}"]
        if present
        else []
    )
    return reference, violations, {}


def _checkpoint_gate(
    data_root: Path,
    *,
    run_stamp: str,
    arm: str,
    terminal_step: int,
    job_id: int | None = None,
) -> dict[str, Any]:
    if arm == "grpo":
        return E56.E53_CHECKPOINT_GATE(
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
        violations.append("checkpoint unexpectedly has a direct MaxEnt controller")
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
        not isinstance(semantic, dict)
        or semantic.get("schema")
        != "semantic_shannon_tracker_v4_open_set_inverse"
        or semantic.get("open_set_inverse_adaptation") is not True
        or not isinstance(controller, dict)
        or controller.get("controller_kind") != "semantic_open_set_inverse"
        or controller.get("controller_rule")
        != "unprojected_warmup_inverse_open_set_entropy_v1"
        or controller.get("base_coefficient") != SEMANTIC_BASE
        or controller.get("warmup_steps") != WARMUP
        or controller.get("ema_decay") != EMA_DECAY
    ):
        violations.append("checkpoint semantic controller mismatch")
    bank = state.get("online_canonical_bank_state")
    if (
        not isinstance(bank, dict)
        or bank.get("retain_exemplars") is not True
        or bank.get("replay_capacity") != CAPACITY
    ):
        violations.append("checkpoint verified bank is incompatible")
    return {
        "status": "pass" if not violations else "fail",
        "path": str(path),
        "tag": expected_tag,
        "violations": violations,
    }


BASE.REPLAY_ARM = ARM
BASE._check_direct_controller = _check_no_direct_controller
BASE._check_replay_controller = E56._check_split_replay
BASE.checkpoint_gate = _checkpoint_gate


def _cold_start_gate(
    records: list[dict[str, Any]], *, complete: bool
) -> dict[str, Any]:
    rows = BASE._latest_by_step(
        records, "train/canonical_replay_available_groups"
    )
    violations: list[str] = []
    first_discovery: int | None = None
    checked = 0
    for row in rows:
        step = int(_finite(row, "trainer/global_step") or -1)
        direct = sorted(key for key in row if key.startswith("train/maxent_"))
        if direct:
            violations.append(f"step {step}: direct MaxEnt telemetry present")
        groups = _finite(row, "train/canonical_replay_available_groups")
        if groups is not None and groups > 0:
            if first_discovery is None:
                first_discovery = step
            continue
        if first_discovery is not None:
            continue
        checked += 1
        expected_zero = (
            "train/online_canonical_task_reward_mean",
            "train/semantic_shannon_augmented_reward_mean",
            "train/policy_grad_norm",
        )
        for key in expected_zero:
            value = _finite(row, key)
            if value is None or not math.isclose(
                value, 0.0, rel_tol=0.0, abs_tol=1e-12
            ):
                violations.append(
                    f"step {step}: pre-discovery {key}={value!r}, expected zero"
                )
    if complete and first_discovery is None:
        violations.append("terminal run never made a verifier-positive discovery")
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
        "pre_discovery_points": checked,
        "first_discovery_step": first_discovery,
        "direct_token_entropy": "absent" if not violations else "violation",
        "violations": violations,
    }


def audit(data_root: Path) -> dict[str, Any]:
    binding, binding_violations = approval_binding()
    treatment_jobs = binding.get("treatment_jobs", {})
    control_jobs = binding.get("control_jobs", {})
    domains: dict[str, Any] = {}
    for domain, config in DOMAINS.items():
        pool = int(config["pool"])
        control_stamp = f"{config['control_prefix']}_grpo_s{SEED}"
        treatment_stamp = f"{config['treatment_prefix']}_{ARM}_s{SEED}"
        control_records, control_attempt_violations = E56._load_bound_records(
            data_root,
            run_stamp=control_stamp,
            job_id=control_jobs.get(domain),
        )
        treatment_records, treatment_attempt_violations = (
            E56._load_bound_records(
                data_root,
                run_stamp=treatment_stamp,
                job_id=treatment_jobs.get(domain),
            )
        )
        control = BASE.audit_run(
            control_records, arm="grpo", prompt_pool_size=pool
        )
        treatment = BASE.audit_run(
            treatment_records, arm=ARM, prompt_pool_size=pool
        )
        control["violations"].extend(control_attempt_violations)
        treatment["violations"].extend(treatment_attempt_violations)
        for run, stamp, run_arm, job_id in (
            (control, control_stamp, "grpo", control_jobs.get(domain)),
            (treatment, treatment_stamp, ARM, treatment_jobs.get(domain)),
        ):
            run["checkpoint_gate"] = (
                BASE.checkpoint_gate(
                    data_root,
                    run_stamp=stamp,
                    arm=run_arm,
                    terminal_step=run["terminal_step"],
                    job_id=job_id,
                )
                if run["status"] == "complete"
                else {"status": "pending"}
            )
            run["violations"].extend(
                run["checkpoint_gate"].get("violations", [])
            )
        semantic = E56._semantic_audit(
            treatment_records,
            complete=treatment["status"] == "complete",
        )
        cold_start = _cold_start_gate(
            treatment_records,
            complete=treatment["status"] == "complete",
        )
        domains[domain] = {
            "runs": {"e53_grpo": control, "e57": treatment},
            "cold_start_gate": cold_start,
            "semantic_gate": semantic,
            "behavioral_gate": BASE.behavioral_gate(control, treatment),
            "safety_gate": E56._safety_gate(control, treatment),
        }
    violations = list(binding_violations)
    for domain, payload in domains.items():
        for arm, run in payload["runs"].items():
            violations.extend(
                f"{domain}/{arm}: {item}" for item in run["violations"]
            )
        for gate in ("cold_start_gate", "semantic_gate"):
            violations.extend(
                f"{domain}/{gate}: {item}"
                for item in payload[gate]["violations"]
            )
    gate_states = [
        payload[key]["status"]
        for payload in domains.values()
        for key in (
            "behavioral_gate",
            "safety_gate",
            "semantic_gate",
            "cold_start_gate",
        )
    ]
    all_complete = all(
        run["status"] == "complete"
        for payload in domains.values()
        for run in payload["runs"].values()
    )
    status = (
        "fail"
        if violations or "fail" in gate_states
        else "pass"
        if all_complete and all(item == "pass" for item in gate_states)
        else "in_progress"
    )
    return {
        "schema": "e57_verified_first_split_sentinel_audit_v1",
        "status": status,
        "authorizes_stage_a": status == "pass",
        "approval_binding": binding,
        "violations": violations,
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
        default=ROOT / "var/artifacts/e57_sentinel_audit_latest.json",
    )
    args = parser.parse_args()
    try:
        payload = audit(args.data_root)
    except Exception as error:  # pragma: no cover - fail-closed CLI
        payload = {
            "schema": "e57_verified_first_split_sentinel_audit_v1",
            "status": "fail",
            "authorizes_stage_a": False,
            "violations": [f"uncaught audit failure: {error}"],
        }
    _atomic_write(args.out, payload)
    if payload["status"] == "pass":
        _atomic_write(APPROVAL_PATH, payload)
    print(
        f"[e57-sentinel-audit] status={payload['status']} "
        f"violations={len(payload.get('violations', []))} out={args.out}"
    )
    return 1 if payload["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
