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
import re
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "ops/exp_scaling/audit_e56_sentinel.py"
SPEC = importlib.util.spec_from_file_location("e58_bound_e56_audit", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import sentinel helpers from {HELPER_PATH}")
E56 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E56)
BASE = E56.BASE

ARM = "verified_first_global_replay_canonical"
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
    ROOT
    / "paper/preregistration/e58_global_verified_replay_canonical_05b.md"
)
LAUNCHER_PATH = (
    ROOT / "ops/exp_scaling/launch_e58_global_replay_sentinel.sh"
)
SMOKE_IDENTITY = (
    ROOT
    / "var/artifacts/e58_global_verified_replay_python_smoke_attempt3_identity.json"
)
SMOKE_AUDIT = (
    ROOT / "var/artifacts/e58_global_replay_smoke_audit_latest.json"
)
E53_IDENTITY = (
    ROOT / "var/artifacts/e53_verified_replay_05b_sentinel_identity.json"
)
APPROVAL_PATH = (
    ROOT / "var/artifacts/e58_global_replay_sentinel_replication_approval.json"
)
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
    if (
        identity.get("arm") != ARM
        or identity.get("seed") != SEED
        or identity.get("num_samples") != 16
        or identity.get("direct_token_entropy")
        != {"coefficient": 0.0, "controller": None}
        or identity.get("global_replay")
        != {
            "groups_per_step": 1,
            "selection": "persistent_prompt_hash_round_robin",
            "capacity": CAPACITY,
        }
        or identity.get("information_firewall")
        != {
            "gold_support_feedback": False,
            "evaluation_feedback": False,
            "desired_entropy": None,
            "desired_mode_count": None,
        }
    ):
        violations.append("E58 sentinel mechanism identity mismatch")
    try:
        smoke_identity = json.loads(SMOKE_IDENTITY.read_text(encoding="utf-8"))
        smoke_audit = json.loads(SMOKE_AUDIT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        violations.append(f"E58 smoke approval is unavailable: {error}")
    else:
        if (
            smoke_identity.get("attempt") != 3
            or smoke_identity.get("arm") != ARM
            or smoke_identity.get("source_hash") != source_hash
            or smoke_identity.get("execution_surface_hash") != execution_hash
            or smoke_audit.get("status") != "pass"
            or smoke_audit.get("violations")
            or Path(str(smoke_audit.get("identity", ""))).resolve()
            != SMOKE_IDENTITY.resolve()
        ):
            violations.append("E58 sentinel lacks a clean bound terminal smoke")
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
    cursor = bank.get("global_replay_cursor") if isinstance(bank, dict) else None
    if (
        not isinstance(bank, dict)
        or bank.get("retain_exemplars") is not True
        or bank.get("replay_capacity") != CAPACITY
        or bank.get("global_replay_groups_per_step") != 1
        or isinstance(cursor, bool)
        or not isinstance(cursor, int)
        or cursor < 0
    ):
        violations.append(
            "checkpoint verified bank/global scheduler is incompatible"
        )
    return {
        "status": "pass" if not violations else "fail",
        "path": str(path),
        "tag": expected_tag,
        "scheduler_cursor": cursor,
        "violations": violations,
    }


BASE.REPLAY_ARM = ARM
BASE._check_direct_controller = _check_no_direct_controller
BASE._check_replay_controller = E56._check_split_replay
BASE.checkpoint_gate = _checkpoint_gate


def _scheduler_and_cold_start_gate(
    records: list[dict[str, Any]], *, complete: bool
) -> dict[str, Any]:
    rows = BASE._latest_by_step(
        records, "train/canonical_replay_available_groups"
    )
    violations: list[str] = []
    first_discovery: int | None = None
    pre_discovery_points = 0
    post_discovery_points = 0
    replay_activations = 0
    previous_mass_observations = 0
    for row in rows:
        step = int(_finite(row, "trainer/global_step") or -1)
        direct = sorted(key for key in row if key.startswith("train/maxent_"))
        if direct:
            violations.append(f"step {step}: direct MaxEnt telemetry present")
        tracked = _finite(row, "train/online_canonical_tracked_outcomes")
        groups = _finite(row, "train/canonical_replay_available_groups")
        mass_observations = _finite(
            row, "train/canonical_replay_mass_observations"
        )
        if tracked is not None and tracked > 0 and first_discovery is None:
            first_discovery = step
        if first_discovery is None:
            pre_discovery_points += 1
            expected_zero = (
                "train/online_canonical_task_reward_mean",
                "train/semantic_shannon_augmented_reward_mean",
                "train/policy_grad_norm",
                "train/canonical_replay_available_groups",
                "train/canonical_replay_mass_observations",
            )
            for key in expected_zero:
                value = _finite(row, key)
                if value is None or not math.isclose(
                    value, 0.0, rel_tol=0.0, abs_tol=1e-12
                ):
                    violations.append(
                        f"step {step}: pre-discovery {key}={value!r}, "
                        "expected zero"
                    )
        else:
            post_discovery_points += 1
            replay_activations += int(groups == 1)
            expected_one = (
                "train/canonical_replay_global_scheduler_active",
                "train/canonical_replay_global_groups_per_step",
                "train/canonical_replay_available_groups",
                "train/canonical_replay_actuator_groups",
            )
            for key in expected_one:
                value = _finite(row, key)
                if value is None or not math.isclose(
                    value, 1.0, rel_tol=0.0, abs_tol=1e-12
                ):
                    violations.append(
                        f"step {step}: post-discovery {key}={value!r}, "
                        "expected one"
                    )
            modes = _finite(row, "train/canonical_replay_available_modes")
            if modes is None or modes < 1:
                violations.append(
                    f"step {step}: global replay lacks a verified mode"
                )
            if (
                mass_observations is None
                or not math.isclose(
                    mass_observations,
                    previous_mass_observations + 1,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                violations.append(
                    f"step {step}: mass observations did not advance once"
                )
        if mass_observations is not None:
            previous_mass_observations = int(mass_observations)
    if complete and first_discovery is None:
        violations.append("terminal run never made a verifier-positive discovery")
    if complete and replay_activations != post_discovery_points:
        violations.append(
            "global replay was not active on every post-discovery update"
        )
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
        "pre_discovery_points": pre_discovery_points,
        "first_discovery_step": first_discovery,
        "post_discovery_points": post_discovery_points,
        "replay_activations": replay_activations,
        "direct_token_entropy": "absent" if not violations else "violation",
        "violations": violations,
    }


_CRASH_SIGNATURES = {
    "python_traceback": re.compile(r"Traceback \(most recent call last\)"),
    "cuda_oom": re.compile(
        r"CUDA out of memory|torch\.OutOfMemoryError",
        re.IGNORECASE,
    ),
    "distributed_worker_failure": re.compile(
        r"ChildFailedError|RayActorError|worker unexpectedly died",
        re.IGNORECASE,
    ),
    "floating_point_failure": re.compile(
        r"FloatingPointError|RuntimeError:[^\n]*non-finite",
        re.IGNORECASE,
    ),
    "segmentation_fault": re.compile(r"segmentation fault", re.IGNORECASE),
}


def _runtime_log_gate(job_id: int | None, *, complete: bool) -> dict[str, Any]:
    """Reject a terminal E58 run if its exact Slurm logs contain a crash."""

    if job_id is None:
        return {
            "status": "fail",
            "violations": ["treatment job id is unavailable"],
        }
    paths = [
        ROOT / f"var/artifacts/logs/xdr_train-{job_id}.out",
        ROOT / f"var/artifacts/logs/xdr_train-{job_id}.err",
    ]
    readable = [path for path in paths if path.is_file()]
    if complete and not readable:
        return {
            "status": "fail",
            "job_id": job_id,
            "paths": [str(path) for path in paths],
            "violations": ["terminal treatment has no exact Slurm logs"],
        }
    matches: dict[str, list[str]] = {}
    violations: list[str] = []
    for path in readable:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            violations.append(f"cannot inspect {path}: {error}")
            continue
        found = [
            name for name, pattern in _CRASH_SIGNATURES.items()
            if pattern.search(text)
        ]
        if found:
            matches[str(path)] = found
            violations.append(
                f"{path.name} contains crash signatures: {found}"
            )
    return {
        "status": (
            "fail"
            if violations
            else "pass"
            if complete
            else "running"
            if readable
            else "pending"
        ),
        "job_id": job_id,
        "paths": [str(path) for path in paths],
        "matches": matches,
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
        scheduler = _scheduler_and_cold_start_gate(
            treatment_records,
            complete=treatment["status"] == "complete",
        )
        runtime = _runtime_log_gate(
            treatment_jobs.get(domain),
            complete=treatment["status"] == "complete",
        )
        domains[domain] = {
            "runs": {"e53_grpo": control, "e58": treatment},
            "scheduler_and_cold_start_gate": scheduler,
            "semantic_gate": semantic,
            "runtime_log_gate": runtime,
            "behavioral_gate": BASE.behavioral_gate(control, treatment),
            "safety_gate": E56._safety_gate(control, treatment),
        }
    violations = list(binding_violations)
    for domain, payload in domains.items():
        for arm, run in payload["runs"].items():
            violations.extend(
                f"{domain}/{arm}: {item}" for item in run["violations"]
            )
        for gate in (
            "scheduler_and_cold_start_gate",
            "semantic_gate",
            "runtime_log_gate",
        ):
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
            "scheduler_and_cold_start_gate",
            "runtime_log_gate",
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
        "schema": "e58_global_verified_replay_sentinel_audit_v1",
        "status": status,
        "authorizes_replication": status == "pass",
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
        default=(
            ROOT
            / "var/artifacts/e58_global_replay_sentinel_audit_latest.json"
        ),
    )
    args = parser.parse_args()
    try:
        payload = audit(args.data_root)
    except Exception as error:  # pragma: no cover - fail-closed CLI
        payload = {
            "schema": "e58_global_verified_replay_sentinel_audit_v1",
            "status": "fail",
            "authorizes_replication": False,
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
