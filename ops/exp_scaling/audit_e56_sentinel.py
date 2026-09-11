#!/usr/bin/env python3
"""Fail-closed audit for E56's three-domain open-set split sentinel."""

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
E55_PATH = ROOT / "ops/exp_scaling/audit_e55_sentinel.py"
SPEC = importlib.util.spec_from_file_location("e56_bound_e55_audit", E55_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import sentinel helpers from {E55_PATH}")
E55 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E55)
BASE = E55.BASE
E53_CHECKPOINT_GATE = BASE.checkpoint_gate

ARM = "open_set_split_canonical"
SEED = 9010
NUM_SAMPLES = 16
CAPACITY = 16
DIRECT_BASE = 0.000075
SEMANTIC_BASE = 0.10
MASS_BASE = 0.10
BALANCE_BASE = 0.10
WARMUP = 64
EMA_DECAY = 0.90
ESTIMATOR_SCALE = 15.0 / 16.0
OBJECTIVE_SCALE = 1.0 / 16.0
SEMANTIC_PREFIX = "train/semantic_shannon_success_conditioned_signed_"
DOMAINS = {
    "countdown": {
        "pool": 384,
        "control_prefix": "cde53_verified_replay_05b_50ep_sentinel_allcs",
        "treatment_prefix": (
            "cde56_open_set_split_canonical_05b_50ep_sentinel_allcs"
        ),
    },
    "graph_coloring": {
        "pool": 192,
        "control_prefix": "gce53_verified_replay_05b_50ep_sentinel",
        "treatment_prefix": (
            "gce56_open_set_split_canonical_05b_50ep_sentinel"
        ),
    },
    "python_factor": {
        "pool": 384,
        "control_prefix": "pye53_verified_replay_05b_50ep_sentinel_allcs",
        "treatment_prefix": (
            "pye56_open_set_split_canonical_05b_50ep_sentinel_allcs"
        ),
    },
}
IDENTITY_PATH = (
    ROOT / "var/artifacts/e56_open_set_split_canonical_05b_sentinel_identity.json"
)
PROTOCOL_PATH = (
    ROOT / "paper/preregistration/e56_open_set_split_controller_05b.md"
)
LAUNCHER_PATH = (
    ROOT / "ops/exp_scaling/launch_e56_open_set_split_smoke.sh"
)
SMOKE_AUDIT_PATH = ROOT / "var/artifacts/e56_smoke_audit_latest.json"
E53_IDENTITY_PATH = (
    ROOT / "var/artifacts/e53_verified_replay_05b_sentinel_identity.json"
)
APPROVAL_PATH = ROOT / "var/artifacts/e56_sentinel_stage_a_approval.json"
TREATMENT_MANIFESTS = {
    domain: ROOT / f"var/artifacts/{config['treatment_prefix']}_comparative_jobs.tsv"
    for domain, config in DOMAINS.items()
}
CONTROL_MANIFESTS = {
    domain: ROOT / f"var/artifacts/{config['control_prefix']}_comparative_jobs.tsv"
    for domain, config in DOMAINS.items()
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tree(root: Path) -> str:
    lines = []
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        lines.append(
            f"{_sha256_file(path)}  ./{path.relative_to(root).as_posix()}\n"
        )
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def _manifest_job(
    path: Path,
    *,
    arm: str,
    run_stamp: str,
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
    except (FileNotFoundError, json.JSONDecodeError) as error:
        return {}, [f"E56 identity is unavailable or invalid: {error}"]
    if (
        not isinstance(identity, dict)
        or identity.get("schema")
        != "e56_open_set_split_canonical_05b_sentinel_v1"
    ):
        return {}, ["E56 identity has an incompatible schema"]
    source_hash = str(identity.get("source_hash", ""))
    execution_hash = str(identity.get("execution_surface_hash", ""))
    source_root = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e56_open_set_split_{source_hash}"
        / "src"
    )
    ops_root = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e56_open_set_split_ops_{execution_hash}"
        / "ops"
    )
    try:
        treatment_manifest_hashes = {
            domain: _sha256_file(path)
            for domain, path in TREATMENT_MANIFESTS.items()
        }
        control_manifest_hashes = {
            domain: _sha256_file(path)
            for domain, path in CONTROL_MANIFESTS.items()
        }
        observed = {
            "identity_sha256": _sha256_file(IDENTITY_PATH),
            "protocol_sha256": _sha256_file(PROTOCOL_PATH),
            "launcher_sha256": _sha256_file(LAUNCHER_PATH),
            "auditor_sha256": _sha256_file(Path(__file__).resolve()),
            "source_hash": _hash_tree(source_root),
            "execution_surface_hash": _hash_tree(ops_root),
            "e53_control_identity_sha256": _sha256_file(E53_IDENTITY_PATH),
            "smoke_audit_sha256": _sha256_file(SMOKE_AUDIT_PATH),
            "job_manifest_sha256": treatment_manifest_hashes,
            "e53_control_manifest_sha256": control_manifest_hashes,
        }
    except OSError as error:
        return {}, [f"E56 approval evidence is unavailable: {error}"]
    violations = [
        (
            f"approval binding mismatch for {key}: "
            f"expected={identity.get(key)!r} observed={observed[key]!r}"
        )
        for key in (
            "protocol_sha256",
            "launcher_sha256",
            "auditor_sha256",
            "source_hash",
            "execution_surface_hash",
            "e53_control_identity_sha256",
            "smoke_audit_sha256",
            "job_manifest_sha256",
            "e53_control_manifest_sha256",
        )
        if observed[key] != identity.get(key)
    ]
    treatment_jobs: dict[str, int] = {}
    control_jobs: dict[str, int] = {}
    for domain, config in DOMAINS.items():
        treatment_stamp = (
            f"{config['treatment_prefix']}_{ARM}_s{SEED}"
        )
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


def _load_bound_records(
    data_root: Path,
    *,
    run_stamp: str,
    job_id: int | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    if job_id is None:
        return [], [f"{run_stamp}: bound job ID is unavailable"]
    run_roots = sorted(data_root.glob(f"*_{run_stamp}"))
    expected_attempt = f"debug_job{job_id}"
    attempt_dirs = sorted(
        path
        for root in run_roots
        for path in root.glob("debug_*")
        if path.is_dir()
    )
    unexpected = [
        str(path)
        for path in attempt_dirs
        if path.name != expected_attempt
    ]
    violations = (
        [f"{run_stamp}: unexpected debug attempts {unexpected}"]
        if unexpected
        else []
    )
    metric_paths = [
        root / expected_attempt / "train_metrics.jsonl"
        for root in run_roots
        if (root / expected_attempt / "train_metrics.jsonl").is_file()
    ]
    if len(metric_paths) > 1:
        violations.append(
            f"{run_stamp}: found {len(metric_paths)} bound metric streams"
        )
    records: list[dict[str, Any]] = []
    for path in metric_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records, violations


def _finite(record: dict[str, Any], key: str) -> float | None:
    return BASE._finite(record, key)


def _controller_update(
    *,
    observations: float,
    used: float,
    next_value: float,
    multiplier: float,
    ema: float | None,
    reference: float | None,
    base: float,
    inverse: bool,
    label: str,
    step: int,
) -> list[str]:
    violations: list[str] = []
    if used <= 0 or next_value <= 0:
        violations.append(f"step {step}: {label} coefficient is nonpositive")
    if observations <= WARMUP:
        if not math.isclose(
            multiplier, 1.0, rel_tol=1e-6, abs_tol=1e-8
        ) or not math.isclose(
            next_value, base, rel_tol=1e-6, abs_tol=1e-8
        ):
            violations.append(f"step {step}: {label} drifted during warmup")
        return violations
    if (
        ema is None
        or reference is None
        or ema <= 0
        or reference <= 0
    ):
        violations.append(f"step {step}: {label} lacks positive state")
        return violations
    expected_multiplier = reference / ema if inverse else ema / reference
    if not math.isclose(
        multiplier, expected_multiplier, rel_tol=2e-5, abs_tol=1e-8
    ) or not math.isclose(
        next_value,
        base * expected_multiplier,
        rel_tol=2e-5,
        abs_tol=1e-8,
    ):
        violations.append(f"step {step}: {label} recurrence mismatch")
    return violations


def _check_split_replay(
    record: dict[str, Any],
    *,
    step: int,
    reference: Any,
) -> tuple[Any, list[str], dict[str, float], bool]:
    violations: list[str] = []
    groups = _finite(record, "train/canonical_replay_available_groups")
    modes = _finite(record, "train/canonical_replay_available_modes")
    capacity = _finite(record, "train/canonical_replay_capacity")
    projection = _finite(
        record, "train/canonical_replay_alpha_projection_active"
    )
    gold = _finite(record, "train/canonical_replay_gold_support_feedback")
    if groups is None or modes is None:
        return reference, [f"step {step}: missing replay availability"], {}, False
    if capacity != CAPACITY:
        violations.append(f"step {step}: replay capacity mismatch")
    if projection != 0 or gold != 0:
        violations.append(f"step {step}: replay projection/gold feedback active")
    if groups <= 0:
        if modes != 0:
            violations.append(f"step {step}: idle replay retained modes")
        for key in (
            "train/canonical_replay_observation_skipped",
            "train/canonical_replay_mass_observation_skipped",
        ):
            if _finite(record, key) != 1:
                violations.append(f"step {step}: idle replay advanced {key}")
        return reference, violations, {}, False

    required_keys = (
        "train/canonical_replay_actuator_loss",
        "train/canonical_replay_balance_loss",
        "train/canonical_replay_weighted_loss",
        "train/canonical_replay_objective_scale",
        "train/canonical_replay_reward_estimator_scale",
        "train/canonical_replay_mass_score_gradient_sum",
        "train/canonical_replay_balance_score_gradient_sum",
        "train/canonical_replay_mass_score_gradient_l2",
        "train/canonical_replay_balance_score_gradient_l2",
        "train/canonical_replay_applied_score_gradient_l2",
        "train/canonical_replay_applied_score_gradient_sum",
        "train/canonical_replay_mass_alpha_used",
        "train/canonical_replay_balance_alpha_used",
        "train/canonical_replay_mass_inverse_multiplier",
        "train/canonical_replay_mass_next_alpha",
        "train/canonical_replay_mass_observations",
        "train/canonical_replay_mass_observed_surprisal",
        "train/canonical_replay_mass_surprisal_ema",
        "train/canonical_replay_mass_projection_active",
        "train/canonical_replay_eligible_groups",
        "train/canonical_replay_inverse_multiplier",
        "train/canonical_replay_next_alpha",
        "train/canonical_replay_observations",
        "train/canonical_replay_projection_active",
    )
    raw = {key: _finite(record, key) for key in required_keys}
    missing = [key for key, value in raw.items() if value is None]
    if missing:
        return (
            reference,
            violations + [f"step {step}: missing split replay {missing}"],
            {},
            True,
        )
    value = {key: float(item) for key, item in raw.items()}
    mass_loss = value["train/canonical_replay_actuator_loss"]
    balance_loss = value["train/canonical_replay_balance_loss"]
    mass_alpha = value["train/canonical_replay_mass_alpha_used"]
    balance_alpha = value["train/canonical_replay_balance_alpha_used"]
    expected_weighted = ESTIMATOR_SCALE * OBJECTIVE_SCALE * (
        mass_alpha * mass_loss + balance_alpha * balance_loss
    )
    checks = (
        (
            "mass raw score-gradient sum",
            value["train/canonical_replay_mass_score_gradient_sum"],
            -1.0,
            2e-6,
        ),
        (
            "balance raw score-gradient sum",
            value["train/canonical_replay_balance_score_gradient_sum"],
            0.0,
            2e-6,
        ),
        (
            "one-pseudo-rollout measure",
            value["train/canonical_replay_objective_scale"],
            OBJECTIVE_SCALE,
            1e-8,
        ),
        (
            "Dr.GRPO estimator scale",
            value["train/canonical_replay_reward_estimator_scale"],
            ESTIMATOR_SCALE,
            1e-8,
        ),
        (
            "combined weighted loss",
            value["train/canonical_replay_weighted_loss"],
            expected_weighted,
            2e-6,
        ),
        (
            "combined applied gradient sum",
            value["train/canonical_replay_applied_score_gradient_sum"],
            -OBJECTIVE_SCALE * mass_alpha,
            2e-6,
        ),
    )
    for label, actual, expected, tolerance in checks:
        if not math.isclose(actual, expected, abs_tol=tolerance, rel_tol=2e-5):
            violations.append(
                f"step {step}: {label}={actual}, expected {expected}"
            )
    if (
        value["train/canonical_replay_mass_score_gradient_l2"] <= 0
        or value["train/canonical_replay_applied_score_gradient_l2"] <= 0
        or value["train/canonical_replay_mass_projection_active"] != 0
        or value["train/canonical_replay_projection_active"] != 0
    ):
        violations.append(f"step {step}: invalid mass/projection telemetry")
    if not math.isclose(
        value["train/canonical_replay_mass_observed_surprisal"],
        mass_loss,
        rel_tol=1e-6,
        abs_tol=1e-8,
    ):
        violations.append(f"step {step}: verified-mass sensor mismatch")
    mass_reference = _finite(
        record, "train/canonical_replay_mass_surprisal_reference"
    )
    violations.extend(
        _controller_update(
            observations=value["train/canonical_replay_mass_observations"],
            used=mass_alpha,
            next_value=value["train/canonical_replay_mass_next_alpha"],
            multiplier=value["train/canonical_replay_mass_inverse_multiplier"],
            ema=value["train/canonical_replay_mass_surprisal_ema"],
            reference=mass_reference,
            base=MASS_BASE,
            inverse=False,
            label="verified-mass controller",
            step=step,
        )
    )

    eligible_groups = value["train/canonical_replay_eligible_groups"]
    if eligible_groups <= 0:
        if (
            modes != groups
            or value["train/canonical_replay_balance_score_gradient_l2"] != 0
            or _finite(record, "train/canonical_replay_observation_skipped") != 1
        ):
            violations.append(f"step {step}: singleton balance was not idle")
    else:
        observed = _finite(
            record, "train/canonical_replay_observed_normalized_entropy"
        )
        entropy_ema = _finite(record, "train/canonical_replay_entropy_ema")
        balance_reference = _finite(
            record, "train/canonical_replay_reference_entropy"
        )
        sensor = _finite(
            record, "train/canonical_replay_normalized_model_entropy"
        )
        if (
            observed is None
            or sensor is None
            or not 0 < observed <= 1 + 1e-6
            or not math.isclose(
                observed, sensor, rel_tol=1e-6, abs_tol=1e-8
            )
            or _finite(record, "train/canonical_replay_observation_skipped")
            != 0
        ):
            violations.append(f"step {step}: known-mode balance sensor mismatch")
        violations.extend(
            _controller_update(
                observations=value["train/canonical_replay_observations"],
                used=balance_alpha,
                next_value=value["train/canonical_replay_next_alpha"],
                multiplier=value["train/canonical_replay_inverse_multiplier"],
                ema=entropy_ema,
                reference=balance_reference,
                base=BALANCE_BASE,
                inverse=True,
                label="known-mode balance controller",
                step=step,
            )
        )
    return reference, violations, {
        "mass_loss": mass_loss,
        "balance_loss": balance_loss,
        "mass_alpha": mass_alpha,
        "balance_alpha": balance_alpha,
        "available_groups": groups,
        "available_modes": modes,
    }, True


def _semantic_audit(
    records: list[dict[str, Any]], *, complete: bool
) -> dict[str, Any]:
    rows = BASE._latest_by_step(
        records,
        f"{SEMANTIC_PREFIX}open_set_inverse_adaptation_active",
    )
    violations: list[str] = []
    positive_steps = 0
    negative_steps = 0
    observation_events = 0
    previous_observations = 0.0
    for row in rows:
        step = int(_finite(row, "trainer/global_step") or -1)
        required_keys = (
            f"{SEMANTIC_PREFIX}open_set_inverse_adaptation_active",
            f"{SEMANTIC_PREFIX}open_set_coefficient_used",
            f"{SEMANTIC_PREFIX}open_set_entropy_ema",
            f"{SEMANTIC_PREFIX}open_set_inverse_multiplier",
            f"{SEMANTIC_PREFIX}open_set_next_coefficient",
            f"{SEMANTIC_PREFIX}open_set_observations",
            f"{SEMANTIC_PREFIX}open_set_projection_active",
            f"{SEMANTIC_PREFIX}open_set_observation_skipped",
            f"{SEMANTIC_PREFIX}advantage_cap",
            f"{SEMANTIC_PREFIX}effective_advantage_positive_fraction",
            f"{SEMANTIC_PREFIX}effective_advantage_negative_fraction",
        )
        raw = {key: _finite(row, key) for key in required_keys}
        missing = [key for key, value in raw.items() if value is None]
        if missing:
            violations.append(f"step {step}: missing open-set state {missing}")
            continue
        value = {key: float(item) for key, item in raw.items()}
        observations = value[f"{SEMANTIC_PREFIX}open_set_observations"]
        positive_steps += int(
            value[
                f"{SEMANTIC_PREFIX}effective_advantage_positive_fraction"
            ]
            > 0
        )
        negative_steps += int(
            value[
                f"{SEMANTIC_PREFIX}effective_advantage_negative_fraction"
            ]
            > 0
        )
        observation_events += int(observations > previous_observations)
        previous_observations = observations
        if (
            value[
                f"{SEMANTIC_PREFIX}open_set_inverse_adaptation_active"
            ]
            != 1
            or value[f"{SEMANTIC_PREFIX}open_set_projection_active"] != 0
            or value[f"{SEMANTIC_PREFIX}advantage_cap"] != 0
        ):
            violations.append(f"step {step}: semantic path/cap/projection mismatch")
        reference = _finite(
            row, f"{SEMANTIC_PREFIX}open_set_reference_entropy"
        )
        violations.extend(
            _controller_update(
                observations=observations,
                used=value[f"{SEMANTIC_PREFIX}open_set_coefficient_used"],
                next_value=value[
                    f"{SEMANTIC_PREFIX}open_set_next_coefficient"
                ],
                multiplier=value[
                    f"{SEMANTIC_PREFIX}open_set_inverse_multiplier"
                ],
                ema=value[f"{SEMANTIC_PREFIX}open_set_entropy_ema"],
                reference=reference,
                base=SEMANTIC_BASE,
                inverse=True,
                label="open-set semantic controller",
                step=step,
            )
        )
    if complete:
        if observation_events == 0:
            violations.append("terminal run never observed open-set entropy")
        if positive_steps == 0:
            violations.append("terminal run never rewarded a new valid mode")
        if negative_steps == 0:
            violations.append("terminal run never penalized a common valid mode")
    return {
        "status": "fail" if violations else "pass" if complete else "running",
        "train_points": len(rows),
        "observation_events": observation_events,
        "positive_pressure_steps": positive_steps,
        "negative_pressure_steps": negative_steps,
        "violations": violations,
    }


def _checkpoint_gate(
    data_root: Path,
    *,
    run_stamp: str,
    arm: str,
    terminal_step: int,
    job_id: int | None = None,
) -> dict[str, Any]:
    if arm == "grpo":
        return E53_CHECKPOINT_GATE(
            data_root,
            run_stamp=run_stamp,
            arm=arm,
            terminal_step=terminal_step,
        )
    expected_tag = f"step_{terminal_step:05d}"
    attempt_glob = f"debug_job{job_id}" if job_id is not None else "debug_*"
    candidates = sorted(
        data_root.glob(
            f"*_{run_stamp}/{attempt_glob}/checkpoints/{expected_tag}/"
            "mp_rank_00_model_states.pt"
        )
    )
    if not candidates:
        return {
            "status": "fail",
            "violations": [f"missing terminal checkpoint {expected_tag}"],
        }
    path = candidates[-1]
    try:
        import torch

        state = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except Exception as error:  # pragma: no cover
        return {
            "status": "fail",
            "path": str(path),
            "violations": [f"cannot inspect terminal checkpoint: {error}"],
        }
    violations: list[str] = []
    expected_states = {
        "maxent_alpha_controller_state": (
            "controller_kind",
            "maxent_inverse",
            "controller_rule",
            "unprojected_warmup_inverse_direct_entropy_v1",
            "base_alpha",
            DIRECT_BASE,
        ),
        "canonical_replay_controller_state": (
            "controller_kind",
            "canonical_replay_inverse",
            "controller_rule",
            "unprojected_warmup_inverse_observed_bank_entropy_v1",
            "base_alpha",
            BALANCE_BASE,
        ),
        "canonical_replay_mass_controller_state": (
            "controller_kind",
            "canonical_replay_likelihood",
            "controller_rule",
            "unprojected_warmup_verified_surprisal_ratio_v1",
            "base_alpha",
            MASS_BASE,
        ),
    }
    for state_key, expected in expected_states.items():
        payload = state.get(state_key)
        if not isinstance(payload, dict):
            violations.append(f"checkpoint lacks {state_key}")
            continue
        for index in range(0, len(expected), 2):
            key, wanted = expected[index], expected[index + 1]
            if payload.get(key) != wanted:
                violations.append(
                    f"{state_key} {key}={payload.get(key)!r}, "
                    f"expected {wanted!r}"
                )
        if (
            payload.get("warmup_steps") != WARMUP
            or payload.get("ema_decay") != EMA_DECAY
        ):
            violations.append(f"{state_key} warmup/EMA mismatch")
    semantic = state.get("semantic_shannon_tracker_state")
    if (
        not isinstance(semantic, dict)
        or semantic.get("schema")
        != "semantic_shannon_tracker_v4_open_set_inverse"
        or semantic.get("open_set_inverse_adaptation") is not True
        or not isinstance(semantic.get("open_set_controller"), dict)
    ):
        violations.append("checkpoint lacks the E56 open-set semantic tracker")
    else:
        controller = semantic["open_set_controller"]
        if (
            controller.get("controller_kind") != "semantic_open_set_inverse"
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
BASE._check_replay_controller = _check_split_replay
BASE.checkpoint_gate = _checkpoint_gate


def _safety_gate(
    control: dict[str, Any],
    treatment: dict[str, Any],
) -> dict[str, Any]:
    """Apply generation safety only to the frozen terminal window."""
    if treatment["safety_window"]["points"] == 0:
        return {"status": "pending"}
    result = E55._safety_gate(control, treatment)
    if treatment.get("status") != "complete":
        return {
            **result,
            "status": "pending",
            "provisional_status": result["status"],
        }
    return result


def audit(data_root: Path) -> dict[str, Any]:
    binding, binding_violations = approval_binding()
    treatment_jobs = binding.get("treatment_jobs", {})
    control_jobs = binding.get("control_jobs", {})
    domains: dict[str, Any] = {}
    for domain, config in DOMAINS.items():
        pool = int(config["pool"])
        control_stamp = f"{config['control_prefix']}_grpo_s{SEED}"
        treatment_stamp = (
            f"{config['treatment_prefix']}_{ARM}_s{SEED}"
        )
        control_records, control_attempt_violations = _load_bound_records(
            data_root,
            run_stamp=control_stamp,
            job_id=control_jobs.get(domain),
        )
        treatment_records, treatment_attempt_violations = (
            _load_bound_records(
                data_root,
                run_stamp=treatment_stamp,
                job_id=treatment_jobs.get(domain),
            )
        )
        control = BASE.audit_run(
            control_records,
            arm="grpo",
            prompt_pool_size=pool,
        )
        treatment = BASE.audit_run(
            treatment_records,
            arm=ARM,
            prompt_pool_size=pool,
        )
        control["violations"].extend(control_attempt_violations)
        treatment["violations"].extend(treatment_attempt_violations)
        for run, stamp, run_arm, job_id in (
            (
                control,
                control_stamp,
                "grpo",
                control_jobs.get(domain),
            ),
            (
                treatment,
                treatment_stamp,
                ARM,
                treatment_jobs.get(domain),
            ),
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
        semantic = _semantic_audit(
            treatment_records,
            complete=treatment["status"] == "complete",
        )
        domains[domain] = {
            "runs": {"e53_grpo": control, "e56": treatment},
            "semantic_gate": semantic,
            "behavioral_gate": BASE.behavioral_gate(control, treatment),
            "safety_gate": _safety_gate(control, treatment),
        }
    violations = list(binding_violations) + [
        f"{domain}/{arm}: {violation}"
        for domain, payload in domains.items()
        for arm, run in payload["runs"].items()
        for violation in run["violations"]
    ] + [
        f"{domain}/semantic: {violation}"
        for domain, payload in domains.items()
        for violation in payload["semantic_gate"]["violations"]
    ]
    gate_states = [
        payload[key]["status"]
        for payload in domains.values()
        for key in ("behavioral_gate", "safety_gate", "semantic_gate")
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
        "schema": "e56_open_set_split_sentinel_audit_v1",
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
        default=ROOT / "var/artifacts/e56_sentinel_audit_latest.json",
    )
    parser.add_argument("--approval-out", type=Path, default=APPROVAL_PATH)
    args = parser.parse_args()
    payload = audit(args.data_root)
    _atomic_write(args.out, payload)
    if payload["status"] == "pass":
        _atomic_write(args.approval_out, payload)
    else:
        args.approval_out.unlink(missing_ok=True)
    print(
        f"[e56-audit] status={payload['status']} "
        f"violations={len(payload['violations'])} out={args.out}"
    )
    for domain, item in payload["domains"].items():
        control = item["runs"]["e53_grpo"]
        treatment = item["runs"]["e56"]
        print(
            f"[e56-audit] {domain}: "
            f"control={control['status']}@{control['training_passes']:.2f}, "
            f"e56={treatment['status']}@{treatment['training_passes']:.2f}; "
            f"behavior={item['behavioral_gate']['status']} "
            f"semantic={item['semantic_gate']['status']}"
        )
    return 1 if payload["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
