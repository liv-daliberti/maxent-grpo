#!/usr/bin/env python3
"""Fail-closed audit for E55 per-rollout verified-anchor sentinels."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "ops/exp_scaling/audit_e53_sentinel.py"
SPEC = importlib.util.spec_from_file_location("e55_bound_e53_audit", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import E53 audit helpers from {BASE_PATH}")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

ARM = "maxent_inverse_canonical_replay"
SEED = 9010
NUM_SAMPLES = 16
REPLAY_CAPACITY = 16
REPLAY_BASE_ALPHA = 0.10
WARMUP_STEPS = 64
ESTIMATOR_SCALE = 15.0 / 16.0
OBJECTIVE_SCALE = 1.0 / 16.0
DOMAINS = {
    "countdown": {
        "pool": 384,
        "control_prefix": "cde53_verified_replay_05b_50ep_sentinel_allcs",
        "treatment_prefix": "cde55_per_rollout_verified_anchor_05b_50ep_sentinel_allcs",
    },
    "graph_coloring": {
        "pool": 192,
        "control_prefix": "gce53_verified_replay_05b_50ep_sentinel",
        "treatment_prefix": "gce55_per_rollout_verified_anchor_05b_50ep_sentinel",
    },
    "python_factor": {
        "pool": 384,
        "control_prefix": "pye53_verified_replay_05b_50ep_sentinel_allcs",
        "treatment_prefix": "pye55_per_rollout_verified_anchor_05b_50ep_sentinel_allcs",
    },
}
IDENTITY_PATH = (
    ROOT / "var/artifacts/e55_per_rollout_verified_anchor_identity.json"
)
E53_IDENTITY_PATH = (
    ROOT / "var/artifacts/e53_verified_replay_05b_sentinel_identity.json"
)
PROTOCOL_PATH = (
    ROOT / "paper/preregistration/e55_per_rollout_verified_anchor_05b.md"
)
LAUNCHER_PATH = (
    ROOT
    / "ops/exp_scaling/launch_e55_per_rollout_verified_anchor_05b.sh"
)
APPROVAL_PATH = ROOT / "var/artifacts/e55_sentinel_stage_a_approval.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tree(root: Path) -> str:
    lines = []
    for path in sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        lines.append(f"{_sha256_file(path)}  ./{relative}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def approval_binding() -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    try:
        identity = json.loads(IDENTITY_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        return {}, [f"E55 identity is unavailable or invalid: {error}"]
    if (
        not isinstance(identity, dict)
        or identity.get("schema")
        != "e55_per_rollout_verified_anchor_05b_sentinel_v1"
    ):
        return {}, ["E55 identity has an incompatible schema"]

    source_hash = str(identity.get("source_hash", ""))
    execution_hash = str(identity.get("execution_surface_hash", ""))
    source_root = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e55_per_rollout_verified_anchor_{source_hash}"
        / "src"
    )
    ops_root = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e55_per_rollout_verified_anchor_ops_{execution_hash}"
        / "ops"
    )
    try:
        observed = {
            "identity_sha256": _sha256_file(IDENTITY_PATH),
            "protocol_sha256": _sha256_file(PROTOCOL_PATH),
            "launcher_sha256": _sha256_file(LAUNCHER_PATH),
            "auditor_sha256": _sha256_file(Path(__file__).resolve()),
            "source_hash": _hash_tree(source_root),
            "execution_surface_hash": _hash_tree(ops_root),
            "e53_control_identity_sha256": _sha256_file(E53_IDENTITY_PATH),
        }
    except OSError as error:
        return {}, [f"E55 approval evidence is unavailable: {error}"]
    for key in (
        "protocol_sha256",
        "launcher_sha256",
        "auditor_sha256",
        "source_hash",
        "execution_surface_hash",
        "e53_control_identity_sha256",
    ):
        if observed[key] != identity.get(key):
            violations.append(
                f"approval binding mismatch for {key}: "
                f"expected={identity.get(key)!r} observed={observed[key]!r}"
            )
    return {
        "identity_path": str(IDENTITY_PATH.resolve()),
        "source_snapshot_root": str(source_root.resolve()),
        "execution_snapshot_root": str(ops_root.resolve()),
        **observed,
    }, violations


def _finite(record: dict[str, Any], key: str) -> float | None:
    return BASE._finite(record, key)


def _check_e55_replay(
    record: dict[str, Any],
    *,
    step: int,
    reference: float | None,
) -> tuple[float | None, list[str], dict[str, float], bool]:
    violations: list[str] = []
    groups = _finite(record, "train/canonical_replay_available_groups")
    modes = _finite(record, "train/canonical_replay_available_modes")
    capacity = _finite(record, "train/canonical_replay_capacity")
    projection = _finite(
        record, "train/canonical_replay_alpha_projection_active"
    )
    gold = _finite(record, "train/canonical_replay_gold_support_feedback")
    if groups is None or modes is None:
        return reference, [f"step {step}: missing replay bank availability"], {}, False
    if capacity != REPLAY_CAPACITY:
        violations.append(f"step {step}: replay capacity mismatch")
    if projection != 0 or gold != 0:
        violations.append(f"step {step}: projection/gold feedback is active")
    active = groups > 0
    skipped = _finite(record, "train/canonical_replay_observation_skipped")
    if not active:
        if modes != 0 or skipped != 1:
            violations.append(f"step {step}: invalid idle replay telemetry")
        return reference, violations, {}, False

    keys = (
        "train/canonical_replay_actuator_loss",
        "train/canonical_replay_balance_loss",
        "train/canonical_replay_weighted_loss",
        "train/canonical_replay_backward_scale",
        "train/canonical_replay_chunk_size",
        "train/canonical_replay_score_passes",
        "train/canonical_replay_normalized_model_entropy",
        "train/canonical_replay_cross_entropy_excess",
        "train/canonical_replay_alpha_used",
        "train/canonical_replay_eligible_groups",
        "train/canonical_replay_retained_modes",
        "train/canonical_replay_actuator_groups",
        "train/canonical_replay_actuator_modes",
        "train/canonical_replay_reward_estimator_scale",
        "train/canonical_replay_score_gradient_sum",
        "train/canonical_replay_objective_scale",
        "train/canonical_replay_applied_score_gradient_sum",
        "train/canonical_replay_verified_likelihood_active",
        "train/canonical_replay_next_alpha",
        "train/canonical_replay_observations",
        "train/canonical_replay_projection_active",
    )
    values = {key: _finite(record, key) for key in keys}
    missing = [key for key, value in values.items() if value is None]
    if missing:
        return (
            reference,
            violations + [f"step {step}: missing/nonfinite E55 replay {missing}"],
            {},
            True,
        )
    value = {key: float(item) for key, item in values.items()}
    actuator = value["train/canonical_replay_actuator_loss"]
    balance = value["train/canonical_replay_balance_loss"]
    weighted = value["train/canonical_replay_weighted_loss"]
    alpha_used = value["train/canonical_replay_alpha_used"]
    sensor = value["train/canonical_replay_normalized_model_entropy"]
    alpha_next = value["train/canonical_replay_next_alpha"]
    observations = value["train/canonical_replay_observations"]
    eligible_groups = value["train/canonical_replay_eligible_groups"]
    eligible_modes = value["train/canonical_replay_retained_modes"]
    if (
        value["train/canonical_replay_actuator_groups"] != groups
        or value["train/canonical_replay_actuator_modes"] != modes
    ):
        violations.append(f"step {step}: replay actuator/bank mismatch")
    if (
        eligible_groups < 0
        or eligible_groups > groups
        or eligible_modes < 2 * eligible_groups
        or eligible_modes > modes
    ):
        violations.append(f"step {step}: replay entropy eligibility mismatch")
    if actuator < -1e-7 or balance < -1e-7:
        violations.append(f"step {step}: negative replay loss")
    if not math.isclose(
        balance,
        value["train/canonical_replay_cross_entropy_excess"],
        rel_tol=1e-6,
        abs_tol=1e-7,
    ):
        violations.append(f"step {step}: detached balance diagnostic mismatch")
    if not math.isclose(
        weighted,
        actuator * alpha_used * ESTIMATOR_SCALE * OBJECTIVE_SCALE,
        rel_tol=2e-5,
        abs_tol=1e-8,
    ):
        violations.append(f"step {step}: actuator weighted-loss mismatch")
    if (
        value["train/canonical_replay_backward_scale"] != NUM_SAMPLES
        or value["train/canonical_replay_chunk_size"] != 1
        or value["train/canonical_replay_score_passes"] != 2
        or value["train/canonical_replay_reward_estimator_scale"]
        != ESTIMATOR_SCALE
        or value["train/canonical_replay_objective_scale"]
        != OBJECTIVE_SCALE
    ):
        violations.append(f"step {step}: replay execution scaling mismatch")
    if (
        not math.isclose(
            value["train/canonical_replay_score_gradient_sum"],
            -1.0,
            rel_tol=1e-6,
            abs_tol=1e-7,
        )
        or not math.isclose(
            value["train/canonical_replay_applied_score_gradient_sum"],
            -OBJECTIVE_SCALE,
            rel_tol=1e-6,
            abs_tol=1e-7,
        )
        or value["train/canonical_replay_verified_likelihood_active"] != 1
    ):
        violations.append(f"step {step}: common-mass actuator is not exact")
    if alpha_used <= 0:
        violations.append(f"step {step}: replay applied nonpositive alpha")
    if value["train/canonical_replay_projection_active"] != 0:
        violations.append(f"step {step}: replay projection is active")
    saved_reference = _finite(
        record, "train/canonical_replay_reference_entropy"
    )
    if saved_reference is not None:
        reference = saved_reference
    if eligible_groups == 0:
        if (
            skipped != 1
            or eligible_modes != 0
            or not math.isclose(sensor, 1.0, rel_tol=0.0, abs_tol=1e-8)
            or not math.isclose(
                alpha_next, alpha_used, rel_tol=1e-6, abs_tol=1e-8
            )
        ):
            violations.append(
                f"step {step}: singleton anchor advanced/faked entropy control"
            )
        return reference, violations, {
            "actuator_loss": actuator,
            "balance_loss": balance,
            "alpha_used": alpha_used,
            "alpha_next": alpha_next,
            "available_groups": groups,
            "available_modes": modes,
            "entropy_eligible_groups": 0.0,
            "score_gradient_sum": -1.0,
            "applied_score_gradient_sum": -OBJECTIVE_SCALE,
        }, True

    controller_keys = (
        "train/canonical_replay_observed_normalized_entropy",
        "train/canonical_replay_entropy_ema",
        "train/canonical_replay_inverse_multiplier",
        "train/canonical_replay_alpha_before",
    )
    controller = {key: _finite(record, key) for key in controller_keys}
    missing_controller = [
        key for key, item in controller.items() if item is None
    ]
    if missing_controller:
        return (
            reference,
            violations
            + [
                f"step {step}: missing multi-mode controller "
                f"{missing_controller}"
            ],
            {},
            True,
        )
    controller_value = {
        key: float(item) for key, item in controller.items()
    }
    observed = controller_value[
        "train/canonical_replay_observed_normalized_entropy"
    ]
    ema = controller_value["train/canonical_replay_entropy_ema"]
    multiplier = controller_value[
        "train/canonical_replay_inverse_multiplier"
    ]
    alpha_before = controller_value["train/canonical_replay_alpha_before"]
    if not 0 < sensor <= 1 + 1e-6 or not math.isclose(
        sensor, observed, rel_tol=1e-6, abs_tol=1e-8
    ):
        violations.append(f"step {step}: replay entropy sensor mismatch")
    if not math.isclose(
        alpha_used, alpha_before, rel_tol=1e-6, abs_tol=1e-8
    ):
        violations.append(f"step {step}: replay applied wrong alpha")
    if skipped != 0:
        violations.append(f"step {step}: multi-mode controller was skipped")
    if observations <= WARMUP_STEPS:
        if not math.isclose(
            alpha_next, REPLAY_BASE_ALPHA, rel_tol=1e-6, abs_tol=1e-8
        ) or not math.isclose(multiplier, 1.0, rel_tol=1e-6, abs_tol=1e-8):
            violations.append(f"step {step}: replay warmup coefficient drift")
    elif reference is None or reference <= 0 or ema <= 0:
        violations.append(f"step {step}: replay inverse lacks positive state")
    else:
        expected_multiplier = reference / ema
        if not math.isclose(
            multiplier, expected_multiplier, rel_tol=2e-5, abs_tol=1e-8
        ) or not math.isclose(
            alpha_next,
            REPLAY_BASE_ALPHA * expected_multiplier,
            rel_tol=2e-5,
            abs_tol=1e-8,
        ):
            violations.append(f"step {step}: unbounded replay inverse mismatch")
    return reference, violations, {
        "actuator_loss": actuator,
        "balance_loss": balance,
        "normalized_model_entropy": sensor,
        "entropy_ema": ema,
        "alpha_used": alpha_used,
        "alpha_next": alpha_next,
        "available_groups": groups,
        "available_modes": modes,
        "entropy_eligible_groups": eligible_groups,
        "score_gradient_sum": value[
            "train/canonical_replay_score_gradient_sum"
        ],
        "applied_score_gradient_sum": value[
            "train/canonical_replay_applied_score_gradient_sum"
        ],
    }, True


BASE._check_replay_controller = _check_e55_replay


def _safety_gate(control: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any]:
    control_window = control["safety_window"]
    treatment_window = treatment["safety_window"]
    control_no_eos = control_window["mean_no_eos_count"]
    control_length = control_window["mean_response_length"]
    if (
        control_window["points"] == 0
        or control_no_eos is None
        or control_length is None
    ):
        return {"status": "pending"}
    no_eos_limit = max(1.5 * control_no_eos, control_no_eos + 1.0)
    length_limit = max(1.5 * control_length, control_length + 32.0)
    checks = {
        "no_eos": treatment_window["mean_no_eos_count"] is not None
        and treatment_window["mean_no_eos_count"] <= no_eos_limit,
        "response_length": treatment_window["mean_response_length"] is not None
        and treatment_window["mean_response_length"] <= length_limit,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "no_eos_limit": no_eos_limit,
        "response_length_limit": length_limit,
    }


def audit(data_root: Path) -> dict[str, Any]:
    binding, binding_violations = approval_binding()
    domains: dict[str, Any] = {}
    for domain, config in DOMAINS.items():
        pool = int(config["pool"])
        control_stamp = f"{config['control_prefix']}_grpo_s{SEED}"
        treatment_stamp = f"{config['treatment_prefix']}_{ARM}_s{SEED}"
        control = BASE.audit_run(
            BASE._load_records(data_root, control_stamp),
            arm="grpo",
            prompt_pool_size=pool,
        )
        treatment = BASE.audit_run(
            BASE._load_records(data_root, treatment_stamp),
            arm=ARM,
            prompt_pool_size=pool,
        )
        for run, stamp, arm in (
            (control, control_stamp, "grpo"),
            (treatment, treatment_stamp, ARM),
        ):
            run["checkpoint_gate"] = (
                BASE.checkpoint_gate(
                    data_root,
                    run_stamp=stamp,
                    arm=arm,
                    terminal_step=run["terminal_step"],
                )
                if run["status"] == "complete"
                else {"status": "pending"}
            )
            run["violations"].extend(
                run["checkpoint_gate"].get("violations", [])
            )
        domains[domain] = {
            "runs": {"e53_grpo": control, "e55": treatment},
            "behavioral_gate": BASE.behavioral_gate(control, treatment),
            "safety_gate": _safety_gate(control, treatment),
        }
    violations = list(binding_violations) + [
        f"{domain}/{arm}: {violation}"
        for domain, payload in domains.items()
        for arm, run in payload["runs"].items()
        for violation in run["violations"]
    ]
    gate_states = [
        payload[key]["status"]
        for payload in domains.values()
        for key in ("behavioral_gate", "safety_gate")
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
        "schema": "e55_sentinel_audit_v1",
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
        default=ROOT / "var/artifacts/e55_sentinel_audit_latest.json",
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
        f"[e55-audit] status={payload['status']} "
        f"violations={len(payload['violations'])} out={args.out}"
    )
    for domain, item in payload["domains"].items():
        control = item["runs"]["e53_grpo"]
        treatment = item["runs"]["e55"]
        print(
            f"[e55-audit] {domain}: "
            f"control={control['status']}@{control['training_passes']:.2f}, "
            f"e55={treatment['status']}@{treatment['training_passes']:.2f}; "
            f"behavior={item['behavioral_gate']['status']}"
        )
    return 1 if payload["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
