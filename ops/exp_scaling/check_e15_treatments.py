#!/usr/bin/env python3
"""Fail-closed validation and comparison for E15 M075/M10.

E15 is a separately preregistered, single-seed continuation of E14's
canonical graph-action calibration.  This gate deliberately reuses E14's
runtime, metric, exact-audit, safety, and diversity checks while adding two
provenance requirements:

* every arm is bound to the exact approved E14 C0 artifact; and
* every arm is bound to, and replays, E14's final no-viable-dose decision.

Run ``validate`` only after a treatment has completed and its exact
``step_00128`` audit has passed.  Initial validation requires the complete
checkpoint schedule.  After validation, the shared archival-receipt machinery
may authorize replay from the exact retained-only 128/129 layout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:  # package import in tests
    from . import check_e14_c0 as c0
    from . import check_e14_treatments as e14
    from .check_e14_preflight import (
        EXPECTED_DATASET_IDENTITY,
        EXPECTED_RUNTIME_IDENTITY,
        GateError,
        _sha256_file,
        _strict_json,
        parse_slurm_status,
        query_slurm,
        source_tree_hash,
    )
    from .e14_archival import (
        ArchiveReceiptError,
        replay_archive_authorization_if_needed,
    )
    from .verify_e14_c0_approval import verify_c0_approval_for_source
except ImportError:  # direct script execution
    import check_e14_c0 as c0
    import check_e14_treatments as e14
    from check_e14_preflight import (
        EXPECTED_DATASET_IDENTITY,
        EXPECTED_RUNTIME_IDENTITY,
        GateError,
        _sha256_file,
        _strict_json,
        parse_slurm_status,
        query_slurm,
        source_tree_hash,
    )
    from e14_archival import (
        ArchiveReceiptError,
        replay_archive_authorization_if_needed,
    )
    from verify_e14_c0_approval import verify_c0_approval_for_source

# Use the exception class emitted by the executable E14 modules.  Importing
# those modules as both package and direct-script modules can otherwise create
# two nominally distinct GateError classes in one test process.
GateError = c0.GateError


GATE_NAME = "e15_canonical_fixed_treatment"
RESULT_SCHEMA = "e15_canonical_treatment_result_v1"
COMPARISON_SCHEMA = "e15_canonical_treatment_comparison_v1"
PROTOCOL = "E15"
ARM_SPECS = {
    "M075": {"phase": "m075", "alpha": 0.075},
    "M10": {"phase": "m10", "alpha": 0.10},
}
IDENTITY_KEYS = {
    "phase",
    "stamp",
    "source_hash",
    "dataset_identity",
    "runtime_identity",
    "target_optimizer_updates",
    "trajectory_query_budget",
    "group_size",
    "protocol",
    "protocol_arm",
    "arm",
    "maxent_alpha",
    "c0_approval",
    "c0_approval_sha256",
    "e14_outcome",
    "e14_outcome_sha256",
}


def _close(observed: float, expected: float, *, label: str) -> None:
    if not math.isfinite(observed) or not math.isclose(
        observed, expected, rel_tol=1e-5, abs_tol=1e-8
    ):
        raise GateError(f"{label}={observed!r}; expected {expected!r}")


def _read_identity_rows(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise GateError(f"E15 treatment identity is missing: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if not rows or rows[0] != ["key", "value"]:
        raise GateError("E15 identity must start with exact key/value header")
    identity: dict[str, str] = {}
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) != 2 or not row[0] or row[0] in identity:
            raise GateError(
                f"malformed or duplicate E15 identity row {line_number}"
            )
        identity[row[0]] = row[1]
    if set(identity) != IDENTITY_KEYS:
        raise GateError(
            "E15 identity fields drifted; "
            f"missing={sorted(IDENTITY_KEYS - set(identity))} "
            f"extra={sorted(set(identity) - IDENTITY_KEYS)}"
        )
    return identity


def read_treatment_identity(
    path: Path,
    *,
    arm: str,
    c0_approval: Path,
    e14_outcome: Path,
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    """Read and fully bind one prospective E15 launch identity."""

    if arm not in ARM_SPECS:
        raise GateError(f"unknown E15 treatment arm {arm!r}")
    identity = _read_identity_rows(path)
    spec = ARM_SPECS[arm]
    expected = {
        "phase": str(spec["phase"]),
        "protocol": PROTOCOL,
        "protocol_arm": arm,
        "arm": "maxent",
        "target_optimizer_updates": str(c0.EXPECTED_UPDATES),
        "trajectory_query_budget": str(c0.EXPECTED_QUERY_BUDGET),
        "group_size": str(c0.EXPECTED_GROUP_SIZE),
    }
    for key, value in expected.items():
        if identity[key] != value:
            raise GateError(
                f"E15 identity {key}={identity[key]!r}; expected {value!r}"
            )
    if path.name != f"{identity['stamp']}_e15_identity.tsv":
        raise GateError("E15 identity filename does not match its stamp")
    _close(
        float(identity["maxent_alpha"]),
        float(spec["alpha"]),
        label="E15 identity alpha",
    )
    bindings = (
        ("c0_approval", c0_approval),
        ("e14_outcome", e14_outcome),
    )
    for field, expected_path in bindings:
        if Path(identity[field]).resolve() != expected_path.resolve():
            raise GateError(f"E15 identity names a different {field}")
        if identity[f"{field}_sha256"] != _sha256_file(expected_path):
            raise GateError(f"E15 identity {field} hash changed")
    dataset = _strict_json(
        identity["dataset_identity"], context="E15 dataset_identity"
    )
    runtime = _strict_json(
        identity["runtime_identity"], context="E15 runtime_identity"
    )
    if dataset != EXPECTED_DATASET_IDENTITY:
        raise GateError("E15 frozen dataset identity drifted")
    if runtime != EXPECTED_RUNTIME_IDENTITY:
        raise GateError("E15 frozen runtime identity drifted")
    if re.fullmatch(r"[0-9a-f]{64}", identity["source_hash"]) is None:
        raise GateError("E15 source hash is not a lowercase SHA-256")
    return identity, dataset, runtime


def replay_e14_outcome(
    outcome_path: Path, *, c0_approval_path: Path
) -> dict[str, Any]:
    """Replay E14's final comparison, rejecting edited summary JSON."""

    if not outcome_path.is_file():
        raise GateError(f"E14 outcome is missing: {outcome_path}")
    payload = _strict_json(
        outcome_path.read_text(encoding="utf-8"), context=str(outcome_path)
    )
    if not isinstance(payload, dict):
        raise GateError("E14 outcome is not a JSON object")
    expected_header = {
        "schema": e14.COMPARISON_SCHEMA,
        "protocol": "E14",
        "status": "no_viable_dose",
        "selected_arm": None,
        "does_not_authorize_scale_or_domain_expansion": True,
    }
    for key, expected in expected_header.items():
        if payload.get(key) != expected:
            raise GateError(
                f"E14 outcome {key}={payload.get(key)!r}; expected {expected!r}"
            )
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != {
        "c0_approval",
        "m01_result",
        "m05_result",
    }:
        raise GateError("E14 outcome evidence fields drifted")
    if evidence["c0_approval"] != {
        "path": str(c0_approval_path.resolve()),
        "sha256": _sha256_file(c0_approval_path),
    }:
        raise GateError("E14 outcome is bound to a different C0 approval")
    try:
        replayed = e14.compare_result_payloads(
            c0_approval_path=c0_approval_path,
            m01_path=Path(evidence["m01_result"]["path"]),
            m05_path=Path(evidence["m05_result"]["path"]),
        )
    except (ArchiveReceiptError, KeyError, TypeError, ValueError) as error:
        raise GateError(f"E14 outcome replay rejected: {error}") from error
    observed = dict(payload)
    expected = dict(replayed)
    observed.pop("compared_at_utc", None)
    expected.pop("compared_at_utc", None)
    if observed != expected:
        raise GateError("E14 outcome does not match replayed treatment evidence")
    return {
        "path": str(outcome_path.resolve()),
        "sha256": _sha256_file(outcome_path),
        "status": "no_viable_dose",
        "selected_arm": None,
        "m01_viable": False,
        "m05_viable": False,
        "fully_replayed": True,
    }


def classify_treatment(
    *,
    arm: str,
    c0_endpoint: dict[str, Any],
    endpoint: dict[str, Any],
    final_reward: float,
) -> dict[str, Any]:
    """Apply E14's unchanged scientific thresholds to an E15 arm."""

    return e14.classify_treatment(
        arm=arm,
        c0_endpoint=c0_endpoint,
        endpoint=endpoint,
        final_reward=final_reward,
    )


def _inspect_treatment_metrics(
    run_dir: Path,
    *,
    arm: str,
    alpha: float,
    archived_removed_steps: Sequence[str] | None,
) -> tuple[dict[str, Any], Path, Path]:
    """Keep E15's reward threshold in scientific classification only."""

    return e14.inspect_treatment_metrics(
        run_dir,
        arm=arm,
        alpha=alpha,
        archived_removed_steps=archived_removed_steps,
        # E15 freezes positive final reward as a scientific safety gate, not
        # a runtime-validity condition. E14 keeps its stricter default.
        require_positive_final_reward=False,
    )


def choose_arm(classifications: Sequence[dict[str, Any]]) -> tuple[str | None, str]:
    """Choose by valid support; use lower alpha for a preregistered 5% tie."""

    observed_arms = {str(row.get("arm")) for row in classifications}
    if observed_arms != set(ARM_SPECS):
        raise GateError(
            f"E15 comparison requires exactly {sorted(ARM_SPECS)}; "
            f"observed={sorted(observed_arms)}"
        )
    viable = [row for row in classifications if row.get("viable") is True]
    if not viable:
        return None, "no runtime-valid arm passed both safety and diversity gates"
    if len(viable) == 1:
        return str(viable[0]["arm"]), "only viable fixed coefficient"
    by_arm = {str(row["arm"]): row for row in viable}
    low = min(
        float(by_arm["M075"]["n_eff_valid_mean"]),
        float(by_arm["M10"]["n_eff_valid_mean"]),
    )
    high = max(
        float(by_arm["M075"]["n_eff_valid_mean"]),
        float(by_arm["M10"]["n_eff_valid_mean"]),
    )
    if low / high >= 1.0 - e14.TIE_RELATIVE_WIDTH:
        return "M075", (
            "valid-mode supports are within 5%; preregistered smaller "
            "coefficient wins"
        )
    selected = max(viable, key=lambda row: float(row["n_eff_valid_mean"]))
    return str(selected["arm"]), "larger exact mean valid-mode effective support"


def check_treatment(
    *,
    arm: str,
    run_dir: Path,
    identity_path: Path,
    c0_approval_path: Path,
    e14_outcome_path: Path,
    endpoint_audit_path: Path,
    source_root: Path,
    stdout_path: Path,
    stderr_path: Path | None,
    slurm_state: str,
    slurm_exit_code: str,
    job_id: str,
    logical_repo_root: Path,
    archived_removed_steps: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate one E15 run and return its immutable result payload."""

    if arm not in ARM_SPECS:
        raise GateError(f"unknown E15 arm {arm!r}")
    if re.fullmatch(r"[0-9]+", str(job_id)) is None:
        raise GateError(f"{arm} has invalid Slurm job ID {job_id!r}")
    alpha = float(ARM_SPECS[arm]["alpha"])
    identity, dataset, runtime = read_treatment_identity(
        identity_path,
        arm=arm,
        c0_approval=c0_approval_path,
        e14_outcome=e14_outcome_path,
    )
    stamp = identity["stamp"]
    run_dir = run_dir.resolve()
    if not run_dir.is_dir() or not run_dir.name.endswith(
        f"_{stamp}_maxent_s9005"
    ):
        raise GateError(f"{arm} run directory does not match stamp/arm/seed")
    observed_source_hash = source_tree_hash(
        source_root, logical_repo_root=logical_repo_root
    )
    if identity["source_hash"] != observed_source_hash:
        raise GateError(f"{arm} immutable source snapshot hash drifted")
    c0_summary = verify_c0_approval_for_source(
        c0_approval_path,
        expected_source_hash=observed_source_hash,
        logical_repo_root=logical_repo_root,
    )
    e14_outcome_summary = replay_e14_outcome(
        e14_outcome_path, c0_approval_path=c0_approval_path
    )
    if slurm_state.strip() != "COMPLETED" or slurm_exit_code.strip() != "0:0":
        raise GateError(
            f"{arm} job {job_id} is not clean terminal success: "
            f"{slurm_state}/{slurm_exit_code}"
        )
    log_summary = e14.inspect_treatment_logs(
        stdout_path,
        stderr_path,
        run_dir=run_dir,
        source_root=source_root,
        job_id=job_id,
        arm=arm,
        alpha=alpha,
    )
    # archived_removed_steps is intentionally unavailable from the validate
    # CLI.  Thus first validation always requires the complete checkpoint
    # schedule; only replay of an already validated result can use a receipt.
    metrics_summary, metrics_path, checkpoint = _inspect_treatment_metrics(
        run_dir,
        arm=arm,
        alpha=alpha,
        archived_removed_steps=archived_removed_steps,
    )
    endpoint_summary = c0.inspect_endpoint_audit(
        endpoint_audit_path,
        checkpoint=checkpoint,
        expected_source_root=source_root,
        expected_source_hash=observed_source_hash,
    )
    c0_payload = _strict_json(
        c0_approval_path.read_text(encoding="utf-8"),
        context=str(c0_approval_path),
    )
    c0_audit_path = Path(c0_payload["endpoint_summary"]["audit"])
    if e14._audit_semantic_fingerprint(
        endpoint_audit_path
    ) != e14._audit_semantic_fingerprint(c0_audit_path):
        raise GateError(
            f"{arm} endpoint audit does not use C0's exact prompt/reward alignment"
        )
    classification = classify_treatment(
        arm=arm,
        c0_endpoint=c0_payload["endpoint_summary"],
        endpoint=endpoint_summary,
        final_reward=float(metrics_summary["final_32_mean_rollout_reward"]),
    )
    evidence_paths = {
        "identity": identity_path,
        "c0_approval": c0_approval_path,
        "e14_outcome": e14_outcome_path,
        "metrics": metrics_path,
        "source_snapshot_marker": source_root / "oat_drgrpo" / "__init__.py",
        "stdout": stdout_path,
        "endpoint_audit": endpoint_audit_path,
        "checkpoint_config": checkpoint / "config.json",
    }
    if stderr_path is not None:
        evidence_paths["stderr"] = stderr_path
    for label, path in evidence_paths.items():
        if not path.is_file():
            raise GateError(f"{arm} evidence {label!r} is missing: {path}")
    evidence = {
        key: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
        for key, path in evidence_paths.items()
    }
    return {
        "schema": RESULT_SCHEMA,
        "gate": GATE_NAME,
        "protocol": PROTOCOL,
        "arm": arm,
        "alpha": alpha,
        "runtime_valid": True,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": job_id,
        "run_dir": str(run_dir),
        "slurm": {"state": "COMPLETED", "exit_code": "0:0"},
        "identity": {
            "stamp": stamp,
            "source_hash": observed_source_hash,
            "dataset": dataset,
            "runtime": runtime,
        },
        "c0_summary": c0_summary,
        "e14_outcome_summary": e14_outcome_summary,
        "checks": {
            "c0_approval_fully_replayed": True,
            "e14_no_dose_outcome_fully_replayed": True,
            "immutable_source_and_identity": True,
            "terminal_success": True,
            "full_checkpoint_schedule_at_initial_validation": True,
            "updates_97_128_contiguous_finite": True,
            "canonical_rollouts_and_behavior_overlap": True,
            "fixed_alpha_on_every_update": True,
            "single_shared_outer_normalization": True,
            "adaptive_and_length_controllers_absent": True,
            "exact_step_00128_audit": True,
            "same_prompt_reward_alignment_as_c0": True,
        },
        "log_summary": log_summary,
        "metrics_summary": metrics_summary,
        "endpoint_summary": endpoint_summary,
        "classification": classification,
        "evidence": evidence,
    }


def _replay_result(
    *,
    expected_arm: str,
    path: Path,
    c0_approval_path: Path,
    e14_outcome_path: Path,
    logical_repo_root: Path,
) -> dict[str, Any]:
    payload = _strict_json(path.read_text(encoding="utf-8"), context=str(path))
    if not isinstance(payload, dict) or payload.get("schema") != RESULT_SCHEMA:
        raise GateError(f"{expected_arm} E15 result has wrong schema")
    if (
        payload.get("protocol") != PROTOCOL
        or payload.get("gate") != GATE_NAME
        or payload.get("arm") != expected_arm
        or payload.get("runtime_valid") is not True
    ):
        raise GateError(f"{expected_arm} result identity or status drifted")
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        raise GateError(f"{expected_arm} result lacks evidence")
    expected_bindings = {
        "c0_approval": (c0_approval_path, _sha256_file(c0_approval_path)),
        "e14_outcome": (e14_outcome_path, _sha256_file(e14_outcome_path)),
    }
    for label, (expected_path, expected_hash) in expected_bindings.items():
        record = evidence.get(label)
        if record != {
            "path": str(expected_path.resolve()),
            "sha256": expected_hash,
        }:
            raise GateError(f"{expected_arm} is bound to a different {label}")
    for label, record in evidence.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise GateError(f"{expected_arm} evidence {label} is malformed")
        evidence_path = Path(str(record["path"]))
        if (
            not evidence_path.is_file()
            or _sha256_file(evidence_path) != record["sha256"]
        ):
            raise GateError(
                f"{expected_arm} evidence {label} is missing or changed"
            )
    source_marker = Path(evidence["source_snapshot_marker"]["path"]).resolve()
    if source_marker.name != "__init__.py" or source_marker.parent.name != "oat_drgrpo":
        raise GateError(f"{expected_arm} source marker has the wrong location")
    try:
        archive_authorization = replay_archive_authorization_if_needed(path)
    except ArchiveReceiptError as error:
        raise GateError(
            f"{expected_arm} archival replay rejected: {error}"
        ) from error
    replayed = check_treatment(
        arm=expected_arm,
        run_dir=Path(payload["run_dir"]),
        identity_path=Path(evidence["identity"]["path"]),
        c0_approval_path=c0_approval_path,
        e14_outcome_path=e14_outcome_path,
        endpoint_audit_path=Path(evidence["endpoint_audit"]["path"]),
        source_root=source_marker.parents[1],
        stdout_path=Path(evidence["stdout"]["path"]),
        stderr_path=(
            Path(evidence["stderr"]["path"])
            if "stderr" in evidence
            else None
        ),
        slurm_state="COMPLETED",
        slurm_exit_code="0:0",
        job_id=str(payload["job_id"]),
        logical_repo_root=logical_repo_root,
        archived_removed_steps=(
            archive_authorization["removed_steps"]
            if archive_authorization is not None
            else None
        ),
    )
    observed = dict(payload)
    expected = dict(replayed)
    observed.pop("validated_at_utc", None)
    expected.pop("validated_at_utc", None)
    if observed != expected:
        raise GateError(
            f"{expected_arm} result does not match replayed E15 evidence"
        )
    return replayed


def compare_result_payloads(
    *,
    c0_approval_path: Path,
    e14_outcome_path: Path,
    m075_path: Path,
    m10_path: Path,
    logical_repo_root: Path | None = None,
) -> dict[str, Any]:
    """Replay both E15 validations, classify, and apply frozen selection."""

    repo_root = logical_repo_root or Path(__file__).resolve().parents[2]
    replay_e14_outcome(e14_outcome_path, c0_approval_path=c0_approval_path)
    results = []
    for expected_arm, path in (("M075", m075_path), ("M10", m10_path)):
        replayed = _replay_result(
            expected_arm=expected_arm,
            path=path,
            c0_approval_path=c0_approval_path,
            e14_outcome_path=e14_outcome_path,
            logical_repo_root=repo_root,
        )
        results.append(replayed["classification"])
    selected, rationale = choose_arm(results)
    return {
        "schema": COMPARISON_SCHEMA,
        "protocol": PROTOCOL,
        "status": "selected" if selected is not None else "no_viable_dose",
        "selected_arm": selected,
        "selection_rationale": rationale,
        "single_seed_engineering_calibration": True,
        "does_not_authorize_scale_or_domain_expansion": True,
        "thresholds": {
            "p_valid_absolute_strict_min": e14.P_VALID_ABSOLUTE_MIN,
            "p_valid_retention_min": e14.P_VALID_RETENTION_MIN,
            "exact_entropy_gain_min_nats": e14.ENTROPY_GAIN_MIN,
            "n_eff_valid_ratio_min": e14.VALID_SUPPORT_GAIN_MIN,
            "tie_relative_width": e14.TIE_RELATIVE_WIDTH,
        },
        "arms": results,
        "evidence": {
            "c0_approval": {
                "path": str(c0_approval_path.resolve()),
                "sha256": _sha256_file(c0_approval_path),
            },
            "e14_outcome": {
                "path": str(e14_outcome_path.resolve()),
                "sha256": _sha256_file(e14_outcome_path),
            },
            "m075_result": {
                "path": str(m075_path.resolve()),
                "sha256": _sha256_file(m075_path),
            },
            "m10_result": {
                "path": str(m10_path.resolve()),
                "sha256": _sha256_file(m10_path),
            },
        },
        "compared_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _job_status(args: argparse.Namespace) -> tuple[str, str]:
    if args.slurm_status is not None:
        return parse_slurm_status(args.slurm_status, expected_job_id=args.job_id)
    if args.slurm_state is not None or args.slurm_exit_code is not None:
        if args.slurm_state is None or args.slurm_exit_code is None:
            raise GateError(
                "both --slurm-state and --slurm-exit-code are required"
            )
        return args.slurm_state, args.slurm_exit_code
    return query_slurm(args.job_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and compare E15 canonical fixed-alpha treatments."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--arm", choices=tuple(ARM_SPECS), required=True)
    validate.add_argument("--run-dir", type=Path, required=True)
    validate.add_argument("--identity", type=Path, required=True)
    validate.add_argument("--c0-approval", type=Path, required=True)
    validate.add_argument("--e14-outcome", type=Path, required=True)
    validate.add_argument("--endpoint-audit", type=Path, required=True)
    validate.add_argument("--source-root", type=Path, required=True)
    validate.add_argument("--stdout-log", type=Path, required=True)
    validate.add_argument("--stderr-log", type=Path)
    validate.add_argument("--job-id", required=True)
    validate.add_argument("--slurm-status", type=Path)
    validate.add_argument("--slurm-state")
    validate.add_argument("--slurm-exit-code")
    validate.add_argument("--result-out", type=Path, required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--c0-approval", type=Path, required=True)
    compare.add_argument("--e14-outcome", type=Path, required=True)
    compare.add_argument("--m075-result", type=Path, required=True)
    compare.add_argument("--m10-result", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output = args.result_out if args.command == "validate" else args.out
    output.unlink(missing_ok=True)
    if args.command == "validate":
        state, exit_code = _job_status(args)
        payload = check_treatment(
            arm=args.arm,
            run_dir=args.run_dir,
            identity_path=args.identity,
            c0_approval_path=args.c0_approval,
            e14_outcome_path=args.e14_outcome,
            endpoint_audit_path=args.endpoint_audit,
            source_root=args.source_root,
            stdout_path=args.stdout_log,
            stderr_path=args.stderr_log,
            slurm_state=state,
            slurm_exit_code=exit_code,
            job_id=args.job_id,
            logical_repo_root=repo_root,
        )
    else:
        payload = compare_result_payloads(
            c0_approval_path=args.c0_approval,
            e14_outcome_path=args.e14_outcome,
            m075_path=args.m075_result,
            m10_path=args.m10_result,
            logical_repo_root=repo_root,
        )
    _write(output, payload)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "status": payload.get("status", "runtime_valid"),
                "selected_arm": payload.get("selected_arm"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
