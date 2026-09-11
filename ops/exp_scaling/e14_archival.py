#!/usr/bin/env python3
"""Create and verify fail-closed canonical-action archive receipts.

E14's runtime gates intentionally require the complete checkpoint schedule on
their first pass.  Once an approval/result exists, the large intermediate
snapshots may be removed while retaining the audited update-128 checkpoint and
its byte-identical terminal alias.  This module records that storage-only
transition without fabricating the removed checkpoint directories.

Receipts are deliberately external to the immutable validation artifact.  A
receipt binds the artifact by SHA-256, binds the exact E14 run/job/source
identity, names the three removed steps, and records complete file manifests
for both retained checkpoints.  Replay callers may use a receipt only when the
on-disk layout is *exactly* the archived layout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "e14_post_validation_archive_receipt_v1"
E15_SCHEMA = "e15_post_validation_archive_receipt_v1"
SCHEMA_BY_PROTOCOL = {"E14": SCHEMA, "E15": E15_SCHEMA}
EXPECTED_CHECKPOINT_STEPS = (
    "step_00032",
    "step_00064",
    "step_00096",
    "step_00128",
    "step_00129",
)
REMOVED_CHECKPOINT_STEPS = ("step_00032", "step_00064", "step_00096")
RETAINED_CHECKPOINT_STEPS = ("step_00128", "step_00129")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ArchiveReceiptError(ValueError):
    """Raised when an archival transition cannot be proved exactly."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant {value}")
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ArchiveReceiptError(f"cannot read strict JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ArchiveReceiptError(f"JSON artifact is not an object: {path}")
    return payload


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ArchiveReceiptError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _absolute_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ArchiveReceiptError(f"{label} is not an absolute path")
    return Path(value).resolve()


def _artifact_kind_and_identity(
    artifact_path: Path, payload: dict[str, Any]
) -> dict[str, str | Path]:
    arm = payload.get("arm")
    if arm == "C0":
        if (
            payload.get("approved") is not True
            or payload.get("gate") != "e14_canonical_c0"
            or payload.get("protocol") != "E14"
        ):
            raise ArchiveReceiptError("C0 archive source is not an approved E14 C0 artifact")
        kind = "c0_approval"
        validation_time = payload.get("approved_at_utc")
    elif arm in {"M01", "M05"}:
        if (
            payload.get("schema") != "e14_canonical_treatment_result_v1"
            or payload.get("gate") != "e14_canonical_fixed_treatment"
            or payload.get("protocol") != "E14"
            or payload.get("runtime_valid") is not True
        ):
            raise ArchiveReceiptError(
                f"{arm} archive source is not a runtime-valid E14 treatment result"
            )
        kind = "treatment_result"
        validation_time = payload.get("validated_at_utc")
    elif arm in {"M075", "M10"}:
        if (
            payload.get("schema") != "e15_canonical_treatment_result_v1"
            or payload.get("gate") != "e15_canonical_fixed_treatment"
            or payload.get("protocol") != "E15"
            or payload.get("runtime_valid") is not True
        ):
            raise ArchiveReceiptError(
                f"{arm} archive source is not a runtime-valid E15 treatment result"
            )
        kind = "treatment_result"
        validation_time = payload.get("validated_at_utc")
    else:
        raise ArchiveReceiptError(f"unsupported E14 archival arm {arm!r}")

    try:
        parsed_time = datetime.fromisoformat(str(validation_time))
    except ValueError as error:
        raise ArchiveReceiptError("validation artifact timestamp is invalid") from error
    if parsed_time.tzinfo is None or parsed_time.utcoffset() is None:
        raise ArchiveReceiptError("validation artifact timestamp is not timezone-aware")

    job_id = str(payload.get("job_id", ""))
    if re.fullmatch(r"[0-9]+", job_id) is None:
        raise ArchiveReceiptError("validation artifact has an invalid Slurm job ID")
    slurm = payload.get("slurm")
    if slurm != {"state": "COMPLETED", "exit_code": "0:0"}:
        raise ArchiveReceiptError("validation artifact lacks clean terminal Slurm success")

    identity = payload.get("identity")
    if not isinstance(identity, dict):
        raise ArchiveReceiptError("validation artifact has no identity object")
    stamp = identity.get("stamp")
    if not isinstance(stamp, str) or not stamp:
        raise ArchiveReceiptError("validation artifact has no run stamp")
    source_hash = _require_sha256(identity.get("source_hash"), label="source hash")
    run_dir = _absolute_path(payload.get("run_dir"), label="run directory")

    checks = payload.get("checks")
    if not isinstance(checks, dict) or checks.get("exact_step_00128_audit") is not True:
        raise ArchiveReceiptError("validation artifact lacks the exact step-128 audit gate")
    metrics_summary = payload.get("metrics_summary")
    if not isinstance(metrics_summary, dict):
        raise ArchiveReceiptError("validation artifact has no metrics summary")
    endpoint_hash = _require_sha256(
        metrics_summary.get("step_128_weights_manifest_sha256"),
        label="validated step-128 weights manifest",
    )
    if metrics_summary.get("step_129_byte_identical_alias") is not True:
        raise ArchiveReceiptError("validation artifact did not validate the step-129 alias")

    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        raise ArchiveReceiptError("validation artifact has no evidence object")
    for required in ("identity", "metrics", "endpoint_audit", "checkpoint_config"):
        if required not in evidence:
            raise ArchiveReceiptError(f"validation artifact lacks evidence {required!r}")
    for label, record in evidence.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ArchiveReceiptError(f"validation evidence {label!r} is malformed")
        evidence_path = _absolute_path(record["path"], label=f"{label} evidence path")
        evidence_hash = _require_sha256(
            record["sha256"], label=f"{label} evidence hash"
        )
        if not evidence_path.is_file():
            raise ArchiveReceiptError(f"validation evidence {label!r} is missing")
        if _sha256_file(evidence_path) != evidence_hash:
            raise ArchiveReceiptError(f"validation evidence {label!r} changed")

    metrics_path = _absolute_path(
        evidence["metrics"]["path"], label="metrics evidence path"
    )
    if metrics_path.name != "train_metrics.jsonl" or metrics_path.parent.parent != run_dir:
        raise ArchiveReceiptError("metrics evidence is not inside the validated run directory")
    saved_models = metrics_path.parent / "saved_models"
    config_path = _absolute_path(
        evidence["checkpoint_config"]["path"], label="checkpoint config path"
    )
    if config_path != (saved_models / "step_00128" / "config.json").resolve():
        raise ArchiveReceiptError("checkpoint config evidence is not from step_00128")

    return {
        "kind": kind,
        "protocol": str(payload.get("protocol")),
        "arm": str(arm),
        "job_id": job_id,
        "stamp": stamp,
        "source_hash": source_hash,
        "run_dir": run_dir,
        "saved_models": saved_models.resolve(),
        "validation_time": str(validation_time),
        "endpoint_hash": endpoint_hash,
        "artifact": artifact_path.resolve(),
    }


def _regular_file_manifest(root: Path) -> tuple[str, list[dict[str, Any]]]:
    if not root.is_dir() or root.is_symlink():
        raise ArchiveReceiptError(f"retained checkpoint is not a real directory: {root}")
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ArchiveReceiptError(f"retained checkpoint contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ArchiveReceiptError(f"retained checkpoint contains a non-file: {path}")
        relative = path.relative_to(root).as_posix()
        records.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        )
    if not records:
        raise ArchiveReceiptError(f"retained checkpoint is empty: {root}")
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(record["sha256"]))
    return digest.hexdigest(), records


def _weight_manifest(root: Path) -> tuple[str, list[dict[str, Any]]]:
    paths = sorted(root.glob("*.safetensors"), key=lambda path: path.name)
    if not paths:
        raise ArchiveReceiptError(f"retained checkpoint has no safetensors weights: {root}")
    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    for path in paths:
        file_hash = _sha256_file(path)
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(file_hash))
        records.append({"name": path.name, "sha256": file_hash})
    return digest.hexdigest(), records


def _checkpoint_record(path: Path) -> dict[str, Any]:
    tree_hash, files = _regular_file_manifest(path)
    weight_hash, weight_files = _weight_manifest(path)
    return {
        "path": str(path.resolve()),
        "tree_manifest_sha256": tree_hash,
        "files": files,
        "weights_manifest_sha256": weight_hash,
        "weight_files": weight_files,
    }


def _checkpoint_tags(saved_models: Path) -> tuple[str, ...]:
    if not saved_models.is_dir():
        raise ArchiveReceiptError(f"saved_models directory is missing: {saved_models}")
    tags = tuple(sorted(path.name for path in saved_models.iterdir() if path.is_dir()))
    unexpected_files = sorted(path.name for path in saved_models.iterdir() if not path.is_dir())
    if unexpected_files:
        raise ArchiveReceiptError(
            f"saved_models contains unexpected non-directories: {unexpected_files}"
        )
    return tags


def archive_receipt_path(validation_artifact: Path) -> Path:
    artifact = validation_artifact.resolve()
    return artifact.with_name(f"{artifact.stem}_archive_receipt.json")


def build_archive_receipt(
    validation_artifact: Path,
    *,
    allow_post_removal_recovery: bool = False,
) -> dict[str, Any]:
    """Build a receipt after validating either the full or archived layout."""

    artifact = validation_artifact.resolve()
    if not artifact.is_file():
        raise ArchiveReceiptError(f"validation artifact is missing: {artifact}")
    payload = _strict_json(artifact)
    metadata = _artifact_kind_and_identity(artifact, payload)
    saved_models = Path(metadata["saved_models"])
    observed = _checkpoint_tags(saved_models)
    if observed == EXPECTED_CHECKPOINT_STEPS:
        phase = "pre_removal"
    elif observed == RETAINED_CHECKPOINT_STEPS:
        if not allow_post_removal_recovery:
            raise ArchiveReceiptError(
                "intermediate checkpoints are already absent; pass the explicit "
                "post-removal recovery flag to record that chronology"
            )
        phase = "post_removal_recovery"
    else:
        raise ArchiveReceiptError(
            "canonical checkpoint layout is neither the validated full schedule nor the "
            f"exact archived layout: observed={list(observed)}"
        )

    retained = {
        step: _checkpoint_record(saved_models / step)
        for step in RETAINED_CHECKPOINT_STEPS
    }
    endpoint = retained["step_00128"]
    alias = retained["step_00129"]
    if endpoint["weights_manifest_sha256"] != metadata["endpoint_hash"]:
        raise ArchiveReceiptError(
            "retained step_00128 weights differ from the validated endpoint"
        )
    if (
        endpoint["tree_manifest_sha256"] != alias["tree_manifest_sha256"]
        or endpoint["files"] != alias["files"]
        or endpoint["weights_manifest_sha256"] != alias["weights_manifest_sha256"]
        or endpoint["weight_files"] != alias["weight_files"]
    ):
        raise ArchiveReceiptError(
            "retained step_00129 is not a complete byte-identical alias of step_00128"
        )

    return {
        "schema": SCHEMA_BY_PROTOCOL[str(metadata["protocol"])],
        "protocol": metadata["protocol"],
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "recording_phase": phase,
        "validation_artifact": {
            "kind": metadata["kind"],
            "path": str(artifact),
            "sha256": _sha256_file(artifact),
            "validated_at_utc": metadata["validation_time"],
        },
        "identity": {
            "arm": metadata["arm"],
            "job_id": metadata["job_id"],
            "stamp": metadata["stamp"],
            "source_hash": metadata["source_hash"],
            "run_dir": str(metadata["run_dir"]),
        },
        "schedule": {
            "expected_steps_at_initial_validation": list(EXPECTED_CHECKPOINT_STEPS),
            "removed_steps": list(REMOVED_CHECKPOINT_STEPS),
            "retained_steps": list(RETAINED_CHECKPOINT_STEPS),
            "observed_steps_when_recorded": list(observed),
            "initial_validation_remained_full_schedule_only": True,
        },
        "retained_checkpoints": retained,
        "alias_check": {
            "step_00129_complete_tree_byte_identical_to_step_00128": True,
        },
    }


def write_archive_receipt(
    validation_artifact: Path,
    *,
    output: Path | None = None,
    allow_post_removal_recovery: bool = False,
    replace: bool = False,
) -> Path:
    artifact = validation_artifact.resolve()
    destination = (output or archive_receipt_path(artifact)).resolve()
    if destination.exists() and not replace:
        raise ArchiveReceiptError(f"archive receipt already exists: {destination}")
    receipt = build_archive_receipt(
        artifact, allow_post_removal_recovery=allow_post_removal_recovery
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _exact_keys(value: Any, expected: Iterable[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(expected):
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ArchiveReceiptError(
            f"{label} fields drifted: observed={observed} expected={sorted(expected)}"
        )
    return value


def verify_archive_receipt(
    receipt_path: Path,
    *,
    validation_artifact: Path,
) -> dict[str, Any]:
    """Verify a receipt and the exact retained-only on-disk state."""

    receipt_file = receipt_path.resolve()
    artifact = validation_artifact.resolve()
    if not receipt_file.is_file():
        raise ArchiveReceiptError(f"archive receipt is missing: {receipt_file}")
    if not artifact.is_file():
        raise ArchiveReceiptError(f"validation artifact is missing: {artifact}")
    artifact_payload = _strict_json(artifact)
    metadata = _artifact_kind_and_identity(artifact, artifact_payload)
    receipt = _strict_json(receipt_file)
    _exact_keys(
        receipt,
        {
            "schema",
            "protocol",
            "recorded_at_utc",
            "recording_phase",
            "validation_artifact",
            "identity",
            "schedule",
            "retained_checkpoints",
            "alias_check",
        },
        label="archive receipt",
    )
    expected_protocol = str(metadata["protocol"])
    if (
        receipt["schema"] != SCHEMA_BY_PROTOCOL.get(expected_protocol)
        or receipt["protocol"] != expected_protocol
    ):
        raise ArchiveReceiptError("archive receipt has the wrong schema or protocol")
    if receipt["recording_phase"] not in {"pre_removal", "post_removal_recovery"}:
        raise ArchiveReceiptError("archive receipt has an invalid recording phase")
    try:
        recorded = datetime.fromisoformat(str(receipt["recorded_at_utc"]))
    except ValueError as error:
        raise ArchiveReceiptError("archive receipt timestamp is invalid") from error
    if recorded.tzinfo is None or recorded.utcoffset() is None:
        raise ArchiveReceiptError("archive receipt timestamp is not timezone-aware")

    artifact_record = _exact_keys(
        receipt["validation_artifact"],
        {"kind", "path", "sha256", "validated_at_utc"},
        label="validation artifact binding",
    )
    if artifact_record != {
        "kind": metadata["kind"],
        "path": str(artifact),
        "sha256": _sha256_file(artifact),
        "validated_at_utc": metadata["validation_time"],
    }:
        raise ArchiveReceiptError("archive receipt is not bound to this validation artifact")

    identity = _exact_keys(
        receipt["identity"],
        {"arm", "job_id", "stamp", "source_hash", "run_dir"},
        label="archive run identity",
    )
    expected_identity = {
        "arm": metadata["arm"],
        "job_id": metadata["job_id"],
        "stamp": metadata["stamp"],
        "source_hash": metadata["source_hash"],
        "run_dir": str(metadata["run_dir"]),
    }
    if identity != expected_identity:
        raise ArchiveReceiptError("archive receipt run/job/source identity drifted")

    schedule = _exact_keys(
        receipt["schedule"],
        {
            "expected_steps_at_initial_validation",
            "removed_steps",
            "retained_steps",
            "observed_steps_when_recorded",
            "initial_validation_remained_full_schedule_only",
        },
        label="archive schedule",
    )
    if schedule["expected_steps_at_initial_validation"] != list(EXPECTED_CHECKPOINT_STEPS):
        raise ArchiveReceiptError("archive receipt changed the initial checkpoint schedule")
    if schedule["removed_steps"] != list(REMOVED_CHECKPOINT_STEPS):
        raise ArchiveReceiptError("archive receipt does not name exactly steps 32/64/96")
    if schedule["retained_steps"] != list(RETAINED_CHECKPOINT_STEPS):
        raise ArchiveReceiptError("archive receipt does not retain exactly steps 128/129")
    if schedule["initial_validation_remained_full_schedule_only"] is not True:
        raise ArchiveReceiptError("archive receipt weakens initial validation")
    expected_recorded = (
        list(EXPECTED_CHECKPOINT_STEPS)
        if receipt["recording_phase"] == "pre_removal"
        else list(RETAINED_CHECKPOINT_STEPS)
    )
    if schedule["observed_steps_when_recorded"] != expected_recorded:
        raise ArchiveReceiptError("archive receipt chronology and recorded layout disagree")

    saved_models = Path(metadata["saved_models"])
    if _checkpoint_tags(saved_models) != RETAINED_CHECKPOINT_STEPS:
        raise ArchiveReceiptError(
            "archive receipt is usable only after exactly steps 32/64/96 are absent"
        )
    retained = _exact_keys(
        receipt["retained_checkpoints"],
        RETAINED_CHECKPOINT_STEPS,
        label="retained checkpoint bindings",
    )
    current: dict[str, dict[str, Any]] = {}
    for step in RETAINED_CHECKPOINT_STEPS:
        expected_path = (saved_models / step).resolve()
        record = _exact_keys(
            retained[step],
            {"path", "tree_manifest_sha256", "files", "weights_manifest_sha256", "weight_files"},
            label=f"{step} binding",
        )
        if record["path"] != str(expected_path):
            raise ArchiveReceiptError(f"archive receipt {step} path drifted")
        current[step] = _checkpoint_record(expected_path)
        if current[step] != record:
            raise ArchiveReceiptError(f"retained checkpoint {step} changed after archival")

    endpoint = current["step_00128"]
    alias = current["step_00129"]
    if endpoint["weights_manifest_sha256"] != metadata["endpoint_hash"]:
        raise ArchiveReceiptError("archived step_00128 differs from validated weights")
    if endpoint != {**alias, "path": endpoint["path"]}:
        raise ArchiveReceiptError("archived step_00129 is not byte-identical to step_00128")
    alias_check = _exact_keys(
        receipt["alias_check"],
        {"step_00129_complete_tree_byte_identical_to_step_00128"},
        label="alias check",
    )
    if alias_check["step_00129_complete_tree_byte_identical_to_step_00128"] is not True:
        raise ArchiveReceiptError("archive receipt does not attest the terminal alias")

    return {
        "receipt": str(receipt_file),
        "receipt_sha256": _sha256_file(receipt_file),
        "validation_artifact_sha256": artifact_record["sha256"],
        "arm": identity["arm"],
        "job_id": identity["job_id"],
        "run_dir": identity["run_dir"],
        "removed_steps": tuple(REMOVED_CHECKPOINT_STEPS),
        "retained_steps": tuple(RETAINED_CHECKPOINT_STEPS),
        "recording_phase": receipt["recording_phase"],
    }


def replay_archive_authorization_if_needed(
    validation_artifact: Path,
) -> dict[str, Any] | None:
    """Return archival authorization only for the exact retained-only layout."""

    artifact = validation_artifact.resolve()
    payload = _strict_json(artifact)
    metadata = _artifact_kind_and_identity(artifact, payload)
    observed = _checkpoint_tags(Path(metadata["saved_models"]))
    if observed == EXPECTED_CHECKPOINT_STEPS:
        return None
    if observed != RETAINED_CHECKPOINT_STEPS:
        raise ArchiveReceiptError(
            "validated canonical run has a partial/unknown checkpoint layout and cannot replay: "
            f"observed={list(observed)}"
        )
    return verify_archive_receipt(
        archive_receipt_path(artifact), validation_artifact=artifact
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--validation-artifact", type=Path, required=True)
    create.add_argument("--out", type=Path)
    create.add_argument("--allow-post-removal-recovery", action="store_true")
    create.add_argument("--replace", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--validation-artifact", type=Path, required=True)
    verify.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "create":
            path = write_archive_receipt(
                args.validation_artifact,
                output=args.out,
                allow_post_removal_recovery=args.allow_post_removal_recovery,
                replace=args.replace,
            )
            summary = verify_archive_receipt(
                path, validation_artifact=args.validation_artifact
            ) if args.allow_post_removal_recovery else {
                "receipt": str(path),
                "status": "ready_for_post-validation_cleanup",
            }
        else:
            path = args.receipt or archive_receipt_path(args.validation_artifact)
            summary = verify_archive_receipt(
                path, validation_artifact=args.validation_artifact
            )
    except ArchiveReceiptError as error:
        raise SystemExit(f"E14 archival receipt rejected: {error}") from error
    print(json.dumps(summary, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
