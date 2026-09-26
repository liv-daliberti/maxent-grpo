#!/usr/bin/env python3
"""Replay an immutable all-six E16 smoke approval before full submission."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

try:  # package import in tests
    from .verify_e16_canonical_datasets import (
        EXPECTED_COUNTDOWN_COMBINED_CONTENT_HASH,
    )
    from .verify_e14_dataset import (
        EXPECTED_COMBINED_CONTENT_HASH as EXPECTED_GRAPH_COMBINED_CONTENT_HASH,
    )
except ImportError:  # direct script execution
    from verify_e16_canonical_datasets import (  # type: ignore[no-redef]
        EXPECTED_COUNTDOWN_COMBINED_CONTENT_HASH,
    )
    from verify_e14_dataset import (  # type: ignore[no-redef]
        EXPECTED_COMBINED_CONTENT_HASH as EXPECTED_GRAPH_COMBINED_CONTENT_HASH,
    )


TASKS = ("graph_coloring", "countdown")
ARMS = ("maxent", "maxent_control", "maxent_dual")
REQUIRED_CHECKS = {
    "canonical_rollout_shape_and_termination",
    "controller_handoff_and_recurrence",
    "endpoint_cell_run_checkpoint_binding",
    "endpoint_reported_enumeration_within_tolerances",
    "finite_behavior_current_overlap",
    "manifest_cell_binding",
    "normalized_actor_behavior_and_exact_current_policy",
    "positive_rollout_reward_observed",
    "post_update_exact_entropy_enumeration",
    "single_contiguous_32_update_metric_stream",
    "stdout_job_run_campaign_binding",
    "terminal_scheduler_success",
}


class SmokeApprovalError(ValueError):
    """Raised when a purported Stage-S approval cannot be fully replayed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SmokeApprovalError(f"{label} is not JSON") from error
    if not isinstance(payload, dict):
        raise SmokeApprovalError(f"{label} is not an object")
    return payload


def _verify_bound_file(record: Any, label: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise SmokeApprovalError(f"{label} binding is malformed")
    path = Path(str(record["path"]))
    if not path.is_absolute() or not path.is_file():
        raise SmokeApprovalError(f"{label} file is missing")
    if _sha256_file(path) != record["sha256"]:
        raise SmokeApprovalError(f"{label} file changed after approval")
    return path, _json_object(path, label)


def verify_smoke_approval(
    *,
    approval_path: Path,
    expected_approval_sha256: str,
    expected_identity: dict[str, str],
) -> dict[str, Any]:
    expected_identity_keys = {
        "dataset_identity_sha256",
        "e15_outcome_sha256",
        "execution_surface_hash",
        "full_config_digest",
        "protocol_sha256",
        "runtime_identity_sha256",
        "smoke_config_digest",
        "source_hash",
    }
    if set(expected_identity) != expected_identity_keys or any(
        re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
        for value in expected_identity.values()
    ):
        raise SmokeApprovalError("expected approval identity is incomplete")
    observed_approval_hash = _sha256_file(approval_path)
    if (
        len(expected_approval_sha256) != 64
        or observed_approval_hash != expected_approval_sha256
    ):
        raise SmokeApprovalError(
            "smoke approval does not match the separately reviewed SHA-256"
        )
    approval = _json_object(approval_path, "smoke approval")
    if (
        approval.get("schema") != "e16_canonical_smoke_approval_v1"
        or approval.get("protocol") != "E16"
        or approval.get("stage") != "smoke"
        or approval.get("status") != "approved"
        or approval.get("all_six_passed") is not True
        or approval.get("seed") != 9006
    ):
        raise SmokeApprovalError("smoke approval identity/status is wrong")
    if approval.get("identity") != expected_identity:
        raise SmokeApprovalError(
            "current protocol/source/data/runtime/config differs from Stage S"
        )
    cells = approval.get("cells")
    if not isinstance(cells, list) or len(cells) != 6:
        raise SmokeApprovalError("smoke approval must contain exactly six cells")
    observed_cells: set[tuple[str, str, int]] = set()
    unique_paths: dict[str, set[str]] = {
        "result": set(),
        "endpoint_audit": set(),
        "metrics": set(),
        "stdout": set(),
        "checkpoint_config": set(),
        "run_dir": set(),
        "checkpoint": set(),
    }
    unique_hashes: dict[str, set[str]] = {
        "result": set(),
        "endpoint_audit": set(),
        "stdout": set(),
    }
    job_ids: set[str] = set()
    run_stamps: set[str] = set()
    shared_evidence_tasks: dict[tuple[str, str], str] = {}
    task_shared_bindings: dict[tuple[str, str], tuple[str, str]] = {}
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict) or set(cell) != {
            "arm",
            "campaign_identity",
            "checkpoint",
            "endpoint_audit",
            "job_id",
            "result",
            "run_dir",
            "run_stamp",
            "seed",
            "task",
        }:
            raise SmokeApprovalError(f"smoke cell {index} is malformed")
        task = str(cell["task"])
        arm = str(cell["arm"])
        seed = int(cell["seed"])
        key = (task, arm, seed)
        if task not in TASKS or arm not in ARMS or seed != 9006:
            raise SmokeApprovalError(f"smoke cell {index} has the wrong identity")
        if key in observed_cells:
            raise SmokeApprovalError(f"duplicate smoke cell {key!r}")
        observed_cells.add(key)

        job_id = str(cell["job_id"])
        run_stamp = str(cell["run_stamp"])
        run_dir = Path(str(cell["run_dir"])).resolve()
        expected_prefix = (
            "gce16_canonical_maxent_joint_smoke_v3"
            if task == "graph_coloring"
            else "cde16_canonical_maxent_joint_smoke_v3"
        )
        if (
            re.fullmatch(r"[1-9][0-9]*", job_id) is None
            or job_id in job_ids
            or run_stamp != f"{expected_prefix}_{arm}_s9006"
            or run_stamp in run_stamps
            or not run_dir.is_dir()
        ):
            raise SmokeApprovalError(f"smoke cell {index} reuses/changes its run")
        job_ids.add(job_id)
        run_stamps.add(run_stamp)

        endpoint_path, endpoint = _verify_bound_file(
            cell["endpoint_audit"], f"{task}/{arm} endpoint audit"
        )
        result_path, result = _verify_bound_file(
            cell["result"], f"{task}/{arm} smoke result"
        )
        for label, path, digest in (
            ("endpoint_audit", endpoint_path, _sha256_file(endpoint_path)),
            ("result", result_path, _sha256_file(result_path)),
        ):
            resolved = str(path.resolve())
            if resolved in unique_paths[label] or digest in unique_hashes[label]:
                raise SmokeApprovalError(f"cell-specific {label} was reused")
            unique_paths[label].add(resolved)
            unique_hashes[label].add(digest)
        checkpoint_record = cell["checkpoint"]
        campaign_binding = cell["campaign_identity"]
        if not isinstance(checkpoint_record, dict) or not isinstance(campaign_binding, dict):
            raise SmokeApprovalError(f"{task}/{arm} lacks exact checkpoint/campaign bindings")
        checkpoint_path = Path(str(checkpoint_record.get("path", ""))).resolve()
        if (
            str(run_dir) in unique_paths["run_dir"]
            or str(checkpoint_path) in unique_paths["checkpoint"]
        ):
            raise SmokeApprovalError("one run/checkpoint was reused across cells")
        unique_paths["run_dir"].add(str(run_dir))
        unique_paths["checkpoint"].add(str(checkpoint_path))
        if (
            endpoint.get("schema") != "e16_exact_canonical_endpoint_audit_v1"
            or endpoint.get("status") != "pass"
            or endpoint.get("task") != task
            or endpoint.get("cell_identity")
            != {
                "task": task,
                "arm": arm,
                "seed": seed,
                "run_stamp": run_stamp,
                "job_id": job_id,
            }
            or endpoint.get("campaign_identity")
            != {
                key: campaign_binding[key]
                for key in (
                    "path",
                    "sha256",
                    "protocol_path",
                    "protocol_sha256",
                    "execution_snapshot_root",
                    "execution_surface_hash",
                    "auditor",
                )
            }
            or endpoint.get("run") != {"path": str(run_dir)}
            or endpoint.get("formulation")
            != "e15_derived_direct_on_policy_canonical_maxent"
        ):
            raise SmokeApprovalError(f"{task}/{arm} endpoint audit is invalid")
        checkpoint = endpoint.get("checkpoint")
        data = endpoint.get("data")
        source = endpoint.get("source")
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("optimizer_updates") != 32
            or checkpoint.get("oat_step_tag") != 32
            or checkpoint.get("role") != "scheduled_update_boundary"
            or Path(str(checkpoint.get("path", ""))).resolve() != checkpoint_path
            or checkpoint_record
            != {
                "path": str(checkpoint_path),
                "weights_manifest_sha256": checkpoint.get(
                    "weights_manifest_sha256"
                ),
            }
        ):
            raise SmokeApprovalError(f"{task}/{arm} endpoint is not update 32")
        expected_data_hash = (
            EXPECTED_GRAPH_COMBINED_CONTENT_HASH
            if task == "graph_coloring"
            else EXPECTED_COUNTDOWN_COMBINED_CONTENT_HASH
        )
        if not isinstance(data, dict) or data.get("combined_content_hash") != expected_data_hash:
            raise SmokeApprovalError(f"{task}/{arm} endpoint uses different data")
        if not isinstance(source, dict) or source.get("python_source_sha256") != expected_identity["source_hash"]:
            raise SmokeApprovalError(f"{task}/{arm} endpoint uses different source")

        if (
            result.get("schema") != "e16_canonical_smoke_cell_result_v1"
            or result.get("protocol") != "E16"
            or result.get("stage") != "smoke"
            or result.get("status") != "pass"
            or result.get("task") != task
            or result.get("arm") != arm
            or result.get("seed") != seed
            or result.get("identity") != expected_identity
            or result.get("job_id") != job_id
            or result.get("run_stamp") != run_stamp
            or Path(str(result.get("run_dir", ""))).resolve() != run_dir
            or result.get("checkpoint") != checkpoint_record
            or result.get("campaign_identity") != campaign_binding
        ):
            raise SmokeApprovalError(f"{task}/{arm} smoke result identity is invalid")
        endpoint_binding = result.get("endpoint_audit")
        if endpoint_binding != {
            "path": str(endpoint_path.resolve()),
            "sha256": _sha256_file(endpoint_path),
        }:
            raise SmokeApprovalError(f"{task}/{arm} result binds another endpoint")
        checks = result.get("checks")
        if (
            not isinstance(checks, dict)
            or set(checks) != REQUIRED_CHECKS
            or any(value is not True for value in checks.values())
        ):
            raise SmokeApprovalError(f"{task}/{arm} did not pass every smoke check")
        if result_path.resolve() == endpoint_path.resolve():
            raise SmokeApprovalError(f"{task}/{arm} result and audit must be separate")
        evidence = result.get("evidence")
        slurm = result.get("slurm")
        if not isinstance(evidence, dict) or not isinstance(slurm, dict):
            raise SmokeApprovalError(f"{task}/{arm} lacks replayable raw evidence")
        expected_evidence = {
            "checkpoint_config",
            "endpoint_audit",
            "identity",
            "manifest",
            "metrics",
            "stdout",
        }
        if set(evidence) != expected_evidence:
            raise SmokeApprovalError(f"{task}/{arm} has an incomplete evidence set")
        for label in expected_evidence:
            record = evidence[label]
            if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
                raise SmokeApprovalError(f"{task}/{arm} {label} binding is malformed")
            evidence_path = Path(str(record["path"]))
            if not evidence_path.is_file() or _sha256_file(evidence_path) != record["sha256"]:
                raise SmokeApprovalError(f"{task}/{arm} {label} evidence changed")
        if evidence["endpoint_audit"] != endpoint_binding:
            raise SmokeApprovalError(f"{task}/{arm} result evidence binds another endpoint")
        if (
            campaign_binding.get("path") != evidence["identity"]["path"]
            or campaign_binding.get("sha256") != evidence["identity"]["sha256"]
        ):
            raise SmokeApprovalError(f"{task}/{arm} campaign identity is inconsistent")
        verifier = campaign_binding.get("approval_verifier")
        if (
            not isinstance(verifier, dict)
            or verifier.get("path") != str(Path(__file__).resolve())
            or verifier.get("sha256") != _sha256_file(Path(__file__).resolve())
        ):
            raise SmokeApprovalError("approval was not replayed by the frozen verifier")
        for label in ("endpoint_audit", "metrics", "stdout", "checkpoint_config"):
            resolved = str(Path(evidence[label]["path"]).resolve())
            if label != "endpoint_audit" and resolved in unique_paths[label]:
                raise SmokeApprovalError(f"cell-specific {label} path was reused")
            unique_paths[label].add(resolved)
        for label in ("stdout",):
            digest = str(evidence[label]["sha256"])
            if digest in unique_hashes[label]:
                raise SmokeApprovalError(f"cell-specific {label} content was reused")
            unique_hashes[label].add(digest)
        for label in ("identity", "manifest"):
            resolved = str(Path(evidence[label]["path"]).resolve())
            share_key = (label, resolved)
            prior_task = shared_evidence_tasks.setdefault(share_key, task)
            if prior_task != task:
                raise SmokeApprovalError(f"{label} evidence was reused across tasks")
            binding = (resolved, str(evidence[label]["sha256"]))
            expected_binding = task_shared_bindings.setdefault(
                (task, label), binding
            )
            if binding != expected_binding:
                raise SmokeApprovalError(
                    f"{task} arms do not share one exact {label} binding"
                )
        try:
            try:  # package import
                from .check_e16_canonical_smoke import validate_cell
            except ImportError:  # direct script execution
                from check_e16_canonical_smoke import validate_cell
            replayed = validate_cell(
                task=task,
                arm=arm,
                identity_path=Path(evidence["identity"]["path"]),
                manifest_path=Path(evidence["manifest"]["path"]),
                metrics_path=Path(evidence["metrics"]["path"]),
                stdout_path=Path(evidence["stdout"]["path"]),
                endpoint_path=Path(evidence["endpoint_audit"]["path"]),
                run_dir=run_dir,
                job_id=job_id,
                slurm_state=str(slurm.get("state")),
                slurm_exit_code=str(slurm.get("exit_code")),
            )
        except (KeyError, ValueError) as error:
            raise SmokeApprovalError(
                f"{task}/{arm} raw-evidence replay failed: {error}"
            ) from error
        if replayed != result:
            raise SmokeApprovalError(
                f"{task}/{arm} result differs from raw-evidence replay"
            )
    expected_cells = {
        (task, arm, 9006) for task in TASKS for arm in ARMS
    }
    if observed_cells != expected_cells:
        raise SmokeApprovalError("smoke approval does not cover the full 2 x 3 grid")
    return {
        "all_six_replayed": True,
        "approval": str(approval_path.resolve()),
        "approval_sha256": observed_approval_hash,
        "cells": len(observed_cells),
        "identity": expected_identity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--approval-sha256", required=True)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--e15-outcome-sha256", required=True)
    parser.add_argument("--dataset-identity-sha256", required=True)
    parser.add_argument("--runtime-identity-sha256", required=True)
    parser.add_argument("--smoke-config-digest", required=True)
    parser.add_argument("--full-config-digest", required=True)
    parser.add_argument("--execution-surface-hash", required=True)
    args = parser.parse_args()
    identity = {
        "dataset_identity_sha256": args.dataset_identity_sha256,
        "e15_outcome_sha256": args.e15_outcome_sha256,
        "execution_surface_hash": args.execution_surface_hash,
        "full_config_digest": args.full_config_digest,
        "protocol_sha256": args.protocol_sha256,
        "runtime_identity_sha256": args.runtime_identity_sha256,
        "smoke_config_digest": args.smoke_config_digest,
        "source_hash": args.source_hash,
    }
    try:
        summary = verify_smoke_approval(
            approval_path=args.approval,
            expected_approval_sha256=args.approval_sha256,
            expected_identity=identity,
        )
    except (FileNotFoundError, KeyError, SmokeApprovalError) as error:
        raise SystemExit(f"E16 full authorization rejected: {error}") from error
    print(json.dumps(summary, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
