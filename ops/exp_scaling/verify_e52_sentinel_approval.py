#!/usr/bin/env python3
"""Replay E52's exact terminal sentinel approval before Stage A."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_APPROVAL = (
    ROOT / "var/artifacts/e52_sentinel_stage_a_approval.json"
)
IDENTITY_PATH = (
    ROOT
    / "var/artifacts/"
    "e52_direct_inverse_entropy_canonical_05b_sentinel_v2_identity.json"
)
PROTOCOL_PATH = (
    ROOT
    / "paper/preregistration/"
    "e52_direct_inverse_entropy_canonical_05b.md"
)
REPAIR_PROTOCOL_PATH = (
    ROOT
    / "paper/preregistration/"
    "e52_runtime_validation_repair_20260726.md"
)
STABILITY_AMENDMENT_PATH = (
    ROOT
    / "paper/preregistration/"
    "e52_scale_free_stability_gate_amendment_20260726.md"
)
SENTINEL_LAUNCHER_PATH = (
    ROOT
    / "ops/exp_scaling/"
    "launch_e52_direct_inverse_entropy_canonical_05b.sh"
)
AUDITOR_PATH = ROOT / "ops/exp_scaling/audit_e52_sentinel.py"
DOMAINS = {"countdown", "graph_coloring", "python_factor"}
ARMS = {"grpo", "maxent_inverse", "maxent_inverse_canonical"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ApprovalError(RuntimeError):
    """Raised when the terminal sentinel cannot authorize Stage A."""


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
        key=lambda path: path.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        lines.append(f"{_sha256_file(path)}  ./{relative}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ApprovalError(f"{label} is unavailable or invalid: {error}") from error
    if not isinstance(payload, dict):
        raise ApprovalError(f"{label} must be a JSON object")
    return payload


def _require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ApprovalError(f"{label} is not a lowercase SHA-256 digest")
    return value


def verify_approval(
    *,
    approval_path: Path,
    expected_approval_sha256: str,
) -> dict[str, Any]:
    expected_approval_sha256 = _require_digest(
        expected_approval_sha256,
        "expected approval hash",
    )
    observed_approval_sha256 = _sha256_file(approval_path)
    if observed_approval_sha256 != expected_approval_sha256:
        raise ApprovalError(
            "sentinel approval does not match the launcher's reviewed hash"
        )
    approval = _json_object(approval_path, "sentinel approval")
    if (
        approval.get("schema") != "e52_sentinel_audit_v2"
        or approval.get("status") != "pass"
        or approval.get("authorizes_stage_a") is not True
        or approval.get("violations") != []
    ):
        raise ApprovalError("sentinel approval identity/status is not positive")

    domains = approval.get("domains")
    if not isinstance(domains, dict) or set(domains) != DOMAINS:
        raise ApprovalError("sentinel approval does not cover exactly three domains")
    for domain, payload in domains.items():
        if not isinstance(payload, dict):
            raise ApprovalError(f"{domain} approval payload is invalid")
        if payload.get("behavioral_gate", {}).get("status") != "pass":
            raise ApprovalError(f"{domain} behavioral gate did not pass")
        if payload.get("safety_gate", {}).get("status") != "pass":
            raise ApprovalError(f"{domain} safety gate did not pass")
        runs = payload.get("runs")
        if not isinstance(runs, dict) or set(runs) != ARMS:
            raise ApprovalError(f"{domain} approval lacks the exact three arms")
        for arm, run in runs.items():
            if (
                not isinstance(run, dict)
                or run.get("status") != "complete"
                or run.get("terminal_evaluation_present") is not True
                or float(run.get("training_passes", 0.0)) < 50.0
                or run.get("violations") != []
            ):
                raise ApprovalError(
                    f"{domain}/{arm} lacks exact clean terminal evidence"
                )

    binding = approval.get("approval_binding")
    if not isinstance(binding, dict):
        raise ApprovalError("sentinel approval lacks its immutable binding")
    identity = _json_object(IDENTITY_PATH, "sentinel identity")
    if (
        identity.get("schema")
        != "e52_direct_inverse_entropy_canonical_05b_sentinel_v2"
    ):
        raise ApprovalError("sentinel identity schema is incompatible")

    source_hash = _require_digest(identity.get("source_hash"), "source hash")
    execution_hash = _require_digest(
        identity.get("execution_surface_hash"),
        "execution-surface hash",
    )
    source_root = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e52_direct_inverse_entropy_{source_hash}"
        / "src"
    )
    execution_root = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e52_direct_inverse_entropy_ops_{execution_hash}"
        / "ops"
    )
    observed_binding = {
        "identity_path": str(IDENTITY_PATH.resolve()),
        "source_snapshot_root": str(source_root.resolve()),
        "execution_snapshot_root": str(execution_root.resolve()),
        "identity_sha256": _sha256_file(IDENTITY_PATH),
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "repair_protocol_sha256": _sha256_file(REPAIR_PROTOCOL_PATH),
        "stability_amendment_sha256": _sha256_file(
            STABILITY_AMENDMENT_PATH
        ),
        "sentinel_launcher_sha256": _sha256_file(
            SENTINEL_LAUNCHER_PATH
        ),
        "source_hash": _hash_tree(source_root),
        "execution_surface_hash": _hash_tree(execution_root),
        "auditor_sha256": _sha256_file(AUDITOR_PATH),
    }
    if binding != observed_binding:
        raise ApprovalError("sentinel evidence changed after terminal approval")
    identity_expectations = {
        "protocol_sha256": observed_binding["protocol_sha256"],
        "repair_protocol_sha256": observed_binding[
            "repair_protocol_sha256"
        ],
        "launcher_sha256": observed_binding["sentinel_launcher_sha256"],
        "source_hash": observed_binding["source_hash"],
        "execution_surface_hash": observed_binding[
            "execution_surface_hash"
        ],
    }
    for key, expected in identity_expectations.items():
        if identity.get(key) != expected:
            raise ApprovalError(f"sentinel identity mismatch for {key}")

    return {
        "approval": str(approval_path.resolve()),
        "approval_sha256": observed_approval_sha256,
        "identity_sha256": observed_binding["identity_sha256"],
        "source_root": observed_binding["source_snapshot_root"],
        "source_hash": observed_binding["source_hash"],
        "execution_root": observed_binding["execution_snapshot_root"],
        "execution_surface_hash": observed_binding[
            "execution_surface_hash"
        ],
        "stability_amendment_sha256": observed_binding[
            "stability_amendment_sha256"
        ],
        "auditor_sha256": observed_binding["auditor_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval", type=Path, default=DEFAULT_APPROVAL)
    parser.add_argument("--approval-sha256", required=True)
    args = parser.parse_args()
    try:
        summary = verify_approval(
            approval_path=args.approval,
            expected_approval_sha256=args.approval_sha256,
        )
    except ApprovalError as error:
        raise SystemExit(f"E52 Stage-A approval rejected: {error}") from error
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
