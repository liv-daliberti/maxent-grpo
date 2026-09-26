#!/usr/bin/env python3
"""Verify that an E53 Stage-A approval is terminal and identity-bound."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
IDENTITY = ROOT / "var/artifacts/e53_verified_replay_05b_sentinel_identity.json"


class ApprovalError(RuntimeError):
    """Raised when the E53 sentinel approval is not launch-authoritative."""


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


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ApprovalError(f"{label} is unavailable or invalid: {error}") from error
    if not isinstance(payload, dict):
        raise ApprovalError(f"{label} must be a JSON object")
    return payload


def verify_approval(
    *,
    approval_path: Path,
    expected_approval_sha256: str | None = None,
) -> dict[str, Any]:
    approval = _json_object(approval_path, "E53 sentinel approval")
    identity = _json_object(IDENTITY, "E53 sentinel identity")
    approval_sha256 = _sha256_file(approval_path)
    if (
        expected_approval_sha256 is not None
        and approval_sha256 != expected_approval_sha256
    ):
        raise ApprovalError("E53 sentinel approval hash mismatch")
    if (
        approval.get("schema") != "e53_sentinel_audit_v2"
        or approval.get("status") != "pass"
        or approval.get("authorizes_stage_a") is not True
        or approval.get("violations") != []
    ):
        raise ApprovalError("E53 sentinel audit is not an unqualified pass")
    binding = approval.get("approval_binding")
    if not isinstance(binding, dict):
        raise ApprovalError("E53 sentinel approval lacks its binding")
    if binding.get("identity_sha256") != _sha256_file(IDENTITY):
        raise ApprovalError("E53 sentinel approval is bound to another identity")
    repair_auditor = ROOT / "ops/exp_scaling/audit_e53_sentinel_v2.py"
    repair_protocol = (
        ROOT
        / "paper/preregistration/e53_runtime_audit_scaling_repair_20260726.md"
    )
    if (
        binding.get("bound_base_auditor_sha256")
        != identity.get("auditor_sha256")
        or binding.get("runtime_repair_auditor_sha256")
        != _sha256_file(repair_auditor)
        or binding.get("runtime_repair_protocol_sha256")
        != _sha256_file(repair_protocol)
    ):
        raise ApprovalError("E53 runtime-audit repair binding mismatch")
    source_root = Path(str(binding.get("source_snapshot_root", "")))
    execution_root = Path(str(binding.get("execution_snapshot_root", "")))
    if _hash_tree(source_root) != identity.get("source_hash"):
        raise ApprovalError("E53 source snapshot no longer matches the identity")
    if _hash_tree(execution_root) != identity.get("execution_surface_hash"):
        raise ApprovalError("E53 execution snapshot no longer matches the identity")
    for domain in ("countdown", "graph_coloring", "python_factor"):
        payload = approval.get("domains", {}).get(domain, {})
        if (
            payload.get("behavioral_gate", {}).get("status") != "pass"
            or payload.get("safety_gate", {}).get("status") != "pass"
            or any(
                run.get("status") != "complete"
                or run.get("checkpoint_gate", {}).get("status") != "pass"
                for run in payload.get("runs", {}).values()
            )
        ):
            raise ApprovalError(f"E53 approval domain is incomplete: {domain}")
    return {
        "approval_sha256": approval_sha256,
        "identity_sha256": _sha256_file(IDENTITY),
        "source_root": str(source_root),
        "execution_root": str(execution_root),
        "source_hash": str(identity["source_hash"]),
        "execution_surface_hash": str(identity["execution_surface_hash"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--approval-sha256")
    args = parser.parse_args()
    result = verify_approval(
        approval_path=args.approval,
        expected_approval_sha256=args.approval_sha256,
    )
    print(
        "[e53-approval] verified terminal sentinel pass "
        f"approval_sha256={result['approval_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
