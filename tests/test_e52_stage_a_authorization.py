from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "ops/exp_scaling/verify_e52_sentinel_approval.py"
SPEC = importlib.util.spec_from_file_location(
    "verify_e52_sentinel_approval",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _approved_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict]:
    protocol = tmp_path / "protocol.md"
    repair = tmp_path / "repair.md"
    stability = tmp_path / "stability.md"
    launcher = tmp_path / "launcher.sh"
    auditor = tmp_path / "auditor.py"
    identity_path = tmp_path / "identity.json"
    source_root = tmp_path / "source" / "src"
    execution_root = tmp_path / "execution" / "ops"
    for path, text in (
        (protocol, "protocol\n"),
        (repair, "repair\n"),
        (stability, "stability\n"),
        (launcher, "launcher\n"),
        (auditor, "auditor\n"),
        (source_root / "module.py", "source\n"),
        (execution_root / "run.sh", "execution\n"),
    ):
        _write(path, text)

    source_hash = MODULE._hash_tree(source_root)
    execution_hash = MODULE._hash_tree(execution_root)
    monkeypatch.setattr(MODULE, "IDENTITY_PATH", identity_path)
    monkeypatch.setattr(MODULE, "PROTOCOL_PATH", protocol)
    monkeypatch.setattr(MODULE, "REPAIR_PROTOCOL_PATH", repair)
    monkeypatch.setattr(MODULE, "STABILITY_AMENDMENT_PATH", stability)
    monkeypatch.setattr(MODULE, "SENTINEL_LAUNCHER_PATH", launcher)
    monkeypatch.setattr(MODULE, "AUDITOR_PATH", auditor)

    identity = {
        "schema": "e52_direct_inverse_entropy_canonical_05b_sentinel_v2",
        "protocol_sha256": MODULE._sha256_file(protocol),
        "repair_protocol_sha256": MODULE._sha256_file(repair),
        "launcher_sha256": MODULE._sha256_file(launcher),
        "source_hash": source_hash,
        "execution_surface_hash": execution_hash,
    }
    _write(identity_path, json.dumps(identity))
    monkeypatch.setattr(
        MODULE,
        "ROOT",
        tmp_path,
    )
    expected_source_root = (
        tmp_path
        / "var/artifacts/source_snapshots"
        / f"e52_direct_inverse_entropy_{source_hash}"
        / "src"
    )
    expected_execution_root = (
        tmp_path
        / "var/artifacts/source_snapshots"
        / f"e52_direct_inverse_entropy_ops_{execution_hash}"
        / "ops"
    )
    expected_source_root.parent.mkdir(parents=True, exist_ok=True)
    expected_execution_root.parent.mkdir(parents=True, exist_ok=True)
    source_root.rename(expected_source_root)
    execution_root.rename(expected_execution_root)

    binding = {
        "identity_path": str(identity_path.resolve()),
        "source_snapshot_root": str(expected_source_root.resolve()),
        "execution_snapshot_root": str(expected_execution_root.resolve()),
        "identity_sha256": MODULE._sha256_file(identity_path),
        "protocol_sha256": MODULE._sha256_file(protocol),
        "repair_protocol_sha256": MODULE._sha256_file(repair),
        "stability_amendment_sha256": MODULE._sha256_file(stability),
        "sentinel_launcher_sha256": MODULE._sha256_file(launcher),
        "source_hash": MODULE._hash_tree(expected_source_root),
        "execution_surface_hash": MODULE._hash_tree(expected_execution_root),
        "auditor_sha256": MODULE._sha256_file(auditor),
    }
    run = {
        "status": "complete",
        "terminal_evaluation_present": True,
        "training_passes": 50.0,
        "violations": [],
    }
    domain = {
        "behavioral_gate": {"status": "pass"},
        "safety_gate": {"status": "pass"},
        "runs": {
            arm: dict(run)
            for arm in (
                "grpo",
                "maxent_inverse",
                "maxent_inverse_canonical",
            )
        },
    }
    approval = {
        "schema": "e52_sentinel_audit_v2",
        "status": "pass",
        "authorizes_stage_a": True,
        "violations": [],
        "approval_binding": binding,
        "domains": {
            domain_name: json.loads(json.dumps(domain))
            for domain_name in (
                "countdown",
                "graph_coloring",
                "python_factor",
            )
        },
    }
    approval_path = tmp_path / "approval.json"
    _write(approval_path, json.dumps(approval, sort_keys=True))
    return approval_path, approval


def test_exact_positive_terminal_approval_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    approval_path, _ = _approved_fixture(tmp_path, monkeypatch)

    summary = MODULE.verify_approval(
        approval_path=approval_path,
        expected_approval_sha256=MODULE._sha256_file(approval_path),
    )

    assert summary["approval_sha256"] == MODULE._sha256_file(approval_path)
    assert summary["source_hash"] == MODULE._hash_tree(
        Path(summary["source_root"])
    )


def test_nonterminal_or_failed_domain_cannot_authorize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    approval_path, approval = _approved_fixture(tmp_path, monkeypatch)
    approval["domains"]["python_factor"]["behavioral_gate"]["status"] = "fail"
    _write(approval_path, json.dumps(approval, sort_keys=True))

    with pytest.raises(MODULE.ApprovalError, match="python_factor behavioral"):
        MODULE.verify_approval(
            approval_path=approval_path,
            expected_approval_sha256=MODULE._sha256_file(approval_path),
        )


def test_bound_evidence_mutation_invalidates_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    approval_path, _ = _approved_fixture(tmp_path, monkeypatch)
    MODULE.STABILITY_AMENDMENT_PATH.write_text(
        "changed\n",
        encoding="utf-8",
    )

    with pytest.raises(MODULE.ApprovalError, match="changed after"):
        MODULE.verify_approval(
            approval_path=approval_path,
            expected_approval_sha256=MODULE._sha256_file(approval_path),
        )


def test_launcher_supplied_approval_hash_is_mandatory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    approval_path, _ = _approved_fixture(tmp_path, monkeypatch)

    with pytest.raises(MODULE.ApprovalError, match="reviewed hash"):
        MODULE.verify_approval(
            approval_path=approval_path,
            expected_approval_sha256="0" * 64,
        )
