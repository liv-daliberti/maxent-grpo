"""CPU-only seeded-launcher regressions; fixtures are never evaluation evidence."""
import os
from pathlib import Path
import subprocess

import pytest

import probe_constructive_code_seeded_20260922 as probe


def test_seeded_changes_preserve_original_kernel_isolation():
    probe.check_kernel_source_unchanged(probe.SOURCE, probe.ORIGINAL)


def test_unrelated_security_change_is_rejected(tmp_path):
    changed = tmp_path / "changed.c"
    changed.write_text(probe.SOURCE.read_text().replace("DENY_SYSCALL(socket),", "/* removed */"))
    with pytest.raises(ValueError, match="kernel isolation"):
        probe.check_kernel_source_unchanged(changed, probe.ORIGINAL)


def test_same_execution_signature_passes_production_replay_gate():
    row = probe.replay_control("fixed output", "fixed output")
    assert row["accepted"] and row["stability_recheck_required"]
    assert not row["hard_violations"]


def test_changed_execution_signature_is_a_hard_failure():
    row = probe.replay_control("first output", "different output")
    assert not row["accepted"] and row["canonical_key"] is None
    assert row["hard_violations"] == ["accepted program changed canonical mode on independent full-suite recheck"]


def test_actual_pinned_runtime_controls(tmp_path):
    raw = os.environ.get("SEEDED_SANDBOX_RUNTIME_ROOT")
    if raw is None:
        pytest.skip("set SEEDED_SANDBOX_RUNTIME_ROOT to execute actual pinned Python CPU controls")
    runtime = Path(raw)
    assert (runtime / "usr/local/bin/python3.10").exists()
    binary = tmp_path / "sandbox"
    subprocess.run(["cc", "-O2", "-Wall", "-Wextra", "-Werror", "-o", str(binary), str(probe.SOURCE)], check=True)
    observations = probe.run_controls(probe.SandboxProbe(binary, runtime))
    assert observations["production_replay_gate_control_fixture"]["changed"]["accepted"] is False
