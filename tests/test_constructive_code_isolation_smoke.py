from __future__ import annotations

import json
from pathlib import Path
import signal

import pytest

import smoke_constructive_code_isolation as isolation
from oat_drgrpo.constructive_code_sandbox import (
    CandidateResult,
    PINNED_IMAGE_BYTES,
    PINNED_IMAGE_SHA256,
    RuntimeIdentity,
    SandboxLimits,
    percentile,
    run_candidate,
)


def _result(
    stdout: bytes,
    *,
    returncode: int = 0,
    output_limited: bool = False,
) -> CandidateResult:
    return CandidateResult(
        returncode=returncode,
        stdout=stdout,
        stderr=b"",
        wall_seconds=0.01,
        timed_out=False,
        output_limited=output_limited,
        sandbox_violation=False,
    )


def _patch_successful_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(isolation, "build_launcher", lambda *_args: "launcher-sha")
    monkeypatch.setattr(isolation, "sha256_file", lambda _path: "source-sha")
    monkeypatch.setattr(isolation, "_landlock_abi", lambda: 6)
    monkeypatch.setattr(
        isolation,
        "prepare_runtime",
        lambda *_args: RuntimeIdentity(
            image_sha256=PINNED_IMAGE_SHA256,
            image_bytes=PINNED_IMAGE_BYTES,
            oci_base_digest="sha256:base",
            critical_file_sha256={"python": "sha"},
        ),
    )
    base = json.dumps(
        {
            "python_version": "3.10.20",
            "hash_seed": "0",
            "locale": "C.UTF-8",
            "timezone": "UTC",
            "path": "",
            "sentinel_secret_visible": False,
            "host_repository_absent": True,
            "host_home_credentials_absent": True,
            "host_host_etc_absent": True,
            "host_root_write_blocked": True,
            "fresh_tmp_writable": True,
        }
    ).encode()

    def fake_run(_launcher: Path, _root: Path, source: str, **_kwargs):
        if source == isolation.BASE_PROBE:
            return _result(base)
        if source in {
            isolation.NETWORK_PROBE,
            isolation.PROCESS_PROBE,
            isolation.HOST_EXEC_PROBE,
            isolation.CROSS_PROCESS_PROBE,
        }:
            return _result(b"blocked\n")
        if source in {isolation.MEMORY_PROBE, isolation.FILE_COUNT_PROBE}:
            return _result(b"limited\n")
        if source == isolation.OUTPUT_PROBE:
            return _result(b"x", output_limited=True)
        if source == isolation.CPU_PROBE:
            return _result(b"", returncode=-signal.SIGKILL)
        if source == isolation.THROUGHPUT_PROBE:
            return _result(b"1 2 3\n")
        raise AssertionError("unexpected smoke probe")

    monkeypatch.setattr(isolation, "_run", fake_run)


def test_isolation_smoke_contract_passes_all_boundaries(monkeypatch, tmp_path):
    _patch_successful_smoke(monkeypatch)

    audit = isolation.run_smoke(
        image=tmp_path / "image.sqsh",
        launcher_source=tmp_path / "sandbox.c",
        launcher=tmp_path / "sandbox",
        runtime_root=tmp_path / "root",
    )

    assert audit["status"] == "pass"
    assert audit["violations"] == []
    assert audit["kernel"]["landlock_abi"] == 6


def test_isolation_smoke_fails_closed_on_open_network(monkeypatch, tmp_path):
    _patch_successful_smoke(monkeypatch)
    original = isolation._run

    def open_network(launcher, runtime_root, source, **kwargs):
        if source == isolation.NETWORK_PROBE:
            return _result(b"open\n")
        return original(launcher, runtime_root, source, **kwargs)

    monkeypatch.setattr(isolation, "_run", open_network)
    audit = isolation.run_smoke(
        image=tmp_path / "image.sqsh",
        launcher_source=tmp_path / "sandbox.c",
        launcher=tmp_path / "sandbox",
        runtime_root=tmp_path / "root",
    )

    assert audit["status"] == "fail"
    assert "network_boundary_open" in audit["violations"]


def test_failure_audit_is_fail_closed():
    audit = isolation._failure_audit(RuntimeError("broken"))

    assert audit["status"] == "fail"
    assert audit["violations"] == ["isolation_smoke_exception"]
    assert audit["error"]["type"] == "RuntimeError"


def test_sandbox_limits_and_percentile_validation():
    with pytest.raises(ValueError, match="positive"):
        SandboxLimits(cpu_seconds=0)
    with pytest.raises(ValueError, match="shorter"):
        SandboxLimits(cpu_seconds=2, wall_seconds=1)
    assert percentile([0.4, 0.1, 0.2, 0.3], 0.75) == 0.3


def test_candidate_source_limit_fails_before_launch(tmp_path):
    with pytest.raises(ValueError, match="source exceeds"):
        run_candidate(
            launcher=tmp_path / "missing-launcher",
            runtime_root=tmp_path / "missing-runtime",
            source="xx",
            limits=SandboxLimits(source_bytes=1),
        )
