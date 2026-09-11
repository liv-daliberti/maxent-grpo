#!/usr/bin/env python3
"""Fail-closed kernel-isolation smoke for ConstructiveCode candidates."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import signal
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oat_drgrpo.constructive_code_sandbox import (  # noqa: E402
    CandidateResult,
    PINNED_IMAGE_BYTES,
    PINNED_IMAGE_SHA256,
    SandboxLimits,
    build_launcher,
    percentile,
    prepare_runtime,
    run_candidate,
    sha256_file,
)


DEFAULT_IMAGE = ROOT / "var/images/python-3.10-slim-c1e4e6c01eb4.sqsh"
DEFAULT_OUTPUT = ROOT / "var/artifacts/constructive_code_isolation_smoke.json"
DEFAULT_SOURCE = ROOT / "ops/constructive_code_sandbox.c"
BASE_LIMITS = SandboxLimits(
    cpu_seconds=1,
    wall_seconds=3.0,
    memory_bytes=128 * 1024 * 1024,
    output_bytes=64 * 1024,
    file_count=32,
)
THROUGHPUT_REPEATS = 24
MAX_MEDIAN_LAUNCH_SECONDS = 0.25
MAX_P95_LAUNCH_SECONDS = 0.50

BASE_PROBE = r"""
import json
import os
from pathlib import Path

result = {
    "python_version": os.sys.version.split()[0],
    "hash_seed": os.environ.get("PYTHONHASHSEED"),
    "locale": os.environ.get("LC_ALL"),
    "timezone": os.environ.get("TZ"),
    "path": os.environ.get("PATH"),
    "sentinel_secret_visible": "CONSTRUCTIVE_CODE_SENTINEL_SECRET" in os.environ,
}
for label, raw_path in {
    "repository": "/n/fs/similarity/maxent-grpo/paper/main.tex",
    "home_credentials": "/u/od2961/.ssh",
    "host_etc": "/etc/passwd",
}.items():
    try:
        Path(raw_path).read_bytes()
    except OSError:
        result[f"host_{label}_absent"] = True
    else:
        result[f"host_{label}_absent"] = False
try:
    Path("/constructive-code-host-write-probe").write_text("forbidden")
except OSError:
    result["host_root_write_blocked"] = True
else:
    result["host_root_write_blocked"] = False
probe = Path("probe.txt")
probe.write_text("ok", encoding="ascii")
result["fresh_tmp_writable"] = probe.read_text(encoding="ascii") == "ok"
print(json.dumps(result, sort_keys=True))
"""

NETWORK_PROBE = r"""
import errno
import socket
try:
    socket.socket()
except OSError as error:
    print("blocked" if error.errno == errno.EACCES else f"wrong:{error.errno}")
else:
    print("open")
"""

PROCESS_PROBE = r"""
import errno
import subprocess
try:
    subprocess.run(["/bin/true"], check=False)
except OSError as error:
    print("blocked" if error.errno == errno.EACCES else f"wrong:{error.errno}")
else:
    print("open")
"""

HOST_EXEC_PROBE = r"""
import errno
import os
try:
    os.execv("/bin/true", ["true"])
except OSError as error:
    print("blocked" if error.errno == errno.EACCES else f"wrong:{error.errno}")
"""

CROSS_PROCESS_PROBE = r"""
import errno
import os
import resource
checks = []
try:
    os.kill(os.getppid(), 0)
except OSError as error:
    checks.append(error.errno == errno.EACCES)
else:
    checks.append(False)
try:
    resource.prlimit(os.getppid(), resource.RLIMIT_NOFILE)
except OSError as error:
    checks.append(error.errno == errno.EACCES)
else:
    checks.append(False)
print("blocked" if all(checks) else "open")
"""

MEMORY_PROBE = r"""
try:
    bytearray(512 * 1024 * 1024)
except MemoryError:
    print("limited")
else:
    print("open")
"""

FILE_COUNT_PROBE = r"""
import errno
handles = []
try:
    for index in range(128):
        handles.append(open(f"file-{index}", "wb"))
except OSError as error:
    print("limited" if error.errno == errno.EMFILE else f"wrong:{error.errno}")
else:
    print("open")
"""

OUTPUT_PROBE = "import os\nos.write(1, b'x' * (128 * 1024))\n"
CPU_PROBE = "while True:\n    pass\n"
THROUGHPUT_PROBE = "values = sorted({3, 1, 2})\nprint(*values)\n"


def _landlock_abi() -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    result = int(libc.syscall(444, 0, 0, 1))
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def _result_payload(result: CandidateResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["stdout_bytes"] = len(result.stdout)
    payload["stderr_bytes"] = len(result.stderr)
    payload["stdout"] = result.stdout[:4096].decode("utf-8", errors="replace")
    payload["stderr"] = result.stderr[:4096].decode("utf-8", errors="replace")
    return payload


def _run(
    launcher: Path,
    runtime_root: Path,
    source: str,
    *,
    limits: SandboxLimits = BASE_LIMITS,
) -> CandidateResult:
    return run_candidate(
        launcher=launcher,
        runtime_root=runtime_root,
        source=source,
        limits=limits,
        scratch_root=runtime_root.parent,
        runtime_is_preverified=True,
    )


def run_smoke(
    *,
    image: Path,
    launcher_source: Path,
    launcher: Path,
    runtime_root: Path,
) -> dict[str, Any]:
    violations: list[str] = []
    launcher_sha256 = build_launcher(launcher_source, launcher)
    runtime_identity = prepare_runtime(image, runtime_root)
    landlock_abi = _landlock_abi()
    if landlock_abi < 5:
        violations.append("landlock_abi_too_old")

    results = {
        "base": _run(launcher, runtime_root, BASE_PROBE),
        "network": _run(launcher, runtime_root, NETWORK_PROBE),
        "process": _run(launcher, runtime_root, PROCESS_PROBE),
        "host_execute": _run(launcher, runtime_root, HOST_EXEC_PROBE),
        "cross_process": _run(launcher, runtime_root, CROSS_PROCESS_PROBE),
        "memory": _run(launcher, runtime_root, MEMORY_PROBE),
        "file_count": _run(launcher, runtime_root, FILE_COUNT_PROBE),
        "output": _run(launcher, runtime_root, OUTPUT_PROBE),
        "cpu": _run(launcher, runtime_root, CPU_PROBE),
    }

    try:
        base = json.loads(results["base"].stdout)
    except json.JSONDecodeError:
        base = {}
        violations.append("base_probe_invalid_json")
    required_base = {
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
    for key, expected in required_base.items():
        if base.get(key) != expected:
            violations.append(f"base_{key}_failed")
    for name in ("base", "network", "process", "host_execute", "cross_process"):
        if not results[name].completed_cleanly:
            violations.append(f"{name}_probe_nonzero")
    for name in ("network", "process", "host_execute", "cross_process"):
        if results[name].stdout.strip() != b"blocked":
            violations.append(f"{name}_boundary_open")
    if results["memory"].stdout.strip() != b"limited":
        violations.append("memory_limit_failed")
    if results["file_count"].stdout.strip() != b"limited":
        violations.append("file_count_limit_failed")
    if not results["output"].output_limited:
        violations.append("output_limit_failed")
    cpu_limited = (
        results["cpu"].timed_out
        or results["cpu"].returncode in {-signal.SIGKILL, -signal.SIGXCPU}
    )
    if not cpu_limited:
        violations.append("cpu_or_wall_limit_failed")

    throughput_results = [
        _run(launcher, runtime_root, THROUGHPUT_PROBE)
        for _ in range(THROUGHPUT_REPEATS)
    ]
    if not all(result.completed_cleanly for result in throughput_results):
        violations.append("throughput_probe_nonzero")
    durations = [result.wall_seconds for result in throughput_results]
    median_seconds = statistics.median(durations)
    p95_seconds = percentile(durations, 0.95)
    if median_seconds > MAX_MEDIAN_LAUNCH_SECONDS:
        violations.append("median_launch_latency_exceeded")
    if p95_seconds > MAX_P95_LAUNCH_SECONDS:
        violations.append("p95_launch_latency_exceeded")

    return {
        "schema_version": "constructive-code-isolation-smoke-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not violations else "fail",
        "decision_boundary": (
            "This gate establishes pinned-runtime identity, candidate containment, "
            "resource ceilings, and launch latency. Released-checker equivalence "
            "and full reward-worker throughput remain separate fail-closed gates."
        ),
        "kernel": {
            "release": platform.release(),
            "machine": platform.machine(),
            "landlock_abi": landlock_abi,
            "seccomp_policy": "explicit escape-surface deny rules",
        },
        "runtime": {
            "image_path": str(image.resolve()),
            "image_sha256": runtime_identity.image_sha256,
            "image_bytes": runtime_identity.image_bytes,
            "expected_image_sha256": PINNED_IMAGE_SHA256,
            "expected_image_bytes": PINNED_IMAGE_BYTES,
            "oci_base_digest": runtime_identity.oci_base_digest,
            "critical_file_sha256": runtime_identity.critical_file_sha256,
        },
        "launcher": {
            "source_path": str(launcher_source.resolve()),
            "source_sha256": sha256_file(launcher_source),
            "binary_sha256": launcher_sha256,
        },
        "limits": asdict(BASE_LIMITS),
        "probes": {name: _result_payload(result) for name, result in results.items()},
        "throughput": {
            "repeats": THROUGHPUT_REPEATS,
            "median_launch_seconds": median_seconds,
            "p95_launch_seconds": p95_seconds,
            "maximum_median_launch_seconds": MAX_MEDIAN_LAUNCH_SECONDS,
            "maximum_p95_launch_seconds": MAX_P95_LAUNCH_SECONDS,
        },
        "violations": violations,
    }


def _failure_audit(error: BaseException) -> dict[str, Any]:
    return {
        "schema_version": "constructive-code-isolation-smoke-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "fail",
        "decision_boundary": "Infrastructure and setup errors fail closed.",
        "error": {"type": type(error).__name__, "message": str(error)},
        "violations": ["isolation_smoke_exception"],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--launcher-source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        audit = run_smoke(
            image=args.image,
            launcher_source=args.launcher_source,
            launcher=args.launcher,
            runtime_root=args.runtime_root,
        )
    except Exception as error:
        audit = _failure_audit(error)
    _write_json(args.output, audit)
    print(
        "[constructive-code-isolation] "
        f"status={audit['status']} "
        f"violations={len(audit['violations'])} "
        f"output={args.output}",
        flush=True,
    )
    raise SystemExit(0 if audit["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
