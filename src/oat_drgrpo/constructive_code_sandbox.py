"""Trusted supervisor for deterministic ConstructiveCode candidate execution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time


PINNED_IMAGE_SHA256 = "6d036dfa4a6e216d71e2ddae4cb673c0ff588d2c8ff8af3ed2ab7b3fed309437"
PINNED_IMAGE_BYTES = 44_068_864
PINNED_OCI_BASE_DIGEST = (
    "sha256:c1e4e6c01eb489c422288b2de34b0761ca316f7a2d98e2c33f47659a73ed108a"
)
CRITICAL_RUNTIME_FILES = {
    "usr/lib64/ld-linux-x86-64.so.2": (
        "438c546d8e8cc48496bf3a95f753051afd9db66a629a74e31a9ded71586b56e0"
    ),
    "usr/local/bin/python3.10": (
        "590a8c6d6f33dd13991b43285f0acb8999f4ce338eacbdda2faec0f25ca2a0b6"
    ),
    "usr/local/lib/libpython3.10.so.1.0": (
        "988df48b3ba1c6e2dec55b332c046fe3fec744f487b05cb686d08ed6336f2ab5"
    ),
}
SANDBOX_SETUP_RETURN_CODES = frozenset({120, 121, 122, 123, 124, 125})


@dataclass(frozen=True)
class SandboxLimits:
    cpu_seconds: int = 2
    wall_seconds: float = 4.0
    memory_bytes: int = 512 * 1024 * 1024
    output_bytes: int = 1024 * 1024
    file_count: int = 32
    source_bytes: int = 256 * 1024

    def __post_init__(self) -> None:
        values = (
            self.cpu_seconds,
            self.wall_seconds,
            self.memory_bytes,
            self.output_bytes,
            self.file_count,
            self.source_bytes,
        )
        if any(value <= 0 for value in values):
            raise ValueError("all sandbox limits must be positive")
        if self.wall_seconds < self.cpu_seconds:
            raise ValueError("wall limit must not be shorter than the CPU limit")


@dataclass(frozen=True)
class RuntimeIdentity:
    image_sha256: str
    image_bytes: int
    oci_base_digest: str
    critical_file_sha256: dict[str, str]


@dataclass(frozen=True)
class CandidateResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    wall_seconds: float
    timed_out: bool
    output_limited: bool
    sandbox_violation: bool

    @property
    def completed_cleanly(self) -> bool:
        return (
            self.returncode == 0
            and not self.timed_out
            and not self.output_limited
            and not self.sandbox_violation
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_runtime_root(root: Path) -> dict[str, str]:
    root = root.resolve()
    actual: dict[str, str] = {}
    for relative, expected in CRITICAL_RUNTIME_FILES.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"critical runtime file is missing: {path}")
        actual[relative] = sha256_file(path)
        if actual[relative] != expected:
            raise ValueError(f"critical runtime hash mismatch: {relative}")
    return actual


def prepare_runtime(image: Path, destination: Path) -> RuntimeIdentity:
    image = image.resolve()
    if not image.is_file():
        raise FileNotFoundError(f"pinned SquashFS image is missing: {image}")
    before_sha256 = sha256_file(image)
    if image.stat().st_size != PINNED_IMAGE_BYTES:
        raise ValueError("pinned SquashFS image byte size mismatch")
    if before_sha256 != PINNED_IMAGE_SHA256:
        raise ValueError("pinned SquashFS image hash mismatch")
    if destination.exists():
        raise FileExistsError(f"runtime extraction destination exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            shutil.which("unsquashfs") or "unsquashfs",
            "-no-progress",
            "-d",
            str(destination),
            str(image),
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=180,
    )
    after_sha256 = sha256_file(image)
    if after_sha256 != before_sha256:
        raise RuntimeError("pinned SquashFS image changed during extraction")
    critical = verify_runtime_root(destination)
    return RuntimeIdentity(
        image_sha256=before_sha256,
        image_bytes=image.stat().st_size,
        oci_base_digest=PINNED_OCI_BASE_DIGEST,
        critical_file_sha256=critical,
    )


def build_launcher(source: Path, output: Path) -> str:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"sandbox launcher source is missing: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            shutil.which("gcc") or "gcc",
            "-std=c17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(source),
            "-o",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    output.chmod(0o500)
    return sha256_file(output)


def _read_bounded(path: Path, limit: int) -> bytes:
    with path.open("rb") as handle:
        return handle.read(limit)


def run_candidate(
    *,
    launcher: Path,
    runtime_root: Path,
    source: str,
    stdin: bytes = b"",
    limits: SandboxLimits = SandboxLimits(),
    scratch_root: Path | None = None,
    runtime_is_preverified: bool = False,
) -> CandidateResult:
    source_payload = source.encode("utf-8")
    if len(source_payload) > limits.source_bytes:
        raise ValueError("candidate source exceeds the configured byte limit")
    launcher = launcher.resolve()
    runtime_root = runtime_root.resolve()
    if not launcher.is_file():
        raise FileNotFoundError(f"sandbox launcher is missing: {launcher}")
    if not runtime_is_preverified:
        verify_runtime_root(runtime_root)

    with tempfile.TemporaryDirectory(
        prefix="constructive-code-candidate-",
        dir=scratch_root,
    ) as raw_directory:
        directory = Path(raw_directory)
        workdir = directory / "work"
        workdir.mkdir(mode=0o700)
        program = workdir / "program.py"
        program.write_bytes(source_payload)
        program.chmod(0o400)
        stdin_path = directory / "stdin"
        stdout_path = directory / "stdout"
        stderr_path = directory / "stderr"
        stdin_path.write_bytes(stdin)

        command = [
            str(launcher),
            str(runtime_root),
            str(workdir),
            program.name,
            str(limits.cpu_seconds),
            str(limits.memory_bytes),
            str(limits.output_bytes),
            str(limits.file_count),
        ]
        started = time.monotonic()
        timed_out = False
        with (
            stdin_path.open("rb") as stdin_handle,
            stdout_path.open("wb") as stdout_handle,
            stderr_path.open("wb") as stderr_handle,
        ):
            process = subprocess.Popen(
                command,
                cwd=workdir,
                env={},
                stdin=stdin_handle,
                stdout=stdout_handle,
                stderr=stderr_handle,
                close_fds=True,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=limits.wall_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait(timeout=5)
        elapsed = time.monotonic() - started

        stdout_size = stdout_path.stat().st_size
        stderr_size = stderr_path.stat().st_size
        output_limited = (
            stdout_size >= limits.output_bytes
            or stderr_size >= limits.output_bytes
            or returncode == -signal.SIGXFSZ
        )
        sandbox_violation = (
            returncode == -signal.SIGSYS
            or returncode in SANDBOX_SETUP_RETURN_CODES
        )
        return CandidateResult(
            returncode=returncode,
            stdout=_read_bounded(stdout_path, limits.output_bytes),
            stderr=_read_bounded(stderr_path, limits.output_bytes),
            wall_seconds=elapsed,
            timed_out=timed_out,
            output_limited=output_limited,
            sandbox_violation=sandbox_violation,
        )


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must be in [0, 1]")
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def runtime_identity_json(identity: RuntimeIdentity) -> str:
    return json.dumps(
        {
            "critical_file_sha256": identity.critical_file_sha256,
            "image_bytes": identity.image_bytes,
            "image_sha256": identity.image_sha256,
            "oci_base_digest": identity.oci_base_digest,
        },
        sort_keys=True,
    )
