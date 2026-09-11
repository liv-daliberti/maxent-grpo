"""Killable external MuJoCo boundary for maze language-action programs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import select
import subprocess
import threading
from typing import Any, Mapping

from .maze_modebench import MazeValidation


class MazeVerifierProcess:
    """Execute maze action programs outside the trainer process."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 25.0,
        worker_python: str | Path | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        root = Path(
            os.environ.get(
                "OAT_ZERO_REPO_ROOT", Path(__file__).resolve().parents[2]
            )
        ).resolve()
        configured = worker_python or os.environ.get("OAT_ZERO_MAZE_WORKER_PYTHON")
        self.worker_python = Path(configured) if configured else (
            root / "var/maze_runtime/venv/bin/python"
        )
        self.timeout_seconds = float(timeout_seconds)
        self._process: subprocess.Popen[str] | None = None
        self._owner_pid = os.getpid()
        self._lock = threading.Lock()

    def _start(self) -> subprocess.Popen[str]:
        if self._owner_pid != os.getpid():
            self._process = None
            self._owner_pid = os.getpid()
        process = self._process
        if process is not None and process.poll() is None:
            return process
        if not self.worker_python.is_file():
            raise FileNotFoundError(f"maze worker Python is missing: {self.worker_python}")
        source_root = str(Path(__file__).resolve().parents[1])
        entrypoint = (
            "import runpy,sys;"
            f"sys.path.insert(0,{source_root!r});"
            "runpy.run_module('oat_drgrpo.maze_modebench_worker',run_name='__main__')"
        )
        worker_env = os.environ.copy()
        worker_env.setdefault("MUJOCO_GL", "egl")
        self._process = subprocess.Popen(
            [str(self.worker_python), "-B", "-I", "-c", entrypoint],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
            start_new_session=True,
            env=worker_env,
        )
        return self._process

    def _stop(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()

    def close(self) -> None:
        with self._lock:
            self._stop()

    def validate(
        self,
        candidate: str,
        spec: Mapping[str, Any],
    ) -> MazeValidation | None:
        result = self.validate_with_execution(candidate, spec)
        return result[0] if result is not None else None

    def validate_with_execution(
        self,
        candidate: str,
        spec: Mapping[str, Any],
    ) -> tuple[MazeValidation, dict[str, Any]] | None:
        """Return validation plus the trusted raw record for offline audits."""

        request = json.dumps(
            {"candidate": str(candidate), "spec": dict(spec)},
            allow_nan=False,
        )
        with self._lock:
            for attempt in range(2):
                try:
                    process = self._start()
                    assert process.stdin is not None and process.stdout is not None
                    process.stdin.write(request + "\n")
                    process.stdin.flush()
                    readable, _, _ = select.select(
                        [process.stdout], [], [], self.timeout_seconds
                    )
                    if not readable:
                        self._stop()
                        return None
                    line = process.stdout.readline()
                    if not line:
                        raise BrokenPipeError("maze worker closed stdout")
                    payload = json.loads(line)
                    if not payload.get("valid"):
                        return None
                    validation = MazeValidation(
                        canonical_key=str(payload["canonical_key"]),
                        directed_gates=tuple(
                            str(value) for value in payload["directed_gates"]
                        ),
                        action_tokens=tuple(
                            str(value) for value in payload["action_tokens"]
                        ),
                        simulator_steps=int(payload["simulator_steps"]),
                    )
                    return validation, dict(payload["execution"])
                except (
                    BrokenPipeError,
                    FileNotFoundError,
                    OSError,
                    ValueError,
                    json.JSONDecodeError,
                ):
                    self._stop()
                    if attempt:
                        return None
        return None

    def __del__(self) -> None:
        try:
            self._stop()
        except Exception:
            pass


_SHARED_VERIFIER = MazeVerifierProcess()


def validate_maze_action_program_external(
    candidate: str,
    spec: Mapping[str, Any],
) -> MazeValidation | None:
    """Validate a candidate through the shared pinned MuJoCo worker."""

    return _SHARED_VERIFIER.validate(candidate, spec)
