"""Client boundary for the persistent networkless PointMaze waypoint worker."""

from __future__ import annotations

import json
import os
from pathlib import Path
import select
import subprocess
from typing import Any, Mapping, Sequence


ROOT = Path(
    os.environ.get("OAT_ZERO_REPO_ROOT", Path(__file__).resolve().parents[2])
).resolve()
DEFAULT_WORKER_PYTHON = ROOT / "var/maze_runtime/venv/bin/python"

SOURCE_ROOT = Path(os.environ.get("OAT_ZERO_SOURCE_ROOT", ROOT / "src")).resolve()


class PointMazeWaypointProcess:
    """Persistent JSON-lines client with bounded, fail-closed requests."""

    def __init__(
        self,
        worker_python: str | Path | None = None,
        *,
        timeout_seconds: float = 130.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        python = Path(
            worker_python
            or os.environ.get(
                "OAT_ZERO_MAZE_WORKER_PYTHON",
                str(DEFAULT_WORKER_PYTHON),
            )
        )
        environment = os.environ.copy()
        environment.setdefault("MUJOCO_GL", "egl")
        environment["OAT_ZERO_REPO_ROOT"] = str(ROOT)
        source_root = str(SOURCE_ROOT)
        inherited = environment.get("PYTHONPATH", "")
        if source_root not in inherited.split(os.pathsep):
            environment["PYTHONPATH"] = (
                f"{source_root}{os.pathsep}{inherited}" if inherited else source_root
            )
        self.timeout_seconds = float(timeout_seconds)
        self.process = subprocess.Popen(
            [str(python), "-m", "oat_drgrpo.point_maze_waypoint_worker"],
            cwd=ROOT,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

    def _request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if (
            self.process.poll() is not None
            or self.process.stdin is None
            or self.process.stdout is None
        ):
            raise RuntimeError("interactive Point waypoint worker is unavailable")
        self.process.stdin.write(json.dumps(payload, allow_nan=False) + "\n")
        self.process.stdin.flush()
        readable, _, _ = select.select(
            [self.process.stdout], [], [], self.timeout_seconds
        )
        if not readable:
            self.process.kill()
            self.process.wait(timeout=5)
            raise TimeoutError("interactive Point waypoint worker timed out")
        line = self.process.stdout.readline()
        if not line:
            stderr = (
                self.process.stderr.read() if self.process.stderr is not None else ""
            )
            raise RuntimeError(
                "interactive Point waypoint worker exited without reply: " + stderr
            )
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(
                "interactive Point waypoint worker rejected request: "
                + str(response.get("error"))
            )
        return response

    def reset_batch(
        self,
        sessions: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        return self._request({"command": "reset_batch", "sessions": list(sessions)})[
            "results"
        ]

    def step_batch(
        self,
        steps: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        return self._request({"command": "step_batch", "steps": list(steps)})["results"]

    def abort_batch(self, session_ids: Sequence[str]) -> None:
        self._request({"command": "abort_batch", "session_ids": list(session_ids)})

    def execute_program(
        self,
        *,
        session_id: str,
        spec: Mapping[str, Any],
        actions: Sequence[str],
    ) -> dict[str, Any]:
        """Execute one logical program through the same stepwise boundary."""

        reset = self.reset_batch([{"session_id": session_id, "spec": dict(spec)}])[0]
        result = reset
        consumed = 0
        try:
            for action in actions:
                if result.get("done"):
                    raise RuntimeError(
                        "waypoint program contains actions after terminal"
                    )
                result = self.step_batch(
                    [{"session_id": session_id, "action": str(action)}]
                )[0]
                consumed += 1
            if not result.get("done"):
                raise RuntimeError("waypoint program ended before a terminal state")
            return result
        finally:
            if consumed < len(actions) or not result.get("done"):
                try:
                    self.abort_batch([session_id])
                except Exception:
                    pass

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._request({"command": "close"})
            except Exception:
                pass
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)

    def __enter__(self) -> "PointMazeWaypointProcess":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
