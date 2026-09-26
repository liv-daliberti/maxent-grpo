"""Client boundary for the persistent networkless PointMaze worker."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence


ROOT = Path(
    os.environ.get("OAT_ZERO_REPO_ROOT", Path(__file__).resolve().parents[2])
).resolve()
DEFAULT_WORKER_PYTHON = ROOT / "var/maze_runtime/venv/bin/python"


class PointMazeInteractiveProcess:
    """Persistent JSON-lines subprocess with fail-closed batch requests."""

    def __init__(self, worker_python: Path | None = None) -> None:
        python = Path(
            worker_python
            or os.environ.get(
                "OAT_ZERO_MAZE_WORKER_PYTHON",
                str(DEFAULT_WORKER_PYTHON),
            )
        )
        environment = os.environ.copy()
        environment["OAT_ZERO_REPO_ROOT"] = str(ROOT)
        # The worker runs under a separate maze runtime that does not have this
        # package installed, so the import path must be supplied explicitly
        # rather than inherited from whatever launched the parent.
        source_root = str(ROOT / "src")
        inherited = environment.get("PYTHONPATH", "")
        if source_root not in inherited.split(os.pathsep):
            environment["PYTHONPATH"] = (
                f"{source_root}{os.pathsep}{inherited}"
                if inherited
                else source_root
            )
        self.process = subprocess.Popen(
            [
                str(python),
                "-m",
                "oat_drgrpo.point_maze_interactive_worker",
            ],
            cwd=ROOT,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def _request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if (
            self.process.poll() is not None
            or self.process.stdin is None
            or self.process.stdout is None
        ):
            raise RuntimeError("interactive PointMaze worker is unavailable")
        self.process.stdin.write(json.dumps(payload, allow_nan=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            stderr = (
                self.process.stderr.read()
                if self.process.stderr is not None
                else ""
            )
            raise RuntimeError(
                f"interactive PointMaze worker exited without reply: {stderr}"
            )
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(
                "interactive PointMaze worker rejected request: "
                + str(response.get("error"))
            )
        return response

    def reset_batch(
        self,
        sessions: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        return self._request(
            {"command": "reset_batch", "sessions": list(sessions)}
        )["results"]

    def step_batch(
        self,
        steps: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        return self._request(
            {"command": "step_batch", "steps": list(steps)}
        )["results"]

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._request({"command": "close"})
            finally:
                if self.process.stdin is not None:
                    self.process.stdin.close()
                self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)

    def __enter__(self) -> "PointMazeInteractiveProcess":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
