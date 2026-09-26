"""Client boundary for the persistent networkless AntMaze v18 worker."""

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


class AntMazeInteractiveProcessV18:
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
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            environment[key] = "1"
        source_root = str(Path(__file__).resolve().parents[1])
        entrypoint = (
            "import runpy,sys;"
            f"sys.path.insert(0,{source_root!r});"
            "runpy.run_module("
            "'oat_drgrpo.ant_maze_interactive_worker_v18',"
            "run_name='__main__')"
        )
        self.process = subprocess.Popen(
            [str(python), "-B", "-I", "-c", entrypoint],
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
            raise RuntimeError("interactive AntMaze v18 worker is unavailable")
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
                f"interactive AntMaze v18 worker exited without reply: {stderr}"
            )
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(
                "interactive AntMaze v18 worker rejected request: "
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
        return self._request({"command": "step_batch", "steps": list(steps)})[
            "results"
        ]

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._request({"command": "close"})
            finally:
                if self.process.stdin is not None:
                    self.process.stdin.close()
                self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)

    def __enter__(self) -> "AntMazeInteractiveProcessV18":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
