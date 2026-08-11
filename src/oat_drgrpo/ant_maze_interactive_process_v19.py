"""Client boundary for the persistent networkless AntMaze v19 worker."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

from .ant_maze_interactive_process_v18 import AntMazeInteractiveProcessV18


class AntMazeInteractiveProcessV19(AntMazeInteractiveProcessV18):
    def _start(self) -> subprocess.Popen[str]:
        if self._owner_pid != os.getpid():
            self._process = None
            self._owner_pid = os.getpid()
        process = self._process
        if process is not None and process.poll() is None:
            return process
        if not self.worker_python.is_file():
            raise FileNotFoundError(
                f"maze worker Python is missing: {self.worker_python}"
            )
        source_root = str(Path(__file__).resolve().parents[1])
        entrypoint = (
            "import runpy,sys;"
            f"sys.path.insert(0,{source_root!r});"
            "runpy.run_module("
            "'oat_drgrpo.ant_maze_interactive_worker_v19',"
            "run_name='__main__')"
        )
        environment = os.environ.copy()
        environment.setdefault("MUJOCO_GL", "egl")
        self._process = subprocess.Popen(
            [str(self.worker_python), "-B", "-I", "-c", entrypoint],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            start_new_session=True,
            env=environment,
        )
        return self._process
