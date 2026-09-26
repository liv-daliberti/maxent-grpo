"""Hash the exact installed runtime used by maze execution workers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _file_receipt(path: Path, package_root: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(package_root.parent).as_posix(),
        "sha256": _sha256(path),
    }


def maze_runtime_identity() -> dict[str, Any]:
    """Return source-bound PointMaze and AntMaze environment identities."""

    import gymnasium
    import gymnasium_robotics

    robotics_root = Path(gymnasium_robotics.__file__).resolve().parent
    site_packages = robotics_root.parent
    gymnasium_root = Path(gymnasium.__file__).resolve().parent
    common = {
        "python": platform.python_version(),
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("gymnasium", "gymnasium-robotics", "mujoco", "numpy")
        },
        "maze_v4": _file_receipt(
            robotics_root / "envs/maze/maze_v4.py", site_packages
        ),
    }
    point_payload = {
        "schema_version": "point-maze-runtime-v1",
        "common": common,
        "point_maze": _file_receipt(
            robotics_root / "envs/maze/point_maze.py", site_packages
        ),
        "point_xml": _file_receipt(
            robotics_root / "envs/assets/point/point.xml", site_packages
        ),
    }
    ant_payload = {
        "schema_version": "ant-maze-runtime-v1",
        "common": common,
        "ant_maze": _file_receipt(
            robotics_root / "envs/maze/ant_maze_v5.py", site_packages
        ),
        "ant_v5": _file_receipt(
            gymnasium_root / "envs/mujoco/ant_v5.py", site_packages
        ),
        "ant_xml": _file_receipt(
            gymnasium_root / "envs/mujoco/assets/ant.xml", site_packages
        ),
    }
    return {
        "schema_version": "maze-runtime-identity-v1",
        "point": point_payload,
        "ant": ant_payload,
        "point_environment_sha256": _canonical_sha256(point_payload),
        "ant_environment_sha256": _canonical_sha256(ant_payload),
    }
