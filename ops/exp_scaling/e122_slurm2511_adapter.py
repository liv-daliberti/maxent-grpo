#!/usr/bin/env python3
"""Additive, pinned Slurm 25.11 display adapter for the frozen E122 campaign.

The sole accepted display amendment is ``NumNodes=1-1`` to ``NumNodes=1``
for the original strict held-job audit. Raw scheduler evidence is retained.
The CLI uses the unchanged release controller with an amendment-bound journal.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve()
TEST_SOURCE = ROOT / "tests/test_e122_slurm2511_adapter.py"
PLAN_PATH = ROOT / "var/artifacts/e122_level3_factorial_plan.json"
PLAN_SHA256 = "67c506a40b9a7fb7b984335d2ac47c859e01c9a62e5eddba5891aca9401cec2f"
LAUNCHER_PATH = ROOT / "ops/exp_scaling/launch_e122_level3_factorial.py"
CONTROLLER_PATH = ROOT / "ops/exp_scaling/control_e122_level3_release.py"
FROZEN_SOURCES = {
    str(LAUNCHER_PATH): "bdf72437c4971361c54982726c4a49e0ef04daf2d3d9b8e9ac01edfb5b8f85cd",
    str(CONTROLLER_PATH): "e123029bf55dde97d87eced134ee1b5db2c3dc4c4dd10c57f47200113a6f43c1",
}
NORMALIZATION = {"field": "NumNodes", "from": "1-1", "to": "1",
                 "semantics": "minimum_1_maximum_1_exact_one_node"}
NODE_FIELD = re.compile(r"(?<!\S)NumNodes=([^\s]+)(?=\s|$)")


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned_json(path: Path, expected: str) -> dict[str, Any]:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("explicit lowercase SHA256 pin is required")
    contents = Path(path).read_bytes()
    if hashlib.sha256(contents).hexdigest() != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    result = json.loads(contents)
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object: {path}")
    return result


def authenticate_amendment(path: Path, sha: str) -> dict[str, Any]:
    """Authenticate the additive document and every pin, without scheduler I/O."""
    path = Path(path).resolve()
    amendment = pinned_json(path, sha)
    if amendment.get("schema") != "e122_slurm_2511_display_amendment_v1":
        raise ValueError("unexpected Slurm display amendment schema")
    if amendment.get("normalization") != NORMALIZATION:
        raise ValueError("only the exact single-node display amendment is permitted")
    if (amendment.get("model_choice") != "05b"
            or amendment.get("plan_sha256") != PLAN_SHA256
            or Path(amendment.get("plan_path", "")).resolve() != PLAN_PATH):
        raise ValueError("amendment differs from the frozen E122 Qwen-0.5B plan")
    plan = pinned_json(PLAN_PATH, PLAN_SHA256)
    if plan.get("model_choice") != "05b":
        raise ValueError("frozen plan model choice differs")
    pins = amendment.get("files_sha256")
    if not isinstance(pins, dict):
        raise ValueError("amendment requires a files_sha256 mapping")
    required = {str(SOURCE), str(TEST_SOURCE), str(PLAN_PATH), *FROZEN_SOURCES}
    if not required.issubset(pins):
        raise ValueError("amendment lacks required original or additive source pins")
    if pins[str(PLAN_PATH)] != PLAN_SHA256:
        raise ValueError("amendment plan file pin differs")
    for name, expected in FROZEN_SOURCES.items():
        if pins[name] != expected or plan.get("files_sha256", {}).get(name) != expected:
            raise ValueError(f"original frozen source pin differs: {name}")
    for name, expected in pins.items():
        if not isinstance(name, str) or str(Path(name).resolve()) != name:
            raise ValueError("amendment file pins must use canonical absolute paths")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"invalid file SHA256 pin: {name}")
        if digest(Path(name)) != expected:
            raise ValueError(f"amendment file SHA256 mismatch: {name}")
    if digest(path) != sha:
        raise ValueError("amendment changed during authentication")
    return amendment


def normalize_num_nodes(record: str) -> str:
    """Change only one isolated exact token; reject missing/ambiguous values."""
    matches = list(NODE_FIELD.finditer(record))
    if len(matches) != 1 or matches[0].group(1) not in {"1", "1-1"}:
        raise ValueError("expected one isolated NumNodes=1 or NumNodes=1-1 field")
    match = matches[0]
    if match.group(1) == "1":
        return record
    return record[:match.start(1)] + "1" + record[match.end(1):]


class LauncherAdapter:
    """Forward the frozen launcher API except its two held-record entrypoints."""
    def __init__(self, frozen_launcher: Any):
        self._original = frozen_launcher

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    def audit_held_record(self, record: str, job_id: int, cell: dict[str, Any]) -> str:
        self._original.audit_held_record(normalize_num_nodes(record), job_id, cell)
        return record

    def audit_held(self, job_id: int, cell: dict[str, Any]) -> str:
        result = subprocess.run(["scontrol", "show", "job", "-dd", "-o", str(job_id)],
                                capture_output=True, text=True, check=False, timeout=60)
        if result.returncode != 0:
            raise ValueError("cannot audit E122 held job: " + result.stderr)
        return self.audit_held_record(result.stdout, job_id, cell)


def load_private_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name + "_" + uuid4().hex, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load pinned module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bind_controller(controller: Any, amendment_path: Path, amendment_sha256: str) -> Any:
    """Wrap a private module in memory; its source and release logic stay frozen."""
    path = Path(amendment_path).resolve()
    authenticate_amendment(path, amendment_sha256)
    original_load = controller.load_campaign

    def authenticated_load(args: argparse.Namespace, launcher: Any) -> dict[str, Any]:
        amendment = authenticate_amendment(path, amendment_sha256)
        if (Path(args.plan).resolve() != Path(amendment["plan_path"])
                or args.plan_sha256 != amendment["plan_sha256"]
                or args.model_choice != amendment["model_choice"]):
            raise ValueError("controller arguments differ from the display amendment")
        campaign = original_load(args, launcher)
        authenticate_amendment(path, amendment_sha256)
        campaign["binding"] = dict(campaign["binding"],
            slurm2511_amendment_path=str(path), slurm2511_amendment_sha256=amendment_sha256)
        return campaign

    controller.load_campaign = authenticated_load
    return controller


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amendment-sha256", required=True)
    options, remaining = parser.parse_known_args(argv)
    try:
        authenticate_amendment(options.amendment, options.amendment_sha256)
        controller = load_private_module(CONTROLLER_PATH, "e122_slurm2511_controller")
        controller = bind_controller(controller, options.amendment, options.amendment_sha256)
        args = controller.parse_args(remaining)
        sys.path.insert(0, str(LAUNCHER_PATH.parent))
        launcher = LauncherAdapter(load_private_module(LAUNCHER_PATH, "e122_slurm2511_launcher"))
        authenticate_amendment(options.amendment, options.amendment_sha256)
        while True:
            payload = (controller.advance_once(args, launcher) if args.advance or args.watch
                       else controller.status(controller.load_campaign(args, launcher), controller.JOURNAL_ROOT))
            print(json.dumps(payload, sort_keys=True), flush=True)
            if not args.watch or payload["blocked_reason"] == "complete":
                return 0
            if payload["blocked_reason"] in {"unknown_scheduler_state", "ambiguous_or_external_activity", "needs_operator_review"}:
                return 2
            time.sleep(args.interval_seconds)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"schema": "e122_slurm2511_adapter_error_v1",
                          "error": f"{type(exc).__name__}: {exc}", "releases_stopped": True}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
