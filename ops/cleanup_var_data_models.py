#!/usr/bin/env python3
"""Safely prune obsolete model state beneath ``var/data``.

The cleanup is intentionally narrower than its name: datasets, telemetry,
evaluation outputs, and run directories are never candidates.  Active Slurm
run stamps protect their entire run roots.  For inactive runs we remove raw
optimizer checkpoints, discard development/smoke exports, and retain only the
highest exported step from the most advanced attempt of each analytical run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = ROOT / "var/data"
DEFAULT_REPORT = ROOT / "var/artifacts/var_data_model_cleanup_latest.json"
STEP_RE = re.compile(r"^step_(\d+)$")
SEED_RE = re.compile(r"_s(\d+)$")
DEVELOPMENT_MARKERS = (
    "smoke",
    "pilot",
    "preflight",
    "probe",
    "diagnostic",
    "unit_test",
)


def _run(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout


def active_run_stamps() -> set[str]:
    user = os.environ.get("USER", "")
    if not user:
        raise RuntimeError("USER is unset; refusing to infer active jobs")
    job_ids = [
        line.strip()
        for line in _run(["squeue", "-u", user, "-h", "-o", "%i"]).splitlines()
        if line.strip().isdigit()
    ]
    stamps: set[str] = set()
    for job_id in job_ids:
        record = _run(["scontrol", "show", "job", job_id, "-o"])
        match = re.search(r"(?:^|[, ])RUN_STAMP=([^, ]+)", record)
        if match:
            stamps.add(match.group(1))
    return stamps


def _is_active_root(run_root: Path, stamps: set[str]) -> bool:
    return any(run_root.name.endswith("_" + stamp) for stamp in stamps)


def _is_development_root(run_root: Path) -> bool:
    lowered = run_root.name.lower()
    if any(marker in lowered for marker in DEVELOPMENT_MARKERS):
        return True
    match = SEED_RE.search(run_root.name)
    return bool(match and int(match.group(1)) >= 9000)


def _max_metric_step(debug_dir: Path) -> int:
    path = debug_dir / "train_metrics.jsonl"
    maximum = -1
    if not path.is_file():
        return maximum
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
            value = row.get("trainer/global_step", row.get("misc/global_step", -1))
            maximum = max(maximum, int(float(value)))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return maximum


def _max_export_step(saved_models: Path) -> int:
    return max(
        (
            int(match.group(1))
            for child in saved_models.iterdir()
            if child.is_dir() and (match := STEP_RE.match(child.name))
        ),
        default=-1,
    )


def _sizes(path: Path) -> tuple[int, int, int]:
    logical = physical = files = 0
    if path.is_symlink() or path.is_file():
        stat = path.lstat()
        return stat.st_size, stat.st_blocks * 512, 1
    for item in path.rglob("*"):
        if not item.is_file() and not item.is_symlink():
            continue
        try:
            stat = item.lstat()
        except FileNotFoundError:
            continue
        logical += stat.st_size
        physical += stat.st_blocks * 512
        files += 1
    return logical, physical, files


def _validated(path: Path, data_root: Path, active_roots: set[Path]) -> Path:
    resolved = path.resolve()
    root = data_root.resolve()
    if resolved == root or root not in resolved.parents:
        raise RuntimeError(f"cleanup target escapes var/data: {path}")
    run_root = next(
        (parent for parent in (resolved, *resolved.parents) if parent.parent == root),
        None,
    )
    if run_root is None:
        raise RuntimeError(f"cleanup target has no run root: {path}")
    if run_root in active_roots:
        raise RuntimeError(f"cleanup target belongs to an active run: {path}")
    parts = resolved.relative_to(run_root).parts
    if "checkpoints" not in parts and "saved_models" not in parts:
        raise RuntimeError(f"cleanup target is not model state: {path}")
    return resolved


def discover(data_root: Path, stamps: set[str]) -> tuple[list[dict[str, Any]], list[str]]:
    data_root = data_root.resolve()
    run_roots = [path.resolve() for path in data_root.iterdir() if path.is_dir()]
    active_roots = {path for path in run_roots if _is_active_root(path, stamps)}
    reasons: dict[Path, str] = {}

    # Raw optimizer state is useful only for resuming a live job.  Exported
    # weights and all result files remain available for inactive runs.
    for path in data_root.glob("*/debug_*/checkpoints"):
        if path.parents[1].resolve() not in active_roots:
            reasons[path.resolve()] = "inactive_raw_optimizer_checkpoint"

    for run_root in run_roots:
        if run_root in active_roots:
            continue
        exports = sorted(run_root.glob("debug_*/saved_models"))
        if not exports:
            continue
        if _is_development_root(run_root):
            for path in exports:
                reasons[path.resolve()] = "development_or_smoke_model_export"
            continue

        # Retain only the most advanced attempt's exports.  Metric progress is
        # the primary key, then export step, then timestamped debug directory.
        keeper = max(
            exports,
            key=lambda path: (
                _max_metric_step(path.parent),
                _max_export_step(path),
                path.parent.name,
            ),
        )
        for path in exports:
            if path != keeper:
                reasons[path.resolve()] = "superseded_attempt_model_export"

        # Within the retained attempt, one highest-step export is sufficient.
        step_dirs = [
            child
            for child in keeper.iterdir()
            if child.is_dir() and STEP_RE.match(child.name)
        ]
        if len(step_dirs) > 1:
            terminal = max(step_dirs, key=lambda path: int(STEP_RE.match(path.name).group(1)))
            for path in step_dirs:
                if path != terminal:
                    reasons[path.resolve()] = "redundant_preterminal_model_export"

    # Remove descendants when an entire ancestor is already a candidate.
    ordered = sorted(reasons, key=lambda path: (len(path.parts), str(path)))
    selected: list[Path] = []
    for path in ordered:
        if any(parent == existing or existing in parent.parents for existing in selected for parent in (path,)):
            continue
        selected.append(path)

    records: list[dict[str, Any]] = []
    for path in selected:
        checked = _validated(path, data_root, active_roots)
        logical, physical, files = _sizes(checked)
        records.append(
            {
                "path": str(checked),
                "reason": reasons[path],
                "logical_bytes": logical,
                "physical_bytes": physical,
                "files": files,
            }
        )
    return records, sorted(str(path) for path in active_roots)


def _atomic_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    if data_root != DEFAULT_DATA_ROOT.resolve():
        raise RuntimeError("refusing a non-default cleanup root")

    stamps = active_run_stamps()
    records, active_roots = discover(data_root, stamps)
    totals = {
        "targets": len(records),
        "logical_bytes": sum(row["logical_bytes"] for row in records),
        "physical_bytes": sum(row["physical_bytes"] for row in records),
        "files": sum(row["files"] for row in records),
    }
    payload: dict[str, Any] = {
        "schema": "var_data_model_cleanup_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "executed": bool(args.execute),
        "active_run_stamps": sorted(stamps),
        "protected_active_roots": active_roots,
        "policy": {
            "datasets_metrics_and_evaluations": "retained",
            "active_run_roots": "fully retained",
            "inactive_optimizer_checkpoints": "removed",
            "development_seed_900x_exports": "removed",
            "inactive_analytical_exports": "highest step of most advanced attempt retained",
        },
        "totals": totals,
        "targets": records,
    }
    _atomic_report(args.report, payload)
    print(json.dumps({"report": str(args.report.resolve()), **totals}, sort_keys=True))
    if not args.execute:
        return

    active_root_set = {Path(path) for path in active_roots}
    for row in records:
        path = _validated(Path(row["path"]), data_root, active_root_set)
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)

    payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    payload["completed"] = True
    _atomic_report(args.report, payload)


if __name__ == "__main__":
    main()
