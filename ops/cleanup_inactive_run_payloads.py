#!/usr/bin/env python3
"""Remove heavyweight model state from inactive experiment run directories.

Training/evaluation metrics and run structure are retained. By default this is
a dry run; pass --apply to delete only ``saved_models`` and ``checkpoints``
trees whose run stamp is not present in the user's live Slurm queue.  A final
``saved_models/step_*`` checkpoint can be retained explicitly while redundant
saved-model snapshots and all optimizer-resume checkpoints are removed.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUN_PREFIXES = ("xdr_", "oat_zero_tiny_")
PAYLOAD_NAMES = {"saved_models", "checkpoints"}


def active_run_stamps(user: str) -> set[str]:
    queue = subprocess.run(
        ["squeue", "-h", "-u", user, "-o", "%i"],
        check=True,
        capture_output=True,
        text=True,
    )
    stamps: set[str] = set()
    for job_id in queue.stdout.split():
        job = subprocess.run(
            ["scontrol", "show", "job", "-o", job_id],
            check=False,
            capture_output=True,
            text=True,
        )
        match = re.search(r"(?:^|,)RUN_STAMP=([^, ]+)", job.stdout)
        if match:
            stamps.add(match.group(1))
    return stamps


def allocated_bytes(path: Path) -> int:
    total = 0
    for root, dirs, files in os.walk(path):
        for name in dirs + files:
            candidate = Path(root, name)
            try:
                total += candidate.lstat().st_blocks * 512
            except FileNotFoundError:
                pass
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "var/data")
    parser.add_argument("--user", default=os.environ.get("USER", ""))
    parser.add_argument("--min-age-minutes", type=float, default=10.0)
    parser.add_argument(
        "--run-name-contains",
        action="append",
        default=[],
        help="limit cleanup to run names containing one of these strings",
    )
    parser.add_argument(
        "--keep-saved-step",
        type=int,
        action="append",
        default=[],
        help=(
            "retain this saved_models/step_N checkpoint while deleting the "
            "other saved-model snapshots; repeat to preserve terminal aliases; "
            "optimizer checkpoints are still removed"
        ),
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    active = active_run_stamps(args.user)
    cutoff = time.time() - args.min_age_minutes * 60
    candidates: list[tuple[Path, int]] = []
    protected = 0
    retained = 0
    retained_steps = {f"step_{step:05d}" for step in args.keep_saved_step}

    for run_dir in data_root.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith(RUN_PREFIXES):
            continue
        if args.run_name_contains and not any(
            token in run_dir.name for token in args.run_name_contains
        ):
            continue
        is_active = any(stamp in run_dir.name for stamp in active)
        for payload in run_dir.glob("debug_*/*"):
            if (
                payload.name not in PAYLOAD_NAMES
                or not payload.is_dir()
                or payload.is_symlink()
            ):
                continue
            if is_active or payload.stat().st_mtime > cutoff:
                protected += 1
                continue
            if payload.name == "saved_models" and retained_steps:
                for snapshot in payload.iterdir():
                    if snapshot.name in retained_steps:
                        retained += 1
                        continue
                    if snapshot.is_dir() and not snapshot.is_symlink():
                        candidates.append((snapshot, allocated_bytes(snapshot)))
                continue
            candidates.append((payload, allocated_bytes(payload)))

    total = sum(size for _, size in candidates)
    action = "deleting" if args.apply else "would delete"
    print(
        f"[cleanup] active_stamps={len(active)} protected_payloads={protected} "
        f"retained_final_snapshots={retained} "
        f"candidates={len(candidates)} {action}={total / 2**30:.1f} GiB"
    )
    for path, size in sorted(candidates, key=lambda item: item[1], reverse=True):
        print(f"[cleanup] {size / 2**30:8.1f} GiB {path}")
        if args.apply:
            shutil.rmtree(path)

    if not args.apply:
        print("[cleanup] dry run; rerun with --apply to remove these payloads")


if __name__ == "__main__":
    main()
