#!/usr/bin/env python3
"""Quickly reject partial DeepSpeed ZIP checkpoints before auto-resume."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import zipfile

STEP = re.compile(r"^step_(\d+)$")


def validate_checkpoint(path: Path) -> list[str]:
    violations: list[str] = []
    if not path.is_dir():
        return ["checkpoint path is not a directory"]
    archives = sorted(path.glob("*.pt"))
    if not archives:
        return ["checkpoint has no .pt archives"]
    if not any(item.name.endswith("_model_states.pt") for item in archives):
        violations.append("checkpoint has no model-state archive")
    if not any(item.name.endswith("_optim_states.pt") for item in archives):
        violations.append("checkpoint has no optimizer-state archive")
    for archive_path in archives:
        try:
            with zipfile.ZipFile(archive_path) as archive:
                names = archive.namelist()
        except (OSError, zipfile.BadZipFile) as exc:
            violations.append(
                f"{archive_path.name}: unreadable ZIP directory ({type(exc).__name__})"
            )
            continue
        if not names:
            violations.append(f"{archive_path.name}: empty ZIP directory")
    return violations


def select_latest_checkpoint(root: Path) -> tuple[Path | None, dict[str, list[str]]]:
    candidates: list[tuple[int, int, str, Path]] = []
    rejected: dict[str, list[str]] = {}
    if root.is_dir():
        for path in root.glob("**/checkpoints/step_*"):
            if not path.is_dir() or len(path.relative_to(root).parts) > 4:
                continue
            match = STEP.fullmatch(path.name)
            if match is None:
                continue
            violations = validate_checkpoint(path)
            if violations:
                rejected[str(path)] = violations
                continue
            candidates.append(
                (int(match.group(1)), path.stat().st_mtime_ns, str(path), path)
            )
    if not candidates:
        return None, rejected
    return max(candidates)[-1], rejected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=Path)
    group.add_argument("--select-under", type=Path)
    args = parser.parse_args()
    if args.select_under is not None:
        selected, rejected = select_latest_checkpoint(args.select_under)
        for path, violations in sorted(rejected.items()):
            for violation in violations:
                print(f"[checkpoint-invalid] {path}: {violation}", file=sys.stderr)
        if selected is not None:
            print(selected)
        return 0
    assert args.checkpoint is not None
    violations = validate_checkpoint(args.checkpoint)
    if violations:
        for violation in violations:
            print(
                f"[checkpoint-invalid] {args.checkpoint}: {violation}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
