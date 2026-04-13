#!/usr/bin/env python3
"""Upload the latest DR.GRPO save root for each discovered seed to Hugging Face.

This is a thin wrapper around ``upload_drgrpo_seed_run_to_hf.py`` that:

- scans ``var/data/`` for OAT zero exact save roots
- groups matching roots by seed
- selects the latest save root per seed
- invokes the per-seed uploader once for each selected seed
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SAVE_ROOT_RE = re.compile(
    r"oat_zero_exact_1p5b_(?P<timestamp>\d{8}_\d{6})(?:_.*)?_seed(?P<seed>\d+)$"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--namespace",
        type=str,
        required=True,
        help="HF namespace/user/org for auto-generated repo ids.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("var/data"),
        help="Directory containing OAT save roots.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        default=[],
        help="Seed(s) to upload. Repeat to select multiple seeds. Defaults to all discovered seeds.",
    )
    parser.add_argument(
        "--repo-prefix",
        type=str,
        default="qwen2.5-Math-1.5b-drgrpo-readmeflash-a6000-5epoch",
        help="Prefix used to form <namespace>/<repo-prefix>-seed<seed>.",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="all",
        help="Comma-separated steps like 16,32,240 or 'all'.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("var/hf_exports/manifests"),
        help="Local directory where upload manifests/README previews are written.",
    )
    parser.add_argument(
        "--commit-message-prefix",
        type=str,
        default="Upload",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repos as private if they do not already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print and execute dry-run uploads only.",
    )
    parser.add_argument(
        "--include-eval-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload eval_results/*.json alongside checkpoints.",
    )
    parser.add_argument(
        "--include-deepspeed-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload DeepSpeed optimizer checkpoints too.",
    )
    parser.add_argument(
        "--include-initial-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include save_root/saved_models/step_00000 when present.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip checkpoint steps already present remotely.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue uploading later seeds if one upload fails.",
    )
    return parser.parse_args()


def _root_sort_key(path: Path, timestamp_text: str) -> tuple[datetime, float]:
    return (datetime.strptime(timestamp_text, "%Y%m%d_%H%M%S"), path.stat().st_mtime)


def _is_uploadable_save_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "saved_models" / "step_00000").is_dir():
        return False
    for child in path.iterdir():
        if child.is_dir() and (child / "saved_models").is_dir() and (child / "eval_results").is_dir():
            return True
    return False


def _discover_latest_save_roots(data_root: Path) -> dict[int, Path]:
    latest_by_seed: dict[int, Path] = {}
    latest_key_by_seed: dict[int, tuple[datetime, float]] = {}
    for child in sorted(data_root.iterdir()):
        match = SAVE_ROOT_RE.match(child.name)
        if match is None:
            continue
        if not _is_uploadable_save_root(child):
            continue
        seed = int(match.group("seed"))
        sort_key = _root_sort_key(child, match.group("timestamp"))
        current_key = latest_key_by_seed.get(seed)
        if current_key is None or sort_key > current_key:
            latest_by_seed[seed] = child.resolve()
            latest_key_by_seed[seed] = sort_key
    return latest_by_seed


def _selected_roots(args: argparse.Namespace) -> list[tuple[int, Path]]:
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(f"data_root does not exist or is not a directory: {data_root}")
    latest_by_seed = _discover_latest_save_roots(data_root)
    if not latest_by_seed:
        raise SystemExit(f"No matching save roots found under {data_root}")

    if args.seed:
        selected = []
        for seed in sorted(set(args.seed)):
            if seed not in latest_by_seed:
                raise SystemExit(f"No matching save root found for seed {seed} under {data_root}")
            selected.append((seed, latest_by_seed[seed]))
        return selected

    return sorted(latest_by_seed.items())


def _build_command(args: argparse.Namespace, save_root: Path) -> list[str]:
    script_path = Path(__file__).with_name("upload_drgrpo_seed_run_to_hf.py")
    cmd = [
        sys.executable,
        str(script_path),
        "--save-root",
        str(save_root),
        "--namespace",
        args.namespace,
        "--repo-prefix",
        args.repo_prefix,
        "--steps",
        args.steps,
        "--manifest-dir",
        str(args.manifest_dir),
        "--commit-message-prefix",
        args.commit_message_prefix,
    ]
    cmd.append("--include-eval-results" if args.include_eval_results else "--no-include-eval-results")
    cmd.append(
        "--include-deepspeed-checkpoints"
        if args.include_deepspeed_checkpoints
        else "--no-include-deepspeed-checkpoints"
    )
    cmd.append("--include-initial-step" if args.include_initial_step else "--no-include-initial-step")
    cmd.append("--skip-existing" if args.skip_existing else "--no-skip-existing")
    if args.private:
        cmd.append("--private")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def main() -> int:
    args = _parse_args()
    selected = _selected_roots(args)

    print("selected save roots:")
    for seed, save_root in selected:
        print(f"  seed {seed}: {save_root}")

    failures: list[tuple[int, int]] = []
    for seed, save_root in selected:
        cmd = _build_command(args, save_root)
        print()
        print(f"=== seed {seed} ===")
        print("command:", " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            continue
        failures.append((seed, result.returncode))
        if not args.continue_on_error:
            break

    if failures:
        for seed, returncode in failures:
            print(f"failed: seed {seed} exited with code {returncode}", file=sys.stderr)
        return 1

    print()
    print("all requested uploads finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
