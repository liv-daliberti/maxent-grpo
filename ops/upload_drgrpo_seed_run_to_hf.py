#!/usr/bin/env python3
"""Upload organized DR.GRPO seed checkpoints to Hugging Face.

This script mirrors the OAT run layout we use locally into a clean per-seed
model repo:

- checkpoints/step_XXXXX/
- eval_results/<step>_<benchmark>.json
- metadata/checkpoints_index.json
- metadata/run_manifest.json
- README.md

It is safe to rerun. By default it skips checkpoint steps that already exist on
the remote repo.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, get_token


STEP_RE = re.compile(r"step_(\d+)$")
SEED_RE = re.compile(r"seed(\d+)")
EVAL_RE = re.compile(r"(?P<step>\d+)_(?P<benchmark>.+)\.json$")


@dataclass
class RunLayout:
    save_root: Path
    run_dir: Path
    initial_step_dir: Path | None
    saved_model_steps: dict[str, Path]
    ds_checkpoint_steps: dict[str, Path]
    eval_files: list[Path]
    seed: int | None
    run_name: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--save-root",
        type=Path,
        help="Top-level OAT save root, e.g. var/data/oat_zero_exact_1p5b_..._seed42",
    )
    source.add_argument(
        "--run-dir",
        type=Path,
        help="Specific experiment run dir containing saved_models/ and eval_results/",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="",
        help="Full Hugging Face repo id. If omitted, use --namespace/--repo-prefix.",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="",
        help="HF namespace/user/org for auto-generated repo ids.",
    )
    parser.add_argument(
        "--repo-prefix",
        type=str,
        default="qwen2.5-Math-1.5b-drgrpo-readmeflash-a6000-5epoch",
        help="Prefix used with --namespace to form <namespace>/<repo-prefix>-seed<seed>.",
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default="model",
        choices=["model", "dataset", "space"],
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it does not already exist.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default="",
        help="HF token. Defaults to HF_TOKEN/HUGGINGFACE_HUB_TOKEN if set.",
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
        help="Upload DeepSpeed optimizer checkpoints under deepspeed_checkpoints/ too.",
    )
    parser.add_argument(
        "--include-initial-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include save_root/saved_models/step_00000 when present.",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="all",
        help="Comma-separated steps like 16,32,240 or 'all'.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip checkpoint steps already present remotely.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and write local metadata, but do not upload anything.",
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
    return parser.parse_args()


def _require_dir(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise SystemExit(f"{label} does not exist or is not a directory: {path}")
    return path


def _sorted_step_dirs(parent: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    if not parent.is_dir():
        return result
    for child in sorted(parent.iterdir()):
        if not child.is_dir():
            continue
        match = STEP_RE.match(child.name)
        if match is None:
            continue
        result[child.name] = child
    return result


def _infer_seed_from_paths(*paths: Path) -> int | None:
    for path in paths:
        for piece in (path.name, str(path)):
            match = SEED_RE.search(piece)
            if match is not None:
                return int(match.group(1))
    return None


def _discover_run_dir(save_root: Path) -> Path:
    candidates = [
        child
        for child in save_root.iterdir()
        if child.is_dir()
        and (child / "saved_models").is_dir()
        and (child / "eval_results").is_dir()
    ]
    if not candidates:
        raise SystemExit(
            f"No experiment subdirectory with saved_models/ and eval_results/ found under {save_root}"
        )
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _discover_layout(args: argparse.Namespace) -> RunLayout:
    if args.save_root is not None:
        save_root = _require_dir(args.save_root, "save_root")
        run_dir = _discover_run_dir(save_root)
    else:
        run_dir = _require_dir(args.run_dir, "run_dir")
        save_root = run_dir.parent

    initial_step_dir = save_root / "saved_models" / "step_00000"
    if not initial_step_dir.is_dir():
        initial_step_dir = None

    saved_model_steps = _sorted_step_dirs(run_dir / "saved_models")
    ds_checkpoint_steps = _sorted_step_dirs(run_dir / "checkpoints")
    eval_files = sorted((run_dir / "eval_results").glob("*.json"))
    seed = _infer_seed_from_paths(save_root, run_dir)
    return RunLayout(
        save_root=save_root,
        run_dir=run_dir,
        initial_step_dir=initial_step_dir,
        saved_model_steps=saved_model_steps,
        ds_checkpoint_steps=ds_checkpoint_steps,
        eval_files=eval_files,
        seed=seed,
        run_name=run_dir.name,
    )


def _normalize_step_name(value: str) -> str:
    if value.startswith("step_"):
        return value
    return f"step_{int(value):05d}"


def _selected_steps(layout: RunLayout, include_initial: bool, spec: str) -> list[str]:
    steps = []
    if include_initial and layout.initial_step_dir is not None:
        steps.append("step_00000")
    steps.extend(layout.saved_model_steps.keys())
    steps = sorted(set(steps))
    if spec == "all":
        return steps
    requested = {_normalize_step_name(piece.strip()) for piece in spec.split(",") if piece.strip()}
    missing = sorted(requested.difference(steps))
    if missing:
        raise SystemExit(f"Requested steps not found locally: {missing}")
    return [step for step in steps if step in requested]


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def _eval_index(eval_files: Iterable[Path]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for path in eval_files:
        match = EVAL_RE.match(path.name)
        if match is None:
            continue
        step_key = _normalize_step_name(match.group("step"))
        bench = match.group("benchmark")
        index.setdefault(step_key, {})[bench] = path.name
    return index


def _build_repo_id(args: argparse.Namespace, layout: RunLayout) -> str:
    if args.repo_id:
        return args.repo_id
    if not args.namespace:
        raise SystemExit("Provide either --repo-id or --namespace.")
    if layout.seed is None:
        raise SystemExit("Could not infer seed from the run layout; pass --repo-id explicitly.")
    return f"{args.namespace}/{args.repo_prefix}-seed{layout.seed}"


def _build_readme(layout: RunLayout, repo_id: str, steps: list[str], eval_map: dict[str, dict[str, str]]) -> str:
    seed_label = f"seed {layout.seed}" if layout.seed is not None else "unknown seed"
    rows = []
    for step in steps:
        model_dir = (
            layout.initial_step_dir if step == "step_00000" else layout.saved_model_steps.get(step)
        )
        size_gb = _dir_size_bytes(model_dir) / (1024 ** 3) if model_dir is not None else 0.0
        eval_benchmarks = ", ".join(sorted(eval_map.get(step, {}).keys())) or "-"
        rows.append(f"| `{step}` | `{size_gb:.2f}` | {eval_benchmarks} |")
    table = "\n".join(rows) if rows else "| - | - | - |"
    return (
        f"""\
---
license: apache-2.0
library_name: transformers
base_model: Qwen/Qwen2.5-Math-1.5B
tags:
- math
- reinforcement-learning
- grpo
- checkpoint
---

# {repo_id}

This repo stores organized DR.GRPO checkpoints for **{seed_label}** from our
`Qwen2.5-Math-1.5B` readme-flash math-reasoning training run.

## Layout

- `checkpoints/step_XXXXX/`: exported model checkpoints ready for inference
- `eval_results/`: per-benchmark JSON outputs saved at eval boundaries
- `metadata/checkpoints_index.json`: machine-readable checkpoint manifest
- `metadata/run_manifest.json`: local run provenance

## Source Run

- Local save root: `{layout.save_root}`
- Local run dir: `{layout.run_dir}`
- Run name: `{layout.run_name}`
- Objective: `dr.grpo`

## Available Checkpoints

| Step | Size (GiB) | Eval files |
| --- | ---: | --- |
{table}
"""
    ).strip() + "\n"


def _write_metadata(
    args: argparse.Namespace,
    layout: RunLayout,
    repo_id: str,
    steps: list[str],
    eval_map: dict[str, dict[str, str]],
) -> tuple[Path, Path]:
    manifest_root = args.manifest_dir.expanduser().resolve()
    manifest_root.mkdir(parents=True, exist_ok=True)
    repo_slug = repo_id.replace("/", "__")
    staging_dir = manifest_root / repo_slug
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    (staging_dir / "metadata").mkdir(parents=True, exist_ok=True)

    checkpoints_index = []
    for step in steps:
        model_dir = (
            layout.initial_step_dir if step == "step_00000" else layout.saved_model_steps.get(step)
        )
        checkpoints_index.append(
            {
                "step": step,
                "model_dir": str(model_dir) if model_dir is not None else None,
                "size_bytes": _dir_size_bytes(model_dir) if model_dir is not None else 0,
                "eval_files": eval_map.get(step, {}),
            }
        )

    run_manifest = {
        "repo_id": repo_id,
        "seed": layout.seed,
        "objective": "dr.grpo",
        "save_root": str(layout.save_root),
        "run_dir": str(layout.run_dir),
        "run_name": layout.run_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_steps": steps,
        "include_eval_results": bool(args.include_eval_results),
        "include_deepspeed_checkpoints": bool(args.include_deepspeed_checkpoints),
    }

    (staging_dir / "README.md").write_text(
        _build_readme(layout, repo_id, steps, eval_map),
        encoding="utf-8",
    )
    (staging_dir / "metadata" / "checkpoints_index.json").write_text(
        json.dumps(checkpoints_index, indent=2) + "\n",
        encoding="utf-8",
    )
    (staging_dir / "metadata" / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return staging_dir, staging_dir / "metadata" / "run_manifest.json"


def _resolve_token(args: argparse.Namespace) -> str:
    token = (
        args.token
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or get_token()
    )
    if not token and not args.dry_run:
        raise SystemExit(
            "No Hugging Face token found. Run `hf auth login` or pass --token / set HF_TOKEN."
        )
    return token or ""


def _list_existing_files(api: HfApi, repo_id: str, repo_type: str, token: str) -> set[str]:
    try:
        return set(api.list_repo_files(repo_id=repo_id, repo_type=repo_type, token=token))
    except Exception:
        return set()


def main() -> int:
    args = _parse_args()
    layout = _discover_layout(args)
    repo_id = _build_repo_id(args, layout)
    steps = _selected_steps(layout, args.include_initial_step, args.steps)
    eval_map = _eval_index(layout.eval_files)
    staging_dir, manifest_path = _write_metadata(args, layout, repo_id, steps, eval_map)

    print(f"repo_id: {repo_id}")
    print(f"seed: {layout.seed}")
    print(f"save_root: {layout.save_root}")
    print(f"run_dir: {layout.run_dir}")
    print(f"manifest: {manifest_path}")
    print(f"steps: {', '.join(steps)}")

    token = _resolve_token(args)
    if args.dry_run:
        print("dry_run: no uploads performed")
        return 0

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
        token=token,
    )
    existing = _list_existing_files(api, repo_id, args.repo_type, token) if args.skip_existing else set()

    api.upload_folder(
        repo_id=repo_id,
        repo_type=args.repo_type,
        folder_path=str(staging_dir),
        path_in_repo="",
        token=token,
        commit_message=f"{args.commit_message_prefix} metadata for {layout.run_name}",
    )

    for step in steps:
        model_dir = layout.initial_step_dir if step == "step_00000" else layout.saved_model_steps.get(step)
        if model_dir is None:
            continue
        marker = f"checkpoints/{step}/config.json"
        if args.skip_existing and marker in existing:
            print(f"skip existing checkpoint {step}")
            continue
        print(f"upload checkpoint {step} -> {repo_id}/checkpoints/{step}")
        api.upload_folder(
            repo_id=repo_id,
            repo_type=args.repo_type,
            folder_path=str(model_dir),
            path_in_repo=f"checkpoints/{step}",
            token=token,
            commit_message=f"{args.commit_message_prefix} {step}",
        )

    if args.include_eval_results:
        for eval_file in layout.eval_files:
            rel = f"eval_results/{eval_file.name}"
            if args.skip_existing and rel in existing:
                continue
            api.upload_file(
                repo_id=repo_id,
                repo_type=args.repo_type,
                path_or_fileobj=str(eval_file),
                path_in_repo=rel,
                token=token,
                commit_message=f"{args.commit_message_prefix} eval {eval_file.name}",
            )

    if args.include_deepspeed_checkpoints:
        for step, ds_dir in sorted(layout.ds_checkpoint_steps.items()):
            marker = f"deepspeed_checkpoints/{step}/latest"
            if args.skip_existing and marker in existing:
                print(f"skip existing deepspeed checkpoint {step}")
                continue
            print(f"upload deepspeed checkpoint {step}")
            api.upload_folder(
                repo_id=repo_id,
                repo_type=args.repo_type,
                folder_path=str(ds_dir),
                path_in_repo=f"deepspeed_checkpoints/{step}",
                token=token,
                commit_message=f"{args.commit_message_prefix} deepspeed {step}",
            )

    print(f"done: https://huggingface.co/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
