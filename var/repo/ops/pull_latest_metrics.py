#!/usr/bin/env python3
"""Export metrics for the most recent maxent + grpo paired runs.

This script scans local W&B run directories (from WANDB_DIR) and pulls
the history/summary files for the latest run group that contains both
maxent and grpo variants (as created by run_two_trainings.sh).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _list_runs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    runs = [p for p in root.glob("run-*") if p.is_dir()]
    runs.extend([p for p in root.glob("offline-run-*") if p.is_dir()])
    return runs


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_run_metadata(run_dir: Path) -> Optional[Dict[str, Any]]:
    meta_path = run_dir / "files" / "wandb-metadata.json"
    if not meta_path.exists():
        return None
    meta = _read_json(meta_path)
    args = meta.get("args") or []
    run_name = None
    run_group = None
    if isinstance(args, list):
        for idx, item in enumerate(args):
            if item == "--run_name" and idx + 1 < len(args):
                run_name = args[idx + 1]
            elif item == "--wandb_run_group" and idx + 1 < len(args):
                run_group = args[idx + 1]
    program = meta.get("program") or meta.get("codePath")
    return {
        "run_name": run_name,
        "run_group": run_group,
        "program": program,
        "run_dir": run_dir,
    }


def _infer_variant(meta: Dict[str, Any]) -> Optional[str]:
    run_name = meta.get("run_name") or ""
    program = meta.get("program") or ""
    if isinstance(run_name, str):
        if run_name.endswith("-maxent"):
            return "maxent"
        if run_name.endswith("-grpo"):
            return "grpo"
        if "maxent" in run_name and "grpo" not in run_name:
            return "maxent"
        if "grpo" in run_name and "maxent" not in run_name:
            return "grpo"
    if isinstance(program, str):
        if "maxent_grpo.py" in program:
            return "maxent"
        if program.endswith("grpo.py"):
            return "grpo"
    return None


def _group_runs(runs: Iterable[Path]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for run_dir in runs:
        meta = _parse_run_metadata(run_dir)
        if not meta:
            continue
        run_group = meta.get("run_group")
        variant = _infer_variant(meta)
        if not run_group or not variant:
            continue
        grouped[run_group][variant] = meta
    return grouped


def _pick_latest_group(grouped: Dict[str, Dict[str, Dict[str, Any]]]) -> Optional[str]:
    candidates: List[Tuple[float, str]] = []
    for group, variants in grouped.items():
        if "maxent" not in variants or "grpo" not in variants:
            continue
        latest_mtime = max(
            variants["maxent"]["run_dir"].stat().st_mtime,
            variants["grpo"]["run_dir"].stat().st_mtime,
        )
        candidates.append((latest_mtime, group))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _copy_metrics(run_dir: Path, out_dir: Path, prefix: str) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    history_src = run_dir / "files" / "wandb-history.jsonl"
    summary_src = run_dir / "files" / "wandb-summary.json"
    history_dst = out_dir / f"{prefix}_history.jsonl"
    summary_dst = out_dir / f"{prefix}_summary.json"
    if history_src.exists():
        shutil.copy2(history_src, history_dst)
    if summary_src.exists():
        shutil.copy2(summary_src, summary_dst)
    return {
        "history": history_dst,
        "summary": summary_dst,
    }


def _history_to_csv(history_path: Path, csv_path: Path) -> None:
    if not history_path.exists():
        return
    rows: List[Dict[str, Any]] = []
    fieldnames: List[str] = []
    with history_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            rows.append(payload)
            for key in payload.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    if not rows:
        return
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export metrics for the latest paired maxent + grpo runs."
    )
    parser.add_argument(
        "--wandb-dir",
        default="var/artifacts/wandb/wandb",
        help="Path to W&B run directory (default: var/artifacts/wandb/wandb).",
    )
    parser.add_argument(
        "--out-dir",
        default="var/artifacts/metrics",
        help="Directory to write exported metrics into.",
    )
    parser.add_argument(
        "--run-group",
        default=None,
        help="Optional W&B run group to export (default: latest paired group).",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also write CSV exports for the history files.",
    )
    args = parser.parse_args()

    wandb_root = Path(args.wandb_dir)
    runs = _list_runs(wandb_root)
    if not runs:
        print(f"No W&B runs found under {wandb_root}")
        return 1

    grouped = _group_runs(runs)
    group = args.run_group or _pick_latest_group(grouped)
    if not group:
        print("No paired maxent + grpo runs found.")
        return 1
    variants = grouped[group]
    out_dir = Path(args.out_dir) / group
    out_dir.mkdir(parents=True, exist_ok=True)

    exported: Dict[str, Dict[str, Path]] = {}
    for variant in ("maxent", "grpo"):
        meta = variants.get(variant)
        if not meta:
            continue
        exported[variant] = _copy_metrics(meta["run_dir"], out_dir, variant)
        if args.csv:
            history_path = exported[variant]["history"]
            csv_path = out_dir / f"{variant}_history.csv"
            _history_to_csv(history_path, csv_path)

    info = {
        "run_group": group,
        "wandb_root": str(wandb_root),
        "out_dir": str(out_dir),
        "runs": {
            variant: {
                "run_name": variants[variant].get("run_name"),
                "run_dir": str(variants[variant].get("run_dir")),
                "history": str(exported.get(variant, {}).get("history", "")),
                "summary": str(exported.get(variant, {}).get("summary", "")),
            }
            for variant in ("maxent", "grpo")
            if variant in variants
        },
    }
    with (out_dir / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, sort_keys=True)

    print(f"Exported metrics for run group: {group}")
    print(f"Output directory: {out_dir}")
    for variant in ("maxent", "grpo"):
        if variant not in exported:
            continue
        history = exported[variant].get("history")
        summary = exported[variant].get("summary")
        print(f"{variant}:")
        print(f"  history: {history}")
        print(f"  summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
