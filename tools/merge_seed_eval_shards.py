#!/usr/bin/env python3
"""Merge per-task SEED eval shards into a full-suite summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_dict_values(payload: dict[str, float]) -> float | None:
    if not payload:
        return None
    return float(sum(float(value) for value in payload.values()) / len(payload))


def _find_single_summary(task_dir: Path) -> Path:
    matches = sorted(task_dir.glob("*.summary.json"))
    if not matches:
        raise FileNotFoundError(f"No shard summary found under {task_dir}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Expected exactly one shard summary under {task_dir}, found {len(matches)}"
        )
    return matches[0]


def _merge_metric_map(
    shard_payloads: dict[str, dict[str, Any]],
    key: str,
) -> dict[str, float]:
    merged: dict[str, float] = {}
    for summary in shard_payloads.values():
        raw = summary.get(key, {})
        if not isinstance(raw, dict):
            continue
        for task_name, value in raw.items():
            merged[str(task_name)] = float(value)
    return merged


def _extend_records(
    summary: dict[str, Any],
    kind: str,
    records: list[dict[str, Any]],
) -> str | None:
    saved = summary.get("saved_output_paths", {})
    if not isinstance(saved, dict):
        return None
    candidate = saved.get(kind)
    if not candidate:
        return None
    path = Path(str(candidate))
    if not path.exists():
        return None
    payload = _load_json(path)
    if not isinstance(payload, list):
        return None
    for record in payload:
        if isinstance(record, dict):
            records.append(record)
    return path.name


def _sort_output_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(record: dict[str, Any]) -> tuple[str, int, int]:
        task_name = str(record.get("task_name", ""))
        prompt_index = int(record.get("prompt_index", -1) or -1)
        sample_index = int(record.get("sample_index", -1) or -1)
        return (task_name, prompt_index, sample_index)

    return sorted(records, key=sort_key)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent-dir",
        required=True,
        help="Directory containing one subdirectory per task shard.",
    )
    parser.add_argument(
        "--tasks",
        default="aime,amc,math,minerva,olympiad_bench",
        help="Comma-separated task list to merge.",
    )
    parser.add_argument(
        "--summary-name",
        default="seed_paper_eval_sharded.summary.json",
        help="Output filename for the merged summary.",
    )
    return parser.parse_args()


def _iter_tasks(raw_tasks: str) -> Iterable[str]:
    for item in raw_tasks.split(","):
        task = item.strip()
        if task:
            yield task


def main() -> int:
    args = _parse_args()
    parent_dir = Path(args.parent_dir).resolve()
    parent_dir.mkdir(parents=True, exist_ok=True)

    tasks = list(_iter_tasks(args.tasks))
    if not tasks:
        raise SystemExit("No tasks were provided.")

    shard_summaries: dict[str, dict[str, Any]] = {}
    shard_summary_paths: dict[str, str] = {}
    for task in tasks:
        task_dir = parent_dir / task
        summary_path = _find_single_summary(task_dir)
        shard_summary_paths[task] = str(summary_path)
        shard_summaries[task] = _load_json(summary_path)

    results = _merge_metric_map(shard_summaries, "results")
    pass_at_8 = _merge_metric_map(shard_summaries, "pass_at_8")
    mean_at_8 = _merge_metric_map(shard_summaries, "mean_at_8")
    avg_lens = _merge_metric_map(shard_summaries, "avg_lens")
    max_lens = _merge_metric_map(shard_summaries, "max_lens")
    formatted = _merge_metric_map(shard_summaries, "formatted")

    single_records: list[dict[str, Any]] = []
    pass8_records: list[dict[str, Any]] = []
    single_name: str | None = None
    pass8_name: str | None = None
    for task in tasks:
        single_name = single_name or _extend_records(
            shard_summaries[task], "single", single_records
        )
        pass8_name = pass8_name or _extend_records(
            shard_summaries[task], "pass_at_8", pass8_records
        )

    merged_summary: dict[str, Any] = {
        "mode": "vllm_server",
        "merged_from_task_shards": True,
        "tasks": tasks,
        "task_summary_paths": shard_summary_paths,
        "results": results,
        "avg": _mean_dict_values(results),
        "pass_at_8": pass_at_8,
        "pass_at_8_avg": _mean_dict_values(pass_at_8),
        "mean_at_8": mean_at_8,
        "mean_at_8_avg": _mean_dict_values(mean_at_8),
        "avg_lens": avg_lens,
        "max_lens": max_lens,
        "formatted": formatted,
        "process_exit_code": max(
            int(shard_summaries[task].get("process_exit_code", 0) or 0) for task in tasks
        ),
    }

    first_summary = shard_summaries[tasks[0]]
    if isinstance(first_summary.get("pass_at_8_config"), dict):
        merged_summary["pass_at_8_config"] = first_summary["pass_at_8_config"]

    saved_output_paths: dict[str, str] = {}
    if single_records:
        single_out = parent_dir / (single_name or "seed_paper_eval_outputs_single_n1.json")
        _write_json(single_out, _sort_output_records(single_records))
        saved_output_paths["single"] = str(single_out)
    if pass8_records:
        pass8_out = parent_dir / (
            pass8_name or "seed_paper_eval_outputs_pass_at_8_n8.json"
        )
        _write_json(pass8_out, _sort_output_records(pass8_records))
        saved_output_paths["pass_at_8"] = str(pass8_out)
    if saved_output_paths:
        merged_summary["saved_output_paths"] = saved_output_paths

    summary_path = parent_dir / args.summary_name
    _write_json(summary_path, merged_summary)

    print(f"Merged summary: {summary_path}")
    print(json.dumps(merged_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
