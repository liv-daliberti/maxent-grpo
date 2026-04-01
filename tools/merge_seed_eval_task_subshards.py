#!/usr/bin/env python3
"""Merge prompt-range SEED eval subshards for a single task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sort_output_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(record: dict[str, Any]) -> tuple[str, int, int]:
        task_name = str(record.get("task_name", ""))
        prompt_index = int(record.get("prompt_index", -1) or -1)
        sample_index = int(record.get("sample_index", -1) or -1)
        return (task_name, prompt_index, sample_index)

    return sorted(records, key=sort_key)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True, help="Task directory containing shard subdirs.")
    parser.add_argument(
        "--summary-name",
        default="seed_paper_eval_sharded.summary.json",
        help="Output filename written at the task-dir root.",
    )
    return parser.parse_args()


def _find_subshard_summaries(task_dir: Path) -> list[Path]:
    matches = sorted(
        path
        for path in task_dir.rglob("*.summary.json")
        if path.parent != task_dir
    )
    if not matches:
        raise FileNotFoundError(f"No subshard summaries found under {task_dir}")
    return matches


def _extract_task_name(summary: dict[str, Any], summary_path: Path) -> str:
    results = summary.get("results", {})
    if not isinstance(results, dict) or len(results) != 1:
        raise RuntimeError(
            f"Expected exactly one task result in {summary_path}, found {results!r}"
        )
    return str(next(iter(results)))


def _infer_prompt_count(summary: dict[str, Any], task_name: str) -> int:
    counts = summary.get("task_prompt_counts")
    if isinstance(counts, dict) and task_name in counts:
        return int(counts[task_name])
    ranges = summary.get("task_prompt_ranges")
    if isinstance(ranges, dict):
        task_range = ranges.get(task_name)
        if isinstance(task_range, dict):
            start = int(task_range.get("start", 0) or 0)
            end = int(task_range.get("end", start) or start)
            return max(0, end - start)
    saved = summary.get("saved_output_paths", {})
    if isinstance(saved, dict):
        candidate = saved.get("single")
        if candidate:
            payload = _load_json(Path(str(candidate)))
            if isinstance(payload, list):
                prompt_ids = {
                    int(record.get("prompt_index", -1))
                    for record in payload
                    if isinstance(record, dict)
                    and str(record.get("task_name", "")) == task_name
                }
                if prompt_ids:
                    return len(prompt_ids)
    raise RuntimeError(f"Unable to infer prompt count for task {task_name!r}")


def _weighted_metric(
    summaries: list[tuple[dict[str, Any], int]],
    dict_key: str,
    task_name: str,
) -> float | None:
    numerator = 0.0
    denominator = 0
    for summary, prompt_count in summaries:
        values = summary.get(dict_key, {})
        if not isinstance(values, dict) or task_name not in values:
            continue
        numerator += float(values[task_name]) * prompt_count
        denominator += prompt_count
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _max_metric(
    summaries: list[tuple[dict[str, Any], int]],
    dict_key: str,
    task_name: str,
) -> float | None:
    candidates: list[float] = []
    for summary, _ in summaries:
        values = summary.get(dict_key, {})
        if isinstance(values, dict) and task_name in values:
            candidates.append(float(values[task_name]))
    if not candidates:
        return None
    return float(max(candidates))


def _extend_records(
    summary: dict[str, Any],
    kind: str,
    records: list[dict[str, Any]],
) -> None:
    saved = summary.get("saved_output_paths", {})
    if not isinstance(saved, dict):
        return
    candidate = saved.get(kind)
    if not candidate:
        return
    path = Path(str(candidate))
    if not path.exists():
        return
    payload = _load_json(path)
    if not isinstance(payload, list):
        return
    for record in payload:
        if isinstance(record, dict):
            records.append(record)


def main() -> int:
    args = _parse_args()
    task_dir = Path(args.task_dir).resolve()
    task_dir.mkdir(parents=True, exist_ok=True)

    shard_summary_paths = _find_subshard_summaries(task_dir)
    shard_summaries = [_load_json(path) for path in shard_summary_paths]
    task_names = {
        _extract_task_name(summary, path)
        for summary, path in zip(shard_summaries, shard_summary_paths)
    }
    if len(task_names) != 1:
        raise RuntimeError(f"Expected exactly one task across subshards, found {task_names}")
    task_name = next(iter(task_names))

    summaries_with_counts = [
        (summary, _infer_prompt_count(summary, task_name))
        for summary in shard_summaries
    ]
    total_prompt_count = sum(prompt_count for _, prompt_count in summaries_with_counts)
    if total_prompt_count <= 0:
        raise RuntimeError(f"No prompts found across subshards in {task_dir}")

    task_ranges: list[dict[str, Any]] = []
    for summary in shard_summaries:
        ranges = summary.get("task_prompt_ranges", {})
        if isinstance(ranges, dict) and isinstance(ranges.get(task_name), dict):
            task_ranges.append(dict(ranges[task_name]))

    results_value = _weighted_metric(summaries_with_counts, "results", task_name)
    if results_value is None:
        raise RuntimeError(f"Missing results for task {task_name!r} in {task_dir}")
    pass8_value = _weighted_metric(summaries_with_counts, "pass_at_8", task_name)
    mean8_value = _weighted_metric(summaries_with_counts, "mean_at_8", task_name)
    avg_len_value = _weighted_metric(summaries_with_counts, "avg_lens", task_name)
    formatted_value = _weighted_metric(summaries_with_counts, "formatted", task_name)
    max_len_value = _max_metric(summaries_with_counts, "max_lens", task_name)

    merged_summary: dict[str, Any] = {
        "mode": "vllm_server",
        "merged_from_prompt_shards": True,
        "task_name": task_name,
        "task_prompt_counts": {task_name: int(total_prompt_count)},
        "results": {task_name: float(results_value)},
        "avg": float(results_value),
        "avg_lens": {task_name: float(avg_len_value or 0.0)},
        "max_lens": {task_name: float(max_len_value or 0.0)},
        "formatted": (
            {task_name: float(formatted_value)}
            if formatted_value is not None
            else {}
        ),
        "process_exit_code": max(
            int(summary.get("process_exit_code", 0) or 0) for summary in shard_summaries
        ),
        "subshard_summary_paths": [str(path) for path in shard_summary_paths],
    }
    if pass8_value is not None:
        merged_summary["pass_at_8"] = {task_name: float(pass8_value)}
        merged_summary["pass_at_8_avg"] = float(pass8_value)
    else:
        merged_summary["pass_at_8"] = {}
        merged_summary["pass_at_8_avg"] = None
    if mean8_value is not None:
        merged_summary["mean_at_8"] = {task_name: float(mean8_value)}
        merged_summary["mean_at_8_avg"] = float(mean8_value)
    else:
        merged_summary["mean_at_8"] = {}
        merged_summary["mean_at_8_avg"] = None
    if task_ranges:
        merged_summary["task_prompt_ranges"] = {
            task_name: {
                "start": min(int(entry.get("start", 0) or 0) for entry in task_ranges),
                "end": max(int(entry.get("end", 0) or 0) for entry in task_ranges),
                "available": max(
                    int(entry.get("available", 0) or 0) for entry in task_ranges
                ),
            }
        }

    first_summary = shard_summaries[0]
    if isinstance(first_summary.get("pass_at_8_config"), dict):
        merged_summary["pass_at_8_config"] = first_summary["pass_at_8_config"]
    if "vllm_batch_size" in first_summary:
        merged_summary["vllm_batch_size"] = int(first_summary["vllm_batch_size"])
    if "vllm_url" in first_summary:
        merged_summary["vllm_url"] = first_summary["vllm_url"]

    single_records: list[dict[str, Any]] = []
    pass8_records: list[dict[str, Any]] = []
    for summary in shard_summaries:
        _extend_records(summary, "single", single_records)
        _extend_records(summary, "pass_at_8", pass8_records)

    saved_output_paths: dict[str, str] = {}
    if single_records:
        single_path = task_dir / "seed_paper_eval_outputs_single_n1.json"
        _write_json(single_path, _sort_output_records(single_records))
        saved_output_paths["single"] = str(single_path)
    if pass8_records:
        pass8_path = task_dir / "seed_paper_eval_outputs_pass_at_8_n8.json"
        _write_json(pass8_path, _sort_output_records(pass8_records))
        saved_output_paths["pass_at_8"] = str(pass8_path)
    if saved_output_paths:
        merged_summary["saved_output_paths"] = saved_output_paths

    summary_path = task_dir / args.summary_name
    _write_json(summary_path, merged_summary)
    print(f"Merged task summary: {summary_path}")
    print(json.dumps(merged_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
