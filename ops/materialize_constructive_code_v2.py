#!/usr/bin/env python3
"""Materialize the frozen held-out Python-3-only ConstructiveCode v2 slate."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from huggingface_hub import HfApi

from audit_constructive_code_sources import (
    OVERLAY_REPO,
    OVERLAY_REVISION,
    PLUS_REPO,
    PLUS_REVISION,
    candidate_key,
    raw_sha256,
)
from materialize_constructive_code_review_slate import (
    OVERLAY_PAYLOAD_COLUMNS,
    PLUS_PAYLOAD_COLUMNS,
    _canonical_json_bytes,
    _overlay_inputs,
    _selected_index_records,
    _source_sizes,
    build_overlay_suite,
)
from materialize_constructive_code_plus_suites import (
    _plus_inputs,
    _read_selected_parquet_rows,
)


ROOT = Path(os.environ.get("OAT_ZERO_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
DEFAULT_INDEX = Path(os.environ.get("OAT_ZERO_CONSTRUCTIVE_V2_INDEX", ROOT / "var/artifacts/constructive_code_candidate_source_index.json")).resolve()
DEFAULT_V1 = ROOT / "var/data/constructive_code_review_slate_v1"
DEFAULT_OUTPUT = ROOT / "var/data/constructive_code_v2"
PROTOCOL = ROOT / "paper/preregistration/constructive_code_executable_slate_v2_20260729.md"
SLATE_SCHEMA = "constructive-code-slate-v2"
TASK_SCHEMA = "constructive-code-task-v2"
VERSION_LABEL = "constructive-v2"
LOGICAL_V1_ROOT: str | None = None
SPLIT_ASSIGNMENT: dict[str, str] = {}
V2_TASKS = {
    "327_B": ("ordered_sequence", "fixed_integer_sequence_v1"),
    "1294_C": ("unordered_set", "multi_case_status_integer_set_v1"),
    "1283_C": ("assignment", "implicit_assignment_v1"),
    "1102_B": ("unordered_partition", "status_label_partition_v1"),
}
PYTHON3_LABELS = frozenset({"py3", "python3", "pypy3"})
REPLAYS_PER_LABEL = 100
EXCLUDE_V1_HASHES = True
V1_LEDGER_COUNT_FIELD = "excluded_v1_hash_count"
SELECTION_DESCRIPTION = (
    "explicit Python-3 labels, v1 hashes excluded, unique SHA-256 "
    "ascending, 100 per known label"
)
PLUS_COLUMNS = tuple(PLUS_PAYLOAD_COLUMNS) + (
    "test_cases.list.element.input",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _execution_limits(row: Mapping[str, Any]) -> dict[str, int]:
    limits = {}
    for source_key, output_key in (
        ("time_limit", "time_milliseconds"),
        ("memory_limit", "memory_megabytes"),
    ):
        value = row.get(source_key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"invalid positive integer {source_key}: {value!r}")
        limits[output_key] = value
    return limits


def select_heldout_python3(
    submissions: Any,
    excluded_hashes: set[str],
    limit: int = REPLAYS_PER_LABEL,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Select unique explicit-Python-3 programs disjoint from v1."""

    by_hash: dict[str, dict[str, str]] = {}
    explicit_python3 = 0
    excluded = 0
    if isinstance(submissions, list):
        for submission in submissions:
            if not isinstance(submission, Mapping):
                continue
            language = str(submission.get("language") or "").strip().lower()
            code = submission.get("code")
            if language not in PYTHON3_LABELS or not isinstance(code, str):
                continue
            if not code.strip():
                continue
            explicit_python3 += 1
            digest = raw_sha256(code)
            if digest in excluded_hashes:
                excluded += 1
                continue
            by_hash.setdefault(
                digest,
                {
                    "submission_sha256": digest,
                    "language": language,
                    "code": code,
                },
            )
    ordered = [by_hash[digest] for digest in sorted(by_hash)]
    if len(ordered) < limit:
        raise ValueError(
            f"only {len(ordered)} held-out Python-3 programs; {limit} required"
        )
    return ordered[:limit], {
        "explicit_python3_records": explicit_python3,
        "excluded_v1_records": excluded,
        "available_unique_heldout_python3": len(ordered),
        "selected": limit,
    }


def _load_v1_hashes(path: Path) -> set[str]:
    hashes = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            hashes.add(str(row["submission_sha256"]))
    return hashes


def _read_selected_plus(
    selected: Mapping[str, Mapping[str, Any]],
    sizes: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    by_shard: dict[str, set[str]] = defaultdict(set)
    for problem_id, record in selected.items():
        by_shard[record["codecontests_plus"]["source_shard"]].add(problem_id)
    result: dict[str, dict[str, Any]] = {}
    for completed, (shard, ids) in enumerate(sorted(by_shard.items()), start=1):
        source_row_indices = [
            int(selected[problem_id]["codecontests_plus"]["source_row_index"])
            for problem_id in ids
        ]
        for row in _read_selected_parquet_rows(
            PLUS_REPO,
            PLUS_REVISION,
            shard,
            sizes[shard],
            PLUS_COLUMNS,
            source_row_indices,
        ):
            problem_id = str(row.get("id") or "")
            if problem_id not in ids:
                continue
            expected = selected[problem_id]
            if candidate_key(row) != expected["problem_key"]:
                raise RuntimeError(f"Plus identity drift for {problem_id}")
            if row["_source_row_index"] != expected["codecontests_plus"][
                "source_row_index"
            ]:
                raise RuntimeError(f"Plus row drift for {problem_id}")
            if raw_sha256(row.get("checker")) != expected["codecontests_plus"][
                "checker_raw_sha256"
            ]:
                raise RuntimeError(f"Plus checker drift for {problem_id}")
            result[problem_id] = row
        print(
            f"[constructive-v2] plus_shards={completed}/{len(by_shard)} "
            f"rows={len(result)}/{len(selected)}",
            flush=True,
        )
    if set(result) != set(selected):
        raise RuntimeError("not all selected Plus rows were recovered")
    return result


def _read_selected_overlay(
    selected: Mapping[str, Mapping[str, Any]],
    sizes: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    by_shard: dict[str, set[str]] = defaultdict(set)
    for problem_id, record in selected.items():
        by_shard[record["codecontests_o"]["source_shard"]].add(problem_id)
    result: dict[str, dict[str, Any]] = {}
    for completed, (shard, ids) in enumerate(sorted(by_shard.items()), start=1):
        source_row_indices = [
            int(selected[problem_id]["codecontests_o"]["source_row_index"])
            for problem_id in ids
        ]
        for row in _read_selected_parquet_rows(
            OVERLAY_REPO,
            OVERLAY_REVISION,
            shard,
            sizes[shard],
            OVERLAY_PAYLOAD_COLUMNS,
            source_row_indices,
        ):
            name = str(row.get("name") or "")
            matches = [
                problem_id
                for problem_id in ids
                if name == selected[problem_id]["codecontests_o"]["name"]
            ]
            if len(matches) != 1:
                continue
            problem_id = matches[0]
            expected = selected[problem_id]
            if row["_source_row_index"] != expected["codecontests_o"][
                "source_row_index"
            ]:
                raise RuntimeError(f"overlay row drift for {problem_id}")
            if raw_sha256(row.get("checker")) != expected["codecontests_o"][
                "checker_raw_sha256"
            ]:
                raise RuntimeError(f"overlay checker drift for {problem_id}")
            result[problem_id] = row
        print(
            f"[constructive-v2] overlay_shards={completed}/{len(by_shard)} "
            f"rows={len(result)}/{len(selected)}",
            flush=True,
        )
    if set(result) != set(selected):
        raise RuntimeError("not all selected overlay rows were recovered")
    return result


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> str:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return _sha256(path)


def _write_suite(path: Path, inputs: list[str]) -> dict[str, Any]:
    records, suite_sha256 = build_overlay_suite(inputs)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            for record in records:
                compressed.write(
                    (json.dumps(record, sort_keys=True) + "\n").encode("ascii")
                )
    return {
        "test_count": len(records),
        "suite_sha256": suite_sha256,
        "compressed_jsonl_sha256": _sha256(path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--v1-root", type=Path, default=DEFAULT_V1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output exists: {args.output}")
    index = json.loads(args.index.read_text(encoding="utf-8"))
    all_selected = _selected_index_records(index)
    selected = {problem_id: all_selected[problem_id] for problem_id in V2_TASKS}
    api = HfApi()
    plus_info = api.dataset_info(
        PLUS_REPO, revision=PLUS_REVISION, files_metadata=True
    )
    overlay_info = api.dataset_info(
        OVERLAY_REPO, revision=OVERLAY_REVISION, files_metadata=True
    )
    if plus_info.sha != PLUS_REVISION or overlay_info.sha != OVERLAY_REVISION:
        raise RuntimeError("source revision drift")
    plus = _read_selected_plus(selected, _source_sizes(plus_info))
    overlay = _read_selected_overlay(selected, _source_sizes(overlay_info))

    staging = args.output.with_name(f"{args.output.name}.tmp.{os.getpid()}")
    staging.mkdir(parents=True)
    tasks = []
    for problem_id, (family, adapter) in V2_TASKS.items():
        relative = problem_id.lower()
        task_dir = staging / relative
        task_dir.mkdir()
        v1_hashes = _load_v1_hashes(
            args.v1_root / relative / "python_replays.jsonl"
        )
        excluded = v1_hashes if EXCLUDE_V1_HASHES else set()
        correct, correct_counts = select_heldout_python3(
            plus[problem_id].get("correct_submissions"),
            excluded,
            limit=REPLAYS_PER_LABEL,
        )
        incorrect, incorrect_counts = select_heldout_python3(
            plus[problem_id].get("incorrect_submissions"),
            excluded,
            limit=REPLAYS_PER_LABEL,
        )
        replays = [
            {"known_label": "correct", **row} for row in correct
        ] + [{"known_label": "incorrect", **row} for row in incorrect]
        replays.sort(
            key=lambda row: (row["known_label"], row["submission_sha256"])
        )
        replays_sha256 = _write_jsonl(task_dir / "py3_replays.jsonl", replays)
        checker_path = task_dir / "checker.cpp"
        checker_path.write_text(str(plus[problem_id].get("checker") or ""))
        overlay_suite = _write_suite(
            task_dir / "overlay_inputs.jsonl.gz",
            _overlay_inputs(overlay[problem_id]),
        )
        plus_suite = _write_suite(
            task_dir / "plus_5x_inputs.jsonl.gz",
            _plus_inputs(plus[problem_id]),
        )
        task = {
            "schema_version": TASK_SCHEMA,
            "problem_key": selected[problem_id]["problem_key"],
            "source_problem_id": problem_id,
            "title": selected[problem_id]["title"],
            "statement": selected[problem_id]["statement"],
            "witness_family": family,
            "task_adapter": adapter,
            "limits": _execution_limits(plus[problem_id]),
            "language_contract": {
                "accepted_labels": sorted(PYTHON3_LABELS),
                "runtime": "Python 3.10.20",
                V1_LEDGER_COUNT_FIELD: len(v1_hashes),
            },
            "replays": {
                "correct": correct_counts,
                "incorrect": incorrect_counts,
                "jsonl_sha256": replays_sha256,
            },
            "checker_sha256": _sha256(checker_path),
            "suites": {
                "codecontests_o_corner_cases_v2": overlay_suite,
                "codecontests_plus_5x_v2": plus_suite,
            },
            "source_locations": {
                "codecontests_plus": selected[problem_id]["codecontests_plus"],
                "codecontests_o": selected[problem_id]["codecontests_o"],
            },
        }
        task["task_record_sha256"] = _canonical_sha256(task)
        (task_dir / "task.json").write_text(
            json.dumps(task, indent=2, sort_keys=True) + "\n"
        )
        tasks.append(
            {
                "problem_key": task["problem_key"],
                "source_problem_id": problem_id,
                "relative_path": relative,
                "witness_family": family,
                "task_adapter": adapter,
                "task_record_sha256": task["task_record_sha256"],
                "overlay_test_count": overlay_suite["test_count"],
                "plus_5x_test_count": plus_suite["test_count"],
                "correct_replays": len(correct),
                "incorrect_replays": len(incorrect),
            }
        )
    manifest = {
        "schema_version": SLATE_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending_executable_replay",
        "selection": SELECTION_DESCRIPTION,
        "sources": {
            "codecontests_plus": {
                "repo": PLUS_REPO,
                "revision": PLUS_REVISION,
            },
            "codecontests_o": {
                "repo": OVERLAY_REPO,
                "revision": OVERLAY_REVISION,
            },
            "candidate_index_sha256": _sha256(args.index),
            "v1_root": LOGICAL_V1_ROOT or str(args.v1_root.relative_to(ROOT)),
        },
        "preregistration_sha256": _sha256(PROTOCOL),
        "split_assignment": SPLIT_ASSIGNMENT,
        "tasks": tasks,
        "tasks_sha256": _canonical_sha256(tasks),
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    staging.replace(args.output)
    print(
        f"[{VERSION_LABEL}] status={manifest['status']} "
        f"tasks={len(tasks)} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
