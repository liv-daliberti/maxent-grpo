#!/usr/bin/env python3
"""Materialize only frozen CodeContests+ 5x inputs for the review slate."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import aiohttp
import fsspec
from huggingface_hub import HfApi
import pyarrow.parquet as pq

from audit_constructive_code_sources import (
    PLUS_REPO,
    PLUS_REVISION,
    _remote_url,
    candidate_key,
)
from materialize_constructive_code_review_slate import (
    SLATE,
    _canonical_json_bytes,
    _selected_index_records,
    _sha256_bytes,
    _source_sizes,
    _write_gzip_jsonl,
    build_overlay_suite,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "var/artifacts/constructive_code_candidate_source_index.json"
DEFAULT_OUTPUT = ROOT / "var/data/constructive_code_plus_5x_suites_v1"
SCHEMA_VERSION = "constructive-code-plus-5x-suites-v1"
PLUS_SUITE_COLUMNS = (
    "source",
    "id",
    "title",
    "test_cases.list.element.input",
)
HTTP_TIMEOUT = aiohttp.ClientTimeout(
    total=None,
    connect=60,
    sock_read=30 * 60,
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _plus_inputs(row: Mapping[str, Any]) -> list[str]:
    cases = row.get("test_cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("CodeContests+ 5x cases are missing")
    inputs: list[str] = []
    for case in cases:
        if not isinstance(case, Mapping) or not isinstance(case.get("input"), str):
            raise ValueError("CodeContests+ 5x input is malformed")
        inputs.append(case["input"])
    return inputs


def _selected_row_groups(
    row_group_sizes: Sequence[int],
    row_indices: Sequence[int],
) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    """Map frozen global row indices to Parquet groups and local offsets."""

    selected = sorted(set(row_indices))
    if not selected or any(index < 0 for index in selected):
        raise ValueError("selected Parquet row indices must be non-negative")
    result: list[tuple[int, int, tuple[int, ...]]] = []
    selected_position = 0
    group_start = 0
    for group_index, raw_size in enumerate(row_group_sizes):
        group_size = int(raw_size)
        if group_size < 0:
            raise ValueError("Parquet row-group size cannot be negative")
        group_end = group_start + group_size
        local_indices: list[int] = []
        while (
            selected_position < len(selected)
            and selected[selected_position] < group_end
        ):
            source_index = selected[selected_position]
            if source_index < group_start:
                raise ValueError("selected Parquet row index is inconsistent")
            local_indices.append(source_index - group_start)
            selected_position += 1
        if local_indices:
            result.append((group_index, group_start, tuple(local_indices)))
        group_start = group_end
    if selected_position != len(selected):
        raise ValueError("selected Parquet row index is outside the shard")
    return tuple(result)


def _read_selected_parquet_rows(
    repo: str,
    revision: str,
    path: str,
    file_size: int,
    columns: Sequence[str],
    row_indices: Sequence[int],
    attempts: int = 8,
) -> list[dict[str, Any]]:
    url = _remote_url(repo, revision, path)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            filesystem = fsspec.filesystem(
                "http",
                client_kwargs={"timeout": HTTP_TIMEOUT},
            )
            with filesystem.open(
                url,
                "rb",
                block_size=64 * 1024,
                cache_type="readahead",
                size=file_size,
            ) as handle:
                parquet = pq.ParquetFile(handle)
                row_group_sizes = [
                    parquet.metadata.row_group(index).num_rows
                    for index in range(parquet.num_row_groups)
                ]
                groups = _selected_row_groups(row_group_sizes, row_indices)
                rows: list[dict[str, Any]] = []
                for group_index, group_start, local_indices in groups:
                    wanted = set(local_indices)
                    last_wanted = max(wanted)
                    local_index = 0
                    for batch in parquet.iter_batches(
                        batch_size=1,
                        row_groups=[group_index],
                        columns=list(columns),
                        use_threads=False,
                    ):
                        if local_index in wanted:
                            row = batch.to_pylist()[0]
                            row["_source_shard"] = path
                            row["_source_row_index"] = group_start + local_index
                            rows.append(row)
                        if local_index >= last_wanted:
                            break
                        local_index += 1
            if len(rows) != len(set(row_indices)):
                raise RuntimeError("selected Parquet rows were not recovered exactly")
            return rows
        except Exception as error:  # pragma: no cover - network fault path
            last_error = error
            if attempt + 1 < attempts:
                delay = min(60.0, 5.0 * (2**attempt))
                delay += random.uniform(0.0, 2.0)
                print(
                    "[constructive-code-plus-suites] "
                    f"retry={attempt + 1}/{attempts} shard={path} "
                    f"delay={delay:.1f}s error={type(error).__name__}",
                    flush=True,
                )
                time.sleep(delay)
    raise RuntimeError(f"failed to read selected rows from {path}") from last_error


def _read_selected_rows(
    selected: Mapping[str, Mapping[str, Any]],
    file_sizes: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    ids_by_shard: dict[str, set[str]] = defaultdict(set)
    for problem_id, record in selected.items():
        ids_by_shard[record["codecontests_plus"]["source_shard"]].add(problem_id)
    rows_by_id: dict[str, dict[str, Any]] = {}
    for completed, (shard, problem_ids) in enumerate(
        sorted(ids_by_shard.items()),
        start=1,
    ):
        source_row_indices = [
            int(selected[problem_id]["codecontests_plus"]["source_row_index"])
            for problem_id in problem_ids
        ]
        rows = _read_selected_parquet_rows(
            PLUS_REPO,
            PLUS_REVISION,
            shard,
            file_sizes[shard],
            PLUS_SUITE_COLUMNS,
            source_row_indices,
        )
        for row in rows:
            problem_id = str(row.get("id") or "")
            if problem_id not in problem_ids:
                continue
            expected = selected[problem_id]
            if candidate_key(row) != expected["problem_key"]:
                raise RuntimeError(f"Plus identity drift for {problem_id}")
            if (
                row["_source_row_index"]
                != expected["codecontests_plus"]["source_row_index"]
            ):
                raise RuntimeError(f"Plus row-location drift for {problem_id}")
            rows_by_id[problem_id] = row
        print(
            "[constructive-code-plus-suites] "
            f"shards={completed}/{len(ids_by_shard)} "
            f"selected_rows={len(rows_by_id)}/{len(selected)}",
            flush=True,
        )
    if set(rows_by_id) != set(selected):
        raise RuntimeError("not all Plus suite rows were recovered")
    return rows_by_id


def materialize(index_path: Path, output: Path) -> dict[str, Any]:
    index = _load_json(index_path)
    candidates = _selected_index_records(index)
    selected = {problem_id: candidates[problem_id] for problem_id in SLATE}
    info = HfApi().dataset_info(
        PLUS_REPO,
        revision=PLUS_REVISION,
        files_metadata=True,
    )
    if info.sha != PLUS_REVISION:
        raise RuntimeError("CodeContests+ revision drift")
    file_sizes = _source_sizes(info)
    rows = _read_selected_rows(selected, file_sizes)

    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    staging = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    staging.mkdir(parents=True)
    tasks: list[dict[str, Any]] = []
    for problem_id in SLATE:
        task_dir = staging / problem_id.lower()
        task_dir.mkdir()
        records, suite_sha256 = build_overlay_suite(_plus_inputs(rows[problem_id]))
        compressed_sha256 = _write_gzip_jsonl(
            task_dir / "plus_5x_inputs.jsonl.gz",
            records,
        )
        tasks.append(
            {
                "problem_key": selected[problem_id]["problem_key"],
                "source_problem_id": problem_id,
                "relative_path": problem_id.lower(),
                "suite_id": "codecontests_plus_5x_v1",
                "suite_sha256": suite_sha256,
                "test_count": len(records),
                "compressed_jsonl_sha256": compressed_sha256,
                "checker_sha256": selected[problem_id]["codecontests_plus"][
                    "checker_raw_sha256"
                ],
                "source_shard": selected[problem_id]["codecontests_plus"][
                    "source_shard"
                ],
                "source_row_index": selected[problem_id]["codecontests_plus"][
                    "source_row_index"
                ],
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "input_only_pending_replay",
        "source": {
            "repo": PLUS_REPO,
            "revision": PLUS_REVISION,
            "candidate_index_sha256": _sha256_bytes(index_path.read_bytes()),
        },
        "excluded_payloads": [
            "reference outputs",
            "unselected submissions",
            "non-slate problem rows",
        ],
        "tasks": tasks,
        "tasks_sha256": _sha256_bytes(_canonical_json_bytes(tasks)),
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staging.replace(output)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = materialize(args.index, args.output)
    print(
        "[constructive-code-plus-suites] "
        f"status={manifest['status']} tasks={len(manifest['tasks'])} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
