#!/usr/bin/env python3
"""Materialize a bounded ConstructiveCode schema-review slate.

Only selected Python replay programs and final CodeContests-O input probes are
written. CodeContests+ 5x tests, overlay result histories, reference outputs,
and unselected submissions remain remote.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from huggingface_hub import HfApi

from audit_constructive_code_sources import (
    OVERLAY_REPO,
    OVERLAY_REVISION,
    PLUS_REPO,
    PLUS_REVISION,
    _read_parquet_shard,
    candidate_key,
    raw_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "var/artifacts/constructive_code_candidate_source_index.json"
DEFAULT_OUTPUT = ROOT / "var/data/constructive_code_review_slate_v1"
SLATE_SCHEMA = "constructive-code-review-slate-v1"

PLUS_PAYLOAD_COLUMNS = (
    "source",
    "id",
    "title",
    "time_limit",
    "memory_limit",
    "validator",
    "generator",
    "generator_cmd",
    "checker",
    "correct_submissions.list.element.code",
    "correct_submissions.list.element.language",
    "incorrect_submissions.list.element.code",
    "incorrect_submissions.list.element.language",
)
OVERLAY_PAYLOAD_COLUMNS = (
    "name",
    "checker",
    "corner_cases.list.element.input.stdin",
)
_PYTHON_LANGUAGES = frozenset(
    {"py", "py2", "py3", "python", "python2", "python3", "pypy", "pypy3"}
)
_SHA256 = re.compile(r"[0-9a-f]{64}")

# This is an antecedent schema-review slate, not an admitted benchmark split.
SLATE: dict[str, tuple[str, str]] = {
    "327_B": ("ordered_sequence", "fixed_integer_sequence_v1"),
    "359_B": ("ordered_sequence", "fixed_integer_sequence_v1"),
    "361_B": ("ordered_sequence", "sentinel_integer_sequence_v1"),
    "482_A": ("ordered_sequence", "fixed_integer_sequence_v1"),
    "659_C": ("unordered_set", "counted_integer_set_v1"),
    "988_A": ("unordered_set", "status_integer_set_v1"),
    "1294_C": ("unordered_set", "multi_case_status_integer_set_v1"),
    "1516_C": ("unordered_set", "counted_integer_set_v1"),
    "1153_B": ("assignment", "matrix_assignment_v1"),
    "1208_C": ("assignment", "matrix_assignment_v1"),
    "1283_C": ("assignment", "implicit_assignment_v1"),
    "1408_A": ("assignment", "multi_case_implicit_assignment_v1"),
    "1051_B": ("unordered_partition", "status_pair_partition_v1"),
    "1102_B": ("unordered_partition", "status_label_partition_v1"),
    "1399_D": ("unordered_partition", "multi_case_label_partition_v1"),
    "149_C": ("unordered_partition", "two_group_partition_v1"),
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def select_python_replays(
    submissions: Any,
    limit: int,
) -> tuple[list[dict[str, str]], int]:
    """Deduplicate by exact code hash and take a deterministic bounded sample."""

    if limit < 1:
        raise ValueError("replay limit must be positive")
    by_hash: dict[str, dict[str, str]] = {}
    if isinstance(submissions, list):
        for submission in submissions:
            if not isinstance(submission, Mapping):
                continue
            language = str(submission.get("language") or "").strip().lower()
            code = submission.get("code")
            if language not in _PYTHON_LANGUAGES or not isinstance(code, str):
                continue
            if not code.strip():
                continue
            digest = raw_sha256(code)
            by_hash.setdefault(
                digest,
                {
                    "submission_sha256": digest,
                    "language": language,
                    "code": code,
                },
            )
    ordered = [by_hash[digest] for digest in sorted(by_hash)]
    return ordered[:limit], len(ordered)


def build_overlay_suite(inputs: Iterable[str]) -> tuple[list[dict[str, Any]], str]:
    records: list[dict[str, Any]] = []
    for index, value in enumerate(inputs):
        if not isinstance(value, str):
            raise ValueError("overlay input must be text")
        encoded = value.encode("utf-8")
        records.append(
            {
                "test_index": index,
                "input_sha256": _sha256_bytes(encoded),
                "input_bytes": len(encoded),
                "stdin": value,
            }
        )
    if not records:
        raise ValueError("overlay suite is empty")
    identity = [
        {
            "test_index": record["test_index"],
            "input_sha256": record["input_sha256"],
            "input_bytes": record["input_bytes"],
        }
        for record in records
    ]
    return records, _sha256_bytes(_canonical_json_bytes(identity))


def _index_record_digest(records: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(_canonical_json_bytes(list(records)))


def _source_sizes(info: Any) -> dict[str, int]:
    return {sibling.rfilename: int(sibling.size or 0) for sibling in info.siblings}


def _selected_index_records(index: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records = index.get("records")
    if not isinstance(records, list):
        raise ValueError("candidate index records are missing")
    if index.get("record_count") != len(records):
        raise ValueError("candidate index record count mismatch")
    if index.get("records_sha256") != _index_record_digest(records):
        raise ValueError("candidate index records SHA-256 mismatch")
    by_id = {
        str(record.get("source_problem_id") or ""): record
        for record in records
        if isinstance(record, dict)
    }
    missing = sorted(set(SLATE) - set(by_id))
    if missing:
        raise ValueError(f"review slate IDs absent from candidate index: {missing}")
    return {problem_id: by_id[problem_id] for problem_id in SLATE}


def _read_selected_plus_rows(
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
        rows = _read_parquet_shard(
            PLUS_REPO,
            PLUS_REVISION,
            shard,
            file_sizes[shard],
            PLUS_PAYLOAD_COLUMNS,
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
            if (
                raw_sha256(row.get("checker"))
                != expected["codecontests_plus"]["checker_raw_sha256"]
            ):
                raise RuntimeError(f"Plus checker drift for {problem_id}")
            rows_by_id[problem_id] = row
        print(
            "[constructive-code-materialize] "
            f"plus_shards={completed}/{len(ids_by_shard)} "
            f"selected_rows={len(rows_by_id)}/{len(selected)}",
            flush=True,
        )
    if set(rows_by_id) != set(selected):
        raise RuntimeError("not all review-slate rows were recovered from Plus")
    return rows_by_id


def _read_selected_overlay_rows(
    selected: Mapping[str, Mapping[str, Any]],
    file_sizes: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    ids_by_shard: dict[str, set[str]] = defaultdict(set)
    for problem_id, record in selected.items():
        ids_by_shard[record["codecontests_o"]["source_shard"]].add(problem_id)

    rows_by_id: dict[str, dict[str, Any]] = {}
    for completed, (shard, problem_ids) in enumerate(
        sorted(ids_by_shard.items()),
        start=1,
    ):
        rows = _read_parquet_shard(
            OVERLAY_REPO,
            OVERLAY_REVISION,
            shard,
            file_sizes[shard],
            OVERLAY_PAYLOAD_COLUMNS,
        )
        for row in rows:
            name = str(row.get("name") or "")
            matching_ids = [
                problem_id
                for problem_id in problem_ids
                if name == selected[problem_id]["codecontests_o"]["name"]
            ]
            if not matching_ids:
                continue
            if len(matching_ids) != 1:
                raise RuntimeError(f"ambiguous overlay identity in {shard}")
            problem_id = matching_ids[0]
            expected = selected[problem_id]
            if (
                row["_source_row_index"]
                != expected["codecontests_o"]["source_row_index"]
            ):
                raise RuntimeError(f"overlay row-location drift for {problem_id}")
            if (
                raw_sha256(row.get("checker"))
                != expected["codecontests_o"]["checker_raw_sha256"]
            ):
                raise RuntimeError(f"overlay checker drift for {problem_id}")
            rows_by_id[problem_id] = row
        print(
            "[constructive-code-materialize] "
            f"overlay_shards={completed}/{len(ids_by_shard)} "
            f"selected_rows={len(rows_by_id)}/{len(selected)}",
            flush=True,
        )
    if set(rows_by_id) != set(selected):
        raise RuntimeError("not all review-slate rows were recovered from overlay")
    return rows_by_id


def _overlay_inputs(row: Mapping[str, Any]) -> list[str]:
    cases = row.get("corner_cases")
    if not isinstance(cases, list):
        raise ValueError("overlay corner cases are missing")
    inputs: list[str] = []
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("overlay corner case is malformed")
        input_record = case.get("input")
        if not isinstance(input_record, Mapping):
            raise ValueError("overlay corner-case input is malformed")
        stdin = input_record.get("stdin")
        if not isinstance(stdin, str):
            raise ValueError("overlay stdin is not text")
        inputs.append(stdin)
    return inputs


def _write_text(path: Path, value: str) -> str:
    path.write_text(value, encoding="utf-8")
    return _sha256_bytes(path.read_bytes())


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> str:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    return _sha256_bytes(path.read_bytes())


def _write_gzip_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> str:
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as gzip_handle:
            for record in records:
                line = (
                    json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n"
                ).encode("ascii")
                gzip_handle.write(line)
    return _sha256_bytes(path.read_bytes())


def _write_payload(
    output: Path,
    index: Mapping[str, Any],
    selected: Mapping[str, Mapping[str, Any]],
    plus_rows: Mapping[str, Mapping[str, Any]],
    overlay_rows: Mapping[str, Mapping[str, Any]],
    replay_limit: int,
) -> dict[str, Any]:
    staging = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    staging.mkdir(parents=True)

    task_records: list[dict[str, Any]] = []
    for problem_id in SLATE:
        source = selected[problem_id]
        plus = plus_rows[problem_id]
        overlay = overlay_rows[problem_id]
        family, adapter = SLATE[problem_id]
        task_dir = staging / problem_id.lower()
        task_dir.mkdir()

        correct, available_correct = select_python_replays(
            plus.get("correct_submissions"), replay_limit
        )
        incorrect, available_incorrect = select_python_replays(
            plus.get("incorrect_submissions"), replay_limit
        )
        replay_records = [
            {"known_label": "correct", **record} for record in correct
        ] + [{"known_label": "incorrect", **record} for record in incorrect]
        replay_records.sort(
            key=lambda record: (
                record["known_label"],
                record["submission_sha256"],
            )
        )
        submissions_sha = _write_jsonl(
            task_dir / "python_replays.jsonl", replay_records
        )

        overlay_tests, overlay_suite_sha = build_overlay_suite(_overlay_inputs(overlay))
        overlay_tests_sha = _write_gzip_jsonl(
            task_dir / "overlay_inputs.jsonl.gz", overlay_tests
        )
        checker_sha = _write_text(
            task_dir / "checker.cpp", str(plus.get("checker") or "")
        )
        validator_sha = _write_text(
            task_dir / "validator.cpp", str(plus.get("validator") or "")
        )
        generator_sha = _write_text(
            task_dir / "generator.cpp", str(plus.get("generator") or "")
        )

        task = {
            "schema_version": "constructive-code-review-task-v1",
            "problem_key": source["problem_key"],
            "source_problem_id": problem_id,
            "title": source["title"],
            "witness_family": family,
            "task_adapter": adapter,
            "admission_status": "pending_executable_equivalence",
            "statement": source["statement"],
            "limits": {
                "time_milliseconds": plus.get("time_limit"),
                "memory_megabytes": plus.get("memory_limit"),
            },
            "source_locations": {
                "codecontests_plus": source["codecontests_plus"],
                "codecontests_o": source["codecontests_o"],
            },
            "source_hashes": {
                "checker_cpp_sha256": checker_sha,
                "validator_cpp_sha256": validator_sha,
                "generator_cpp_sha256": generator_sha,
                "generator_command_sha256": raw_sha256(plus.get("generator_cmd")),
            },
            "replays": {
                "available_unique_correct_python": available_correct,
                "available_unique_incorrect_python": available_incorrect,
                "selected_correct_python": len(correct),
                "selected_incorrect_python": len(incorrect),
                "selection_rule": (
                    "unique exact-code SHA-256 ascending, capped per label"
                ),
                "python_replays_jsonl_sha256": submissions_sha,
            },
            "overlay_suite": {
                "suite_id": "codecontests_o_corner_cases_v1",
                "test_count": len(overlay_tests),
                "suite_sha256": overlay_suite_sha,
                "compressed_jsonl_sha256": overlay_tests_sha,
                "reference_outputs_materialized": False,
            },
            "pending_gates": [
                "task adapter implementation and adversarial identity tests",
                "released checker compilation and replay equivalence",
                "CodeContests+ 5x test replay",
                "isolated execution throughput",
                "two distinct accepted behavior keys",
            ],
        }
        task_sha = _sha256_bytes(_canonical_json_bytes(task))
        task["task_record_sha256"] = task_sha
        (task_dir / "task.json").write_text(
            json.dumps(task, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        task_records.append(
            {
                "problem_key": source["problem_key"],
                "source_problem_id": problem_id,
                "relative_path": str(task_dir.relative_to(staging)),
                "task_record_sha256": task_sha,
                "witness_family": family,
                "task_adapter": adapter,
                "overlay_test_count": len(overlay_tests),
                "selected_correct_python": len(correct),
                "selected_incorrect_python": len(incorrect),
            }
        )

    manifest = {
        "schema_version": SLATE_SCHEMA,
        "status": "pending_executable_equivalence",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision_boundary": (
            "This bounded review package authorizes isolation and checker "
            "development only. It does not admit ConstructiveCode tasks or "
            "authorize policy sampling."
        ),
        "sources": {
            "codecontests_plus": {
                "repo": PLUS_REPO,
                "revision": PLUS_REVISION,
                "license": "CC-BY-4.0",
            },
            "codecontests_o": {
                "repo": OVERLAY_REPO,
                "revision": OVERLAY_REVISION,
                "license": "Apache-2.0",
            },
        },
        "candidate_index": {
            "schema_version": index["schema_version"],
            "records_sha256": index["records_sha256"],
            "candidate_keys_sha256": index["source_audit"]["candidate_keys_sha256"],
        },
        "access_contract": {
            "selected_problem_count": len(task_records),
            "python_replays_per_label_cap": replay_limit,
            "overlay_reference_outputs_materialized": False,
            "codecontests_plus_5x_tests_materialized": False,
            "overlay_iteration_histories_materialized": False,
            "unselected_submission_code_materialized": False,
        },
        "family_counts": {
            family: sum(record["witness_family"] == family for record in task_records)
            for family in (
                "ordered_sequence",
                "unordered_set",
                "assignment",
                "unordered_partition",
            )
        },
        "task_count": len(task_records),
        "tasks_sha256": _sha256_bytes(_canonical_json_bytes(task_records)),
        "tasks": task_records,
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staging.replace(output)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replays-per-label", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replays_per_label < 1:
        raise ValueError("--replays-per-label must be positive")
    index = json.loads(args.candidate_index.read_text(encoding="utf-8"))
    if not isinstance(index, dict):
        raise ValueError("candidate index must be a JSON object")
    selected = _selected_index_records(index)

    api = HfApi()
    plus_info = api.dataset_info(PLUS_REPO, revision=PLUS_REVISION, files_metadata=True)
    overlay_info = api.dataset_info(
        OVERLAY_REPO, revision=OVERLAY_REVISION, files_metadata=True
    )
    if plus_info.sha != PLUS_REVISION or overlay_info.sha != OVERLAY_REVISION:
        raise RuntimeError("source revision drift")
    plus_rows = _read_selected_plus_rows(selected, _source_sizes(plus_info))
    overlay_rows = _read_selected_overlay_rows(selected, _source_sizes(overlay_info))
    manifest = _write_payload(
        args.output,
        index,
        selected,
        plus_rows,
        overlay_rows,
        args.replays_per_label,
    )
    print(
        "[constructive-code-materialize] "
        f"status={manifest['status']} "
        f"tasks={manifest['task_count']} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
