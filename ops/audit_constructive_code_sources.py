#!/usr/bin/env python3
"""Count-only source audit for the proposed ConstructiveCode domain.

The audit intentionally projects narrow Parquet columns from pinned Hugging
Face revisions. It never downloads submission source, generated tests, or
iteration histories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import quote

import fsspec
import pyarrow.parquet as pq
from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "var/artifacts/constructive_code_source_overlap_audit.json"
DEFAULT_CANDIDATE_INDEX_OUTPUT = (
    ROOT / "var/artifacts/constructive_code_candidate_source_index.json"
)

PLUS_REPO = "ByteDance-Seed/Code-Contests-Plus"
PLUS_REVISION = "96c850540fade31d384a25766461e0da6b08f5fc"
OVERLAY_REPO = "caijanfeng/CodeContests-O"
OVERLAY_REVISION = "1a765191567b429f633bbd1c6e67b5890dfaf267"

PLUS_COLUMNS = (
    "source",
    "id",
    "title",
    "description",
    "checker",
    "correct_submissions.list.element.language",
    "true_positive_rate",
    "true_negative_rate",
)
OVERLAY_COLUMNS = ("name", "description", "checker")

_COMMENT_OR_LITERAL = re.compile(
    r"""
    //[^\n]*                       # line comment
    | /\*.*?\*/                    # block comment
    | "(?:\\.|[^"\\])*"           # string literal
    | '(?:\\.|[^'\\])*'           # character literal
    """,
    re.DOTALL | re.VERBOSE,
)
_PYTHON_LANGUAGES = frozenset(
    {"py", "py2", "py3", "python", "python2", "python3", "pypy", "pypy3"}
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def raw_sha256(value: Any) -> str:
    if value is None:
        value = ""
    return _sha256_text(str(value))


def normalize_text(value: Any) -> str:
    """Normalize source text for an alignment hash, not for semantic identity."""

    if value is None:
        return ""
    normalized = unicodedata.normalize("NFC", str(value))
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(normalized.split())


def normalized_sha256(value: Any) -> str:
    return _sha256_text(normalize_text(value))


def checker_is_reference_independent(checker: Any) -> bool:
    """Return a conservative lower bound on logic-based witness checkers.

    A candidate must inspect both the problem input and contestant output, and
    must not inspect the reference-answer stream. This omits valid constructive
    checkers that compare an objective value against ``ans``; those require
    manual semantic review.
    """

    if not isinstance(checker, str) or not checker.strip():
        return False
    code = _COMMENT_OR_LITERAL.sub(" ", checker)
    return (
        re.search(r"\binf\b", code) is not None
        and re.search(r"\bouf\b", code) is not None
        and re.search(r"\bans\b", code) is None
    )


def python_submission_count(submissions: Any) -> int:
    if not isinstance(submissions, list):
        return 0
    count = 0
    for submission in submissions:
        if not isinstance(submission, dict):
            continue
        language = str(submission.get("language") or "").strip().lower()
        if language in _PYTHON_LANGUAGES:
            count += 1
    return count


def plus_overlay_name(row: dict[str, Any]) -> str | None:
    """Build the stable ``id. title`` name exposed by CodeContests-O."""

    problem_id = str(row.get("id") or "").strip()
    title = str(row.get("title") or "").strip()
    if not problem_id or not title:
        return None
    return f"{problem_id}. {title}"


def _remote_url(repo: str, revision: str, path: str) -> str:
    encoded_path = quote(path, safe="/")
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{encoded_path}"


def _read_parquet_shard(
    repo: str,
    revision: str,
    path: str,
    file_size: int,
    columns: Sequence[str],
    attempts: int = 8,
) -> list[dict[str, Any]]:
    url = _remote_url(repo, revision, path)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            filesystem = fsspec.filesystem("http")
            with filesystem.open(
                url,
                "rb",
                block_size=64 * 1024,
                cache_type="readahead",
                size=file_size,
            ) as handle:
                table = pq.ParquetFile(handle).read(
                    columns=list(columns),
                    use_threads=False,
                )
            rows = table.to_pylist()
            for row_index, row in enumerate(rows):
                row["_source_shard"] = path
                row["_source_row_index"] = row_index
            return rows
        except Exception as error:  # pragma: no cover - exercised on network faults
            last_error = error
            if attempt + 1 < attempts:
                delay = min(60.0, 5.0 * (2**attempt))
                delay += random.uniform(0.0, 2.0)
                print(
                    f"[constructive-code-audit] retry={attempt + 1}/{attempts} "
                    f"shard={path} delay={delay:.1f}s "
                    f"error={type(error).__name__}",
                    flush=True,
                )
                time.sleep(delay)
    raise RuntimeError(f"failed to read projected columns from {path}") from last_error


def _read_all_shards(
    repo: str,
    revision: str,
    paths: Sequence[str],
    file_sizes: dict[str, int],
    columns: Sequence[str],
    workers: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {}
    try:
        futures = {
            executor.submit(
                _read_parquet_shard,
                repo,
                revision,
                path,
                file_sizes[path],
                columns,
            ): path
            for path in paths
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            shard_rows = future.result()
            rows.extend(shard_rows)
            progress_interval = 10 if len(paths) < 100 else 100
            if completed % progress_interval == 0 or completed == len(paths):
                print(
                    f"[constructive-code-audit] {repo}: "
                    f"{completed}/{len(paths)} shards, {len(rows)} rows",
                    flush=True,
                )
            if not shard_rows:
                raise RuntimeError(f"empty Parquet shard: {path}")
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return rows


def _repository_record(
    repo: str,
    requested_revision: str,
    info: Any,
    parquet_paths: Sequence[str],
) -> dict[str, Any]:
    sizes = {sibling.rfilename: int(sibling.size or 0) for sibling in info.siblings}
    return {
        "repo": repo,
        "requested_revision": requested_revision,
        "resolved_revision": info.sha,
        "repository_file_count": len(info.siblings),
        "repository_bytes": sum(sizes.values()),
        "projected_parquet_file_count": len(parquet_paths),
        "scanned_parquet_logical_bytes": sum(sizes[path] for path in parquet_paths),
    }


def _duplicate_count(values: Iterable[str]) -> int:
    return sum(count - 1 for count in Counter(values).values() if count > 1)


def candidate_key(row: dict[str, Any]) -> str:
    return f"{row.get('source')}:{row.get('id')}:{row.get('title')}"


def select_audited_candidate_pairs(
    plus_rows: Sequence[dict[str, Any]],
    overlay_rows: Sequence[dict[str, Any]],
    verified_threshold: float,
    minimum_python_submissions: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Return the exact candidate pairs counted by :func:`build_audit`."""

    overlay_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in overlay_rows:
        overlay_by_name[str(row.get("name") or "").strip()].append(row)
    plus_names = [plus_overlay_name(row) for row in plus_rows]
    plus_name_counts = Counter(name for name in plus_names if name)

    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row, overlay_name in zip(plus_rows, plus_names):
        tpr = row.get("true_positive_rate")
        tnr = row.get("true_negative_rate")
        if not (
            isinstance(tpr, (int, float))
            and isinstance(tnr, (int, float))
            and tpr >= verified_threshold
            and tnr >= verified_threshold
        ):
            continue
        if not checker_is_reference_independent(row.get("checker")):
            continue
        if (
            python_submission_count(row.get("correct_submissions"))
            < minimum_python_submissions
        ):
            continue
        if overlay_name is None or plus_name_counts[overlay_name] != 1:
            continue
        matches = overlay_by_name.get(overlay_name, [])
        if len(matches) != 1:
            continue
        overlay = matches[0]
        if normalized_sha256(row.get("description")) != normalized_sha256(
            overlay.get("description")
        ):
            continue
        if normalized_sha256(row.get("checker")) != normalized_sha256(
            overlay.get("checker")
        ):
            continue
        selected.append((row, overlay))
    return sorted(selected, key=lambda pair: candidate_key(pair[0]))


def build_candidate_index(
    candidate_pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    source_audit: dict[str, Any],
) -> dict[str, Any]:
    """Materialize only statement/checker metadata for manual schema review."""

    keys = [candidate_key(plus) for plus, _ in candidate_pairs]
    expected_count = int(source_audit["audited_candidate_count"])
    expected_digest = str(source_audit["audited_candidate_keys_sha256"])
    actual_digest = _sha256_text(
        json.dumps(keys, separators=(",", ":"), ensure_ascii=True)
    )
    if len(keys) != expected_count or actual_digest != expected_digest:
        raise RuntimeError("candidate index differs from the count-only source audit")

    records: list[dict[str, Any]] = []
    for plus, overlay in candidate_pairs:
        statement = str(plus.get("description") or "")
        checker = str(plus.get("checker") or "")
        overlay_statement = str(overlay.get("description") or "")
        overlay_checker = str(overlay.get("checker") or "")
        overlay_shard = str(overlay.get("_source_shard") or "")
        split_match = re.search(r"/(train|valid|test)-", overlay_shard)
        record = {
            "problem_key": candidate_key(plus),
            "source": str(plus.get("source") or ""),
            "source_problem_id": str(plus.get("id") or ""),
            "title": str(plus.get("title") or ""),
            "statement": statement,
            "checker": checker,
            "codecontests_plus": {
                "source_shard": str(plus.get("_source_shard") or ""),
                "source_row_index": int(plus.get("_source_row_index", -1)),
                "true_positive_rate": plus.get("true_positive_rate"),
                "true_negative_rate": plus.get("true_negative_rate"),
                "correct_python_submission_count": python_submission_count(
                    plus.get("correct_submissions")
                ),
                "statement_raw_sha256": raw_sha256(statement),
                "statement_normalized_sha256": normalized_sha256(statement),
                "checker_raw_sha256": raw_sha256(checker),
                "checker_normalized_sha256": normalized_sha256(checker),
            },
            "codecontests_o": {
                "name": str(overlay.get("name") or ""),
                "split": split_match.group(1) if split_match else "unknown",
                "source_shard": overlay_shard,
                "source_row_index": int(overlay.get("_source_row_index", -1)),
                "statement_raw_sha256": raw_sha256(overlay_statement),
                "statement_normalized_sha256": normalized_sha256(overlay_statement),
                "checker_raw_sha256": raw_sha256(overlay_checker),
                "checker_normalized_sha256": normalized_sha256(overlay_checker),
            },
            "manual_review": {
                "status": "pending",
                "witness_family": None,
                "task_adapter_version": None,
            },
        }
        records.append(record)

    records_digest = _sha256_text(
        json.dumps(records, separators=(",", ":"), ensure_ascii=True, sort_keys=True)
    )
    return {
        "schema_version": 1,
        "status": "pending_manual_schema_review",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision_boundary": (
            "Selected statement/checker metadata only. A row remains inadmissible "
            "until task-specific witness schema review and executable checker "
            "equivalence pass."
        ),
        "source_audit": {
            "schema_version": source_audit["schema_version"],
            "candidate_count": expected_count,
            "candidate_keys_sha256": expected_digest,
            "sources": source_audit["sources"],
            "thresholds": source_audit["thresholds"],
        },
        "access_contract": {
            "included_payloads": [
                "problem identity",
                "statement",
                "checker",
                "source shard and row location",
                "quality rates",
                "correct Python submission count",
            ],
            "excluded_payloads": source_audit["access_contract"]["excluded_payloads"],
            "full_dataset_materialized": False,
        },
        "record_count": len(records),
        "records_sha256": records_digest,
        "records": records,
    }


def build_audit(
    plus_rows: Sequence[dict[str, Any]],
    overlay_rows: Sequence[dict[str, Any]],
    plus_source: dict[str, Any],
    overlay_source: dict[str, Any],
    verified_threshold: float,
    minimum_python_submissions: int,
) -> dict[str, Any]:
    overlay_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    overlay_split_counts: Counter[str] = Counter()
    empty_overlay_names = 0
    for row in overlay_rows:
        name = str(row.get("name") or "").strip()
        if not name:
            empty_overlay_names += 1
        overlay_by_name[name].append(row)
        shard = str(row.get("_source_shard") or "")
        match = re.search(r"/(train|valid|test)-", shard)
        overlay_split_counts[match.group(1) if match else "unknown"] += 1

    funnel: Counter[str] = Counter()
    first_rejection: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    candidate_keys: list[str] = []
    raw_name_overlap = 0
    raw_statement_match = 0
    raw_checker_match = 0
    plus_names = [plus_overlay_name(row) for row in plus_rows]
    plus_name_counts = Counter(name for name in plus_names if name)

    for row, overlay_name in zip(plus_rows, plus_names):
        funnel["all_plus_5x_rows"] += 1
        source_counts[str(row.get("source") or "<missing>")] += 1
        stable_plus_name = (
            overlay_name is not None and plus_name_counts[overlay_name] == 1
        )
        matches = overlay_by_name.get(overlay_name, []) if stable_plus_name else []
        unique_overlay = matches[0] if len(matches) == 1 else None

        if unique_overlay is not None:
            raw_name_overlap += 1
            if normalized_sha256(row.get("description")) == normalized_sha256(
                unique_overlay.get("description")
            ):
                raw_statement_match += 1
            if normalized_sha256(row.get("checker")) == normalized_sha256(
                unique_overlay.get("checker")
            ):
                raw_checker_match += 1

        tpr = row.get("true_positive_rate")
        tnr = row.get("true_negative_rate")
        verified = (
            isinstance(tpr, (int, float))
            and isinstance(tnr, (int, float))
            and tpr >= verified_threshold
            and tnr >= verified_threshold
        )
        if not verified:
            first_rejection["not_codecontests_plus_verified"] += 1
            continue
        funnel["verified_tpr_tnr"] += 1

        if not checker_is_reference_independent(row.get("checker")):
            first_rejection["checker_requires_reference_or_is_ambiguous"] += 1
            continue
        funnel["reference_independent_checker_lower_bound"] += 1

        py_count = python_submission_count(row.get("correct_submissions"))
        if py_count < minimum_python_submissions:
            first_rejection["insufficient_correct_python_submissions"] += 1
            continue
        funnel["python_supported"] += 1

        if overlay_name is None:
            first_rejection["missing_plus_overlay_name"] += 1
            continue
        if not stable_plus_name:
            first_rejection["duplicate_plus_overlay_name"] += 1
            continue
        if len(matches) == 0:
            first_rejection["missing_overlay_name"] += 1
            continue
        if len(matches) > 1:
            first_rejection["duplicate_overlay_name"] += 1
            continue
        funnel["unique_cross_source_name_match"] += 1

        if normalized_sha256(row.get("description")) != normalized_sha256(
            unique_overlay.get("description")
        ):
            first_rejection["normalized_statement_hash_mismatch"] += 1
            continue
        funnel["normalized_statement_hash_match"] += 1

        if normalized_sha256(row.get("checker")) != normalized_sha256(
            unique_overlay.get("checker")
        ):
            first_rejection["normalized_checker_hash_mismatch"] += 1
            continue
        funnel["normalized_checker_hash_match"] += 1

        candidate_keys.append(candidate_key(row))

    candidate_keys.sort()
    plus_ids = [f"{row.get('source')}:{row.get('id')}" for row in plus_rows]
    overlay_names = [str(row.get("name") or "").strip() for row in overlay_rows]
    nonempty_overlay_names = [name for name in overlay_names if name]

    return {
        "schema_version": 2,
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision_boundary": (
            "Count-only source audit. No task is admitted to ConstructiveCode "
            "until checker execution, witness canonicalization, and local replay pass."
        ),
        "sources": {
            "codecontests_plus_5x": plus_source,
            "codecontests_o": overlay_source,
        },
        "access_contract": {
            "codecontests_plus_columns": list(PLUS_COLUMNS),
            "codecontests_o_columns": list(OVERLAY_COLUMNS),
            "excluded_payloads": [
                "correct_submissions.code",
                "incorrect_submissions",
                "test_cases",
                "corner_cases",
                "commands",
                "generator",
                "results",
            ],
            "full_dataset_materialized": False,
            "http_transfer_bytes_measured": False,
        },
        "thresholds": {
            "verified_tpr_tnr_minimum_inclusive": verified_threshold,
            "minimum_correct_python_submissions": minimum_python_submissions,
            "cross_source_join_rule": (
                "unique exact {id}. {title} name on both sources, followed by "
                "normalized statement and checker SHA-256 equality"
            ),
            "multiple_answer_rule": (
                "checker references inf and ouf but not ans after removing "
                "C/C++ comments and literals; conservative lower bound"
            ),
        },
        "row_integrity": {
            "codecontests_plus_5x_rows": len(plus_rows),
            "codecontests_plus_unique_source_ids": len(set(plus_ids)),
            "codecontests_plus_duplicate_source_ids": _duplicate_count(plus_ids),
            "codecontests_plus_unique_overlay_names": len(plus_name_counts),
            "codecontests_plus_duplicate_overlay_names": _duplicate_count(
                name for name in plus_names if name
            ),
            "codecontests_plus_source_counts": dict(sorted(source_counts.items())),
            "codecontests_o_rows": len(overlay_rows),
            "codecontests_o_empty_names": empty_overlay_names,
            "codecontests_o_unique_nonempty_names": len(set(nonempty_overlay_names)),
            "codecontests_o_duplicate_nonempty_names": _duplicate_count(
                nonempty_overlay_names
            ),
            "codecontests_o_split_counts": dict(sorted(overlay_split_counts.items())),
        },
        "raw_cross_source_alignment": {
            "unique_exact_name_matches": raw_name_overlap,
            "normalized_statement_hash_matches": raw_statement_match,
            "normalized_checker_hash_matches": raw_checker_match,
        },
        "eligibility_funnel": dict(funnel),
        "first_rejection_counts": dict(sorted(first_rejection.items())),
        "audited_candidate_count": len(candidate_keys),
        "audited_candidate_keys_sha256": _sha256_text(
            json.dumps(candidate_keys, separators=(",", ":"), ensure_ascii=True)
        ),
        "next_gate": (
            "Manual witness-schema review and executable checker equivalence; "
            "this audit alone does not authorize policy sampling."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--candidate-index-output",
        type=Path,
        help=(
            "also write selected statement/checker metadata for manual schema "
            "review; submission code and tests remain excluded"
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verified-threshold", type=float, default=0.9)
    parser.add_argument("--minimum-python-submissions", type=int, default=2)
    parser.add_argument("--plus-repo", default=PLUS_REPO)
    parser.add_argument("--plus-revision", default=PLUS_REVISION)
    parser.add_argument("--overlay-repo", default=OVERLAY_REPO)
    parser.add_argument("--overlay-revision", default=OVERLAY_REVISION)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if not 0.0 <= args.verified_threshold <= 1.0:
        raise ValueError("--verified-threshold must be in [0, 1]")
    if args.minimum_python_submissions < 1:
        raise ValueError("--minimum-python-submissions must be positive")

    api = HfApi()
    plus_info = api.dataset_info(
        args.plus_repo,
        revision=args.plus_revision,
        files_metadata=True,
    )
    overlay_info = api.dataset_info(
        args.overlay_repo,
        revision=args.overlay_revision,
        files_metadata=True,
    )
    if plus_info.sha != args.plus_revision:
        raise RuntimeError(
            f"CodeContests+ revision drift: {plus_info.sha} != {args.plus_revision}"
        )
    if overlay_info.sha != args.overlay_revision:
        raise RuntimeError(
            f"CodeContests-O revision drift: "
            f"{overlay_info.sha} != {args.overlay_revision}"
        )

    plus_paths = sorted(
        sibling.rfilename
        for sibling in plus_info.siblings
        if sibling.rfilename.startswith("ccplus_5x/")
        and sibling.rfilename.endswith(".parquet")
    )
    overlay_paths = sorted(
        sibling.rfilename
        for sibling in overlay_info.siblings
        if sibling.rfilename.startswith("data/")
        and sibling.rfilename.endswith(".parquet")
    )
    if not plus_paths or not overlay_paths:
        raise RuntimeError("expected pinned Parquet shards are missing")
    plus_sizes = {
        sibling.rfilename: int(sibling.size or 0) for sibling in plus_info.siblings
    }
    overlay_sizes = {
        sibling.rfilename: int(sibling.size or 0) for sibling in overlay_info.siblings
    }
    if any(plus_sizes[path] <= 0 for path in plus_paths):
        raise RuntimeError("CodeContests+ contains a shard with unknown size")
    if any(overlay_sizes[path] <= 0 for path in overlay_paths):
        raise RuntimeError("CodeContests-O contains a shard with unknown size")

    plus_rows = _read_all_shards(
        args.plus_repo,
        args.plus_revision,
        plus_paths,
        plus_sizes,
        PLUS_COLUMNS,
        args.workers,
    )
    overlay_rows = _read_all_shards(
        args.overlay_repo,
        args.overlay_revision,
        overlay_paths,
        overlay_sizes,
        OVERLAY_COLUMNS,
        args.workers,
    )

    audit = build_audit(
        plus_rows,
        overlay_rows,
        _repository_record(
            args.plus_repo,
            args.plus_revision,
            plus_info,
            plus_paths,
        ),
        _repository_record(
            args.overlay_repo,
            args.overlay_revision,
            overlay_info,
            overlay_paths,
        ),
        args.verified_threshold,
        args.minimum_python_submissions,
    )
    _write_json(args.output, audit)
    if args.candidate_index_output is not None:
        candidate_pairs = select_audited_candidate_pairs(
            plus_rows,
            overlay_rows,
            args.verified_threshold,
            args.minimum_python_submissions,
        )
        candidate_index = build_candidate_index(candidate_pairs, audit)
        _write_json(args.candidate_index_output, candidate_index)
    print(
        "[constructive-code-audit] "
        f"status={audit['status']} "
        f"plus={audit['row_integrity']['codecontests_plus_5x_rows']} "
        f"overlay={audit['row_integrity']['codecontests_o_rows']} "
        f"candidates={audit['audited_candidate_count']} "
        f"output={args.output}",
        flush=True,
    )
    if args.candidate_index_output is not None:
        print(
            f"[constructive-code-audit] candidate_index={args.candidate_index_output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
