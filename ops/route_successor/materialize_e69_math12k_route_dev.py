#!/usr/bin/env python3
"""Materialize E69's frozen MATH12K-384 train and MATH12K-128 route dev.

MATH-500 participates only as a sealed set of normalized problem identities for
the leakage firewall. No MATH-500 row is written to the output.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from ops.math500 import materialize_e39_math12k_384 as e39
from oat_drgrpo.templates import apply_qwen_math_route_json_v1_template


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "var/data/math12k_384_route_dev128_v1"
MANIFEST_NAME = "MATERIALIZATION_MANIFEST.json"
SCHEMA = "e69_math12k_384_route_dev128_materialization_v1"
SELECTION_SEED = "e69-verified-route-dev-v1"
TRAIN_ROWS = 384
DEV_ROWS = 128
PROMPT_MAX_LENGTH = 1_024
DEV_SPLIT = "math_dev"
EXPECTED_SELECTION_INDEX_HASH = (
    "327d843e7da08a1e0298ca3b623176fc78d54a3757f7917459bad78896115b17"
)
EXPECTED_DEV_ORDERED_ROW_HASH = (
    "1a3b5c2cae6dfd28faae68b8de480477e4e11ce53ad562dfdc2ffe3a01b5d229"
)
EXPECTED_DEV_ORDERED_PROBLEM_HASH = (
    "2f37f517a6ab6badce6c8d2fcc4e05483449a06d3aee89d87b5c4ecb7400fa43"
)
EXPECTED_DEV_PROMPT_LENGTH_HASH = (
    "d3e26c3484358b699b2825d283ef0a40126de1a039fa660b87335aeec81c628b"
)


class MaterializationError(RuntimeError):
    """Raised when the sealed E69 data contract does not hold."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _text_sha256(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _row_identity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    row_hashes = [_json_sha256(dict(row)) for row in rows]
    problem_hashes = [_text_sha256(row["problem"]) for row in rows]
    return {
        "ordered_row_sha256": _json_sha256(row_hashes),
        "ordered_problem_sha256": _json_sha256(problem_hashes),
        "row_sha256": row_hashes,
        "problem_sha256": problem_hashes,
    }


def _stratum(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["subject"]), str(int(row["level"]))


def largest_remainder_quotas(
    counts: Mapping[tuple[str, str], int],
    total: int,
) -> dict[tuple[str, str], int]:
    """Allocate ``total`` proportionally with deterministic lexical ties."""

    clean = {tuple(key): int(value) for key, value in counts.items() if value > 0}
    population = sum(clean.values())
    if total < 1 or total > population:
        raise MaterializationError("invalid stratified allocation size")
    exact = {key: total * value / population for key, value in clean.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remaining = total - sum(quotas.values())
    order = sorted(
        clean,
        key=lambda key: (-(exact[key] - quotas[key]), key),
    )
    for key in order[:remaining]:
        quotas[key] += 1
    if sum(quotas.values()) != total:
        raise MaterializationError("largest-remainder allocation failed")
    if any(quotas[key] > clean[key] for key in clean):
        raise MaterializationError("a stratum quota exceeds its population")
    return quotas


def _selection_priority(source_index: int, normalized_problem: str) -> str:
    payload = (
        f"{SELECTION_SEED}\0{int(source_index)}\0{normalized_problem}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prompt_lengths(problems: Sequence[str]) -> list[int]:
    e39._verify_files(e39.TOKENIZER_ROOT, e39.TOKENIZER_FILES, label="tokenizer")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise MaterializationError("transformers is required") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        str(e39.TOKENIZER_ROOT),
        local_files_only=True,
    )
    encoded = tokenizer(
        [apply_qwen_math_route_json_v1_template(problem) for problem in problems],
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    return [len(tokens) for tokens in encoded]


def select_dev_indices(
    source_rows: Sequence[Mapping[str, Any]],
    sealed_math500_rows: Sequence[Mapping[str, Any]],
    *,
    prompt_lengths: Sequence[int],
) -> tuple[list[int], dict[str, Any]]:
    """Select the deterministic, stratified, leakage-free dev population."""

    if len(source_rows) != e39.TRAIN_SOURCE_ROWS:
        raise MaterializationError("MATH12K source row count drifted")
    if len(prompt_lengths) != len(source_rows):
        raise MaterializationError("prompt-length vector has the wrong size")

    normalized_train = {
        e39.normalize_problem(row["problem"]) for row in source_rows[:TRAIN_ROWS]
    }
    normalized_sealed = {
        e39.normalize_problem(row["problem"]) for row in sealed_math500_rows
    }
    seen = set(normalized_train)
    eligible: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    excluded = Counter(
        {
            "blank_problem": 0,
            "blank_answer": 0,
            "train_overlap": 0,
            "sealed_math500_overlap": 0,
            "duplicate_normalized_problem": 0,
            "prompt_too_long": 0,
        }
    )
    for source_index in range(TRAIN_ROWS, len(source_rows)):
        row = source_rows[source_index]
        normalized = e39.normalize_problem(row["problem"])
        if not normalized:
            excluded["blank_problem"] += 1
            continue
        if not str(row["answer"]).strip():
            excluded["blank_answer"] += 1
            continue
        if normalized in normalized_train:
            excluded["train_overlap"] += 1
            continue
        if normalized in normalized_sealed:
            excluded["sealed_math500_overlap"] += 1
            continue
        if normalized in seen:
            excluded["duplicate_normalized_problem"] += 1
            continue
        if int(prompt_lengths[source_index]) > PROMPT_MAX_LENGTH:
            excluded["prompt_too_long"] += 1
            continue
        seen.add(normalized)
        eligible[_stratum(row)].append(
            (_selection_priority(source_index, normalized), source_index)
        )

    counts = {key: len(values) for key, values in eligible.items()}
    quotas = largest_remainder_quotas(counts, DEV_ROWS)
    selected: list[int] = []
    for key in sorted(eligible):
        candidates = sorted(eligible[key])
        selected.extend(index for _priority, index in candidates[: quotas[key]])
    selected.sort()
    if len(selected) != DEV_ROWS or len(set(selected)) != DEV_ROWS:
        raise MaterializationError("dev selection is not exactly 128 unique rows")

    audit = {
        "candidate_source_indices": [TRAIN_ROWS, len(source_rows) - 1],
        "eligible_rows": sum(counts.values()),
        "eligible_stratum_counts": {
            f"{subject}|{level}": counts[(subject, level)]
            for subject, level in sorted(counts)
        },
        "excluded_counts": dict(sorted(excluded.items())),
        "quotas": {
            f"{subject}|{level}": quotas[(subject, level)]
            for subject, level in sorted(quotas)
        },
        "selected_source_indices": selected,
        "selected_source_indices_sha256": _json_sha256(selected),
    }
    return selected, audit


def _source_bundle() -> tuple[
    dict[str, Any],
    Any,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[int],
    dict[str, Any],
]:
    source, source_dataset, _eval_dataset, train_rows, sealed_rows = (
        e39._source_bundle()
    )
    source_rows = [dict(row) for row in source_dataset]
    lengths = _prompt_lengths([str(row["problem"]) for row in source_rows])
    selected_indices, selection_audit = select_dev_indices(
        source_rows,
        sealed_rows,
        prompt_lengths=lengths,
    )
    dev_rows = [source_rows[index] for index in selected_indices]
    if train_rows != source_rows[:TRAIN_ROWS]:
        raise MaterializationError("E69 train rows differ from frozen E39 train")
    if selection_audit["selected_source_indices_sha256"] != (
        EXPECTED_SELECTION_INDEX_HASH
    ):
        raise MaterializationError("frozen E69 development selection drifted")
    normalized_dev = {
        e39.normalize_problem(row["problem"]) for row in dev_rows
    }
    normalized_sealed = {
        e39.normalize_problem(row["problem"]) for row in sealed_rows
    }
    if normalized_dev.intersection(normalized_sealed):
        raise MaterializationError("development/MATH-500 problem leakage")
    return (
        source,
        source_dataset,
        train_rows,
        dev_rows,
        lengths,
        selection_audit,
    )


def _population_audit(
    train_rows: Sequence[Mapping[str, Any]],
    dev_rows: Sequence[Mapping[str, Any]],
    *,
    dev_prompt_lengths: Sequence[int],
    selection_audit: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_train = {
        e39.normalize_problem(row["problem"]) for row in train_rows
    }
    normalized_dev = [e39.normalize_problem(row["problem"]) for row in dev_rows]
    if len(normalized_train) != TRAIN_ROWS:
        raise MaterializationError("training problems are not unique")
    if len(set(normalized_dev)) != DEV_ROWS:
        raise MaterializationError("development problems are not unique")
    if normalized_train.intersection(normalized_dev):
        raise MaterializationError("train/development problem leakage")
    if max(dev_prompt_lengths) > PROMPT_MAX_LENGTH:
        raise MaterializationError("development prompt exceeds token admission")
    subject_counts = dict(
        sorted(Counter(str(row["subject"]) for row in dev_rows).items())
    )
    level_counts = dict(
        sorted(
            Counter(str(int(row["level"])) for row in dev_rows).items(),
            key=lambda item: int(item[0]),
        )
    )
    stratum_counts = dict(
        sorted(
            (
                f"{subject}|{level}",
                count,
            )
            for (subject, level), count in Counter(
                _stratum(row) for row in dev_rows
            ).items()
        )
    )
    dev_identity = _row_identity(dev_rows)
    prompt_length_hash = _json_sha256(dev_prompt_lengths)
    expected = {
        "development ordered rows": (
            EXPECTED_DEV_ORDERED_ROW_HASH,
            dev_identity["ordered_row_sha256"],
        ),
        "development ordered problems": (
            EXPECTED_DEV_ORDERED_PROBLEM_HASH,
            dev_identity["ordered_problem_sha256"],
        ),
        "development prompt lengths": (
            EXPECTED_DEV_PROMPT_LENGTH_HASH,
            prompt_length_hash,
        ),
    }
    for label, (frozen, observed) in expected.items():
        if frozen != observed:
            raise MaterializationError(
                f"{label} drifted: expected={frozen} observed={observed}"
            )
    return {
        "train_rows": len(train_rows),
        "train_identity": _row_identity(train_rows),
        "dev_rows": len(dev_rows),
        "dev_identity": dev_identity,
        "dev_subject_counts": subject_counts,
        "dev_level_counts": level_counts,
        "dev_stratum_counts": stratum_counts,
        "dev_prompt_admission": {
            "template": "qwen_math_route",
            "prompt_max_length": PROMPT_MAX_LENGTH,
            "minimum_tokens": min(dev_prompt_lengths),
            "maximum_tokens": max(dev_prompt_lengths),
            "mean_tokens": sum(dev_prompt_lengths) / len(dev_prompt_lengths),
            "ordered_token_lengths_sha256": prompt_length_hash,
        },
        "train_dev_normalized_problem_overlap": 0,
        "selection": dict(selection_audit),
    }


def _output_file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): e39.sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != MANIFEST_NAME
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _load_output_rows(output_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:  # pragma: no cover
        raise MaterializationError("datasets is required") from exc
    train_root = load_from_disk(str(output_root / "train"))
    eval_root = load_from_disk(str(output_root / "eval"))
    if not isinstance(train_root, DatasetDict) or set(train_root) != {"train"}:
        raise MaterializationError("materialized train split drifted")
    if not isinstance(eval_root, DatasetDict) or set(eval_root) != {DEV_SPLIT}:
        raise MaterializationError("materialized route-dev split drifted")
    return (
        [dict(row) for row in train_root["train"]],
        [dict(row) for row in eval_root[DEV_SPLIT]],
    )


def materialize(output_root: Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Build and atomically publish the sealed E69 DatasetDict pair."""

    try:
        from datasets import DatasetDict
    except ImportError as exc:  # pragma: no cover
        raise MaterializationError("datasets is required") from exc

    output_root = output_root.resolve()
    if output_root.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output_root}; pass --overwrite to replace it"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    (
        source,
        source_dataset,
        train_rows,
        dev_rows,
        all_prompt_lengths,
        selection_audit,
    ) = _source_bundle()
    selected_indices = selection_audit["selected_source_indices"]
    dev_prompt_lengths = [all_prompt_lengths[index] for index in selected_indices]
    audit = _population_audit(
        train_rows,
        dev_rows,
        dev_prompt_lengths=dev_prompt_lengths,
        selection_audit=selection_audit,
    )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    backup: Path | None = None
    try:
        DatasetDict(
            {"train": source_dataset.select(range(TRAIN_ROWS))}
        ).save_to_disk(str(staging / "train"))
        DatasetDict(
            {DEV_SPLIT: source_dataset.select(selected_indices)}
        ).save_to_disk(str(staging / "eval"))
        observed_train, observed_dev = _load_output_rows(staging)
        if observed_train != train_rows or observed_dev != dev_rows:
            raise MaterializationError("materialized rows differ from selection")
        manifest = {
            "schema": SCHEMA,
            "selection_rule": {
                "seed": SELECTION_SEED,
                "candidate_source_indices": [TRAIN_ROWS, e39.TRAIN_SOURCE_ROWS - 1],
                "strata": ["subject", "level"],
                "allocation": "largest_remainder_proportional",
                "within_stratum_order": "sha256(seed,source_index,normalized_problem)",
                "output_order": "source_index",
                "train_split": "train",
                "dev_split": DEV_SPLIT,
            },
            "sealed_math500_firewall": {
                "policy": (
                    "identity-only overlap check; no MATH-500 row or score is "
                    "materialized"
                ),
                "rows": e39.EVAL_ROWS,
                "ordered_problem_sha256": e39.EVAL_ORDERED_PROBLEM_HASH,
                "normalized_problem_overlap": 0,
            },
            "source": source,
            "audit": audit,
            "output": {
                "root": _repo_relative(output_root),
                "train_splits": ["train"],
                "train_rows": len(observed_train),
                "dev_splits": [DEV_SPLIT],
                "dev_rows": len(observed_dev),
                "files_sha256": _output_file_hashes(staging),
            },
        }
        _atomic_write_json(staging / MANIFEST_NAME, manifest)

        if output_root.exists():
            backup = output_root.parent / f".{output_root.name}.backup.{os.getpid()}"
            if backup.exists():
                raise FileExistsError(f"backup path already exists: {backup}")
            os.replace(output_root, backup)
        try:
            os.replace(staging, output_root)
        except Exception:
            if backup is not None and backup.exists() and not output_root.exists():
                os.replace(backup, output_root)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def audit_materialized(output_root: Path) -> dict[str, Any]:
    """Fail closed unless an existing materialization matches the frozen source."""

    output_root = output_root.resolve()
    manifest_path = output_root / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationError("materialization manifest is invalid") from exc
    if manifest.get("schema") != SCHEMA:
        raise MaterializationError("materialization schema drifted")
    (
        source,
        _source_dataset,
        train_rows,
        dev_rows,
        all_prompt_lengths,
        selection_audit,
    ) = _source_bundle()
    selected_indices = selection_audit["selected_source_indices"]
    expected_audit = _population_audit(
        train_rows,
        dev_rows,
        dev_prompt_lengths=[all_prompt_lengths[index] for index in selected_indices],
        selection_audit=selection_audit,
    )
    observed_train, observed_dev = _load_output_rows(output_root)
    if observed_train != train_rows or observed_dev != dev_rows:
        raise MaterializationError("materialized population drifted")
    if manifest.get("source") != source or manifest.get("audit") != expected_audit:
        raise MaterializationError("materialization provenance drifted")
    output = manifest.get("output")
    if not isinstance(output, dict):
        raise MaterializationError("materialization output record is missing")
    expected_output = {
        "root": _repo_relative(output_root),
        "train_splits": ["train"],
        "train_rows": TRAIN_ROWS,
        "dev_splits": [DEV_SPLIT],
        "dev_rows": DEV_ROWS,
        "files_sha256": _output_file_hashes(output_root),
    }
    if output != expected_output:
        raise MaterializationError("materialized file hashes drifted")
    return expected_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.audit_only:
        audit = audit_materialized(args.output_root)
        print(json.dumps(audit, indent=2, sort_keys=True))
        return
    manifest = materialize(args.output_root, overwrite=args.overwrite)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
