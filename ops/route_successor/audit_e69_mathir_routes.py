#!/usr/bin/env python3
"""Audit exact endpoint and cross-prompt route support for E69 MathIR."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from oat_drgrpo.mathir import (
    enumerate_mathir_action_menu_validations,
    validate_mathir_action_menu,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = ROOT / "var/data/mathir_action_menu_v1"
DEFAULT_OUTPUT = ROOT / "var/artifacts/e69_mathir_route_offline_audit.json"
EXPECTED_FAMILIES = {
    "ax_plus_b_eq_c",
    "ax_plus_b_eq_c_minus_dx",
    "ax_plus_b_eq_dx_plus_c",
    "x_over_a_plus_b_eq_c",
}
EXPECTED_SPLIT_ROWS = {"train": 384, "eval": 128}
EXPECTED_FAMILY_SPLIT_ROWS = {"train": 96, "eval": 32}
EXPECTED_SUPPORT = 5


class AuditError(RuntimeError):
    """Raised when the frozen MathIR route-support contract fails."""


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _audit_one(record: Mapping[str, Any]) -> dict[str, Any]:
    spec = json.loads(str(record["answer"]))
    validations = enumerate_mathir_action_menu_validations(spec)
    endpoint_keys = sorted(
        {validation.canonical_key for validation in validations}
    )
    route_signatures = sorted(
        {validation.route_signature for validation in validations}
    )
    violations: list[str] = []
    if len(validations) != EXPECTED_SUPPORT:
        violations.append(
            f"validation_count={len(validations)} expected={EXPECTED_SUPPORT}"
        )
    if len(endpoint_keys) != EXPECTED_SUPPORT:
        violations.append(
            f"endpoint_support={len(endpoint_keys)} expected={EXPECTED_SUPPORT}"
        )
    if len(route_signatures) != EXPECTED_SUPPORT:
        violations.append(
            f"route_support={len(route_signatures)} expected={EXPECTED_SUPPORT}"
        )
    for validation in validations:
        replay = validate_mathir_action_menu(
            ";".join(validation.action_ids),
            spec,
        )
        if replay is None:
            violations.append("enumerated program failed public validator replay")
            continue
        if replay.canonical_key != validation.canonical_key:
            violations.append("endpoint key changed on public validator replay")
        if replay.route_signature != validation.route_signature:
            violations.append("route signature changed on public validator replay")
    return {
        "split": str(record["split"]),
        "row_index": int(record["row_index"]),
        "family": str(record["family"]),
        "endpoint_keys": endpoint_keys,
        "route_signatures": route_signatures,
        "programs": [
            ";".join(validation.action_ids) for validation in validations
        ],
        "violations": violations,
    }


def _load_records(data_root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:  # pragma: no cover
        raise AuditError("datasets is required") from exc
    records: list[dict[str, Any]] = []
    split_roots = {
        "train": (data_root / "train", "train"),
        "eval": (data_root / "eval", "multi_answer"),
    }
    for split, (root, dataset_split) in split_roots.items():
        loaded = load_from_disk(str(root))
        if not isinstance(loaded, DatasetDict) or set(loaded) != {dataset_split}:
            raise AuditError(f"{split} DatasetDict schema drifted")
        dataset = loaded[dataset_split]
        if len(dataset) != EXPECTED_SPLIT_ROWS[split]:
            raise AuditError(f"{split} row count drifted")
        for row_index, row in enumerate(dataset):
            records.append(
                {
                    "split": split,
                    "row_index": row_index,
                    "family": row["mathir_family"],
                    "answer": row["answer"],
                }
            )
    files = {
        path.relative_to(data_root).as_posix(): sha256(path)
        for path in sorted(data_root.rglob("*"))
        if path.is_file()
    }
    return records, files


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    by_family_split: dict[tuple[str, str], list[set[str]]] = defaultdict(list)
    endpoint_support = Counter()
    route_support = Counter()
    for result in results:
        if result["violations"]:
            violations.append(
                {
                    "split": result["split"],
                    "row_index": result["row_index"],
                    "family": result["family"],
                    "violations": result["violations"],
                }
            )
        key = (result["family"], result["split"])
        by_family_split[key].append(set(result["route_signatures"]))
        endpoint_support[len(result["endpoint_keys"])] += 1
        route_support[len(result["route_signatures"])] += 1

    observed_families = {family for family, _split in by_family_split}
    if observed_families != EXPECTED_FAMILIES:
        violations.append(
            {
                "scope": "families",
                "observed": sorted(observed_families),
                "expected": sorted(EXPECTED_FAMILIES),
            }
        )

    family_summaries: dict[str, Any] = {}
    for family in sorted(EXPECTED_FAMILIES):
        split_summary: dict[str, Any] = {}
        split_unions: dict[str, set[str]] = {}
        for split in ("train", "eval"):
            support_sets = by_family_split[(family, split)]
            if len(support_sets) != EXPECTED_FAMILY_SPLIT_ROWS[split]:
                violations.append(
                    {
                        "scope": "family_split_rows",
                        "family": family,
                        "split": split,
                        "observed": len(support_sets),
                        "expected": EXPECTED_FAMILY_SPLIT_ROWS[split],
                    }
                )
            union = set().union(*support_sets) if support_sets else set()
            intersection = (
                set(support_sets[0]).intersection(*support_sets[1:])
                if support_sets
                else set()
            )
            frequency = Counter(
                signature
                for support in support_sets
                for signature in support
            )
            split_unions[split] = union
            split_summary[split] = {
                "prompts": len(support_sets),
                "distinct_support_sets": len(
                    {frozenset(support) for support in support_sets}
                ),
                "union_support": len(union),
                "intersection_support": len(intersection),
                "minimum_signature_frequency": min(frequency.values(), default=0),
                "union_signature_sha256": _json_hash(sorted(union)),
                "intersection_signature_sha256": _json_hash(
                    sorted(intersection)
                ),
            }
            if not intersection:
                violations.append(
                    {
                        "scope": "cross_prompt_recurrence",
                        "family": family,
                        "split": split,
                        "reason": "empty route-signature intersection",
                    }
                )
        cross_split = split_unions["train"] & split_unions["eval"]
        split_summary["cross_split"] = {
            "overlap_support": len(cross_split),
            "overlap_signature_sha256": _json_hash(sorted(cross_split)),
        }
        if not cross_split:
            violations.append(
                {
                    "scope": "cross_split_recurrence",
                    "family": family,
                    "reason": "no train/eval route-signature overlap",
                }
            )
        family_summaries[family] = split_summary

    return {
        "status": "pass" if not violations else "fail",
        "prompts": len(results),
        "public_validator_replays": sum(
            len(result["programs"]) for result in results
        ),
        "endpoint_support_histogram": {
            str(key): value for key, value in sorted(endpoint_support.items())
        },
        "route_support_histogram": {
            str(key): value for key, value in sorted(route_support.items())
        },
        "families": family_summaries,
        "violation_count": len(violations),
        "violations": violations,
    }


def audit(data_root: Path, *, workers: int) -> dict[str, Any]:
    records, files = _load_records(data_root)
    if workers <= 1:
        results = [_audit_one(record) for record in records]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_audit_one, records, chunksize=4))
    summary = _summarize(results)
    return {
        "schema": "e69_mathir_route_offline_audit_v1",
        "data_root": _repo_relative(data_root),
        "input_files_sha256": files,
        "workers": workers,
        "summary": summary,
        "prompt_records_sha256": _json_hash(results),
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(args.data_root, workers=args.workers)
    _atomic_write(args.output, result)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    if result["summary"]["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
