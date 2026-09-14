#!/usr/bin/env python3
"""Build the exact response-free E117 Stage 1 table and run the v11 analyzer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import e117_stage1_statistics as statistics


CHECKPOINTS = statistics.CHECKPOINTS
DRAWS = statistics.EVALUATION_DRAWS
SEED_BASE = 117900
SAMPLED_KIND = "fixed_seed_sampled_k_neutral"
PROMPT_FIELDS = (
    "answer_mode_count",
    "option_ids",
    "prompt",
    "prompt_index",
    "reference",
)
REQUEST_FIELDS = ("option_ids", "prompt_index", "request_seeds_by_option")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def native_finite(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{where} must be a native JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{where} is non-finite")
    return result


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
            json.dump(payload, sink, indent=2, sort_keys=True)
            sink.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
            for row in rows:
                sink.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                sink.write("\n")
                count += 1
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return count


def _projection(
    row: dict[str, Any], *, path: Path, line_number: int
) -> tuple[str, str]:
    prompts = row.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != 128:
        raise RuntimeError(f"{path}:{line_number}: expected exactly 128 prompts")
    if any(not isinstance(prompt, dict) for prompt in prompts):
        raise RuntimeError(f"{path}:{line_number}: malformed prompt surface")
    try:
        prompt_projection = [
            {field: prompt[field] for field in PROMPT_FIELDS} for prompt in prompts
        ]
        request_projection = [
            {field: prompt[field] for field in REQUEST_FIELDS} for prompt in prompts
        ]
    except KeyError as error:
        raise RuntimeError(f"{path}:{line_number}: incomplete response-free identity") from error
    expected_seed = int(row["seed"])
    for prompt_index, prompt in enumerate(request_projection):
        seeds = prompt["request_seeds_by_option"]
        if seeds != [expected_seed]:
            raise RuntimeError(
                f"{path}:{line_number}: prompt {prompt_index} request seed "
                f"{seeds!r} != {[expected_seed]!r}"
            )
    return (
        canonical_sha256(prompt_projection),
        canonical_sha256({"row_seed": expected_seed, "prompts": request_projection}),
    )


def sampled_run_rows(run: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    run_dir = Path(str(run["run_dir"]))
    expected = {(step, draw) for step in CHECKPOINTS for draw in DRAWS}
    observed: dict[tuple[int, int], dict[str, Any]] = {}
    sources: dict[str, str] = {}
    paths = sorted(run_dir.glob("debug_job*/eval_mode_coverage_draws.jsonl"))
    if not paths:
        raise RuntimeError(f"{run_dir}: no evaluation sidecar exists")
    for path in paths:
        used = False
        with path.open(encoding="utf-8", errors="strict") as source:
            for line_number, raw in enumerate(source, start=1):
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if row.get("evaluation_kind") != SAMPLED_KIND:
                    continue
                step = row.get("step")
                draw = row.get("draw_index")
                if type(step) is not int or type(draw) is not int:
                    raise RuntimeError(f"{path}:{line_number}: non-integer sampled identity")
                key = (step, draw)
                if key not in expected:
                    raise RuntimeError(f"{path}:{line_number}: extra sampled row {key}")
                contract = {
                    "benchmark": "multi_answer",
                    "draw_index": draw,
                    "sample_count": 8,
                    "schema_version": 1,
                    "seed": SEED_BASE + draw,
                    "temperature": 1.0,
                }
                drift = {
                    field: (row.get(field), value)
                    for field, value in contract.items()
                    if row.get(field) != value
                }
                if drift:
                    raise RuntimeError(f"{path}:{line_number}: sampled contract drift: {drift}")
                metrics = row.get("metrics")
                if not isinstance(metrics, dict):
                    raise RuntimeError(f"{path}:{line_number}: sampled metrics absent")
                prompt_sha, request_sha = _projection(
                    row, path=path, line_number=line_number
                )
                normalized = {
                    "sentinel": f"{run['scale']}/{run['domain']}",
                    "training_seed": int(run["seed"]),
                    "arm": str(run["arm"]),
                    "checkpoint": step,
                    "evaluation_draw": draw,
                    "pass_at_8": native_finite(
                        metrics.get("any_correct_at_k"),
                        where=f"{path}:{line_number}:any_correct_at_k",
                    ),
                    "raw_distinct_at_8": native_finite(
                        metrics.get("distinct_correct_modes_at_k"),
                        where=f"{path}:{line_number}:distinct_correct_modes_at_k",
                    ),
                    "prompt_surface_sha256": prompt_sha,
                    "request_surface_sha256": request_sha,
                }
                if key in observed and observed[key] != normalized:
                    raise RuntimeError(f"{run_dir}: conflicting duplicate sampled row {key}")
                observed[key] = normalized
                used = True
        if used:
            sources[str(path.resolve())] = sha256(path)
    missing = expected.difference(observed)
    if missing:
        raise RuntimeError(
            f"{run_dir}: missing {len(missing)} sampled rows; first={min(missing)}"
        )
    return [observed[key] for key in sorted(observed)], sources


def build(
    ledger_path: Path,
    *,
    table_path: Path,
    output_path: Path,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    runs = list(ledger.get("runs", []))
    if len(runs) != 36 or ledger.get("released") is not True:
        raise RuntimeError("E117 Stage 1 ledger is not a released 36-cell grid")
    table: list[dict[str, Any]] = []
    sources: dict[str, str] = {}
    for run in runs:
        rows, run_sources = sampled_run_rows(run)
        table.extend(rows)
        sources.update(run_sources)
    expected_rows = 4 * 3 * 3 * len(CHECKPOINTS) * len(DRAWS)
    if len(table) != expected_rows:
        raise RuntimeError(f"Stage 1 table has {len(table)} rows, expected {expected_rows}")
    result = statistics.analyze_registered_table(table)
    count = atomic_jsonl(table_path, table)
    if count != expected_rows:
        raise RuntimeError("atomic Stage 1 table write lost rows")
    payload = {
        "schema": "e117_stage1_development_results_v1",
        "development_only": True,
        "confirmation_reserve_read": False,
        "ledger": str(ledger_path.resolve()),
        "ledger_sha256": sha256(ledger_path),
        "table": str(table_path.resolve()),
        "table_sha256": sha256(table_path),
        "table_rows": count,
        "source_sidecars": sources,
        "source_sidecar_count": len(sources),
        "identity_projection": {
            "prompt_fields": list(PROMPT_FIELDS),
            "request_fields": ["row_seed", *REQUEST_FIELDS],
            "excluded": ["answer_keys", "metrics", "responses", "rewards"],
        },
        "provenance": provenance or {},
        "analysis": result,
    }
    atomic_json(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(
        args.ledger.resolve(),
        table_path=args.table.resolve(),
        output_path=args.output.resolve(),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
