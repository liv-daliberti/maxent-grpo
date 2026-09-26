#!/usr/bin/env python3
"""Read source-admitted response samples without estimating concentration.

The frozen trajectory snapshot decides scientific admission. This reader never
searches live run directories or selects a replacement endpoint. A stricter raw
sample check can mark a checkpoint unavailable, while its original cohort and
terminal-admission metadata remain intact. File appends are ignored beyond the
hashed frozen prefix. Both initial arms retain their independently recorded draws.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
# The original analysis stays bound to its preserved 74/67 census even after
# the manuscript's live snapshot advances. The extension has its own manifest.
DEFAULT_SNAPSHOT = ROOT / "paper/audits/conditional_concentration_20260911/source/training_curve_snapshot_initial.json"
METHODS = ("drgrpo", "replay_drgrpo", "maxrl", "replay_maxrl")
CELL_FIELDS = ("level", "scale", "domain", "method", "seed")
EVALUATION_KIND = "fixed_seed_sampled_k_neutral"


class SourceIntegrityError(RuntimeError):
    """Source bytes or response identities do not justify a sample estimate."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def resolve_source(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def prompt_identity(prompt: str, reference: Any) -> dict[str, str]:
    """Exact task+surface identity; never join solely by row/instance number."""
    reference = json.loads(reference) if isinstance(reference, str) else reference
    return {
        "prompt_id": digest(canonical_json({"prompt": prompt, "reference": reference}).encode()),
        "prompt_sha256": digest(prompt.encode()),
        "reference_sha256": digest(canonical_json(reference).encode()),
    }


def normalize_draw(row: dict, *, origin: dict | None = None) -> dict:
    """Normalize one intact K8/128-prompt draw; do not calculate new metrics.

    Failed responses can have canonical keys in historical telemetry. Their raw
    keys remain available, but ``verified_keys`` gates them by positive reward.
    """
    if (row.get("evaluation_kind") != EVALUATION_KIND
            or type(row.get("step")) is not int
            or type(row.get("draw_index")) is not int
            or row["draw_index"] not in range(4)
            or row.get("sample_count") != 8
            or row.get("temperature") != 1.0):
        raise SourceIntegrityError("draw violates the frozen sampled-K8 contract")
    if row.get("top_p") not in (None, 1.0):
        raise SourceIntegrityError("draw has a non-unit top_p outside this analysis")
    prompts = row.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != 128:
        raise SourceIntegrityError("draw must contain exactly 128 prompt records")
    normalized = []
    for prompt in prompts:
        index = prompt.get("prompt_index")
        if type(index) is not int or index not in range(128):
            raise SourceIntegrityError("prompt_index must be an integer in [0,128)")
        text, reference = prompt.get("prompt"), prompt.get("reference")
        if not isinstance(text, str) or reference is None:
            raise SourceIntegrityError("prompt or reference missing from raw samples")
        try:
            reference = json.loads(reference) if isinstance(reference, str) else reference
            identity = prompt_identity(text, reference)
        except (ValueError, TypeError) as exc:
            raise SourceIntegrityError("invalid canonical prompt/reference identity") from exc
        keys, rewards = prompt.get("answer_keys"), prompt.get("rewards")
        if not isinstance(keys, list) or not isinstance(rewards, list) or len(keys) != 8 or len(rewards) != 8:
            raise SourceIntegrityError("prompt must retain eight keys and eight rewards")
        verified = []
        for key, reward in zip(keys, rewards):
            if key is not None and (not isinstance(key, str) or not key):
                raise SourceIntegrityError("canonical key must be a nonempty string or null")
            if (isinstance(reward, bool) or not isinstance(reward, (int, float))
                    or not math.isfinite(reward) or reward not in (0.0, 1.0)):
                raise SourceIntegrityError("non-binary or nonfinite verification reward")
            if reward > 0 and key is None:
                raise SourceIntegrityError("positive reward lacks a canonical outcome key")
            verified.append(key if reward > 0 else None)
        normalized.append({
            **identity, "prompt_index": index, "reference": reference,
            "answer_keys": keys, "rewards": rewards, "verified_keys": verified,
            "request_seeds_by_option": prompt.get("request_seeds_by_option"),
            "option_ids": prompt.get("option_ids"),
            "logged_answer_mode_count": prompt.get("answer_mode_count"),
            "observed_metrics": prompt.get("metrics", {}),
        })
    if {p["prompt_index"] for p in normalized} != set(range(128)):
        raise SourceIntegrityError("duplicate or missing prompt index")
    if len({p["prompt_id"] for p in normalized}) != 128:
        raise SourceIntegrityError("duplicate exact prompt/reference identity")
    metadata = {key: row.get(key) for key in (
        "benchmark", "evaluation_kind", "sample_count", "seed", "temperature", "top_p")}
    metadata["prompt_count"] = 128
    return {
        "step": row["step"], "draw_index": row["draw_index"], "metadata": metadata,
        "origins": [origin] if origin else [], "prompts": sorted(normalized, key=lambda p: p["prompt_index"]),
        "observed_metrics": row.get("metrics", {}),
        "raw_payload_sha256": digest(canonical_json(row).encode()),
    }


def read_frozen_sources(source_files: list[dict], wanted_origins: list[dict]) -> tuple[dict, list[dict]]:
    """Return only exact requested lines after verifying each entire frozen prefix.

    Supports trajectory ``read_bytes/sha256_read_prefix`` records and baseline
    ``byte_length/sha256`` records. Keys are (absolute source path, 1-based line).
    """
    wanted: dict[str, set[int]] = {}
    for origin in wanted_origins:
        path = str(resolve_source(origin["path"]))
        line = origin["line"]
        if type(line) is not int or line < 1:
            raise SourceIntegrityError("source origin has an invalid line number")
        wanted.setdefault(path, set()).add(line)
    bindings = {}
    for source in source_files:
        path = str(resolve_source(source["path"]))
        if path in bindings and bindings[path] != source:
            raise SourceIntegrityError("conflicting source-prefix bindings")
        bindings[path] = source
    if set(wanted) - set(bindings):
        raise SourceIntegrityError("source origin is outside frozen source bindings")
    rows, checks = {}, []
    for name, requested_lines in sorted(wanted.items()):
        source = bindings[name]
        size = source.get("read_bytes", source.get("byte_length"))
        expected = source.get("sha256_read_prefix", source.get("sha256"))
        if type(size) is not int or size < 0 or not isinstance(expected, str):
            raise SourceIntegrityError("frozen source lacks prefix size/hash")
        path = Path(name)
        if path.stat().st_size < size:
            raise SourceIntegrityError(f"frozen source was truncated: {path}")
        hasher, consumed, line_number = hashlib.sha256(), 0, 0
        with path.open("rb") as handle:
            while consumed < size:
                raw = handle.readline(size - consumed)
                if not raw:
                    raise SourceIntegrityError(f"unexpected EOF in frozen source: {path}")
                consumed += len(raw)
                line_number += 1
                hasher.update(raw)
                if line_number in requested_lines:
                    if not raw.endswith(b"\n"):
                        raise SourceIntegrityError("selected source line is incomplete")
                    try:
                        rows[name, line_number] = json.loads(raw)
                    except (ValueError, UnicodeDecodeError) as exc:
                        raise SourceIntegrityError("selected source line is invalid JSON") from exc
        if hasher.hexdigest() != expected:
            raise SourceIntegrityError(f"frozen source prefix hash differs: {path}")
        if any((name, line) not in rows for line in requested_lines):
            raise SourceIntegrityError(f"source origin falls outside frozen prefix: {path}")
        checks.append({"path": name, "read_bytes": size, "sha256_read_prefix": expected,
                       "prefix_verified": True, "selected_lines": sorted(requested_lines),
                       "appended_bytes_ignored": path.stat().st_size - size})
    return rows, checks


def assemble_checkpoint(draws: list[dict], *, step: int) -> dict:
    """Certify shared prompt and recorded decoding fields across four draws."""
    draws = sorted(draws, key=lambda d: d["draw_index"])
    if [d["draw_index"] for d in draws] != [0, 1, 2, 3] or any(d["step"] != step for d in draws):
        raise SourceIntegrityError("checkpoint lacks four unique exact-step draws")
    ids = [[p["prompt_id"] for p in d["prompts"]] for d in draws]
    if any(prompt_ids != ids[0] for prompt_ids in ids[1:]):
        raise SourceIntegrityError("prompt/reference identity or ordering differs across draws")
    common_fields = ("benchmark", "evaluation_kind", "sample_count", "temperature", "top_p", "prompt_count")
    common = {k: draws[0]["metadata"][k] for k in common_fields}
    if any({k: d["metadata"][k] for k in common_fields} != common for d in draws[1:]):
        raise SourceIntegrityError("recorded decoding fields differ across draws")
    seeds = [d["metadata"]["seed"] for d in draws]
    fingerprint = digest(canonical_json({"common": common, "seeds": seeds,
                                        "draw_payloads": [d["raw_payload_sha256"] for d in draws]}).encode())
    return {"step": step, "draws": draws, "sample_status": "available",
            "sampling_certificate": {"same_prompt_identities": True, "same_recorded_decoder_fields": True,
                                     "common_metadata": common, "draw_seeds": seeds,
                                     "distinct_integer_draw_seeds": all(type(s) is int for s in seeds) and len(set(seeds)) == 4,
                                     "top_p_recorded": common["top_p"] is not None,
                                     "source_bound_fingerprint": fingerprint,
                                     "scope": "Recorded fields and frozen response identities; not an independent proof of sampler independence."}}


def load_primary_manifest(snapshot_path: str | Path = DEFAULT_SNAPSHOT) -> dict:
    """Load cohort metadata only; no raw samples or new outcome calculations."""
    path = Path(snapshot_path)
    raw = path.read_bytes()
    snapshot = json.loads(raw)
    if snapshot.get("schema") != "training-curve-frozen-snapshot-v1" or snapshot.get("target_step") != 3072:
        raise SourceIntegrityError("unexpected frozen training snapshot contract")
    cohorts = [{k: deepcopy(panel[k]) for k in ("level", "scale", "domain", "paired_cohorts")}
               for panel in snapshot["panels"]]
    totals = Counter()
    for panel in cohorts:
        for objective, seeds in panel["paired_cohorts"].items():
            if len(seeds) != len(set(seeds)):
                raise SourceIntegrityError("duplicate seed in frozen paired cohort")
            totals[panel["level"], objective] += len(seeds)
    if (totals["level1", "drgrpo"], totals["level1", "maxrl"]) != (74, 67):
        raise SourceIntegrityError("frozen Level-1 primary terminal cohorts differ from 74/67")
    cells = snapshot["cells"]
    if len({tuple(c[k] for k in CELL_FIELDS) for c in cells}) != len(cells):
        raise SourceIntegrityError("duplicate scientific cell in frozen snapshot")
    return {"schema": "paper-collision-source-manifest-v1", "snapshot": snapshot,
            "source_snapshot": {"path": str(path.resolve()), "sha256": digest(raw)},
            "cohorts": cohorts, "cohort_counts": [{"level": level, "objective": objective, "n": n}
                                                    for (level, objective), n in sorted(totals.items())]}


def _load_cell(cell: dict, paired: bool, steps: tuple[int, ...]) -> dict:
    result = {k: deepcopy(cell.get(k)) for k in (*CELL_FIELDS, "run_dir", "ledger", "registered_job_id",
              "terminal_admitted", "terminal_matches_census", "terminal_reference", "approved_exclusion")}
    result.update(in_terminal_paired_cohort=paired, checkpoints={}, sample_issues=[], source_checks=[])
    requested = {str(step): cell["complete_checkpoints"].get(str(step)) for step in steps}
    wanted = [origin for cp in requested.values() if cp for draw in cp["draws"] for origin in draw["origins"]]
    try:
        rows, result["source_checks"] = read_frozen_sources(cell["source_files"], wanted)
    except (OSError, SourceIntegrityError) as exc:
        result["sample_issues"].append({"kind": "source_integrity_failure", "reason": str(exc), "steps": list(steps)})
        rows = None
    for step_text, checkpoint in requested.items():
        if checkpoint is None:
            result["checkpoints"][step_text] = None
            result["sample_issues"].append({"kind": "not_admitted_in_frozen_snapshot", "step": int(step_text)})
            continue
        if rows is None:
            result["checkpoints"][step_text] = None
            continue
        try:
            normalized = []
            for admitted in checkpoint["draws"]:
                originals = [rows[str(resolve_source(o["path"])), o["line"]] for o in admitted["origins"]]
                signatures = {canonical_json(row) for row in originals}
                if len(signatures) != 1:
                    raise SourceIntegrityError("metric-equivalent duplicate origins have conflicting raw response payloads")
                raw = originals[0]
                if raw.get("step") != int(step_text) or raw.get("draw_index") != admitted["draw_index"]:
                    raise SourceIntegrityError("raw source step/draw differs from exact frozen origin")
                if raw.get("metrics") != admitted["metrics"]:
                    raise SourceIntegrityError("raw draw metrics differ from frozen admitted metrics")
                draw = normalize_draw(raw)
                if any(draw["metadata"].get(k) != v for k, v in admitted["metadata"].items()):
                    raise SourceIntegrityError("raw draw metadata differs from frozen admission")
                draw["origins"] = deepcopy(admitted["origins"])
                normalized.append(draw)
            result["checkpoints"][step_text] = assemble_checkpoint(normalized, step=int(step_text))
        except (SourceIntegrityError, ValueError, TypeError, KeyError) as exc:
            result["checkpoints"][step_text] = None
            result["sample_issues"].append({"kind": "raw_sample_integrity_failure", "step": int(step_text), "reason": str(exc)})
    result["before_after_available"] = all(result["checkpoints"].get(str(step)) is not None for step in (0, 3072))
    return result


def iter_primary_sample_cells(manifest: dict | None = None, *, snapshot_path: str | Path = DEFAULT_SNAPSHOT,
                              steps: tuple[int, ...] = (0, 3072), paired_only: bool = False,
                              workers: int = 4) -> Iterator[dict]:
    """Stream one normalized scientific cell at a time in stable snapshot order."""
    manifest = manifest or load_primary_manifest(snapshot_path)
    if any(step not in (0, 3072) for step in steps):
        raise ValueError("this retrospective intake admits only step0 and3072")
    cohort = {(p["level"], p["scale"], p["domain"], method, seed)
              for p in manifest["cohorts"] for objective, seeds in p["paired_cohorts"].items()
              for method in (objective, "replay_" + objective) for seed in seeds}
    tasks = [(cell, tuple(cell[k] for k in CELL_FIELDS) in cohort)
             for cell in manifest["snapshot"]["cells"]]
    tasks = [(cell, paired) for cell, paired in tasks if paired or not paired_only]
    for cell, paired in tasks:
        if paired and (not cell["terminal_admitted"] or not cell["terminal_matches_census"]
                       or "3072" not in cell["complete_checkpoints"]):
            raise SourceIntegrityError("paired cell lacks its original exact terminal admission")
    # Bounded batches prevent Executor.map from retaining every completed cell.
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for start in range(0, len(tasks), max(1, workers)):
            futures = [pool.submit(_load_cell, cell, paired, tuple(steps))
                       for cell, paired in tasks[start:start + max(1, workers)]]
            for future in futures:
                yield future.result()


def load_primary_samples(snapshot_path: str | Path = DEFAULT_SNAPSHOT, *, steps: tuple[int, ...] = (0, 3072),
                         paired_only: bool = False, workers: int = 4) -> dict:
    manifest = load_primary_manifest(snapshot_path)
    return {k: v for k, v in manifest.items() if k != "snapshot"} | {
        "schema": "paper-collision-normalized-samples-v1",
        "cells": list(iter_primary_sample_cells(manifest, steps=steps, paired_only=paired_only, workers=workers))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, required=True, help="Fresh JSONL path; one cell per line")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--paired-only", action="store_true")
    args = parser.parse_args()
    manifest = load_primary_manifest(args.snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as sink:
        sink.write(canonical_json({k: v for k, v in manifest.items() if k != "snapshot"}) + "\n")
        for cell in iter_primary_sample_cells(manifest, paired_only=args.paired_only, workers=args.workers):
            sink.write(canonical_json(cell) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
