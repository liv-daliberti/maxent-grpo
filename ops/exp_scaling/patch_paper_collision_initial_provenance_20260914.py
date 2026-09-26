#!/usr/bin/env python3
"""Apply amendment 4: admit conflicted step-0 endpoints by attempt provenance.

A conflicted initial checkpoint is a watchdog-requeue duplicate: the restart
re-evaluated the initial model and appended a second step-0 block, and the two
attempts disagree because they executed on different GPU models. The rule
frozen in the amendment admits the block that the cell's admitted trajectory
continues from, chosen by file order alone.

The rule is expressed here as a patch to the frozen training-curve snapshot, so
that the unchanged loader performs every integrity check on the result: frozen
prefix hashes, raw-versus-admitted metric equality, prompt identity across
draws, and recorded decoder fields. This program reconstructs existing P,D,M
measurements only. It never computes collision, eligibility, effects, or any
inferential summary.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "paper/audits/conditional_concentration_20260914"
BINDING = AUDIT / "amendment_4_initial_checkpoint_provenance_binding.json"
PATCHED_SNAPSHOT = AUDIT / "training_curve_snapshot_initial_provenance.json"
OUTPUT = AUDIT / "verified_samples_initial_provenance.jsonl.gz"
RECEIPT = AUDIT / "collection_receipt_initial_provenance.json"
FROZEN_PHASE = "frozen_before_new_source_effects"

sys.path.insert(0, str(ROOT / "ops"))
from exp_scaling import load_paper_collision_samples as loader  # noqa: E402
# The compact reconstruction is reused from the amendment-3 extension unchanged,
# so an admitted checkpoint is stored exactly as its predecessors were.
from exp_scaling.extend_paper_collision_complete_pantry_20260912 import (  # noqa: E402
    compact_checkpoint,
)

FIELDS = ("level", "scale", "domain", "method", "seed")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key_of(cell: dict) -> tuple:
    return tuple(cell[k] for k in FIELDS)


def continued_step0_block(cell: dict) -> list[dict] | None:
    """The step-0 draws of the attempt whose trajectory the analysis uses.

    Reads file order, step numbers and draw indices only. A source file with no
    training-step row belongs to an attempt that trained nothing and is
    discarded; within a retained file a step-0 block counts as continued when
    the next evaluation row is a training step. Exactly one continued block may
    exist across the cell's files, or the endpoint stays unavailable.
    """
    continued: list[list[dict]] = []
    for source in cell.get("source_files") or []:
        path = Path(source["path"])
        if not path.is_file():
            return None
        rows = []
        with path.open() as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                rows.append((number, record))
        if not any((r.get("step") or 0) > 0 for _, r in rows):
            continue
        blocks: list[tuple[list[dict], bool]] = []
        current: list[dict] = []
        for number, record in rows:
            if record.get("step") != 0:
                if current:
                    blocks.append((current, True))
                    current = []
                continue
            if record.get("draw_index") is None and current:
                blocks.append((current, False))
                current = []
            if record.get("draw_index") is not None:
                current.append({
                    "draw_index": record["draw_index"],
                    "line": number,
                    "path": str(path),
                    "metrics": record["metrics"],
                    "metadata": {k: record.get(k) for k in (
                        "benchmark", "evaluation_kind", "prompt_count",
                        "sample_count", "seed", "temperature")},
                })
        if current:
            blocks.append((current, False))
        continued.extend(block for block, is_continued in blocks if is_continued)
    if len(continued) != 1 or sorted(d["draw_index"] for d in continued[0]) != [0, 1, 2, 3]:
        return None
    return sorted(continued[0], key=lambda d: d["draw_index"])


def admitted_checkpoint(draws: list[dict]) -> dict:
    """Snapshot-shaped step-0 admission built from the selected block."""
    keys = sorted({k for d in draws for k in d["metrics"]})
    return {
        "draw_count": len(draws),
        "step": 0,
        "draws": [{"draw_index": d["draw_index"],
                   "metadata": {k: v for k, v in d["metadata"].items() if v is not None},
                   "metrics": d["metrics"],
                   "origins": [{"line": d["line"], "path": d["path"]}]} for d in draws],
        "mean_metrics": {k: sum(d["metrics"][k] for d in draws) / len(draws)
                         for k in keys if all(k in d["metrics"] for d in draws)},
    }


def patch_snapshot(snapshot: dict, wanted: set[tuple]) -> tuple[dict, list[dict]]:
    patched = deepcopy(snapshot)
    applied = []
    for cell in patched["cells"]:
        key = key_of(cell)
        if key not in wanted:
            continue
        block = continued_step0_block(cell)
        if block is None:
            raise ValueError(f"amendment 4 names {key} but its step 0 is not resolvable now")
        cell["complete_checkpoints"]["0"] = admitted_checkpoint(block)
        cell["complete_steps"] = sorted(set(cell["complete_steps"]) | {0})
        cell["invalid_or_conflicted_steps"] = [s for s in cell["invalid_or_conflicted_steps"] if s != 0]
        cell["missing_registered_steps"] = [s for s in cell["missing_registered_steps"] if s != 0]
        cell["incomplete_checkpoints"].pop("0", None)
        applied.append({"cell": list(key),
                        "origins": [d["origins"][0] for d in cell["complete_checkpoints"]["0"]["draws"]]})
    missing = wanted - {key_of(c) for c in patched["cells"]}
    if missing:
        raise ValueError(f"amendment 4 names cells absent from the snapshot: {sorted(missing)}")
    if len(applied) != len(wanted):
        raise ValueError("patched cell count differs from the frozen affected population")
    return patched, applied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if any(p.exists() for p in (PATCHED_SNAPSHOT, OUTPUT, RECEIPT)):
        raise SystemExit("refusing to overwrite a completed amendment-4 artifact")

    binding = json.loads(BINDING.read_text())
    amendment = ROOT / binding["amendment_path"]
    if sha(amendment) != binding["amendment_sha256"] or binding["status"] != FROZEN_PHASE:
        raise SystemExit("amendment 4 binding differs from its frozen text")
    snapshot_path = ROOT / binding["source_snapshot"]["path"]
    if sha(snapshot_path) != binding["source_snapshot"]["sha256"]:
        raise SystemExit("bound training-curve snapshot changed")
    parent = binding["preserved_predecessor"]
    for record in (parent["amendment"], parent["cache"], parent["receipt"], parent["result"]):
        if sha(ROOT / record["path"]) != record["sha256"]:
            raise SystemExit(f"preserved predecessor changed: {record['path']}")

    wanted = {(c["level"], c["scale"], c["domain"], c["method"], c["seed"])
              for c in binding["affected_cells"]}
    snapshot = json.loads(snapshot_path.read_text())
    patched, applied = patch_snapshot(snapshot, wanted)
    PATCHED_SNAPSHOT.write_text(json.dumps(patched, sort_keys=True) + "\n")
    print(f"Patched {len(applied)} step-0 admissions into {PATCHED_SNAPSHOT.name}")

    index = {key_of(c): c for c in patched["cells"]}
    with gzip.open(ROOT / parent["cache"]["path"], "rt") as handle:
        header = json.loads(next(handle))
        cached = [json.loads(line) for line in handle]

    cohort = {key_of(cell) for cell in cached if cell.get("in_terminal_paired_cohort")}
    reloaded = {}
    for number, key in enumerate(sorted(wanted), 1):
        reloaded[key] = loader._load_cell(index[key], key in cohort, (0, 3072))
        print(f"Revalidated provenance cell {number}/{len(wanted)}: {key}", flush=True)

    failures, result_cells, stream_rows = [], [], []
    for cell in cached:
        key = key_of(cell)
        record = deepcopy(cell)
        if key in reloaded:
            fresh = reloaded[key]
            loaded_step0 = fresh["checkpoints"].get("0")
            if loaded_step0 is None:
                failures.append({"cell": list(key), "issues": fresh["sample_issues"]})
                step0 = None
            else:
                try:
                    step0, stream_checks = compact_checkpoint(loaded_step0)
                except ValueError as exc:
                    failures.append({"cell": list(key), "issues": [{"step": 0, "reason": str(exc)}]})
                    step0 = None
                else:
                    stream_rows.append({"cell": list(key), "step": 0, "draws": stream_checks})
            record["checkpoints"]["0"] = deepcopy(step0)
            record["sample_issues"] = [i for i in record["sample_issues"] if i.get("step") != 0]
            record["sample_issues"].extend(deepcopy([i for i in fresh["sample_issues"] if i.get("step") == 0]))
            record["source_checks"] = deepcopy(fresh["source_checks"])
            record["before_after_available"] = all(
                record["checkpoints"].get(s) is not None for s in ("0", "3072"))
            record["initial_provenance_admission"] = {
                "amendment": binding["amendment_path"],
                "origins": next(a["origins"] for a in applied if a["cell"] == list(key)),
            }
        result_cells.append(record)

    if failures:
        raise SystemExit("provenance admission failed integrity revalidation: "
                         + json.dumps(failures)[:2000])

    new_header = deepcopy(header)
    new_header.update({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_kind": "paper-collision-samples-initial-provenance-v1",
        "initial_provenance_amendment": {
            "path": binding["amendment_path"],
            "sha256": binding["amendment_sha256"],
            "patched_snapshot": {"path": str(PATCHED_SNAPSHOT.relative_to(ROOT)),
                                 "sha256": sha(PATCHED_SNAPSHOT)},
            "affected_cells": binding["affected_cells"],
            "extension_code_sha256": sha(Path(__file__)),
        },
        "parent_cache": {"path": parent["cache"]["path"], "sha256": parent["cache"]["sha256"]},
    })
    with gzip.open(OUTPUT, "wt") as handle:
        handle.write(json.dumps(new_header, sort_keys=True) + "\n")
        for cell in result_cells:
            handle.write(json.dumps(cell, sort_keys=True) + "\n")

    admitted = sum(1 for c in result_cells if c["checkpoints"].get("0") is not None)
    RECEIPT.write_text(json.dumps({
        "status": "collected",
        "path": str(OUTPUT.relative_to(ROOT)),
        "sha256": sha(OUTPUT),
        "binding_path": str(BINDING.relative_to(ROOT)),
        "binding_sha256": sha(BINDING),
        "amendment_sha256": binding["amendment_sha256"],
        "patched_snapshot_sha256": sha(PATCHED_SNAPSHOT),
        "extension_code_sha256": sha(Path(__file__)),
        "cells": len(result_cells),
        "newly_admitted_initial_checkpoints": len(wanted),
        "cells_with_initial_checkpoint": admitted,
        "stream_metadata_checks": stream_rows,
        "effects_computed": False,
    }, indent=1, sort_keys=True) + "\n")
    print(f"Wrote {OUTPUT.name} ({len(result_cells)} cells, "
          f"{admitted} with an initial checkpoint) and its receipt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
