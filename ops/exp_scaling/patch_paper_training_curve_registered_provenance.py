#!/usr/bin/env python3
"""Apply amendment 5: admit conflicted registered checkpoints by attempt provenance.

Amendment 4 resolved conflicted *initial* checkpoints by reading which attempt's
trajectory the analysis already uses. The same conflict occurs at registered
training steps, from the same cause: a watchdog requeue re-evaluates a step and
appends a second block, and separate attempts land on different GPU models, so
the blocks disagree and the checkpoint is invalidated. The training-curve
figures draw those invalidations as gaps.

This program extends amendment 4's rule to any registered step and expresses it,
as amendment 4 did, as a patch to a frozen training-curve snapshot rather than a
re-collection. It reads file order, step numbers, draw indices and the presence
of later training steps. It never reads a metric value.

Unlike amendment 4's patch, this one also rebuilds the snapshot's panels from
the patched cells. Amendment 4 left panels untouched, which is why its patched
snapshot renders identically to its source: the plotting code reads panels, not
cells. Panels are rebuilt with each panel's existing fixed cohort, so the
amendment changes which checkpoints a seed contributes and never which seeds
are in the cohort.

    python ops/exp_scaling/patch_paper_training_curve_registered_provenance.py \
        --binding paper/audits/training_curve_provenance_20260919/binding.json
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
from exp_scaling.build_paper_training_curve_snapshot import series  # noqa: E402

FIELDS = ("level", "scale", "domain", "method", "seed")
SCHEMA = "training-curve-frozen-snapshot-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key_of(cell: dict) -> tuple:
    return tuple(cell[field] for field in FIELDS)


def continued_block(rows: list[tuple[int, dict]], step: int) -> list[list[dict]]:
    """Blocks of draws at `step` that the attempt's training advanced past.

    A block is closed by the next evaluation row. It counts as continued when
    that row carries a later step, which is the file-order evidence that this
    attempt kept training after writing the block. Amendment 4 applies exactly
    this test at step 0, where "a later step" is any training step.
    """
    blocks: list[tuple[list[dict], bool]] = []
    current: list[dict] = []
    for number, record in rows:
        if record.get("step") != step:
            if current:
                blocks.append((current, (record.get("step") or 0) > step))
                current = []
            continue
        if record.get("draw_index") is None and current:
            blocks.append((current, False))
            current = []
        if record.get("draw_index") is not None:
            # prompt_count is not a field of the raw draw row; the snapshot
            # derives it from the prompt list, and validate_checkpoint requires
            # it, so it is derived here the same way.
            metadata = {field: record.get(field) for field in (
                "benchmark", "evaluation_kind", "sample_count", "seed", "temperature")}
            metadata["prompt_count"] = len(record.get("prompts") or ())
            current.append({
                "draw_index": record["draw_index"], "line": number,
                "metrics": record["metrics"], "metadata": metadata,
            })
    if current:
        blocks.append((current, False))
    return [block for block, is_continued in blocks if is_continued]


def admitted_block(cell: dict, step: int) -> list[dict] | None:
    """The one continued four-draw block at `step`, or nothing."""
    found: list[list[dict]] = []
    for source in cell.get("source_files") or []:
        path = Path(source["path"])
        if not path.is_file():
            return None
        rows = []
        with path.open() as handle:
            for number, line in enumerate(handle, 1):
                if line.strip():
                    rows.append((number, json.loads(line)))
        if not any((record.get("step") or 0) > step for _, record in rows):
            continue  # an attempt that never trained past this step
        for block in continued_block(rows, step):
            block = [dict(draw, path=str(path)) for draw in block]
            found.append(block)
    complete = [block for block in found
                if sorted(draw["draw_index"] for draw in block) == [0, 1, 2, 3]]
    if len(complete) != 1:
        return None
    return sorted(complete[0], key=lambda draw: draw["draw_index"])


def checkpoint_record(draws: list[dict], step: int) -> dict:
    """Snapshot-shaped admission, built exactly as amendment 4 builds step 0."""
    keys = sorted({k for draw in draws for k in draw["metrics"]})
    return {
        "draw_count": len(draws),
        "step": step,
        "draws": [{"draw_index": draw["draw_index"],
                   "metadata": {k: v for k, v in draw["metadata"].items() if v is not None},
                   "metrics": draw["metrics"],
                   "origins": [{"line": draw["line"], "path": draw["path"]}]}
                  for draw in draws],
        "mean_metrics": {k: sum(draw["metrics"][k] for draw in draws) / len(draws)
                         for k in keys if all(k in draw["metrics"] for draw in draws)},
    }


def patch_cells(snapshot: dict, population: dict[tuple, list[int]]) -> list[dict]:
    index = {key_of(cell): cell for cell in snapshot["cells"]}
    missing = set(population) - set(index)
    if missing:
        raise SystemExit(f"amendment 5 names cells absent from the snapshot: {sorted(missing)}")
    applied = []
    for cell_key, steps in sorted(population.items()):
        cell = index[cell_key]
        for step in steps:
            block = admitted_block(cell, step)
            if block is None:
                raise SystemExit(
                    f"amendment 5 names {cell_key} step {step}, which is not resolvable now")
            cell["complete_checkpoints"][str(step)] = checkpoint_record(block, step)
            cell["complete_steps"] = sorted(set(cell["complete_steps"]) | {step})
            cell["invalid_or_conflicted_steps"] = [
                s for s in cell["invalid_or_conflicted_steps"] if s != step]
            cell["missing_registered_steps"] = [
                s for s in cell["missing_registered_steps"] if s != step]
            cell["incomplete_checkpoints"].pop(str(step), None)
            applied.append({"cell": list(cell_key), "step": step,
                            "origins": [draw["origins"][0] for draw
                                        in cell["complete_checkpoints"][str(step)]["draws"]]})
    return applied


def rebuild_panels(snapshot: dict) -> None:
    """Recompute every panel series from the patched cells, cohorts unchanged.

    The fixed cohort is a property of the terminal paired seeds and is not what
    this amendment touches, so each series is rebuilt with the seed list and
    policy the frozen panel already carries.
    """
    index = {key_of(cell): cell for cell in snapshot["cells"]}
    for panel in snapshot["panels"]:
        prefix = (panel["level"], panel["scale"], panel["domain"])
        for group in ("methods", "supplementary_methods"):
            for method, record in panel[group].items():
                panel[group][method] = series(
                    index, prefix, method, record["cohort_seeds"], record["cohort_policy"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    binding = json.loads(args.binding.read_text())
    amendment = ROOT / binding["amendment_path"]
    if sha(amendment) != binding["amendment_sha256"]:
        raise SystemExit("amendment 5 binding differs from its frozen text")
    source_path = ROOT / binding["source_snapshot"]["path"]
    if sha(source_path) != binding["source_snapshot"]["sha256"]:
        raise SystemExit("bound training-curve snapshot changed")
    output = args.output or (ROOT / binding["output_snapshot"])
    if output.exists():
        raise SystemExit(f"refusing to overwrite an existing snapshot: {output}")

    snapshot = json.loads(source_path.read_text())
    if snapshot.get("schema") != SCHEMA:
        raise SystemExit("unexpected training-curve snapshot schema")
    population: dict[tuple, list[int]] = {}
    for entry in binding["affected_cells"]:
        population.setdefault(tuple(entry["cell"]), []).extend(entry["steps"])
    patched = deepcopy(snapshot)
    applied = patch_cells(patched, population)
    if len(applied) != binding["affected_checkpoint_count"]:
        raise SystemExit("patched checkpoint count differs from the frozen affected population")
    rebuild_panels(patched)
    patched["checkpoint_policy"] = (
        snapshot["checkpoint_policy"]
        + " Conflicted registered checkpoints named by amendment 5 are admitted by attempt"
          " provenance, never by metric value.")
    patched["initial_provenance_amendment"] = {
        "path": binding["amendment_path"], "sha256": binding["amendment_sha256"],
        "source_snapshot": binding["source_snapshot"],
        "patched_at_utc": datetime.now(timezone.utc).isoformat(),
        "admissions": applied,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(patched, sort_keys=True) + "\n")
    print(json.dumps({"event": "patched", "output": str(output.relative_to(ROOT)),
                      "sha256": sha(output), "checkpoints_admitted": len(applied),
                      "cells_touched": len(population)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
