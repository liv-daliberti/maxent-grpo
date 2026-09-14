#!/usr/bin/env python3
"""Partition unfinished paper cells into explicit execution waves.

The output is a planning manifest, not a launcher. Registered cells retain
their existing job identity and are never emitted as new submissions. Missing
cells are grouped by scientific priority and protocol gate so future launchers
can consume a frozen, reviewed slice instead of reconstructing scope from
cohort names.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_matrix as matrix  # noqa: E402


@dataclass(frozen=True)
class Wave:
    key: str
    priority: int
    label: str
    kind: str
    protocol_gate: str


WAVES: tuple[Wave, ...] = (
    Wave(
        "finish_released",
        0,
        "Finish, repair, or resolve gates for released scientific cells",
        "existing",
        (
            "Use the immutable source ledger; do not create a duplicate job. "
            "Blocked cells require a registered gate resolution, not resubmission."
        ),
    ),
    Wave(
        "direct_comparators",
        1,
        "Complete GRPO, UCPO, and RLEP-Dr controls",
        "new",
        (
            "Freeze model/domain extensions against the existing paired design; "
            "RLEP-Dr cells require a passed replay-pool feasibility gate."
        ),
    ),
    Wave(
        "core_scale_extensions",
        2,
        "Complete remaining Dr.GRPO and Re:Dr.GRPO scale cells",
        "new",
        "Extend the frozen five-domain paired protocol without changing the intervention.",
    ),
    Wave(
        "adaptive_replay",
        3,
        "Extend Adaptive Re:Dr.GRPO",
        "new",
        "Freeze the bank-normalized dose rule from E90 without scale-specific tuning.",
    ),
    Wave(
        "fixed_semantic",
        4,
        "Complete the fixed semantic factorial",
        "new",
        "Reuse the fixed coefficient and repaired PantryPlan actuator contract.",
    ),
    Wave(
        "adaptive_semantic_without_replay",
        5,
        "Create Adaptive Semantic MaxEnt without replay",
        "new protocol",
        (
            "Implement and preregister the genuinely missing no-replay controller; "
            "the adaptive-plus-replay cohorts cannot substitute for it."
        ),
    ),
    Wave(
        "adaptive_semantic_with_replay",
        6,
        "Complete Adaptive Semantic MaxEnt plus Re:Dr.GRPO",
        "new",
        "Freeze the reachable controller identity used by E89/E91/E92.",
    ),
)

WAVE_BY_KEY = {wave.key: wave for wave in WAVES}

MISSING_WAVE_BY_METHOD = {
    "grpo": "direct_comparators",
    "ucpo": "direct_comparators",
    "rlep_dr": "direct_comparators",
    "drgrpo": "core_scale_extensions",
    "replay_grpo": "core_scale_extensions",
    "adaptive_replay_grpo": "adaptive_replay",
    "semantic_maxent": "fixed_semantic",
    "replay_semantic_maxent": "fixed_semantic",
    "adaptive_semantic_maxent": "adaptive_semantic_without_replay",
    "adaptive_semantic_replay": "adaptive_semantic_with_replay",
}


def build_plan(cells: dict[matrix.CellKey, matrix.Cell]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {wave.key: [] for wave in WAVES}
    terminal = 0
    for key in matrix.desired_keys():
        cell = cells.get(key)
        record: dict[str, Any] = asdict(key)
        if cell is not None and cell.status == "terminal":
            terminal += 1
            continue
        if cell is not None:
            wave_key = "finish_released"
            record.update(
                {
                    "status": cell.status,
                    "source": cell.source,
                    "ledger": cell.ledger,
                    "job_id": cell.job_id,
                    "step": cell.step,
                    "target": cell.target,
                    "blocked_because": cell.blocked_because,
                    "launch_new_job": False,
                }
            )
        else:
            wave_key = MISSING_WAVE_BY_METHOD[key.method]
            record.update(
                {
                    "status": "missing",
                    "launch_new_job": True,
                }
            )
        buckets[wave_key].append(record)

    waves = []
    for wave in WAVES:
        records = buckets[wave.key]
        states = Counter(str(record["status"]) for record in records)
        waves.append(
            {
                **asdict(wave),
                "cells": len(records),
                "status_counts": dict(sorted(states.items())),
                "records": records,
            }
        )
    unfinished = sum(len(records) for records in buckets.values())
    target_cells = len(matrix.desired_keys())
    if terminal + unfinished != target_cells:
        raise ValueError(
            f"launch plan does not partition the matrix: {terminal} + {unfinished}"
        )
    return {
        "schema": "modebench-paper-launch-plan-v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "terminal_cells": terminal,
        "unfinished_cells": unfinished,
        "target_cells": target_cells,
        "safety": (
            "planning only; existing records must not be resubmitted and missing "
            "records require a separately reviewed frozen launcher manifest"
        ),
        "waves": waves,
    }


def render(plan: dict[str, Any], *, markdown: bool) -> str:
    if markdown:
        out = [
            "| priority | wave | kind | cells | status | protocol gate |",
            "|---:|---|---|---:|---|---|",
        ]
        for wave in plan["waves"]:
            statuses = ", ".join(
                f"{key}={value}"
                for key, value in wave["status_counts"].items()
            )
            out.append(
                f"| P{wave['priority']} | {wave['label']} | {wave['kind']} "
                f"| {wave['cells']} | {statuses} | {wave['protocol_gate']} |"
            )
        return "\n".join(out)

    width = max(len(wave["label"]) for wave in plan["waves"])
    out = [
        f"{'wave':<{width}} {'kind':<12} {'cells':>5}  status",
        "-" * (width + 34),
    ]
    for wave in plan["waves"]:
        statuses = ", ".join(
            f"{key}={value}" for key, value in wave["status_counts"].items()
        )
        out.append(
            f"{wave['label']:<{width}} {wave['kind']:<12} "
            f"{wave['cells']:>5}  {statuses}"
        )
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="also write the full JSON manifest to this path",
    )
    args = parser.parse_args()

    plan = build_plan(matrix.load_cells())
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(plan, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        print(
            f"paper launch plan: {plan['terminal_cells']} terminal, "
            f"{plan['unfinished_cells']} unfinished"
        )
        print()
        print(render(plan, markdown=args.markdown))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
