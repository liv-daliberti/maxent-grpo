#!/usr/bin/env python3
"""Freeze planning batches for missing GRPO, UCPO, and RLEP-Dr cells.

This is deliberately not a launcher. It partitions every missing direct-
comparator cell exactly once, records the closest immutable protocol source,
and keeps submission authorization false until a reviewed preregistration and
launcher manifest exist.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_matrix as matrix  # noqa: E402


COMPARATORS = ("grpo", "ucpo", "rlep_dr")
STRATA = ("static",)


def _source_tag(method: str, scale: str) -> str:
    if method == "grpo":
        return {
            "qwen05b": "e95_05b",
            "falcon1b": "e95_1b",
            "qwen3b": "e95_3b",
        }[scale]
    return {"ucpo": "e97", "rlep_dr": "e98"}[method]


def _templates(method: str, scale: str, stratum: str) -> list[str]:
    if method == "grpo":
        out = [
            "paper/preregistration/e95_plain_grpo_cross_scale_20260812.md",
            "ops/exp_scaling/launch_e95_plain_grpo_control.py",
            _source_tag(method, scale),
        ]
        return out
    if method == "ucpo":
        return [
            "paper/preregistration/e97_ucpo_05b_20260812.md",
            "ops/exp_scaling/launch_e97_ucpo_05b.py",
            "e97",
        ]
    return [
        "paper/preregistration/e98_rlep_dr_05b_20260812.md",
        "ops/exp_scaling/launch_e98_rlep_05b.py",
        "ops/exp_scaling/audit_e98_rlep_pool.py",
        "e98",
    ]


def _protocol(method: str, scale: str, stratum: str) -> tuple[str, str]:
    if method == "rlep_dr":
        return (
            "blocked_feasibility_amendment",
            (
                "Do not extend E98 as written. Preregister a non-post-hoc pool "
                "feasibility amendment, pass per-cell collection audits, then "
                "pass a fresh shared learner smoke before releasing science jobs."
            ),
        )
    if method == "grpo" and scale == "qwen3b" and stratum == "static":
        return (
            "frozen_seed_extension",
            (
                "Extend the frozen E95 Qwen2.5-3B protocol from seed 70 to seeds "
                "71-74 without changing optimizer, horizon, or evaluation."
            ),
        )
    if method == "grpo" and stratum == "static":
        return (
            "frozen_protocol_extension",
            (
                "Recover any absent E95 static cell under its frozen scale-specific "
                "protocol without changing optimizer, horizon, or evaluation."
            ),
        )
    if scale == "qwen05b" and stratum == "static":
        return (
            "frozen_domain_extension",
            (
                "Extend the frozen E97 UCPO treatment to the missing static "
                "domains with the matched E78 control budget and evaluator."
            ),
        )
    return (
        "scale_port",
        (
            "Preregister the E97 UCPO treatment at this model scale; freeze "
            "the same tau, rollout budget, horizon, and domain evaluators."
        ),
    )


def build_plan(cells: dict[matrix.CellKey, matrix.Cell]) -> dict[str, Any]:
    domain_stratum = {domain.key: domain.stratum for domain in matrix.DOMAINS}
    target = [key for key in matrix.desired_keys() if key.method in COMPARATORS]
    missing = [key for key in target if key not in cells]
    batches: list[dict[str, Any]] = []
    assigned: list[matrix.CellKey] = []

    for method in COMPARATORS:
        for scale in (item.key for item in matrix.SCALES):
            for stratum in STRATA:
                records = [
                    key
                    for key in missing
                    if key.method == method
                    and key.scale == scale
                    and domain_stratum[key.domain] == stratum
                ]
                if not records:
                    continue
                protocol_state, protocol_gate = _protocol(method, scale, stratum)
                assigned.extend(records)
                batches.append(
                    {
                        "key": f"{method}_{scale}_{stratum}",
                        "method": method,
                        "scale": scale,
                        "stratum": stratum,
                        "cells": len(records),
                        "domains": sorted({key.domain for key in records}),
                        "seeds": sorted({key.seed for key in records}),
                        "protocol_state": protocol_state,
                        "protocol_gate": protocol_gate,
                        "template_sources": _templates(method, scale, stratum),
                        "submission_authorized": False,
                        "records": [asdict(key) for key in records],
                    }
                )

    if len(assigned) != len(set(assigned)):
        raise ValueError("a missing comparator cell was assigned to multiple batches")
    if set(assigned) != set(missing):
        omitted = sorted(set(missing) - set(assigned))
        extra = sorted(set(assigned) - set(missing))
        raise ValueError(f"comparator partition drift: omitted={omitted}, extra={extra}")
    states = Counter(str(batch["protocol_state"]) for batch in batches)
    return {
        "schema": "modebench-paper-comparator-batches-v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "methods": list(COMPARATORS),
        "target_cells": len(target),
        "registered_cells": len(target) - len(missing),
        "missing_cells": len(missing),
        "batches": len(batches),
        "protocol_state_counts": dict(sorted(states.items())),
        "submission_authorized": False,
        "safety": (
            "planning only; this artifact cannot submit work. Every batch needs a "
            "reviewed preregistration and immutable launcher manifest."
        ),
        "records": batches,
    }


def render(plan: dict[str, Any]) -> str:
    headings = ("batch", "cells", "domains", "seeds", "protocol state")
    widths = (44, 5, 7, 5, 31)
    out = [
        " ".join(f"{name:<{width}}" for name, width in zip(headings, widths)),
        "-" * (sum(widths) + len(widths) - 1),
    ]
    for batch in plan["records"]:
        values = (
            batch["key"],
            str(batch["cells"]),
            str(len(batch["domains"])),
            str(len(batch["seeds"])),
            batch["protocol_state"],
        )
        out.append(
            " ".join(
                f"{value:<{width}}" for value, width in zip(values, widths)
            )
        )
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    plan = build_plan(matrix.load_cells())
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        print(
            f"direct comparators: {plan['registered_cells']}/{plan['target_cells']} registered; "
            f"{plan['missing_cells']} missing in {plan['batches']} batches"
        )
        print()
        print(render(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
