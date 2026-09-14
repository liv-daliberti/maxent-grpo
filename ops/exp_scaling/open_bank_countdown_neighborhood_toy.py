#!/usr/bin/env python3
"""Audit a support-blind local Countdown proposal neighborhood.

The candidate generator sees only the public three-position action grammar,
the prompt operands, one verified model-authored anchor, and the ordinary
validator.  This script may compare its accepted candidates with the frozen
exact enumerator because it is an offline mechanism audit; training never
receives that comparison or the enumerated support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
OPS = ROOT / "ops"
for path in (SRC, OPS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from make_modebench_data import _countdown_expression_map  # noqa: E402
from oat_drgrpo.canonical_actions import (  # noqa: E402
    decode_countdown_action_code,
    enumerate_countdown_action_codes,
)
from oat_drgrpo.math_grader import (  # noqa: E402
    _canonical_countdown_expression_key,
    validated_modebench_outcome_key,
)


def hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("codes must have the same length")
    return sum(a != b for a, b in zip(left, right))


def exact_keys(reference: dict[str, object]) -> set[str]:
    numbers = [int(value) for value in reference["numbers"]]
    target = int(reference["target"])
    expressions = _countdown_expression_map(numbers).get(target, set())
    return {
        key
        for expression in expressions
        if (key := _canonical_countdown_expression_key(expression, reference))
        is not None
    }


def valid_code_keys(reference: dict[str, object]) -> dict[str, str]:
    serialized = json.dumps(reference, sort_keys=True)
    accepted: dict[str, str] = {}
    for code in enumerate_countdown_action_codes():
        expression = decode_countdown_action_code(code, reference)
        key = validated_modebench_outcome_key(
            f"\\boxed{{{expression}}}",
            serialized,
        )
        if key is not None:
            accepted[code] = key
    return accepted


def audit_rows(rows: list[dict[str, object]], radius: int) -> dict[str, object]:
    anchors = 0
    expandable = 0
    novel_counts: list[int] = []
    rows_with_any = 0
    rows_with_all = 0
    support_mismatches: list[dict[str, object]] = []
    example: dict[str, object] | None = None

    for row_index, row in enumerate(rows):
        reference = json.loads(str(row["answer"]))
        exact = exact_keys(reference)
        by_code = valid_code_keys(reference)
        codes_by_key: dict[str, list[str]] = defaultdict(list)
        for code, key in by_code.items():
            codes_by_key[key].append(code)
        if set(codes_by_key) != exact:
            support_mismatches.append(
                {
                    "row": row_index,
                    "enumerated_only": sorted(exact - set(codes_by_key)),
                    "codec_only": sorted(set(codes_by_key) - exact),
                }
            )

        row_novel_counts: list[int] = []
        for anchor_key, anchor_codes in sorted(codes_by_key.items()):
            neighbor_keys = {
                candidate_key
                for candidate_code, candidate_key in by_code.items()
                if candidate_key != anchor_key
                and any(
                    0 < hamming_distance(anchor_code, candidate_code) <= radius
                    for anchor_code in anchor_codes
                )
            }
            if not neighbor_keys.issubset(exact):
                raise RuntimeError("local grammar admitted a key outside exact support")
            anchors += 1
            novel_counts.append(len(neighbor_keys))
            row_novel_counts.append(len(neighbor_keys))
            expandable += int(bool(neighbor_keys))
            if example is None and neighbor_keys:
                target_key = sorted(neighbor_keys)[0]
                target_code = min(
                    code for code, key in by_code.items() if key == target_key
                )
                source_code = min(
                    code
                    for code in anchor_codes
                    if 0 < hamming_distance(code, target_code) <= radius
                )
                example = {
                    "numbers": reference["numbers"],
                    "target": reference["target"],
                    "anchor_code": source_code,
                    "anchor_expression": decode_countdown_action_code(
                        source_code, reference
                    ),
                    "anchor_key": anchor_key,
                    "proposal_code": target_code,
                    "proposal_expression": decode_countdown_action_code(
                        target_code, reference
                    ),
                    "proposal_key": target_key,
                    "distance": hamming_distance(source_code, target_code),
                }
        rows_with_any += int(any(value > 0 for value in row_novel_counts))
        rows_with_all += int(all(value > 0 for value in row_novel_counts))

    return {
        "radius": radius,
        "rows": len(rows),
        "anchors": anchors,
        "expandable_anchors": expandable,
        "expandable_anchor_fraction": expandable / anchors if anchors else 0.0,
        "rows_with_any_expandable_anchor": rows_with_any,
        "rows_with_all_anchors_expandable": rows_with_all,
        "mean_novel_keys_per_anchor": (
            sum(novel_counts) / len(novel_counts) if novel_counts else 0.0
        ),
        "max_novel_keys_per_anchor": max(novel_counts, default=0),
        "support_mismatches": support_mismatches,
        "example": example,
    }


def audit_fixed_long_jumps(
    rows: list[dict[str, object]],
    *,
    budget: int,
    seed: int = 10_104,
) -> dict[str, object]:
    """Audit radius two plus a target-blind subset of distance-three codes."""

    anchors = 0
    expandable = 0
    local_failures = 0
    local_failures_recovered = 0
    novel_counts: list[int] = []
    selected_counts: list[int] = []
    rows_with_any = 0
    rows_with_all = 0
    all_codes = enumerate_countdown_action_codes()

    for row in rows:
        reference = json.loads(str(row["answer"]))
        by_code = valid_code_keys(reference)
        codes_by_key: dict[str, list[str]] = defaultdict(list)
        for code, key in by_code.items():
            codes_by_key[key].append(code)
        row_counts: list[int] = []
        for anchor_key, anchor_codes in sorted(codes_by_key.items()):
            local_keys = {
                candidate_key
                for candidate_code, candidate_key in by_code.items()
                if candidate_key != anchor_key
                and any(
                    0 < hamming_distance(anchor_code, candidate_code) <= 2
                    for anchor_code in anchor_codes
                )
            }
            distance_three = [
                code
                for code in all_codes
                if any(
                    hamming_distance(anchor_code, code) == 3
                    for anchor_code in anchor_codes
                )
            ]
            identity = json.dumps(
                {
                    "seed": seed,
                    "numbers": reference["numbers"],
                    "target": reference["target"],
                    "anchor_key": anchor_key,
                },
                sort_keys=True,
            )
            ordered = sorted(
                distance_three,
                key=lambda code: hashlib.sha256(
                    f"{identity}|{code}".encode()
                ).hexdigest(),
            )
            selected = ordered[:budget]
            selected_keys = {
                by_code[code]
                for code in selected
                if code in by_code and by_code[code] != anchor_key
            }
            neighbor_keys = local_keys | selected_keys
            anchors += 1
            expandable += int(bool(neighbor_keys))
            local_failures += int(not local_keys)
            local_failures_recovered += int(not local_keys and bool(selected_keys))
            novel_counts.append(len(neighbor_keys))
            selected_counts.append(len(selected))
            row_counts.append(len(neighbor_keys))
        rows_with_any += int(any(value > 0 for value in row_counts))
        rows_with_all += int(all(value > 0 for value in row_counts))

    return {
        "radius": 2,
        "distance_three_budget": budget,
        "selection_seed": seed,
        "rows": len(rows),
        "anchors": anchors,
        "expandable_anchors": expandable,
        "expandable_anchor_fraction": expandable / anchors if anchors else 0.0,
        "radius_two_failures": local_failures,
        "radius_two_failures_recovered": local_failures_recovered,
        "rows_with_any_expandable_anchor": rows_with_any,
        "rows_with_all_anchors_expandable": rows_with_all,
        "mean_novel_keys_per_anchor": (
            sum(novel_counts) / len(novel_counts) if novel_counts else 0.0
        ),
        "mean_distance_three_codes_selected": (
            sum(selected_counts) / len(selected_counts) if selected_counts else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "var/data/e101_open_bank_countdown_tiny",
    )
    args = parser.parse_args()
    train = load_from_disk(str(args.data_root / "train"))["train"]
    rows = [dict(row) for row in train]
    payload = {
        "schema": "open_bank_countdown_neighborhood_toy_v1",
        "data_root": str(args.data_root),
        "radii": [audit_rows(rows, radius) for radius in (1, 2)],
        "fixed_long_jumps": [
            audit_fixed_long_jumps(rows, budget=budget)
            for budget in (4, 8, 12, 16, 24, 32)
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
