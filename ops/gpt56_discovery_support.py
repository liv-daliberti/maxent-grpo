#!/usr/bin/env python3
"""Certify available-mode references without reading model outcomes.

Graph enumerates every allowed missing-color assignment. Countdown enumerates
all binary expression trees, but deliberately reports a lower bound: the hosted
validator also accepts unary negation and retains it in its canonical key.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
import hashlib
import importlib
from itertools import permutations, product
import json
from pathlib import Path
import sys
from typing import Callable, Optional

Verifier = Callable[[str, str], Optional[str]]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def graph_modes(spec: dict) -> dict[str, str]:
    """Enumerate full-color-vector keys compatible with fixed colors and edges."""
    n = int(spec["n"])
    partial = spec.get("partial_colors", [None] * n)
    if len(partial) != n or not 1 <= n <= 20:
        raise ValueError("Invalid graph size or partial coloring")
    if any(c is not None and c not in (1, 2, 3) for c in partial):
        raise ValueError("Fixed graph colors must belong to {1,2,3}")
    hidden = [i for i, c in enumerate(partial) if c is None]
    if len(hidden) > 12:
        raise ValueError("Graph exceeds this certificate's enumeration budget")
    edges = [(int(u) - 1, int(v) - 1) for u, v in spec["edges"]]
    if any(not (0 <= u < n and 0 <= v < n) for u, v in edges):
        raise ValueError("Graph edge outside vertex range")
    modes = {}
    for fill in product((1, 2, 3), repeat=len(hidden)):
        colors = list(partial)
        for i, c in zip(hidden, fill):
            colors[i] = c
        if all(colors[u] != colors[v] for u, v in edges):
            text = "".join(str(c) for c in colors)
            modes["graph_coloring:" + text] = text
    return modes


def binary_countdown_modes(spec: dict) -> dict[str, str]:
    """All binary +/-/*// trees, exact rational values, commutative-key aliases merged."""
    numbers = tuple(spec["numbers"])
    if not 1 <= len(numbers) <= 4 or any(type(n) is not int or n <= 0 for n in numbers):
        raise ValueError("Expected one to four positive integer Countdown operands")
    target = Fraction(int(spec["target"]))

    @lru_cache(None)
    def build(order: tuple[int, ...]) -> dict:
        if len(order) == 1:
            atom = str(order[0])
            return {Fraction(order[0]): {atom: atom}}
        result = defaultdict(dict)
        for split in range(1, len(order)):
            for lv, lefts in build(order[:split]).items():
                for rv, rights in build(order[split:]).items():
                    for lk, le in lefts.items():
                        for rk, re in rights.items():
                            a, b = sorted((lk, rk))
                            candidates = [(lv + rv, f"add({a},{b})", f"({le}+{re})"),
                                          (lv - rv, f"sub({lk},{rk})", f"({le}-{re})"),
                                          (lv * rv, f"mul({a},{b})", f"({le}*{re})")]
                            if rv:
                                candidates.append((lv / rv, f"div({lk},{rk})", f"({le}/{re})"))
                            for value, key, expression in candidates:
                                result[value].setdefault(key, expression)
        return dict(result)

    modes = {}
    for order in sorted(set(permutations(numbers))):
        for key, expression in build(order).get(target, {}).items():
            modes.setdefault("countdown:" + key, expression)
    return modes


def certify_row(row: dict, verifier: Verifier) -> dict:
    answer = row["answer"]
    spec = json.loads(answer) if isinstance(answer, str) else answer
    answer_text = answer if isinstance(answer, str) else json.dumps(answer, sort_keys=True)
    domain = row["domain"]
    if domain == "graph_coloring":
        modes = graph_modes(spec)
        kind = "exact"
        rationale = ("Exhaustive enumeration of all assignments from {1,2,3} to missing "
                     "vertices, retaining fixed colors and enforcing every edge. Every "
                     "canonical full-color-vector key is checked by the frozen validator.")
    elif domain == "countdown":
        modes = binary_countdown_modes(spec)
        kind = "certified_lower_bound"
        rationale = ("Exhaustive binary-tree enumeration using each given operand once "
                     "with +,-,*,/ and exact rational execution; commutative aliases merge. "
                     "This is a lower bound for the hosted interface: unary negation is "
                     "also accepted and remains in the canonical key. No exhaustive "
                     "finite hosted support total is asserted.")
    else:
        raise ValueError(f"Unsupported new support domain: {domain}")
    if not modes:
        raise ValueError("No feasible modes found")
    for key, expression in modes.items():
        actual = verifier("\\boxed{" + expression + "}", answer_text)
        if actual != key:
            raise ValueError(f"Frozen verifier disagrees with support witness: {key}, {actual}")
    declared = spec.get("num_completions")
    if declared is not None and len(modes) != int(declared):
        raise ValueError(f"Enumerated {len(modes)} modes but metadata declared {declared}")
    unary_witness = None
    if domain == "countdown":
        expression = next(iter(modes.values()))
        text = "\\boxed{-(-(" + expression + "))}"
        actual = verifier(text, answer_text)
        if actual is None or actual in modes:
            raise ValueError("Frozen Countdown unary-key behavior differs from certificate rationale")
        unary_witness = {"text": text, "canonical_key": actual,
                         "included_in_reference_count": False}
    keys = sorted(modes)
    return {"pair_id": f"L{row['level']}_{domain}_{row['row_index']:03d}",
            "level": row["level"], "domain": domain, "row_index": row["row_index"],
            "support_count": len(modes), "support_kind": kind,
            "source_field": "independent_enumeration_with_frozen_verifier",
            "declared_answer_mode_count": declared, "rationale": rationale,
            "canonical_keys": keys,
            "key_sha256": hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest(),
            "witnesses": modes, "unary_extension_witness": unary_witness}


def build_reference(rows_path: Path, legacy_path: Path, code_root: Path) -> dict:
    # A fresh CLI process imports the exact collection's snapshot, not active code.
    if "oat_drgrpo.math_grader" in sys.modules:
        raise ValueError("Run in a fresh process to avoid a previously imported verifier")
    sys.path.insert(0, str((code_root / "src").resolve()))
    grader = importlib.import_module("oat_drgrpo.math_grader")
    grader_file = Path(grader.__file__).resolve()
    if not grader_file.is_relative_to(code_root.resolve()):
        raise ValueError("Verifier import escaped the frozen code root")
    legacy = json.loads(legacy_path.read_text())
    refs = dict(legacy["references"])
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    if len(rows) != 64 or {(r["level"], r["domain"]) for r in rows} != {
            (l, d) for l in (2, 3) for d in ("graph_coloring", "countdown")}:
        raise ValueError("Expected 16 new problems in each of four Graph/Countdown cells")
    by_cell = defaultdict(int)
    for row in rows:
        ref = certify_row(row, grader.validated_modebench_outcome_key)
        if ref["pair_id"] in refs:
            raise ValueError("Duplicate support reference")
        refs[ref["pair_id"]] = ref
        by_cell[row["level"], row["domain"]] += 1
    if set(by_cell.values()) != {16}:
        raise ValueError("Unequal Graph/Countdown cell sizes")
    cells = defaultdict(list)
    for ref in refs.values():
        cells[f"level{ref['level']}/{ref['domain']}"].append(ref)
    summaries = {cell: {"prompts": len(rs),
                       "mean_support_count": sum(r["support_count"] for r in rs) / len(rs),
                       "min_support_count": min(r["support_count"] for r in rs),
                       "max_support_count": max(r["support_count"] for r in rs),
                       "support_kinds": sorted({r["support_kind"] for r in rs})}
                 for cell, rs in sorted(cells.items())}
    return {"schema": "gpt56-five-domain-discovery-support-v1", "status": "complete",
            "sources": {"new_rows": binding(rows_path), "legacy_support": binding(legacy_path),
                        "frozen_grader": binding(grader_file), "support_helper": binding(Path(__file__))},
            "outcomes_read": False, "selection_from_outcomes": False,
            "selected_rows": len(refs), "references": refs, "cells": summaries,
            "support_kind_definitions": {"exact": "All canonical keys accepted by the declared interface are counted.",
                "certified_lower_bound": "At least this many distinct valid canonical keys exist; this is not total support."},
            "extension_analysis_contract": {
                "new_draws_are_independent": True,
                "no_empirical_64_pool_extrapolation": True,
                "retain_fixed_original_problem_sets_and_controls": True,
                "report_all_completed_budgets": True,
                "beyond64_requires_actual_new_native_receipts": True,
                "late_gain": "Mean D(2k)-D(k) using actual cumulative pools, with whole-problem uncertainty.",
                "stopping": "Any data-dependent stopping and pointwise intervals are exploratory; a finite observed plateau does not establish zero mass on unobserved modes.",
                "support": "Observed/reference-count ratios are exact coverage only for exact references; lower-bound ratios may exceed one."}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--legacy-support", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to replace an existing support certificate")
    result = build_reference(args.rows, args.legacy_support, args.code_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "cells": result["cells"]}, indent=2))


if __name__ == "__main__":
    main()
