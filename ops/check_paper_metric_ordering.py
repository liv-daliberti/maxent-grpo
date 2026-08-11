#!/usr/bin/env python3
"""Check every printed metric triple in the manuscript against Lemma 2.2.

The lemma orders the quartet pointwise: mean@K <= pass@K <= distinct@K. A table
that prints a triple violating it is not a debatable presentation choice, it is
evidence that the wrong telemetry key was read --- which is exactly what
happened to the cross-family table, where `sampled_mode_coverage_at_8`
(distinct modes divided by the domain's total mode count) was printed under a
`distinct@8` heading and landed below `pass@8`.

The check is deliberately mechanical and reads the manuscript rather than the
artifacts, because the artifacts were right and the transcription was wrong. It
parses the tables whose column layout is declared below, so a new table is
covered by describing it here rather than by hoping someone re-derives the
lemma by eye.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# Tables that print a metric triple, and where the numbers sit in each row.
# `arms` gives, per arm, the (pass_index, distinct_index) positions among the
# numeric fields of a row, counting from zero after the leading epoch column.
TABLES = {
    "tab:cross-family": {
        "description": "cross-family replication (Falcon3-1B)",
        # One row per arm: p@1, mean@8, pass@8, #modes, modes-per-success.
        # The triple is (mean@8, pass@8, distinct@8), so the full ordering of
        # Lemma 2.2 is checked rather than only its second inequality.
        "triples": [(1, 2, 3)],
        "expected_fields": 5,
    },
}

NUMBER = re.compile(r"(?<![\w.])(\d*\.\d+|\d+\.\d*|\d+)(?![\d.])")


def numeric_fields(row: str) -> list[float]:
    """Every number in a table row, with the leading epoch column dropped.

    LaTeX bolding, alignment marks, and inter-arm spacer columns are stripped
    first so a row reads as the sequence of quantities a reader sees.
    """
    body = row.replace("\\textbf", " ").replace("\\\\", " ")
    body = re.sub(r"[{}&]", " ", body)
    values = [float(match) for match in NUMBER.findall(body)]
    # The epoch column is an integer count, not a metric.
    return values[1:] if values and values[0] >= 1 and values[0] == int(values[0]) else values


def table_rows(text: str, label: str) -> list[str]:
    start = text.find("\\begin{table}", 0)
    marker = f"\\label{{{label}}}"
    end = text.find(marker)
    if end < 0:
        return []
    start = text.rfind("\\begin{table}", 0, end)
    block = text[start:end]
    body = block[block.find("\\midrule") : block.find("\\bottomrule")]
    return [row.strip() for row in body.split("\\\\") if "&" in row]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, default=root / "paper" / "main.tex")
    args = parser.parse_args()

    text = args.manuscript.read_text(encoding="utf-8")
    problems: list[str] = []
    checked = 0

    for label, spec in TABLES.items():
        rows = table_rows(text, label)
        if not rows:
            problems.append(f"{label}: table not found in {args.manuscript}")
            continue
        for row in rows:
            values = numeric_fields(row)
            if len(values) != spec["expected_fields"]:
                problems.append(
                    f"{label}: row has {len(values)} numeric fields, expected "
                    f"{spec['expected_fields']}: {row[:70]}"
                )
                continue
            leading = " ".join(row.split("&")[:2])
            leading = re.sub(r"\\[A-Za-z]+|[{}]|\\midrule|\\addlinespace.*", " ", leading)
            leading = " ".join(leading.split())
            for mean_at, pass_at, distinct_at in spec["triples"]:
                checked += 1
                ordered = (
                    values[mean_at] <= values[pass_at] + 1e-9
                    and values[pass_at] <= values[distinct_at] + 1e-9
                )
                if not ordered:
                    problems.append(
                        f"{label} [{leading}]: mean@8 {values[mean_at]}, pass@8 "
                        f"{values[pass_at]}, distinct@8 {values[distinct_at]} "
                        "violates Lemma 2.2 (is a printed column mode coverage "
                        "rather than the distinct-mode count?)"
                    )

    for problem in problems:
        print(f"  {problem}")
    if problems:
        print(f"[metric-ordering] FAILED: {len(problems)} problem(s)")
        return 1
    print(f"[metric-ordering] {checked} printed metric pairs satisfy Lemma 2.2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
