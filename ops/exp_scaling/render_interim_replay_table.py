#!/usr/bin/env python3
"""Render the manuscript's interim replay table body from its frozen snapshot.

The table in `tab:interim-replay` restates numbers that live in
`paper/results/figure4_interim_20260806_table.json`. Hand-editing it after each
snapshot refresh is how the two drift apart, so this renders the body straight
from the snapshot using exactly the conventions
`ops/check_paper_figure1_contract.py` enforces: paired-seed means to three
decimals with the leading zero stripped, `--` for unavailable cells, and
`\\bettercell{}` on an x-Mode value strictly greater than its matched Dr.GRPO
value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "paper/results/figure4_interim_20260806_table.json"
MANUSCRIPT = ROOT / "paper/main.tex"
FAMILY_ORDER = ("Qwen2.5-0.5B", "Falcon3-1B", "Qwen2.5-3B")
FAMILY_TEX = {
    "Qwen2.5-0.5B": r"\qwenmark{}2.5-0.5B",
    "Falcon3-1B": "Falcon3-1B",
    "Qwen2.5-3B": r"\qwenmark{}2.5-3B",
}
sys.path.insert(0, str(Path(__file__).resolve().parent))
from status_e78 import DOMAIN_TITLES  # noqa: E402
METRICS = ("pass1", "pass8", "mean8", "distinct8")


def decimal3(value: float) -> str:
    rendered = f"{value:.3f}"
    if rendered.startswith("0."):
        return rendered[1:]
    if rendered.startswith("-0."):
        return "-" + rendered[2:]
    return rendered


def cells(domain_record: dict[str, Any]) -> list[str]:
    control = domain_record["arms"]["control"]["means"]
    replay = domain_record["arms"]["replay"]["means"]
    row = [
        f"{len(domain_record['paired_seeds'])}@{float(domain_record['pass']):.1f}"
    ]
    for metric in METRICS:
        value = control.get(metric)
        row.append("--" if value is None else decimal3(float(value)))
    for metric in METRICS:
        value = replay.get(metric)
        if value is None:
            row.append("--")
            continue
        rendered = decimal3(float(value))
        against = control.get(metric)
        better = against is not None and float(value) > float(against)
        row.append(rf"\bettercell{{{rendered}}}" if better else rendered)
    return row


def body(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    for index, family in enumerate(FAMILY_ORDER):
        record = payload["families"].get(family)
        if not record:
            continue
        if index:
            lines.append(r"    \addlinespace[2pt]")
        for position, (domain, domain_record) in enumerate(
            record["domains"].items()
        ):
            title = DOMAIN_TITLES[domain]
            head = (
                f"    {FAMILY_TEX[family]} & {title}"
                if position == 0
                else f"      & {title:<14}"
            )
            lines.append(head + " & " + " & ".join(cells(domain_record)) + r" \\")
    return "\n".join(lines)


def splice(manuscript: str, rendered: str) -> str:
    label = r"\label{tab:interim-replay}"
    anchor = manuscript.find(label)
    if anchor < 0:
        raise SystemExit("manuscript has no tab:interim-replay")
    start = manuscript.find(r"\midrule", anchor)
    stop = manuscript.find(r"\bottomrule", start)
    if start < 0 or stop < 0:
        raise SystemExit("could not locate the interim table body")
    head = manuscript[: start + len(r"\midrule")]
    tail = manuscript[stop:]
    return head + "\n" + rendered + "\n    " + tail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=TABLE)
    parser.add_argument("--manuscript", type=Path, default=MANUSCRIPT)
    parser.add_argument(
        "--write",
        action="store_true",
        help="splice the rendered body into the manuscript in place",
    )
    args = parser.parse_args()

    payload = json.loads(args.table.read_text(encoding="utf-8"))
    rendered = body(payload)
    if not args.write:
        print(rendered)
        return 0

    manuscript = args.manuscript.read_text(encoding="utf-8")
    updated = splice(manuscript, rendered)
    if updated == manuscript:
        print("interim replay table already matches the snapshot")
        return 0
    args.manuscript.write_text(updated, encoding="utf-8")
    rows = len(re.findall(r"\\\\", rendered))
    print(f"rewrote {rows} interim table rows in {args.manuscript}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
