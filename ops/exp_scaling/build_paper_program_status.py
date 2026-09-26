#!/usr/bin/env python3
"""Freeze the canonical experiment-program coverage table used by the paper.

The 10 x 3 x 6 x 5 matrix is an organization target, not one completed
factorial. This builder records that distinction and derives every count from
paper_matrix.py so manuscript prose cannot silently conflate missing,
incomplete, blocked, and terminal cells.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import paper_matrix as matrix


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = ROOT / "paper/results/paper_program_status.json"
DEFAULT_TEX = ROOT / "paper/results/paper_program_status_table_body.tex"


def _latex(text: str) -> str:
    return (
        text.replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def build_payload(cells: dict[matrix.CellKey, matrix.Cell]) -> dict[str, Any]:
    rows = []
    desired = matrix.desired_keys()
    for method in matrix.METHODS:
        keys = [key for key in desired if key.method == method.key]
        status = matrix.counts(cells, keys)
        registered = len(keys) - status["missing"]
        active = status["running"] + status["pending"]
        rows.append(
            {
                "key": method.key,
                "label": method.label,
                "paper_role": method.paper_role,
                "description": method.description,
                "target": len(keys),
                "registered": registered,
                "terminal": status["terminal"],
                "active": active,
                "blocked": status["blocked"],
                "partial": status["partial"],
                "failed": status["failed"],
                "inactive": status["inactive"],
                "missing": status["missing"],
            }
        )
    return {
        "schema": "modebench-paper-program-status-v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status_scope": "five-domain coverage ledger; figures retain every available checkpoint with exact n",
        "shape": {
            "methods": len(matrix.METHODS),
            "scales": len(matrix.SCALES),
            "domains": len(matrix.DOMAINS),
            "seeds_per_scale": 5,
            "cells": len(desired),
        },
        "registered": len(cells),
        "terminal": sum(row["terminal"] for row in rows),
        "missing": sum(row["missing"] for row in rows),
        "methods": rows,
    }


def _open_state(row: dict[str, Any]) -> str:
    parts = []
    for field, label in (
        ("active", "active"),
        ("blocked", "blocked"),
        ("partial", "partial"),
        ("failed", "failed"),
        ("inactive", "inactive"),
        ("missing", "unregistered"),
    ):
        value = int(row[field])
        if value:
            parts.append(f"{value} {label}")
    return "; ".join(parts) if parts else "none"


def render_tex(payload: dict[str, Any]) -> str:
    lines = []
    for row in payload["methods"]:
        lines.append(
            " & ".join(
                (
                    rf"\textbf{{{_latex(str(row['label']))}}}",
                    _latex(str(row["paper_role"])),
                    _latex(str(row["description"])),
                    f"{row['registered']}/{row['target']}",
                    str(row["terminal"]),
                    _latex(_open_state(row)),
                )
            )
            + r" \\" 
        )
    lines.append(r"    \bottomrule")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--tex-output", type=Path, default=DEFAULT_TEX)
    args = parser.parse_args()

    payload = build_payload(matrix.load_cells())
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.tex_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.tex_output.write_text(render_tex(payload), encoding="utf-8")
    print(
        f"paper program: {payload['terminal']}/{payload['shape']['cells']} terminal; "
        f"{payload['registered']} registered; wrote {args.json_output} and "
        f"{args.tex_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
