#!/usr/bin/env python3
"""Verify the manuscript's results tables against the live aggregated surface.

Every number in Table `tab:main-results` and in the two per-seed appendix tables
is transcribed from the checkpoint aggregator. This check re-derives those
numbers from the scaling curves and compares them to what `main.tex` actually
says, so a cohort switch (E70 -> E71) or a checkpoint advancing from provisional
to terminal can never silently leave a stale figure in the paper.

Exit status is 0 when the manuscript matches, 1 when it does not. With
``--print-latex`` it emits the corrected table bodies so the swap is a paste,
not a re-typing.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_TEX = ROOT / "paper/main.tex"

_spec = importlib.util.spec_from_file_location(
    "e70agg", ROOT / "ops/exp_scaling/aggregate_e70_paper_checkpoints.py"
)
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)

# Table 1 label -> aggregator domain key. The manuscript shortens some names.
MAIN_ROW_TO_DOMAIN = {
    "Graph coloring": "Graph coloring",
    "Countdown": "Countdown",
    "Python factors": "Python factors",
    "MathIR action menu": "MathIR action menu",
    "PantryPlan": "PantryPlan",
}
# Appendix per-seed blocks use short names.
SEED_ROW_TO_DOMAIN = {
    "Graph": "Graph coloring",
    "Countdown": "Countdown",
    "Python": "Python factors",
    "MathIR": "MathIR action menu",
    "PantryPlan": "PantryPlan",
}
# Table 1 column order, then appendix column order.
MAIN_METRICS = ("greedy", "mean8", "pass8", "distinct8")
SEED_METRICS = ("pass8", "mean8", "distinct8", "greedy")


def fmt(value: float) -> str:
    return f"{value:.3f}".lstrip("0")


def load_surface() -> dict:
    """Re-derive per-domain endpoints and per-seed rows from the curves."""
    surface = {}
    for domain, (candidates, stage) in agg.DOMAINS.items():
        filename, index, terminal, _ = agg.resolve_curve(
            candidates, ROOT / "var/artifacts"
        )
        means, per_seed = {}, {}
        for arm in agg.ARMS:
            rows = [index[(arm, s, float(terminal))] for s in agg.SEEDS]
            means[arm] = {m: statistics.mean(r[m] for r in rows) for m in agg.METRICS}
            per_seed[arm] = {
                s: {m: index[(arm, s, float(terminal))][m] for m in agg.METRICS}
                for s in agg.SEEDS
            }
        surface[domain] = {
            "cohort": filename.split("_")[0],
            "terminal_pass": terminal,
            "complete": terminal == 12,
            "means": means,
            "per_seed": per_seed,
        }
    return surface


def parse_main_table(tex: str) -> list[tuple[str, int, list[str]]]:
    """Extract (domain, pass, eight cell strings) from tab:main-results."""
    block = re.search(
        r"\\label\{tab:main-results\}", tex
    )
    if not block:
        raise SystemExit("tab:main-results not found")
    body = tex[: block.start()]
    body = body[body.rindex("\\begin{table}") :]
    rows = []
    pattern = re.compile(
        r"^\s*([A-Za-z][A-Za-z ]*?)(?:\$\^\{\\[a-z]+\}\$)?\s*&\s*(\d+)\s*&(.*?)\\\\",
        re.M | re.S,
    )
    for match in pattern.finditer(body):
        name = match.group(1).strip()
        if name not in MAIN_ROW_TO_DOMAIN:
            continue
        cells = re.sub(r"\\textbf\{|\}", "", match.group(3))
        values = [v.strip() for v in cells.split("&") if v.strip()]
        rows.append((name, int(match.group(2)), values))
    return rows


def parse_seed_tables(tex: str) -> dict[str, dict]:
    """Extract per-seed appendix rows keyed by domain."""
    out: dict[str, dict] = {}
    current_domain = None
    current_arm = None
    header = re.compile(
        r"^\s*([A-Za-z]+)\s*\((\d+)\)\s*&\s*(Dr\.GRPO|\\xdr\{\})\s*&\s*(\d+)\s*&(.*?)\\\\"
    )
    cont_arm = re.compile(r"^\s*&\s*(Dr\.GRPO|\\xdr\{\})\s*&\s*(\d+)\s*&(.*?)\\\\")
    cont = re.compile(r"^\s*&\s*&\s*(\d+)\s*&(.*?)\\\\")
    for line in tex.splitlines():
        m = header.match(line)
        if m and m.group(1) in SEED_ROW_TO_DOMAIN:
            current_domain = SEED_ROW_TO_DOMAIN[m.group(1)]
            current_arm = agg.ARMS[0] if m.group(3) == "Dr.GRPO" else agg.ARMS[1]
            entry = out.setdefault(
                current_domain, {"pass": int(m.group(2)), "rows": {}}
            )
            entry["rows"].setdefault(current_arm, {})[int(m.group(4))] = [
                v.strip() for v in m.group(5).split("&") if v.strip()
            ]
            continue
        if current_domain is None:
            continue
        m = cont_arm.match(line)
        if m:
            current_arm = agg.ARMS[0] if m.group(1) == "Dr.GRPO" else agg.ARMS[1]
            out[current_domain]["rows"].setdefault(current_arm, {})[
                int(m.group(2))
            ] = [v.strip() for v in m.group(3).split("&") if v.strip()]
            continue
        m = cont.match(line)
        if m:
            out[current_domain]["rows"].setdefault(current_arm, {})[
                int(m.group(1))
            ] = [v.strip() for v in m.group(2).split("&") if v.strip()]
    return out


def latex_main_rows(surface: dict) -> list[str]:
    lines = []
    for name, domain in MAIN_ROW_TO_DOMAIN.items():
        info = surface[domain]
        control = info["means"][agg.ARMS[0]]
        treat = info["means"][agg.ARMS[1]]
        cells = [fmt(control[m]) for m in MAIN_METRICS]
        bold = [
            f"\\textbf{{{fmt(treat[m])}}}" if treat[m] > control[m] else fmt(treat[m])
            for m in MAIN_METRICS
        ]
        lines.append(
            f"    {name} & {info['terminal_pass']} & "
            + " & ".join(cells)
            + " & &\n      "
            + " & ".join(bold)
            + " \\\\"
        )
    return lines


def latex_seed_rows(surface: dict, domains: list[str]) -> list[str]:
    lines = []
    for short, domain in SEED_ROW_TO_DOMAIN.items():
        if domain not in domains:
            continue
        info = surface[domain]
        first = True
        for arm in agg.ARMS:
            label = "Dr.GRPO" if arm == agg.ARMS[0] else "\\xdr{}"
            for i, seed in enumerate(agg.SEEDS):
                vals = info["per_seed"][arm][seed]
                cells = " & ".join(fmt(vals[m]) for m in SEED_METRICS)
                if first:
                    lines.append(
                        f"    {short} ({info['terminal_pass']}) & {label} & {seed} & {cells} \\\\"
                    )
                    first = False
                elif i == 0:
                    lines.append(f"    & {label} & {seed} & {cells} \\\\")
                else:
                    lines.append(f"    & & {seed} & {cells} \\\\")
        lines.append("    \\midrule")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-latex", action="store_true")
    args = parser.parse_args()

    tex = MAIN_TEX.read_text(encoding="utf-8")
    surface = load_surface()
    problems: list[str] = []

    print("cohort / endpoint per domain:")
    for domain, info in surface.items():
        flag = "" if info["complete"] else "  PROVISIONAL"
        print(
            f"  {domain:20s} [{info['cohort']}] pass {info['terminal_pass']}{flag}"
        )

    main_rows = parse_main_table(tex)
    seen = {name for name, _, _ in main_rows}
    if seen != set(MAIN_ROW_TO_DOMAIN):
        problems.append(
            f"tab:main-results rows {sorted(seen)} != {sorted(MAIN_ROW_TO_DOMAIN)}"
        )
    for name, tex_pass, values in main_rows:
        info = surface[MAIN_ROW_TO_DOMAIN[name]]
        if tex_pass != info["terminal_pass"]:
            problems.append(
                f"tab:main-results {name}: pass {tex_pass} in tex, "
                f"{info['terminal_pass']} in data"
            )
        expected = [
            fmt(info["means"][arm][m]) for arm in agg.ARMS for m in MAIN_METRICS
        ]
        if values != expected:
            problems.append(
                f"tab:main-results {name}: tex={values} expected={expected}"
            )

    seed_tables = parse_seed_tables(tex)
    for short, domain in SEED_ROW_TO_DOMAIN.items():
        if domain not in seed_tables:
            problems.append(f"per-seed table missing block for {short}")
            continue
        info = surface[domain]
        entry = seed_tables[domain]
        if entry["pass"] != info["terminal_pass"]:
            problems.append(
                f"per-seed {short}: pass {entry['pass']} in tex, "
                f"{info['terminal_pass']} in data"
            )
        for arm in agg.ARMS:
            for seed in agg.SEEDS:
                got = entry["rows"].get(arm, {}).get(seed)
                want = [fmt(info["per_seed"][arm][seed][m]) for m in SEED_METRICS]
                if got is None:
                    problems.append(f"per-seed {short}/{arm}/s{seed}: missing")
                elif got != want:
                    problems.append(
                        f"per-seed {short}/{arm}/s{seed}: tex={got} expected={want}"
                    )

    if args.print_latex:
        print("\n=== tab:main-results body ===")
        print("\n".join(latex_main_rows(surface)))
        print("\n=== tab:per-seed body (four Stage-A domains) ===")
        print(
            "\n".join(
                latex_seed_rows(
                    surface,
                    ["Graph coloring", "Countdown", "Python factors",
                     "MathIR action menu"],
                )
            )
        )
        print("\n=== tab:pantry-per-seed body ===")
        print("\n".join(latex_seed_rows(surface, ["PantryPlan"])))

    print()
    if problems:
        print(f"MANUSCRIPT OUT OF DATE: {len(problems)} mismatch(es)")
        for problem in problems:
            print(f"  - {problem}")
        print("\nRe-run with --print-latex to emit the corrected table bodies.")
        return 1
    print("manuscript results tables match the aggregated surface")
    return 0


if __name__ == "__main__":
    sys.exit(main())
