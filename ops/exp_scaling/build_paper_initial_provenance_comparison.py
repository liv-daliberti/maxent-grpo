#!/usr/bin/env python3
"""Report the before/after contrast under both initial-checkpoint admissions.

Amendment 4 admits conflicted step-0 endpoints by attempt provenance. It was
adopted after the original analysis was specified, so its reporting requirement
is that every affected longitudinal block appears under both admissions with
its defined-seed count, whatever direction the re-admission produces.

This program reads the two frozen results and renders that comparison. It
computes no new estimate: every value is copied from a bound analysis artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PRESERVED = ROOT / "paper/results/conditional_concentration_20260912.json"
AMENDED = ROOT / "paper/results/conditional_concentration_20260914.json"
BINDING = (ROOT / "paper/audits/conditional_concentration_20260914"
           / "amendment_4_initial_checkpoint_provenance_binding.json")
DEFAULT_OUTPUT = ROOT / "paper/results/initial_provenance_comparison"

DOMAIN_LABEL = {
    "graph_coloring": "Graph", "countdown": "Countdown", "python_factors": "Python",
    "mathir": "MathIR", "pantry_plan": "PantryPlan",
}
METHOD_LABEL = {
    "drgrpo": "Dr.GRPO", "grpo": "GRPO", "maxrl": "MaxRL",
    "replay_drgrpo": "Re:Dr", "replay_maxrl": "Re:Max",
}
SCALE_LABEL = {"qwen05b": "Qwen2.5-0.5B", "falcon1b": "Falcon3-1B", "qwen3b": "Qwen2.5-3B"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index(result: dict) -> dict:
    return {(b["kind"], b["level"], b["scale"], b["domain"], b["method"]): b
            for b in result["blocks"]}


TOLERANCE = 1e-9


def deviation(a: dict | None, b: dict | None) -> float:
    """Largest absolute difference between two block summaries.

    Unaffected blocks are recomputed from identical inputs, so they agree to
    summation order rather than bitwise; a tolerance separates that from a real
    change without hiding one.
    """
    if a is None or b is None:
        return 0.0 if a == b else float("inf")
    if a["n"] != b["n"]:
        return float("inf")
    worst = 0.0
    for field in ("mean",):
        if (a[field] is None) != (b[field] is None):
            return float("inf")
        if a[field] is not None:
            worst = max(worst, abs(a[field] - b[field]))
    if (a["ci95"] is None) != (b["ci95"] is None):
        return float("inf")
    if a["ci95"] is not None:
        worst = max(worst, max(abs(x - y) for x, y in zip(a["ci95"], b["ci95"])))
    return worst


def summary(block: dict | None) -> dict | None:
    if block is None:
        return None
    s = block["summaries"]["distinct_streams"]
    return {"n": s.get("n", 0), "mean": s.get("mean"), "ci95": s.get("ci95")}


def build_record() -> dict[str, Any]:
    binding = json.loads(BINDING.read_text())
    before, after = index(json.loads(PRESERVED.read_text())), index(json.loads(AMENDED.read_text()))
    affected = {(c["level"], c["scale"], c["domain"], c["method"]) for c in binding["affected_cells"]}
    rows, unchanged_headline, worst_unaffected = [], [], 0.0
    for key in sorted(set(before) | set(after)):
        if key[0] != "before_after":
            continue
        a, b = summary(before.get(key)), summary(after.get(key))
        if key[1:] in affected:
            rows.append({"level": key[1], "scale": key[2], "domain": key[3], "method": key[4],
                         "preserved": a, "amended": b})
        else:
            drift = deviation(a, b)
            if drift > TOLERANCE:
                raise ValueError(f"amendment 4 changed an unaffected block: {key} by {drift}")
            worst_unaffected = max(worst_unaffected, drift)
        if key[4] in ("drgrpo", "grpo") and key[3] in ("graph_coloring", "pantry_plan"):
            unchanged_headline.append({"block": list(key[1:]),
                                       "deviation": deviation(a, b)})
    replay_changed = [list(k[1:]) for k in set(before) | set(after)
                      if k[0] == "replay_effect"
                      and deviation(summary(before.get(k)), summary(after.get(k))) > TOLERANCE]
    return {
        "schema": "initial-provenance-comparison-v1",
        "analysis_code_sha256": sha(Path(__file__)),
        "amendment": {"path": binding["amendment_path"], "sha256": binding["amendment_sha256"]},
        "sources": {"preserved": {"path": str(PRESERVED.relative_to(ROOT)), "sha256": sha(PRESERVED)},
                    "amended": {"path": str(AMENDED.relative_to(ROOT)), "sha256": sha(AMENDED)}},
        "rows": rows,
        "headline_twelve_unchanged": all(r["deviation"] <= TOLERANCE for r in unchanged_headline),
        "worst_unaffected_deviation": worst_unaffected,
        "unaffected_tolerance": TOLERANCE,
        "headline_twelve_checked": len(unchanged_headline),
        "replay_effect_blocks_changed": replay_changed,
    }


def render_table(record: dict[str, Any]) -> str:
    lines = []
    for row in record["rows"]:
        def cell(s):
            if s is None or s["mean"] is None:
                return r"\multicolumn{1}{c}{---}"
            ci = s["ci95"]
            # Keep the interval in math mode so its signs match the mean's.
            interval = f" $[{ci[0]:+.3f}, {ci[1]:+.3f}]$" if ci else ""
            return f"{s['n']} & ${s['mean']:+.3f}${interval}"
        lines.append(
            f"    {SCALE_LABEL.get(row['scale'], row['scale'])} & "
            f"{DOMAIN_LABEL.get(row['domain'], row['domain'])} & "
            f"{METHOD_LABEL.get(row['method'], row['method'])} & "
            f"{cell(row['preserved'])} & {cell(row['amended'])} \\\\")
    return "\n".join(lines) + "\n    \\bottomrule\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    record = build_record()
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    Path(str(args.output) + "_table_body.tex").write_text(render_table(record))
    reversals = [r for r in record["rows"]
                 if r["amended"] and r["amended"]["mean"] is not None and r["amended"]["mean"] < 0]
    print(f"{len(record['rows'])} affected blocks; headline twelve unchanged: "
          f"{record['headline_twelve_unchanged']}; replay-effect blocks changed: "
          f"{len(record['replay_effect_blocks_changed'])}")
    for r in reversals:
        print(f"  opposite direction: {r['scale']}/{r['domain']}/{r['method']} "
              f"{r['amended']['mean']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
