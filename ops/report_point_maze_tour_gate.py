#!/usr/bin/env python
"""Report the PointMaze Tour admission gates from whatever has finished.

Safe to run at any time: it reads only the metrics files the runs append as
they go, so a partially complete cohort reports partial trends rather than
failing. It applies the registered thresholds and never selects a checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics as st

ARTIFACTS = Path("var/artifacts")

STAGE1_GATES = {
    "pass8": ("pass@8", lambda v: v >= 0.70, ">= 0.70"),
    "mean8": ("mean@8", lambda v: 0.20 <= v <= 0.65, "0.20-0.65"),
    "distinct8": ("distinct@8", lambda v: v >= 2.5, ">= 2.5"),
}
COLLAPSE_DROP = 0.5


EVAL_SCHEMA = "point-maze-tour-evaluation-v1"


def load(stem: Path) -> list[dict]:
    """Evaluation rows only.

    The metrics file interleaves one training row per update with the periodic
    evaluations; only the latter carry distinct@8.
    """

    if not stem.exists():
        return []
    rows = [json.loads(line) for line in stem.read_text().splitlines() if line.strip()]
    return [row for row in rows if row.get("schema") == EVAL_SCHEMA]


def load_any(stem: Path) -> list[dict]:
    if not stem.exists():
        return []
    return [json.loads(line) for line in stem.read_text().splitlines() if line.strip()]


def report_stage1(pattern: str) -> None:
    print("STAGE 1 - base policy viability (untouched model, no warm start)")
    found = sorted(ARTIFACTS.glob(f"{pattern}*.metrics.jsonl"))
    if not found:
        print("  (no stage-1 metrics yet)\n")
        return
    for path in found:
        rows = load(path)
        if not rows:
            continue
        r = rows[-1]
        pm = r.get("per_map", [])
        two = sum(1 for m in pm if m["distinct8"] >= 2) / len(pm) if pm else 0.0
        verdicts = []
        for key, (label, ok, text) in STAGE1_GATES.items():
            v = r[key]
            verdicts.append(f"{label}={v:.3f} {'PASS' if ok(v) else 'FAIL'} ({text})")
        verdicts.append(f">=2 modes={two:.3f} {'PASS' if two >= 0.60 else 'FAIL'} (>= 0.60)")
        print(f"  {path.name}")
        print(f"    " + "  ".join(verdicts))
    print()


def report_collapse(prefix: str) -> None:
    print("STAGE 2 - collapse gate: control must LOSE >= 0.50 distinct@8")
    paths = sorted(ARTIFACTS.glob(f"{prefix}_control_s*.metrics.jsonl"))
    if not paths:
        print("  (no control metrics yet - jobs still queued)\n")
        return
    drops = []
    for path in paths:
        rows = load(path)
        if not rows:
            print(f"  {path.name}: started, no evaluation written yet")
            continue
        trend = " -> ".join(f"{r['distinct8']:.2f}" for r in rows)
        first, last = rows[0]["distinct8"], rows[-1]["distinct8"]
        drop = first - last
        drops.append(drop)
        passes = rows[-1]["learning_round"] / 384
        print(f"  {path.name}")
        print(f"    distinct@8 by pass: {trend}   (through pass {passes:.0f})")
        print(f"    mean@8     by pass: " + " -> ".join(f"{r['mean8']:.2f}" for r in rows))
        print(f"    drop = {drop:+.3f}   {'PASS' if drop >= COLLAPSE_DROP else 'not yet'}")
    if drops:
        mean_drop = st.mean(drops)
        print(f"  mean drop across {len(drops)} seed(s): {mean_drop:+.3f} "
              f"-> {'DOMAIN ADMITTED' if mean_drop >= COLLAPSE_DROP else 'gate not met yet'}")
    print()


def report_smoke(prefix: str) -> None:
    print("STAGE 3 - matched smoke (replay arm plumbing)")
    paths = sorted(ARTIFACTS.glob(f"{prefix}_replay_s*.metrics.jsonl"))
    if not paths:
        print("  (no replay metrics yet)\n")
        return
    for path in paths:
        rows = [r for r in load_any(path) if "replay_active_modes" in r]
        if not rows:
            print(f"  {path.name}: started, no training evaluation yet")
            continue
        r = rows[-1]
        banked = r.get("replay_bank_tracked_outcomes", 0.0)
        active = r.get("replay_active_modes", 0.0)
        applied = r.get("replay_applied_score_gradient_l2", 0.0)
        print(f"  {path.name}")
        print(f"    banked outcomes={banked:.0f}  active replay modes={active:.1f}  "
              f"applied replay grad L2={applied:.4f}")
        print(f"    {'PASS' if banked > 0 and applied > 0 else 'FAIL'} "
              "(bank must fill and the replay derivative must be nonzero)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-pattern", default="tour_stage1")
    parser.add_argument("--stage2-prefix", default="tour_stage2")
    parser.add_argument("--stage3-prefix", default="tour_stage23")
    args = parser.parse_args()
    report_stage1(args.stage1_pattern)
    report_collapse(args.stage2_prefix)
    report_smoke(args.stage3_prefix)


if __name__ == "__main__":
    main()
