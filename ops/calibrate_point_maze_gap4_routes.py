#!/usr/bin/env python3
"""Calibrate token counts for the redesigned four-gap PointMaze geometry.

The v1 environment hardcodes two routes per map, so `distinct@8` can never
exceed 2. This geometry puts a vertical wall down the middle with four separated
gaps, giving four homotopy classes. Token counts per leg are not analytic -- the
point mass accelerates -- so each route is calibrated against the real simulator
and kept only if the verifier certifies it AND attributes it to the intended
gap.

Writes the calibrated counts to var/artifacts/point_maze_gap4_calibration.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oat_drgrpo.maze_modebench import (  # noqa: E402
    MAZE_ACTION_VERSION,
    POINT_ACTIONS,
    POINT_MAZE_VERIFIER,
)
from oat_drgrpo.maze_modebench_process import MazeVerifierProcess  # noqa: E402

WORKER = ROOT / "var/maze_runtime/venv/bin/python"
OUT = ROOT / "var/artifacts/point_maze_gap4_calibration.json"
ENV_SHA = json.loads(
    (ROOT / "var/data/point_maze_modebench_v1/identity.json").read_text()
)["runtime_identity"]["point_environment_sha256"]

BORDER = [1] * 9
GAP_ROW = [1, 0, 0, 0, 0, 0, 0, 0, 1]
WALL_ROW = [1, 0, 0, 0, 1, 0, 0, 0, 1]
# rows 1,3,5,7 open at col 4; rows 2,4,6 blocked -> exactly four passages.
MAZE = [BORDER, GAP_ROW, WALL_ROW, GAP_ROW, WALL_ROW, GAP_ROW, WALL_ROW, GAP_ROW, BORDER]
GAPS = [("gap_n2", 3.0), ("gap_n1", 1.0), ("gap_s1", -1.0), ("gap_s2", -3.0)]


def canon(value) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, ensure_ascii=True,
                   separators=(",", ":"), sort_keys=True).encode("ascii")
    ).hexdigest()


def make_spec(reset_col: int, goal_col: int) -> dict:
    spec = {
        "verifier": POINT_MAZE_VERIFIER,
        "maze_action_version": MAZE_ACTION_VERSION,
        "environment_id": "PointMaze_UMaze-v3",
        "environment_sha256": ENV_SHA,
        "controller_sha256": None,
        "map_id": f"gap4_c{reset_col}{goal_col}",
        "maze_map": MAZE,
        "reset_cell": [4, reset_col],
        "goal_cell": [4, goal_col],
        "reset_seed": 91_001,
        "min_actions": 2,
        "max_actions": 96,
        "action_repeat": 5,
        "action_tokens": list(POINT_ACTIONS),
        "success_threshold": 0.45,
        "max_segment_length": 0.2,
        "bounds_xy": [[-4.5, 4.5], [-4.5, 4.5]],
        "route_gates": [
            {"id": gid, "axis": "x", "coordinate": 0.0,
             "span": [y - 0.4, y + 0.4], "hysteresis": 0.1}
            for gid, y in GAPS
        ],
    }
    spec["spec_sha256"] = canon(spec)
    return spec


def calibrate(verifier, spec, gid, y, span_cells, budget):
    """Shortest certified program that the verifier attributes to `gid`.

    Seeded from the one hand-verified point (3-cell leg, 6-cell span certifies
    at 14/26/14) and scaled linearly, so the search is a short ordered shortlist
    rather than a grid sweep.
    """
    lead = "N" if y > 0 else "S"
    back = "S" if y > 0 else "N"
    cells = int(abs(y))
    leg0 = max(3, round(14 * cells / 3))
    mid0 = max(8, round(26 * span_cells / 6))
    tried = 0
    best = None
    for dmid in (0, 2, -2, 4, -4):
        mid = mid0 + dmid
        for dleg in (0, 1, -1, 2, -2, 3, -3):
            first = leg0 + dleg
            for last in (first, first + 1, first - 1):
                if first < 2 or last < 2 or first + mid + last > 96:
                    continue
                tried += 1
                if tried > budget:
                    return best, tried
                program = " ".join([lead] * first + ["E"] * mid + [back] * last)
                result = verifier.validate(program, spec)
                if result is None:
                    continue
                gates = tuple(result.directed_gates)
                if len(gates) == 1 and gates[0].startswith(gid):
                    best = {"gap": gid, "first": first, "mid": mid, "last": last,
                            "tokens": first + mid + last, "lead": lead, "back": back,
                            "directed_gates": list(gates),
                            "simulator_steps": result.simulator_steps,
                            "canonical_key": result.canonical_key,
                            "program": program}
                    return best, tried
    return best, tried


def main() -> int:
    verifier = MazeVerifierProcess(worker_python=WORKER)
    report = {"schema": "point-maze-gap4-calibration-v1", "maze_map": MAZE,
              "variants": []}
    try:
        for reset_col, goal_col in ((1, 7), (2, 6)):
            spec = make_spec(reset_col, goal_col)
            routes, calls = [], 0
            for gid, y in GAPS:
                best, tried = calibrate(verifier, spec, gid, y, goal_col - reset_col, budget=36)
                calls += tried
                routes.append(best)
                print(f"[{reset_col}->{goal_col}] {gid}: "
                      + (f"tokens={best['tokens']} ({best['first']}/{best['mid']}/"
                         f"{best['last']}) steps={best['simulator_steps']}"
                         if best else "NO CERTIFIED PROGRAM")
                      + f"   [{tried} probes]", flush=True)
            ok = [r for r in routes if r]
            keys = {r["canonical_key"] for r in ok}
            report["variants"].append({
                "reset_col": reset_col, "goal_col": goal_col,
                "span_cells": goal_col - reset_col,
                "certified_routes": len(ok), "distinct_keys": len(keys),
                "max_tokens": max((r["tokens"] for r in ok), default=None),
                "min_tokens": min((r["tokens"] for r in ok), default=None),
                "verifier_calls": calls,
                "routes": routes,
            })
            print(f"  -> {len(ok)}/4 certified, {len(keys)} distinct keys\n", flush=True)
    finally:
        verifier.close()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
