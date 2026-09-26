#!/usr/bin/env python3
"""Materialize PointMaze v2: four balanced routes, all rotations, held-out maps.

v1 had four defects that together made the domain unable to measure anything:

  1. every map hardcoded exactly two routes, so `distinct@8` could not exceed 2;
  2. three of its four families shared identical certified programs;
  3. the evaluation rotation (r3) never appeared in training, so every eval map
     demanded an unseen rotation -- the reason nothing was learned;
  4. eight training maps and 96 optimizer updates.

v2 uses a channelled geometry: a free vertical corridor, stub walls forming one
bounded channel per gap, and a main wall whose gaps give four homotopy classes.
The channels catch overshoot, which brings per-mode discoverability from a 16.2x
imbalance down to 2.1x (measured against the simulator). Geometries -- not
rotations -- are held out, and every split carries all four rotations.

Token counts are not analytic (the point accelerates), so each distinct leg
geometry is calibrated against the real simulator and cached by leg signature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oat_drgrpo.maze_modebench import (  # noqa: E402
    MAZE_ACTION_VERSION,
    POINT_ACTIONS,
    POINT_MAZE_VERIFIER,
)
from oat_drgrpo.maze_modebench_process import MazeVerifierProcess  # noqa: E402

DEFAULT_WORKER = ROOT / "var/maze_runtime/venv/bin/python"
DEFAULT_OUTPUT = ROOT / "var/data/point_maze_modebench_v2"
SIZES = tuple(range(7, 16))    # 7x7 .. 15x15; below 7 there is no
                               # interior room for corridor+wall+goal
GAP_COUNTS = tuple(range(3, 7))  # floor of 3: a 2-gap map caps distinct@8 at 2
CORRIDOR_COL = 1               # free vertical corridor
ROTATE_CW = {"N": "E", "E": "S", "S": "W", "W": "N",
             "NE": "SE", "SE": "SW", "SW": "NW", "NW": "NE", "COAST": "COAST"}
ROTATIONS = (0, 1, 2, 3)       # every split carries all four


def canon(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, ensure_ascii=True,
                   separators=(",", ":"), sort_keys=True).encode("ascii")
    ).hexdigest()


def runtime_identity(worker: Path) -> dict:
    source = ("import json,sys;sys.path.insert(0,'src');"
              "from oat_drgrpo.maze_runtime_identity import maze_runtime_identity;"
              "print(json.dumps(maze_runtime_identity(),sort_keys=True))")
    done = subprocess.run([str(worker), "-c", source], cwd=ROOT, check=True,
                          capture_output=True, text=True)
    return json.loads(done.stdout.splitlines()[-1])


def build_map(size: int, wall_col: int,
              channel_rows: tuple[int, ...]) -> list[list[int]]:
    """Corridor at CORRIDOR_COL, stub walls everywhere except the channel rows."""
    grid = [[0] * size for _ in range(size)]
    for i in range(size):
        grid[0][i] = grid[size - 1][i] = 1
        grid[i][0] = grid[i][size - 1] = 1
    for row in range(1, size - 1):
        if row in channel_rows:
            continue
        for col in range(CORRIDOR_COL + 1, wall_col + 1):
            grid[row][col] = 1
    return grid


def geometries() -> list[dict]:
    """Sweep grid size and gap count for maximum distinct-program coverage.

    The certified program depends only on (distance profile, span, rotation), so
    variety comes from those -- not from wall placement, which changes the map
    picture but not the action sequence. Sweeping size and gap count raises the
    distinct-program pool from 27 keys to >2000 and lets the per-prompt support
    size vary from 2 to 6 instead of being pinned at one value.
    """
    import itertools
    out = []
    for size in SIZES:
        interior = list(range(1, size - 1))
        walls = list(range(3, size - 2))
        if not walls:
            continue
        for gaps in GAP_COUNTS:
            if len(interior) < 2 * gaps - 1:
                continue
            for channels in itertools.combinations(interior, gaps):
                if any(b - a < 2 for a, b in zip(channels, channels[1:])):
                    continue
                for wall_col in walls:
                    # Measured: a goal abutting the east border certifies at
                    # ~53%, an interior goal at ~6% -- the border wall stops the
                    # point, an interior goal must be decelerated onto. Spending
                    # candidates on interior goals is nearly pure waste, and the
                    # span axis they provided is recovered from grid size.
                    for goal_col in (size - 2,):
                        if goal_col < wall_col + 2:
                            continue
                        for start_row in interior:
                            if start_row in channels:
                                continue
                            spread = [abs(start_row - c) for c in channels]
                            if max(spread) > 5:
                                continue
                            span = goal_col - CORRIDOR_COL
                            longest = max(2 * (round(3.2 * v) + 1)
                                          + round(3.4 * span) for v in spread)
                            if longest > 96:
                                continue
                            out.append({
                                "size": size,
                                "gaps": gaps,
                                "max_spread": max(spread),
                                "family": f"s{size}g{gaps}"
                                          f"_c{''.join(f'{c:02d}' for c in channels)}"
                                          f"_w{wall_col}_g{goal_col}_r{start_row}",
                                "channel_rows": channels,
                                "wall_col": wall_col,
                                "goal_col": goal_col,
                                "start_row": start_row,
                                "program_key": (tuple(sorted(spread)), span),
                            })
    # Interleave distinct program keys so a truncated pool spans as many
    # distinct required action sequences as possible; within a key prefer the
    # better-balanced (tighter-spread) geometry.
    # Prefer more gaps: a 2-gap map caps distinct@8 at 2, which is the ceiling
    # that made v1 unmeasurable. Ordering keys by descending gap count keeps the
    # per-prompt support ceiling high while still spanning distinct programs.
    buckets: dict = {}
    for g in sorted(out, key=lambda g: (-g["gaps"], g["max_spread"], g["family"])):
        buckets.setdefault(g["program_key"], []).append(g)
    ordered = []
    keys = sorted(buckets, key=lambda k: (-buckets[k][0]["gaps"], k))
    while any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k]:
                ordered.append(buckets[k].pop(0))
    return ordered


def gates_for(geo: dict) -> list[dict]:
    centre = geo["size"] // 2
    gates = []
    for row in geo["channel_rows"]:
        y = centre - row
        gates.append({"id": f"gap_r{row}", "axis": "x",
                      "coordinate": float(geo["wall_col"] - centre),
                      "span": [y - 0.4, y + 0.4], "hysteresis": 0.1})
    return gates


def rotate_map(grid): return [list(r) for r in zip(*grid[::-1])]
def rotate_cell(cell, size): return (cell[1], size - 1 - cell[0])
def rotate_point(p): return (p[1], -p[0])


def rotate_gate(gate: dict) -> dict:
    axis, coord = gate["axis"], float(gate["coordinate"])
    lo, hi = (float(v) for v in gate["span"])
    ends = ((coord, lo), (coord, hi)) if axis == "x" else ((lo, coord), (hi, coord))
    a, b = (rotate_point(p) for p in ends)
    if abs(a[0] - b[0]) < 1e-12:
        return {"id": gate["id"], "axis": "x", "coordinate": a[0],
                "span": sorted((a[1], b[1])), "hysteresis": gate["hysteresis"]}
    return {"id": gate["id"], "axis": "y", "coordinate": a[1],
            "span": sorted((a[0], b[0])), "hysteresis": gate["hysteresis"]}


def make_spec(geo, grid, reset, goal, gates, rotation, split, env_sha, seed) -> dict:
    spec = {
        "verifier": POINT_MAZE_VERIFIER,
        "maze_action_version": MAZE_ACTION_VERSION,
        "environment_id": "PointMaze_UMaze-v3",
        "environment_sha256": env_sha,
        "controller_sha256": None,
        "map_id": f"{geo['family']}_{split}_r{rotation}",
        "maze_map": grid,
        "reset_cell": list(reset),
        "goal_cell": list(goal),
        "reset_seed": seed,
        "min_actions": 2,
        "max_actions": 96,
        "action_repeat": 5,
        "action_tokens": list(POINT_ACTIONS),
        "success_threshold": 0.45,
        "max_segment_length": 0.2,
        "bounds_xy": [[-geo["size"] / 2, geo["size"] / 2],
                      [-geo["size"] / 2, geo["size"] / 2]],
        "route_gates": gates,
    }
    spec["spec_sha256"] = canon(spec)
    return spec


def calibrate(verifier, spec, gate_id, vert_cells, lead, span_cells, cache, budget=220):
    """Certified (first, mid, last) for one channel, cached by leg signature."""
    key = (vert_cells, span_cells, lead)
    if key in cache:
        return cache[key]
    leg0 = max(2, round(3.2 * vert_cells) + 1)
    back = {"N": "S", "S": "N"}[lead]
    tried = 0
    # The horizontal leg is NOT a fixed cost per cell: a goal that abuts the far
    # border is caught by the wall and tolerates a wide band of counts, while an
    # interior goal must be stopped at and admits only a couple. A point estimate
    # therefore misses every short-span geometry, so scan the whole plausible
    # band (~2.8-5.5 tokens/cell) shortest-first.
    mids = list(range(max(6, round(2.9 * span_cells)),
                      max(9, round(4.6 * span_cells)) + 1))
    # Order matters far more than range here. The vertical estimate 3.2v+1 is
    # simulator-validated, so sweep `mid` for the best leg guess FIRST and only
    # then perturb the legs; the old mid-outermost cross-product paid ~28 probes
    # per mid before advancing and never finished.
    for dleg, dlast in ((0, 0), (0, 1), (1, 0), (-1, 0), (1, 1), (-1, -1),
                        (2, 0), (-2, 0), (2, 1), (-2, -1), (3, 0), (-3, 0)):
        first = leg0 + dleg
        last = first + dlast
        if first < 2 or last < 2:
            continue
        for mid in mids:
            if first + mid + last > 96:
                continue
            tried += 1
            if tried > budget:
                cache[key] = None
                return None
            program = " ".join([lead] * first + ["E"] * mid + [back] * last)
            res = verifier.validate(program, spec)
            if res is None:
                continue
            gates = tuple(res.directed_gates)
            if len(gates) == 1 and gates[0].startswith(gate_id):
                cache[key] = (first, mid, last)
                return cache[key]
    cache[key] = None
    return None


def prompt_text(spec: dict) -> str:
    tokens = ", ".join(spec["action_tokens"])
    return (
        "Drive the point robot from its reset cell to the goal cell in this maze.\n"
        f"Maze grid (1 = wall, 0 = free), row major:\n{spec['maze_map']}\n"
        f"Reset cell (row, col): {spec['reset_cell']}\n"
        f"Goal cell (row, col): {spec['goal_cell']}\n"
        f"Emit between {spec['min_actions']} and {spec['max_actions']} action "
        f"tokens separated by spaces, chosen from: {tokens}.\n"
        f"Each token is held for {spec['action_repeat']} simulator steps.\n"
        "There are several distinct passages through the wall; any one that "
        "reaches the goal is correct."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker-python", type=Path, default=DEFAULT_WORKER)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--train-geometries", type=int, default=213)
    ap.add_argument("--dev-geometries", type=int, default=36)
    ap.add_argument("--eval-geometries", type=int, default=71)
    ap.add_argument("--pilot", type=int, default=0,
                    help="if >0, use only this many geometries total (smoke test)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = args.output_root.resolve()
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite")
        if "point_maze_modebench_v2" not in out.name:
            raise SystemExit("refusing overwrite outside point_maze_modebench_v2")
        shutil.rmtree(out)

    runtime = runtime_identity(args.worker_python)
    env_sha = runtime["point_environment_sha256"]
    geos = geometries()
    need = (args.pilot if args.pilot else
            args.train_geometries + args.dev_geometries + args.eval_geometries)
    candidates = geos[:need]
    print(f"attempting {len(candidates)} geometries "
          f"({len({g['program_key'] for g in candidates})} distinct program keys)",
          flush=True)

    verifier = MazeVerifierProcess(worker_python=args.worker_python)
    cache: dict = {}
    rows: dict[str, list] = {"train": [], "dev": [], "eval": []}
    certification: list[dict] = []
    dropped: list[str] = []
    try:
        # Phase 1 -- calibrate every candidate. A geometry survives only if all
        # of its channels certify; one failure drops it.
        survivors = []
        for gi, geo in enumerate(candidates):
            grid = build_map(geo["size"], geo["wall_col"], geo["channel_rows"])
            reset = (geo["start_row"], CORRIDOR_COL)
            goal = (geo["start_row"], geo["goal_col"])
            gates = gates_for(geo)
            base_spec = make_spec(geo, grid, reset, goal, gates, 0, "probe",
                                  env_sha, 92_000 + gi)
            legs = []
            for row, gate in zip(geo["channel_rows"], gates):
                lead = "N" if row < geo["start_row"] else "S"
                counts = calibrate(verifier, base_spec, gate["id"],
                                   abs(row - geo["start_row"]), lead,
                                   geo["goal_col"] - CORRIDOR_COL, cache)
                legs.append((gate["id"], lead, counts))
            if any(c is None for _g, _l, c in legs):
                dropped.append(geo["family"])
                print(f"[calib {gi+1}/{len(candidates)}] {geo['family']}: DROPPED",
                      flush=True)
                continue
            survivors.append((geo, grid, reset, goal, gates, legs))
            print(f"[calib {gi+1}/{len(candidates)}] {geo['family']}: ok "
                  f"({geo['gaps']} routes)", flush=True)

        # Phase 2 -- size the splits to the realised yield, keeping the 6:1:2
        # ratio and the difficulty interleave.
        pattern = ["train"] * 6 + ["dev"] + ["eval"] * 2
        assigned = [(pattern[i % len(pattern)], *rec)
                    for i, rec in enumerate(survivors)]
        print(f"\ncalibrated {len(survivors)}/{len(candidates)} geometries "
              f"({100*len(survivors)/max(1,len(candidates)):.0f}% yield)", flush=True)

        # Phase 3 -- certify every rotation of every survivor.
        for split, geo, grid, reset, goal, gates, legs in assigned:
            for rotation in ROTATIONS:
                g2, r2, go2, gt2 = grid, reset, goal, gates
                progs = [[l] * c[0] + ["E"] * c[1]
                         + [{"N": "S", "S": "N"}[l]] * c[2] for _i, l, c in legs]
                for _ in range(rotation):
                    g2 = rotate_map(g2)
                    r2 = rotate_cell(r2, geo["size"])
                    go2 = rotate_cell(go2, geo["size"])
                    gt2 = [rotate_gate(x) for x in gt2]
                    progs = [[ROTATE_CW[t] for t in p] for p in progs]
                spec = make_spec(geo, g2, r2, go2, gt2, rotation, split,
                                 env_sha, 92_000 + rotation)
                records = []
                for prog in progs:
                    cand = " ".join(prog)
                    res = verifier.validate(cand, spec)
                    if res is None:
                        records = []
                        break
                    records.append({
                        "program": cand,
                        "program_sha256": hashlib.sha256(cand.encode()).hexdigest(),
                        "canonical_key": res.canonical_key,
                        "directed_gates": list(res.directed_gates),
                        "simulator_steps": res.simulator_steps,
                    })
                if len({r["canonical_key"] for r in records}) != len(progs):
                    dropped.append(spec["map_id"])
                    continue
                certification.append({
                    "map_id": spec["map_id"], "split": split,
                    "family": geo["family"], "rotation": rotation,
                    "size": geo["size"], "gaps": geo["gaps"],
                    "spec_sha256": spec["spec_sha256"], "routes": records,
                })
                rows[split].append({
                    "problem": prompt_text(spec),
                    "answer": json.dumps(spec, sort_keys=True,
                                         separators=(",", ":")),
                    "modebench_task": POINT_MAZE_VERIFIER,
                    "answer_mode_family": geo["family"],
                    "answer_mode_split": split,
                    "answer_mode_count": len(records),
                    "certified_simple_route_count": len(records),
                    "instance_fingerprint": spec["spec_sha256"],
                })
    finally:
        verifier.close()

    out.mkdir(parents=True, exist_ok=True)
    for split, key in (("train", "train"), ("dev", "multi_answer"), ("eval", "multi_answer")):
        if not rows[split]:
            print(f"WARNING: {split} split is empty; not written")
            continue
        DatasetDict({key: Dataset.from_list(rows[split])}).save_to_disk(str(out / split))

    counts = [len(c["routes"]) for c in certification]
    identity = {
        "schema_version": "point-maze-modebench-data-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "grid_sizes": sorted({c["size"] for c in certification}) if certification else [],
        "gap_counts": sorted({len(c["routes"]) for c in certification}) if certification else [],
        "rotations_per_split": list(ROTATIONS),
        "holdout_unit": "geometry",
        "split_rows": {k: len(v) for k, v in rows.items()},
        "distinct_families": {k: len({r["answer_mode_family"] for r in v})
                              for k, v in rows.items()},
        "routes_per_map": {"min": min(counts, default=0), "max": max(counts, default=0),
                           "mean": (sum(counts) / len(counts)) if counts else 0},
        "distinct_program_sets": len({
            tuple(sorted(r["program_sha256"] for r in c["routes"]))
            for c in certification}),
        "map_count": len(certification),
        "dropped": dropped,
        "runtime_identity": runtime,
        "certification": certification,
    }
    (out / "identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    print(f"\nrows: {identity['split_rows']}  families: {identity['distinct_families']}")
    print(f"routes/map: {identity['routes_per_map']}  "
          f"distinct program sets: {identity['distinct_program_sets']}/{len(certification)}")
    print(f"dropped: {len(dropped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
