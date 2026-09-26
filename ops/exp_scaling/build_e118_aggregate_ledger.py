#!/usr/bin/env python3
"""Build the monitor-only aggregate of all E118 MaxRL extension ledgers."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    ("qwen05b", "e118r2_maxrl_verified_replay_factorial_jobs.json"),
    ("qwen05b", "e118q5_maxrl_verified_replay_extension_jobs.json"),
    ("falcon1b", "e118f1_maxrl_verified_replay_extension_jobs.json"),
    ("qwen3b", "e118q3_maxrl_verified_replay_extension_jobs.json"),
)
OUTPUT = ROOT / "var/artifacts/e118_all_scales_maxrl_verified_replay_jobs.json"


def main() -> int:
    runs = []
    sources = []
    for scale, name in SOURCES:
        path = ROOT / "var/artifacts" / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("released") is not True or payload.get("target_steps") != 3072:
            raise SystemExit(f"invalid E118 source ledger: {path}")
        source_runs = payload.get("runs", [])
        for run in source_runs:
            runs.append(dict(run, scale=scale))
        sources.append({"scale": scale, "ledger": str(path), "cells": len(source_runs)})
    if len(runs) != 150 or len({int(run["job_id"]) for run in runs}) != 150:
        raise SystemExit("E118 aggregate must contain exactly 150 unique cells")
    payload = {
        "schema": "e118_all_scales_maxrl_verified_replay_jobs_v1",
        "released": True,
        "train_rows": 384,
        "passes": 8,
        "target_steps": 3072,
        "checkpoint_interval_steps": 192,
        "arms": ["maxrl", "replay_maxrl"],
        "domains": ["graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan"],
        "sources": sources,
        "runs": runs,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT} with {len(runs)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
