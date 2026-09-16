#!/usr/bin/env python3
"""Build the terminal Qwen-0.5B Dr/Replay x MaxRL/ReplayMaxRL record."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "paper/results/core_terminal_endpoints.json"
E118 = ROOT / "var/artifacts/e118_all_scales_maxrl_verified_replay_jobs.json"
OUTPUT = ROOT / "paper/results/e118_qwen05b_factorial.json"
DOMAINS = ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan")
SEEDS = (43, 44, 45, 46, 47)
TARGET = 3072
METHODS = ("drgrpo", "replay_drgrpo", "maxrl", "replay_maxrl")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def last_lines(path: Path, count: int = 32) -> list[str]:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        end = handle.tell()
        size = min(end, 32 * 1024 * 1024)
        handle.seek(end - size)
        lines = handle.read().decode("utf-8", "replace").splitlines()
    return lines[-count:]


def endpoint(run_dir: Path) -> tuple[dict[str, float], Path]:
    candidates = sorted(run_dir.glob("debug_job*/eval_mode_coverage_draws.jsonl"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one fixed-draw record under {run_dir}, found {len(candidates)}")
    path = candidates[0]
    draws: dict[int, dict[str, Any]] = {}
    for line in last_lines(path):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(row.get("step", -1)) == TARGET and row.get("draw_index") is not None:
            draws[int(row["draw_index"])] = row
    if set(draws) != {0, 1, 2, 3}:
        raise RuntimeError(f"incomplete terminal fixed draws: {path}: {sorted(draws)}")
    return {
        "pass8": statistics.fmean(float(row["metrics"]["any_correct_at_k"]) for row in draws.values()),
        "distinct8": statistics.fmean(float(row["metrics"]["distinct_correct_modes_at_k"]) for row in draws.values()),
    }, path


def summary(values: list[float]) -> dict[str, Any]:
    if len(values) != 5:
        raise RuntimeError("paper summary requires five paired seeds")
    mean = statistics.fmean(values)
    half = 2.7764451051977987 * statistics.stdev(values) / math.sqrt(5)
    return {"mean": mean, "student_t_95": [mean - half, mean + half], "per_seed": dict(zip(map(str, SEEDS), values))}


def main() -> int:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    ledger = json.loads(E118.read_text(encoding="utf-8"))
    runs = {(r["domain"], r["arm"], int(r["seed"])): r for r in ledger["runs"] if r.get("scale") == "qwen05b"}
    sources = {str(BASELINE): digest(BASELINE), str(E118): digest(E118)}
    domains: dict[str, Any] = {}
    for domain in DOMAINS:
        per_method: dict[str, dict[str, dict[str, float]]] = {method: {} for method in METHODS}
        base = baseline["models"]["Qwen2.5-0.5B"]["domains"][domain]["methods"]
        for seed in SEEDS:
            for method, arm in (("drgrpo", "control"), ("replay_drgrpo", "replay")):
                row = base[arm]["per_seed"][str(seed)]
                per_method[method][str(seed)] = {"pass8": float(row["pass8"]), "distinct8": float(row["distinct8"])}
            for method, arm in (("maxrl", "maxrl"), ("replay_maxrl", "replay_maxrl")):
                run = runs[(domain, arm, seed)]
                values, path = endpoint(Path(run["run_dir"]))
                per_method[method][str(seed)] = values
                sources[str(path)] = digest(path)
        method_summaries = {
            method: {metric: summary([per_method[method][str(seed)][metric] for seed in SEEDS]) for metric in ("pass8", "distinct8")}
            for method in METHODS
        }
        contrasts = {}
        for name, treatment, control in (
            ("replay_on_drgrpo", "replay_drgrpo", "drgrpo"),
            ("replay_on_maxrl", "replay_maxrl", "maxrl"),
        ):
            contrasts[name] = {
                metric: summary([per_method[treatment][str(seed)][metric] - per_method[control][str(seed)][metric] for seed in SEEDS])
                for metric in ("pass8", "distinct8")
            }
        contrasts["factorial_interaction"] = {
            metric: summary([
                (per_method["replay_maxrl"][str(seed)][metric] - per_method["maxrl"][str(seed)][metric])
                - (per_method["replay_drgrpo"][str(seed)][metric] - per_method["drgrpo"][str(seed)][metric])
                for seed in SEEDS
            ]) for metric in ("pass8", "distinct8")
        }
        domains[domain] = {"per_method": per_method, "summaries": method_summaries, "contrasts": contrasts}
    payload = {
        "schema": "e118-qwen05b-terminal-factorial-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "Qwen2.5-0.5B-Instruct", "target_step": TARGET,
        "evaluation": "four fixed-seed temperature-1 draws of K=8 per prompt",
        "domains": domains, "domain_order": list(DOMAINS), "seeds": list(SEEDS),
        "methods": list(METHODS), "sources_sha256": sources,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
