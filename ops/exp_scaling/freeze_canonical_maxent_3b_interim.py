#!/usr/bin/env python3
"""Freeze the latest paired 3B common-horizon result used by the paper.

The live curve artifacts continue to advance.  This command reads one coherent
byte snapshot of each input, records its SHA-256 digest, and computes the latest
training-pass checkpoint shared by every method and seeds 43--45 within each
domain.  Later unequal-horizon observations are deliberately excluded.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "paper/results/canonical_maxent_vs_reward_only_3b_interim.json"
SEEDS = (43, 44, 45)
METRICS = ("pass8", "mean8", "coverage8", "distinct8", "greedy")
ARM_MAPPING = {
    "reward_only": "grpo",
    "fixed": "maxent",
    "proportional": "maxent_control",
    "dual": "maxent_dual",
}
SOURCE_PATHS = {
    "graph_coloring": {
        "graph_coloring_maxent": (
            ROOT / "var/artifacts/gce17_canonical_maxent_3b_v5_scaling_curve.json"
        ),
        "graph_coloring_reward_only": (
            ROOT / "var/artifacts/gce18_canonical_drgrpo_3b_v1_scaling_curve.json"
        ),
    },
    "countdown": {
        "countdown_maxent": (
            ROOT / "var/artifacts/cde17_canonical_maxent_3b_v5_scaling_curve.json"
        ),
        "countdown_reward_only": (
            ROOT / "var/artifacts/cde18_canonical_drgrpo_3b_v1_scaling_curve.json"
        ),
    },
}


def _read_json_bytes(path: Path, retries: int = 5) -> tuple[bytes, list[dict[str, Any]]]:
    """Read a live atomically-published JSON array, retrying transient races."""

    error: Exception | None = None
    for attempt in range(retries):
        try:
            raw = path.read_bytes()
            payload = json.loads(raw)
            if not isinstance(payload, list):
                raise ValueError(f"expected a JSON array, got {type(payload).__name__}")
            return raw, [row for row in payload if isinstance(row, dict)]
        except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError) as exc:
            error = exc
            if attempt + 1 < retries:
                time.sleep(0.15 * (attempt + 1))
    raise RuntimeError(f"could not read curve artifact {path}: {error}") from error


def _finite(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non-numeric {field}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite {field}: {value!r}")
    return result


def _metric_values(row: dict[str, Any]) -> dict[str, float]:
    return {metric: _finite(row.get(metric), field=metric) for metric in METRICS}


def _domain_result(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latest: dict[tuple[str, int, float], dict[str, Any]] = {}
    eligible_arms = frozenset(ARM_MAPPING.values())
    for row in rows:
        if row.get("split") != "multi_answer" or row.get("arm") not in eligible_arms:
            continue
        try:
            seed = int(row["seed"])
            training_passes = _finite(row.get("training_passes"), field="training_passes")
            step = int(row["step"])
        except (KeyError, TypeError, ValueError):
            continue
        if seed not in SEEDS:
            continue
        key = (str(row["arm"]), seed, training_passes)
        previous = latest.get(key)
        if previous is None or step >= int(previous["step"]):
            latest[key] = row

    maxima: list[float] = []
    for arm in ARM_MAPPING.values():
        for seed in SEEDS:
            passes = [
                training_passes
                for candidate_arm, candidate_seed, training_passes in latest
                if candidate_arm == arm and candidate_seed == seed
            ]
            if not passes:
                raise ValueError(f"missing 3B curve for arm={arm}, seed={seed}")
            maxima.append(max(passes))
    common_horizon = min(maxima)

    selected: dict[str, dict[int, dict[str, float]]] = {}
    for method, arm in ARM_MAPPING.items():
        selected[method] = {}
        for seed in SEEDS:
            key = (arm, seed, common_horizon)
            if key not in latest:
                raise ValueError(
                    f"missing common-horizon row for arm={arm}, seed={seed}, "
                    f"passes={common_horizon:g}"
                )
            selected[method][seed] = _metric_values(latest[key])

    baseline = selected["reward_only"]
    methods: dict[str, Any] = {}
    for method in ARM_MAPPING:
        by_seed = selected[method]
        method_result: dict[str, Any] = {
            "mean": {
                metric: mean(by_seed[seed][metric] for seed in SEEDS)
                for metric in METRICS
            },
            "by_seed": {str(seed): by_seed[seed] for seed in SEEDS},
        }
        if method != "reward_only":
            paired_by_seed = {
                str(seed): {
                    metric: by_seed[seed][metric] - baseline[seed][metric]
                    for metric in METRICS
                }
                for seed in SEEDS
            }
            method_result["paired_delta"] = {
                metric: mean(paired_by_seed[str(seed)][metric] for seed in SEEDS)
                for metric in METRICS
            }
            method_result["paired_delta_by_seed"] = paired_by_seed
            method_result["positive_seed_count"] = {
                metric: sum(
                    paired_by_seed[str(seed)][metric] > 0.0 for seed in SEEDS
                )
                for metric in METRICS
            }
        methods[method] = method_result

    return {"common_horizon_passes": common_horizon, "methods": methods}


def build_snapshot() -> dict[str, Any]:
    source_artifacts: dict[str, dict[str, str]] = {}
    domains: dict[str, Any] = {}
    for domain, named_paths in SOURCE_PATHS.items():
        rows: list[dict[str, Any]] = []
        for source_name, path in named_paths.items():
            raw, source_rows = _read_json_bytes(path)
            rows.extend(source_rows)
            source_artifacts[source_name] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        domains[domain] = _domain_result(rows)

    return {
        "status": "FROZEN_INTERIM_PAIRED_COMMON_HORIZON",
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": "Qwen2.5-3B-Instruct",
        "seeds": list(SEEDS),
        "analysis": (
            "Latest training-pass checkpoint shared by every method and all "
            "three paired seeds within each domain. Later unequal-horizon "
            "observations are excluded from numerical comparisons."
        ),
        "source_arm_mapping": ARM_MAPPING,
        "source_artifacts": source_artifacts,
        "domains": domains,
    }


def write_snapshot(payload: dict[str, Any], output: Path = OUTPUT) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    payload = build_snapshot()
    write_snapshot(payload)
    horizons = payload["domains"]
    print(
        f"wrote {OUTPUT}: graph={horizons['graph_coloring']['common_horizon_passes']:g} "
        f"passes, countdown={horizons['countdown']['common_horizon_passes']:g} passes"
    )


if __name__ == "__main__":
    main()
