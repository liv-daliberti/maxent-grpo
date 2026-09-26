#!/usr/bin/env python3
"""Audit and aggregate the registered E72 B1a confirmation.

The confirmation compares semantic-MaxEnt-off B1a with a freshly trained,
seed-paired xGRPO arm on seeds 48--52.  It never mixes the exploratory seeds
43--47 into the estimate.  A domain is emitted only when all ten cells retain
the pass-12 evaluation and satisfy the frozen arm-integrity contract.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

SEEDS = (48, 49, 50, 51, 52)
DOMAINS = (
    ("graph_coloring", "Graph coloring"),
    ("countdown", "Countdown"),
    ("python_factors", "Python factors"),
    ("mathir", "MathIR action menu"),
    ("pantry_plan", "PantryPlan"),
)
ARMS = ("b1aconf", "xgrpoconf")
TERMINAL_STEP = 4608
METRIC_KEY = "eval/multi_answer/sampled_distinct_correct_at_8"
MARGIN = 0.15
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 72_014_852
INTERVAL_PERCENTILES = (2.5, 97.5)

SEMANTIC_MAXENT_OVERRIDES = {
    "OAT_ZERO_SEMANTIC_SHANNON_COEF": "0.0",
    "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE": "0",
    "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE": "0",
}
ACTIVE_SEMANTIC_MAXENT_KEYS = (
    "train/semantic_shannon_separate_advantage_active",
    "train/semantic_shannon_success_conditioned_signed_advantage_active",
)
SEMANTIC_MAXENT_METRIC = "train/semantic_shannon_separate_semantic_advantage_rms"
REPLAY_KEYS = (
    "train/canonical_replay_applied_score_gradient_l2",
    "train/canonical_replay_mass_score_gradient_l2",
)
COMPUTE_ONLY_KEYS = (
    "train/canonical_replay_compute_only",
    "train/canonical_replay_compute_only_configured",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def read_run(run_dir: Path, arm: str) -> dict[str, Any]:
    """Read a cell once, collecting its terminal value and integrity evidence."""

    terminal_values: list[float] = []
    seen_counts = {
        "metric_files": 0,
        "records": 0,
        "replay_gradient_records": 0,
        "compute_only_records": 0,
        "semantic_maxent_records": 0,
    }
    replay_max = 0.0
    semantic_maxent_max = 0.0
    compute_only_values: set[float] = set()
    semantic_maxent_active_values: dict[str, set[float]] = {
        key: set() for key in ACTIVE_SEMANTIC_MAXENT_KEYS
    }

    for path_text in sorted(
        glob.glob(str(run_dir / "debug_job*" / "train_metrics.jsonl"))
    ):
        seen_counts["metric_files"] += 1
        path = Path(path_text)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                seen_counts["records"] += 1
                if (
                    int(record.get("misc/global_step", -1)) == TERMINAL_STEP
                    and finite(record.get(METRIC_KEY))
                ):
                    terminal_values.append(float(record[METRIC_KEY]))

                for key in REPLAY_KEYS:
                    value = record.get(key)
                    if finite(value):
                        seen_counts["replay_gradient_records"] += 1
                        replay_max = max(replay_max, abs(float(value)))
                for key in COMPUTE_ONLY_KEYS:
                    value = record.get(key)
                    if finite(value):
                        seen_counts["compute_only_records"] += 1
                        compute_only_values.add(float(value))
                semantic_maxent = record.get(SEMANTIC_MAXENT_METRIC)
                if finite(semantic_maxent):
                    seen_counts["semantic_maxent_records"] += 1
                    semantic_maxent_max = max(
                        semantic_maxent_max, abs(float(semantic_maxent))
                    )
                for key in ACTIVE_SEMANTIC_MAXENT_KEYS:
                    value = record.get(key)
                    if finite(value):
                        semantic_maxent_active_values[key].add(float(value))

    violations: list[str] = []
    if not (run_dir / "TRAINING_COMPLETE.json").is_file():
        violations.append("missing TRAINING_COMPLETE.json")
    if not terminal_values:
        violations.append("missing pass-12 distinct@8")
        terminal_value = None
    else:
        rounded = {round(value, 12) for value in terminal_values}
        if len(rounded) != 1:
            violations.append("conflicting pass-12 distinct@8 values")
        terminal_value = terminal_values[-1]
    if seen_counts["replay_gradient_records"] == 0 or replay_max <= 0:
        violations.append("nonzero replay gradient not observed")
    if not compute_only_values or compute_only_values != {0.0}:
        violations.append(
            f"replay compute-only values are {sorted(compute_only_values)}"
        )

    if arm == "b1aconf":
        if semantic_maxent_max != 0.0:
            violations.append("semantic MaxEnt advantage was nonzero")
        if any(values - {0.0} for values in semantic_maxent_active_values.values()):
            violations.append("a disabled semantic MaxEnt switch became active")
    else:
        if semantic_maxent_max <= 0.0:
            violations.append("xGRPO semantic MaxEnt advantage never became active")
        for key, values in semantic_maxent_active_values.items():
            if values != {1.0}:
                violations.append(f"xGRPO semantic MaxEnt switch {key} values are {sorted(values)}")

    return {
        "run_dir": str(run_dir),
        "terminal": not violations,
        "distinct8": terminal_value,
        "integrity": {
            **seen_counts,
            "replay_gradient_l2_max": replay_max,
            "compute_only_values": sorted(compute_only_values),
            "semantic_maxent_advantage_rms_max": semantic_maxent_max,
            "semantic_maxent_active_values": {
                key: sorted(values) for key, values in semantic_maxent_active_values.items()
            },
            "violations": violations,
        },
    }


def load_cells(root: Path) -> tuple[dict[tuple[str, int, str], dict[str, Any]], dict]:
    cells: dict[tuple[str, int, str], dict[str, Any]] = {}
    ledgers: dict[str, Any] = {}
    expected = {(domain, seed) for domain, _ in DOMAINS for seed in SEEDS}
    for arm in ARMS:
        path = root / "var" / "artifacts" / f"e72_{arm}_ablation_jobs.json"
        ledger = json.loads(path.read_text(encoding="utf-8"))
        ledgers[arm] = ledger
        registered = {
            (str(run["domain"]), int(run["seed"])) for run in ledger.get("runs", [])
        }
        if registered != expected or len(ledger.get("runs", [])) != len(expected):
            raise ValueError(f"{arm}: launch manifest is not the frozen 5x5 grid")
        if arm == "b1aconf" and ledger.get("overrides") != SEMANTIC_MAXENT_OVERRIDES:
            raise ValueError("B1a effective semantic MaxEnt overrides differ from protocol")
        if arm == "xgrpoconf" and ledger.get("overrides"):
            raise ValueError("xGRPO confirmation unexpectedly has objective overrides")
        for run in ledger["runs"]:
            domain, seed = str(run["domain"]), int(run["seed"])
            cells[(domain, seed, arm)] = read_run(Path(run["save_path"]), arm)
    return cells, ledgers


def percentile_interval(deltas: list[float], rng: np.random.Generator) -> list[float]:
    values = np.asarray(deltas, dtype=float)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, INTERVAL_PERCENTILES)
    return [float(low), float(high)]


def verdict(interval: list[float]) -> str:
    low, high = interval
    if low > -MARGIN and high < MARGIN:
        return "E"
    if high < 0.0:
        return "W"
    if low > MARGIN:
        return "B"
    return "U"


def summarize(cells: dict[tuple[str, int, str], dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    domains: list[dict[str, Any]] = []
    for domain, title in DOMAINS:
        per_seed = []
        violations = []
        for seed in SEEDS:
            b1a = cells[(domain, seed, "b1aconf")]
            xgrpo = cells[(domain, seed, "xgrpoconf")]
            for arm, row in (("b1a", b1a), ("xgrpo", xgrpo)):
                if row["integrity"]["violations"]:
                    violations.append(
                        {"seed": seed, "arm": arm, "violations": row["integrity"]["violations"]}
                    )
            if b1a["distinct8"] is not None and xgrpo["distinct8"] is not None:
                per_seed.append(
                    {
                        "seed": seed,
                        "b1a": b1a["distinct8"],
                        "xgrpo": xgrpo["distinct8"],
                        "difference": b1a["distinct8"] - xgrpo["distinct8"],
                        "integrity": {"b1a": b1a["integrity"], "xgrpo": xgrpo["integrity"]},
                    }
                )
        reportable = len(per_seed) == len(SEEDS) and not violations
        if not reportable:
            domains.append(
                {
                    "domain": domain,
                    "title": title,
                    "reportable": False,
                    "violations": violations,
                    "complete_pairs": len(per_seed),
                }
            )
            continue
        deltas = [float(row["difference"]) for row in per_seed]
        interval = percentile_interval(deltas, rng)
        domains.append(
            {
                "domain": domain,
                "title": title,
                "reportable": True,
                "seeds": list(SEEDS),
                "means": {
                    "b1a": statistics.fmean(float(row["b1a"]) for row in per_seed),
                    "xgrpo": statistics.fmean(float(row["xgrpo"]) for row in per_seed),
                },
                "paired_difference": {
                    "mean": statistics.fmean(deltas),
                    "interval": interval,
                    "per_seed": deltas,
                    "wins": sum(delta > 0 for delta in deltas),
                },
                "verdict": verdict(interval),
                "per_seed": per_seed,
                "violations": [],
            }
        )
    counts = {
        code: sum(row.get("verdict") == code for row in domains)
        for code in ("E", "W", "B", "U")
    }
    all_reportable = all(row["reportable"] for row in domains)
    if not all_reportable:
        interpretation = "WITHHELD"
    elif counts["E"] >= 4:
        interpretation = "C1"
    elif counts["W"] >= 2:
        interpretation = "C2"
    elif counts["E"] and counts["W"]:
        interpretation = "C3"
    else:
        interpretation = "PER_DOMAIN_ONLY"
    return {
        "schema": "e72_b1a_confirmation_summary_v1",
        "terminal_step": TERMINAL_STEP,
        "metric": "distinct@8",
        "seeds": list(SEEDS),
        "bootstrap": {
            "method": "paired seed-level percentile",
            "resamples": BOOTSTRAP_RESAMPLES,
            "rng": "numpy.PCG64",
            "seed": BOOTSTRAP_SEED,
            "interval_percentiles": list(INTERVAL_PERCENTILES),
            "equivalence_margin": MARGIN,
        },
        "all_50_cells_terminal_and_valid": all_reportable,
        "verdict_counts": counts,
        "registered_interpretation": interpretation,
        "domains": domains,
    }


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# E72 B1a confirmation — terminal registered analysis",
        "",
        "Seeds 48--52 only; exploratory seeds 43--47 are not pooled.",
        "",
        "| domain | no semantic MaxEnt | xGRPO | paired difference | 95% paired bootstrap | verdict |",
        "| --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in summary["domains"]:
        if not row["reportable"]:
            lines.append(f"| {row['title']} | _withheld_ | | | | |")
            continue
        means = row["means"]
        paired = row["paired_difference"]
        low, high = paired["interval"]
        lines.append(
            f"| {row['title']} | {means['b1a']:.3f} | {means['xgrpo']:.3f} | "
            f"{paired['mean']:+.3f} | [{low:+.3f}, {high:+.3f}] | {row['verdict']} |"
        )
    lines += [
        "",
        f"Registered interpretation: **{summary['registered_interpretation']}**; "
        f"verdict counts {summary['verdict_counts']}.",
        "",
        "E = interval wholly within (-0.15, +0.15); W = interval excludes zero below; "
        "B = interval wholly above +0.15; otherwise U.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "var" / "artifacts" / "e72_b1a_confirmation_summary.json",
    )
    args = parser.parse_args()
    cells, _ = load_cells(root)
    summary = summarize(cells)
    atomic_json(summary, args.output)
    report = root / "paper" / "results" / "e72_b1a_confirmation_terminal.md"
    write_report(summary, report)
    print(
        f"[e72-b1a-confirmation] {summary['registered_interpretation']} "
        f"{summary['verdict_counts']}"
    )
    print(f"[e72-b1a-confirmation] wrote {args.output} and {report}")
    return 0 if summary["all_50_cells_terminal_and_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
