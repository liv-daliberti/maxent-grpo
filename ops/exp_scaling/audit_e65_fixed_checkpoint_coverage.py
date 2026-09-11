#!/usr/bin/env python3
"""Audit the complete seed-level paper checkpoint surface for all five domains."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "var/artifacts/e65_five_domain_confirmation_results_latest.json"
OUT = ROOT / "var/artifacts/e65_fixed_checkpoint_coverage_audit_latest.json"
SEEDS = ("43", "44", "45")
MODEBENCH_PASSES = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12)
MATH_PASSES = (0, 2, 4, 6, 8, 10, 12)
MODEBENCH_METRICS = ("greedy", "mean8", "pass8", "distinct8")
MATH_METRICS = ("greedy", "mean8", "pass8")
MODEBENCH_DOMAINS = (
    "Graph coloring",
    "Countdown",
    "Python factors",
    "MathIR action menu",
)
MATH_DOMAIN = "Held-out MATH-500 transfer"
COHORTS = {
    "e61": {
        "audit": ROOT
        / "var/artifacts/e61r1_e58_vs_grpo_12pass_audit_latest.json",
        "terminal_runs": 24,
        "arms": ("grpo", "verified_first_global_replay_canonical"),
        "domains": MODEBENCH_DOMAINS,
        "passes": MODEBENCH_PASSES,
        "metrics": MODEBENCH_METRICS,
    },
    "e66": {
        "audit": ROOT
        / "var/artifacts/e66_same_plumbing_actuator_ablation_audit_latest.json",
        "terminal_runs": 12,
        "arms": ("same_plumbing_actuator_off_control",),
        "domains": MODEBENCH_DOMAINS,
        "passes": MODEBENCH_PASSES,
        "metrics": MODEBENCH_METRICS,
    },
    "e68": {
        "audit": ROOT
        / "var/artifacts/"
        "e68_separated_support_actuator_ablation_audit_latest.json",
        "terminal_runs": 12,
        "arms": ("verified_entropy_gated_singleton_escape_canonical",),
        "domains": MODEBENCH_DOMAINS,
        "passes": MODEBENCH_PASSES,
        "metrics": MODEBENCH_METRICS,
    },
    "e64": {
        "audit": ROOT
        / "var/artifacts/e64_math500_realism_matched_audit_latest.json",
        "terminal_runs": 6,
        "arms": ("grpo", "verified_first_global_replay_canonical"),
        "domains": (MATH_DOMAIN,),
        "passes": MATH_PASSES,
        "metrics": MATH_METRICS,
    },
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _points(
    results: dict[str, Any],
    *,
    domain: str,
    arm: str,
    metric: str,
) -> dict[int, list[dict[str, Any]]]:
    records = (
        results.get("domains", {})
        .get(domain, {})
        .get("arms", {})
        .get(arm, {})
        .get("metrics", {})
        .get(metric, {})
        .get("points", [])
    )
    indexed: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        training_pass = record.get("pass")
        if not _finite(training_pass):
            continue
        indexed.setdefault(int(training_pass), []).append(record)
    return indexed


def _complete_point(record: dict[str, Any]) -> bool:
    seeds = record.get("seeds")
    return isinstance(seeds, dict) and all(
        _finite(seeds.get(seed)) for seed in SEEDS
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    results = _load(RESULTS)
    violations: list[str] = []
    tracks: dict[str, Any] = {}
    expected_checkpoint_cells = 0
    landed_checkpoint_cells = 0
    expected_metric_seed_values = 0
    landed_metric_seed_values = 0
    all_cohorts_terminal = True

    domain_requirements: dict[str, dict[str, Any]] = {
        domain: {
            "passes": MODEBENCH_PASSES,
            "requirements": [],
        }
        for domain in MODEBENCH_DOMAINS
    }
    domain_requirements[MATH_DOMAIN] = {
        "passes": MATH_PASSES,
        "requirements": [],
    }

    for cohort, config in COHORTS.items():
        audit = _load(config["audit"])
        terminal_runs = int(audit.get("summary", {}).get("terminal_runs", 0))
        cohort_terminal = terminal_runs == int(config["terminal_runs"])
        all_cohorts_terminal = all_cohorts_terminal and cohort_terminal
        cohort_expected = 0
        cohort_landed = 0
        cohort_pending: list[dict[str, Any]] = []

        for domain in config["domains"]:
            for arm in config["arms"]:
                domain_requirements[domain]["requirements"].append(
                    {
                        "cohort": cohort,
                        "arm": arm,
                        "metrics": config["metrics"],
                    }
                )
                metric_indexes = {
                    metric: _points(
                        results,
                        domain=domain,
                        arm=arm,
                        metric=metric,
                    )
                    for metric in config["metrics"]
                }
                for training_pass in config["passes"]:
                    cohort_expected += 1
                    expected_checkpoint_cells += 1
                    checkpoint_complete = True
                    missing: list[str] = []
                    for metric, indexed in metric_indexes.items():
                        records = indexed.get(training_pass, [])
                        expected_metric_seed_values += len(SEEDS)
                        if len(records) > 1:
                            violations.append(
                                f"{cohort}/{domain}/{arm}/{metric}/"
                                f"pass{training_pass}: duplicate checkpoint rows"
                            )
                        if len(records) != 1:
                            checkpoint_complete = False
                            missing.extend(f"{metric}/s{seed}" for seed in SEEDS)
                            continue
                        seeds = records[0].get("seeds", {})
                        for seed in SEEDS:
                            if _finite(seeds.get(seed)):
                                landed_metric_seed_values += 1
                            else:
                                checkpoint_complete = False
                                missing.append(f"{metric}/s{seed}")
                    if checkpoint_complete:
                        cohort_landed += 1
                        landed_checkpoint_cells += 1
                    else:
                        pending = {
                            "domain": domain,
                            "arm": arm,
                            "pass": training_pass,
                            "missing": missing,
                        }
                        cohort_pending.append(pending)
                        if cohort_terminal:
                            violations.append(
                                f"{cohort}/{domain}/{arm}/pass{training_pass}: "
                                f"terminal cohort is missing {', '.join(missing)}"
                            )

        tracks[cohort] = {
            "terminal_runs": terminal_runs,
            "expected_terminal_runs": int(config["terminal_runs"]),
            "cohort_terminal": cohort_terminal,
            "expected_checkpoint_cells": cohort_expected,
            "landed_checkpoint_cells": cohort_landed,
            "pending_checkpoint_cells": cohort_pending,
        }

    domains: dict[str, Any] = {}
    for domain, config in domain_requirements.items():
        complete_passes: list[int] = []
        for training_pass in config["passes"]:
            complete = True
            for requirement in config["requirements"]:
                for metric in requirement["metrics"]:
                    records = _points(
                        results,
                        domain=domain,
                        arm=requirement["arm"],
                        metric=metric,
                    ).get(training_pass, [])
                    if len(records) != 1 or not _complete_point(records[0]):
                        complete = False
            if complete:
                complete_passes.append(training_pass)
        domains[domain] = {
            "maximum_registered_checkpoints": len(config["passes"]),
            "registered_passes": list(config["passes"]),
            "complete_full_surface_passes": complete_passes,
            "complete_full_surface_checkpoint_count": len(complete_passes),
            "required_arms": [
                requirement["arm"] for requirement in config["requirements"]
            ],
        }

    if any(
        domain["maximum_registered_checkpoints"] > 10
        for domain in domains.values()
    ):
        violations.append("a domain registers more than 10 paper checkpoints")
    complete = (
        landed_checkpoint_cells == expected_checkpoint_cells
        and landed_metric_seed_values == expected_metric_seed_values
    )
    status = (
        "fail"
        if violations
        else "pass"
        if all_cohorts_terminal and complete
        else "in_progress"
    )
    payload = {
        "schema": "e65_fixed_checkpoint_coverage_audit_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_results": str(RESULTS.relative_to(ROOT)),
        "source_results_generated_at": results.get("generated_at"),
        "status": status,
        "policy": {
            "seeds": [int(seed) for seed in SEEDS],
            "modebench_passes": list(MODEBENCH_PASSES),
            "math500_passes": list(MATH_PASSES),
            "maximum_checkpoints_per_domain": 10,
            "complete_checkpoint_requires_every_registered_metric_and_seed": True,
        },
        "summary": {
            "domains": len(domains),
            "expected_checkpoint_cells": expected_checkpoint_cells,
            "landed_checkpoint_cells": landed_checkpoint_cells,
            "expected_metric_seed_values": expected_metric_seed_values,
            "landed_metric_seed_values": landed_metric_seed_values,
            "all_cohorts_terminal": all_cohorts_terminal,
            "violation_count": len(violations),
        },
        "domains": domains,
        "tracks": tracks,
        "violations": sorted(set(violations)),
    }
    _atomic_json(OUT, payload)
    print(
        f"[fixed-checkpoint-coverage] status={status} "
        f"cells={landed_checkpoint_cells}/{expected_checkpoint_cells} "
        f"values={landed_metric_seed_values}/{expected_metric_seed_values} "
        f"violations={len(violations)}"
    )


if __name__ == "__main__":
    main()
