#!/usr/bin/env python3
"""Fail-closed terminal closure for the paired E66/E68 MathIR recovery."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "var" / "artifacts"
TERMINAL_STEP = 4608
SEEDS = (43, 44, 45)
EXPECTED_JOBS = {
    ("e66", 43): 30156845,
    ("e66", 44): 30156846,
    ("e66", 45): 30156847,
    ("e68", 43): 30156848,
    ("e68", 44): 30156849,
    ("e68", 45): 30156874,
}
INPUTS = {
    "recovery_manifest": ARTIFACTS
    / "e66_e68_mathir_seed_overflow_recovery.json",
    "scheduler_accounting": ARTIFACTS
    / "e66_e68_mathir_recovery_sacct_20260728.tsv",
    "e66_audit": ARTIFACTS
    / "e66_same_plumbing_actuator_ablation_audit_latest.json",
    "e68_audit": ARTIFACTS
    / "e68_separated_support_actuator_ablation_audit_latest.json",
    "e66_curve": ARTIFACTS
    / "mie66_same_plumbing_control_05b_12ep_scaling_curve.json",
    "e68_curve": ARTIFACTS
    / "mie68_separated_support_actuator_05b_12ep_scaling_curve.json",
    "figure_current": ROOT
    / "paper"
    / "figures"
    / "e61r1_e58_vs_grpo_05b_12ep_live.png",
    "figure_e68_provenance": ROOT
    / "paper"
    / "figures"
    / "e68_e58_vs_grpo_05b_12ep_live.png",
}
OUTPUT = ARTIFACTS / "e66_e68_mathir_terminal_recovery_summary.json"
METRICS = (
    "greedy",
    "mean8",
    "pass8",
    "distinct8",
    "online_canonical_tracked_outcomes",
    "online_canonical_mean_support_per_prompt",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _audit_runs(payload: dict[str, Any], cohort: str) -> list[dict[str, Any]]:
    if payload.get("violations") != []:
        raise ValueError(f"{cohort} audit has violations")
    domain = payload.get("domains", {}).get("mathir")
    if not isinstance(domain, dict) or not isinstance(domain.get("runs"), list):
        raise ValueError(f"{cohort} audit lacks MathIR runs")
    runs = domain["runs"]
    if len(runs) != len(SEEDS):
        raise ValueError(f"{cohort} audit does not contain exactly three runs")
    for run in runs:
        seed = int(run.get("seed", -1))
        if seed not in SEEDS:
            raise ValueError(f"{cohort} audit contains unexpected seed {seed}")
        if (
            int(run.get("recovery_job_id", -1))
            != EXPECTED_JOBS[(cohort, seed)]
            or not bool(run.get("terminal"))
            or int(run.get("latest_step", -1)) != TERMINAL_STEP
            or int(run.get("recovery_latest_step", -1)) != TERMINAL_STEP
            or float(run.get("training_passes", -1)) != 12.0
        ):
            raise ValueError(
                f"{cohort} seed {seed} is not an exact terminal recovery"
            )
    return sorted(runs, key=lambda row: int(row["seed"]))


def _terminal_rows(
    payload: list[dict[str, Any]], cohort: str
) -> list[dict[str, Any]]:
    per_seed_counts = {
        seed: sum(
            1
            for row in payload
            if int(row.get("seed", -1)) == seed
            and row.get("split") == "multi_answer"
            and row.get("pass8") is not None
        )
        for seed in SEEDS
    }
    if per_seed_counts != {seed: 49 for seed in SEEDS}:
        raise ValueError(
            f"{cohort} fixed evaluation cadence is not 49 rows per seed: "
            f"{per_seed_counts}"
        )
    terminal = [
        row
        for row in payload
        if int(row.get("seed", -1)) in SEEDS
        and int(row.get("step", -1)) == TERMINAL_STEP
        and row.get("split") == "multi_answer"
        and row.get("pass8") is not None
    ]
    if len(terminal) != len(SEEDS):
        raise ValueError(
            f"{cohort} lacks one primary terminal row for every seed"
        )
    if sorted(int(row["seed"]) for row in terminal) != list(SEEDS):
        raise ValueError(f"{cohort} terminal seed set is invalid")
    result: list[dict[str, Any]] = []
    for row in sorted(terminal, key=lambda value: int(value["seed"])):
        normalized = {
            "seed": int(row["seed"]),
            "step": int(row["step"]),
        }
        for metric in METRICS:
            normalized[metric] = _finite_number(
                row.get(metric),
                label=f"{cohort} seed {row['seed']} {metric}",
            )
        result.append(normalized)
    return result


def main() -> None:
    missing = [str(path) for path in INPUTS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing closure inputs: {missing}")

    recovery = _load_json(INPUTS["recovery_manifest"])
    manifest_jobs = {
        (str(row["cohort"]), int(row["seed"])): int(row["recovery_job_id"])
        for row in recovery.get("jobs", [])
    }
    if manifest_jobs != EXPECTED_JOBS:
        raise ValueError("recovery manifest job mapping drifted")
    if bool(recovery.get("scientific_settings_changed")):
        raise ValueError("recovery manifest reports changed scientific settings")

    with INPUTS["scheduler_accounting"].open(
        newline="", encoding="utf-8"
    ) as handle:
        accounting = list(csv.DictReader(handle, delimiter="\t"))
    accounting_jobs = {int(row["JobIDRaw"]): row for row in accounting}
    if set(accounting_jobs) != set(EXPECTED_JOBS.values()):
        raise ValueError("scheduler evidence does not contain the six exact jobs")
    for job_id, row in accounting_jobs.items():
        if (
            row["JobName"] != "xdr_train"
            or row["State"] != "COMPLETED"
            or row["ExitCode"] != "0:0"
            or row["NodeList"] != "node302"
        ):
            raise ValueError(f"job {job_id} lacks clean terminal accounting")

    audits = {
        cohort: _load_json(INPUTS[f"{cohort}_audit"])
        for cohort in ("e66", "e68")
    }
    audited_runs = {
        cohort: _audit_runs(payload, cohort)
        for cohort, payload in audits.items()
    }
    curve_rows = {
        cohort: _terminal_rows(
            _load_json(INPUTS[f"{cohort}_curve"]),
            cohort,
        )
        for cohort in ("e66", "e68")
    }
    summaries = {
        cohort: {
            metric: fmean(row[metric] for row in rows)
            for metric in METRICS
        }
        for cohort, rows in curve_rows.items()
    }
    deltas = {
        metric: summaries["e68"][metric] - summaries["e66"][metric]
        for metric in METRICS
    }
    paired_seed_deltas = {
        metric: [
            curve_rows["e68"][index][metric]
            - curve_rows["e66"][index][metric]
            for index in range(len(SEEDS))
        ]
        for metric in METRICS
    }
    if not all(deltas[name] > 0 for name in METRICS):
        raise ValueError("terminal E68 three-seed means are not all positive")
    for metric in ("greedy", "pass8", "distinct8"):
        if not all(value > 0 for value in paired_seed_deltas[metric]):
            raise ValueError(
                f"terminal E68 {metric} is not positive for all paired seeds"
            )
    if (
        _sha256(INPUTS["figure_current"])
        != _sha256(INPUTS["figure_e68_provenance"])
    ):
        raise ValueError("E68 provenance figure is not an exact current copy")

    payload = {
        "schema": "e66_e68_mathir_terminal_recovery_summary_v1",
        "created_at": "2026-07-28",
        "status": "pass",
        "scope": (
            "Paired terminal MathIR causal result only; the broader frozen "
            "E66/E68 campaign remains incomplete and is not claimed."
        ),
        "terminal_contract": {
            "seeds": list(SEEDS),
            "step": TERMINAL_STEP,
            "training_passes": 12,
            "primary_split": "multi_answer",
            "fixed_evaluations_per_seed": 49,
            "recovery_jobs": [
                {
                    "cohort": cohort,
                    "seed": seed,
                    "job_id": job_id,
                    "state": accounting_jobs[job_id]["State"],
                    "exit_code": accounting_jobs[job_id]["ExitCode"],
                    "elapsed": accounting_jobs[job_id]["Elapsed"],
                    "node": accounting_jobs[job_id]["NodeList"],
                }
                for (cohort, seed), job_id in EXPECTED_JOBS.items()
            ],
        },
        "audit_status": {
            cohort: {
                "campaign_status": audits[cohort]["status"],
                "campaign_violations": audits[cohort]["violations"],
                "mathir_terminal_runs": len(audited_runs[cohort]),
                "entropy_gated_interventions": (
                    audits[cohort]["summary"].get(
                        "entropy_gated_interventions", 0
                    )
                ),
            }
            for cohort in ("e66", "e68")
        },
        "terminal_seed_rows": curve_rows,
        "three_seed_means": summaries,
        "e68_minus_e66": deltas,
        "paired_seed_deltas": paired_seed_deltas,
        "decision": {
            "choice": "successor_first",
            "finish_entire_frozen_54_run_campaign_now": False,
            "rationale": (
                "The paired MathIR causal result is terminal and positive, "
                "while terminal E64 MATH-500 failed the frozen directional "
                "transfer gate. Preserve the incomplete cells and prioritize "
                "the preregistered verified-route successor."
            ),
            "outcome_tuning": False,
        },
        "input_sha256": {
            label: _sha256(path) for label, path in INPUTS.items()
        },
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"status=pass jobs=6 terminal_runs=6 "
        f"e68_minus_e66_pass8={deltas['pass8']:.9f} "
        f"output={OUTPUT.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
