#!/usr/bin/env python3
"""Fail closed if a five-domain run goes more than one epoch without eval."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
INPUTS = {
    "e61": ROOT / "var/artifacts/e61r1_e58_vs_grpo_12pass_audit_latest.json",
    "e64": ROOT / "var/artifacts/e64_math500_realism_matched_audit_latest.json",
    "e66": (
        ROOT
        / "var/artifacts/e66_same_plumbing_actuator_ablation_audit_latest.json"
    ),
    "e68": (
        ROOT
        / "var/artifacts/"
        "e68_separated_support_actuator_ablation_audit_latest.json"
    ),
}
OUT = ROOT / "var/artifacts/e65_eval_cadence_audit_latest.json"
EVAL_NAME = re.compile(r"^(\d+)_.*\.json$")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _runs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    flat = payload.get("runs")
    if isinstance(flat, list):
        return flat
    return [
        run
        for domain in payload.get("domains", {}).values()
        for run in domain.get("runs", [])
    ]


def _run_dir(run: dict[str, Any]) -> Path | None:
    recorded = run.get("run_dir")
    if recorded:
        path = Path(str(recorded))
        return path if path.is_absolute() else ROOT / path
    stamp = run.get("run_stamp")
    job_id = run.get("job_id")
    if not stamp or job_id is None:
        return None
    candidates = sorted(
        (ROOT / "var/data").glob(f"*_{stamp}/debug_job{int(job_id)}")
    )
    return candidates[0] if len(candidates) == 1 else None


def _eval_steps(run_dir: Path) -> list[int]:
    steps: set[int] = set()
    for path in (run_dir / "eval_results").glob("*.json"):
        match = EVAL_NAME.match(path.name)
        if match:
            steps.add(int(match.group(1)))
    return sorted(steps)


def _audit_steps(
    eval_steps: list[int],
    *,
    latest_step: int,
    expected_step: int,
    epoch_steps: int,
) -> list[str]:
    violations: list[str] = []
    if latest_step < 0:
        return violations
    landed = sorted(set(step for step in eval_steps if 0 <= step <= latest_step))
    if not landed:
        return ["no evaluation at or before the latest optimizer step"]
    boundaries = landed
    if boundaries[0] > epoch_steps:
        violations.append(
            f"first evaluation step {boundaries[0]} exceeds one epoch "
            f"({epoch_steps} steps)"
        )
    for left, right in zip(boundaries, boundaries[1:]):
        if right - left > epoch_steps:
            violations.append(
                f"evaluation gap {left}->{right} is {right - left} steps, "
                f"exceeding one epoch ({epoch_steps})"
            )
    if latest_step - boundaries[-1] > epoch_steps:
        violations.append(
            f"trailing evaluation gap {boundaries[-1]}->{latest_step} is "
            f"{latest_step - boundaries[-1]} steps, exceeding one epoch "
            f"({epoch_steps})"
        )
    if latest_step >= expected_step and expected_step not in eval_steps:
        violations.append(
            f"terminal step {expected_step} has no exact evaluation artifact"
        )
    return violations


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    records: list[dict[str, Any]] = []
    violations: list[str] = []
    expected_runs = 0
    materialized_runs = 0
    audited_runs = 0
    preoptimizer_runs = 0

    for cohort, path in INPUTS.items():
        payload = _load(path)
        expected_runs += int(payload.get("summary", {}).get("expected_runs", 0))
        for run in _runs(payload):
            run_dir = _run_dir(run)
            if run_dir is None:
                continue
            materialized_runs += 1
            latest_step = int(run.get("latest_step", -1))
            expected_step = int(run["expected_step"])
            if expected_step <= 0 or expected_step % 12:
                label = f"{cohort}/{run.get('run_stamp')}/j{run.get('job_id')}"
                violations.append(
                    f"{label}: expected step {expected_step} is not 12 epochs"
                )
                continue
            epoch_steps = expected_step // 12
            steps = _eval_steps(run_dir)
            label = f"{cohort}/{run.get('run_stamp')}/j{run.get('job_id')}"
            if latest_step < 0:
                preoptimizer_runs += 1
                run_violations: list[str] = []
            else:
                audited_runs += 1
                run_violations = _audit_steps(
                    steps,
                    latest_step=latest_step,
                    expected_step=expected_step,
                    epoch_steps=epoch_steps,
                )
            violations.extend(
                f"{label}: {violation}" for violation in run_violations
            )
            landed = [step for step in steps if 0 <= step <= latest_step]
            gaps = [
                right - left
                for left, right in zip(landed, landed[1:])
            ]
            records.append(
                {
                    "cohort": cohort,
                    "job_id": int(run["job_id"]),
                    "run_stamp": run["run_stamp"],
                    "run_dir": str(run_dir.relative_to(ROOT)),
                    "latest_step": latest_step,
                    "expected_step": expected_step,
                    "epoch_steps": epoch_steps,
                    "eval_steps": steps,
                    "latest_landed_eval_step": landed[-1] if landed else None,
                    "maximum_landed_gap_steps": max(gaps) if gaps else 0,
                    "violations": run_violations,
                }
            )

    status = "pass" if audited_runs > 0 and not violations else "fail"
    payload = {
        "schema": "e65_eval_cadence_audit_v1",
        "status": status,
        "policy": {
            "minimum_frequency": "at least once per prompt epoch",
            "maximum_gap_epochs": 1,
            "terminal_exact_evaluation_required": True,
            "evidence": "durable eval_results JSON artifacts",
        },
        "summary": {
            "expected_runs": expected_runs,
            "materialized_runs": materialized_runs,
            "audited_runs": audited_runs,
            "preoptimizer_runs": preoptimizer_runs,
            "violation_count": len(violations),
        },
        "runs": records,
        "violations": sorted(set(violations)),
    }
    _atomic_json(OUT, payload)
    print(
        f"[eval-cadence-audit] status={status} "
        f"audited={audited_runs}/{expected_runs} "
        f"preoptimizer={preoptimizer_runs} violations={len(violations)}"
    )


if __name__ == "__main__":
    main()
