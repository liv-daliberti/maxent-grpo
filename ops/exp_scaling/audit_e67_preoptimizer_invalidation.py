#!/usr/bin/env python3
"""Confirm that E67 was invalidated before any optimizer update."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[2]
IDENTITY = (
    ROOT
    / "var/artifacts/e67_corrected_same_objective_actuator_ablation_identity.json"
)
OUT = ROOT / "var/artifacts/e67_preoptimizer_invalidation.json"
EXPECTED_ERROR = (
    "ValueError: proposal support cannot feed an on-policy "
    "canonical-bank advantage"
)


def main() -> None:
    violations: list[str] = []
    identity = json.loads(IDENTITY.read_text(encoding="utf-8"))
    if (
        identity.get("schema")
        != "e67_corrected_same_objective_actuator_ablation_v1"
    ):
        violations.append("E67 identity schema mismatch")
    jobs = [
        row
        for domain_rows in identity.get("jobs", {}).values()
        for row in domain_rows
    ]
    if len(jobs) != 12:
        violations.append(f"E67 identity has {len(jobs)} jobs, expected 12")

    metric_paths: list[str] = []
    validator_failures: list[int] = []
    other_tracebacks: list[int] = []
    for row in jobs:
        job_id = int(row["job_id"])
        run_stamp = str(row["run_stamp"])
        metric_paths.extend(
            str(path.relative_to(ROOT))
            for path in (ROOT / "var/data").glob(
                f"*_{run_stamp}/debug_job{job_id}/train_metrics.jsonl"
            )
        )
        texts = []
        for suffix in ("out", "err"):
            path = (
                ROOT
                / "var/artifacts/logs"
                / f"xdr_train-{job_id}.{suffix}"
            )
            if path.is_file():
                texts.append(
                    path.read_text(encoding="utf-8", errors="replace")
                )
        text = "\n".join(texts)
        if EXPECTED_ERROR in text:
            validator_failures.append(job_id)
        elif "Traceback (most recent call last)" in text:
            other_tracebacks.append(job_id)

    if metric_paths:
        violations.append(
            "E67 unexpectedly wrote optimizer metrics: "
            + ", ".join(sorted(metric_paths))
        )
    if not validator_failures:
        violations.append("no E67 job reproduced the common argument validator")
    if other_tracebacks:
        violations.append(
            "E67 had unrelated tracebacks: "
            + ", ".join(str(job_id) for job_id in other_tracebacks)
        )

    status = "confirmed" if not violations else "fail"
    payload = {
        "schema": "e67_preoptimizer_invalidation_v1",
        "status": status,
        "reason": (
            "shared frozen configuration mixed proposal support with the "
            "on-policy E58 novelty-count table"
        ),
        "expected_runs": 12,
        "optimizer_metric_files": sorted(metric_paths),
        "validator_failure_jobs": sorted(validator_failures),
        "invalidated_job_ids": sorted(int(row["job_id"]) for row in jobs),
        "performance_evidence_eligible": False,
        "violations": violations,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{OUT.name}.", dir=OUT.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, OUT)
    print(
        f"[e67-invalidation] status={status} "
        f"validator_failures={len(validator_failures)} "
        f"optimizer_metrics={len(metric_paths)} "
        f"violations={len(violations)}"
    )


if __name__ == "__main__":
    main()
