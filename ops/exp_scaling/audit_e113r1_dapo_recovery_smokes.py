#!/usr/bin/env python3
"""Audit the two E113-R1 operational smokes against their frozen pass rule."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = ROOT / "var/artifacts/e113r1_dapo_recovery_smoke_jobs.json"
EXPECTED_SCHEMA = "e113r1_dapo_recovery_smoke_jobs_v1"
EXPECTED_FAMILIES = {"qwen05b", "falcon1b"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_smoke(
    family: str,
    smoke: dict[str, Any],
    *,
    expected_receipt_step: int | None = None,
) -> dict[str, Any]:
    violations: list[str] = []
    run_dir = Path(str(smoke.get("run_dir", "")))
    marker = run_dir / "TRAINING_COMPLETE.json"
    if not marker.is_file():
        return {
            "family": family,
            "passed": False,
            "violations": [f"missing completion receipt: {marker}"],
        }

    receipt = load_json(marker)
    if receipt.get("schema") != "oat_zero_training_complete_v1":
        violations.append("completion receipt schema drifted")
    terminal_step = int(receipt.get("terminal_step", -1))
    target = int(smoke.get("max_train", -1))
    # The frozen R1 protocol expected the receipt's outer runner step to equal
    # the optimizer-step target. Keep that historical check as the default.
    # A prospective successor may explicitly admit the implementation's
    # target+1 terminal-summary convention while still requiring policy steps
    # 1..target below.
    receipt_target = (
        target if expected_receipt_step is None else expected_receipt_step
    )
    if terminal_step != receipt_target or target != 32:
        violations.append(
            f"terminal step is {terminal_step}; expected exactly {receipt_target}"
        )

    attempt = Path(str(receipt.get("terminal_attempt", "")))
    metrics_path = attempt / "train_metrics.jsonl"
    if not metrics_path.is_file():
        violations.append(f"missing terminal metrics: {metrics_path}")
        rows: list[dict[str, Any]] = []
    else:
        rows = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    raw_accepted = [
        row
        for row in rows
        if float(row.get("actor/dapo_accepted_groups", 0.0)) == 1.0
    ]
    accepted_by_step: dict[int, dict[str, Any]] = {}
    records_without_step: list[dict[str, Any]] = []
    for row in raw_accepted:
        raw_step = row.get("trainer/global_step", row.get("misc/global_step"))
        if raw_step is None:
            records_without_step.append(row)
            continue
        step = int(raw_step)
        if 1 <= step <= target:
            accepted_by_step.setdefault(step, row)
    if accepted_by_step:
        accepted = [accepted_by_step[step] for step in sorted(accepted_by_step)]
        expected_steps = set(range(1, target + 1))
        if set(accepted_by_step) != expected_steps:
            violations.append(
                "accepted DAPO optimizer-step set is incomplete or non-contiguous"
            )
    else:
        # Backward-compatible path for synthetic and early metrics without an
        # explicit policy/global-step field.
        accepted = records_without_step
    if len(accepted) != 32:
        violations.append(
            f"accepted DAPO metric records={len(accepted)}; expected 32"
        )

    finite_fields = (
        "train/pg_loss",
        "train/policy_grad_norm",
        "train/dapo_token_level_active_tokens",
    )
    for index, row in enumerate(accepted, start=1):
        if float(row.get("actor/dapo_dynamic_sampling_enabled", 0.0)) != 1.0:
            violations.append(f"update {index}: dynamic sampling not enabled")
        generation_batches = float(
            row.get("actor/dapo_generation_batches", math.nan)
        )
        if not math.isfinite(generation_batches) or not (
            1.0 <= generation_batches <= 10.0
        ):
            violations.append(
                f"update {index}: invalid generation batches {generation_batches}"
            )
        if float(row.get("train/dapo_enabled", 0.0)) != 1.0:
            violations.append(f"update {index}: DAPO loss path not enabled")
        if not math.isclose(
            float(row.get("train/dapo_clip_low", math.nan)),
            0.20,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            violations.append(f"update {index}: lower clip drifted")
        if not math.isclose(
            float(row.get("train/dapo_clip_high", math.nan)),
            0.28,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            violations.append(f"update {index}: upper clip drifted")
        for field in finite_fields:
            value = float(row.get(field, math.nan))
            if not math.isfinite(value):
                violations.append(f"update {index}: non-finite or missing {field}")
        if float(row.get("train/dapo_token_level_active_tokens", 0.0)) <= 0.0:
            violations.append(f"update {index}: no active DAPO tokens")

    sampled_rows = max(
        (float(row.get("misc/query_step", 0.0)) for row in rows),
        default=0.0,
    )
    max_queries = int(smoke.get("max_queries", -1))
    if max_queries != 5120 or sampled_rows > max_queries:
        violations.append(
            f"sampled rows {sampled_rows:g} exceed or drift from ceiling "
            f"{max_queries}"
        )
    return {
        "family": family,
        "job_id": int(smoke.get("job_id", -1)),
        "run_dir": str(run_dir),
        "terminal_step": terminal_step,
        "expected_receipt_step": receipt_target,
        "accepted_updates": len(accepted),
        "raw_accepted_records": len(raw_accepted),
        "duplicate_accepted_records": len(raw_accepted) - len(accepted),
        "sampled_rows": sampled_rows,
        "max_queries": max_queries,
        "passed": not violations,
        "violations": violations,
    }


def audit(ledger_path: Path) -> dict[str, Any]:
    if not ledger_path.is_file():
        return {
            "schema": "e113r1_dapo_recovery_smoke_audit_v1",
            "passed": False,
            "violations": [f"missing ledger: {ledger_path}"],
            "smokes": [],
        }
    ledger = load_json(ledger_path)
    top_violations: list[str] = []
    if ledger.get("schema") != EXPECTED_SCHEMA:
        top_violations.append("ledger schema drifted")
    if ledger.get("released") is not True:
        top_violations.append("ledger is not released")
    if int(ledger.get("scientific_cells", -1)) != 0 or ledger.get("runs") != []:
        top_violations.append("recovery launcher included scientific cells")
    if ledger.get("smoke_domain") != "graph_coloring":
        top_violations.append("recovery smoke domain drifted")
    smokes = ledger.get("smokes")
    if not isinstance(smokes, dict) or set(smokes) != EXPECTED_FAMILIES:
        top_violations.append("recovery ledger must contain exactly two families")
        smoke_audits: list[dict[str, Any]] = []
    else:
        smoke_audits = [
            audit_smoke(family, smokes[family])
            for family in sorted(EXPECTED_FAMILIES)
        ]
    return {
        "schema": "e113r1_dapo_recovery_smoke_audit_v1",
        "ledger": str(ledger_path),
        "passed": not top_violations
        and len(smoke_audits) == 2
        and all(row["passed"] for row in smoke_audits),
        "violations": top_violations,
        "smokes": smoke_audits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args()
    payload = audit(args.ledger.resolve())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
