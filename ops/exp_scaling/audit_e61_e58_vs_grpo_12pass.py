#!/usr/bin/env python3
"""Fail-closed filesystem audit for E61's exact 24-job cohort."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pathlib
import re
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
IDENTITY = ROOT / "var/artifacts/e61_e58_vs_grpo_05b_12pass_identity.json"
PROTOCOL = ROOT / "paper/preregistration/e61_e58_vs_grpo_05b_12pass.md"
LAUNCHER = ROOT / "ops/exp_scaling/launch_e61_e58_vs_grpo_12pass.sh"
OUT = ROOT / "var/artifacts/e61_e58_vs_grpo_12pass_audit_latest.json"
CONTROL = "grpo"
TREATMENT = "verified_first_global_replay_canonical"
EXPECTED_STEPS = {
    "graph_coloring": 192 * 12,
    "countdown": 384 * 12,
    "python_factor": 384 * 12,
    "mathir": 384 * 12,
}
CRASH = re.compile(
    r"Traceback \(most recent call last\)|CUDA out of memory|"
    r"torch\.OutOfMemoryError|ChildFailedError|RayActorError|"
    r"worker unexpectedly died|RuntimeError:[^\n]*non-finite|"
    r"segmentation fault",
    re.IGNORECASE,
)


def _digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _last_jsonl(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    last = None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            if raw.strip():
                try:
                    last = json.loads(raw)
                except json.JSONDecodeError:
                    continue
    return last


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _manifest_rows(identity: dict[str, Any], domain: str) -> list[dict[str, Any]]:
    expected = identity["jobs"][domain]
    prefix = expected[0]["run_stamp"].split("_grpo_s", 1)[0].split(
        "_verified_first_global_replay_canonical_s", 1
    )[0]
    path = ROOT / f"var/artifacts/{prefix}_comparative_jobs.tsv"
    if _digest(path) != identity["manifest_sha256"][domain]:
        raise RuntimeError(f"{domain} manifest hash mismatch")
    rows = list(csv.DictReader(path.open(), delimiter="\t"))
    normalized = [
        {
            "arm": row["arm"],
            "seed": int(row["seed"]),
            "job_id": int(row["job_id"]),
            "run_stamp": row["run_stamp"],
        }
        for row in rows
    ]
    if normalized != expected:
        raise RuntimeError(f"{domain} manifest contents differ from identity")
    return normalized


def _run_dir(run_stamp: str, job_id: int) -> pathlib.Path | None:
    candidates = sorted(
        (ROOT / "var/data").glob(f"*_{run_stamp}/debug_job{job_id}")
    )
    return candidates[0] if len(candidates) == 1 else None


def _mechanism_violations(
    arm: str,
    row: dict[str, Any],
    label: str,
) -> list[str]:
    violations: list[str] = []
    if arm == TREATMENT:
        required = {
            "train/canonical_replay_gold_support_feedback": 0.0,
            "train/canonical_replay_alpha_projection_active": 0.0,
            "train/canonical_replay_global_scheduler_active": 1.0,
            "train/canonical_replay_global_groups_per_step": 1.0,
            "train/canonical_replay_global_bootstrap_steps": 0.0,
            (
                "train/semantic_shannon_success_conditioned_signed_"
                "open_set_projection_active"
            ): 0.0,
        }
        for key, expected in required.items():
            if key in row and float(row[key]) != expected:
                violations.append(
                    f"{label}: {key}={row[key]!r}, expected {expected}"
                )
    elif arm == CONTROL:
        key = "train/canonical_replay_global_scheduler_active"
        if key in row and float(row[key]) != 0.0:
            violations.append(f"{label}: baseline global replay is active")

    for key, value in row.items():
        if not key.startswith("train/"):
            continue
        if key.endswith(("_nan", "_inf")) and _finite(value) and value > 0:
            violations.append(f"{label}: non-finite diagnostic {key}={value}")
        if (
            ("next_alpha" in key or "next_coefficient" in key)
            and value is not None
            and not _finite(value)
        ):
            violations.append(f"{label}: non-finite coefficient {key}={value}")
    return violations


def main() -> None:
    violations: list[str] = []
    try:
        identity = _load(IDENTITY)
    except Exception as exc:
        identity = {}
        violations.append(f"cannot load identity: {exc}")

    if identity.get("schema") != "e61_e58_vs_grpo_05b_12pass_v1":
        violations.append("identity schema mismatch")
    for key, path in (
        ("protocol_sha256", PROTOCOL),
        ("launcher_sha256", LAUNCHER),
    ):
        try:
            if _digest(path) != identity.get(key):
                violations.append(f"{key} mismatch")
        except Exception as exc:
            violations.append(f"cannot verify {path}: {exc}")

    domain_payload: dict[str, Any] = {}
    all_terminal = True
    total_runs = 0
    materialized_runs = 0
    metric_runs = 0
    terminal_runs = 0

    for domain, expected_step in EXPECTED_STEPS.items():
        runs: list[dict[str, Any]] = []
        try:
            rows = _manifest_rows(identity, domain)
        except Exception as exc:
            violations.append(str(exc))
            rows = identity.get("jobs", {}).get(domain, [])
        if len(rows) != 6:
            violations.append(f"{domain}: expected 6 exact jobs, found {len(rows)}")
        arms_and_seeds = {(row["arm"], int(row["seed"])) for row in rows}
        expected_pairs = {
            (arm, seed)
            for arm in (CONTROL, TREATMENT)
            for seed in (43, 44, 45)
        }
        if arms_and_seeds != expected_pairs:
            violations.append(f"{domain}: arm/seed set mismatch")

        for record in rows:
            total_runs += 1
            job_id = int(record["job_id"])
            run_stamp = str(record["run_stamp"])
            label = f"{domain}/{record['arm']}/s{record['seed']}/j{job_id}"
            run_dir = _run_dir(run_stamp, job_id)
            latest = None
            if run_dir is not None:
                materialized_runs += 1
                latest = _last_jsonl(run_dir / "train_metrics.jsonl")
            latest_step = -1
            if latest is not None:
                metric_runs += 1
                raw_step = latest.get(
                    "trainer/global_step", latest.get("trainer/step", -1)
                )
                if _finite(raw_step):
                    latest_step = int(raw_step)
                violations.extend(
                    _mechanism_violations(str(record["arm"]), latest, label)
                )

            log_crash = False
            for suffix in ("out", "err"):
                log = ROOT / f"var/artifacts/logs/xdr_train-{job_id}.{suffix}"
                if log.is_file():
                    match = CRASH.search(
                        log.read_text(encoding="utf-8", errors="replace")
                    )
                    if match:
                        log_crash = True
                        violations.append(
                            f"{label}: crash signature {match.group(0)!r} in {log.name}"
                        )
            terminal = latest_step >= expected_step
            if terminal:
                terminal_runs += 1
            else:
                all_terminal = False
            runs.append(
                {
                    **record,
                    "run_dir": (
                        str(run_dir.relative_to(ROOT)) if run_dir is not None else None
                    ),
                    "latest_step": latest_step,
                    "expected_step": expected_step,
                    "training_passes": (
                        latest_step / (expected_step / 12)
                        if latest_step >= 0
                        else None
                    ),
                    "terminal": terminal,
                    "crash_signature": log_crash,
                }
            )
        domain_payload[domain] = {
            "expected_terminal_step": expected_step,
            "runs": runs,
        }

    status = "fail" if violations else ("pass" if all_terminal else "in_progress")
    payload = {
        "schema": "e61_e58_vs_grpo_05b_12pass_audit_v1",
        "status": status,
        "summary": {
            "expected_runs": 24,
            "identity_runs": total_runs,
            "materialized_runs": materialized_runs,
            "metric_runs": metric_runs,
            "terminal_runs": terminal_runs,
        },
        "domains": domain_payload,
        "violations": sorted(set(violations)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{OUT.name}.", dir=OUT.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, OUT)
    print(
        f"[e61-audit] status={status} "
        f"materialized={materialized_runs}/24 metrics={metric_runs}/24 "
        f"terminal={terminal_runs}/24 violations={len(payload['violations'])} "
        f"out={OUT}"
    )


if __name__ == "__main__":
    main()
