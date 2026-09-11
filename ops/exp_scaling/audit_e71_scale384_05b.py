#!/usr/bin/env python3
"""Fail-closed filesystem audit for E71's exact 20-job 384/128 cohort.

E71 re-runs Graph coloring and PantryPlan on the common 384-train / 128-eval
design. Beyond the per-run terminal and mechanism checks that E70 Stage A
performs, this audit enforces the two properties E71 exists to establish: that
both domains really trained on 384 prompts for 12 passes, and that neither
dataset was mutated after submission.
"""

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
IDENTITY = ROOT / "var/artifacts/e71_scale384_05b_identity.json"
PROTOCOL = ROOT / "paper/preregistration/e71_scale384_graph_pantry_05b_20260730.md"
LAUNCHER = ROOT / "ops/exp_scaling/launch_e71_scale384_05b.sh"
OUT = ROOT / "var/artifacts/e71_scale384_05b_audit_latest.json"
CONTROL = "grpo"
TREATMENT = "verified_first_global_replay_canonical"
SEEDS = (43, 44, 45, 46, 47)
TRAIN_ROWS = 384
EVAL_ROWS = 128
PASSES = 12
EXPECTED_STEPS = {
    "graph_coloring": TRAIN_ROWS * PASSES,
    "pantry_plan": TRAIN_ROWS * PASSES,
}
DATA_ROOTS = {
    "graph_coloring": ROOT / "var/data/graph_coloring_modebench_v2",
    "pantry_plan": ROOT / "var/data/pantry_plan_modebench_v2",
}
# Domain-specific actuator expectations. PantryPlan drives a six-step support
# mask through the canonical action space; Graph coloring runs the plain boxed
# surface with no canonical action task at all.
DOMAIN_MECHANISM = {
    "pantry_plan": {"actor/canonical_pantry_support_mask_actions": 1.0},
    "graph_coloring": {"actor/canonical_pantry_support_mask_actions": 0.0},
}
CRASH = re.compile(
    r"\[rank\d+\]: Traceback \(most recent call last\)|CUDA out of memory|"
    r"torch\.OutOfMemoryError|ChildFailedError|RayActorError|"
    r"worker unexpectedly died|RuntimeError:[^\n]*non-finite|"
    r"segmentation fault",
    re.IGNORECASE,
)
SIGTERM = re.compile(r"SIGTERM Signal received", re.IGNORECASE)
CRASH_BYTES = re.compile(CRASH.pattern.encode("ascii"), re.IGNORECASE)
SIGTERM_BYTES = re.compile(SIGTERM.pattern.encode("ascii"), re.IGNORECASE)


def _scan_process_failures(path: pathlib.Path) -> tuple[list[str], int]:
    """Stream multi-GB logs while preserving the 3,000-character SIGTERM context."""

    failures: list[str] = []
    infrastructure_interruptions = 0
    carry = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            data = carry + chunk
            carry_size = len(carry)
            for match in CRASH_BYTES.finditer(data):
                if match.end() <= carry_size:
                    continue
                nearby = data[max(0, match.start() - 3000) : match.start()]
                text = match.group(0).decode("ascii", errors="replace")
                if "Traceback" in text and SIGTERM_BYTES.search(nearby):
                    infrastructure_interruptions += 1
                else:
                    failures.append(text)
            carry = data[-3000:]
    return failures, infrastructure_interruptions


def _digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hash(root: pathlib.Path) -> str:
    """Reproduce the launcher's find | sort | sha256sum | sha256sum digest."""

    # `find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum`:
    # sort on the literal "./path" bytes, not on Path component order.
    names = sorted(
        "./" + str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
    )
    inner = hashlib.sha256()
    for name in names:
        blob = (root / name[2:]).read_bytes()
        inner.update(f"{hashlib.sha256(blob).hexdigest()}  {name}\n".encode())
    return inner.hexdigest()


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
    domain: str,
    row: dict[str, Any],
    label: str,
) -> list[str]:
    violations: list[str] = []
    common = {
        "train/canonical_replay_gold_support_feedback": 0.0,
        "train/canonical_replay_alpha_projection_active": 0.0,
        "train/canonical_replay_global_scheduler_active": 1.0,
        "train/canonical_replay_global_groups_per_step": 1.0,
        "train/canonical_replay_global_bootstrap_steps": 0.0,
    }
    arm_specific = {
        TREATMENT: {"train/canonical_replay_compute_only": 0.0},
        CONTROL: {"train/canonical_replay_compute_only": 1.0},
    }
    required = {
        **common,
        **arm_specific.get(arm, {}),
        **DOMAIN_MECHANISM.get(domain, {}),
    }
    for key, expected in required.items():
        if key in row and float(row[key]) != expected:
            violations.append(f"{label}: {key}={row[key]!r}, expected {expected}")

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

    if identity.get("schema") != "e71_scale384_05b_v1":
        violations.append("identity schema mismatch")
    for field, expected in (
        ("train_rows", TRAIN_ROWS),
        ("eval_rows", EVAL_ROWS),
        ("optimizer_updates_per_run", TRAIN_ROWS * PASSES),
    ):
        if identity.get(field) != expected:
            violations.append(
                f"identity {field}={identity.get(field)!r}, expected {expected}"
            )
    for key, path in (
        ("protocol_sha256", PROTOCOL),
        ("launcher_sha256", LAUNCHER),
    ):
        try:
            if _digest(path) != identity.get(key):
                violations.append(f"{key} mismatch")
        except Exception as exc:
            violations.append(f"cannot verify {path}: {exc}")

    # E71 exists to change the dataset size deliberately; prove neither dataset
    # drifted after submission.
    data_state: dict[str, Any] = {}
    for domain, data_root in DATA_ROOTS.items():
        recorded = (identity.get("data_tree_sha256") or {}).get(domain)
        try:
            observed = _tree_hash(data_root)
        except Exception as exc:
            observed = None
            violations.append(f"{domain}: cannot hash {data_root}: {exc}")
        if observed is not None and recorded is not None and observed != recorded:
            violations.append(
                f"{domain}: data tree changed since submission "
                f"({observed[:12]} != {recorded[:12]})"
            )
        data_state[domain] = {
            "path": str(data_root.relative_to(ROOT)),
            "recorded_sha256": recorded,
            "observed_sha256": observed,
            "unchanged": observed == recorded,
        }

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
        if len(rows) != 10:
            violations.append(f"{domain}: expected 10 exact jobs, found {len(rows)}")
        arms_and_seeds = {(row["arm"], int(row["seed"])) for row in rows}
        expected_pairs = {
            (arm, seed) for arm in (CONTROL, TREATMENT) for seed in SEEDS
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
                    _mechanism_violations(
                        str(record["arm"]), domain, latest, label
                    )
                )
            log_crash = False
            infrastructure_interruptions = 0
            for suffix in ("out", "err"):
                log = ROOT / f"var/artifacts/logs/xdr_train-{job_id}.{suffix}"
                if log.is_file():
                    failures, interruptions = _scan_process_failures(log)
                    infrastructure_interruptions += interruptions
                    for failure in failures:
                        log_crash = True
                        violations.append(
                            f"{label}: crash signature {failure!r} in {log.name}"
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
                        latest_step / (expected_step / PASSES)
                        if latest_step >= 0
                        else None
                    ),
                    "terminal": terminal,
                    "crash_signature": log_crash,
                    "infrastructure_interruptions": infrastructure_interruptions,
                }
            )
        domain_payload[domain] = {
            "expected_terminal_step": expected_step,
            "train_rows": TRAIN_ROWS,
            "eval_rows": EVAL_ROWS,
            "runs": runs,
        }

    status = "fail" if violations else ("pass" if all_terminal else "in_progress")
    payload = {
        "schema": "e71_scale384_05b_audit_v1",
        "status": status,
        "summary": {
            "expected_runs": 20,
            "identity_runs": total_runs,
            "materialized_runs": materialized_runs,
            "metric_runs": metric_runs,
            "terminal_runs": terminal_runs,
        },
        "design": {
            "train_rows": TRAIN_ROWS,
            "eval_rows": EVAL_ROWS,
            "prompt_passes": PASSES,
            "seeds": list(SEEDS),
        },
        "data_integrity": data_state,
        "supersedes": identity.get("supersedes"),
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
        f"[e71-scale384-audit] status={status} "
        f"materialized={materialized_runs}/20 metrics={metric_runs}/20 "
        f"terminal={terminal_runs}/20 violations={len(payload['violations'])} "
        f"out={OUT}"
    )


if __name__ == "__main__":
    main()
