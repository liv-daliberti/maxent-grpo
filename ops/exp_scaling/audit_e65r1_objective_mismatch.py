#!/usr/bin/env python3
"""Record the non-outcome runtime mismatch that invalidated E65R1."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile


ROOT = Path(__file__).resolve().parents[2]
E65_IDENTITY = (
    ROOT
    / "var/artifacts/e65r1_entropy_gated_singleton_confirmation_identity.json"
)
E66_IDENTITY = (
    ROOT / "var/artifacts/e66_same_plumbing_actuator_ablation_identity.json"
)
FROZEN_RUNTIME = (
    ROOT
    / "var/artifacts/source_snapshots/"
    "e65_entropy_gate_ops_"
    "ff2a4f37d8653ba1a5888538d22743269c73d541b7bc6c8db214cedaec9c5d8f/"
    "ops/run_experiment.sh"
)
OUT = ROOT / "var/artifacts/e65r1_objective_mismatch_invalidation.json"
CONFIG = re.compile(
    r"online_canonical_bank_alpha=(\S+) novelty_beta=(\S+)"
)


def _runtime_config(job_id: int) -> dict[str, float] | None:
    path = ROOT / f"var/artifacts/logs/xdr_train-{job_id}.out"
    if not path.is_file():
        return None
    match = CONFIG.search(path.read_text(encoding="utf-8", errors="replace"))
    if match is None:
        return None
    return {
        "online_canonical_bank_alpha": float(match.group(1)),
        "online_canonical_novelty_beta": float(match.group(2)),
    }


def main() -> None:
    e65 = json.loads(E65_IDENTITY.read_text(encoding="utf-8"))
    e66 = json.loads(E66_IDENTITY.read_text(encoding="utf-8"))
    e65_jobs = [
        int(run["job_id"])
        for domain in e65["jobs"].values()
        for run in domain
    ]
    e66_jobs = [
        int(run["job_id"])
        for domain in e66["jobs"].values()
        for run in domain
    ]
    e65_configs = {
        str(job_id): _runtime_config(job_id) for job_id in e65_jobs
    }
    e66_configs = {
        str(job_id): config
        for job_id in e66_jobs
        if (config := _runtime_config(job_id)) is not None
    }
    runtime_text = FROZEN_RUNTIME.read_text(encoding="utf-8")
    repair_start = runtime_text.index(
        "  verified_entropy_gated_singleton_escape_canonical)"
    )
    repair_end = runtime_text.index("\n    ;;", repair_start)
    repair_branch = runtime_text[repair_start:repair_end]
    branch_forces_zero = (
        "export OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.0"
        in repair_branch
    )
    all_e65_zero = len(e65_configs) == 12 and all(
        config is not None
        and config["online_canonical_novelty_beta"] == 0.0
        for config in e65_configs.values()
    )
    e66_observed_literal = bool(e66_configs) and all(
        config["online_canonical_novelty_beta"] == 0.5
        for config in e66_configs.values()
    )
    confirmed = branch_forces_zero and all_e65_zero and e66_observed_literal
    violations = []
    if not branch_forces_zero:
        violations.append("frozen E65 repair branch does not force beta zero")
    if not all_e65_zero:
        violations.append("not all 12 E65 jobs report runtime beta zero")
    if not e66_observed_literal:
        violations.append(
            "materialized E66 controls do not uniformly report beta 0.5"
        )
    payload = {
        "schema": "e65r1_objective_mismatch_invalidation_v1",
        "status": "confirmed" if confirmed else "fail",
        "scientific_disposition": (
            "exclude E65R1 from confirmatory denominator and preserve as "
            "engineering evidence"
        ),
        "expected_literal_e58_novelty_beta": 0.5,
        "e65_runtime_configs": e65_configs,
        "e66_materialized_runtime_configs": e66_configs,
        "frozen_repair_branch_forces_zero": branch_forces_zero,
        "e65_jobs_checked": len(e65_configs),
        "e66_jobs_checked": len(e66_configs),
        "violations": violations,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{OUT.name}.", dir=OUT.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, OUT)
    print(
        f"[e65r1-invalidation] status={payload['status']} "
        f"e65={payload['e65_jobs_checked']} "
        f"e66={payload['e66_jobs_checked']} "
        f"violations={len(violations)}"
    )


if __name__ == "__main__":
    main()
