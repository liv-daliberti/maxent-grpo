#!/usr/bin/env python3
"""Hash the shell/Python execution surface that sits outside ``src/``."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXECUTION_FILES = (
    "ops/repo_env.sh",
    "ops/resolve_eval_cadence.py",
    "ops/run_experiment.sh",
    "ops/train.sh",
    "ops/submit_countdown_comparative.sh",
    "ops/slurm/train_node302.slurm",
    "ops/make_modebench_data.py",
    "ops/exp_scaling/audit_e14_checkpoint.py",
    "ops/exp_scaling/check_e14_preflight.py",
    "ops/exp_scaling/e16_canonical_plan.py",
    "ops/exp_scaling/audit_e16_canonical_endpoint.py",
    "ops/exp_scaling/check_e16_canonical_smoke.py",
    "ops/exp_scaling/launch_e16_canonical_maxent_replication.sh",
    "ops/exp_scaling/verify_e16_antecedent.py",
    "ops/exp_scaling/verify_e16_canonical_datasets.py",
    "ops/exp_scaling/verify_e16_canonical_runtime.py",
    "ops/exp_scaling/verify_e16_execution_surface.py",
    "ops/exp_scaling/verify_e16_held_cohort.py",
    "ops/exp_scaling/verify_e16_smoke_approval.py",
    "ops/exp_scaling/verify_e14_dataset.py",
    "ops/exp_scaling/verify_e14_runtime.py",
)


def execution_surface_hash(repo_root: Path) -> tuple[str, list[dict[str, str]]]:
    digest = hashlib.sha256()
    records = []
    for relative in EXECUTION_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing E16 execution file: {path}")
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(bytes.fromhex(file_hash))
        records.append({"path": relative, "sha256": file_hash})
    return digest.hexdigest(), records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    digest, files = execution_surface_hash(args.repo_root.resolve())
    print(
        json.dumps(
            {
                "files": files,
                "schema": "e16_execution_surface_identity_v1",
                "sha256": digest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
