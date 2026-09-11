#!/usr/bin/env python3
"""Post-outcome semantic correction for Pantry's final-v1 dose audit.

The immutable v1 receipt correctly audited the ten completed cells, but its
strengthened dose check treated ``canonical_replay_mass_alpha_used`` as a
constant configured coefficient.  In the frozen learner that field is the
effective coefficient emitted by CanonicalReplayLikelihoodController:

    next_alpha = base_alpha * surprisal_ema / surprisal_reference

after a 64-observation warmup.  This amendment preserves and hashes the failed
v1 receipt, reuses all of its unrelated checks, and validates the complete
controller trajectory instead of requiring the effective coefficient to remain
equal to its registered 0.20 base.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


SEEDS = (76411, 76412, 76413, 76414, 76415)
TREATMENT = "verified_first_global_replay_canonical"
UPDATES = 384
BASE_ALPHA = 0.20
WARMUP = 64
ABS_TOL = 2e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=2e-6, abs_tol=ABS_TOL)


def value(row: dict[str, Any], suffix: str) -> float:
    return float(row[f"train/{suffix}"])


def locate_run(root: Path, run_stamp: str, job_id: int) -> Path:
    matches = sorted(
        path
        for path in (root / "var/data").glob(
            f"xdr_*_{run_stamp}/debug_job{job_id}"
        )
        if (path / "train_metrics.jsonl").is_file()
    )
    if len(matches) != 1:
        raise ValueError(
            f"{run_stamp}/j{job_id}: expected one metric directory, "
            f"found {len(matches)}"
        )
    return matches[0]


def controller_check(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        row
        for row in metric_rows
        if "train/canonical_replay_mass_alpha_used" in row
    ]
    errors: list[str] = []
    if len(rows) != UPDATES:
        errors.append(f"expected {UPDATES} controller rows, found {len(rows)}")

    alpha_used: list[float] = []
    mass_used: list[float] = []
    for index, row in enumerate(rows):
        observation = index + 1
        try:
            replay_used = value(row, "canonical_replay_alpha_used")
            replay_before = value(row, "canonical_replay_alpha_before")
            replay_next = value(row, "canonical_replay_next_alpha")
            replay_observations = value(row, "canonical_replay_observations")
            replay_ema = value(row, "canonical_replay_entropy_ema")
            replay_warm = value(row, "canonical_replay_warmup_complete")

            used = value(row, "canonical_replay_mass_alpha_used")
            before = value(row, "canonical_replay_mass_alpha_before")
            next_alpha = value(row, "canonical_replay_mass_next_alpha")
            observations = value(row, "canonical_replay_mass_observations")
            ema = value(row, "canonical_replay_mass_surprisal_ema")
            warm = value(row, "canonical_replay_mass_warmup_complete")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"row {observation}: incomplete controller telemetry: {exc}")
            continue

        scalars = (
            replay_used,
            replay_before,
            replay_next,
            replay_observations,
            replay_ema,
            replay_warm,
            used,
            before,
            next_alpha,
            observations,
            ema,
            warm,
        )
        if not all(math.isfinite(item) for item in scalars):
            errors.append(f"row {observation}: non-finite controller telemetry")
            continue
        if replay_used <= 0 or used <= 0 or replay_ema <= 0 or ema <= 0:
            errors.append(f"row {observation}: non-positive controller state")

        alpha_used.append(replay_used)
        mass_used.append(used)
        if not close(replay_observations, observation):
            errors.append(f"row {observation}: replay observation index mismatch")
        if not close(observations, observation):
            errors.append(f"row {observation}: mass observation index mismatch")
        if not close(replay_used, replay_before):
            errors.append(f"row {observation}: replay used/before discontinuity")
        if not close(used, before):
            errors.append(f"row {observation}: mass used/before discontinuity")
        if index == 0:
            if not close(replay_used, BASE_ALPHA):
                errors.append("row 1: replay controller did not start at base alpha")
            if not close(used, BASE_ALPHA):
                errors.append("row 1: mass controller did not start at base alpha")
        else:
            previous = rows[index - 1]
            if not close(
                replay_used,
                value(previous, "canonical_replay_next_alpha"),
            ):
                errors.append(
                    f"row {observation}: replay next-to-used trajectory discontinuity"
                )
            if not close(
                used,
                value(previous, "canonical_replay_mass_next_alpha"),
            ):
                errors.append(
                    f"row {observation}: mass next-to-used trajectory discontinuity"
                )

        expected_warm = 1.0 if observation >= WARMUP else 0.0
        if not close(replay_warm, expected_warm):
            errors.append(f"row {observation}: replay warmup flag mismatch")
        if not close(warm, expected_warm):
            errors.append(f"row {observation}: mass warmup flag mismatch")

        if observation <= WARMUP:
            expected_replay_next = BASE_ALPHA
            expected_mass_next = BASE_ALPHA
        else:
            try:
                entropy_reference = value(
                    row, "canonical_replay_reference_entropy"
                )
                surprisal_reference = value(
                    row, "canonical_replay_mass_surprisal_reference"
                )
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(
                    f"row {observation}: post-warmup reference absent: {exc}"
                )
                continue
            if entropy_reference <= 0 or surprisal_reference <= 0:
                errors.append(
                    f"row {observation}: post-warmup reference is non-positive"
                )
                continue
            expected_replay_next = BASE_ALPHA * entropy_reference / replay_ema
            expected_mass_next = BASE_ALPHA * ema / surprisal_reference
        if not close(replay_next, expected_replay_next):
            errors.append(f"row {observation}: replay controller law mismatch")
        if not close(next_alpha, expected_mass_next):
            errors.append(f"row {observation}: mass controller law mismatch")

    return {
        "passed": not errors and len(rows) == UPDATES,
        "row_count": len(rows),
        "warmup_observations": WARMUP,
        "registered_base_alpha": BASE_ALPHA,
        "minimum_replay_alpha_used": min(alpha_used) if alpha_used else None,
        "maximum_replay_alpha_used": max(alpha_used) if alpha_used else None,
        "minimum_mass_alpha_used": min(mass_used) if mass_used else None,
        "maximum_mass_alpha_used": max(mass_used) if mass_used else None,
        "errors": errors,
    }


def atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp = Path(handle.name)
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "var/artifacts/"
            "pantry_support_retention_final_v1_audit_amendment_v2.json"
        ),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output
    if not output.is_absolute():
        output = root / output

    original_path = (
        root / "var/artifacts/pantry_support_retention_final_v1_audit.json"
    )
    identity_path = (
        root / "var/artifacts/pantry_support_retention_final_v1_identity.json"
    )
    manifest_path = (
        root
        / "var/artifacts/pprepair_support_retention_final_v1_comparative_jobs.tsv"
    )
    original = json.loads(original_path.read_text())
    identity = json.loads(identity_path.read_text())
    manifest = list(csv.DictReader(manifest_path.open(), delimiter="\t"))

    expected_old_errors = {
        "failed check: treatment_replay_dose_020",
        *{
            (
                f"{TREATMENT}/s{seed}/j"
                f"{next(int(row['job_id']) for row in manifest if row['arm'] == TREATMENT and int(row['seed']) == seed)}"
                ": replay dose differs from 0.20"
            )
            for seed in SEEDS
        },
    }
    unrelated_checks = {
        key: bool(passed)
        for key, passed in original.get("checks", {}).items()
        if key != "treatment_replay_dose_020"
    }

    treatment_rows = [
        row for row in manifest if row.get("arm") == TREATMENT
    ]
    trajectories: dict[str, Any] = {}
    metric_hashes: dict[str, str] = {}
    for row in treatment_rows:
        seed = int(row["seed"])
        job_id = int(row["job_id"])
        directory = locate_run(root, row["run_stamp"], job_id)
        metrics_path = directory / "train_metrics.jsonl"
        metric_rows = [
            json.loads(line)
            for line in metrics_path.read_text().splitlines()
            if line.strip()
        ]
        trajectories[f"s{seed}"] = controller_check(metric_rows)
        metric_hashes[f"s{seed}/j{job_id}"] = sha256(metrics_path)

    scheduler = identity.get("held_scheduler_after", {})
    submission_checks = []
    for seed in SEEDS:
        text = str(scheduler.get(f"{TREATMENT}/s{seed}", ""))
        submission_checks.append(
            all(
                token in text
                for token in (
                    "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.20",
                    "OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA=0.20",
                    "OAT_ZERO_ONLINE_CANONICAL_REPLAY_WARMUP_STEPS=64",
                    "OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_WARMUP_STEPS=64",
                    (
                        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE="
                        "split_mass_balance_per_rollout"
                    ),
                )
            )
        )

    source_root = Path(identity["source_root"])
    controller_source = source_root / "oat_drgrpo/canonical_replay.py"
    learner_init_source = source_root / "oat_drgrpo/learner/init.py"
    controller_text = controller_source.read_text()
    init_text = learner_init_source.read_text()
    frozen_semantics = (
        "class CanonicalReplayLikelihoodController" in controller_text
        and "self.base_alpha * multiplier" in controller_text
        and "self.surprisal_ema / self.surprisal_reference" in controller_text
        and (
            "base_alpha=float(args.online_canonical_replay_mass_alpha)"
            in init_text
        )
    )

    checks = {
        "original_v1_receipt_preserved_as_failure": (
            original.get("status") == "fail"
            and original.get("decision")
            == "pantry_support_retention_final_stopped"
        ),
        "amendment_scope_exactly_one_misinterpreted_check": (
            set(original.get("errors", [])) == expected_old_errors
            and len(unrelated_checks) > 0
            and all(unrelated_checks.values())
        ),
        "manifest_exact_five_treatment_seeds": (
            len(manifest) == 10
            and len(treatment_rows) == 5
            and {int(row["seed"]) for row in treatment_rows} == set(SEEDS)
        ),
        "registered_base_coefficients_are_020": (
            identity.get("replay_alpha") == BASE_ALPHA
            and identity.get("replay_mass_alpha") == BASE_ALPHA
        ),
        "frozen_submissions_use_registered_controller": all(submission_checks),
        "frozen_source_implements_adaptive_effective_coefficients": (
            frozen_semantics
        ),
        "all_five_controller_trajectories_match_frozen_laws": (
            len(trajectories) == len(SEEDS)
            and all(item["passed"] for item in trajectories.values())
        ),
    }
    errors = [f"failed check: {key}" for key, passed in checks.items() if not passed]
    for seed, trajectory in trajectories.items():
        errors.extend(
            f"{seed}: {error}" for error in trajectory.get("errors", [])
        )

    payload = {
        "schema": "pantry-support-retention-final-audit-amendment-v2",
        "status": "pass" if not errors else "fail",
        "decision": (
            "pantry_support_retention_final_eligible_via_audit_amendment"
            if not errors
            else "pantry_support_retention_final_amendment_stopped"
        ),
        "post_outcome_amendment": True,
        "amends_schema": original.get("schema"),
        "amends_path": str(original_path),
        "amends_sha256": sha256(original_path),
        "correction": (
            "The v1 audit conflated the configured 0.20 base coefficient "
            "with the post-warmup adaptive effective mass coefficient. "
            "No run, endpoint, efficacy criterion, or original receipt was changed."
        ),
        "checks": checks,
        "errors": errors,
        "controller_trajectories": trajectories,
        "provenance": {
            "identity_path": str(identity_path),
            "identity_sha256": sha256(identity_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "controller_source_path": str(controller_source),
            "controller_source_sha256": sha256(controller_source),
            "learner_init_source_path": str(learner_init_source),
            "learner_init_source_sha256": sha256(learner_init_source),
            "treatment_metric_sha256": metric_hashes,
        },
        "terminal_endpoints_copied_without_reevaluation": original.get(
            "endpoints", {}
        ),
    }
    atomic(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "output": str(output),
                "errors": len(errors),
            },
            sort_keys=True,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
