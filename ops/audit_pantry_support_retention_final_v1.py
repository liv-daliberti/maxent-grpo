#!/usr/bin/env python3
"""Audit PantryPlan's five-seed support-retention final cohort."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import audit_pantry_stage_b_05b_12pass as base


base.SEEDS = (76411, 76412, 76413, 76414, 76415)
base.PREFIX = "pprepair_support_retention_final_v1"
base.QUALIFICATION = "pantry_support_retention_repair_v1_audit.json"
FINAL_DATA = "pantry_plan_modebench_v3_repair_final_view"
_tree_hash = base.tree_hash
_loads = base.json.loads
_atomic = base.atomic


def tree_hash(path: Path) -> str:
    if path.as_posix().endswith("var/data/pantry_plan_modebench_v2"):
        path = path.parent / FINAL_DATA
    return _tree_hash(path)


def loads(value):
    payload = _loads(value)
    if payload.get("schema") == "pantry-support-retention-repair-audit-v1":
        payload = {
            **payload,
            "decision": "eligible_for_ten_stage_b_jobs",
        }
    return payload


def atomic(path: Path, payload: dict) -> None:
    if payload.get("schema") != "pantry-stage-b-05b-12pass-audit-v1":
        _atomic(path, payload)
        return
    root = path.resolve().parents[2]
    manifest = (
        root
        / "var/artifacts/pprepair_support_retention_final_v1_comparative_jobs.tsv"
    )
    errors = list(payload.get("errors", []))
    checks = dict(payload.get("checks", {}))
    endpoints = {}
    alpha_checks = {}
    if not manifest.is_file():
        errors.append("repair final manifest absent during strengthened audit")
    else:
        rows = list(csv.DictReader(manifest.open(), delimiter="\t"))
        for row in rows:
            arm = row["arm"]
            seed = int(row["seed"])
            job_id = int(row["job_id"])
            label = f"{arm}/s{seed}/j{job_id}"
            directory = base.run_dir(root, row["run_stamp"], job_id)
            if directory is None:
                errors.append(f"{label}: unique repair-final directory absent")
                continue
            metrics_path = directory / "train_metrics.jsonl"
            metric_rows = [
                json.loads(line)
                for line in metrics_path.read_text().splitlines()
                if line.strip()
            ]
            evaluations = [
                item
                for item in metric_rows[:-1]
                if "eval/multi_answer/sampled_distinct_correct_at_8" in item
            ]
            if not evaluations:
                errors.append(f"{label}: evaluation rows absent")
                continue
            if any(
                float(item.get("eval/multi_answer/eval_count", -1)) != 64.0
                for item in evaluations
            ):
                errors.append(f"{label}: final evaluation did not use 64 rows")
            terminal = evaluations[-1]
            endpoints[f"{arm}/s{seed}"] = {
                metric: float(terminal[f"eval/multi_answer/{metric}"])
                for metric in (
                    "accuracy",
                    "sampled_mean_at_8",
                    "sampled_any_correct_at_8",
                    "sampled_distinct_correct_at_8",
                )
            }
            if arm == base.TREATMENT:
                alpha = [
                    float(
                        item.get(
                            "train/canonical_replay_alpha_used",
                            float("nan"),
                        )
                    )
                    for item in metric_rows[1 : base.UPDATES + 1]
                ]
                mass_alpha = [
                    float(
                        item.get(
                            "train/canonical_replay_mass_alpha_used",
                            float("nan"),
                        )
                    )
                    for item in metric_rows[1 : base.UPDATES + 1]
                ]
                passed = len(alpha) == base.UPDATES and all(
                    math.isfinite(value)
                    and math.isclose(
                        value, 0.20, rel_tol=0.05, abs_tol=0.005
                    )
                    for value in (*alpha, *mass_alpha)
                )
                alpha_checks[f"s{seed}"] = {
                    "passed": passed,
                    "count": len(alpha),
                    "minimum_replay_alpha": min(alpha),
                    "maximum_replay_alpha": max(alpha),
                    "minimum_mass_alpha": min(mass_alpha),
                    "maximum_mass_alpha": max(mass_alpha),
                }
                if not passed:
                    errors.append(f"{label}: replay dose differs from 0.20")
    identity_path = (
        root
        / "var/artifacts/pantry_support_retention_final_v1_identity.json"
    )
    identity = json.loads(identity_path.read_text())
    view_identity = (
        root
        / f"var/data/{FINAL_DATA}/final_view_identity.json"
    )
    checks.update(
        repair_final_identity=(
            identity.get("repair_schema")
            == "pantry-support-retention-final-identity-v1"
            and identity.get("evaluation_rows") == 64
            and identity.get("evaluation_source") == "previously_untouched_dev"
            and identity.get("replay_alpha") == 0.20
            and identity.get("replay_mass_alpha") == 0.20
        ),
        final_view_identity=(
            view_identity.is_file()
            and json.loads(view_identity.read_text()).get(
                "evaluation_rows_previously_loaded_by_calibration"
            )
            is False
        ),
        treatment_replay_dose_020=(
            len(alpha_checks) == len(base.SEEDS)
            and all(item["passed"] for item in alpha_checks.values())
        ),
        finite_terminal_endpoints=(
            len(endpoints) == 2 * len(base.SEEDS)
            and all(
                math.isfinite(value)
                for endpoint in endpoints.values()
                for value in endpoint.values()
            )
        ),
    )
    for key, passed in checks.items():
        if not passed:
            errors.append(f"failed check: {key}")
    payload.update(
        schema="pantry-support-retention-final-audit-v1",
        checks=checks,
        endpoints=endpoints,
        alpha_checks=alpha_checks,
        errors=sorted(set(errors)),
    )
    payload["status"] = "pass" if not payload["errors"] else "fail"
    payload["decision"] = (
        "pantry_support_retention_final_eligible"
        if not payload["errors"]
        else "pantry_support_retention_final_stopped"
    )
    _atomic(path, payload)


base.tree_hash = tree_hash
base.json.loads = loads
base.atomic = atomic


if __name__ == "__main__":
    base.main()
