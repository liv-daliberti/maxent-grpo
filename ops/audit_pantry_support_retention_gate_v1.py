#!/usr/bin/env python3
"""Audit PantryPlan's fresh-split support-retention repair pair."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import audit_pantry_support_mask_paired_smoke_v1 as base


base.SEED = 76401
base.PREFIX = "pprepair_support_retention_v1"
base.UPDATES = 96
_tree_hash = base.tree_hash


def _repair_tree_hash(path: Path) -> str:
    if path.as_posix().endswith("var/data/pantry_plan_modebench_v2"):
        path = path.parent / "pantry_plan_modebench_v3_repair"
    return _tree_hash(path)


base.tree_hash = _repair_tree_hash


def _paths() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parsed, _ = parser.parse_known_args()
    return parsed


def _endpoint(row: dict[str, object], anchor: str) -> dict[str, float]:
    return {
        f"{anchor}_{metric}": float(row[f"eval/multi_answer/{metric}"])
        for metric in (
            "sampled_mean_at_8",
            "sampled_any_correct_at_8",
            "sampled_distinct_correct_at_8",
        )
    }


def strengthen(parsed: argparse.Namespace) -> None:
    payload = json.loads(parsed.output.read_text())
    rows = list(csv.DictReader(parsed.manifest.open(), delimiter="\t"))
    errors = list(payload.get("errors", []))
    endpoints: dict[str, dict[str, float]] = {}
    alpha_checks: dict[str, dict[str, float | int | bool]] = {}
    for row in rows:
        arm = row["arm"]
        job_id = int(row["job_id"])
        directory = base.run_dir(parsed.repo_root, row["run_stamp"], job_id)
        if directory is None:
            errors.append(f"{arm}/j{job_id}: unique repair run directory absent")
            continue
        metric_rows = [
            json.loads(line)
            for line in (directory / "train_metrics.jsonl").read_text().splitlines()
            if line.strip()
        ]
        evaluations = [
            item
            for item in metric_rows[:-1]
            if "eval/multi_answer/sampled_distinct_correct_at_8" in item
        ]
        if len(evaluations) != 13:
            errors.append(f"{arm}/j{job_id}: exact 13 development coordinates absent")
            continue
        endpoints[arm] = {
            **_endpoint(evaluations[0], "initial"),
            **_endpoint(evaluations[-1], "terminal"),
        }
        if arm == base.TREATMENT:
            values = [
                float(item.get("train/canonical_replay_alpha_used", float("nan")))
                for item in metric_rows[1:97]
            ]
            passed = all(
                math.isfinite(value)
                and math.isclose(value, 0.20, rel_tol=0.05, abs_tol=0.005)
                for value in values
            )
            alpha_checks[arm] = {
                "passed": passed,
                "count": len(values),
                "minimum": min(values),
                "maximum": max(values),
            }
            if not passed:
                errors.append(f"{arm}/j{job_id}: replay alpha differs from 0.20 repair dose")

    checks = dict(payload.get("checks", {}))
    if set(endpoints) == {base.CONTROL, base.TREATMENT}:
        control = endpoints[base.CONTROL]
        treatment = endpoints[base.TREATMENT]
        checks.update(
            terminal_distinct_exceeds_control=(
                treatment["terminal_sampled_distinct_correct_at_8"]
                > control["terminal_sampled_distinct_correct_at_8"]
            ),
            terminal_pass_not_below_control=(
                treatment["terminal_sampled_any_correct_at_8"]
                >= control["terminal_sampled_any_correct_at_8"]
            ),
            terminal_mean_at_least_90pct_control=(
                treatment["terminal_sampled_mean_at_8"]
                >= 0.90 * control["terminal_sampled_mean_at_8"]
            ),
            terminal_distinct_retains_60pct_initial=(
                treatment["terminal_sampled_distinct_correct_at_8"]
                >= 0.60 * treatment["initial_sampled_distinct_correct_at_8"]
            ),
            treatment_replay_alpha_020=bool(
                alpha_checks.get(base.TREATMENT, {}).get("passed", False)
            ),
        )
    else:
        checks["terminal_retention_gate"] = False
    for key, passed in checks.items():
        if not passed:
            errors.append(f"failed check: {key}")

    payload.update(
        schema="pantry-support-retention-repair-audit-v1",
        checks=checks,
        endpoints=endpoints,
        alpha_checks=alpha_checks,
        errors=sorted(set(errors)),
    )
    payload["status"] = "pass" if not payload["errors"] else "fail"
    payload["decision"] = (
        "eligible_for_pantry_support_retention_final"
        if not payload["errors"]
        else "pantry_support_retention_repair_stopped"
    )
    base.atomic(parsed.output, payload)
    print(json.dumps({"status": payload["status"], "errors": len(payload["errors"])}))
    if payload["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parsed_paths = _paths()
    base.main()
    strengthen(parsed_paths)

