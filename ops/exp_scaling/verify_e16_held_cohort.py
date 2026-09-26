#!/usr/bin/env python3
"""Verify every E16 job is submitted and user-held before cohort release."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import subprocess

try:
    from .e16_canonical_plan import ARMS, campaign_plan
except ImportError:
    from e16_canonical_plan import ARMS, campaign_plan


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or set(rows[0]) != {"arm", "seed", "job_id", "run_stamp"}:
        raise ValueError(f"malformed E16 manifest: {path}")
    return rows


def verify_cohort(
    *, stage: str, graph_manifest: Path, countdown_manifest: Path
) -> dict[str, object]:
    plan = campaign_plan(stage)
    expected = {
        (task, arm, str(seed))
        for task in ("graph_coloring", "countdown")
        for arm in ARMS
        for seed in plan["seeds"]
    }
    observed: dict[tuple[str, str, str], tuple[str, str]] = {}
    for task, path in (
        ("graph_coloring", graph_manifest),
        ("countdown", countdown_manifest),
    ):
        prefix = str(plan["tasks"][task]["prefix"])
        for row in _manifest_rows(path):
            key = (task, str(row["arm"]), str(row["seed"]))
            if key in observed:
                raise ValueError(f"duplicate E16 cohort cell: {key!r}")
            expected_stamp = f"{prefix}_{row['arm']}_s{row['seed']}"
            job_id = str(row["job_id"])
            if not re.fullmatch(r"[1-9][0-9]*", job_id):
                raise ValueError(f"non-numeric E16 job ID: {job_id!r}")
            if row["run_stamp"] != expected_stamp:
                raise ValueError(f"wrong E16 run stamp for {key!r}")
            observed[key] = (job_id, expected_stamp)
    if set(observed) != expected:
        raise ValueError(
            f"E16 held cohort incomplete: missing={sorted(expected-set(observed))!r} "
            f"extra={sorted(set(observed)-expected)!r}"
        )
    job_ids = [observed[key][0] for key in sorted(observed)]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("E16 held cohort reuses a Slurm job ID")
    for job_id in job_ids:
        completed = subprocess.run(
            ["scontrol", "show", "job", "-o", job_id],
            check=True,
            capture_output=True,
            text=True,
        )
        output = completed.stdout.strip()
        if "JobState=PENDING" not in output or "Reason=JobHeldUser" not in output:
            raise ValueError(f"E16 job {job_id} is not pending on a user hold")
    return {
        "cell_count": len(observed),
        "job_ids": job_ids,
        "release_argument": ",".join(job_ids),
        "schema": "e16_held_cohort_v1",
        "stage": stage,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--graph-manifest", type=Path, required=True)
    parser.add_argument("--countdown-manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify_cohort(
            stage=args.stage,
            graph_manifest=args.graph_manifest,
            countdown_manifest=args.countdown_manifest,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as error:
        raise SystemExit(f"E16 held cohort rejected; no jobs released: {error}") from error
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
