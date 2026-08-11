#!/usr/bin/env python3
"""Derive B2b's per-domain token-entropy targets from the treatment's telemetry.

B2b holds a Dr.GRPO policy at xGRPO's *own* measured token entropy, so the
comparison is not "some entropy bonus" but "the same entropy, spent on tokens
instead of on executed outcomes". The target therefore has to come from the
frozen runs rather than from a guess.

The target has to be stated in the units the controller actually observes, and
the first version of this script got that wrong in a way worth recording. It
converted the measured per-token entropy into a *sequence* target by
multiplying by response length, because the dual controller reads
`maxent_sequence_entropy` by default. Sequence entropy is a sum over generated
tokens, so even a low-entropy eleven-token response already exceeded the
resulting target at step one: the controller could only ever push down, pinned
its coefficient at the floor, and the residual bonus on a *sum* rewarded length
until responses hit the 192-token cap at eight nats per token. The arm measured
nothing.

Under `maxent_objective=conditional_token_mean` the controller instead observes
`maxent_conditional_token_entropy`, a per-token mean, so the treatment's
measured per-token entropy is the target directly with no conversion at all --
and the entropy term becomes a mean rather than a sum, removing the length
incentive that caused the runaway. PantryPlan is the exception: a canonical
action task overrides the objective, and its controller regulates exact
canonical sequence entropy over the 64-mode support, so its target is measured
in those units instead.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import tempfile
from pathlib import Path
from typing import Any

# Window over which the terminal entropy is averaged: the last quarter-pass of
# logged updates, which is where the policy has settled.
TAIL_UPDATES = 200

DOMAIN_PREFIXES = {
    "graph_coloring": "gce71_scale384_05b_12pass",
    "countdown": "cde70_clean_stage_a_05b_12pass",
    "python_factors": "pye70_clean_stage_a_05b_12pass",
    "mathir": "mie70_clean_stage_a_05b_12pass",
    "pantry_plan": "ppe71_scale384_05b_12pass",
}
TREATMENT = "verified_first_global_replay_canonical"
CONTROL = "grpo_compute_matched"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def tail_means(metrics: Path, keys: tuple[str, ...]) -> dict[str, float | None]:
    """Mean of each key over the final logged updates of one run."""
    series: dict[str, list[float]] = {key: [] for key in keys}
    with metrics.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            for key in keys:
                value = record.get(key)
                if value is not None:
                    series[key].append(float(value))
    return {
        key: statistics.fmean(values[-TAIL_UPDATES:]) if values else None
        for key, values in series.items()
    }


def arm_measurements(root: Path, prefix: str, variant: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pattern = f"var/data/xdr_qwen25_0p5b_instruct_{variant}_{prefix}_*_s4[3-7]"
    for run_dir in sorted(root.glob(pattern)):
        seed = int(re.search(r"_s(\d+)$", run_dir.name).group(1))
        for path in glob.glob(str(run_dir / "debug_job*" / "train_metrics.jsonl")):
            measured = tail_means(
                Path(path),
                (
                    "train/entropy",
                    "actor/response_tok_len",
                    "train/canonical_exact_sequence_entropy",
                ),
            )
            if measured["train/entropy"] is None:
                continue
            rows.append(
                {
                    "seed": seed,
                    "token_entropy": measured["train/entropy"],
                    "response_tokens": measured["actor/response_tok_len"],
                    "canonical_entropy": measured[
                        "train/canonical_exact_sequence_entropy"
                    ],
                }
            )
            break
    return rows


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "var" / "artifacts" / "e72_token_entropy_targets.json",
    )
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "schema": "e72_token_entropy_targets_v1",
        "tail_updates": TAIL_UPDATES,
        "units": (
            "controller_target is in the units the dual controller observes: "
            "conditional content-token nats (mean) where maxent_objective is "
            "conditional_token_mean, and exact canonical sequence nats where a "
            "canonical action task overrides the objective"
        ),
        "domains": {},
    }
    problems: list[str] = []
    for domain, prefix in DOMAIN_PREFIXES.items():
        treatment = arm_measurements(root, prefix, TREATMENT)
        control = arm_measurements(root, prefix, CONTROL)
        if len(treatment) < 5 or len(control) < 5:
            problems.append(
                f"{domain}: found {len(treatment)} treatment and {len(control)} "
                "control runs, expected 5 each"
            )
            continue
        token_entropy = statistics.fmean(row["token_entropy"] for row in treatment)
        lengths = [
            row["response_tokens"] for row in treatment if row["response_tokens"]
        ]
        mean_length = statistics.fmean(lengths) if lengths else None
        canonical = [
            row["canonical_entropy"] for row in treatment if row["canonical_entropy"]
        ]
        canonical_entropy = statistics.fmean(canonical) if canonical else None
        payload["domains"][domain] = {
            # What the launcher injects. A canonical action task overrides the
            # per-token objective, so those domains are targeted in canonical
            # units; everywhere else the measured per-token mean is the target
            # as-is, with no conversion.
            # The controller is configured to observe `train/entropy`, so the
            # target is that measurement directly, in every domain. Earlier
            # versions converted into whichever estimator the objective
            # happened to expose; each conversion was a unit error in a
            # different disguise, and the arm measured nothing three times
            # before the estimator was made to match the measurement.
            "controller_target": token_entropy,
            "controller_units": "masked_mean_token_nats_v1",
            "treatment_canonical_entropy": canonical_entropy,
            "treatment_token_entropy": token_entropy,
            "control_token_entropy": statistics.fmean(
                row["token_entropy"] for row in control
            ),
            "treatment_mean_response_tokens": mean_length,

            "treatment_per_seed": sorted(
                (row["seed"], round(row["token_entropy"], 6)) for row in treatment
            ),
        }
    payload["problems"] = problems

    handle, temporary = tempfile.mkstemp(
        prefix=f".{args.output.name}.", dir=args.output.parent
    )
    with os.fdopen(handle, "w", encoding="utf-8") as sink:
        json.dump(payload, sink, indent=2, sort_keys=True)
        sink.write("\n")
    os.replace(temporary, args.output)

    for domain, entry in payload["domains"].items():
        print(
            f"  {domain:16} treatment {entry['treatment_token_entropy']:.4f} "
            f"vs control {entry['control_token_entropy']:.4f} per token; "
            f"target {entry['controller_target']:.4f} "
            f"[{entry['controller_units'].split('_nats')[0]}]"
        )
    if problems:
        for problem in problems:
            print(f"  problem: {problem}")
        return 1
    print(f"[e72-entropy-targets] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
