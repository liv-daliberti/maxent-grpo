#!/usr/bin/env python3
"""Bind prior Falcon Python admission/replay telemetry without outcomes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "var/artifacts/e106_prior_falcon_python_bootstrap_audit.json"
RUNS = {
    "plain_grpo": ROOT
    / "var/checkpoints/e95_falcon1b_python_factors_grpo_s55/"
    "debug_job30516430/train_metrics.jsonl",
    "replay_semantic": ROOT
    / "var/data/xdr_falcon3_1b_instruct_verified_replay_semantic_maxent_"
    "e82_falcon_semantic_maxent_python_semantic_s55/debug_job30374918/"
    "train_metrics.jsonl",
}
FIELDS = (
    "train/online_canonical_bank_size_after_mean",
    "train/online_canonical_eligible_fraction",
    "train/canonical_replay_available_groups",
    "train/canonical_replay_applied_score_gradient_l2",
    "train/semantic_shannon_success_conditioned_signed_eligible_fraction",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(path: Path) -> dict[str, Any]:
    maxima = {field: 0.0 for field in FIELDS}
    last_step = -1
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            last_step = max(last_step, int(row.get("trainer/global_step", -1)))
            for field in FIELDS:
                value = row.get(field)
                if isinstance(value, (int, float)):
                    maxima[field] = max(maxima[field], float(value))
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest(path),
        "rows": rows,
        "last_step": last_step,
        "maxima": maxima,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    reports = {name: summarize(path) for name, path in RUNS.items()}
    plain = reports["plain_grpo"]["maxima"]
    replay = reports["replay_semantic"]["maxima"]
    violations: list[str] = []
    if plain["train/online_canonical_bank_size_after_mean"] <= 0.0:
        violations.append("Falcon plain-GRPO never populated the verified bank")
    if plain["train/online_canonical_eligible_fraction"] <= 0.0:
        violations.append("Falcon plain-GRPO admitted no verified rollout")
    for field, label in (
        ("train/online_canonical_bank_size_after_mean", "bank"),
        ("train/online_canonical_eligible_fraction", "verified rollout"),
        ("train/canonical_replay_available_groups", "replay group"),
        ("train/canonical_replay_applied_score_gradient_l2", "replay gradient"),
        (
            "train/semantic_shannon_success_conditioned_signed_eligible_fraction",
            "semantic eligibility",
        ),
    ):
        if replay[field] <= 0.0:
            violations.append(f"Falcon replay+semantic run had no {label}")
    payload = {
        "schema": "e106_prior_falcon_python_bootstrap_audit_v1",
        "passed": not violations,
        "scope": "prior analyzed Falcon Python train telemetry only",
        "fields_read": ["trainer/global_step", *FIELDS],
        "evaluation_outcomes_read": False,
        "post_e104_or_e106_update_outcomes_inspected": False,
        "pointmaze": "excluded",
        "runs": reports,
        "interpretation": (
            "Falcon's stochastic rollout surface can already seed verified "
            "Python support and drive replay; E106 need not broaden its parser."
        ),
        "violations": violations,
    }
    atomic_json(OUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
