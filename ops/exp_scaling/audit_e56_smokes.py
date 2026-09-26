#!/usr/bin/env python3
"""Fail-closed runtime audit for E56's two engineering smokes."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARM = "open_set_split_canonical"
SEED = 9056
DIRECT_BASE_ALPHA = 0.000075
CONTROLLER_WARMUP = 64
PREFIXES = {
    "graph_coloring": "e56_open_set_split_smoke_graph_telemetry_v2",
    "python_factor": "e56_open_set_split_smoke_python_telemetry_v2",
}
# One naturally multi-success graph group is enough to execute the shared
# open-set sensor/controller code. Python's one-pass smoke separately proves
# that validator-produced executable-program keys enter replay; requiring two
# successes for the same previously unseen prompt would make that domain's
# engineering gate a stochastic quality test rather than a mechanics test.
SEMANTIC_OBSERVATION_REQUIRED = frozenset({"graph_coloring"})
REPLAY_REQUIRED = (
    "train/canonical_replay_actuator_loss",
    "train/canonical_replay_balance_loss",
    "train/canonical_replay_objective_scale",
    "train/canonical_replay_reward_estimator_scale",
    "train/canonical_replay_mass_score_gradient_sum",
    "train/canonical_replay_balance_score_gradient_sum",
    "train/canonical_replay_mass_score_gradient_l2",
    "train/canonical_replay_balance_score_gradient_l2",
    "train/canonical_replay_applied_score_gradient_l2",
    "train/canonical_replay_mass_alpha_used",
    "train/canonical_replay_balance_alpha_used",
)
SEMANTIC_PREFIX = (
    "train/semantic_shannon_success_conditioned_signed_"
)


def _step(row: dict[str, Any]) -> Any:
    """Return the trainer's canonical step with legacy aliases as fallbacks."""
    for key in (
        "trainer/global_step",
        "misc/global_step",
        "train/global_step",
        "global_step",
    ):
        value = row.get(key)
        if _finite(value):
            return value
    return None


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _run_dir(prefix: str) -> Path | None:
    pattern = (
        ROOT
        / "var/data"
        / (
            "xdr_qwen25_0p5b_instruct_"
            f"{ARM}_{prefix}_{ARM}_s{SEED}"
        )
    )
    matches = [Path(path) for path in glob.glob(str(pattern))]
    if len(matches) > 1:
        raise RuntimeError(f"multiple E56 smoke run directories match {prefix}")
    return matches[0] if matches else None


def _rows(run_dir: Path) -> list[dict[str, Any]]:
    rows_by_step: dict[int, dict[str, Any]] = {}
    paths = sorted(run_dir.glob("debug_*/train_metrics.jsonl"))
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                raw_step = _step(row)
                if not _finite(raw_step):
                    continue
                rows_by_step[int(raw_step)] = row
    return [rows_by_step[step] for step in sorted(rows_by_step)]


def _audit_domain(domain: str, prefix: str) -> dict[str, Any]:
    run_dir = _run_dir(prefix)
    if run_dir is None:
        return {
            "domain": domain,
            "status": "pending",
            "violations": [],
            "run_dir": None,
        }
    rows = _rows(run_dir)
    complete = (run_dir / "TRAINING_COMPLETE.json").is_file()
    violations: list[str] = []
    if not rows:
        if complete:
            violations.append("training completed without readable metrics")
        return {
            "domain": domain,
            "status": "fail" if violations else "running",
            "violations": violations,
            "run_dir": str(run_dir),
            "train_points": 0,
        }

    for row in rows:
        for key, value in row.items():
            if (
                key.startswith(
                    (
                        "train/canonical_replay_",
                        SEMANTIC_PREFIX,
                        "train/maxent_inverse_",
                    )
                )
                and isinstance(value, (int, float))
                and not _finite(value)
            ):
                violations.append(f"{key} is non-finite")
        for key, value in row.items():
            if (
                key.startswith("train/canonical_replay_")
                and key.endswith(("_nan", "_inf"))
                and _finite(value)
                and float(value) != 0.0
            ):
                violations.append(f"{key}={value}, expected zero")

    replay_rows = [
        row
        for row in rows
        if _finite(row.get("train/canonical_replay_actuator_groups"))
        and float(row["train/canonical_replay_actuator_groups"]) > 0
    ]
    semantic_rows = [
        row
        for row in rows
        if _finite(
            row.get(
                f"{SEMANTIC_PREFIX}open_set_inverse_adaptation_active"
            )
        )
    ]
    direct_rows = [
        row
        for row in rows
        if _finite(row.get("train/maxent_inverse_observations"))
    ]
    semantic_observed = [
        row
        for row in semantic_rows
        if _finite(row.get(f"{SEMANTIC_PREFIX}open_set_observations"))
        and float(row[f"{SEMANTIC_PREFIX}open_set_observations"]) > 0
    ]
    if complete and not replay_rows:
        violations.append("completed smoke never activated canonical replay")
    if complete and not direct_rows:
        violations.append("completed smoke never emitted direct-controller state")
    if (
        complete
        and domain in SEMANTIC_OBSERVATION_REQUIRED
        and not semantic_observed
    ):
        violations.append(
            "completed smoke never obtained an eligible open-set observation"
        )

    for row in replay_rows:
        missing = [key for key in REPLAY_REQUIRED if not _finite(row.get(key))]
        if missing:
            violations.append(f"replay row is missing/non-finite: {missing}")
            continue
        checks = {
            "mass raw score-gradient sum": (
                float(row["train/canonical_replay_mass_score_gradient_sum"]),
                -1.0,
                2e-6,
            ),
            "balance raw score-gradient sum": (
                float(row["train/canonical_replay_balance_score_gradient_sum"]),
                0.0,
                2e-6,
            ),
            "one-pseudo-rollout measure": (
                float(row["train/canonical_replay_objective_scale"]),
                1.0 / 16.0,
                1e-8,
            ),
            "Dr.GRPO estimator scale": (
                float(row["train/canonical_replay_reward_estimator_scale"]),
                15.0 / 16.0,
                1e-8,
            ),
        }
        for label, (actual, expected, tolerance) in checks.items():
            if not math.isclose(actual, expected, abs_tol=tolerance):
                violations.append(
                    f"{label}={actual}, expected {expected}"
                )
        for key in (
            "train/canonical_replay_projection_active",
            "train/canonical_replay_mass_projection_active",
            "train/canonical_replay_alpha_projection_active",
            "train/canonical_replay_gold_support_feedback",
        ):
            if _finite(row.get(key)) and float(row[key]) != 0.0:
                violations.append(f"{key}={row[key]}, expected zero")

    for row in direct_rows:
        required = {
            key: row.get(key)
            for key in (
                "train/maxent_conditional_token_entropy",
                "train/maxent_inverse_observed_entropy",
                "train/maxent_inverse_entropy_ema",
                "train/maxent_inverse_multiplier",
                "train/maxent_alpha_used",
                "train/maxent_inverse_next_alpha",
                "train/maxent_inverse_observations",
                "train/maxent_inverse_projection_active",
            )
        }
        missing = [key for key, value in required.items() if not _finite(value)]
        if missing:
            violations.append(
                f"direct controller row is missing/non-finite: {missing}"
            )
            continue
        value = {key: float(item) for key, item in required.items()}
        entropy = value["train/maxent_conditional_token_entropy"]
        observed = value["train/maxent_inverse_observed_entropy"]
        multiplier = value["train/maxent_inverse_multiplier"]
        next_alpha = value["train/maxent_inverse_next_alpha"]
        observations = value["train/maxent_inverse_observations"]
        if not math.isclose(entropy, observed, rel_tol=1e-6, abs_tol=1e-8):
            violations.append("direct objective/controller sensor mismatch")
        if value["train/maxent_inverse_projection_active"] != 0.0:
            violations.append("direct entropy projection became active")
        if (
            value["train/maxent_alpha_used"] <= 0.0
            or next_alpha <= 0.0
        ):
            violations.append("direct entropy coefficient became nonpositive")
        if observations <= CONTROLLER_WARMUP:
            if not math.isclose(
                multiplier, 1.0, rel_tol=1e-6, abs_tol=1e-8
            ) or not math.isclose(
                next_alpha,
                DIRECT_BASE_ALPHA,
                rel_tol=1e-6,
                abs_tol=1e-9,
            ):
                violations.append("direct entropy coefficient drifted in warmup")
        else:
            reference = row.get("train/maxent_inverse_reference_entropy")
            ema = value["train/maxent_inverse_entropy_ema"]
            if not _finite(reference) or float(reference) <= 0.0 or ema <= 0.0:
                violations.append("direct inverse controller lacks positive state")
                continue
            expected_multiplier = float(reference) / ema
            if not math.isclose(
                multiplier,
                expected_multiplier,
                rel_tol=2e-5,
                abs_tol=1e-8,
            ) or not math.isclose(
                next_alpha,
                DIRECT_BASE_ALPHA * expected_multiplier,
                rel_tol=2e-5,
                abs_tol=1e-9,
            ):
                violations.append("direct unbounded inverse arithmetic mismatch")

    for row in semantic_rows:
        for suffix in (
            "open_set_inverse_adaptation_active",
            "open_set_coefficient_used",
            "open_set_next_coefficient",
            "open_set_projection_active",
        ):
            key = f"{SEMANTIC_PREFIX}{suffix}"
            if not _finite(row.get(key)):
                violations.append(f"{key} is missing/non-finite")
        projection = row.get(f"{SEMANTIC_PREFIX}open_set_projection_active")
        if _finite(projection) and float(projection) != 0.0:
            violations.append("open-set semantic projection became active")
        active = row.get(
            f"{SEMANTIC_PREFIX}open_set_inverse_adaptation_active"
        )
        if _finite(active) and float(active) != 1.0:
            violations.append("open-set inverse semantic path is inactive")
        cap = row.get(f"{SEMANTIC_PREFIX}advantage_cap")
        if _finite(cap) and float(cap) != 0.0:
            violations.append("open-set semantic advantage cap is nonzero")

    status = "fail" if violations else "pass" if complete else "running"
    latest = rows[-1]
    return {
        "domain": domain,
        "status": status,
        "violations": sorted(set(violations)),
        "run_dir": str(run_dir),
        "train_points": len(rows),
        "latest_step": _step(latest),
        "replay_activations": len(replay_rows),
        "semantic_observed_points": len(semantic_observed),
        "semantic_observation_required": (
            domain in SEMANTIC_OBSERVATION_REQUIRED
        ),
        "complete": complete,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "var/artifacts/e56_smoke_audit_latest.json",
    )
    args = parser.parse_args()
    domains = [
        _audit_domain(domain, prefix)
        for domain, prefix in PREFIXES.items()
    ]
    status = (
        "fail"
        if any(item["status"] == "fail" for item in domains)
        else "pass"
        if all(item["status"] == "pass" for item in domains)
        else "in_progress"
    )
    payload = {
        "schema": "e56_open_set_split_smoke_audit_v1",
        "status": status,
        "domains": domains,
        "violations": [
            f"{item['domain']}: {violation}"
            for item in domains
            for violation in item["violations"]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        f"[e56-smoke-audit] status={status} "
        + " ".join(
            f"{item['domain']}={item['status']}" for item in domains
        )
    )


if __name__ == "__main__":
    main()
