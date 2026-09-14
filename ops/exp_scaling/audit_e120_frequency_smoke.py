#!/usr/bin/env python3
"""Fail-closed E120 smoke audit over persisted per-update telemetry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REQUIRED = {
    "canonical_replay_key_weighting_frequency": 1.0,
    "canonical_replay_frequency_count_fresh_only": 1.0,
    "canonical_replay_frequency_count_from_replay": 0.0,
    "canonical_replay_frequency_count_from_proposals": 0.0,
    "canonical_replay_score_passes": 2.0,
    "canonical_replay_alpha_used": 0.1,
    "canonical_replay_objective_scale": 1.0 / 16.0,
    "canonical_replay_global_groups_per_step": 1.0,
}


def suffix_value(record: dict[str, object], suffix: str) -> float | None:
    matches = [value for key, value in record.items() if key.endswith(suffix)]
    if not matches:
        return None
    value = float(matches[0])
    return value if math.isfinite(value) else None


def resolve_metrics(run_dir: Path) -> Path:
    """Resolve metrics without escaping the submitted stable run directory."""

    stable_root = run_dir.resolve()
    direct = stable_root / "train_metrics.jsonl"
    if direct.is_file():
        return direct
    completion = stable_root / "TRAINING_COMPLETE.json"
    if not completion.is_file():
        raise SystemExit(f"E120 smoke lacks {direct} and {completion}")
    try:
        payload = json.loads(completion.read_text(encoding="utf-8"))
        terminal_attempt = Path(str(payload["terminal_attempt"])).resolve()
        terminal_attempt.relative_to(stable_root)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"E120 smoke has invalid terminal attempt receipt: {error}"
        ) from error
    metrics = terminal_attempt / "train_metrics.jsonl"
    if not metrics.is_file():
        raise SystemExit(f"E120 smoke lacks {metrics}")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    metrics = resolve_metrics(args.run_dir)

    rows = [
        json.loads(line)
        for line in metrics.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    replay_rows = [
        row
        for row in rows
        if suffix_value(row, "canonical_replay_key_weighting_frequency")
        is not None
    ]
    if not replay_rows:
        raise SystemExit("E120 smoke produced no replay telemetry row")

    for row in replay_rows:
        for suffix, expected in REQUIRED.items():
            value = suffix_value(row, suffix)
            if value is None or not math.isclose(
                value, expected, rel_tol=0.0, abs_tol=1e-7
            ):
                raise SystemExit(
                    f"E120 smoke telemetry mismatch for {suffix}: "
                    f"{value!r} != {expected!r}"
                )
        minimum = suffix_value(
            row, "canonical_replay_fresh_observation_count_min"
        )
        if minimum is None or minimum < 1.0:
            raise SystemExit("E120 smoke used a non-fresh replay key")
        target_sum = suffix_value(row, "canonical_replay_target_weight_sum")
        modes = suffix_value(row, "canonical_replay_actuator_modes")
        if target_sum is None or modes is None or not math.isclose(
            target_sum, modes, rel_tol=0.0, abs_tol=1e-5
        ):
            raise SystemExit("E120 smoke target weights do not preserve bank budget")
        dynamic = [
            key
            for key in row
            if "canonical_replay_target_weight_row_" in key
        ]
        counts = [
            key for key in row if "canonical_replay_fresh_count_row_" in key
        ]
        outcomes = [
            key
            for key in row
            if "canonical_replay_outcome_fingerprint_row_" in key
        ]
        if not dynamic or not (len(dynamic) == len(counts) == len(outcomes)):
            raise SystemExit("E120 smoke lacks aligned per-key telemetry")

    receipt = {
        "schema": "e120_frequency_weighting_smoke_audit_v1",
        "run_dir": str(args.run_dir),
        "metrics": str(metrics),
        "metric_rows": len(rows),
        "replay_rows": len(replay_rows),
        "required": REQUIRED,
        "passed": True,
        "outcomes_inspected": False,
    }
    path = args.run_dir / "e120_smoke_audit.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"[e120-smoke-audit] passed replay_rows={len(replay_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

