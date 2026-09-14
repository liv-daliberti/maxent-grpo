#!/usr/bin/env python3
"""Fail-closed mechanism audit for the E98-R1 sparse RLEP learner smoke."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as sink:
            json.dump(payload, sink, indent=2, sort_keys=True)
            sink.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def audit_smoke(root: Path, *, expected_terminal_step: int = 32) -> dict[str, Any]:
    root = root.resolve()
    receipts = sorted(root.glob("**/TRAINING_COMPLETE.json"))
    metrics_paths = sorted(root.glob("**/train_metrics.jsonl"))
    if len(receipts) != 1:
        raise ValueError(
            f"expected one TRAINING_COMPLETE.json under {root}, found {len(receipts)}"
        )
    if len(metrics_paths) != 1:
        raise ValueError(
            f"expected one train_metrics.jsonl under {root}, found {len(metrics_paths)}"
        )
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    terminal_step = int(receipt.get("terminal_step", 0))
    if receipt.get("schema") != "oat_zero_training_complete_v1":
        raise ValueError("E98-R1 smoke has the wrong completion schema")
    if terminal_step < expected_terminal_step:
        raise ValueError(
            f"E98-R1 smoke stopped at step {terminal_step}, expected "
            f"{expected_terminal_step}"
        )

    rows: list[dict[str, Any]] = []
    for line in metrics_paths[0].read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if "train/rlep_replay_eligible" in row:
            rows.append(row)
    if not rows:
        raise ValueError("E98-R1 smoke has no sparse RLEP telemetry")

    eligibility = {int(round(float(row["train/rlep_replay_eligible"]))) for row in rows}
    if eligibility != {0, 1}:
        raise ValueError(
            f"E98-R1 smoke did not exercise both replay branches: {eligibility}"
        )
    replay_rows = {
        (
            int(round(float(row["train/rlep_replay_eligible"]))),
            int(round(float(row["train/rlep_replay_rows"]))),
        )
        for row in rows
    }
    if (0, 0) not in replay_rows or (1, 2) not in replay_rows:
        raise ValueError(f"E98-R1 smoke replay dose drift: {sorted(replay_rows)}")
    forbidden = sorted(
        pair for pair in replay_rows if pair not in {(0, 0), (1, 2)}
    )
    if forbidden:
        raise ValueError(f"E98-R1 smoke has unregistered replay doses: {forbidden}")

    finite_keys = (
        "train/rlep_replay_loss",
        "train/rlep_replay_advantage",
        "train/rlep_mixed_reward_mean",
    )
    observed_finite: dict[str, int] = {}
    for key in finite_keys:
        values = [float(row[key]) for row in rows if key in row]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError(f"E98-R1 smoke lacks finite {key}")
        observed_finite[key] = len(values)

    payload = {
        "schema": "e98r1_sparse_rlep_smoke_complete_v1",
        "audited_at_unix": time.time(),
        "run_root": str(root),
        "terminal_step": terminal_step,
        "telemetry_rows": len(rows),
        "eligibility_values": sorted(eligibility),
        "replay_dose_pairs": [list(pair) for pair in sorted(replay_rows)],
        "finite_key_counts": observed_finite,
        "receipt": str(receipts[0]),
        "metrics": str(metrics_paths[0]),
    }
    atomic_json(root / "E98R1_SMOKE_COMPLETE.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-terminal-step", type=int, default=32)
    args = parser.parse_args()
    payload = audit_smoke(
        args.run_root, expected_terminal_step=args.expected_terminal_step
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
