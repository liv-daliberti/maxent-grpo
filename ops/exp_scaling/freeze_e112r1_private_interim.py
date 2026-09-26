#!/usr/bin/env python3
"""Freeze the outcome-blind membership for a private E112-R1 interim look."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "var/artifacts/e112r1_verified_support_discovery_full_three_scale_jobs.json"
PROTOCOL = ROOT / "paper/preregistration/e112r1_user_requested_private_interim_unblinding_20260820.md"
OUTPUT = ROOT / "var/artifacts/e112r1_private_interim_unblinding_freeze.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    output = args.output.resolve()
    protocol = args.protocol.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite frozen interim membership: {output}")
    if not protocol.is_file():
        raise SystemExit(f"private interim protocol is absent: {protocol}")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    if ledger.get("released") is not True or len(ledger.get("runs", [])) != 75:
        raise SystemExit("E112-R1 ledger is not the released 75-cell cohort")
    target = int(ledger.get("target_steps", -1))
    if target != 3072:
        raise SystemExit(f"unexpected E112-R1 target: {target}")

    cells: list[dict[str, object]] = []
    for run in ledger["runs"]:
        run_dir = Path(str(run["run_dir"]))
        marker = run_dir / "TRAINING_COMPLETE.json"
        if not marker.is_file():
            continue
        payload = json.loads(marker.read_text(encoding="utf-8"))
        terminal_step = payload.get("terminal_step")
        if not isinstance(terminal_step, int) or terminal_step < target:
            raise SystemExit(f"invalid completion marker: {marker}")
        cells.append(
            {
                "scale": str(run["scale"]),
                "domain": str(run["domain"]),
                "seed": int(run["seed"]),
                "arm": str(run["arm"]),
                "job_id": int(run["job_id"]),
                "run_dir": relative(run_dir),
                "terminal_step": terminal_step,
                "completion_marker": relative(marker),
                "completion_marker_sha256": sha256(marker),
                "paired_replay": run["paired_replay"],
            }
        )
    cells.sort(key=lambda row: (str(row["scale"]), str(row["domain"]), int(row["seed"])))
    if not cells:
        raise SystemExit("no terminal E112-R1 cells are available")

    record = {
        "schema": "e112r1_private_interim_unblinding_freeze_v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "all valid E112-R1 completion markers present at freeze time",
        "user_requested_early_unblinding": True,
        "confirmatory_outcome_blindness_broken": True,
        "efficacy_fields_read_before_freeze": False,
        "campaign_mutation_allowed_from_interim": False,
        "paper_efficacy_output_allowed": False,
        "pointmaze": "excluded",
        "target_steps": target,
        "terminal_cells": len(cells),
        "ledger": relative(LEDGER),
        "ledger_sha256": sha256(LEDGER),
        "protocol": relative(protocol),
        "protocol_sha256": sha256(protocol),
        "cells": cells,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[e112r1-private-freeze] terminal_cells={len(cells)} output={output}")


if __name__ == "__main__":
    main()
