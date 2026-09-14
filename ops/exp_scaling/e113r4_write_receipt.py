#!/usr/bin/env python3
"""Write an atomic terminal receipt after upstream DAPO exits successfully."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--snapshot", required=True)
    args = parser.parse_args()
    payload = {
        "schema": "e113r4_official_verl_dapo_training_complete_v1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "family": args.family,
        "domain": args.domain,
        "seed": args.seed,
        "total_training_steps": args.steps,
        "runtime_snapshot": args.snapshot,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    target = args.output / "TRAINING_COMPLETE.json"
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as sink:
            json.dump(payload, sink, indent=2, sort_keys=True)
            sink.write("\n")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

