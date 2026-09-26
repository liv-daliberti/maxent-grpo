#!/usr/bin/env python3
"""Fail-closed mechanism audit for the E100 Falcon sparse-RLEP smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_e98r1_sparse_rlep_smoke as shared  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-terminal-step", type=int, default=32)
    args = parser.parse_args()
    payload = shared.audit_smoke(
        args.run_root, expected_terminal_step=args.expected_terminal_step
    )
    # The shared checker writes its historical receipt as part of its atomic
    # validation. Add a cohort-specific receipt so E100 never depends on an
    # ambiguously named marker.
    payload = dict(payload)
    payload["schema"] = "e100_sparse_rlep_smoke_complete_v1"
    shared.atomic_json(args.run_root.resolve() / "E100_SMOKE_COMPLETE.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
