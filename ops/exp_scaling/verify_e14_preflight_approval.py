#!/usr/bin/env python3
"""Verify that an E14 approval authorizes the current prospective C0 source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from check_e14_preflight import GateError, verify_approval_for_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--expected-source-hash", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = verify_approval_for_source(
            args.approval,
            expected_source_hash=args.expected_source_hash,
            logical_repo_root=args.repo_root,
        )
    except GateError as error:
        raise SystemExit(f"E14 preflight approval rejected: {error}") from error
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
