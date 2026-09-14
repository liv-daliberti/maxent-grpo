#!/usr/bin/env python3
"""Fail-closed audit for one immutable E98 RLEP experience pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from oat_drgrpo.rlep import RLEPExperiencePool


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def audit_pool(
    root: Path, *, expected_prompts: int, allow_sparse: bool = False
) -> dict[str, Any]:
    root = root.resolve()
    completion = sorted(root.glob("**/EVAL_ONLY_COMPLETE.json"))
    sidecars = sorted(root.glob("**/eval_mode_coverage_draws.jsonl"))
    if len(completion) != 1:
        raise ValueError(
            f"expected one EVAL_ONLY_COMPLETE.json under {root}, found {len(completion)}"
        )
    if len(sidecars) != 1:
        raise ValueError(
            f"expected one eval_mode_coverage_draws.jsonl under {root}, found {len(sidecars)}"
        )
    marker = json.loads(completion[0].read_text(encoding="utf-8"))
    required = {
        "eval_mode_coverage_k": 16,
        "eval_mode_coverage_temperature": 0.7,
        "eval_mode_coverage_top_p": 0.95,
        "eval_mode_coverage_draws": 4,
        "test_split": "multi_answer",
    }
    mismatch = {
        key: (marker.get(key), value)
        for key, value in required.items()
        if marker.get(key) != value
    }
    if mismatch:
        raise ValueError(f"RLEP collection completion marker drift: {mismatch}")

    pool = RLEPExperiencePool.from_directory(root, allow_sparse=allow_sparse)
    diagnostics = pool.diagnostics
    if diagnostics.prompts != expected_prompts:
        raise ValueError(
            f"RLEP pool covers {diagnostics.prompts} prompts, expected {expected_prompts}"
        )
    if allow_sparse and diagnostics.eligible_prompts <= 0:
        raise ValueError("sparse RLEP pool has no replay-eligible prompt")
    if not allow_sparse and diagnostics.minimum_trajectories_per_prompt < 2:
        raise ValueError("RLEP pool contains an ineligible prompt")
    payload = {
        "schema": (
            "e98r1_sparse_rlep_pool_complete_v1"
            if allow_sparse else "e98_rlep_pool_complete_v1"
        ),
        "audited_at_unix": time.time(),
        "pool_root": str(root),
        "completion_marker": str(completion[0]),
        "completion_marker_sha256": digest(completion[0]),
        "sidecar": str(sidecars[0]),
        "sidecar_sha256": digest(sidecars[0]),
        "prompts": diagnostics.prompts,
        "trajectories": diagnostics.trajectories,
        "minimum_trajectories_per_prompt": diagnostics.minimum_trajectories_per_prompt,
        "maximum_trajectories_per_prompt": diagnostics.maximum_trajectories_per_prompt,
        "eligible_prompts": diagnostics.eligible_prompts,
        "ineligible_prompts": diagnostics.ineligible_prompts,
        "eligible_fraction": diagnostics.eligible_prompts / diagnostics.prompts,
        "replay_count_on_eligible_prompt": 2,
        "fallback_on_ineligible_prompt": "unchanged_16_row_drgrpo_update",
        "sample_count": 16,
        "draws": 4,
        "temperature": 0.7,
        "top_p": 0.95,
    }
    receipt = (
        "RLEP_SPARSE_POOL_COMPLETE.json" if allow_sparse else "RLEP_POOL_COMPLETE.json"
    )
    atomic_json(root / receipt, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-root", type=Path, required=True)
    parser.add_argument("--expected-prompts", type=int, default=384)
    parser.add_argument("--allow-sparse", action="store_true")
    args = parser.parse_args()
    payload = audit_pool(
        args.pool_root,
        expected_prompts=args.expected_prompts,
        allow_sparse=args.allow_sparse,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
