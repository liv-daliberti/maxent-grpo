#!/usr/bin/env python3
"""Cheap C/P/F runtime smoke for E117 repaired same-plumbing releases."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Any

import torch


PREFIX = "train/semantic_shannon_success_conditioned_verified_support"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
            json.dump(payload, sink, indent=2, sort_keys=True)
            sink.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def semantic_args(coefficient: float) -> SimpleNamespace:
    return SimpleNamespace(
        semantic_shannon_coef=coefficient,
        semantic_shannon_allow_zero_coefficient_control=True,
        semantic_shannon_surprisal_clip=5.0,
        semantic_shannon_pseudocount=1.0,
        semantic_shannon_quality_gated_advantage=False,
        semantic_shannon_quality_gated_cap=0.05,
        semantic_shannon_success_conditioned_signed_advantage=False,
        semantic_shannon_success_conditioned_signed_cap=0.05,
        semantic_shannon_success_conditioned_group_centered_advantage=False,
        semantic_shannon_success_conditioned_verified_support_advantage=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot = args.snapshot_root.resolve()
    identity = snapshot / "SNAPSHOT_IDENTITY.json"
    if not identity.is_file():
        raise SystemExit(f"snapshot identity is absent: {identity}")
    sys.path.insert(0, str(snapshot / "src"))

    from oat_drgrpo.learner.init import build_semantic_shannon_tracker
    from oat_drgrpo.semantic_shannon import (
        SemanticShannonSuccessConditionedSignedDiagnostics,
        add_semantic_shannon_separate_advantage,
        success_conditioned_semantic_metric_values,
    )
    import oat_drgrpo.learner.init as init_module
    import oat_drgrpo.semantic_shannon as semantic_module

    for module in (init_module, semantic_module):
        if snapshot not in Path(module.__file__).resolve().parents:
            raise SystemExit(f"smoke imported non-snapshot module: {module.__file__}")

    arms: dict[str, dict[str, Any]] = {}
    states: list[dict[str, Any]] = []
    expected_keys = {
        f"{PREFIX}_{field.name}"
        for field in fields(SemanticShannonSuccessConditionedSignedDiagnostics)
    }
    for arm, coefficient in (("c", 0.0), ("p", 0.0), ("f", 0.1)):
        tracker = build_semantic_shannon_tracker(semantic_args(coefficient))
        if tracker is None:
            raise SystemExit(f"{arm}: semantic estimator was not constructed")
        (
            advantages,
            diagnostics,
        ) = tracker.score_success_conditioned_signed_advantages_and_update(
            prompt_token_ids=[[11, 12]] * 4,
            answer_keys=["a", "a", "a", "b"],
            task_rewards=[1.0] * 4,
            active_mask=[True] * 4,
            num_samples=4,
            verified_support_keys_by_group=[["a", "b"]],
        )
        metrics = success_conditioned_semantic_metric_values(PREFIX, diagnostics)
        if set(metrics) != expected_keys:
            raise SystemExit(f"{arm}: semantic metric namespace drifted")
        state = tracker.state_dict()
        state_without_coefficient = dict(state)
        state_without_coefficient.pop("coefficient")
        states.append(state_without_coefficient)
        arms[arm] = {
            "advantages": advantages,
            "advantage_hex": [value.hex() for value in advantages],
            "coefficient": metrics[PREFIX + "_open_set_coefficient_used"],
            "raw_rms": metrics[PREFIX + "_raw_eligible_advantage_rms"],
            "effective_rms": metrics[PREFIX + "_effective_advantage_rms"],
            "eligible_fraction": metrics[PREFIX + "_eligible_fraction"],
            "verified_support_at_least_two_eligible_fraction": metrics[
                PREFIX + "_verified_support_at_least_two_eligible_fraction"
            ],
            "history_rows_added": metrics[PREFIX + "_history_rows_added"],
            "tracked_outcomes": metrics[PREFIX + "_tracked_outcomes"],
            "metric_key_count": len(metrics),
            "metric_keys_sha256": hashlib.sha256(
                "\n".join(sorted(metrics)).encode()
            ).hexdigest(),
        }

    if states[0] != states[1] or states[1] != states[2]:
        raise SystemExit("C/P/F estimator state plumbing differs beyond coefficient")
    for arm in ("c", "p"):
        row = arms[arm]
        if row["advantage_hex"] != ["0x0.0p+0"] * 4:
            raise SystemExit(f"{arm}: semantic advantage was not positive bitwise zero")
        if any(row[key] != 0.0 for key in ("coefficient", "raw_rms", "effective_rms")):
            raise SystemExit(f"{arm}: zero-dose semantic telemetry was nonzero")
        task = torch.tensor([0.25, -0.5, 1.0, -2.0], dtype=torch.float32)
        semantic = torch.tensor(row["advantages"], dtype=torch.float32)
        combined = add_semantic_shannon_separate_advantage(task, semantic)
        if not torch.equal(combined.view(torch.int32), task.view(torch.int32)):
            raise SystemExit(f"{arm}: zero-dose semantic path changed policy advantage")
    if arms["c"] != arms["p"]:
        raise SystemExit("C/P semantic runtime receipts differ")
    if arms["f"]["coefficient"] != 0.1 or arms["f"]["effective_rms"] <= 0.0:
        raise SystemExit("F semantic pressure did not activate")
    if any(row["eligible_fraction"] <= 0.0 for row in arms.values()):
        raise SystemExit("semantic eligibility was not observed in every arm")

    payload = {
        "schema": "e117r2_same_plumbing_runtime_smoke_v1",
        "passed": True,
        "snapshot_root": str(snapshot),
        "snapshot_identity_sha256": digest(identity),
        "efficacy_outcomes_used": False,
        "endpoint_contrasts_computed": False,
        "metric_key_parity": True,
        "estimator_state_parity_except_coefficient": True,
        "arms": arms,
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
