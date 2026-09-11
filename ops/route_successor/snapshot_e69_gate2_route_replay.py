#!/usr/bin/env python3
"""Persist compact, prospective E69 route-replay checkpoint observations.

The frozen training counter only counts a first neutral admission after a replay
marker. This observer does not change training. It records each checkpoint's
monotone replay-target set and neutral counts so a later count increase is a
conservative temporal lower bound on neutral route reproduction after replay.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "var/artifacts/e69_gate2_compute_matched_screen_audit_latest.json"
AMENDMENT = (
    ROOT
    / "paper/preregistration/"
    "e69_gate2_route_temporal_observer_amendment_20260728.md"
)
PARENT_OUT = ROOT / "var/artifacts/e69_gate2_route_temporal_snapshots.json"
R1_IDENTITY = ROOT / "var/artifacts/e69_gate2_r1_execution_repair_identity.json"
R1_PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e69_gate2_r1_execution_repair_20260728.md"
)
R1_OUT = ROOT / "var/artifacts/e69_gate2_r1_route_temporal_snapshots.json"
R2_IDENTITY = (
    ROOT
    / "var/artifacts/"
    "e69_gate2_r2_route_endpoint_bookkeeping_repair_identity.json"
)
R2_PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e69_gate2_r2_route_endpoint_bookkeeping_repair_20260728.md"
)
R2_OUT = ROOT / "var/artifacts/e69_gate2_r2_route_temporal_snapshots.json"
R1_ACTIVE = R1_IDENTITY.is_file()
R2_ACTIVE = R2_IDENTITY.is_file()
if R2_ACTIVE:
    AMENDMENT = R2_PROTOCOL
    OUT = R2_OUT
elif R1_ACTIVE:
    AMENDMENT = R1_PROTOCOL
    OUT = R1_OUT
else:
    OUT = PARENT_OUT
SUCCESSOR = "verified_route_successor"
POOL_SIZE = {
    "graph_coloring": 192,
    "countdown": 384,
    "python_factor": 384,
    "mathir": 384,
}
MINIMUM_ACTIVE_CHECKPOINT_AGE_SECONDS = 30


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _pair_key(route_key: str, prompt_key: str) -> str:
    return hashlib.sha256(
        route_key.encode("utf-8") + b"\0" + prompt_key.encode("ascii")
    ).hexdigest()


def _empty_payload() -> dict[str, Any]:
    payload = {
        "schema": (
            "e69_gate2_r2_route_temporal_snapshots_v1"
            if R2_ACTIVE
            else (
                "e69_gate2_r1_route_temporal_snapshots_v1"
                if R1_ACTIVE
                else "e69_gate2_route_temporal_snapshots_v1"
            )
        ),
        "amendment": str(AMENDMENT.resolve()),
        "amendment_sha256": _sha256(AMENDMENT),
        "observation_rule": (
            "later_neutral_count_increase_for_pair_already_replay_marked_"
            "at_an_earlier_checkpoint"
        ),
        "snapshots": {domain: {} for domain in POOL_SIZE},
        "temporal_reproductions": {},
    }
    if R2_ACTIVE:
        identity = json.loads(R2_IDENTITY.read_text(encoding="utf-8"))
        parent_hash = _sha256(R1_OUT)
        if identity.get("parent_route_temporal_snapshot_sha256") != parent_hash:
            raise SystemExit("E69-R2 parent temporal snapshot hash drift")
        parent = json.loads(R1_OUT.read_text(encoding="utf-8"))
        if parent.get("schema") != "e69_gate2_r1_route_temporal_snapshots_v1":
            raise SystemExit("E69-R2 parent temporal snapshot schema drift")
        for domain in ("countdown", "graph_coloring", "mathir"):
            payload["snapshots"][domain] = json.loads(
                json.dumps(parent["snapshots"][domain])
            )
        payload.update(
            {
                "r2_identity": str(R2_IDENTITY.resolve()),
                "r2_identity_sha256": _sha256(R2_IDENTITY),
                "parent_snapshot": str(R1_OUT.resolve()),
                "parent_snapshot_sha256": parent_hash,
                "carried_parent_domains": [
                    "countdown",
                    "graph_coloring",
                    "mathir",
                ],
                "fresh_r2_domains": ["python_factor"],
            }
        )
    elif R1_ACTIVE:
        identity = json.loads(R1_IDENTITY.read_text(encoding="utf-8"))
        parent_hash = _sha256(PARENT_OUT)
        if identity.get("parent_route_temporal_snapshot_sha256") != parent_hash:
            raise SystemExit("E69-R1 parent temporal snapshot hash drift")
        parent = json.loads(PARENT_OUT.read_text(encoding="utf-8"))
        if parent.get("schema") != "e69_gate2_route_temporal_snapshots_v1":
            raise SystemExit("E69-R1 parent temporal snapshot schema drift")
        for domain in ("countdown", "mathir"):
            payload["snapshots"][domain] = json.loads(
                json.dumps(parent["snapshots"][domain])
            )
        payload.update(
            {
                "r1_identity": str(R1_IDENTITY.resolve()),
                "r1_identity_sha256": _sha256(R1_IDENTITY),
                "parent_snapshot": str(PARENT_OUT.resolve()),
                "parent_snapshot_sha256": parent_hash,
                "carried_parent_domains": ["countdown", "mathir"],
                "fresh_r1_domains": ["graph_coloring", "python_factor"],
            }
        )
    return payload


def _load_payload() -> dict[str, Any]:
    if not OUT.is_file():
        return _empty_payload()
    payload = json.loads(OUT.read_text(encoding="utf-8"))
    expected_schema = (
        "e69_gate2_r2_route_temporal_snapshots_v1"
        if R2_ACTIVE
        else (
            "e69_gate2_r1_route_temporal_snapshots_v1"
            if R1_ACTIVE
            else "e69_gate2_route_temporal_snapshots_v1"
        )
    )
    if payload.get("schema") != expected_schema:
        raise SystemExit("unexpected E69 route temporal snapshot schema")
    if payload.get("amendment_sha256") != _sha256(AMENDMENT):
        raise SystemExit("E69 route temporal amendment hash drift")
    if set(payload.get("snapshots", {})) != set(POOL_SIZE):
        raise SystemExit("E69 route temporal snapshot domain drift")
    if R2_ACTIVE and (
        payload.get("r2_identity_sha256") != _sha256(R2_IDENTITY)
        or payload.get("parent_snapshot_sha256") != _sha256(R1_OUT)
        or payload.get("carried_parent_domains")
        != ["countdown", "graph_coloring", "mathir"]
        or payload.get("fresh_r2_domains") != ["python_factor"]
    ):
        raise SystemExit("E69-R2 route temporal provenance drift")
    if R1_ACTIVE and not R2_ACTIVE and (
        payload.get("r1_identity_sha256") != _sha256(R1_IDENTITY)
        or payload.get("parent_snapshot_sha256") != _sha256(PARENT_OUT)
        or payload.get("carried_parent_domains") != ["countdown", "mathir"]
        or payload.get("fresh_r1_domains")
        != ["graph_coloring", "python_factor"]
    ):
        raise SystemExit("E69-R1 route temporal provenance drift")
    return payload


def _checkpoint_snapshot(
    checkpoint: Path,
    *,
    domain: str,
    step: int,
) -> dict[str, Any]:
    import torch

    state = torch.load(
        checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if int(state.get("global_step", -1)) != step:
        raise SystemExit(f"{checkpoint}: global step mismatch")
    route_state = state.get("verified_route_library_state")
    if (
        not isinstance(route_state, dict)
        or route_state.get("schema")
        != "verified_cross_prompt_route_library_v1"
    ):
        raise SystemExit(f"{checkpoint}: route-library state is absent")
    if int(route_state.get("replay_groups_per_step", -1)) != 1:
        raise SystemExit(f"{checkpoint}: replay budget drift")
    records = route_state.get("records")
    targets = route_state.get("replayed_route_targets")
    counters = route_state.get("counters")
    if not isinstance(records, dict) or not isinstance(targets, list):
        raise SystemExit(f"{checkpoint}: malformed route-library state")
    if not isinstance(counters, dict):
        raise SystemExit(f"{checkpoint}: malformed route-library counters")

    target_counts: dict[str, int] = {}
    for raw_pair in targets:
        if (
            not isinstance(raw_pair, list)
            or len(raw_pair) != 2
            or not all(isinstance(value, str) for value in raw_pair)
        ):
            raise SystemExit(f"{checkpoint}: malformed replay-target pair")
        route_key, prompt_key = raw_pair
        record = records.get(route_key, {}).get(prompt_key)
        neutral_count = 0 if record is None else int(record["neutral_count"])
        if neutral_count < 0:
            raise SystemExit(f"{checkpoint}: negative neutral count")
        key = _pair_key(route_key, prompt_key)
        prior = target_counts.get(key)
        if prior is not None and prior != neutral_count:
            raise SystemExit(f"{checkpoint}: replay-target hash collision")
        target_counts[key] = neutral_count

    selected_counters = {
        key: int(counters.get(key, 0))
        for key in (
            "neutral_rows_observed",
            "neutral_routes_observed",
            "proposal_rows_admitted",
            "proposal_graduations",
            "cross_prompt_neutral_reproductions",
            "post_replay_cross_prompt_neutral_reproductions",
            "cross_prompt_replay_updates",
            "cross_prompt_replay_groups",
            "cross_prompt_replay_rows",
        )
    }
    del state
    return {
        "domain": domain,
        "step": step,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "replayed_target_neutral_counts": target_counts,
        "summary": {
            "replayed_target_pairs": len(target_counts),
            "replayed_target_pairs_with_neutral_support": sum(
                count > 0 for count in target_counts.values()
            ),
            "neutral_count_over_replayed_targets": sum(target_counts.values()),
        },
        "counters": selected_counters,
    }


def _derive_temporal_reproductions(
    snapshots: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for domain in POOL_SIZE:
        earlier_counts: dict[str, int] = {}
        reproduced_pairs: set[str] = set()
        occurrence_lower_bound = 0
        transitions: list[dict[str, Any]] = []
        ordered = sorted(
            snapshots[domain].values(),
            key=lambda row: int(row["step"]),
        )
        previous_step: int | None = None
        for snapshot in ordered:
            step = int(snapshot["step"])
            current = {
                str(key): int(value)
                for key, value in snapshot[
                    "replayed_target_neutral_counts"
                ].items()
            }
            missing = set(earlier_counts) - set(current)
            if missing:
                raise SystemExit(
                    f"{domain}/step{step}: replay-target set is not monotone"
                )
            new_pairs: list[str] = []
            transition_occurrences = 0
            if previous_step is not None:
                for pair, prior_count in earlier_counts.items():
                    current_count = current[pair]
                    if current_count < prior_count:
                        raise SystemExit(
                            f"{domain}/step{step}: neutral count decreased"
                        )
                    if current_count > prior_count:
                        reproduced_pairs.add(pair)
                        new_pairs.append(pair)
                        transition_occurrences += current_count - prior_count
                if new_pairs:
                    transitions.append(
                        {
                            "after_step": previous_step,
                            "observed_at_step": step,
                            "reproduced_pair_count": len(new_pairs),
                            "neutral_occurrence_lower_bound": (
                                transition_occurrences
                            ),
                            "pair_sha256": sorted(new_pairs),
                        }
                    )
                    occurrence_lower_bound += transition_occurrences
            earlier_counts = current
            previous_step = step
        result[domain] = {
            "snapshot_steps": [int(row["step"]) for row in ordered],
            "snapshot_count": len(ordered),
            "post_replay_neutral_reproduction_pairs": len(reproduced_pairs),
            "post_replay_neutral_occurrence_lower_bound": (
                occurrence_lower_bound
            ),
            "transitions": transitions,
        }
    return result


def main() -> None:
    if not AUDIT.is_file():
        raise SystemExit(f"E69 Gate 2 live audit is absent: {AUDIT}")
    if not AMENDMENT.is_file():
        raise SystemExit(f"E69 route temporal amendment is absent: {AMENDMENT}")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    runs = {
        str(row["domain"]): row
        for row in audit.get("physical_runs", [])
        if row.get("arm") == SUCCESSOR and row.get("domain") in POOL_SIZE
    }
    if set(runs) != set(POOL_SIZE):
        raise SystemExit("E69 live audit lacks the four exact successor runs")

    payload = _load_payload()
    added: list[str] = []
    now = time.time()
    for domain, pool_size in POOL_SIZE.items():
        run_dir = Path(str(runs[domain]["run_dir"]))
        if not run_dir.is_dir():
            raise SystemExit(f"{domain}: successor run directory is absent")
        training_complete = (run_dir / "TRAINING_COMPLETE.json").is_file()
        for pass_index in range(1, 7):
            step = pool_size * pass_index
            step_key = str(step)
            if step_key in payload["snapshots"][domain]:
                continue
            checkpoint = (
                run_dir
                / "checkpoints"
                / f"step_{step:05d}"
                / "mp_rank_00_model_states.pt"
            )
            evaluation = run_dir / "eval_results" / f"{step}_multi_answer.json"
            if not checkpoint.is_file() or not evaluation.is_file():
                continue
            if (
                not training_complete
                and now - checkpoint.stat().st_mtime
                < MINIMUM_ACTIVE_CHECKPOINT_AGE_SECONDS
            ):
                continue
            snapshot = _checkpoint_snapshot(
                checkpoint,
                domain=domain,
                step=step,
            )
            payload["snapshots"][domain][step_key] = snapshot
            added.append(f"{domain}:{step}")

    payload["temporal_reproductions"] = _derive_temporal_reproductions(
        payload["snapshots"]
    )
    payload["summary"] = {
        "observed_snapshots": sum(
            len(rows) for rows in payload["snapshots"].values()
        ),
        "expected_snapshots": len(POOL_SIZE) * 6,
        "domains_with_post_replay_neutral_reproduction": sum(
            row["post_replay_neutral_reproduction_pairs"] > 0
            for row in payload["temporal_reproductions"].values()
        ),
    }
    _atomic_json(OUT, payload)
    print(
        "[e69-route-snapshot] "
        f"added={','.join(added) if added else 'none'} "
        f"observed={payload['summary']['observed_snapshots']}/24 "
        "reproducing_domains="
        f"{payload['summary']['domains_with_post_replay_neutral_reproduction']}"
    )


if __name__ == "__main__":
    main()
