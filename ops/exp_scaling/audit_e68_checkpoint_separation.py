#!/usr/bin/env python3
"""Audit E68's latest durable checkpoint for objective/replay separation."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
IDENTITY = (
    ROOT / "var/artifacts/e68_separated_support_actuator_ablation_identity.json"
)
OUT = ROOT / "var/artifacts/e68_checkpoint_separation_audit_latest.json"
EXPECTED_STEPS = {
    "graph_coloring": 192 * 12,
    "countdown": 384 * 12,
    "python_factor": 384 * 12,
    "mathir": 384 * 12,
}
EXPECTED_SCHEMA = (
    "online_growing_support_canonical_maxent_replay_"
    "separated_proposal_v3"
)
STEP_DIR = re.compile(r"step_(\d+)$")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _close(value: Any, expected: float) -> bool:
    return _finite(value) and math.isclose(
        float(value),
        expected,
        rel_tol=0.0,
        abs_tol=1e-9,
    )


def _run_dir(run_stamp: str, job_id: int) -> Path | None:
    candidates = sorted(
        (ROOT / "var/data").glob(f"*_{run_stamp}/debug_job{job_id}")
    )
    return candidates[0] if len(candidates) == 1 else None


def _latest_durable_checkpoint(run_dir: Path) -> tuple[int, Path] | None:
    candidates: list[tuple[int, Path]] = []
    checkpoint_root = run_dir / "checkpoints"
    for directory in checkpoint_root.glob("step_*"):
        match = STEP_DIR.fullmatch(directory.name)
        if match is None:
            continue
        model_state = directory / "mp_rank_00_model_states.pt"
        optimizer_state = (
            directory / "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"
        )
        if not model_state.is_file() or not optimizer_state.is_file():
            continue
        model_stat = model_state.stat()
        optimizer_stat = optimizer_state.stat()
        if model_stat.st_size <= 0 or optimizer_stat.st_size <= 0:
            continue
        newest_write_ns = max(model_stat.st_mtime_ns, optimizer_stat.st_mtime_ns)
        if time.time_ns() - newest_write_ns < 30_000_000_000:
            continue
        candidates.append((int(match.group(1)), model_state))
    return max(candidates, default=None)


def _audit_checkpoint(
    path: Path,
    *,
    step: int,
    domain: str,
    seed: int,
    job_id: int,
) -> dict[str, Any]:
    violations: list[str] = []
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError("model state is not a dictionary")
    if int(state.get("global_step", -1)) != step:
        violations.append("top-level global step does not match checkpoint")

    bank = state.get("online_canonical_bank_state")
    if not isinstance(bank, dict):
        violations.append("online canonical bank state is missing")
        bank = {}
    if bank.get("schema") != EXPECTED_SCHEMA:
        violations.append("separated-support checkpoint schema mismatch")
    if bank.get("separate_proposal_objective_support") is not True:
        violations.append("separated proposal/objective state is disabled")
    if not _close(bank.get("entropy_alpha"), 0.0):
        violations.append("checkpoint canonical entropy alpha is not zero")
    if not _close(bank.get("novelty_beta"), 0.5):
        violations.append("checkpoint novelty beta is not 0.5")

    counts = bank.get("counts")
    exemplars = bank.get("exemplars")
    proposal_only = bank.get("proposal_only_outcomes")
    if not isinstance(counts, dict):
        violations.append("checkpoint on-policy counts are malformed")
        counts = {}
    if not isinstance(exemplars, dict):
        violations.append("checkpoint replay exemplars are malformed")
        exemplars = {}
    if not isinstance(proposal_only, dict):
        violations.append("checkpoint proposal-only outcomes are malformed")
        proposal_only = {}

    proposal_only_total = 0
    objective_overlap = 0
    replay_present = 0
    for prompt_key, outcomes in proposal_only.items():
        if not isinstance(outcomes, list) or len(outcomes) != len(set(outcomes)):
            violations.append(
                f"proposal-only outcomes malformed for prompt {prompt_key}"
            )
            continue
        proposal_set = set(outcomes)
        proposal_only_total += len(proposal_set)
        count_set = set(counts.get(prompt_key, {}))
        exemplar_set = set(exemplars.get(prompt_key, {}))
        objective_overlap += len(proposal_set & count_set)
        replay_present += len(proposal_set & exemplar_set)
    if objective_overlap:
        violations.append(
            f"{objective_overlap} proposal-only outcomes leaked into counts"
        )
    if replay_present != proposal_only_total:
        violations.append(
            "one or more proposal-only outcomes are absent from replay exemplars"
        )

    proposal_groups = bank.get("proposal_groups")
    proposal_rows = bank.get("proposal_rows")
    proposal_new_outcomes = bank.get("proposal_new_outcomes")
    for name, value in (
        ("proposal_groups", proposal_groups),
        ("proposal_rows", proposal_rows),
        ("proposal_new_outcomes", proposal_new_outcomes),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            violations.append(f"checkpoint {name} is invalid")
    if (
        isinstance(proposal_new_outcomes, int)
        and not isinstance(proposal_new_outcomes, bool)
        and proposal_only_total > proposal_new_outcomes
    ):
        violations.append("proposal-only support exceeds cumulative discoveries")
    proposal_graduated = (
        proposal_new_outcomes - proposal_only_total
        if isinstance(proposal_new_outcomes, int)
        and not isinstance(proposal_new_outcomes, bool)
        and proposal_new_outcomes >= proposal_only_total
        else None
    )

    tracker = state.get("semantic_shannon_tracker_state")
    if not isinstance(tracker, dict):
        violations.append("semantic entropy tracker state is missing")
        tracker = {}
    if (
        tracker.get("schema")
        != "semantic_shannon_tracker_v4_open_set_inverse"
        or tracker.get("open_set_inverse_adaptation") is not True
    ):
        violations.append("open-set entropy tracker schema mismatch")
    controller = tracker.get("open_set_controller")
    if not isinstance(controller, dict):
        violations.append("open-set inverse controller state is missing")
        controller = {}
    if (
        controller.get("controller_rule")
        != "unprojected_warmup_inverse_open_set_entropy_v1"
        or not _close(controller.get("base_coefficient"), 0.1)
        or controller.get("warmup_steps") != 64
        or not _finite(controller.get("current_coefficient"))
        or float(controller.get("current_coefficient", 0.0)) <= 0
    ):
        violations.append("unprojected entropy controller state mismatch")

    checkpoint_stat = path.stat()
    return {
        "domain": domain,
        "seed": seed,
        "job_id": job_id,
        "checkpoint_step": step,
        "checkpoint_path": str(path.relative_to(ROOT)),
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "bank_schema": bank.get("schema"),
        "separate_proposal_objective_support": bank.get(
            "separate_proposal_objective_support"
        ),
        "proposal_only_outcomes": proposal_only_total,
        "proposal_objective_overlap": objective_overlap,
        "proposal_replay_exemplars_present": replay_present,
        "proposal_groups": proposal_groups,
        "proposal_rows": proposal_rows,
        "proposal_new_outcomes": proposal_new_outcomes,
        "proposal_graduated_to_on_policy": proposal_graduated,
        "violations": violations,
    }


def main() -> None:
    auditor_sha256 = _digest(Path(__file__))
    previous: dict[str, Any] = {}
    if OUT.is_file():
        try:
            previous_payload = json.loads(OUT.read_text(encoding="utf-8"))
            if previous_payload.get("auditor_sha256") == auditor_sha256:
                previous = {
                    record["checkpoint_path"]: record
                    for record in previous_payload.get("checkpoints", [])
                    if not record.get("violations")
                }
        except (json.JSONDecodeError, KeyError, TypeError):
            previous = {}

    violations: list[str] = []
    checkpoints: list[dict[str, Any]] = []
    checkpointed_runs = 0
    terminal_checkpoint_runs = 0
    if not IDENTITY.is_file():
        identity = {}
        violations.append("E68 identity is missing")
    else:
        identity = json.loads(IDENTITY.read_text(encoding="utf-8"))

    for domain, expected_step in EXPECTED_STEPS.items():
        pass_steps = expected_step // 12
        for expected in identity.get("jobs", {}).get(domain, []):
            seed = int(expected["seed"])
            job_id = int(expected["job_id"])
            run_dir = _run_dir(expected["run_stamp"], job_id)
            if run_dir is None:
                continue
            latest = _latest_durable_checkpoint(run_dir)
            if latest is None:
                continue
            checkpointed_runs += 1
            step, path = latest
            label = f"{domain}/s{seed}/j{job_id}/step{step}"
            if step <= 0 or step > expected_step or step % pass_steps:
                violations.append(f"{label}: unregistered checkpoint step")
            terminal_checkpoint_runs += int(step == expected_step)
            relative_path = str(path.relative_to(ROOT))
            path_stat = path.stat()
            cached = previous.get(relative_path)
            if (
                cached is not None
                and cached.get("checkpoint_size") == path_stat.st_size
                and cached.get("checkpoint_mtime_ns") == path_stat.st_mtime_ns
            ):
                record = cached
            else:
                try:
                    record = _audit_checkpoint(
                        path,
                        step=step,
                        domain=domain,
                        seed=seed,
                        job_id=job_id,
                    )
                except Exception as error:  # fail closed on durable state
                    record = {
                        "domain": domain,
                        "seed": seed,
                        "job_id": job_id,
                        "checkpoint_step": step,
                        "checkpoint_path": relative_path,
                        "checkpoint_size": path_stat.st_size,
                        "checkpoint_mtime_ns": path_stat.st_mtime_ns,
                        "violations": [
                            f"checkpoint load/audit failed: "
                            f"{type(error).__name__}: {error}"
                        ],
                    }
            violations.extend(
                f"{label}: {message}"
                for message in record.get("violations", [])
            )
            checkpoints.append(record)

    status = (
        "fail"
        if violations
        else "pass"
        if terminal_checkpoint_runs == 12
        else "in_progress"
    )
    payload = {
        "schema": "e68_checkpoint_separation_audit_v1",
        "auditor_sha256": auditor_sha256,
        "status": status,
        "summary": {
            "expected_runs": 12,
            "checkpointed_runs": checkpointed_runs,
            "terminal_checkpoint_runs": terminal_checkpoint_runs,
            "audited_checkpoint_files": len(checkpoints),
            "proposal_new_outcomes": sum(
                int(record.get("proposal_new_outcomes", 0))
                for record in checkpoints
            ),
            "proposal_only_outcomes": sum(
                int(record.get("proposal_only_outcomes", 0))
                for record in checkpoints
            ),
            "proposal_graduated_to_on_policy": sum(
                int(record.get("proposal_graduated_to_on_policy", 0))
                for record in checkpoints
            ),
        },
        "checkpoints": checkpoints,
        "violations": sorted(set(violations)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{OUT.name}.", dir=OUT.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, OUT)
    print(
        f"[e68-checkpoint] status={status} "
        f"summary={payload['summary']} "
        f"violations={len(payload['violations'])}"
    )


if __name__ == "__main__":
    main()
