#!/usr/bin/env python3
"""Fail-closed audit for E58's global verified-replay Python smoke."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IDENTITY = (
    ROOT
    / "var"
    / "artifacts"
    / "e58_global_verified_replay_python_smoke_attempt3_identity.json"
)
DEFAULT_OUT = (
    ROOT / "var" / "artifacts" / "e58_global_replay_smoke_audit_latest.json"
)
PREFIX = "e58_global_verified_replay_smoke_attempt3_python_seed9010"
ARM = "verified_first_global_replay_canonical"
TERMINAL_STEP = 256
WARMUP = 64
EMA_DECAY = 0.9
BASE_COEFFICIENT = 0.1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_tree(root: Path) -> str:
    lines = [
        f"{_sha256(path)}  ./{path.relative_to(root).as_posix()}\n"
        for path in sorted(
            (item for item in root.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(root).as_posix(),
        )
    ]
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _zero(value: Any) -> bool:
    return _finite(value) and math.isclose(
        float(value), 0.0, rel_tol=0.0, abs_tol=1e-12
    )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an object")
    return payload


def _find_metrics(job_id: int) -> Path:
    candidates = sorted(
        (ROOT / "var" / "data").glob(
            f"*{PREFIX}_{ARM}_s9010/"
            f"debug_job{job_id}/train_metrics.jsonl"
        )
    )
    if len(candidates) != 1:
        raise ValueError(
            "expected one exact-job E58 metrics file, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _read_records(path: Path) -> list[dict[str, Any]]:
    records_by_step: dict[int, dict[str, Any]] = {}
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        record = json.loads(raw)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        step = int(record.get("trainer/global_step", -1))
        previous = records_by_step.get(step)
        if previous is not None:
            objective_keys = {
                key
                for key in set(previous) | set(record)
                if key.startswith(("train/", "actor/"))
            }
            conflicts = [
                key
                for key in objective_keys
                if previous.get(key) != record.get(key)
            ]
            if conflicts:
                raise ValueError(
                    f"{path}:{line_number} has conflicting duplicate step "
                    f"{step}: {', '.join(sorted(conflicts)[:5])}"
                )
            if len(record) > len(previous):
                records_by_step[step] = record
        else:
            records_by_step[step] = record
    return [records_by_step[step] for step in sorted(records_by_step)]


def _checkpoint_gate(job_id: int) -> dict[str, Any]:
    candidates = sorted(
        (ROOT / "var" / "data").glob(
            f"*{PREFIX}_{ARM}_s9010/debug_job{job_id}/checkpoints/"
            f"step_{TERMINAL_STEP:05d}/mp_rank_00_model_states.pt"
        )
    )
    if len(candidates) != 1:
        return {
            "status": "fail",
            "violations": [
                "expected one exact terminal checkpoint, "
                f"found {len(candidates)}"
            ],
        }
    path = candidates[0]
    try:
        import torch

        state = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except Exception as error:  # pragma: no cover - backend diagnostic
        return {
            "status": "fail",
            "path": str(path),
            "violations": [f"cannot inspect terminal checkpoint: {error}"],
        }

    violations: list[str] = []
    if state.get("maxent_alpha_controller_state") is not None:
        violations.append(
            "checkpoint unexpectedly contains a direct MaxEnt controller"
        )
    expected_controllers = {
        "canonical_replay_controller_state": (
            "canonical_replay_inverse",
            "unprojected_warmup_inverse_observed_bank_entropy_v1",
        ),
        "canonical_replay_mass_controller_state": (
            "canonical_replay_likelihood",
            "unprojected_warmup_verified_surprisal_ratio_v1",
        ),
    }
    for key, (kind, rule) in expected_controllers.items():
        payload = state.get(key)
        if not isinstance(payload, dict):
            violations.append(f"checkpoint lacks {key}")
            continue
        if (
            payload.get("controller_kind") != kind
            or payload.get("controller_rule") != rule
            or payload.get("base_alpha") != BASE_COEFFICIENT
            or payload.get("warmup_steps") != WARMUP
            or payload.get("ema_decay") != EMA_DECAY
        ):
            violations.append(f"checkpoint {key} configuration mismatch")

    semantic = state.get("semantic_shannon_tracker_state")
    semantic_controller = (
        semantic.get("open_set_controller")
        if isinstance(semantic, dict)
        else None
    )
    if (
        not isinstance(semantic_controller, dict)
        or semantic_controller.get("controller_kind")
        != "semantic_open_set_inverse"
        or semantic_controller.get("controller_rule")
        != "unprojected_warmup_inverse_open_set_entropy_v1"
        or semantic_controller.get("base_coefficient")
        != BASE_COEFFICIENT
        or semantic_controller.get("warmup_steps") != WARMUP
        or semantic_controller.get("ema_decay") != EMA_DECAY
    ):
        violations.append("checkpoint semantic controller mismatch")

    bank = state.get("online_canonical_bank_state")
    cursor = bank.get("global_replay_cursor") if isinstance(bank, dict) else None
    if (
        not isinstance(bank, dict)
        or bank.get("retain_exemplars") is not True
        or bank.get("replay_capacity") != 16
        or bank.get("global_replay_groups_per_step") != 1
        or isinstance(cursor, bool)
        or not isinstance(cursor, int)
        or cursor < 0
    ):
        violations.append(
            "checkpoint lacks the configured persistent global replay scheduler"
        )
    return {
        "status": "pass" if not violations else "fail",
        "path": str(path),
        "scheduler_cursor": cursor,
        "violations": violations,
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def audit(identity_path: Path) -> dict[str, Any]:
    violations: list[str] = []
    identity = _load_json(identity_path)
    expected_files = {
        "protocol_sha256": (
            ROOT
            / "paper"
            / "preregistration"
            / "e58_global_verified_replay_canonical_05b.md"
        ),
        "launcher_sha256": (
            ROOT
            / "ops"
            / "exp_scaling"
            / "launch_e58_global_replay_smoke.sh"
        ),
        "auditor_sha256": Path(__file__).resolve(),
        "manifest_sha256": (
            ROOT
            / "var"
            / "artifacts"
            / f"{PREFIX}_comparative_jobs.tsv"
        ),
    }
    if identity.get("schema") != "e58_global_verified_replay_python_smoke_v1":
        violations.append("identity schema mismatch")
    if identity.get("attempt") != 3:
        violations.append("identity attempt mismatch")
    for field, path in expected_files.items():
        if not path.is_file() or identity.get(field) != _sha256(path):
            violations.append(f"{field} binding mismatch")
    source_hash = str(identity.get("source_hash", ""))
    execution_hash = str(identity.get("execution_surface_hash", ""))
    source_root = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e58_global_verified_replay_{source_hash}"
        / "src"
    )
    execution_root = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e58_global_verified_replay_ops_{execution_hash}"
        / "ops"
    )
    try:
        if _hash_tree(source_root) != source_hash:
            violations.append("source snapshot binding mismatch")
        if _hash_tree(execution_root) != execution_hash:
            violations.append("execution snapshot binding mismatch")
    except OSError as exc:
        violations.append(f"cannot inspect frozen source/execution trees: {exc}")
    if (
        identity.get("arm") != ARM
        or identity.get("seed") != 9010
        or identity.get("max_updates") != TERMINAL_STEP
    ):
        violations.append("identity arm, seed, or smoke budget mismatch")
    if identity.get("global_replay") != {
        "groups_per_step": 1,
        "selection": "persistent_prompt_hash_round_robin",
        "capacity": 16,
    }:
        violations.append("identity global replay contract mismatch")
    if identity.get("direct_token_entropy") != {
        "coefficient": 0.0,
        "controller": None,
    }:
        violations.append("direct token entropy is not disabled")
    if identity.get("information_firewall") != {
        "desired_entropy": None,
        "desired_mode_count": None,
        "evaluation_feedback": False,
        "gold_support_feedback": False,
    }:
        violations.append("information firewall mismatch")

    try:
        job_id = int(identity["job_id"])
        with expected_files["manifest_sha256"].open(
            encoding="utf-8", newline=""
        ) as handle:
            manifest_rows = list(csv.DictReader(handle, delimiter="\t"))
        exact_rows = [
            row
            for row in manifest_rows
            if row.get("arm") == ARM
            and row.get("seed") == "9010"
            and row.get("run_stamp") == f"{PREFIX}_{ARM}_s9010"
            and row.get("job_id") == str(job_id)
        ]
        if len(exact_rows) != 1:
            violations.append(
                "manifest does not contain exactly one identity-bound job row"
            )
        metrics_path = _find_metrics(job_id)
        records = _read_records(metrics_path)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        violations.append(str(exc))
        job_id = -1
        metrics_path = None
        records = []

    first_discovery_step: int | None = None
    post_discovery_records = 0
    replay_activations = 0
    latest_step = -1
    previous_mass_observations = 0
    semantic_observations = 0
    balance_observations = 0
    latest_mass_observations = 0
    for index, record in enumerate(records):
        step = int(record.get("trainer/global_step", -1))
        latest_step = max(latest_step, step)
        if index and step <= int(
            records[index - 1].get("trainer/global_step", -1)
        ):
            violations.append("metric steps are not strictly increasing")
        for key, value in record.items():
            if key.startswith("train/maxent_"):
                violations.append(
                    f"step {step}: direct MaxEnt telemetry present: {key}"
                )
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and not math.isfinite(float(value))
            ):
                violations.append(f"step {step}: nonfinite {key}")

        tracked = record.get("train/online_canonical_tracked_outcomes", 0)
        tracked = int(tracked) if _finite(tracked) else 0
        if tracked > 0 and first_discovery_step is None:
            first_discovery_step = step
        groups = record.get("train/canonical_replay_available_groups", 0)
        groups = int(groups) if _finite(groups) else -1
        mass_observations = record.get(
            "train/canonical_replay_mass_observations", 0
        )
        mass_observations = (
            int(mass_observations) if _finite(mass_observations) else -1
        )

        if first_discovery_step is None:
            cold_checks = {
                "task_reward": record.get(
                    "train/online_canonical_task_reward_mean", 0
                ),
                "semantic_reward": record.get(
                    "train/semantic_shannon_augmented_reward_mean", 0
                ),
                "replay_groups": groups,
                "policy_grad_norm": record.get("train/policy_grad_norm", 0),
                "mass_observations": mass_observations,
            }
            for name, value in cold_checks.items():
                if not _zero(value):
                    violations.append(
                        f"step {step}: pre-discovery {name} is not zero"
                    )
        else:
            post_discovery_records += 1
            replay_activations += int(groups > 0)
            post_checks = {
                "scheduler_active": record.get(
                    "train/canonical_replay_global_scheduler_active"
                ),
                "groups_per_step": record.get(
                    "train/canonical_replay_global_groups_per_step"
                ),
                "available_groups": groups,
                "actuator_groups": record.get(
                    "train/canonical_replay_actuator_groups"
                ),
            }
            for name, value in post_checks.items():
                if not _finite(value) or not math.isclose(
                    float(value), 1.0, rel_tol=0.0, abs_tol=1e-12
                ):
                    violations.append(
                        f"step {step}: post-discovery {name} is not one"
                    )
            modes = record.get("train/canonical_replay_available_modes")
            if not _finite(modes) or float(modes) < 1:
                violations.append(
                    f"step {step}: scheduled replay lacks a verified mode"
                )
            if mass_observations != previous_mass_observations + 1:
                violations.append(
                    f"step {step}: mass controller did not advance exactly once"
                )
        previous_mass_observations = max(mass_observations, 0)
        latest_mass_observations = max(
            latest_mass_observations,
            mass_observations,
        )
        semantic_observations = max(
            semantic_observations,
            int(
                record.get(
                    "train/"
                    "semantic_shannon_success_conditioned_signed_"
                    "open_set_observations",
                    0,
                )
            ),
        )
        balance_observations = max(
            balance_observations,
            int(record.get("train/canonical_replay_observations", 0)),
        )
        for key in (
            "train/canonical_replay_alpha_projection_active",
            "train/canonical_replay_mass_projection_active",
            "train/canonical_replay_gold_support_feedback",
            "train/"
            "semantic_shannon_success_conditioned_signed_"
            "open_set_projection_active",
        ):
            value = record.get(key)
            if value is not None and not _zero(value):
                violations.append(
                    f"step {step}: forbidden projection/feedback {key}"
                )

    terminal = latest_step >= TERMINAL_STEP
    checkpoint_gate = (
        _checkpoint_gate(job_id)
        if terminal and job_id >= 0
        else {"status": "pending", "violations": []}
    )
    if terminal:
        if first_discovery_step is None:
            violations.append(
                f"no verifier-positive model discovery by step {TERMINAL_STEP}"
            )
        if post_discovery_records < 32:
            violations.append(
                "fewer than 32 post-discovery global replay observations"
            )
        if replay_activations != post_discovery_records:
            violations.append(
                "global replay was not active on every post-discovery update"
            )
        if latest_mass_observations != post_discovery_records:
            violations.append(
                "mass-controller observation count mismatches replay schedule"
            )
        violations.extend(checkpoint_gate.get("violations", []))

    status = "fail" if violations else "pass" if terminal else "in_progress"
    return {
        "schema": "e58_global_verified_replay_python_smoke_audit_v1",
        "status": status,
        "violations": sorted(set(violations)),
        "identity": str(identity_path),
        "metrics": None if metrics_path is None else str(metrics_path),
        "latest_step": latest_step,
        "terminal_step": TERMINAL_STEP,
        "first_discovery_step": first_discovery_step,
        "post_discovery_records": post_discovery_records,
        "replay_activations": replay_activations,
        "controller_observations": {
            "semantic": semantic_observations,
            "mass": latest_mass_observations,
            "balance": balance_observations,
        },
        "checkpoint_gate": checkpoint_gate,
        "objective_isolation": {
            "direct_token_entropy": "absent",
            "zero_gradient_cold_start": not any(
                "pre-discovery" in violation for violation in violations
            ),
            "gold_support_feedback": "absent",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    try:
        payload = audit(args.identity)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        payload = {
            "schema": "e58_global_verified_replay_python_smoke_audit_v1",
            "status": "fail",
            "violations": [str(exc)],
        }
    _write(args.out, payload)
    print(
        f"[e58-smoke-audit] status={payload['status']} "
        f"violations={len(payload.get('violations', []))} out={args.out}"
    )


if __name__ == "__main__":
    main()
