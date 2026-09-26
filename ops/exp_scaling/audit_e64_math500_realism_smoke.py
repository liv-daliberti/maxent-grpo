#!/usr/bin/env python3
"""Fail-closed audit for E64's verified-answer MATH realism smoke."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
IDENTITY = ROOT / "var/artifacts/e64_math500_realism_smoke_identity.json"
OUT = ROOT / "var/artifacts/e64_math500_realism_smoke_audit_latest.json"
PREFIX = "mte64_math500_realism_smoke_e58_s43"
ARM = "verified_first_global_replay_canonical"
TERMINAL_STEP = 96
SINGLE_KEY = "math_verified_answer:correct"


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


def _zero(value: Any, *, tolerance: float = 1e-10) -> bool:
    return _finite(value) and math.isclose(
        float(value),
        0.0,
        rel_tol=0.0,
        abs_tol=tolerance,
    )


def _read_records(path: Path) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        record = json.loads(raw)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        step = int(record.get("trainer/global_step", -1))
        previous = by_step.get(step)
        if previous is None:
            by_step[step] = record
            continue
        objective_keys = {
            key
            for key in set(previous) | set(record)
            if key.startswith(("train/", "actor/"))
        }
        conflicts = [
            key for key in objective_keys if previous.get(key) != record.get(key)
        ]
        if conflicts:
            raise ValueError(
                f"{path}:{line_number} conflicting step {step}: "
                + ", ".join(sorted(conflicts)[:5])
            )
        if len(record) > len(previous):
            by_step[step] = record
    return [by_step[step] for step in sorted(by_step)]


def _find_metrics(job_id: int) -> Path:
    candidates = sorted(
        (ROOT / "var/data").glob(
            f"*{PREFIX}_{ARM}_s43/"
            f"debug_job{job_id}/train_metrics.jsonl"
        )
    )
    if len(candidates) != 1:
        raise ValueError(
            f"expected one identity-bound E64 metrics file, found "
            f"{len(candidates)}"
        )
    return candidates[0]


def _checkpoint_gate(job_id: int) -> dict[str, Any]:
    candidates = sorted(
        (ROOT / "var/data").glob(
            f"*{PREFIX}_{ARM}_s43/debug_job{job_id}/checkpoints/"
            "step_00096/mp_rank_00_model_states.pt"
        )
    )
    if len(candidates) != 1:
        return {
            "status": "fail",
            "violations": [
                f"expected one terminal checkpoint, found {len(candidates)}"
            ],
        }
    try:
        import torch

        state = torch.load(
            candidates[0],
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except Exception as error:  # pragma: no cover - backend diagnostic
        return {
            "status": "fail",
            "path": str(candidates[0]),
            "violations": [f"cannot inspect terminal checkpoint: {error}"],
        }

    violations: list[str] = []
    if state.get("maxent_alpha_controller_state") is not None:
        violations.append("checkpoint contains a direct MaxEnt controller")
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
        controller = state.get(key)
        if (
            not isinstance(controller, dict)
            or controller.get("controller_kind") != kind
            or controller.get("controller_rule") != rule
            or controller.get("base_alpha") != 0.1
            or controller.get("warmup_steps") != 64
            or controller.get("ema_decay") != 0.9
        ):
            violations.append(f"checkpoint {key} mismatch")

    balance = state.get("canonical_replay_controller_state")
    if isinstance(balance, dict) and balance.get("observation_count") != 0:
        violations.append("checkpoint balance controller consumed an observation")

    semantic = state.get("semantic_shannon_tracker_state")
    semantic_controller = (
        semantic.get("open_set_controller")
        if isinstance(semantic, dict)
        else None
    )
    if (
        not isinstance(semantic_controller, dict)
        or semantic_controller.get("controller_rule")
        != "unprojected_warmup_inverse_open_set_entropy_v1"
    ):
        violations.append("checkpoint semantic controller mismatch")

    bank = state.get("online_canonical_bank_state")
    counts = bank.get("counts") if isinstance(bank, dict) else None
    exemplars = bank.get("exemplars") if isinstance(bank, dict) else None
    if (
        not isinstance(bank, dict)
        or bank.get("retain_exemplars") is not True
        or bank.get("replay_capacity") != 16
        or bank.get("global_replay_groups_per_step") != 1
        or bank.get("global_replay_bootstrap_steps") != 0
        or not isinstance(counts, dict)
        or not isinstance(exemplars, dict)
    ):
        violations.append("checkpoint verified bank/scheduler mismatch")
    else:
        for prompt_key, outcomes in counts.items():
            if (
                not isinstance(outcomes, dict)
                or set(outcomes) != {SINGLE_KEY}
                or set(exemplars.get(prompt_key, {})) != {SINGLE_KEY}
            ):
                violations.append(
                    "checkpoint contains non-singleton or non-verified-answer "
                    "canonical support"
                )
                break

    return {
        "status": "pass" if not violations else "fail",
        "path": str(candidates[0]),
        "tracked_prompts": len(counts) if isinstance(counts, dict) else None,
        "maximum_support": (
            max((len(value) for value in counts.values()), default=0)
            if isinstance(counts, dict)
            else None
        ),
        "violations": violations,
    }


def audit(identity_path: Path) -> dict[str, Any]:
    violations: list[str] = []
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    expected_files = {
        "protocol_sha256": (
            ROOT
            / "paper/preregistration/e64_math500_realism_transfer_05b.md"
        ),
        "launcher_sha256": (
            ROOT
            / "ops/exp_scaling/launch_e64_math500_realism_smoke.sh"
        ),
        "auditor_sha256": Path(__file__).resolve(),
        "data_manifest_sha256": (
            ROOT
            / "var/data/math12k_384_math500/MATERIALIZATION_MANIFEST.json"
        ),
        "manifest_sha256": (
            ROOT / f"var/artifacts/{PREFIX}_comparative_jobs.tsv"
        ),
    }
    if identity.get("schema") != "e64_math500_realism_smoke_v1":
        violations.append("identity schema mismatch")
    for field, path in expected_files.items():
        if not path.is_file() or identity.get(field) != _sha256(path):
            violations.append(f"{field} binding mismatch")
    if (
        identity.get("arm") != ARM
        or identity.get("seed") != 43
        or identity.get("max_updates") != TERMINAL_STEP
        or identity.get("canonical_key_mode") != "math_verified_answer"
    ):
        violations.append("identity arm/seed/budget/key-mode mismatch")
    if identity.get("canonical_contract") != {
        "maximum_support_per_prompt": 1,
        "reasoning_strategy_claim": False,
        "reward_positive_key": SINGLE_KEY,
        "reward_zero_key": None,
    }:
        violations.append("identity canonical contract mismatch")
    if identity.get("evaluation") != {
        "math500_is_training_input": False,
        "purpose": "smoke diagnostics only",
        "sampled_k": 0,
    }:
        violations.append("identity evaluation firewall mismatch")
    if identity.get("information_firewall") != {
        "desired_entropy": None,
        "desired_mode_count": None,
        "evaluation_feedback": False,
        "gold_support_feedback": False,
    }:
        violations.append("identity information firewall mismatch")
    if identity.get("coefficient_projection") is not None:
        violations.append("identity permits coefficient projection")

    data_manifest = json.loads(
        expected_files["data_manifest_sha256"].read_text(encoding="utf-8")
    )
    output = data_manifest.get("output", {})
    audit_data = data_manifest.get("audit", {})
    if (
        data_manifest.get("schema")
        != "e39_math12k_384_math500_materialization_v1"
        or output.get("train_rows") != 384
        or output.get("eval_rows") != 500
        or output.get("train_ordered_row_sha256")
        != "051baa5571a1865518ef200c414178f1e50decea261bcd46ec0328e5837c0f36"
        or output.get("eval_ordered_row_sha256")
        != "1576fd11df21dc705a7c85000f232031212225cd9c00520faa26f6bdfc751166"
        or audit_data.get("normalized_problem_overlap") != 0
    ):
        violations.append("frozen train/eval data boundary mismatch")

    source_hash = str(identity.get("source_hash", ""))
    execution_hash = str(identity.get("execution_surface_hash", ""))
    try:
        source_root = (
            ROOT
            / f"var/artifacts/source_snapshots/"
            f"e64_math500_realism_{source_hash}/src"
        )
        execution_root = (
            ROOT
            / f"var/artifacts/source_snapshots/"
            f"e64_math500_realism_ops_{execution_hash}/ops"
        )
        if _hash_tree(source_root) != source_hash:
            violations.append("source snapshot mismatch")
        if _hash_tree(execution_root) != execution_hash:
            violations.append("execution snapshot mismatch")
    except OSError as error:
        violations.append(f"cannot inspect source snapshots: {error}")

    metrics_path: Path | None = None
    records: list[dict[str, Any]] = []
    job_id = -1
    try:
        job_id = int(identity["job_id"])
        with expected_files["manifest_sha256"].open(
            encoding="utf-8",
            newline="",
        ) as handle:
            manifest = list(csv.DictReader(handle, delimiter="\t"))
        exact = [
            row
            for row in manifest
            if row.get("arm") == ARM
            and row.get("seed") == "43"
            and row.get("job_id") == str(job_id)
        ]
        if len(exact) != 1:
            violations.append("manifest lacks one exact identity-bound job")
        metrics_path = _find_metrics(job_id)
        records = _read_records(metrics_path)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        violations.append(str(error))

    latest_step = -1
    first_discovery_step: int | None = None
    post_discovery_records = 0
    replay_activations = 0
    mass_observations = 0
    balance_observations = 0
    previous_mass = 0
    for record in records:
        step = int(record.get("trainer/global_step", -1))
        latest_step = max(latest_step, step)
        for key, value in record.items():
            if key.startswith("train/maxent_"):
                violations.append(f"step {step}: direct MaxEnt telemetry {key}")
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

        support_two = record.get(
            "train/online_canonical_support_at_least_two_prompt_fraction"
        )
        if support_two is not None and not _zero(support_two):
            violations.append(f"step {step}: verified support exceeded one")

        groups = record.get("train/canonical_replay_available_groups", 0)
        groups = int(groups) if _finite(groups) else -1
        modes = record.get("train/canonical_replay_available_modes", 0)
        modes = int(modes) if _finite(modes) else -1
        current_mass = record.get("train/canonical_replay_mass_observations", 0)
        current_mass = int(current_mass) if _finite(current_mass) else -1
        current_balance = record.get("train/canonical_replay_observations", 0)
        current_balance = (
            int(current_balance) if _finite(current_balance) else -1
        )
        balance_observations = max(balance_observations, current_balance)
        mass_observations = max(mass_observations, current_mass)

        if current_balance != 0:
            violations.append(
                f"step {step}: singleton bank advanced balance controller"
            )
        if first_discovery_step is None:
            for name, value in {
                "task reward": record.get(
                    "train/online_canonical_task_reward_mean",
                    0,
                ),
                "semantic reward": record.get(
                    "train/semantic_shannon_augmented_reward_mean",
                    0,
                ),
                "replay groups": groups,
                "policy gradient": record.get("train/policy_grad_norm", 0),
                "mass observations": current_mass,
            }.items():
                if not _zero(value):
                    violations.append(
                        f"step {step}: pre-discovery {name} is not zero"
                    )
        else:
            post_discovery_records += 1
            expected_one = {
                "global scheduler": record.get(
                    "train/canonical_replay_global_scheduler_active"
                ),
                "groups per step": record.get(
                    "train/canonical_replay_global_groups_per_step"
                ),
                "available groups": groups,
                "available modes": modes,
                "actuator groups": record.get(
                    "train/canonical_replay_actuator_groups"
                ),
                "actuator modes": record.get(
                    "train/canonical_replay_actuator_modes"
                ),
            }
            for name, value in expected_one.items():
                if not _finite(value) or not math.isclose(
                    float(value),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    violations.append(f"step {step}: {name} is not one")
            replay_activations += int(groups == 1 and modes == 1)
            if current_mass != previous_mass + 1:
                violations.append(
                    f"step {step}: mass observations did not advance once"
                )
            singleton_zero_metrics = (
                "train/canonical_replay_eligible_groups",
                "train/canonical_replay_retained_modes",
                "train/canonical_replay_balance_loss",
                "train/canonical_replay_balance_score_gradient_sum",
                "train/canonical_replay_balance_score_gradient_l2",
            )
            for key in singleton_zero_metrics:
                if not _zero(record.get(key)):
                    violations.append(
                        f"step {step}: singleton balance metric {key} "
                        "is not zero"
                    )
            mass_gradient_sum = record.get(
                "train/canonical_replay_mass_score_gradient_sum"
            )
            mass_gradient_l2 = record.get(
                "train/canonical_replay_mass_score_gradient_l2"
            )
            if not _finite(mass_gradient_sum) or not math.isclose(
                float(mass_gradient_sum),
                -1.0,
                rel_tol=0.0,
                abs_tol=2e-6,
            ):
                violations.append(
                    f"step {step}: verified-mass score gradient is not -1"
                )
            if not _finite(mass_gradient_l2) or not math.isclose(
                float(mass_gradient_l2),
                1.0,
                rel_tol=0.0,
                abs_tol=2e-6,
            ):
                violations.append(
                    f"step {step}: verified-mass gradient norm is not one"
                )
        previous_mass = max(current_mass, 0)

        for key in (
            "train/canonical_replay_alpha_projection_active",
            "train/canonical_replay_projection_active",
            "train/canonical_replay_mass_projection_active",
            "train/canonical_replay_gold_support_feedback",
            "train/"
            "semantic_shannon_success_conditioned_signed_"
            "open_set_projection_active",
        ):
            value = record.get(key)
            if value is not None and not _zero(value):
                violations.append(f"step {step}: forbidden {key}")

    terminal = latest_step >= TERMINAL_STEP
    checkpoint = (
        _checkpoint_gate(job_id)
        if terminal and job_id >= 0
        else {"status": "pending", "violations": []}
    )
    if terminal:
        if first_discovery_step is None:
            violations.append("no model-generated verified discovery")
        if post_discovery_records <= 0:
            violations.append("no post-discovery verified-mass replay")
        if replay_activations != post_discovery_records:
            violations.append("global replay missed a post-discovery update")
        if mass_observations != post_discovery_records:
            violations.append("mass observation count mismatches replay")
        if balance_observations != 0:
            violations.append("balance controller was not structurally idle")
        violations.extend(checkpoint.get("violations", []))

    status = "fail" if violations else "pass" if terminal else "in_progress"
    return {
        "schema": "e64_math500_realism_smoke_audit_v1",
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
            "mass": mass_observations,
            "balance": balance_observations,
        },
        "maximum_verified_support": 1 if first_discovery_step is not None else 0,
        "checkpoint_gate": checkpoint,
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", type=Path, default=IDENTITY)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    try:
        payload = audit(args.identity)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        payload = {
            "schema": "e64_math500_realism_smoke_audit_v1",
            "status": "fail",
            "violations": [str(error)],
        }
    _write(args.out, payload)
    print(
        f"[e64-smoke-audit] status={payload['status']} "
        f"violations={len(payload.get('violations', []))} out={args.out}"
    )


if __name__ == "__main__":
    main()
