#!/usr/bin/env python3
"""Fail-closed audit for E53's three-domain verified-replay sentinel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARMS = ("grpo", "maxent_inverse", "maxent_inverse_canonical_replay")
REPLAY_ARM = "maxent_inverse_canonical_replay"
DOMAINS = {
    "countdown": (
        "cde53_verified_replay_05b_50ep_sentinel_allcs",
        384,
    ),
    "graph_coloring": (
        "gce53_verified_replay_05b_50ep_sentinel",
        192,
    ),
    "python_factor": (
        "pye53_verified_replay_05b_50ep_sentinel_allcs",
        384,
    ),
}
SEED = 9010
DIRECT_BASE_ALPHA = 0.000075
REPLAY_BASE_ALPHA = 0.10
WARMUP_STEPS = 64
MAX_PASSES = 50
NUM_SAMPLES = 16
REPLAY_CAPACITY = 16
REPLAY_ESTIMATOR_SCALE = 15.0 / 16.0
DISTINCT_KEY = "eval/multi_answer/sampled_distinct_correct_at_8"
PASS8_KEY = "eval/multi_answer/sampled_any_correct_at_8"
MEAN8_KEY = "eval/multi_answer/sampled_mean_at_8"
IDENTITY_PATH = (
    ROOT / "var/artifacts/e53_verified_replay_05b_sentinel_identity.json"
)
PROTOCOL_PATH = (
    ROOT / "paper/preregistration/e53_verified_exemplar_replay_05b.md"
)
LAUNCHER_PATH = (
    ROOT / "ops/exp_scaling/launch_e53_verified_exemplar_replay_05b.sh"
)
APPROVAL_PATH = ROOT / "var/artifacts/e53_sentinel_stage_a_approval.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tree(root: Path) -> str:
    lines = []
    for path in sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        lines.append(f"{_sha256_file(path)}  ./{relative}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def approval_binding() -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    try:
        identity = json.loads(IDENTITY_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        return {}, [f"sentinel identity is unavailable or invalid: {error}"]
    if (
        not isinstance(identity, dict)
        or identity.get("schema")
        != "e53_verified_exemplar_replay_05b_sentinel_v1"
    ):
        return {}, ["sentinel identity has an incompatible schema"]

    source_hash = str(identity.get("source_hash", ""))
    execution_hash = str(identity.get("execution_surface_hash", ""))
    source_root = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e53_verified_replay_{source_hash}"
        / "src"
    )
    ops_root = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e53_verified_replay_ops_{execution_hash}"
        / "ops"
    )
    try:
        observed = {
            "identity_sha256": _sha256_file(IDENTITY_PATH),
            "protocol_sha256": _sha256_file(PROTOCOL_PATH),
            "sentinel_launcher_sha256": _sha256_file(LAUNCHER_PATH),
            "source_hash": _hash_tree(source_root),
            "execution_surface_hash": _hash_tree(ops_root),
            "auditor_sha256": _sha256_file(Path(__file__).resolve()),
        }
    except OSError as error:
        return {}, [f"sentinel approval evidence is unavailable: {error}"]
    expected = {
        "protocol_sha256": identity.get("protocol_sha256"),
        "sentinel_launcher_sha256": identity.get("launcher_sha256"),
        "source_hash": identity.get("source_hash"),
        "execution_surface_hash": identity.get("execution_surface_hash"),
        "auditor_sha256": identity.get("auditor_sha256"),
    }
    for key, expected_value in expected.items():
        if observed[key] != expected_value:
            violations.append(
                f"approval binding mismatch for {key}: "
                f"expected={expected_value!r} observed={observed[key]!r}"
            )
    binding = {
        "identity_path": str(IDENTITY_PATH.resolve()),
        "source_snapshot_root": str(source_root.resolve()),
        "execution_snapshot_root": str(ops_root.resolve()),
        **observed,
    }
    return binding, violations


def _finite(record: dict[str, Any], key: str) -> float | None:
    value = record.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _load_records(data_root: Path, run_stamp: str) -> list[dict[str, Any]]:
    candidates = sorted(
        data_root.glob(f"*_{run_stamp}/debug_*/train_metrics.jsonl"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )
    records: list[dict[str, Any]] = []
    for path in candidates:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def checkpoint_gate(
    data_root: Path,
    *,
    run_stamp: str,
    arm: str,
    terminal_step: int,
) -> dict[str, Any]:
    """Inspect the exact terminal client state without loading model tensors."""

    expected_tag = f"step_{terminal_step:05d}"
    candidates = sorted(
        data_root.glob(
            f"*_{run_stamp}/debug_*/checkpoints/"
            f"{expected_tag}/mp_rank_00_model_states.pt"
        ),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )
    if not candidates:
        return {
            "status": "fail",
            "violations": [f"missing terminal checkpoint {expected_tag}"],
        }
    path = candidates[-1]
    try:
        import torch

        state = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except Exception as error:  # pragma: no cover - backend-dependent diagnostic
        return {
            "status": "fail",
            "path": str(path),
            "violations": [f"cannot inspect terminal checkpoint: {error}"],
        }
    violations: list[str] = []
    bank = state.get("online_canonical_bank_state")
    if not isinstance(bank, dict):
        violations.append("checkpoint lacks the verified canonical bank")
    direct = state.get("maxent_alpha_controller_state")
    if arm == "grpo":
        if direct is not None:
            violations.append("control checkpoint unexpectedly has direct controller")
    elif not isinstance(direct, dict):
        violations.append("checkpoint lacks the direct inverse controller")
    else:
        expected_direct = {
            "controller_kind": "maxent_inverse",
            "controller_rule": "unprojected_warmup_inverse_direct_entropy_v1",
            "base_alpha": DIRECT_BASE_ALPHA,
            "warmup_steps": WARMUP_STEPS,
            "ema_decay": 0.9,
        }
        for key, expected in expected_direct.items():
            if direct.get(key) != expected:
                violations.append(
                    f"direct checkpoint {key}={direct.get(key)!r}, "
                    f"expected {expected!r}"
                )
    replay = state.get("canonical_replay_controller_state")
    if arm != REPLAY_ARM:
        if replay is not None:
            violations.append("non-replay checkpoint unexpectedly has replay controller")
    elif not isinstance(replay, dict):
        violations.append("checkpoint lacks the replay inverse controller")
    else:
        expected_replay = {
            "controller_kind": "canonical_replay_inverse",
            "controller_rule": (
                "unprojected_warmup_inverse_observed_bank_entropy_v1"
            ),
            "base_alpha": REPLAY_BASE_ALPHA,
            "warmup_steps": WARMUP_STEPS,
            "ema_decay": 0.9,
        }
        for key, expected in expected_replay.items():
            if replay.get(key) != expected:
                violations.append(
                    f"replay checkpoint {key}={replay.get(key)!r}, "
                    f"expected {expected!r}"
                )
        if isinstance(bank, dict):
            if (
                bank.get("schema")
                != "online_growing_support_canonical_maxent_replay_v2"
                or bank.get("retain_exemplars") is not True
                or bank.get("replay_capacity") != REPLAY_CAPACITY
                or not isinstance(bank.get("exemplars"), dict)
                or not isinstance(bank.get("prompt_token_ids"), dict)
            ):
                violations.append("checkpoint replay bank schema/config is incompatible")
    return {
        "status": "pass" if not violations else "fail",
        "path": str(path),
        "tag": expected_tag,
        "direct_observations": (
            direct.get("observation_count") if isinstance(direct, dict) else None
        ),
        "replay_observations": (
            replay.get("observation_count") if isinstance(replay, dict) else None
        ),
        "bank_groups_scored": (
            bank.get("groups_scored") if isinstance(bank, dict) else None
        ),
        "violations": violations,
    }


def _latest_by_step(
    records: list[dict[str, Any]], required_key: str
) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    for record in records:
        if required_key not in record:
            continue
        step = _finite(record, "trainer/global_step")
        if step is not None:
            by_step[int(step)] = record
    return [by_step[step] for step in sorted(by_step)]


def _check_direct_controller(
    record: dict[str, Any],
    *,
    step: int,
    reference: float | None,
) -> tuple[float | None, list[str], dict[str, float]]:
    violations: list[str] = []
    required = {
        key: _finite(record, key)
        for key in (
            "train/maxent_conditional_token_entropy",
            "train/maxent_inverse_observed_entropy",
            "train/maxent_inverse_entropy_ema",
            "train/maxent_inverse_multiplier",
            "train/maxent_alpha_used",
            "train/maxent_inverse_next_alpha",
            "train/maxent_inverse_observations",
            "train/maxent_inverse_projection_active",
            "train/maxent_entropy_loss",
        )
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        return reference, [f"step {step}: missing/nonfinite {missing}"], {}
    values = {key: float(value) for key, value in required.items() if value is not None}
    entropy = values["train/maxent_conditional_token_entropy"]
    observed = values["train/maxent_inverse_observed_entropy"]
    ema = values["train/maxent_inverse_entropy_ema"]
    multiplier = values["train/maxent_inverse_multiplier"]
    alpha_used = values["train/maxent_alpha_used"]
    alpha_next = values["train/maxent_inverse_next_alpha"]
    observations = values["train/maxent_inverse_observations"]
    projection = values["train/maxent_inverse_projection_active"]
    entropy_loss = values["train/maxent_entropy_loss"]
    if not math.isclose(entropy, observed, rel_tol=1e-6, abs_tol=1e-8):
        violations.append(f"step {step}: direct objective/controller sensor mismatch")
    if projection != 0:
        violations.append(f"step {step}: direct alpha projection reported active")
    if alpha_used <= 0 or alpha_next <= 0:
        violations.append(f"step {step}: nonpositive direct coefficient")
    if entropy_loss >= 0:
        violations.append(f"step {step}: direct entropy loss is not negative")
    saved_reference = _finite(record, "train/maxent_inverse_reference_entropy")
    if saved_reference is not None:
        reference = saved_reference
    if observations > WARMUP_STEPS:
        if reference is None or reference <= 0 or ema <= 0:
            violations.append(f"step {step}: direct inverse lacks positive state")
        else:
            expected_multiplier = reference / ema
            expected_alpha = DIRECT_BASE_ALPHA * expected_multiplier
            if not math.isclose(
                multiplier, expected_multiplier, rel_tol=2e-5, abs_tol=1e-9
            ):
                violations.append(f"step {step}: direct multiplier mismatch")
            if not math.isclose(
                alpha_next, expected_alpha, rel_tol=2e-5, abs_tol=1e-9
            ):
                violations.append(f"step {step}: direct next-alpha mismatch")
    return reference, violations, {
        "entropy": entropy,
        "entropy_ema": ema,
        "alpha_used": alpha_used,
        "alpha_next": alpha_next,
    }


def _check_replay_controller(
    record: dict[str, Any],
    *,
    step: int,
    reference: float | None,
) -> tuple[float | None, list[str], dict[str, float], bool]:
    violations: list[str] = []
    available_groups = _finite(record, "train/canonical_replay_available_groups")
    available_modes = _finite(record, "train/canonical_replay_available_modes")
    capacity = _finite(record, "train/canonical_replay_capacity")
    projection = _finite(
        record, "train/canonical_replay_alpha_projection_active"
    )
    gold_feedback = _finite(
        record, "train/canonical_replay_gold_support_feedback"
    )
    if available_groups is None or available_modes is None:
        violations.append(f"step {step}: missing replay bank availability")
        return reference, violations, {}, False
    if capacity != REPLAY_CAPACITY:
        violations.append(
            f"step {step}: replay capacity {capacity!r}, expected {REPLAY_CAPACITY}"
        )
    if projection != 0 or gold_feedback != 0:
        violations.append(
            f"step {step}: replay projection/gold-support feedback is active"
        )
    active = available_groups > 0
    skipped = _finite(record, "train/canonical_replay_observation_skipped")
    if not active:
        if available_modes != 0:
            violations.append(f"step {step}: idle replay reports retained modes")
        if skipped != 1:
            violations.append(f"step {step}: idle replay consumed an observation")
        return reference, violations, {}, False

    required = {
        key: _finite(record, key)
        for key in (
            "train/canonical_replay_balance_loss",
            "train/canonical_replay_weighted_loss",
            "train/canonical_replay_backward_scale",
            "train/canonical_replay_chunk_size",
            "train/canonical_replay_score_passes",
            "train/canonical_replay_normalized_model_entropy",
            "train/canonical_replay_cross_entropy_excess",
            "train/canonical_replay_alpha_used",
            "train/canonical_replay_eligible_groups",
            "train/canonical_replay_retained_modes",
            "train/canonical_replay_reward_estimator_scale",
            "train/canonical_replay_observed_normalized_entropy",
            "train/canonical_replay_entropy_ema",
            "train/canonical_replay_inverse_multiplier",
            "train/canonical_replay_alpha_before",
            "train/canonical_replay_next_alpha",
            "train/canonical_replay_observations",
            "train/canonical_replay_projection_active",
        )
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        return (
            reference,
            violations + [f"step {step}: missing/nonfinite replay {missing}"],
            {},
            True,
        )
    values = {key: float(value) for key, value in required.items() if value is not None}
    loss = values["train/canonical_replay_balance_loss"]
    weighted = values["train/canonical_replay_weighted_loss"]
    backward_scale = values["train/canonical_replay_backward_scale"]
    sensor = values["train/canonical_replay_normalized_model_entropy"]
    observed = values["train/canonical_replay_observed_normalized_entropy"]
    entropy_ema = values["train/canonical_replay_entropy_ema"]
    multiplier = values["train/canonical_replay_inverse_multiplier"]
    alpha_used = values["train/canonical_replay_alpha_used"]
    alpha_before = values["train/canonical_replay_alpha_before"]
    alpha_next = values["train/canonical_replay_next_alpha"]
    observations = values["train/canonical_replay_observations"]
    if available_modes < 2 * available_groups:
        violations.append(f"step {step}: replay group has fewer than two modes")
    if (
        values["train/canonical_replay_eligible_groups"] != available_groups
        or values["train/canonical_replay_retained_modes"] != available_modes
    ):
        violations.append(f"step {step}: replay materialization/bank mismatch")
    if loss < -1e-7 or not math.isclose(
        loss,
        values["train/canonical_replay_cross_entropy_excess"],
        rel_tol=1e-6,
        abs_tol=1e-7,
    ):
        violations.append(f"step {step}: replay KL arithmetic mismatch")
    if not 0 < sensor <= 1 + 1e-6 or not math.isclose(
        sensor, observed, rel_tol=1e-6, abs_tol=1e-8
    ):
        violations.append(f"step {step}: replay entropy sensor mismatch")
    if alpha_used <= 0 or not math.isclose(
        alpha_used, alpha_before, rel_tol=1e-6, abs_tol=1e-8
    ):
        violations.append(f"step {step}: replay applied the wrong alpha")
    if not math.isclose(
        weighted,
        loss * alpha_used * REPLAY_ESTIMATOR_SCALE,
        rel_tol=2e-5,
        abs_tol=1e-8,
    ):
        violations.append(f"step {step}: replay weighted-loss mismatch")
    if (
        values["train/canonical_replay_reward_estimator_scale"]
        != REPLAY_ESTIMATOR_SCALE
        or not math.isclose(
            backward_scale,
            alpha_used * REPLAY_ESTIMATOR_SCALE * NUM_SAMPLES,
            rel_tol=2e-5,
            abs_tol=1e-8,
        )
        or values["train/canonical_replay_chunk_size"] != 1
        or values["train/canonical_replay_score_passes"] != 2
    ):
        violations.append(f"step {step}: replay execution scaling mismatch")
    if values["train/canonical_replay_projection_active"] != 0 or skipped != 0:
        violations.append(f"step {step}: active replay projection/skip mismatch")
    saved_reference = _finite(
        record, "train/canonical_replay_reference_entropy"
    )
    if saved_reference is not None:
        reference = saved_reference
    if observations <= WARMUP_STEPS:
        if not math.isclose(
            alpha_next, REPLAY_BASE_ALPHA, rel_tol=1e-6, abs_tol=1e-8
        ) or not math.isclose(multiplier, 1.0, rel_tol=1e-6, abs_tol=1e-8):
            violations.append(f"step {step}: replay warmup coefficient drift")
    elif reference is None or reference <= 0 or entropy_ema <= 0:
        violations.append(f"step {step}: replay inverse lacks positive state")
    else:
        expected_multiplier = reference / entropy_ema
        expected_alpha = REPLAY_BASE_ALPHA * expected_multiplier
        if not math.isclose(
            multiplier, expected_multiplier, rel_tol=2e-5, abs_tol=1e-8
        ):
            violations.append(f"step {step}: replay multiplier mismatch")
        if not math.isclose(
            alpha_next, expected_alpha, rel_tol=2e-5, abs_tol=1e-8
        ):
            violations.append(f"step {step}: replay next-alpha mismatch")
    return reference, violations, {
        "loss": loss,
        "normalized_model_entropy": sensor,
        "entropy_ema": entropy_ema,
        "alpha_used": alpha_used,
        "alpha_next": alpha_next,
        "available_groups": available_groups,
        "available_modes": available_modes,
    }, True


def audit_run(
    records: list[dict[str, Any]],
    *,
    arm: str,
    prompt_pool_size: int,
) -> dict[str, Any]:
    required_key = (
        "train/canonical_replay_available_groups"
        if arm == REPLAY_ARM
        else "train/online_canonical_entropy_alpha_used"
    )
    train = _latest_by_step(records, required_key)
    evaluations = _latest_by_step(records, DISTINCT_KEY)
    prompt_consumed = max(
        (
            value
            for record in records
            if (value := _finite(record, "misc/prompt_consumed")) is not None
        ),
        default=0.0,
    )
    terminal_step = MAX_PASSES * prompt_pool_size
    terminal_evaluations = [
        record
        for record in evaluations
        if int(_finite(record, "trainer/global_step") or -1) == terminal_step
        and all(
            _finite(record, key) is not None
            for key in (DISTINCT_KEY, PASS8_KEY, MEAN8_KEY)
        )
    ]
    prompt_horizon_reached = (
        prompt_consumed >= MAX_PASSES * prompt_pool_size * NUM_SAMPLES
    )
    status = (
        "not_started"
        if not records
        else "complete"
        if prompt_horizon_reached and terminal_evaluations
        else "terminal_eval_pending"
        if prompt_horizon_reached
        else "running"
    )
    violations: list[str] = []
    direct_reference: float | None = None
    replay_reference: float | None = None
    latest: dict[str, float] = {}
    replay_activations = 0
    no_eos_values: list[float] = []
    response_length_values: list[float] = []
    for record in train:
        step = int(_finite(record, "trainer/global_step") or -1)
        gradient = _finite(record, "train/policy_grad_norm")
        no_eos = _finite(record, "actor/no_eos_count")
        response_length = _finite(record, "actor/response_tok_len")
        if gradient is None:
            violations.append(f"step {step}: missing/nonfinite policy gradient")
        if no_eos is None or response_length is None:
            violations.append(f"step {step}: missing/nonfinite generation safety")
        else:
            no_eos_values.append(no_eos)
            response_length_values.append(response_length)
        bank_alpha = _finite(
            record, "train/online_canonical_entropy_alpha_used"
        )
        if bank_alpha is None or abs(bank_alpha) > 1e-8:
            violations.append(
                f"step {step}: sampled count-bank alpha {bank_alpha!r}, expected 0"
            )
        if arm != "grpo":
            direct_reference, direct_violations, direct_latest = (
                _check_direct_controller(
                    record, step=step, reference=direct_reference
                )
            )
            violations.extend(direct_violations)
            latest.update({f"direct_{key}": value for key, value in direct_latest.items()})
        if arm == REPLAY_ARM:
            replay_reference, replay_violations, replay_latest, active = (
                _check_replay_controller(
                    record, step=step, reference=replay_reference
                )
            )
            violations.extend(replay_violations)
            replay_activations += int(active)
            latest.update(
                {f"replay_{key}": value for key, value in replay_latest.items()}
            )
        latest["step"] = float(step)
        if gradient is not None:
            latest["policy_grad_norm"] = gradient
    if status == "complete" and arm == REPLAY_ARM and replay_activations == 0:
        violations.append("terminal replay arm never activated on a multi-mode bank")
    return {
        "status": status,
        "prompt_consumed": prompt_consumed,
        "training_passes": prompt_consumed / (prompt_pool_size * NUM_SAMPLES),
        "train_points": len(train),
        "evaluation_points": len(evaluations),
        "terminal_step": terminal_step,
        "terminal_evaluation_present": bool(terminal_evaluations),
        "replay_activations": replay_activations,
        "latest": latest,
        "safety_window": {
            "points": min(len(no_eos_values), 64),
            "mean_no_eos_count": (
                sum(no_eos_values[-64:]) / min(len(no_eos_values), 64)
                if no_eos_values
                else None
            ),
            "mean_response_length": (
                sum(response_length_values[-64:])
                / min(len(response_length_values), 64)
                if response_length_values
                else None
            ),
        },
        "violations": violations,
        "evaluations": [
            {
                "step": int(_finite(record, "trainer/global_step") or 0),
                "distinct8": _finite(record, DISTINCT_KEY),
                "pass8": _finite(record, PASS8_KEY),
                "mean8": _finite(record, MEAN8_KEY),
            }
            for record in evaluations
        ],
    }


def behavioral_gate(control: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    def usable(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        return {
            row["step"]: row
            for row in rows
            if row["step"] > 0
            and all(row[key] is not None for key in ("distinct8", "pass8", "mean8"))
        }

    control_by_step = usable(control["evaluations"])
    replay_by_step = usable(replay["evaluations"])
    paired_steps = sorted(set(control_by_step) & set(replay_by_step))[-8:]
    if len(paired_steps) < 8:
        return {"status": "pending", "paired_boundaries": len(paired_steps)}
    control_rows = [control_by_step[step] for step in paired_steps]
    replay_rows = [replay_by_step[step] for step in paired_steps]
    control_distinct = sum(row["distinct8"] for row in control_rows) / 8
    replay_distinct = sum(row["distinct8"] for row in replay_rows) / 8
    control_excess = sum(
        row["distinct8"] - row["pass8"] for row in control_rows
    ) / 8
    replay_excess = sum(
        row["distinct8"] - row["pass8"] for row in replay_rows
    ) / 8
    wins = sum(
        replay_row["distinct8"] > control_row["distinct8"]
        for control_row, replay_row in zip(control_rows, replay_rows)
    )
    excess_wins = sum(
        replay_row["distinct8"] - replay_row["pass8"]
        > control_row["distinct8"] - control_row["pass8"]
        for control_row, replay_row in zip(control_rows, replay_rows)
    )
    positive_excess = sum(
        row["distinct8"] > row["pass8"] for row in replay_rows
    )
    all_replay_rows = [
        replay_by_step[step] for step in sorted(replay_by_step)
    ]
    rolling = [
        sum(row["distinct8"] for row in all_replay_rows[end - 7 : end + 1]) / 8
        for end in range(7, len(all_replay_rows))
    ]
    best_rolling = max(rolling)
    retention = replay_distinct / best_rolling if best_rolling > 0 else 0.0
    checks = {
        "higher_mean_distinct8": replay_distinct > control_distinct,
        "wins_at_least_six": wins >= 6,
        "higher_mean_distinct_excess_over_pass": replay_excess > control_excess,
        "excess_wins_at_least_six": excess_wins >= 6,
        "positive_multiplicity_at_least_six": positive_excess >= 6,
        "retains_75pct_of_own_best_rolling_eight": retention >= 0.75,
        "pass8_guardrail": (
            replay_rows[-1]["pass8"] >= control_rows[-1]["pass8"] - 0.03
        ),
        "mean8_guardrail": (
            replay_rows[-1]["mean8"] >= control_rows[-1]["mean8"] - 0.03
        ),
    }
    result = {
        "status": "pass" if all(checks.values()) else "fail",
        "paired_boundaries": 8,
        "steps": paired_steps,
        "control_mean_distinct8": control_distinct,
        "replay_mean_distinct8": replay_distinct,
        "replay_wins": wins,
        "control_mean_distinct_excess_over_pass": control_excess,
        "replay_mean_distinct_excess_over_pass": replay_excess,
        "replay_excess_wins": excess_wins,
        "replay_positive_multiplicity_boundaries": positive_excess,
        "replay_best_rolling_eight_distinct8": best_rolling,
        "replay_self_retention_ratio": retention,
        "checks": checks,
    }
    if control.get("status") != "complete" or replay.get("status") != "complete":
        return {
            "status": "pending",
            "paired_boundaries": len(set(control_by_step) & set(replay_by_step)),
            "provisional_last_eight": result,
        }
    return result


def safety_gate(runs: dict[str, Any]) -> dict[str, Any]:
    control = runs["grpo"]["safety_window"]
    control_no_eos = control["mean_no_eos_count"]
    control_length = control["mean_response_length"]
    if control["points"] == 0 or control_no_eos is None or control_length is None:
        return {"status": "pending"}
    no_eos_limit = max(1.5 * control_no_eos, control_no_eos + 1.0)
    length_limit = max(1.5 * control_length, control_length + 32.0)
    checks: dict[str, bool] = {}
    observations: dict[str, Any] = {}
    for arm in ("maxent_inverse", REPLAY_ARM):
        treatment = runs[arm]["safety_window"]
        no_eos = treatment["mean_no_eos_count"]
        length = treatment["mean_response_length"]
        observations[arm] = {
            "mean_no_eos_count": no_eos,
            "mean_response_length": length,
        }
        checks[f"{arm}_no_eos"] = no_eos is not None and no_eos <= no_eos_limit
        checks[f"{arm}_response_length"] = (
            length is not None and length <= length_limit
        )
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "control_mean_no_eos_count": control_no_eos,
        "control_mean_response_length": control_length,
        "no_eos_limit": no_eos_limit,
        "response_length_limit": length_limit,
        "observations": observations,
        "checks": checks,
    }


def audit(data_root: Path) -> dict[str, Any]:
    binding, binding_violations = approval_binding()
    domains: dict[str, Any] = {}
    for domain, (prefix, prompt_pool_size) in DOMAINS.items():
        runs: dict[str, Any] = {}
        for arm in ARMS:
            run_stamp = f"{prefix}_{arm}_s{SEED}"
            run = audit_run(
                _load_records(data_root, run_stamp),
                arm=arm,
                prompt_pool_size=prompt_pool_size,
            )
            if run["status"] == "complete":
                checkpoint = checkpoint_gate(
                    data_root,
                    run_stamp=run_stamp,
                    arm=arm,
                    terminal_step=run["terminal_step"],
                )
                run["checkpoint_gate"] = checkpoint
                run["violations"].extend(checkpoint["violations"])
            else:
                run["checkpoint_gate"] = {"status": "pending"}
            runs[arm] = run
        domains[domain] = {
            "runs": runs,
            "safety_gate": safety_gate(runs),
            "behavioral_gate": behavioral_gate(
                runs["grpo"], runs[REPLAY_ARM]
            ),
        }
    violations = list(binding_violations) + [
        f"{domain}/{arm}: {violation}"
        for domain, payload in domains.items()
        for arm, run in payload["runs"].items()
        for violation in run["violations"]
    ]
    all_complete = all(
        run["status"] == "complete"
        for payload in domains.values()
        for run in payload["runs"].values()
    )
    gates = [
        payload[gate]["status"]
        for payload in domains.values()
        for gate in ("behavioral_gate", "safety_gate")
    ]
    status = (
        "fail"
        if violations or "fail" in gates
        else "pass"
        if all_complete and all(value == "pass" for value in gates)
        else "in_progress"
    )
    return {
        "schema": "e53_sentinel_audit_v1",
        "status": status,
        "authorizes_stage_a": status == "pass",
        "approval_binding": binding,
        "violations": violations,
        "domains": domains,
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_audit_outputs(
    *,
    payload: dict[str, Any],
    audit_out: Path,
    approval_out: Path,
) -> None:
    _atomic_write(audit_out, payload)
    if payload.get("status") == "pass":
        _atomic_write(approval_out, payload)
    else:
        approval_out.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "var/data")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "var/artifacts/e53_sentinel_audit_latest.json",
    )
    parser.add_argument("--approval-out", type=Path, default=APPROVAL_PATH)
    args = parser.parse_args()
    payload = audit(args.data_root)
    write_audit_outputs(
        payload=payload,
        audit_out=args.out,
        approval_out=args.approval_out,
    )
    print(
        f"[e53-audit] status={payload['status']} "
        f"violations={len(payload['violations'])} out={args.out}"
    )
    for domain, domain_payload in payload["domains"].items():
        statuses = ", ".join(
            f"{arm}={run['status']}@{run['training_passes']:.2f}"
            for arm, run in domain_payload["runs"].items()
        )
        print(
            f"[e53-audit] {domain}: {statuses}; "
            f"behavior={domain_payload['behavioral_gate']['status']}"
        )
    return 1 if payload["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
