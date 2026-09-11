#!/usr/bin/env python3
"""Audit E52's live or terminal three-domain sentinel without label feedback."""

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
ARMS = ("grpo", "maxent_inverse", "maxent_inverse_canonical")
DOMAINS = {
    "countdown": (
        "cde52_direct_inverse_entropy_canonical_05b_50ep_sentinel_v2_allcs",
        384,
    ),
    "graph_coloring": (
        "gce52_direct_inverse_entropy_canonical_05b_50ep_sentinel_v2",
        192,
    ),
    "python_factor": (
        "pye52_direct_inverse_entropy_canonical_05b_50ep_sentinel_v2_allcs",
        384,
    ),
}
BASE_ALPHA = 0.000075
WARMUP_STEPS = 64
MAX_PASSES = 50
NUM_SAMPLES = 16
DISTINCT_KEY = "eval/multi_answer/sampled_distinct_correct_at_8"
PASS8_KEY = "eval/multi_answer/sampled_any_correct_at_8"
MEAN8_KEY = "eval/multi_answer/sampled_mean_at_8"
IDENTITY_PATH = (
    ROOT
    / "var/artifacts/"
    "e52_direct_inverse_entropy_canonical_05b_sentinel_v2_identity.json"
)
PROTOCOL_PATH = (
    ROOT
    / "paper/preregistration/"
    "e52_direct_inverse_entropy_canonical_05b.md"
)
REPAIR_PROTOCOL_PATH = (
    ROOT
    / "paper/preregistration/"
    "e52_runtime_validation_repair_20260726.md"
)
STABILITY_AMENDMENT_PATH = (
    ROOT
    / "paper/preregistration/"
    "e52_scale_free_stability_gate_amendment_20260726.md"
)
LAUNCHER_PATH = (
    ROOT
    / "ops/exp_scaling/"
    "launch_e52_direct_inverse_entropy_canonical_05b.sh"
)
APPROVAL_PATH = (
    ROOT / "var/artifacts/e52_sentinel_stage_a_approval.json"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tree(root: Path) -> str:
    """Reproduce the launcher's sorted ``sha256sum | sha256sum`` identity."""

    lines = []
    for path in sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
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
        != "e52_direct_inverse_entropy_canonical_05b_sentinel_v2"
    ):
        return {}, ["sentinel identity has an incompatible schema"]

    source_hash = str(identity.get("source_hash", ""))
    execution_hash = str(identity.get("execution_surface_hash", ""))
    source_root = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e52_direct_inverse_entropy_{source_hash}"
        / "src"
    )
    ops_root = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e52_direct_inverse_entropy_ops_{execution_hash}"
        / "ops"
    )
    try:
        observed = {
            "identity_sha256": _sha256_file(IDENTITY_PATH),
            "protocol_sha256": _sha256_file(PROTOCOL_PATH),
            "repair_protocol_sha256": _sha256_file(REPAIR_PROTOCOL_PATH),
            "stability_amendment_sha256": _sha256_file(
                STABILITY_AMENDMENT_PATH
            ),
            "sentinel_launcher_sha256": _sha256_file(LAUNCHER_PATH),
            "source_hash": _hash_tree(source_root),
            "execution_surface_hash": _hash_tree(ops_root),
            "auditor_sha256": _sha256_file(Path(__file__).resolve()),
        }
    except OSError as error:
        return {}, [f"sentinel approval evidence is unavailable: {error}"]
    expected = {
        "protocol_sha256": identity.get("protocol_sha256"),
        "repair_protocol_sha256": identity.get("repair_protocol_sha256"),
        "sentinel_launcher_sha256": identity.get("launcher_sha256"),
        "source_hash": identity.get("source_hash"),
        "execution_surface_hash": identity.get("execution_surface_hash"),
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


def audit_run(
    records: list[dict[str, Any]],
    *,
    arm: str,
    prompt_pool_size: int,
) -> dict[str, Any]:
    train = _latest_by_step(
        records,
        (
            "train/maxent_inverse_observations"
            if arm != "grpo"
            else "train/online_canonical_entropy_alpha_used"
        ),
    )
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
    reference: float | None = None
    latest: dict[str, float] = {}
    no_eos_values: list[float] = []
    response_length_values: list[float] = []
    mechanism_values: dict[str, list[float]] = {
        key: []
        for key in (
            "train/online_canonical_task_reward_mean",
            "train/online_canonical_canonicalizable_correct_fraction",
            "train/online_canonical_support_at_least_two_prompt_fraction",
            "train/online_canonical_entropy_advantage_rms",
            "train/online_canonical_novelty_advantage_rms",
        )
    }

    for record in train:
        step = int(_finite(record, "trainer/global_step") or -1)
        policy_grad_norm = _finite(record, "train/policy_grad_norm")
        if policy_grad_norm is None:
            violations.append(f"step {step}: missing/nonfinite policy gradient norm")
        no_eos_count = _finite(record, "actor/no_eos_count")
        response_length = _finite(record, "actor/response_tok_len")
        if no_eos_count is None or response_length is None:
            violations.append(
                f"step {step}: missing/nonfinite length or no-EOS telemetry"
            )
        else:
            no_eos_values.append(no_eos_count)
            response_length_values.append(response_length)
        for key, values in mechanism_values.items():
            value = _finite(record, key)
            if value is not None:
                values.append(value)
        canonical_alpha = _finite(
            record, "train/online_canonical_entropy_alpha_used"
        )
        expected_canonical = (
            0.10 if arm == "maxent_inverse_canonical" else 0.0
        )
        if canonical_alpha is None or not math.isclose(
            canonical_alpha, expected_canonical, rel_tol=1e-6, abs_tol=1e-8
        ):
            violations.append(
                f"step {step}: canonical alpha {canonical_alpha!r}, "
                f"expected {expected_canonical}"
            )
        if arm == "grpo":
            continue

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
            violations.append(f"step {step}: missing/nonfinite {missing}")
            continue
        entropy = required["train/maxent_conditional_token_entropy"]
        observed = required["train/maxent_inverse_observed_entropy"]
        ema = required["train/maxent_inverse_entropy_ema"]
        multiplier = required["train/maxent_inverse_multiplier"]
        alpha_used = required["train/maxent_alpha_used"]
        alpha_next = required["train/maxent_inverse_next_alpha"]
        observations = required["train/maxent_inverse_observations"]
        projection = required["train/maxent_inverse_projection_active"]
        entropy_loss = required["train/maxent_entropy_loss"]
        assert all(
            value is not None
            for value in (
                entropy,
                observed,
                ema,
                multiplier,
                alpha_used,
                alpha_next,
                observations,
                projection,
                entropy_loss,
            )
        )
        if not math.isclose(entropy, observed, rel_tol=1e-6, abs_tol=1e-8):
            violations.append(f"step {step}: objective/controller sensor mismatch")
        if projection != 0:
            violations.append(f"step {step}: projection reported active")
        if alpha_used <= 0 or alpha_next <= 0:
            violations.append(f"step {step}: nonpositive direct coefficient")
        if entropy_loss >= 0:
            violations.append(f"step {step}: direct entropy loss is not negative")
        saved_reference = _finite(
            record, "train/maxent_inverse_reference_entropy"
        )
        if saved_reference is not None:
            reference = saved_reference
        if observations > WARMUP_STEPS:
            if reference is None or reference <= 0 or ema <= 0:
                violations.append(
                    f"step {step}: post-warmup inverse lacks positive state"
                )
            else:
                expected_multiplier = reference / ema
                expected_alpha = BASE_ALPHA * expected_multiplier
                if not math.isclose(
                    multiplier,
                    expected_multiplier,
                    rel_tol=2e-5,
                    abs_tol=1e-9,
                ):
                    violations.append(
                        f"step {step}: inverse multiplier arithmetic mismatch"
                    )
                if not math.isclose(
                    alpha_next,
                    expected_alpha,
                    rel_tol=2e-5,
                    abs_tol=1e-9,
                ):
                    violations.append(
                        f"step {step}: next-alpha arithmetic mismatch"
                    )
        latest = {
            "step": float(step),
            "entropy": entropy,
            "entropy_ema": ema,
            "reference_entropy": reference or 0.0,
            "multiplier": multiplier,
            "alpha_used": alpha_used,
            "alpha_next": alpha_next,
            "entropy_loss": entropy_loss,
            "canonical_alpha": canonical_alpha or 0.0,
            "policy_grad_norm": policy_grad_norm or 0.0,
            "no_eos_count": no_eos_count or 0.0,
            "response_length": response_length or 0.0,
        }

    if status == "complete" and arm != "grpo":
        entropies = [
            _finite(record, "train/maxent_conditional_token_entropy")
            for record in train[-64:]
        ]
        finite_entropies = [value for value in entropies if value is not None]
        if (
            reference is None
            or len(finite_entropies) < 64
            or sum(finite_entropies) / 64 < 0.5 * reference
        ):
            violations.append("terminal trailing-64 entropy retention gate failed")

    return {
        "status": status,
        "prompt_consumed": prompt_consumed,
        "training_passes": prompt_consumed / (prompt_pool_size * NUM_SAMPLES),
        "train_points": len(train),
        "evaluation_points": len(evaluations),
        "terminal_step": terminal_step,
        "terminal_evaluation_present": bool(terminal_evaluations),
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
        "mechanism_window": {
            "points": min(len(train), 64),
            **{
                key.removeprefix("train/"): (
                    sum(values[-64:]) / min(len(values), 64)
                    if values
                    else None
                )
                for key, values in mechanism_values.items()
            },
            "tracked_outcomes": (
                _finite(
                    train[-1],
                    "train/online_canonical_tracked_outcomes",
                )
                if train
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


def behavioral_gate(control: dict[str, Any], hybrid: dict[str, Any]) -> dict[str, Any]:
    control_by_step = {
        row["step"]: row
        for row in control["evaluations"]
        if (
            row["step"] > 0
            and row["distinct8"] is not None
            and row["pass8"] is not None
            and row["mean8"] is not None
        )
    }
    hybrid_by_step = {
        row["step"]: row
        for row in hybrid["evaluations"]
        if (
            row["step"] > 0
            and row["distinct8"] is not None
            and row["pass8"] is not None
            and row["mean8"] is not None
        )
    }
    paired_steps = sorted(set(control_by_step) & set(hybrid_by_step))[-8:]
    if len(paired_steps) < 8:
        return {"status": "pending", "paired_boundaries": len(paired_steps)}
    control_rows = [control_by_step[step] for step in paired_steps]
    hybrid_rows = [hybrid_by_step[step] for step in paired_steps]
    control_distinct = sum(row["distinct8"] for row in control_rows) / 8
    hybrid_distinct = sum(row["distinct8"] for row in hybrid_rows) / 8
    control_excess = sum(
        row["distinct8"] - row["pass8"] for row in control_rows
    ) / 8
    hybrid_excess = sum(
        row["distinct8"] - row["pass8"] for row in hybrid_rows
    ) / 8
    wins = sum(
        hybrid_row["distinct8"] > control_row["distinct8"]
        for control_row, hybrid_row in zip(control_rows, hybrid_rows)
    )
    excess_wins = sum(
        hybrid_row["distinct8"] - hybrid_row["pass8"]
        > control_row["distinct8"] - control_row["pass8"]
        for control_row, hybrid_row in zip(control_rows, hybrid_rows)
    )
    multiplicity_boundaries = sum(
        row["distinct8"] > row["pass8"] for row in hybrid_rows
    )
    all_hybrid_rows = [
        hybrid_by_step[step] for step in sorted(hybrid_by_step)
    ]
    rolling_hybrid_distinct = [
        sum(row["distinct8"] for row in all_hybrid_rows[end - 7 : end + 1])
        / 8
        for end in range(7, len(all_hybrid_rows))
    ]
    best_rolling_hybrid_distinct = max(rolling_hybrid_distinct)
    self_retention_ratio = (
        hybrid_distinct / best_rolling_hybrid_distinct
        if best_rolling_hybrid_distinct > 0
        else 0.0
    )
    terminal_control = control_rows[-1]
    terminal_hybrid = hybrid_rows[-1]
    checks = {
        "higher_mean_distinct8": hybrid_distinct > control_distinct,
        "wins_at_least_six": wins >= 6,
        "higher_mean_distinct_excess_over_pass": (
            hybrid_excess > control_excess
        ),
        "excess_wins_at_least_six": excess_wins >= 6,
        "positive_multiplicity_at_least_six": (
            multiplicity_boundaries >= 6
        ),
        "retains_half_of_own_best_rolling_eight": (
            self_retention_ratio >= 0.5
        ),
        "pass8_guardrail": (
            terminal_hybrid["pass8"] is not None
            and terminal_control["pass8"] is not None
            and terminal_hybrid["pass8"] >= terminal_control["pass8"] - 0.03
        ),
        "mean8_guardrail": (
            terminal_hybrid["mean8"] is not None
            and terminal_control["mean8"] is not None
            and terminal_hybrid["mean8"] >= terminal_control["mean8"] - 0.03
        ),
    }
    result = {
        "status": (
            "pass"
            if all(checks.values())
            else "fail"
        ),
        "paired_boundaries": 8,
        "steps": paired_steps,
        "control_mean_distinct8": control_distinct,
        "hybrid_mean_distinct8": hybrid_distinct,
        "hybrid_wins": wins,
        "control_mean_distinct_excess_over_pass": control_excess,
        "hybrid_mean_distinct_excess_over_pass": hybrid_excess,
        "hybrid_excess_wins": excess_wins,
        "hybrid_positive_multiplicity_boundaries": multiplicity_boundaries,
        "hybrid_best_rolling_eight_distinct8": (
            best_rolling_hybrid_distinct
        ),
        "hybrid_self_retention_ratio": self_retention_ratio,
        "checks": checks,
    }
    if control.get("status") != "complete" or hybrid.get("status") != "complete":
        return {
            "status": "pending",
            "paired_boundaries": len(set(control_by_step) & set(hybrid_by_step)),
            "provisional_last_eight": result,
        }
    return result


def safety_gate(runs: dict[str, Any]) -> dict[str, Any]:
    control = runs["grpo"]["safety_window"]
    if control["points"] == 0:
        return {"status": "pending"}
    control_no_eos = control["mean_no_eos_count"]
    control_length = control["mean_response_length"]
    if control_no_eos is None or control_length is None:
        return {"status": "pending"}
    no_eos_limit = max(1.5 * control_no_eos, control_no_eos + 1.0)
    length_limit = max(1.5 * control_length, control_length + 32.0)
    checks: dict[str, bool] = {}
    observations: dict[str, Any] = {}
    for arm in ("maxent_inverse", "maxent_inverse_canonical"):
        treatment = runs[arm]["safety_window"]
        no_eos = treatment["mean_no_eos_count"]
        length = treatment["mean_response_length"]
        observations[arm] = {
            "mean_no_eos_count": no_eos,
            "mean_response_length": length,
        }
        checks[f"{arm}_no_eos"] = (
            no_eos is not None and no_eos <= no_eos_limit
        )
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
        runs = {
            arm: audit_run(
                _load_records(data_root, f"{prefix}_{arm}_s9009"),
                arm=arm,
                prompt_pool_size=prompt_pool_size,
            )
            for arm in ARMS
        }
        domains[domain] = {
            "runs": runs,
            "safety_gate": safety_gate(runs),
            "behavioral_gate": behavioral_gate(
                runs["grpo"], runs["maxent_inverse_canonical"]
            ),
        }
    violations = list(binding_violations) + [
        f"{domain}/{arm}: {violation}"
        for domain, domain_payload in domains.items()
        for arm, run in domain_payload["runs"].items()
        for violation in run["violations"]
    ]
    all_complete = all(
        run["status"] == "complete"
        for domain_payload in domains.values()
        for run in domain_payload["runs"].values()
    )
    behavioral = [
        payload["behavioral_gate"]["status"] for payload in domains.values()
    ]
    safety = [payload["safety_gate"]["status"] for payload in domains.values()]
    status = (
        "fail"
        if violations or "fail" in behavioral or "fail" in safety
        else "pass"
        if (
            all_complete
            and all(value == "pass" for value in behavioral)
            and all(value == "pass" for value in safety)
        )
        else "in_progress"
    )
    return {
        "schema": "e52_sentinel_audit_v2",
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
        default=ROOT / "var/artifacts/e52_sentinel_audit_latest.json",
    )
    parser.add_argument(
        "--approval-out",
        type=Path,
        default=APPROVAL_PATH,
    )
    args = parser.parse_args()
    payload = audit(args.data_root)
    write_audit_outputs(
        payload=payload,
        audit_out=args.out,
        approval_out=args.approval_out,
    )
    print(
        f"[e52-audit] status={payload['status']} "
        f"violations={len(payload['violations'])} out={args.out}"
    )
    for domain, domain_payload in payload["domains"].items():
        statuses = ", ".join(
            f"{arm}={run['status']}@{run['training_passes']:.2f}"
            for arm, run in domain_payload["runs"].items()
        )
        print(
            f"[e52-audit] {domain}: {statuses}; "
            f"behavior={domain_payload['behavioral_gate']['status']}"
        )
    return 1 if payload["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
