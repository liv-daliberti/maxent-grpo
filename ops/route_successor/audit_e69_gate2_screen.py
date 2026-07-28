#!/usr/bin/env python3
"""Audit and summarize the prospectively frozen E69 Gate 2 screen."""

from __future__ import annotations

from collections import defaultdict
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
IDENTITY = ROOT / "var/artifacts/e69_gate2_compute_matched_screen_identity.json"
OUT_JSON = ROOT / "var/artifacts/e69_gate2_compute_matched_screen_audit_latest.json"
OUT_MD = ROOT / "paper/results/e69_gate2_compute_matched_screen_live.md"
PASSES = (0, 1, 2, 3, 4, 5, 6)
POOL_SIZE = {
    "graph_coloring": 192,
    "countdown": 384,
    "python_factor": 384,
    "mathir": 384,
    "math_dev": 384,
}
MODEBENCH = ("graph_coloring", "countdown", "python_factor", "mathir")
CONTROL = "grpo"
SUCCESSOR = "verified_route_successor"
ENDPOINT = "verified_first_global_replay_canonical"
METRICS = ("greedy", "mean8", "pass8", "distinct8")
UNCAUGHT = re.compile(
    r"\[rank\d+\]: Traceback \(most recent call last\)|"
    r"CUDA out of memory|torch\.OutOfMemoryError|ChildFailedError|"
    r"RayActorError|segmentation fault",
    re.IGNORECASE,
)


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _metric(row: dict[str, Any], name: str) -> Any:
    for prefix in ("train/", "actor/", ""):
        key = f"{prefix}{name}"
        if key in row:
            return row[key]
    return None


def _close(value: Any, expected: float, *, tolerance: float = 1e-8) -> bool:
    return _finite(value) and math.isclose(
        float(value),
        float(expected),
        rel_tol=0.0,
        abs_tol=tolerance,
    )


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


def _run_dir(run_stamp: str, job_id: int) -> Path | None:
    candidates = sorted(
        (ROOT / "var/data").glob(f"*_{run_stamp}/debug_job{job_id}")
    )
    return candidates[0] if len(candidates) == 1 else None


def _scan_logs(job_id: int) -> list[str]:
    failures: list[str] = []
    for suffix in ("out", "err"):
        path = ROOT / f"var/artifacts/logs/xdr_train-{job_id}.{suffix}"
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        failures.extend(match.group(0) for match in UNCAUGHT.finditer(text))
    return failures


def _training_audit(
    path: Path,
    *,
    label: str,
    arm: str,
    response_limit: int,
) -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    if not path.is_file():
        return {
            "records": 0,
            "latest_step": -1,
            "route_terminal": {},
        }, violations
    records = 0
    latest_step = -1
    control_groups = 0
    control_rows = 0
    control_prompt_tokens = 0
    control_response_tokens = 0
    control_charged_response_tokens = 0
    replay_prompt_tokens = 0
    replay_response_tokens = 0
    replay_charged_response_tokens = 0
    replay_records = 0
    route_terminal: dict[str, float] = {}
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            violations.append(f"{label}: invalid train JSON line {line_number}")
            continue
        if not isinstance(row, dict):
            violations.append(f"{label}: non-object train line {line_number}")
            continue
        step = row.get("trainer/global_step", row.get("misc/global_step", -1))
        if _finite(step):
            latest_step = max(latest_step, int(step))
        if not any(key.startswith(("train/", "actor/")) for key in row):
            continue
        records += 1
        for key, value in row.items():
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and not math.isfinite(float(value))
            ):
                violations.append(f"{label}: non-finite {key} line {line_number}")

        expected = {
            "counterfactual_fixed_control_groups_generated": 3.0,
            "counterfactual_fixed_control_rows_generated": 48.0,
            "counterfactual_fixed_control_rows_sent_to_ppo": 0.0,
        }
        for name, target in expected.items():
            value = _metric(row, name)
            if not _close(value, target):
                violations.append(
                    f"{label}: {name}={value!r}, expected {target} "
                    f"at line {line_number}"
                )
        control_groups += int(
            _metric(row, "counterfactual_fixed_control_groups_generated") or 0
        )
        control_rows += int(
            _metric(row, "counterfactual_fixed_control_rows_generated") or 0
        )
        control_prompt_tokens += int(
            _metric(
                row,
                "counterfactual_fixed_control_realized_prompt_tokens",
            )
            or 0
        )
        control_response_tokens += int(
            _metric(
                row,
                "counterfactual_fixed_control_realized_response_tokens",
            )
            or 0
        )
        charged_control = _metric(
            row,
            "counterfactual_fixed_control_charged_response_token_budget",
        )
        if not _close(charged_control, 48 * response_limit):
            violations.append(
                f"{label}: fixed-control charged budget={charged_control!r}, "
                f"expected {48 * response_limit} at line {line_number}"
            )
        control_charged_response_tokens += int(charged_control or 0)

        replay_charged = _metric(
            row,
            "canonical_replay_charged_response_token_budget",
        )
        if not _close(replay_charged, 16 * response_limit):
            violations.append(
                f"{label}: replay charged budget={replay_charged!r}, "
                f"expected {16 * response_limit} at line {line_number}"
            )
        replay_charged_response_tokens += int(replay_charged or 0)
        replay_prompt_tokens += int(
            _metric(row, "canonical_replay_realized_prompt_tokens") or 0
        )
        replay_response_tokens += int(
            _metric(row, "canonical_replay_realized_response_tokens") or 0
        )

        replay_passes = _metric(row, "canonical_replay_score_passes")
        if _finite(replay_passes):
            replay_records += 1
            if not _close(replay_passes, 2.0):
                violations.append(
                    f"{label}: replay score passes={replay_passes!r} "
                    f"at line {line_number}"
                )
            if arm == CONTROL:
                for name in (
                    "canonical_replay_compute_only",
                    "canonical_replay_compute_only_configured",
                ):
                    if not _close(_metric(row, name), 1.0):
                        violations.append(
                            f"{label}: Dr.GRPO {name} is not one "
                            f"at line {line_number}"
                        )
                for name in (
                    "canonical_replay_weighted_loss",
                    "canonical_replay_applied_score_gradient_sum",
                    "canonical_replay_applied_score_gradient_l2",
                ):
                    if not _close(_metric(row, name), 0.0):
                        violations.append(
                            f"{label}: Dr.GRPO {name} is nonzero "
                            f"at line {line_number}"
                        )
        for key, value in row.items():
            if key.endswith(
                (
                    "rows_sent_to_ppo",
                    "gold_support_feedback",
                    "eval_feedback",
                    "desired_mode_count_feedback",
                    "projection_active",
                )
            ) and _finite(value) and not _close(value, 0.0):
                violations.append(
                    f"{label}: forbidden nonzero {key} at line {line_number}"
                )
        for name in (
            "verified_route_post_replay_cross_prompt_neutral_reproductions",
            "verified_route_cross_prompt_replay_updates",
            "verified_route_cross_prompt_replay_groups",
            "verified_route_proposal_graduations",
        ):
            value = _metric(row, name)
            if _finite(value):
                route_terminal[name.removeprefix("verified_route_")] = float(value)

    return (
        {
            "records": records,
            "latest_step": latest_step,
            "fixed_control_groups": control_groups,
            "fixed_control_rows": control_rows,
            "fixed_control_realized_prompt_tokens": control_prompt_tokens,
            "fixed_control_realized_response_tokens": control_response_tokens,
            "fixed_control_charged_response_tokens": (
                control_charged_response_tokens
            ),
            "replay_records": replay_records,
            "replay_realized_prompt_tokens": replay_prompt_tokens,
            "replay_realized_response_tokens": replay_response_tokens,
            "replay_charged_response_tokens": replay_charged_response_tokens,
            "route_terminal": route_terminal,
        },
        violations,
    )


def _evaluations(path: Path) -> tuple[dict[int, dict[str, float]], list[str]]:
    by_key: dict[tuple[int, str, int | None], dict[str, Any]] = {}
    violations: list[str] = []
    if not path.is_file():
        return {}, violations
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            violations.append(f"{path}: invalid JSON line {line_number}")
            continue
        if not isinstance(row, dict) or not isinstance(row.get("metrics"), dict):
            continue
        kind = str(row.get("evaluation_kind", ""))
        if kind not in {
            "deterministic_greedy_trace_neutral",
            "fixed_seed_sampled_k_neutral",
        }:
            continue
        step = int(row["step"])
        draw = row.get("draw_index")
        key = (step, kind, draw if isinstance(draw, int) else None)
        prior = by_key.get(key)
        if prior is not None and prior.get("metrics") != row.get("metrics"):
            violations.append(f"{path}: conflicting repeated evaluation {key}")
        by_key[key] = row

    result: dict[int, dict[str, float]] = defaultdict(dict)
    for (step, kind, _draw), row in by_key.items():
        metrics = row["metrics"]
        if kind == "deterministic_greedy_trace_neutral":
            result[step]["greedy"] = float(metrics["any_correct_at_k"])
        else:
            result[step]["mean8"] = float(metrics["mean_at_k"])
            result[step]["pass8"] = float(metrics["any_correct_at_k"])
            result[step]["distinct8"] = float(
                metrics["distinct_correct_modes_at_k"]
            )
    return dict(result), violations


def evaluate_gate(
    curves: dict[str, dict[str, dict[int, dict[str, float]]]],
    route_terminal: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Evaluate only the prospectively frozen E69-minus-control thresholds."""

    checks: dict[str, bool] = {}
    deltas: dict[str, dict[int, dict[str, float]]] = {}
    for domain in POOL_SIZE:
        treatment = ENDPOINT if domain == "math_dev" else SUCCESSOR
        deltas[domain] = {}
        for pass_index in PASSES:
            deltas[domain][pass_index] = {
                metric: (
                    curves[domain][treatment][pass_index][metric]
                    - curves[domain][CONTROL][pass_index][metric]
                )
                for metric in METRICS
            }

    for pass_index in (5, 6):
        checks[f"quality_noninferior_pass_{pass_index}"] = all(
            deltas[domain][pass_index][metric] >= -0.02 - 1e-12
            for domain in POOL_SIZE
            for metric in ("greedy", "pass8")
        )
        checks[f"three_modebench_distinct_gains_pass_{pass_index}"] = (
            sum(
                deltas[domain][pass_index]["distinct8"] > 0
                for domain in MODEBENCH
            )
            >= 3
        )
        checks[f"mathir_pass_and_distinct_gain_pass_{pass_index}"] = (
            deltas["mathir"][pass_index]["pass8"] > 0
            and deltas["mathir"][pass_index]["distinct8"] > 0
        )

    math_delta = deltas["math_dev"][6]
    checks["math_parent_quality_gate"] = (
        math_delta["greedy"] >= -0.01 - 1e-12
        and math_delta["pass8"] >= -0.01 - 1e-12
        and (math_delta["greedy"] > 0 or math_delta["pass8"] > 0)
    )
    checks["math_greedy_mean_gate"] = (
        math_delta["greedy"] >= -0.01 - 1e-12
        and math_delta["mean8"] >= -0.01 - 1e-12
        and (math_delta["greedy"] > 0 or math_delta["mean8"] > 0)
    )
    checks["post_replay_reuse_three_domains"] = (
        sum(
            route_terminal.get(domain, {}).get(
                "post_replay_cross_prompt_neutral_reproductions",
                0.0,
            )
            > 0
            for domain in MODEBENCH
        )
        >= 3
    )
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "deltas": {
            domain: {str(key): value for key, value in by_pass.items()}
            for domain, by_pass in deltas.items()
        },
    }


def _write_markdown(payload: dict[str, Any]) -> None:
    lines = [
        "# E69 Gate 2 compute-matched screen",
        "",
        f"Status: **{payload['status']}**.",
        "",
        "MATH-500 remains sealed. MATH E66/E68/E69 are one physical "
        "endpoint-only run with three reporting aliases.",
        "",
        "| Domain | Arm | Pass | Greedy | Mean@8 | Pass@8 | Distinct@8 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for domain, arms in payload.get("curves", {}).items():
        for arm, by_pass in arms.items():
            for pass_index in PASSES:
                row = by_pass.get(str(pass_index))
                if not row:
                    continue
                lines.append(
                    f"| {domain} | {arm} | {pass_index} | "
                    f"{row['greedy']:.4f} | {row['mean8']:.4f} | "
                    f"{row['pass8']:.4f} | {row['distinct8']:.4f} |"
                )
    if payload.get("outcome_gate"):
        lines.extend(["", "## Frozen gate checks", ""])
        for name, passed in payload["outcome_gate"]["checks"].items():
            lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
    if payload.get("pending"):
        lines.extend(["", "## Pending", ""])
        lines.extend(f"- {value}" for value in payload["pending"])
    if payload.get("violations"):
        lines.extend(["", "## Integrity violations", ""])
        lines.extend(f"- {value}" for value in payload["violations"])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not IDENTITY.is_file():
        raise SystemExit(f"E69 Gate 2 identity is absent: {IDENTITY}")
    identity = json.loads(IDENTITY.read_text(encoding="utf-8"))
    if identity.get("schema") != "e69_gate2_compute_matched_screen_v1":
        raise SystemExit("unexpected E69 Gate 2 identity schema")

    violations: list[str] = []
    pending: list[str] = []
    physical_runs: list[dict[str, Any]] = []
    curves: dict[str, dict[str, dict[int, dict[str, float]]]] = defaultdict(dict)
    route_terminal: dict[str, dict[str, float]] = {}
    for domain, jobs in identity["jobs"].items():
        expected_step = POOL_SIZE[domain] * 6
        response_limit = 1024 if domain == "math_dev" else (
            64 if domain == "mathir" else 192
        )
        for job in jobs:
            arm = str(job["arm"])
            job_id = int(job["job_id"])
            label = f"{domain}/{arm}/job{job_id}"
            run_dir = _run_dir(str(job["run_stamp"]), job_id)
            if run_dir is None:
                pending.append(f"{label}: run directory absent")
                physical_runs.append(
                    {
                        "domain": domain,
                        "arm": arm,
                        "job_id": job_id,
                        "run_dir": None,
                        "terminal": False,
                    }
                )
                violations.extend(
                    f"{label}: {failure}" for failure in _scan_logs(job_id)
                )
                continue
            training, run_violations = _training_audit(
                run_dir / "train_metrics.jsonl",
                label=label,
                arm=arm,
                response_limit=response_limit,
            )
            violations.extend(run_violations)
            violations.extend(
                f"{label}: {failure}" for failure in _scan_logs(job_id)
            )
            evaluations, eval_violations = _evaluations(
                run_dir / "eval_mode_coverage_draws.jsonl"
            )
            violations.extend(f"{label}: {value}" for value in eval_violations)
            pass_curve: dict[int, dict[str, float]] = {}
            for pass_index in PASSES:
                step = POOL_SIZE[domain] * pass_index
                row = evaluations.get(step)
                if row is None or not all(metric in row for metric in METRICS):
                    pending.append(f"{label}: missing evaluation pass {pass_index}")
                else:
                    pass_curve[pass_index] = {
                        metric: float(row[metric]) for metric in METRICS
                    }
            curves[domain][arm] = pass_curve
            terminal = (
                int(training["latest_step"]) >= expected_step
                and len(pass_curve) == len(PASSES)
            )
            if not terminal:
                pending.append(
                    f"{label}: latest step {training['latest_step']}/{expected_step}"
                )
            if arm == SUCCESSOR:
                route_terminal[domain] = dict(training["route_terminal"])
            physical_runs.append(
                {
                    "domain": domain,
                    "arm": arm,
                    "job_id": job_id,
                    "run_dir": str(run_dir.resolve()),
                    "terminal": terminal,
                    "training": training,
                    "evaluations": {
                        str(key): value for key, value in pass_curve.items()
                    },
                }
            )

    if "math_dev" in curves and ENDPOINT in curves["math_dev"]:
        curves["math_dev"][SUCCESSOR] = curves["math_dev"][ENDPOINT]
        curves["math_dev"]["verified_entropy_gated_singleton_escape_canonical"] = (
            curves["math_dev"][ENDPOINT]
        )

    complete = (
        len(physical_runs) == 18
        and all(run.get("terminal") for run in physical_runs)
        and not pending
    )
    outcome_gate = None
    if complete and not violations:
        outcome_gate = evaluate_gate(curves, route_terminal)
    status = (
        "fail"
        if violations
        else outcome_gate["status"]
        if outcome_gate is not None
        else "in_progress"
    )
    payload = {
        "schema": "e69_gate2_compute_matched_screen_audit_v1",
        "status": status,
        "identity": str(IDENTITY.resolve()),
        "math500_sealed": True,
        "summary": {
            "expected_physical_runs": 18,
            "observed_physical_runs": len(physical_runs),
            "terminal_physical_runs": sum(
                bool(run.get("terminal")) for run in physical_runs
            ),
            "pending_items": len(set(pending)),
            "integrity_violations": len(set(violations)),
        },
        "physical_runs": physical_runs,
        "curves": {
            domain: {
                arm: {str(key): value for key, value in by_pass.items()}
                for arm, by_pass in arms.items()
            }
            for domain, arms in curves.items()
        },
        "route_terminal": route_terminal,
        "outcome_gate": outcome_gate,
        "pending": sorted(set(pending)),
        "violations": sorted(set(violations)),
    }
    _atomic_json(OUT_JSON, payload)
    _write_markdown(payload)
    print(
        f"[e69-gate2-audit] status={status} "
        f"terminal={payload['summary']['terminal_physical_runs']}/18 "
        f"pending={payload['summary']['pending_items']} "
        f"violations={payload['summary']['integrity_violations']}"
    )


if __name__ == "__main__":
    main()
