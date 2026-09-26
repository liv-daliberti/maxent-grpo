#!/usr/bin/env python3
"""Audit and summarize the frozen E69 Gate 3 confirmatory panel."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any

from ops.route_successor.audit_e69_gate2_screen import (
    METRICS,
    MODEBENCH,
    PASSES,
    POOL_SIZE,
    _evaluations,
    _run_dir,
    _scan_logs,
    _training_audit,
)


ROOT = Path(__file__).resolve().parents[2]
IDENTITY = ROOT / "var/artifacts/e69_gate3_confirmatory_identity.json"
OUT_JSON = ROOT / "var/artifacts/e69_gate3_confirmatory_audit_latest.json"
OUT_CSV = ROOT / "paper/results/e69_gate3_confirmatory_curves_live.csv"
OUT_MD = ROOT / "paper/results/e69_gate3_confirmatory_live.md"
CONTROL = "grpo"
SUCCESSOR = "verified_route_successor"
ENDPOINT = "verified_first_global_replay_canonical"
SEEDS = (43, 44, 45)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 690301


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


def _successor(domain: str) -> str:
    return ENDPOINT if domain == "math_dev" else SUCCESSOR


def crossed_bootstrap_interval(
    paired_prompt_deltas: dict[int, list[float]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Crossed paired-seed/prompt percentile interval for a mean delta."""

    if set(paired_prompt_deltas) != set(SEEDS):
        raise ValueError("crossed bootstrap requires paired seeds 43, 44, and 45")
    if replicates <= 0:
        raise ValueError("crossed bootstrap replicates must be positive")
    arrays: dict[int, tuple[float, ...]] = {}
    for paired_seed, values in paired_prompt_deltas.items():
        array = tuple(float(value) for value in values)
        if not array or not all(math.isfinite(value) for value in array):
            raise ValueError(f"invalid prompt deltas for seed {paired_seed}")
        arrays[paired_seed] = array

    generator = random.Random(seed)
    draws: list[float] = []
    for _ in range(replicates):
        seed_means: list[float] = []
        for _ in SEEDS:
            paired_seed = generator.choice(SEEDS)
            values = arrays[paired_seed]
            seed_means.append(
                sum(generator.choice(values) for _ in values) / len(values)
            )
        draws.append(sum(seed_means) / len(seed_means))
    draws.sort()
    lower_index = max(0, math.floor(0.025 * (replicates - 1)))
    upper_index = min(replicates - 1, math.ceil(0.975 * (replicates - 1)))
    return draws[lower_index], draws[upper_index]


def _prompt_metrics(
    path: Path,
    *,
    step: int,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Return terminal per-prompt metrics keyed by exact prompt text."""

    selected: dict[tuple[str, int | None], dict[str, Any]] = {}
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
            continue
        if not isinstance(row, dict) or row.get("step") != step:
            continue
        kind = str(row.get("evaluation_kind", ""))
        if kind not in {
            "deterministic_greedy_trace_neutral",
            "fixed_seed_sampled_k_neutral",
        }:
            continue
        draw = row.get("draw_index")
        key = (kind, draw if isinstance(draw, int) else None)
        prior = selected.get(key)
        if prior is not None and prior.get("prompts") != row.get("prompts"):
            violations.append(
                f"{path}: conflicting prompt evaluation {key} at line "
                f"{line_number}"
            )
        selected[key] = row

    result: dict[str, dict[str, float]] = defaultdict(dict)
    for (kind, _draw), row in selected.items():
        prompts = row.get("prompts")
        if not isinstance(prompts, list):
            violations.append(f"{path}: evaluation has no prompt records")
            continue
        for prompt_row in prompts:
            if not isinstance(prompt_row, dict):
                continue
            prompt = prompt_row.get("prompt")
            metrics = prompt_row.get("metrics")
            if not isinstance(prompt, str) or not isinstance(metrics, dict):
                continue
            if kind == "deterministic_greedy_trace_neutral":
                result[prompt]["greedy"] = float(metrics["any_correct_at_k"])
            else:
                result[prompt]["mean8"] = float(metrics["mean_at_k"])
                result[prompt]["pass8"] = float(metrics["any_correct_at_k"])
                result[prompt]["distinct8"] = float(
                    metrics["distinct_correct_modes_at_k"]
                )
    return dict(result), violations


def classify_internal_result(
    aggregate_deltas: dict[str, dict[int, dict[str, float]]],
    seed_deltas: dict[str, dict[int, dict[int, dict[str, float]]]],
    route_terminal: dict[str, dict[int, dict[str, float]]],
) -> dict[str, Any]:
    """Apply only the prospectively frozen Gate 3 classifications."""

    task_quality: dict[str, bool] = {}
    positive_support: dict[str, bool] = {}
    mechanism_by_domain: dict[str, bool] = {}
    for domain in POOL_SIZE:
        terminal = aggregate_deltas[domain][6]
        task_quality[domain] = (
            terminal["greedy"] >= -0.02 - 1e-12
            and terminal["pass8"] >= -0.02 - 1e-12
            and (
                domain != "math_dev"
                or terminal["mean8"] >= -0.01 - 1e-12
            )
        )
        if domain in MODEBENCH:
            positive_support[domain] = (
                terminal["distinct8"] > 0
                and sum(
                    seed_deltas[domain][seed][6]["distinct8"] > 0
                    for seed in SEEDS
                )
                >= 2
                and aggregate_deltas[domain][5]["distinct8"] > 0
            )
            mechanism_by_domain[domain] = (
                sum(
                    route_terminal.get(domain, {})
                    .get(seed, {})
                    .get(
                        "post_replay_cross_prompt_neutral_reproductions",
                        0.0,
                    )
                    > 0
                    for seed in SEEDS
                )
                >= 2
            )

    checks = {
        "all_five_task_quality_noninferior": all(task_quality.values()),
        "three_executable_domains_positive_support": (
            sum(positive_support.values()) >= 3
        ),
        "mathir_positive_pass8_and_distinct8": (
            aggregate_deltas["mathir"][6]["pass8"] > 0
            and aggregate_deltas["mathir"][6]["distinct8"] > 0
        ),
        "mechanism_present_in_three_executable_domains": (
            sum(mechanism_by_domain.values()) >= 3
        ),
    }
    return {
        "status": "success" if all(checks.values()) else "mechanism_or_null",
        "checks": checks,
        "task_quality_noninferior": task_quality,
        "positive_support": positive_support,
        "mechanism_by_domain": mechanism_by_domain,
    }


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        metric: sum(row[metric] for row in rows) / len(rows)
        for metric in METRICS
    }


def _write_csv(payload: dict[str, Any]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "domain",
                "arm",
                "seed",
                "pass",
                "greedy",
                "mean8",
                "pass8",
                "distinct8",
            ]
        )
        for domain, arms in payload["curves"].items():
            for arm, seeds in arms.items():
                for seed, passes in seeds.items():
                    for pass_index, metrics in passes.items():
                        writer.writerow(
                            [
                                domain,
                                arm,
                                seed,
                                pass_index,
                                *(metrics[metric] for metric in METRICS),
                            ]
                        )


def _write_markdown(payload: dict[str, Any]) -> None:
    lines = [
        "# E69 Gate 3 confirmatory panel",
        "",
        f"Status: **{payload['status']}**.",
        "",
        "MATH-500 remains sealed. Seed 43 is reused exactly from Gate 2; "
        "no duplicate attempt is included.",
        "",
        "| Domain | Pass | Metric | Successor − control mean | "
        "Seed 43 / 44 / 45 |",
        "|---|---:|---|---:|---|",
    ]
    for domain, passes in payload.get("aggregate_deltas", {}).items():
        for pass_index in ("5", "6"):
            if pass_index not in passes:
                continue
            for metric in METRICS:
                seed_values = [
                    payload["seed_deltas"][domain][str(seed)][pass_index][metric]
                    for seed in SEEDS
                ]
                lines.append(
                    f"| {domain} | {pass_index} | {metric} | "
                    f"{passes[pass_index][metric]:+.4f} | "
                    + " / ".join(f"{value:+.4f}" for value in seed_values)
                    + " |"
                )
    if payload.get("bootstrap_terminal"):
        lines.extend(
            [
                "",
                "## Terminal crossed-bootstrap intervals",
                "",
                "| Domain | Metric | Mean delta | Descriptive 95% interval |",
                "|---|---|---:|---:|",
            ]
        )
        for domain, metrics in payload["bootstrap_terminal"].items():
            for metric, row in metrics.items():
                lines.append(
                    f"| {domain} | {metric} | {row['mean']:+.4f} | "
                    f"[{row['lower']:+.4f}, {row['upper']:+.4f}] |"
                )
    if payload.get("classification"):
        lines.extend(["", "## Frozen classification", ""])
        for name, passed in payload["classification"]["checks"].items():
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
        raise SystemExit(f"E69 Gate 3 identity is absent: {IDENTITY}")
    identity = json.loads(IDENTITY.read_text(encoding="utf-8"))
    if identity.get("schema") != "e69_gate3_confirmatory_v1":
        raise SystemExit("unexpected E69 Gate 3 identity schema")

    violations: list[str] = []
    pending: list[str] = []
    physical_runs: list[dict[str, Any]] = []
    curves: dict[
        str,
        dict[str, dict[int, dict[int, dict[str, float]]]],
    ] = defaultdict(lambda: defaultdict(dict))
    prompt_metrics: dict[
        str,
        dict[str, dict[int, dict[str, dict[str, float]]]],
    ] = defaultdict(lambda: defaultdict(dict))
    route_terminal: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)

    for domain, jobs in identity["jobs"].items():
        expected_step = POOL_SIZE[domain] * 6
        response_limit = 1024 if domain == "math_dev" else (
            64 if domain == "mathir" else 192
        )
        for job in jobs:
            arm = str(job["arm"])
            seed = int(job["seed"])
            job_id = int(job["job_id"])
            label = f"{domain}/{arm}/seed{seed}/job{job_id}"
            run_dir = _run_dir(str(job["run_stamp"]), job_id)
            if run_dir is None:
                pending.append(f"{label}: run directory absent")
                violations.extend(
                    f"{label}: {failure}" for failure in _scan_logs(job_id)
                )
                physical_runs.append(
                    {
                        "domain": domain,
                        "arm": arm,
                        "seed": seed,
                        "job_id": job_id,
                        "origin": job["origin"],
                        "run_dir": None,
                        "terminal": False,
                    }
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
            curves[domain][arm][seed] = pass_curve
            terminal_prompts, prompt_violations = _prompt_metrics(
                run_dir / "eval_mode_coverage_draws.jsonl",
                step=expected_step,
            )
            violations.extend(f"{label}: {value}" for value in prompt_violations)
            if terminal_prompts:
                prompt_metrics[domain][arm][seed] = terminal_prompts
            terminal = (
                int(training["latest_step"]) >= expected_step
                and len(pass_curve) == len(PASSES)
                and bool(terminal_prompts)
            )
            if not terminal:
                pending.append(
                    f"{label}: latest step {training['latest_step']}/"
                    f"{expected_step}"
                )
            if arm == _successor(domain):
                route_terminal[domain][seed] = dict(training["route_terminal"])
            physical_runs.append(
                {
                    "domain": domain,
                    "arm": arm,
                    "seed": seed,
                    "job_id": job_id,
                    "origin": job["origin"],
                    "run_dir": str(run_dir.resolve()),
                    "terminal": terminal,
                    "training": training,
                    "evaluations": {
                        str(key): value for key, value in pass_curve.items()
                    },
                }
            )

    complete = (
        len(physical_runs) == 30
        and all(run.get("terminal") for run in physical_runs)
        and not pending
    )
    aggregate_curves: dict[
        str,
        dict[str, dict[int, dict[str, float]]],
    ] = defaultdict(lambda: defaultdict(dict))
    seed_deltas: dict[
        str,
        dict[int, dict[int, dict[str, float]]],
    ] = defaultdict(dict)
    aggregate_deltas: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    bootstrap_terminal: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    classification = None

    if complete and not violations:
        for domain in POOL_SIZE:
            treatment = _successor(domain)
            for arm in (CONTROL, treatment):
                for pass_index in PASSES:
                    aggregate_curves[domain][arm][pass_index] = _mean_rows(
                        [curves[domain][arm][seed][pass_index] for seed in SEEDS]
                    )
            for seed in SEEDS:
                seed_deltas[domain][seed] = {}
                for pass_index in PASSES:
                    seed_deltas[domain][seed][pass_index] = {
                        metric: (
                            curves[domain][treatment][seed][pass_index][metric]
                            - curves[domain][CONTROL][seed][pass_index][metric]
                        )
                        for metric in METRICS
                    }
            for pass_index in PASSES:
                aggregate_deltas[domain][pass_index] = _mean_rows(
                    [
                        seed_deltas[domain][seed][pass_index]
                        for seed in SEEDS
                    ]
                )

            for metric in METRICS:
                paired: dict[int, list[float]] = {}
                for seed in SEEDS:
                    control_prompts = prompt_metrics[domain][CONTROL][seed]
                    treatment_prompts = prompt_metrics[domain][treatment][seed]
                    if set(control_prompts) != set(treatment_prompts):
                        violations.append(
                            f"{domain}/seed{seed}: terminal prompt sets differ"
                        )
                        continue
                    paired[seed] = [
                        treatment_prompts[prompt][metric]
                        - control_prompts[prompt][metric]
                        for prompt in sorted(control_prompts)
                    ]
                if len(paired) == len(SEEDS):
                    lower, upper = crossed_bootstrap_interval(paired)
                    bootstrap_terminal[domain][metric] = {
                        "mean": aggregate_deltas[domain][6][metric],
                        "lower": lower,
                        "upper": upper,
                    }
        if not violations:
            classification = classify_internal_result(
                aggregate_deltas,
                seed_deltas,
                route_terminal,
            )

    status = (
        "fail_integrity"
        if violations
        else "complete"
        if complete
        else "in_progress"
    )
    payload = {
        "schema": "e69_gate3_confirmatory_audit_v1",
        "status": status,
        "identity": str(IDENTITY.resolve()),
        "math500_sealed": True,
        "summary": {
            "expected_physical_runs": 30,
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
                arm: {
                    str(seed): {
                        str(pass_index): metrics
                        for pass_index, metrics in passes.items()
                    }
                    for seed, passes in seeds.items()
                }
                for arm, seeds in arms.items()
            }
            for domain, arms in curves.items()
        },
        "aggregate_curves": {
            domain: {
                arm: {
                    str(pass_index): metrics
                    for pass_index, metrics in passes.items()
                }
                for arm, passes in arms.items()
            }
            for domain, arms in aggregate_curves.items()
        },
        "seed_deltas": {
            domain: {
                str(seed): {
                    str(pass_index): metrics
                    for pass_index, metrics in passes.items()
                }
                for seed, passes in seeds.items()
            }
            for domain, seeds in seed_deltas.items()
        },
        "aggregate_deltas": {
            domain: {
                str(pass_index): metrics
                for pass_index, metrics in passes.items()
            }
            for domain, passes in aggregate_deltas.items()
        },
        "bootstrap_terminal": bootstrap_terminal,
        "route_terminal": route_terminal,
        "classification": classification,
        "pending": sorted(set(pending)),
        "violations": sorted(set(violations)),
    }
    _atomic_json(OUT_JSON, payload)
    _write_csv(payload)
    _write_markdown(payload)
    print(
        f"[e69-gate3-audit] status={status} "
        f"terminal={payload['summary']['terminal_physical_runs']}/30 "
        f"pending={payload['summary']['pending_items']} "
        f"violations={payload['summary']['integrity_violations']}"
    )


if __name__ == "__main__":
    main()
