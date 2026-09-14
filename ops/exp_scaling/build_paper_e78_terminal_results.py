#!/usr/bin/env python3
"""Build the paper-facing terminal and trajectory summary for completed E78.

The registered Qwen2.5-0.5B comparison contains five domains, two arms, five
paired training seeds, and four independent K=8 evaluation draws at every
half-pass checkpoint.  This script fails closed unless that complete design is
present.  It reports paired terminal effects and the preregistered trapezoidal
AUC, normalized by the eight-pass training horizon.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "var/artifacts/e78_verified_replay_only_05b_jobs.json"
OUTPUT = ROOT / "paper/results/e78_terminal_05b.json"
TABLE_BODY = ROOT / "paper/results/e78_terminal_05b_table_body.tex"
EXPECTED_ARMS = ("control", "replay")
EXPECTED_DRAWS = 4
EXPECTED_SEEDS = (43, 44, 45, 46, 47)
METRIC_FIELDS = {
    "pass8": "any_correct_at_k",
    "mean8": "mean_at_k",
    "distinct8": "distinct_correct_modes_at_k",
}
DOMAIN_TITLES = {
    "graph_coloring": "Graph coloring",
    "countdown": "Countdown",
    "python_factors": "Python factors",
    "mathir": "MathIR",
    "pantry_plan": "PantryPlan",
}


def _finite(value: Any, *, where: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RuntimeError(f"{where}: expected a finite number, got {value!r}")
    return float(value)


def _curve(
    run_dir: Path,
    *,
    interval: int,
    target: int,
) -> dict[int, dict[str, float]]:
    """Load complete four-draw sampled metrics at each registered checkpoint."""

    records: dict[tuple[int, int], dict[str, Any]] = {}
    for path in sorted(run_dir.glob("debug_job*/eval_mode_coverage_draws.jsonl")):
        with path.open(encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                step = row.get("step")
                draw = row.get("draw_index")
                metrics = row.get("metrics")
                if (
                    row.get("evaluation_kind")
                    != "fixed_seed_sampled_k_neutral"
                    or not isinstance(step, int)
                    or not isinstance(draw, int)
                    or not isinstance(metrics, dict)
                    or step < 0
                    or step > target
                    or step % interval
                    or draw not in range(EXPECTED_DRAWS)
                ):
                    continue
                records[(step, draw)] = metrics

    curve: dict[int, dict[str, float]] = {}
    for step in range(0, target + 1, interval):
        draws = [records.get((step, draw)) for draw in range(EXPECTED_DRAWS)]
        if any(draw is None for draw in draws):
            observed = [
                draw
                for draw in range(EXPECTED_DRAWS)
                if records.get((step, draw)) is not None
            ]
            raise RuntimeError(
                f"{run_dir}: checkpoint {step} has draws {observed}, expected "
                f"0--{EXPECTED_DRAWS - 1}"
            )
        complete = [draw for draw in draws if draw is not None]
        curve[step] = {
            output_name: sum(
                _finite(
                    draw.get(source_name),
                    where=f"{run_dir}:{step}:{source_name}",
                )
                for draw in complete
            )
            / EXPECTED_DRAWS
            for output_name, source_name in METRIC_FIELDS.items()
        }
    return curve


def _normalized_auc(
    curve: dict[int, dict[str, float]],
    *,
    metric: str,
    target: int,
) -> float:
    steps = sorted(curve)
    if not steps or steps[0] != 0 or steps[-1] != target:
        raise RuntimeError(f"AUC curve does not span 0--{target}: {steps}")
    area = 0.0
    for left, right in zip(steps, steps[1:]):
        area += (right - left) * (curve[left][metric] + curve[right][metric]) / 2
    return area / target


def _paired_summary(values: dict[int, float]) -> dict[str, Any]:
    """Return a mean and two-sided 95% Student-t interval over five seeds."""

    if tuple(sorted(values)) != EXPECTED_SEEDS:
        raise RuntimeError(f"expected paired seeds {EXPECTED_SEEDS}, got {sorted(values)}")
    samples = [values[seed] for seed in EXPECTED_SEEDS]
    mean = statistics.fmean(samples)
    # Two-sided 95% t critical value with df=4.  Keeping the constant local
    # avoids adding scipy as a paper-build dependency.
    t_critical_df4 = 2.7764451051977987
    half_width = t_critical_df4 * statistics.stdev(samples) / math.sqrt(len(samples))
    return {
        "mean": mean,
        "student_t_95": [mean - half_width, mean + half_width],
        "range": [min(samples), max(samples)],
        "per_seed": {str(seed): values[seed] for seed in EXPECTED_SEEDS},
    }


def _signed(value: float) -> str:
    return f"{value:+.3f}".replace("+0.", "+.").replace("-0.", "-.")


def main() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    domains = [str(domain) for domain in ledger["domains"]]
    if set(domains) != set(DOMAIN_TITLES):
        raise RuntimeError(f"unexpected E78 domains: {domains}")
    interval = int(ledger["checkpoint_interval_steps"])
    target = int(ledger["target_steps"])
    passes = int(ledger["passes"])
    if target != passes * int(ledger["train_rows"]):
        raise RuntimeError("target steps do not equal passes times training rows")

    curves: dict[str, dict[str, dict[int, dict[int, dict[str, float]]]]] = (
        defaultdict(lambda: defaultdict(dict))
    )
    for run in ledger["runs"]:
        domain = str(run["domain"])
        arm = str(run["arm"])
        seed = int(run["seed"])
        if arm not in EXPECTED_ARMS:
            raise RuntimeError(f"unexpected E78 arm: {arm}")
        curves[domain][arm][seed] = _curve(
            Path(run["run_dir"]), interval=interval, target=target
        )

    output: dict[str, Any] = {
        "schema": "e78_terminal_paper_results_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ledger": str(LEDGER.resolve()),
        "design": {
            "model": "Qwen2.5-0.5B-Instruct",
            "domains": domains,
            "arms": list(EXPECTED_ARMS),
            "paired_seeds": list(EXPECTED_SEEDS),
            "evaluation_draws": EXPECTED_DRAWS,
            "checkpoint_interval_steps": interval,
            "target_steps": target,
            "passes": passes,
        },
        "uncertainty": (
            "two-sided 95% Student-t interval over the five paired training-seed "
            "effects (df=4)"
        ),
        "auc": (
            "trapezoidal AUC over all registered half-pass checkpoints, divided "
            "by the eight-pass horizon"
        ),
        "domains": {},
    }

    for domain in domains:
        domain_curves = curves[domain]
        for arm in EXPECTED_ARMS:
            if tuple(sorted(domain_curves[arm])) != EXPECTED_SEEDS:
                raise RuntimeError(
                    f"{domain}/{arm}: expected seeds {EXPECTED_SEEDS}, "
                    f"got {sorted(domain_curves[arm])}"
                )

        arm_summaries: dict[str, Any] = {}
        for arm in EXPECTED_ARMS:
            arm_summaries[arm] = {
                "terminal_mean": {
                    metric: statistics.fmean(
                        domain_curves[arm][seed][target][metric]
                        for seed in EXPECTED_SEEDS
                    )
                    for metric in METRIC_FIELDS
                },
                "normalized_auc_mean": {
                    metric: statistics.fmean(
                        _normalized_auc(
                            domain_curves[arm][seed], metric=metric, target=target
                        )
                        for seed in EXPECTED_SEEDS
                    )
                    for metric in METRIC_FIELDS
                },
            }

        effects: dict[str, Any] = {}
        for metric in METRIC_FIELDS:
            terminal = {
                seed: domain_curves["replay"][seed][target][metric]
                - domain_curves["control"][seed][target][metric]
                for seed in EXPECTED_SEEDS
            }
            auc = {
                seed: _normalized_auc(
                    domain_curves["replay"][seed],
                    metric=metric,
                    target=target,
                )
                - _normalized_auc(
                    domain_curves["control"][seed],
                    metric=metric,
                    target=target,
                )
                for seed in EXPECTED_SEEDS
            }
            effects[f"terminal_{metric}"] = _paired_summary(terminal)
            effects[f"normalized_auc_{metric}"] = _paired_summary(auc)

        terminal_excess = {
            seed: (
                domain_curves["replay"][seed][target]["distinct8"]
                - domain_curves["replay"][seed][target]["pass8"]
            )
            - (
                domain_curves["control"][seed][target]["distinct8"]
                - domain_curves["control"][seed][target]["pass8"]
            )
            for seed in EXPECTED_SEEDS
        }
        effects["terminal_excess_modes8"] = _paired_summary(terminal_excess)
        output["domains"][domain] = {
            "title": DOMAIN_TITLES[domain],
            "arms": arm_summaries,
            "paired_effects": effects,
        }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    rows = []
    for domain in domains:
        entry = output["domains"][domain]
        effects = entry["paired_effects"]
        distinct = effects["terminal_distinct8"]
        interval_low, interval_high = distinct["student_t_95"]
        rows.append(
            "    "
            + " & ".join(
                [
                    entry["title"],
                    _signed(effects["terminal_pass8"]["mean"]),
                    _signed(distinct["mean"]),
                    f"[{_signed(interval_low)}, {_signed(interval_high)}]",
                    _signed(effects["terminal_excess_modes8"]["mean"]),
                    _signed(effects["normalized_auc_distinct8"]["mean"]),
                ]
            )
            + r" \\"
        )
    TABLE_BODY.write_text("\n".join(rows) + "\n    \\bottomrule\n", encoding="utf-8")
    print(f"wrote {OUTPUT} and {TABLE_BODY}")


if __name__ == "__main__":
    main()
