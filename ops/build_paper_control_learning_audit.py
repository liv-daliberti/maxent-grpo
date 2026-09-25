#!/usr/bin/env python3
"""Did the compute-matched controls train, or did the learning rate stall them?

The objection this answers is specific: the controls run at a small constant
learning rate with no sweep, several terminate near zero, and the untrained
model beats them in two domains, so their accuracy deficit might be a failure
to optimise rather than the concentration this paper reports.

Three quantities decide it, and all three are read from runs already on disk:

  * ``pass@8`` at step 0 against the terminal step, per control cell. A control
    that never trained sits where it started. One that moves by four tenths did
    not fail to optimise, whichever direction it moved in.
  * the fraction of updates whose rollout group carries a task gradient at all.
    Dr.GRPO's group advantage is identically zero when every response in the
    group earns the same reward, so in those updates the learning rate
    multiplies an exact zero. Where that fraction is near zero the coefficient
    is not the binding constraint and no sweep over it can be.
  * the same endpoints for the replay arm, so the comparison the paper actually
    makes stays visible beside the audit.

One prompt group is drawn per update (Table "run contract"), so ``actor/rewards``
is that group's mean binary reward and the group is degenerate exactly when it
is 0 or 1. Computed this way the control fractions reproduce the five
Qwen2.5-0.5B values already reported in App. "Fresh-gradient degeneracy" to
three decimals, which is what licenses extending them to the other two scales.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
try:
    from paper_domain_typography import format_domain_names
except ModuleNotFoundError:
    from ops.paper_domain_typography import format_domain_names

ROOT = Path(__file__).resolve().parents[1]

LEDGERS = (
    ("Qwen2.5-0.5B", "var/artifacts/e78_verified_replay_only_05b_jobs.json"),
    ("Falcon3-1B", "var/artifacts/e79_falcon1b_aligned_verified_replay_jobs.json"),
    ("Qwen2.5-3B", "var/artifacts/e80r1_qwen3b_aligned_verified_replay_jobs.json"),
)
PASS8 = "eval/multi_answer/sampled_any_correct_at_8"
# Per-sample accuracy over the same draws. A policy that still samples
# several ways scores below its own pass@8; one that emits a single
# completion scores level with it, so the gap is a collapse read-out.
MEAN8 = "eval/multi_answer/sampled_mean_at_8"
REWARD = "actor/rewards"
LABEL = {"graph_coloring": "Graph", "countdown": "Countdown",
         "python_factors": "Python", "mathir": "MathIR", "pantry_plan": "PantryPlan"}
ORDER = ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan")
OUT_JSON = ROOT / "paper/results/control_learning_audit.json"
TABLE = ROOT / "paper/results/control_learning_table_body.tex"
MACROS = ROOT / "paper/results/control_learning_macros.tex"


def read_run(run_dir: str) -> dict | None:
    """Endpoints and gradient availability from one run's metrics, in one pass."""
    directory = Path(run_dir)
    if not directory.is_dir():
        return None
    jobs = [item for item in sorted(directory.iterdir()) if item.is_dir()]
    if not jobs:
        return None
    metrics = jobs[-1] / "train_metrics.jsonl"
    if not metrics.is_file():
        return None
    evals: dict[float, float] = {}
    means: dict[float, float] = {}
    groups = mixed = 0
    with metrics.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if PASS8 not in line and REWARD not in line and MEAN8 not in line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            reward = row.get(REWARD)
            if isinstance(reward, (int, float)):
                groups += 1
                if 0.0 < float(reward) < 1.0:
                    mixed += 1
            value = row.get(PASS8)
            step = row.get("trainer/global_step", row.get("trainer/step"))
            if isinstance(value, (int, float)) and step is not None:
                evals[float(step)] = float(value)
            average = row.get(MEAN8)
            if isinstance(average, (int, float)) and step is not None:
                means[float(step)] = float(average)
    if 0.0 not in evals:
        return None
    return {"pass0": evals[0.0], "terminal": evals[max(evals)],
            "mean0": means.get(0.0),
            "mean_terminal": means.get(max(means)) if means else None,
            "mixed": (mixed / groups) if groups else None}


def _avg(records, key):
    values = [r[key] for r in records if r.get(key) is not None]
    return statistics.fmean(values) if values else None


def main() -> int:
    cells: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list))
    for model, ledger in LEDGERS:
        payload = json.loads((ROOT / ledger).read_text(encoding="utf-8"))
        for run in payload["runs"]:
            arm = run.get("arm")
            if arm not in {"control", "replay"}:
                continue
            record = read_run(run["run_dir"])
            if record:
                cells[(model, run["domain"])][arm].append(record)

    rows = {}
    for (model, domain), arms in cells.items():
        control = arms.get("control", [])
        replay = arms.get("replay", [])
        if not control:
            continue
        base = statistics.fmean(c["pass0"] for c in control)
        end = statistics.fmean(c["terminal"] for c in control)
        fractions = [c["mixed"] for c in control if c["mixed"] is not None]
        rows[f"{model}|{domain}"] = {
            "model": model, "domain": domain, "seeds": len(control),
            "base_pass8": base, "control_pass8": end, "delta": end - base,
            "mixed_groups": statistics.fmean(fractions) if fractions else None,
            "base_mean8": _avg(control, "mean0"),
            "control_mean8": _avg(control, "mean_terminal"),
            "replay_pass8": (statistics.fmean(r["terminal"] for r in replay)
                             if replay else None),
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        {"generator": "ops/build_paper_control_learning_audit.py",
         "gradient_rule": "one prompt group per update; degenerate iff mean reward is 0 or 1",
         "cells": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def num(value, places=3):
        return f"{value:.{places}f}".lstrip("0") if value is not None else "---"

    lines = []
    for model, _ in LEDGERS:
        named = False
        for domain in ORDER:
            row = rows.get(f"{model}|{domain}")
            if not row:
                continue
            first, named = named, True
            delta = row["delta"]
            sign = "$+$" if delta > 0 else ("$-$" if delta < 0 else "")
            lines.append(
                f"    {'' if first else model} & {LABEL[domain]} & "
                f"{num(row['base_pass8'])} & {num(row['control_pass8'])} & "
                f"{sign}{num(abs(delta))} & "
                f"{num(row['base_mean8'])} & {num(row['control_mean8'])} & "
                f"{num(row['mixed_groups'])} & "
                f"{num(row['replay_pass8'])} \\\\")
        lines.append(r"    \addlinespace[2pt]")
    TABLE.write_text(
        format_domain_names("% Generated by ops/build_paper_control_learning_audit.py; do not hand edit.\n"
        + "\n".join(lines[:-1]) + "\n    \\bottomrule\n"), encoding="utf-8")

    values = list(rows.values())
    gains = [r for r in values if r["delta"] > 0.005]
    losses = [r for r in values if r["delta"] < -0.005]
    best = max(values, key=lambda r: r["delta"])
    worst = min(values, key=lambda r: r["delta"])
    leanest = min((r for r in values if r["mixed_groups"] is not None),
                  key=lambda r: r["mixed_groups"])
    zero_base = [r for r in values if r["base_pass8"] < 0.005]
    # The Yue pattern: per-sample accuracy up while the k-sample union is flat
    # or down. And a terminal policy whose mean@8 has closed on its own pass@8
    # is emitting one completion, which is the collapse read directly.
    paired = [r for r in values if r["base_mean8"] is not None
              and r["control_mean8"] is not None]
    yue = [r for r in paired
           if r["control_mean8"] > r["base_mean8"] and r["delta"] <= 0.005]
    mean_up = [r for r in paired if r["control_mean8"] > r["base_mean8"]]
    closed = [r for r in paired
              if r["control_pass8"] - r["control_mean8"] < 0.02]
    opened = [r for r in paired if r["base_pass8"] - r["base_mean8"] >= 0.02]
    # The cell that states the pattern most plainly: per-sample accuracy up by
    # the most among the cells whose k-sample union fell.
    fell = [r for r in paired if r["delta"] < -0.005]
    exemplar = max(fell, key=lambda r: r["control_mean8"] - r["base_mean8"]) \
        if fell else None
    MACROS.write_text(
        format_domain_names("% Generated by ops/build_paper_control_learning_audit.py; do not hand edit.\n"
        f"\\newcommand{{\\MDctlCells}}{{{len(values)}}}\n"
        f"\\newcommand{{\\MDctlGains}}{{{len(gains)}}}\n"
        f"\\newcommand{{\\MDctlLosses}}{{{len(losses)}}}\n"
        f"\\newcommand{{\\MDctlBestGain}}{{{best['delta']:.3f}}}\n"
        f"\\newcommand{{\\MDctlBestGainCell}}"
        f"{{{LABEL[best['domain']]} at {best['model']}}}\n"
        f"\\newcommand{{\\MDctlWorstLoss}}{{{abs(worst['delta']):.3f}}}\n"
        f"\\newcommand{{\\MDctlWorstLossCell}}"
        f"{{{LABEL[worst['domain']]} at {worst['model']}}}\n"
        f"\\newcommand{{\\MDctlMeanDelta}}"
        f"{{{statistics.fmean(r['delta'] for r in values):+.3f}}}\n"
        f"\\newcommand{{\\MDctlZeroBaseCells}}{{{len(zero_base)}}}\n"
        f"\\newcommand{{\\MDctlZeroBaseDomain}}"
        f"{{{LABEL[zero_base[0]['domain']] if zero_base else '---'}}}\n"
        f"\\newcommand{{\\MDctlLeanestMixed}}{{{leanest['mixed_groups']:.3f}}}\n"
        f"\\newcommand{{\\MDctlLeanestCell}}"
        f"{{{LABEL[leanest['domain']]} at {leanest['model']}}}\n"
        f"\\newcommand{{\\MDctlLeanestZero}}"
        f"{{{100 * (1 - leanest['mixed_groups']):.1f}}}\n"
        f"\\newcommand{{\\MDctlPaired}}{{{len(paired)}}}\n"
        f"\\newcommand{{\\MDctlMeanUp}}{{{len(mean_up)}}}\n"
        f"\\newcommand{{\\MDctlYue}}{{{len(yue)}}}\n"
        f"\\newcommand{{\\MDctlClosed}}{{{len(closed)}}}\n"
        f"\\newcommand{{\\MDctlOpened}}{{{len(opened)}}}\n"
        + ("" if exemplar is None else
           f"\\newcommand{{\\MDctlPatternCell}}"
           f"{{{LABEL[exemplar['domain']]} at {exemplar['model']}}}\n"
           f"\\newcommand{{\\MDctlPatternMeanFrom}}{{{exemplar['base_mean8']:.3f}}}\n"
           f"\\newcommand{{\\MDctlPatternMeanTo}}{{{exemplar['control_mean8']:.3f}}}\n"
           f"\\newcommand{{\\MDctlPatternPassFrom}}{{{exemplar['base_pass8']:.3f}}}\n"
           f"\\newcommand{{\\MDctlPatternPassTo}}{{{exemplar['control_pass8']:.3f}}}\n")),
        encoding="utf-8")

    print(f"wrote {OUT_JSON.name}, {TABLE.name}, {MACROS.name}")
    print(f"{'model':<14}{'domain':<12}{'base':>7}{'ctrl':>7}{'delta':>8}"
          f"{'mixed':>8}{'replay':>8}")
    for model, _ in LEDGERS:
        for domain in ORDER:
            row = rows.get(f"{model}|{domain}")
            if not row:
                continue
            print(f"{model:<14}{LABEL[domain]:<12}{row['base_pass8']:>7.3f}"
                  f"{row['control_pass8']:>7.3f}{row['delta']:>+8.3f}"
                  f"{(row['mixed_groups'] or 0):>8.3f}"
                  f"{(row['replay_pass8'] or 0):>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
