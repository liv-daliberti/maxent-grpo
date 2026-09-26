#!/usr/bin/env python3
"""Terminal endpoints for the E133 wider-group control, against E128 and Re:Dr.

The compute-budget appendix names three ways a control could spend replay's
share of the budget on learning: more task-gradient passes over existing
groups, larger fresh-rollout groups, and more fresh-sampling updates. E131
measured the third (``ops/build_paper_extended_horizon_control.py``). E133 is
the second: the E128 control objective with only the group width moved, 24
fresh rollouts per prompt instead of 16, at the unchanged 3,072-update horizon,
on the same prompts, seeds, placement, runtime and evaluator. It is the
reallocation that tests whether replay's breadth is a sampling artifact.

Three arms are emitted together because the appendix reads them together:

``e128_control``
    The 16-rollout control, measured by
    ``ops/extract_diversity_comparator_mode_diversity.py`` and reused here
    rather than re-measured, so the two arms differ only in group width.
``e133_control``
    The 24-rollout control, measured here from each cell's terminal
    mode-coverage draws with the same pooled estimator
    (``ops/mode_diversity.py``) every other cell uses.
``redr``
    Re:Dr at eight passes, from the reference-KL cohort, which is the one
    payload carrying both a control and a replay arm on a single evaluation.

Two things the E131 builder does not need are measured here because the
registered predictions name them. The paired PCMD difference against the
E128 control is scored seed by seed with a Student-t interval, since the
breadth prediction is a statement about that difference. And the fraction of
optimizer updates whose fresh group mixed correct and incorrect responses is
read from every cell's training log, for both arms, since the accuracy
prediction is attributed to that fraction rising. The realized group width is
read from the same logs and checked, so the arm is a 24-rollout arm by
measurement rather than by configuration.

The registered predictions live in
``paper/preregistration/e133_wider_groups_control_05b_20260924.md`` and are
scored in the emitted payload, including where the measurement runs against
them. Nothing here filters on reportability; ``reportable`` is emitted per
cell so the appendix can say which columns carry a \\pmd{} estimate at all.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import importlib.util
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))

LEDGER = ROOT / "var/artifacts/e133_wider_groups_control_05b_jobs.json"
E128_LEDGER = ROOT / "var/artifacts/e128_matched_control_05b_jobs.json"
E128 = ROOT / "paper/results/mode_diversity_comparators_diversity_05b.json"
REFKL = ROOT / "paper/results/reference_kl_comparison.json"
PREREG = ROOT / "paper/preregistration/e133_wider_groups_control_05b_20260924.md"

DOMAINS = ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan")
SEEDS = (43, 44, 45, 46, 47)
LABEL = {
    "graph_coloring": r"\texttt{Graph}",
    "countdown": r"\texttt{Countdown}",
    "python_factors": r"\texttt{Python}",
    "mathir": r"\texttt{MathIR}",
    "pantry_plan": r"\texttt{PantryPlan}",
}
SCHEMA = "paper-wider-groups-control-v1"
#: Two-sided 95% Student-t critical values by number of paired seeds.
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
#: Registered threshold for prediction 1: half the matched Re:Dr effect.
PMD_RECOVERY_CEILING = 0.19


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_extractor():
    """Import the frozen comparator extractor for its per-cell estimator."""

    target = ROOT / "ops/extract_diversity_comparator_mode_diversity.py"
    spec = importlib.util.spec_from_file_location("_dc_extract", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, sha256(target)


def fmt(value: float, places: int = 3) -> str:
    """Leading-dot formatting for [0,1); plain otherwise, as elsewhere."""

    text = f"{value:.{places}f}"
    if text.startswith("0."):
        return text[1:]
    if text.startswith("-0."):
        return "-" + text[2:]
    return text


def signed(value: float, places: int = 3) -> str:
    body = fmt(abs(value), places)
    return ("+" if value >= 0 else "-") + body


def neff(pmd: float) -> float:
    return 1.0 / (1.0 - pmd)


def paired(deltas: list[float]) -> dict:
    n = len(deltas)
    mean = st.fmean(deltas)
    half = T95[n] * st.stdev(deltas) / math.sqrt(n) if n > 1 else float("nan")
    return {"mean": mean, "ci95": [mean - half, mean + half], "n": n,
            "interval_type": "paired Student-t 95%"}


def terminal_attempt(run_dir: Path) -> Path:
    marker = run_dir / "TRAINING_COMPLETE.json"
    if marker.is_file():
        return Path(json.loads(marker.read_text(encoding="utf-8"))["terminal_attempt"])
    attempts = sorted(run_dir.glob("debug_job*"))
    if not attempts:
        raise SystemExit(f"no training attempt under {run_dir}")
    return attempts[-1]


def training_groups(run_dir: Path) -> dict:
    """Mixed-group fraction and realized group width from the training log.

    One fresh group is consumed per optimizer update, and the learner logs
    whether its rewards were all one or all zero. A group is mixed when it is
    neither, which is exactly when Dr.GRPO's centered advantage is nonzero
    (App. "Vanishing task gradients"). ``actor/num_data`` is the number of
    rollouts the actor returned for that update, which is the group width.
    """

    metrics = terminal_attempt(run_dir) / "train_metrics.jsonl"
    groups = mixed = 0
    widths: collections.Counter = collections.Counter()
    with metrics.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "train/all_one_rewards_count" not in line:
                continue
            row = json.loads(line)
            groups += 1
            if row["train/all_one_rewards_count"] == 0 and row["train/all_zero_rewards_count"] == 0:
                mixed += 1
            widths[int(row.get("actor/num_data", 0))] += 1
    if not groups:
        raise SystemExit(f"no training groups logged under {metrics}")
    return {"groups": groups, "mixed_fraction": mixed / groups,
            "group_widths": {str(k): v for k, v in sorted(widths.items())}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "paper/results",
        help="artifact directory for the payload, macros and table body",
    )
    parser.add_argument("--stamp", default=dt.date.today().isoformat().replace("-", ""))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    extract, extract_sha = load_extractor()

    # --- E133: measure every cell from its own terminal draws and log -------
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    width = int(ledger["fresh_group_size"])
    base_width = int(ledger["baseline_fresh_group_size"])
    wide_cells: dict[tuple[str, int], dict] = {}
    for run in ledger["runs"]:
        key = (run["domain"], int(run["seed"]))
        cell = extract.cell_pcmd(run["run_dir"])
        cell.update(training_groups(Path(run["run_dir"])))
        if set(cell["group_widths"]) != {str(width)}:
            raise SystemExit(f"E133 {key} did not train at width {width}: {cell['group_widths']}")
        wide_cells[key] = cell
    missing = [k for k in ((d, s) for d in DOMAINS for s in SEEDS) if k not in wide_cells]
    if missing:
        raise SystemExit(f"E133 ledger is short {len(missing)} cells: {missing}")
    steps = {c["terminal_step"] for c in wide_cells.values()}
    if len(steps) != 1:
        raise SystemExit(f"E133 cells end at differing steps: {sorted(steps)}")
    terminal_step = steps.pop()

    # --- E128: reuse the published 16-rollout control endpoints -------------
    base_cells: dict[tuple[str, int], dict] = {}
    for cell in json.loads(E128.read_text(encoding="utf-8"))["cells"]:
        if cell["method"] == "e128_control":
            base_cells[(cell["domain"], int(cell["seed"]))] = dict(cell)
    missing = [k for k in ((d, s) for d in DOMAINS for s in SEEDS) if k not in base_cells]
    if missing:
        raise SystemExit(f"E128 control is short {len(missing)} cells: {missing}")
    # Its mixed-group fraction is not in the published payload; read it from
    # the same logs, under the same definition, so the two arms are paired.
    e128_ledger = json.loads(E128_LEDGER.read_text(encoding="utf-8"))
    for run in e128_ledger["runs"]:
        if run.get("arm") != "control":
            continue
        key = (run["domain"], int(run["seed"]))
        if key not in base_cells:
            continue
        groups = training_groups(Path(run["run_dir"]))
        if set(groups["group_widths"]) != {str(base_width)}:
            raise SystemExit(f"E128 {key} did not train at width {base_width}: {groups['group_widths']}")
        base_cells[key].update(groups)
    short = [k for k, c in base_cells.items() if "mixed_fraction" not in c]
    if short:
        raise SystemExit(f"E128 control logs missing for {short}")

    # --- Re:Dr at eight passes ---------------------------------------------
    refkl = json.loads(REFKL.read_text(encoding="utf-8"))["domains"]

    rows = []
    for domain in DOMAINS:
        wide = [wide_cells[(domain, s)] for s in SEEDS]
        base = [base_cells[(domain, s)] for s in SEEDS]
        rows.append(
            {
                "domain": domain,
                "base_pass8": st.fmean(c["pass8"] for c in base),
                "base_pmd": st.fmean(c["pmd"] for c in base),
                "base_mixed": st.fmean(c["mixed_fraction"] for c in base),
                "wide_pass8": st.fmean(c["pass8"] for c in wide),
                "wide_pmd": st.fmean(c["pmd"] for c in wide),
                "wide_mixed": st.fmean(c["mixed_fraction"] for c in wide),
                "wide_reportable": sum(int(c["reportable"]) for c in wide),
                "wide_defined_prompts": st.fmean(c["defined_prompts"] for c in wide),
                "paired_pmd": paired([w["pmd"] - b["pmd"] for w, b in zip(wide, base)]),
                "paired_pass8": paired([w["pass8"] - b["pass8"] for w, b in zip(wide, base)]),
                "mixed_up_seeds": sum(int(w["mixed_fraction"] > b["mixed_fraction"])
                                      for w, b in zip(wide, base)),
                "redr_pass8": refkl[domain]["replay"]["pass8"],
                "redr_pmd": refkl[domain]["replay"]["pmd"],
                "seeds": len(wide),
            }
        )

    mean = {
        key: st.fmean(r[key] for r in rows)
        for key in (
            "base_pass8", "base_pmd", "base_mixed",
            "wide_pass8", "wide_pmd", "wide_mixed",
            "redr_pass8", "redr_pmd",
        )
    }
    recovered = (mean["wide_pmd"] - mean["base_pmd"]) / (mean["redr_pmd"] - mean["base_pmd"])
    pass_up = sum(1 for r in rows if r["wide_pass8"] > r["base_pass8"])
    pmd_up = sum(1 for r in rows if r["wide_pmd"] > r["base_pmd"])
    mixed_up = sum(1 for r in rows if r["wide_mixed"] > r["base_mixed"])
    largest_pmd_shift = max(abs(r["paired_pmd"]["mean"]) for r in rows)
    pmd_intervals_span_zero = sum(
        1 for r in rows if r["paired_pmd"]["ci95"][0] <= 0 <= r["paired_pmd"]["ci95"][1]
    )
    python = next(r for r in rows if r["domain"] == "python_factors")
    mixed_shift = {r["domain"]: r["wide_mixed"] - r["base_mixed"] for r in rows}
    largest_mixed_rise = max(mixed_shift, key=mixed_shift.get)

    predictions = {
        "diversity_does_not_recover": {
            "registered": "E133 five-domain terminal PCMD stays below .19, half the "
                          "matched Re:Dr effect over this control",
            "held": mean["wide_pmd"] < PMD_RECOVERY_CEILING,
            "five_domain_pmd": mean["wide_pmd"],
            "fraction_of_redr_effect_recovered": recovered,
            "domains_above_base": pmd_up,
            "largest_paired_shift": largest_pmd_shift,
            "paired_intervals_spanning_zero": pmd_intervals_span_zero,
        },
        "correctness_improves": {
            "registered": "E133 five-domain terminal pass@8 exceeds the matched control",
            "held": mean["wide_pass8"] > mean["base_pass8"],
            "shift": mean["wide_pass8"] - mean["base_pass8"],
            "domains_above_base": pass_up,
            "python_shift": python["wide_pass8"] - python["base_pass8"],
        },
        "mixed_groups_rise_everywhere": {
            "registered": "the fraction of updates whose fresh group carries no task "
                          "gradient falls relative to the control in every domain, "
                          "most in PythonFactors",
            "held": mixed_up == len(rows) and largest_mixed_rise == "python_factors",
            "domains_with_more_mixed_groups": mixed_up,
            "largest_rise_in": largest_mixed_rise,
            "python_mixed_base": python["base_mixed"],
            "python_mixed_wide": python["wide_mixed"],
            "shift_by_domain": mixed_shift,
        },
    }

    payload = {
        "schema": SCHEMA,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "estimator": "pooled per-prompt PCMD over verified responses, "
                     "ops/mode_diversity.py, via extract_diversity_comparator_"
                     "mode_diversity.cell_pcmd; mixed-group fraction from "
                     "train/all_one_rewards_count and train/all_zero_rewards_count "
                     "per logged optimizer update",
        "scientific_difference": ledger.get("scientific_difference"),
        "fresh_group_size": width,
        "baseline_fresh_group_size": base_width,
        "terminal_step": terminal_step,
        "cells": len(wide_cells),
        "sources": {
            "e133_ledger": {"path": str(LEDGER.relative_to(ROOT)), "sha256": sha256(LEDGER)},
            "e128_ledger": {"path": str(E128_LEDGER.relative_to(ROOT)), "sha256": sha256(E128_LEDGER)},
            "e128_control": {"path": str(E128.relative_to(ROOT)), "sha256": sha256(E128)},
            "redr": {"path": str(REFKL.relative_to(ROOT)), "sha256": sha256(REFKL)},
            "preregistration": {"path": str(PREREG.relative_to(ROOT)), "sha256": sha256(PREREG)},
            "extractor": {"path": "ops/extract_diversity_comparator_mode_diversity.py",
                          "sha256": extract_sha},
        },
        "domains": rows,
        "means": mean,
        "registered_predictions": predictions,
        "cells_measured": {
            f"{d}/s{s}": {k: v for k, v in wide_cells[(d, s)].items() if k != "draws"}
            for d in DOMAINS for s in SEEDS
        },
    }
    (out / f"wider_groups_control_{args.stamp}.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )

    macros = [
        "% Generated by ops/build_paper_wider_groups_control.py; do not hand edit.",
        rf"\newcommand{{\WGCwidth}}{{{width}}}",
        rf"\newcommand{{\WGCbasewidth}}{{{base_width}}}",
        rf"\newcommand{{\WGCupdates}}{{3{{,}}072}}",
        rf"\newcommand{{\WGCcells}}{{{len(wide_cells)}}}",
        rf"\newcommand{{\WGCbasepass}}{{{fmt(mean['base_pass8'])}}}",
        rf"\newcommand{{\WGCwidepass}}{{{fmt(mean['wide_pass8'])}}}",
        rf"\newcommand{{\WGCredrpass}}{{{fmt(mean['redr_pass8'])}}}",
        rf"\newcommand{{\WGCpassshift}}{{{signed(mean['wide_pass8'] - mean['base_pass8'])}}}",
        rf"\newcommand{{\WGCgap}}{{{fmt(mean['redr_pass8'] - mean['wide_pass8'])}}}",
        rf"\newcommand{{\WGCbasepmd}}{{{fmt(mean['base_pmd'])}}}",
        rf"\newcommand{{\WGCwidepmd}}{{{fmt(mean['wide_pmd'])}}}",
        rf"\newcommand{{\WGCredrpmd}}{{{fmt(mean['redr_pmd'])}}}",
        rf"\newcommand{{\WGCrecovered}}{{{round(100 * recovered)}\%}}",
        rf"\newcommand{{\WGCbaseneff}}{{{neff(mean['base_pmd']):.3f}}}",
        rf"\newcommand{{\WGCwideneff}}{{{neff(mean['wide_pmd']):.3f}}}",
        rf"\newcommand{{\WGCredrneff}}{{{neff(mean['redr_pmd']):.3f}}}",
        rf"\newcommand{{\WGCpassup}}{{{pass_up}}}",
        rf"\newcommand{{\WGCpmdup}}{{{pmd_up}}}",
        rf"\newcommand{{\WGClargestpmdshift}}{{{fmt(largest_pmd_shift)}}}",
        rf"\newcommand{{\WGCmixedup}}{{{mixed_up}}}",
        rf"\newcommand{{\WGCpythonbase}}{{{fmt(python['base_pass8'])}}}",
        rf"\newcommand{{\WGCpythonwide}}{{{fmt(python['wide_pass8'])}}}",
        rf"\newcommand{{\WGCpythonshift}}{{{signed(python['wide_pass8'] - python['base_pass8'])}}}",
        rf"\newcommand{{\WGCpythonmixedbase}}{{{fmt(python['base_mixed'])}}}",
        rf"\newcommand{{\WGCpythonmixedwide}}{{{fmt(python['wide_mixed'])}}}",
        rf"\newcommand{{\WGCpythonreportable}}{{{python['wide_reportable']}}}",
        rf"\newcommand{{\WGCpythondefined}}{{{python['wide_defined_prompts']:.1f}}}",
        rf"\newcommand{{\WGCmathirmixedbase}}{{{fmt(next(r for r in rows if r['domain']=='mathir')['base_mixed'])}}}",
        rf"\newcommand{{\WGCmathirmixedwide}}{{{fmt(next(r for r in rows if r['domain']=='mathir')['wide_mixed'])}}}",
    ]
    (out / f"wider_groups_control_{args.stamp}_macros.tex").write_text(
        "\n".join(macros) + "\n", encoding="utf-8"
    )

    body = [
        "% Generated by ops/build_paper_wider_groups_control.py; do not hand edit."
    ]
    for r in rows:
        body.append(
            f"  {LABEL[r['domain']]} & {fmt(r['base_pass8'])} & {fmt(r['base_pmd'])}"
            f" & {fmt(r['base_mixed'])}"
            f" & {fmt(r['wide_pass8'])} & {fmt(r['wide_pmd'])} & {fmt(r['wide_mixed'])}"
            f" & {fmt(r['redr_pass8'])} & {fmt(r['redr_pmd'])} \\\\"
        )
    body.append(r"  \midrule")
    body.append(
        f"  Mean & {fmt(mean['base_pass8'])} & {fmt(mean['base_pmd'])} & {fmt(mean['base_mixed'])}"
        f" & {fmt(mean['wide_pass8'])} & {fmt(mean['wide_pmd'])} & {fmt(mean['wide_mixed'])}"
        f" & {fmt(mean['redr_pass8'])} & {fmt(mean['redr_pmd'])} \\\\"
    )
    body.append(r"  \bottomrule")
    (out / f"wider_groups_control_{args.stamp}_table_body.tex").write_text(
        "\n".join(body) + "\n", encoding="utf-8"
    )

    print(f"wrote wider_groups_control_{args.stamp}.{{json,_macros.tex,_table_body.tex}}"
          f" -> {out.relative_to(ROOT)}")
    print(f"  cells={len(wide_cells)} terminal_step={terminal_step} width={width} vs {base_width}")
    print(f"  pass@8  base {fmt(mean['base_pass8'])} -> wide {fmt(mean['wide_pass8'])}"
          f"  (Re:Dr {fmt(mean['redr_pass8'])}, gap {fmt(mean['redr_pass8']-mean['wide_pass8'])})")
    print(f"  PCMD    base {fmt(mean['base_pmd'])} -> wide {fmt(mean['wide_pmd'])}"
          f"  (Re:Dr {fmt(mean['redr_pmd'])}, recovered {100*recovered:.1f}%)")
    print(f"  mixed   base {fmt(mean['base_mixed'])} -> wide {fmt(mean['wide_mixed'])}"
          f"  (up in {mixed_up}/5 domains; largest rise {largest_mixed_rise})")
    for r in rows:
        p = r["paired_pmd"]
        print(f"    {r['domain']:15s} dPCMD {signed(p['mean'])} [{signed(p['ci95'][0])},{signed(p['ci95'][1])}]"
              f"  dpass8 {signed(r['paired_pass8']['mean'])}"
              f"  mixed {fmt(r['base_mixed'])}->{fmt(r['wide_mixed'])}"
              f"  reportable {r['wide_reportable']}/5")
    for name, rec in predictions.items():
        print(f"  prediction {name}: {'held' if rec['held'] else 'runs against the registration'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
