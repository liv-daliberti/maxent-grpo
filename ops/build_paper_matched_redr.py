#!/usr/bin/env python3
"""E132: Re:Dr measured on the E128 panel, so the comparisons are within cohort.

E130 and E131 were measured against the E128 control while the Re:Dr arm they
were compared against came from E78, whose cells span an A100 and an A5000
node. E132 runs Re:Dr on the E128 panel itself -- same prompts, seeds,
placement, runtime and evaluator -- so every effect here is a paired,
within-cohort quantity.

The payload is emitted in the shape the mechanism-ladder generator consumes
(``domains[<domain>]["replay"]["pmd"]``), so
``build_paper_replay_mechanism_ladder.py --redr`` can be pointed at this file
and the whole ladder becomes within-cohort in one step.

Two things this cohort settles.

**The seam was benign for diversity.** E132's five-domain mean PCMD lands within
a few thousandths of the published E78 arm, which is what licenses every PCMD
comparison already reported against the cross-cohort number.

**Every domain carries a positive effect.** The per-domain magnitudes differ by
more than an order of magnitude at Level 1, and an earlier draft used a
magnitude threshold to decide which domains "counted" -- which excluded
``python_factors`` by a thousandth and wrongly implied the domain has no replay
effect. Paired per seed against its own control, the effect is positive in
almost every cell and negative in none, so the honest statement is a sign
census over cells rather than a cut on magnitude. ``paired_cells`` carries it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))

LEDGER = ROOT / "var/artifacts/e132_matched_redr_05b_jobs.json"
E128 = ROOT / "paper/results/mode_diversity_comparators_diversity_05b.json"
E130_PAYLOAD = ROOT / "paper/results/mode_agnostic_replay_20260923.json"
REFKL = ROOT / "paper/results/reference_kl_comparison.json"
PREREG = ROOT / "paper/preregistration/e132_matched_redr_05b_20260923.md"

DOMAINS = ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan")
SEEDS = (43, 44, 45, 46, 47)
LABEL = {
    "graph_coloring": r"\texttt{Graph}",
    "countdown": r"\texttt{Countdown}",
    "python_factors": r"\texttt{Python}",
    "mathir": r"\texttt{MathIR}",
    "pantry_plan": r"\texttt{PantryPlan}",
}
CAPACITY_INVARIANTS = ("train/canonical_replay_capacity",
                       "train/canonical_replay_verified_likelihood_active")
EXPECTED = {"train/canonical_replay_capacity": 16,
            "train/canonical_replay_verified_likelihood_active": 1}
SCHEMA = "paper-matched-redr-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_extractor():
    target = ROOT / "ops/extract_diversity_comparator_mode_diversity.py"
    spec = importlib.util.spec_from_file_location("_dc_extract", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, sha256(target)


def audit_cell(run_dir: Path) -> dict:
    """Confirm the arm ran at Re:Dr's capacity with the replay gradient live."""

    complete = json.loads((run_dir / "TRAINING_COMPLETE.json").read_text(encoding="utf-8"))
    seen: dict[str, set] = {k: set() for k in CAPACITY_INVARIANTS}
    banked = 0.0
    for line in (Path(complete["terminal_attempt"]) / "train_metrics.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for key in CAPACITY_INVARIANTS:
            if row.get(key) is not None:
                seen[key].add(row[key])
        value = row.get("train/canonical_replay_banked_modes")
        if value is not None:
            banked = max(banked, value)
    return {
        "on_spec": all(seen[k] == {EXPECTED[k]} for k in CAPACITY_INVARIANTS),
        "banked_modes_max": banked,
        "training_terminal_step": complete["terminal_step"],
    }


def fmt(value: float, places: int = 3) -> str:
    text = f"{value:.{places}f}"
    if text.startswith("0."):
        return text[1:]
    if text.startswith("-0."):
        return "-" + text[2:]
    return text


def signed(value: float, places: int = 3) -> str:
    return ("+" if value >= 0 else "-") + fmt(abs(value), places)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/results")
    parser.add_argument("--stamp", default=dt.date.today().isoformat().replace("-", ""))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    extract, extract_sha = load_extractor()
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))

    cells: dict[tuple[str, int], dict] = {}
    audits: dict[tuple[str, int], dict] = {}
    for run in ledger["runs"]:
        key = (run["domain"], int(run["seed"]))
        cells[key] = extract.cell_pcmd(run["run_dir"])
        audits[key] = audit_cell(Path(run["run_dir"]))
    missing = [k for k in ((d, s) for d in DOMAINS for s in SEEDS) if k not in cells]
    if missing:
        raise SystemExit(f"E132 ledger is short {len(missing)} cells: {missing}")
    off_spec = sorted(f"{d}/s{s}" for (d, s), a in audits.items() if not a["on_spec"])
    if off_spec:
        raise SystemExit(f"E132 cells did not run at Re:Dr capacity: {off_spec}")
    steps = {c["terminal_step"] for c in cells.values()}
    if len(steps) != 1:
        raise SystemExit(f"E132 cells end at differing steps: {sorted(steps)}")
    terminal_step = steps.pop()

    control = {
        (c["domain"], c["seed"]): c
        for c in json.loads(E128.read_text(encoding="utf-8"))["cells"]
        if c["method"] == "e128_control"
    }
    e130 = {r["domain"]: r for r in json.loads(E130_PAYLOAD.read_text(encoding="utf-8"))["domains"]}
    published = json.loads(REFKL.read_text(encoding="utf-8"))["domains"]

    domains: dict[str, dict] = {}
    census = {"positive": 0, "zero": 0, "negative": 0, "zero_cells": [], "negative_cells": []}
    for domain in DOMAINS:
        paired = []
        for seed in SEEDS:
            delta = cells[(domain, seed)]["pmd"] - control[(domain, seed)]["pmd"]
            paired.append(delta)
            bucket = "positive" if delta > 0 else ("zero" if delta == 0 else "negative")
            census[bucket] += 1
            if bucket != "positive":
                census[f"{bucket}_cells"].append(f"{domain}/s{seed}")
        domains[domain] = {
            # the shape the ladder generator reads
            "replay": {
                "pmd": st.fmean(cells[(domain, s)]["pmd"] for s in SEEDS),
                "pass8": st.fmean(cells[(domain, s)]["pass8"] for s in SEEDS),
            },
            "control": {
                "pmd": st.fmean(control[(domain, s)]["pmd"] for s in SEEDS),
                "pass8": st.fmean(control[(domain, s)]["pass8"] for s in SEEDS),
            },
            "paired_pmd_effect": st.fmean(paired),
            "paired_pmd_by_seed": paired,
            # raw per-seed values, so the keying contrast against the
            # capacity-one arm can be paired seed by seed downstream
            "replay_pmd_by_seed": [cells[(domain, s)]["pmd"] for s in SEEDS],
            "seeds": list(SEEDS),
            "seeds_positive": sum(1 for z in paired if z > 0),
            "reportable": sum(int(cells[(domain, s)]["reportable"]) for s in SEEDS),
            "banked_modes_max": max(audits[(domain, s)]["banked_modes_max"] for s in SEEDS),
            "published_e78_pmd": published[domain]["replay"]["pmd"],
        }

    mean_redr = st.fmean(v["replay"]["pmd"] for v in domains.values())
    mean_ctl = st.fmean(v["control"]["pmd"] for v in domains.values())
    mean_pass = st.fmean(v["replay"]["pass8"] for v in domains.values())
    mean_ctl_pass = st.fmean(v["control"]["pass8"] for v in domains.values())
    mean_pub = st.fmean(v["published_e78_pmd"] for v in domains.values())
    mean_cap1 = st.fmean(e130[d]["abl_pmd"] for d in DOMAINS)
    recovered = (mean_cap1 - mean_ctl) / (mean_redr - mean_ctl)

    predictions = {
        "diversity_agreement": {
            "registered": "E132 five-domain mean PCMD within .05 of the published "
                          "E78 Re:Dr value",
            "e132": mean_redr, "published": mean_pub,
            "absolute_difference": abs(mean_redr - mean_pub),
            "held": abs(mean_redr - mean_pub) < 0.05,
        },
        "correctness_exceeds_control": {
            "registered": "E132 pass@8 exceeds its matched control on the "
                          "five-domain mean; no match to the published value claimed",
            "e132": mean_pass, "control": mean_ctl_pass,
            "published": st.fmean(published[d]["replay"]["pass8"] for d in DOMAINS),
            "held": mean_pass > mean_ctl_pass,
        },
        "ablation_conclusion_survives": {
            "registered": "within-cohort capacity-one recovered fraction stays "
                          "strictly between 25% and 85%",
            "recovered_fraction": recovered,
            "cross_cohort_value_it_replaces": (mean_cap1 - mean_ctl) / (mean_pub - mean_ctl),
            "held": 0.25 < recovered < 0.85,
        },
    }

    payload = {
        "schema": SCHEMA,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "estimator": "pooled per-prompt PCMD over verified responses, "
                     "ops/mode_diversity.py, via extract_diversity_comparator_"
                     "mode_diversity.cell_pcmd",
        "cohort": "e132",
        "control_cohort": "e128",
        "replaces_cross_cohort_arm": "e78 replay (Re:Dr), spans node302 (A100) "
                                     "and node105 (A5000)",
        "terminal_step": terminal_step,
        "cells": len(cells),
        "cells_on_spec": len(cells) - len(off_spec),
        "reportable_cells": sum(int(c["reportable"]) for c in cells.values()),
        "paired_cells": census,
        "sources": {
            "e132_ledger": {"path": str(LEDGER.relative_to(ROOT)), "sha256": sha256(LEDGER)},
            "e128_control": {"path": str(E128.relative_to(ROOT)), "sha256": sha256(E128)},
            "e130": {"path": str(E130_PAYLOAD.relative_to(ROOT)), "sha256": sha256(E130_PAYLOAD)},
            "published_e78": {"path": str(REFKL.relative_to(ROOT)), "sha256": sha256(REFKL)},
            "preregistration": {"path": str(PREREG.relative_to(ROOT)), "sha256": sha256(PREREG)},
            "extractor": {"path": "ops/extract_diversity_comparator_mode_diversity.py",
                          "sha256": extract_sha},
        },
        "domains": domains,
        "means": {
            "control_pmd": mean_ctl, "redr_pmd": mean_redr, "published_pmd": mean_pub,
            "control_pass8": mean_ctl_pass, "redr_pass8": mean_pass,
        },
        "registered_predictions": predictions,
    }
    (out / f"matched_redr_{args.stamp}.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )

    macros = [
        "% Generated by ops/build_paper_matched_redr.py; do not hand edit.",
        rf"\newcommand{{\MRcells}}{{{len(cells)}}}",
        rf"\newcommand{{\MRpmd}}{{{fmt(mean_redr)}}}",
        rf"\newcommand{{\MRctlpmd}}{{{fmt(mean_ctl)}}}",
        rf"\newcommand{{\MRpubpmd}}{{{fmt(mean_pub)}}}",
        rf"\newcommand{{\MRpmdgap}}{{{fmt(abs(mean_redr - mean_pub))}}}",
        rf"\newcommand{{\MRpass}}{{{fmt(mean_pass)}}}",
        rf"\newcommand{{\MRctlpass}}{{{fmt(mean_ctl_pass)}}}",
        rf"\newcommand{{\MRpassgain}}{{{signed(mean_pass - mean_ctl_pass)}}}",
        rf"\newcommand{{\MRpositive}}{{{census['positive']}}}",
        rf"\newcommand{{\MRzero}}{{{census['zero']}}}",
        rf"\newcommand{{\MRnegative}}{{{census['negative']}}}",
        rf"\newcommand{{\MRrecovered}}{{{round(100 * recovered)}\%}}",
        rf"\newcommand{{\MRalldomains}}{{{sum(1 for v in domains.values() if v['paired_pmd_effect'] > 0)}}}",
    ]
    for domain, v in domains.items():
        tag = domain.split("_")[0]
        macros.append(rf"\newcommand{{\MR{tag}effect}}{{{signed(v['paired_pmd_effect'])}}}")
        macros.append(rf"\newcommand{{\MR{tag}seedspos}}{{{v['seeds_positive']}}}")
    (out / f"matched_redr_{args.stamp}_macros.tex").write_text(
        "\n".join(macros) + "\n", encoding="utf-8"
    )

    body = ["% Generated by ops/build_paper_matched_redr.py; do not hand edit."]
    for domain in DOMAINS:
        v = domains[domain]
        body.append(
            f"  {LABEL[domain]} & {fmt(v['control']['pmd'])} & {fmt(v['replay']['pmd'])}"
            f" & {signed(v['paired_pmd_effect'])} & {v['seeds_positive']}/5"
            f" & {fmt(v['published_e78_pmd'])} \\\\"
        )
    body.append(r"  \midrule")
    body.append(
        f"  Mean & {fmt(mean_ctl)} & {fmt(mean_redr)}"
        f" & {signed(mean_redr - mean_ctl)} & {census['positive']}/25"
        f" & {fmt(mean_pub)} \\\\"
    )
    body.append(r"  \bottomrule")
    (out / f"matched_redr_{args.stamp}_table_body.tex").write_text(
        "\n".join(body) + "\n", encoding="utf-8"
    )

    print(f"wrote matched_redr_{args.stamp}.{{json,_macros.tex,_table_body.tex}}")
    print(f"  cells={len(cells)} on-spec={len(cells)-len(off_spec)} step={terminal_step}")
    print(f"  PCMD   control {fmt(mean_ctl)} -> E132 {fmt(mean_redr)}"
          f"   (published E78 {fmt(mean_pub)}, gap {fmt(abs(mean_redr-mean_pub))})")
    print(f"  pass@8 control {fmt(mean_ctl_pass)} -> E132 {fmt(mean_pass)}")
    print(f"  paired cells: {census['positive']} positive, {census['zero']} zero, "
          f"{census['negative']} negative")
    if census["zero_cells"]:
        print(f"    zero: {', '.join(census['zero_cells'])}")
    print("  per-domain paired effect (seeds positive):")
    for domain in DOMAINS:
        v = domains[domain]
        print(f"    {domain:16s} {signed(v['paired_pmd_effect'])}  {v['seeds_positive']}/5")
    for name, rec in predictions.items():
        print(f"  prediction {name}: {'held' if rec['held'] else 'NOT BORNE OUT'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
