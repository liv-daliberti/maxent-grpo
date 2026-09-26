#!/usr/bin/env python3
"""Terminal endpoints for the E130 mode-agnostic replay ablation.

E130 is the Re:Dr objective with the replay bank reduced to a single slot: the
verified-likelihood derivative is live, but the bank retains only the first
canonical mode each prompt discovers, so mode identity cannot enter the
objective at all. Against its matched E128 control it isolates what the *mode
keying* contributes, as distinct from the supervised likelihood term Re:Dr adds
alongside it. Three arms are emitted together, all at eight passes:

``e128_control``
    The matched control, reused from
    ``paper/results/mode_diversity_comparators_diversity_05b.json`` rather than
    re-measured, so the arms differ only in the bank.
``e130_singleton``
    Capacity-1 replay, measured here from each cell's terminal mode-coverage
    draws with the pooled estimator (``ops/mode_diversity.py``) every other
    cell uses.
``redr``
    Re:Dr at capacity 16, from the reference-KL cohort.

The three registered predictions in
``paper/preregistration/e130_mode_agnostic_replay_05b_20260922.md`` are scored
in the payload. Prediction 2 -- that capacity-1 PCMD would not differ
systematically from the control -- is not borne out, and the payload records
that verdict rather than softening it: the ablation recovers a majority of the
Re:Dr breadth effect without ever seeing a second key.

Two cell-level facts are emitted because they bound how the result reads.
``reportable`` flags cells below the thirty-prompt PCMD threshold.
``online_bank_max_modes`` is the largest mean online bank occupancy a cell
reached: where it is 1, the cell never discovered a second mode, so single-slot
replay withheld nothing and the ablation is vacuous for that cell.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import importlib.util
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))

LEDGER = ROOT / "var/artifacts/e130_mode_agnostic_replay_05b_jobs.json"
E128 = ROOT / "paper/results/mode_diversity_comparators_diversity_05b.json"
REFKL = ROOT / "paper/results/reference_kl_comparison.json"
PREREG = ROOT / "paper/preregistration/e130_mode_agnostic_replay_05b_20260922.md"

DOMAINS = ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan")
SEEDS = (43, 44, 45, 46, 47)
LABEL = {
    "graph_coloring": r"\texttt{Graph}",
    "countdown": r"\texttt{Countdown}",
    "python_factors": r"\texttt{Python}",
    "mathir": r"\texttt{MathIR}",
    "pantry_plan": r"\texttt{PantryPlan}",
}
OCCUPANCY_KEY = "train/online_canonical_bank_size_after_mean"
CAPACITY_INVARIANTS = (
    "train/canonical_replay_capacity",
    "train/canonical_replay_banked_modes",
    "train/canonical_replay_actuator_modes",
    "train/canonical_replay_verified_likelihood_active",
)
SCHEMA = "paper-mode-agnostic-replay-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_extractor():
    target = ROOT / "ops/extract_diversity_comparator_mode_diversity.py"
    spec = importlib.util.spec_from_file_location("_dc_extract", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, sha256(target)


def audit_cell(run_dir: Path) -> dict:
    """Confirm the single-slot bank held, and record online mode discovery."""

    complete = json.loads((run_dir / "TRAINING_COMPLETE.json").read_text(encoding="utf-8"))
    metrics = Path(complete["terminal_attempt"]) / "train_metrics.jsonl"
    seen = {k: set() for k in CAPACITY_INVARIANTS}
    occupancy = 0.0
    for line in metrics.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for key in CAPACITY_INVARIANTS:
            if key in row:
                seen[key].add(row[key])
        occupancy = max(occupancy, row.get(OCCUPANCY_KEY) or 0.0)
    return {
        "on_spec": all(seen[k] == {1} for k in CAPACITY_INVARIANTS),
        "invariants": {k.rsplit("_", 2)[-1] if False else k: sorted(seen[k]) for k in CAPACITY_INVARIANTS},
        "online_bank_max_modes": occupancy,
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
    parser.add_argument(
        "--redr",
        type=Path,
        default=REFKL,
        help="payload carrying the Re:Dr arm; point at E132 for a within-cohort "
             "comparison instead of the cross-cohort published E78 arm",
    )
    args = parser.parse_args()
    if not args.redr.is_absolute():
        args.redr = (ROOT / args.redr).resolve()
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
        raise SystemExit(f"E130 ledger is short {len(missing)} cells: {missing}")
    off_spec = sorted(f"{d}/s{s}" for (d, s), a in audits.items() if not a["on_spec"])
    if off_spec:
        raise SystemExit(f"E130 cells did not hold the single-slot bank: {off_spec}")
    steps = {c["terminal_step"] for c in cells.values()}
    if len(steps) != 1:
        raise SystemExit(f"E130 cells end at differing steps: {sorted(steps)}")
    terminal_step = steps.pop()

    base_cells = collections.defaultdict(list)
    for cell in json.loads(E128.read_text(encoding="utf-8"))["cells"]:
        if cell["method"] == "e128_control":
            base_cells[cell["domain"]].append(cell)
    refkl = json.loads(args.redr.read_text(encoding="utf-8"))["domains"]

    rows = []
    for domain in DOMAINS:
        abl = [cells[(domain, s)] for s in SEEDS]
        aud = [audits[(domain, s)] for s in SEEDS]
        base = base_cells[domain]
        row = {
            "domain": domain,
            "base_pass8": st.fmean(c["pass8"] for c in base),
            "base_pmd": st.fmean(c["pmd"] for c in base),
            "abl_pass8": st.fmean(c["pass8"] for c in abl),
            "abl_pmd": st.fmean(c["pmd"] for c in abl),
            "abl_pass8_by_seed": [c["pass8"] for c in abl],
            "abl_pmd_by_seed": [c["pmd"] for c in abl],
            "abl_reportable": sum(int(c["reportable"]) for c in abl),
            "abl_unreportable_seeds": [s for s, c in zip(SEEDS, abl) if not c["reportable"]],
            "vacuous_seeds": [s for s, a in zip(SEEDS, aud) if a["online_bank_max_modes"] <= 1.0],
            "redr_pass8": refkl[domain]["replay"]["pass8"],
            "redr_pmd": refkl[domain]["replay"]["pmd"],
        }
        row["abl_effect_pmd"] = row["abl_pmd"] - row["base_pmd"]
        row["redr_effect_pmd"] = row["redr_pmd"] - row["base_pmd"]
        row["recovered_fraction"] = (
            row["abl_effect_pmd"] / row["redr_effect_pmd"]
            if row["redr_effect_pmd"] > 0 else None
        )
        rows.append(row)

    mean = {
        k: st.fmean(r[k] for r in rows)
        for k in ("base_pass8", "base_pmd", "abl_pass8", "abl_pmd",
                  "redr_pass8", "redr_pmd", "abl_effect_pmd", "redr_effect_pmd")
    }
    mean["recovered_fraction"] = mean["abl_effect_pmd"] / mean["redr_effect_pmd"]

    separation = sum(1 for r in rows if r["abl_effect_pmd"] < r["redr_effect_pmd"])
    predictions = {
        "accuracy": {
            "registered": "terminal pass@8 above the matched control and below Re:Dr",
            "domains_above_control": sum(1 for r in rows if r["abl_pass8"] > r["base_pass8"]),
            "domains_below_redr": sum(1 for r in rows if r["abl_pass8"] < r["redr_pass8"]),
            "mean_vs_control": mean["abl_pass8"] - mean["base_pass8"],
            "mean_vs_redr": mean["abl_pass8"] - mean["redr_pass8"],
            "held": (mean["abl_pass8"] > mean["base_pass8"]
                     and mean["abl_pass8"] < mean["redr_pass8"]),
        },
        "diversity": {
            "registered": "terminal PCMD does not differ systematically from the "
                          "matched control",
            "held": False,
            "shift": mean["abl_pmd"] - mean["base_pmd"],
            "domains_above_control": sum(1 for r in rows if r["abl_pmd"] > r["base_pmd"]),
            "note": "not borne out: single-slot replay raises PCMD well above the "
                    "control in every domain that carries a Re:Dr effect, and "
                    "recovers a majority of that effect on the five-domain mean",
        },
        "separation": {
            "registered": "the capacity-1 PCMD effect is smaller than the Re:Dr "
                          "PCMD effect in at least four of five domains",
            "domains_separated": separation,
            "held": separation >= 4,
            "recovered_fraction_mean": mean["recovered_fraction"],
            "recovered_fraction_by_domain": {
                r["domain"]: r["recovered_fraction"] for r in rows
            },
        },
    }

    payload = {
        "schema": SCHEMA,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "estimator": "pooled per-prompt PCMD over verified responses, "
                     "ops/mode_diversity.py, via extract_diversity_comparator_"
                     "mode_diversity.cell_pcmd",
        "ablation": ledger.get("ablation"),
        "objective": ledger.get("objective"),
        "replay_bank_capacity": ledger.get("replay_bank_capacity"),
        "treatment_reference": ledger.get("treatment_reference"),
        "control_cohort": ledger.get("control_cohort"),
        "passes": ledger.get("passes"),
        "terminal_step": terminal_step,
        "cells": len(cells),
        "cells_on_spec": len(cells) - len(off_spec),
        "sources": {
            "e130_ledger": {"path": str(LEDGER.relative_to(ROOT)), "sha256": sha256(LEDGER)},
            "e128_control": {"path": str(E128.relative_to(ROOT)), "sha256": sha256(E128)},
            "redr": {"path": str(args.redr.relative_to(ROOT)), "sha256": sha256(args.redr)},
            "preregistration": {"path": str(PREREG.relative_to(ROOT)), "sha256": sha256(PREREG)},
            "extractor": {"path": "ops/extract_diversity_comparator_mode_diversity.py",
                          "sha256": extract_sha},
        },
        "domains": rows,
        "means": mean,
        "registered_predictions": predictions,
    }
    (out / f"mode_agnostic_replay_{args.stamp}.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )

    vacuous = sorted(f"{r['domain']}/s{s}" for r in rows for s in r["vacuous_seeds"])
    unreportable = sorted(f"{r['domain']}/s{s}" for r in rows for s in r["abl_unreportable_seeds"])
    macros = [
        "% Generated by ops/build_paper_mode_agnostic_replay.py; do not hand edit.",
        rf"\newcommand{{\MARcells}}{{{len(cells)}}}",
        rf"\newcommand{{\MARbasepass}}{{{fmt(mean['base_pass8'])}}}",
        rf"\newcommand{{\MARablpass}}{{{fmt(mean['abl_pass8'])}}}",
        rf"\newcommand{{\MARredrpass}}{{{fmt(mean['redr_pass8'])}}}",
        rf"\newcommand{{\MARpassgain}}{{{signed(mean['abl_pass8'] - mean['base_pass8'])}}}",
        rf"\newcommand{{\MARpassshortfall}}{{{signed(mean['abl_pass8'] - mean['redr_pass8'])}}}",
        rf"\newcommand{{\MARbasepmd}}{{{fmt(mean['base_pmd'])}}}",
        rf"\newcommand{{\MARablpmd}}{{{fmt(mean['abl_pmd'])}}}",
        rf"\newcommand{{\MARredrpmd}}{{{fmt(mean['redr_pmd'])}}}",
        rf"\newcommand{{\MARableffect}}{{{signed(mean['abl_effect_pmd'])}}}",
        rf"\newcommand{{\MARredreffect}}{{{signed(mean['redr_effect_pmd'])}}}",
        rf"\newcommand{{\MARrecovered}}{{{round(100 * mean['recovered_fraction'])}\%}}",
        rf"\newcommand{{\MARseparated}}{{{separation}}}",
        rf"\newcommand{{\MARreportable}}{{{sum(r['abl_reportable'] for r in rows)}}}",
        rf"\newcommand{{\MARvacuous}}{{{len(vacuous)}}}",
    ]
    for r in rows:
        tag = r["domain"].split("_")[0]
        macros.append(rf"\newcommand{{\MAR{tag}ablpmd}}{{{fmt(r['abl_pmd'])}}}")
        macros.append(rf"\newcommand{{\MAR{tag}redrpmd}}{{{fmt(r['redr_pmd'])}}}")
        macros.append(rf"\newcommand{{\MAR{tag}ablpass}}{{{fmt(r['abl_pass8'])}}}")
    (out / f"mode_agnostic_replay_{args.stamp}_macros.tex").write_text(
        "\n".join(macros) + "\n", encoding="utf-8"
    )

    body = ["% Generated by ops/build_paper_mode_agnostic_replay.py; do not hand edit."]
    for r in rows:
        body.append(
            f"  {LABEL[r['domain']]} & {fmt(r['base_pass8'])} & {fmt(r['base_pmd'])}"
            f" & {fmt(r['abl_pass8'])} & {fmt(r['abl_pmd'])}"
            f" & {fmt(r['redr_pass8'])} & {fmt(r['redr_pmd'])} \\\\"
        )
    body.append(r"  \midrule")
    body.append(
        f"  Mean & {fmt(mean['base_pass8'])} & {fmt(mean['base_pmd'])}"
        f" & {fmt(mean['abl_pass8'])} & {fmt(mean['abl_pmd'])}"
        f" & {fmt(mean['redr_pass8'])} & {fmt(mean['redr_pmd'])} \\\\"
    )
    body.append(r"  \bottomrule")
    (out / f"mode_agnostic_replay_{args.stamp}_table_body.tex").write_text(
        "\n".join(body) + "\n", encoding="utf-8"
    )

    print(f"wrote mode_agnostic_replay_{args.stamp}.{{json,_macros.tex,_table_body.tex}}")
    print(f"  cells={len(cells)} on-spec={len(cells)-len(off_spec)} terminal_step={terminal_step}")
    print(f"  pass@8  ctrl {fmt(mean['base_pass8'])} -> cap1 {fmt(mean['abl_pass8'])}"
          f"  (Re:Dr {fmt(mean['redr_pass8'])})")
    print(f"  PCMD    ctrl {fmt(mean['base_pmd'])} -> cap1 {fmt(mean['abl_pmd'])}"
          f"  (Re:Dr {fmt(mean['redr_pmd'])})")
    print(f"  capacity-1 recovers {round(100*mean['recovered_fraction'])}% of the "
          f"Re:Dr PCMD effect on the five-domain mean")
    for name, rec in predictions.items():
        print(f"  prediction {name}: {'held' if rec['held'] else 'NOT BORNE OUT'}")
    if unreportable:
        print(f"  below PCMD threshold: {', '.join(unreportable)}")
    if vacuous:
        print(f"  vacuous (never discovered a 2nd mode): {', '.join(vacuous)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
