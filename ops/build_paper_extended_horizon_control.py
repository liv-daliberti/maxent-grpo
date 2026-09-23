#!/usr/bin/env python3
"""Terminal endpoints for the E131 extended-horizon control, against E128 and Re:Dr.

App. "The missing comparison at equal total compute" needs one number it never
had: what the control does when its share of the budget is spent on more
learning rather than switched off at the replay derivative. E131 is that cell
set -- the E128 control objective with only the horizon moved, twelve prompt
passes instead of eight, 4,608 optimizer updates against 3,072, with 50% more
fresh rollouts on the same prompts, seeds, placement, runtime and evaluator.

Three arms are emitted together because the appendix reads them together:

``e128_control``
    The eight-pass control, measured by
    ``ops/extract_diversity_comparator_mode_diversity.py`` and reused here
    rather than re-measured, so the two horizons differ only in horizon.
``e131_control``
    The twelve-pass control, measured here from each cell's terminal
    mode-coverage draws with the same pooled estimator
    (``ops/mode_diversity.py``) every other cell uses.
``redr``
    Re:Dr at eight passes, from the reference-KL cohort, which is the one
    payload carrying both a control and a replay arm on a single evaluation.

The registered predictions live in
``paper/preregistration/e131_extended_horizon_control_05b_20260922.md`` and are
scored in the emitted payload, including where the measurement runs against
them. Nothing here filters on reportability; ``reportable`` is emitted per cell
so the appendix can say which columns carry a \\pmd{} estimate at all.
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

LEDGER = ROOT / "var/artifacts/e131_extended_horizon_control_05b_jobs.json"
E128 = ROOT / "paper/results/mode_diversity_comparators_diversity_05b.json"
REFKL = ROOT / "paper/results/reference_kl_comparison.json"
PREREG = ROOT / "paper/preregistration/e131_extended_horizon_control_05b_20260922.md"

DOMAINS = ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan")
SEEDS = (43, 44, 45, 46, 47)
LABEL = {
    "graph_coloring": r"\texttt{Graph}",
    "countdown": r"\texttt{Countdown}",
    "python_factors": r"\texttt{Python}",
    "mathir": r"\texttt{MathIR}",
    "pantry_plan": r"\texttt{PantryPlan}",
}
SCHEMA = "paper-extended-horizon-control-v1"


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

    # --- E131: measure every cell from its own terminal draws -------------
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    ext_cells: dict[tuple[str, int], dict] = {}
    for run in ledger["runs"]:
        ext_cells[(run["domain"], int(run["seed"]))] = extract.cell_pcmd(run["run_dir"])
    missing = [k for k in ((d, s) for d in DOMAINS for s in SEEDS) if k not in ext_cells]
    if missing:
        raise SystemExit(f"E131 ledger is short {len(missing)} cells: {missing}")
    steps = {c["terminal_step"] for c in ext_cells.values()}
    if len(steps) != 1:
        raise SystemExit(f"E131 cells end at differing steps: {sorted(steps)}")
    terminal_step = steps.pop()

    # --- E128: reuse the published eight-pass control ----------------------
    base_cells = collections.defaultdict(list)
    for cell in json.loads(E128.read_text(encoding="utf-8"))["cells"]:
        if cell["method"] == "e128_control":
            base_cells[cell["domain"]].append(cell)

    # --- Re:Dr at eight passes --------------------------------------------
    refkl = json.loads(REFKL.read_text(encoding="utf-8"))["domains"]

    rows = []
    for domain in DOMAINS:
        ext = [ext_cells[(domain, s)] for s in SEEDS]
        base = base_cells[domain]
        rows.append(
            {
                "domain": domain,
                "base_pass8": st.fmean(c["pass8"] for c in base),
                "base_pmd": st.fmean(c["pmd"] for c in base),
                "ext_pass8": st.fmean(c["pass8"] for c in ext),
                "ext_pmd": st.fmean(c["pmd"] for c in ext),
                "ext_reportable": sum(int(c["reportable"]) for c in ext),
                "ext_defined_prompts": st.fmean(c["defined_prompts"] for c in ext),
                "redr_pass8": refkl[domain]["replay"]["pass8"],
                "redr_pmd": refkl[domain]["replay"]["pmd"],
                "seeds": len(ext),
            }
        )

    mean = {
        key: st.fmean(r[key] for r in rows)
        for key in (
            "base_pass8", "base_pmd", "ext_pass8", "ext_pmd", "redr_pass8", "redr_pmd"
        )
    }
    below = sum(1 for r in rows if r["ext_pass8"] < r["redr_pass8"])
    pmd_up = sum(1 for r in rows if r["ext_pmd"] > r["base_pmd"])
    python = next(r for r in rows if r["domain"] == "python_factors")

    predictions = {
        "gap_does_not_close": {
            "registered": "E131 terminal pass@8 remains below Re:Dr terminal pass@8",
            "held": mean["ext_pass8"] < mean["redr_pass8"],
            "domains_below": below,
            "margin": mean["redr_pass8"] - mean["ext_pass8"],
        },
        "diversity_does_not_recover": {
            "registered": "E131 PCMD does not differ systematically from the "
                          "eight-pass control, concentrating further or holding",
            "held_as_registered": mean["ext_pmd"] <= mean["base_pmd"],
            "domains_above_base": pmd_up,
            "shift": mean["ext_pmd"] - mean["base_pmd"],
            "note": "measured direction is a small rise rather than a hold; both "
                    "horizons remain within .02 PCMD of a single verified mode",
        },
        "python_is_the_sharpest_case": {
            "registered": "the additional updates change terminal pass@8 by less "
                          "than the additional passes would suggest",
            "held": python["ext_pass8"] <= python["base_pass8"],
            "shift": python["ext_pass8"] - python["base_pass8"],
            "reportable_pmd_cells": python["ext_reportable"],
        },
    }

    payload = {
        "schema": SCHEMA,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "estimator": "pooled per-prompt PCMD over verified responses, "
                     "ops/mode_diversity.py, via extract_diversity_comparator_"
                     "mode_diversity.cell_pcmd",
        "comparison": ledger.get("comparison"),
        "horizon_passes": ledger.get("horizon_passes"),
        "baseline_horizon_passes": ledger.get("baseline_horizon_passes"),
        "terminal_step": terminal_step,
        "cells": len(ext_cells),
        "sources": {
            "e131_ledger": {"path": str(LEDGER.relative_to(ROOT)), "sha256": sha256(LEDGER)},
            "e128_control": {"path": str(E128.relative_to(ROOT)), "sha256": sha256(E128)},
            "redr": {"path": str(REFKL.relative_to(ROOT)), "sha256": sha256(REFKL)},
            "preregistration": {"path": str(PREREG.relative_to(ROOT)), "sha256": sha256(PREREG)},
            "extractor": {"path": "ops/extract_diversity_comparator_mode_diversity.py",
                          "sha256": extract_sha},
        },
        "domains": rows,
        "means": mean,
        "registered_predictions": predictions,
    }
    (out / f"extended_horizon_control_{args.stamp}.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )

    macros = [
        "% Generated by ops/build_paper_extended_horizon_control.py; do not hand edit.",
        rf"\newcommand{{\EHCpasses}}{{{payload['horizon_passes']}}}",
        rf"\newcommand{{\EHCbasepasses}}{{{payload['baseline_horizon_passes']}}}",
        rf"\newcommand{{\EHCupdates}}{{{terminal_step:,}}}".replace(",", "{,}"),
        rf"\newcommand{{\EHCbaseupdates}}{{3{{,}}072}}",
        rf"\newcommand{{\EHCcells}}{{{len(ext_cells)}}}",
        rf"\newcommand{{\EHCbasepass}}{{{fmt(mean['base_pass8'])}}}",
        rf"\newcommand{{\EHCextpass}}{{{fmt(mean['ext_pass8'])}}}",
        rf"\newcommand{{\EHCredrpass}}{{{fmt(mean['redr_pass8'])}}}",
        rf"\newcommand{{\EHCpassshift}}{{{signed(mean['ext_pass8'] - mean['base_pass8'])}}}",
        rf"\newcommand{{\EHCgap}}{{{fmt(mean['redr_pass8'] - mean['ext_pass8'])}}}",
        rf"\newcommand{{\EHCbasepmd}}{{{fmt(mean['base_pmd'])}}}",
        rf"\newcommand{{\EHCextpmd}}{{{fmt(mean['ext_pmd'])}}}",
        rf"\newcommand{{\EHCredrpmd}}{{{fmt(mean['redr_pmd'])}}}",
        rf"\newcommand{{\EHCbaseneff}}{{{neff(mean['base_pmd']):.3f}}}",
        rf"\newcommand{{\EHCextneff}}{{{neff(mean['ext_pmd']):.3f}}}",
        rf"\newcommand{{\EHCredrneff}}{{{neff(mean['redr_pmd']):.3f}}}",
        rf"\newcommand{{\EHCbelow}}{{{below}}}",
        rf"\newcommand{{\EHCpmdup}}{{{pmd_up}}}",
        rf"\newcommand{{\EHCpythonbase}}{{{fmt(python['base_pass8'])}}}",
        rf"\newcommand{{\EHCpythonext}}{{{fmt(python['ext_pass8'])}}}",
        rf"\newcommand{{\EHCpythonshift}}{{{signed(python['ext_pass8'] - python['base_pass8'])}}}",
        rf"\newcommand{{\EHCpythonreportable}}{{{python['ext_reportable']}}}",
        rf"\newcommand{{\EHCpythondefined}}{{{python['ext_defined_prompts']:.1f}}}",
    ]
    (out / f"extended_horizon_control_{args.stamp}_macros.tex").write_text(
        "\n".join(macros) + "\n", encoding="utf-8"
    )

    body = [
        "% Generated by ops/build_paper_extended_horizon_control.py; do not hand edit."
    ]
    for r in rows:
        body.append(
            f"  {LABEL[r['domain']]} & {fmt(r['base_pass8'])} & {fmt(r['base_pmd'])}"
            f" & {fmt(r['ext_pass8'])} & {fmt(r['ext_pmd'])}"
            f" & {fmt(r['redr_pass8'])} & {fmt(r['redr_pmd'])} \\\\"
        )
    body.append(r"  \midrule")
    body.append(
        f"  Mean & {fmt(mean['base_pass8'])} & {fmt(mean['base_pmd'])}"
        f" & {fmt(mean['ext_pass8'])} & {fmt(mean['ext_pmd'])}"
        f" & {fmt(mean['redr_pass8'])} & {fmt(mean['redr_pmd'])} \\\\"
    )
    body.append(r"  \bottomrule")
    (out / f"extended_horizon_control_{args.stamp}_table_body.tex").write_text(
        "\n".join(body) + "\n", encoding="utf-8"
    )

    print(f"wrote extended_horizon_control_{args.stamp}.{{json,_macros.tex,_table_body.tex}}"
          f" -> {out.relative_to(ROOT)}")
    print(f"  cells={len(ext_cells)} terminal_step={terminal_step}")
    print(f"  pass@8  base {fmt(mean['base_pass8'])} -> ext {fmt(mean['ext_pass8'])}"
          f"  (Re:Dr {fmt(mean['redr_pass8'])}, gap {fmt(mean['redr_pass8']-mean['ext_pass8'])})")
    print(f"  PCMD    base {fmt(mean['base_pmd'])} -> ext {fmt(mean['ext_pmd'])}"
          f"  (Re:Dr {fmt(mean['redr_pmd'])})")
    for name, rec in predictions.items():
        flag = rec.get("held", rec.get("held_as_registered"))
        print(f"  prediction {name}: {'held' if flag else 'runs against the registration'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
