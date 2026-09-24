#!/usr/bin/env python3
"""The four-rung replay mechanism ladder, measured on one PCMD axis.

Re:Dr differs from its control in three separable ways: it retains a verified
success at all, it retains several of them, and it rehearses those uniformly
rather than in proportion to how often each was discovered. Three cohorts
between them isolate each step, and this generator places all four rungs on the
same estimator so the decomposition can be read off directly:

``control``      E128, no replay.
``capacity_one`` E130, one slot: a success is retained and rehearsed, but mode
                 identity cannot enter the objective.
``frequency``    E120-R1, full capacity weighted by discovery frequency: many
                 modes retained, rehearsed in proportion to how often each was
                 seen. Its published payload reports ``distinct@8``, the
                 superseded breadth metric, so PCMD is measured here from its
                 own terminal mode-coverage draws.
``uniform``      Re:Dr, full capacity rehearsed uniformly.

The ladder is **not monotonic**, which is the point of emitting it: retaining
many modes under frequency weighting scores below retaining exactly one. A
frequency-weighted buffer concentrates on the mode that was already common, so
uniformity is not a bonus on top of multi-mode retention --- it is what makes
multi-mode retention pay at all.

``recovered_fraction`` is emitted per domain as well as on the mean, because it
ranges from near zero to near one and a single aggregate hides that. No
threshold decides which domains "count": every ratio is reported next to the
denominator it was divided by, so a reader can see directly which ones rest on
a small base. An earlier draft cut at a Re:Dr effect of 0.10 and thereby
excluded ``python_factors`` by 0.001, which also implied the domain has no
replay effect at all. It does --- at Level 3 replay takes it from .000 to .418,
the largest gain of any domain --- so ``level3_reference`` carries that figure
for the two domains whose Level-1 effect is small.

Cohort seam: the four rungs come from three cohorts. For PCMD this is
defensible --- the recorded E128-minus-E78 control shift is +.0028 over 20
paired cells --- and that figure is carried into the payload rather than left
implicit. E132 is measuring a within-cohort Re:Dr; when it lands, ``--redr``
can point this generator at it instead.
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

E120_LEDGER = ROOT / "var/artifacts/e120r1_frequency_weighted_replay_jobs.json"
E130_PAYLOAD = ROOT / "paper/results/mode_agnostic_replay_20260923.json"
COMPARATORS = ROOT / "paper/results/mode_diversity_comparators_diversity_05b.json"
REFKL = ROOT / "paper/results/reference_kl_comparison.json"

DOMAINS = ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan")
SEEDS = (43, 44, 45, 46, 47)
LABEL = {
    "graph_coloring": r"\texttt{Graph}",
    "countdown": r"\texttt{Countdown}",
    "python_factors": r"\texttt{Python}",
    "mathir": r"\texttt{MathIR}",
    "pantry_plan": r"\texttt{PantryPlan}",
}
#: Level-3 replay gains, for the domains whose Level-1 effect is small. Python
#: is the case that matters: its Level-1 Re:Dr effect is a tenth of a mode pair,
#: but at Level 3 replay takes it from .000 to .418 over five seeds in both arms.
#: A reader who sees only this cohort would wrongly conclude the domain has no
#: replay effect, so the ladder carries the Level-3 figure rather than a
#: threshold that would quietly exclude it.
LEVEL3_REFERENCE = {
    "python_factors": {
        "control": 0.0,
        "replay": 0.41763,
        "seeds": 5,
        "note": "largest Level-3 gain of any domain, and also the one that does "
                "not survive key coarsening (App. coarser keys)",
    },
    "mathir": {"control": 0.0, "replay": 0.05272, "seeds": 5, "note": None},
}
SCHEMA = "paper-replay-mechanism-ladder-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_extractor():
    target = ROOT / "ops/extract_diversity_comparator_mode_diversity.py"
    spec = importlib.util.spec_from_file_location("_dc_extract", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, sha256(target)


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
        help="payload carrying the uniform-rehearsal arm; swap for E132 once it lands",
    )
    args = parser.parse_args()
    if not args.redr.is_absolute():
        args.redr = (ROOT / args.redr).resolve()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    extract, extract_sha = load_extractor()

    # --- frequency-weighted rung: measure PCMD from its own draws -----------
    ledger = json.loads(E120_LEDGER.read_text(encoding="utf-8"))
    freq: dict[tuple[str, int], dict] = {}
    for run in ledger["runs"]:
        if "qwen05b" not in Path(run["run_dir"]).name:
            continue
        freq[(run["domain"], int(run["seed"]))] = extract.cell_pcmd(run["run_dir"])
    missing = [k for k in ((d, s) for d in DOMAINS for s in SEEDS) if k not in freq]
    if missing:
        raise SystemExit(f"E120-R1 is short {len(missing)} Qwen-0.5B cells: {missing}")
    freq_steps = {c["terminal_step"] for c in freq.values()}

    # --- the other three rungs ---------------------------------------------
    e130 = {r["domain"]: r for r in json.loads(E130_PAYLOAD.read_text(encoding="utf-8"))["domains"]}
    comparators = json.loads(COMPARATORS.read_text(encoding="utf-8"))
    redr_payload = json.loads(args.redr.read_text(encoding="utf-8"))["domains"]

    rows = []
    for domain in DOMAINS:
        cells = [freq[(domain, s)] for s in SEEDS]
        row = {
            "domain": domain,
            "control": e130[domain]["base_pmd"],
            "capacity_one": e130[domain]["abl_pmd"],
            "frequency": st.fmean(c["pmd"] for c in cells),
            "frequency_reportable": sum(int(c["reportable"]) for c in cells),
            "uniform": redr_payload[domain]["replay"]["pmd"],
        }
        row["redr_effect"] = row["uniform"] - row["control"]
        row["capacity_one_effect"] = row["capacity_one"] - row["control"]
        row["recovered_fraction"] = (
            row["capacity_one_effect"] / row["redr_effect"]
            if row["redr_effect"] > 0 else None
        )
        row["level3_reference"] = LEVEL3_REFERENCE.get(domain)
        rows.append(row)

    mean = {
        k: st.fmean(r[k] for r in rows)
        for k in ("control", "capacity_one", "frequency", "uniform",
                  "redr_effect", "capacity_one_effect")
    }
    total = mean["uniform"] - mean["control"]
    steps = {
        "retain_one": mean["capacity_one"] - mean["control"],
        "retain_many_frequency_weighted": mean["frequency"] - mean["capacity_one"],
        "balance_uniformly": mean["uniform"] - mean["frequency"],
    }
    small_base = sorted(r["domain"] for r in rows if r["redr_effect"] < 0.10)

    payload = {
        "schema": SCHEMA,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "estimator": "pooled per-prompt PCMD over verified responses, "
                     "ops/mode_diversity.py, via extract_diversity_comparator_"
                     "mode_diversity.cell_pcmd",
        "rungs": {
            "control": "E128, no replay",
            "capacity_one": "E130, one slot, mode identity absent",
            "frequency": "E120-R1, full capacity, discovery-frequency weights",
            "uniform": f"Re:Dr, full capacity, uniform rehearsal "
                       f"(from {args.redr.name})",
        },
        "monotonic": steps["retain_many_frequency_weighted"] >= 0,
        "frequency_terminal_steps": sorted(freq_steps),
        "cohort_seam": comparators.get("control_shift_vs_e78"),
        "small_denominator_domains": small_base,
        "small_denominator_note": "reported, not excluded: their Level-1 Re:Dr "
                                  "effect is a few hundredths, so the ratio is "
                                  "unstable; see level3_reference per domain",
        "sources": {
            "e120r1_ledger": {"path": str(E120_LEDGER.relative_to(ROOT)),
                              "sha256": sha256(E120_LEDGER)},
            "e130": {"path": str(E130_PAYLOAD.relative_to(ROOT)),
                     "sha256": sha256(E130_PAYLOAD)},
            "comparators": {"path": str(COMPARATORS.relative_to(ROOT)),
                            "sha256": sha256(COMPARATORS)},
            "redr": {"path": str(args.redr.relative_to(ROOT)),
                     "sha256": sha256(args.redr)},
            "extractor": {"path": "ops/extract_diversity_comparator_mode_diversity.py",
                          "sha256": extract_sha},
        },
        "domains": rows,
        "means": mean,
        "decomposition": {
            "total_redr_effect": total,
            "steps": steps,
            "shares": {k: v / total for k, v in steps.items()},
        },
    }
    (out / f"replay_mechanism_ladder_{args.stamp}.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )

    macros = [
        "% Generated by ops/build_paper_replay_mechanism_ladder.py; do not hand edit.",
        rf"\newcommand{{\LADctl}}{{{fmt(mean['control'])}}}",
        rf"\newcommand{{\LADone}}{{{fmt(mean['capacity_one'])}}}",
        rf"\newcommand{{\LADfreq}}{{{fmt(mean['frequency'])}}}",
        rf"\newcommand{{\LADuni}}{{{fmt(mean['uniform'])}}}",
        rf"\newcommand{{\LADretainstep}}{{{signed(steps['retain_one'])}}}",
        rf"\newcommand{{\LADfreqstep}}{{{signed(steps['retain_many_frequency_weighted'])}}}",
        rf"\newcommand{{\LADbalancestep}}{{{signed(steps['balance_uniformly'])}}}",
        rf"\newcommand{{\LADsmallbase}}{{{len(small_base)}}}",
        rf"\newcommand{{\LADpythonlthree}}{{{fmt(LEVEL3_REFERENCE['python_factors']['replay'])}}}",
        rf"\newcommand{{\LADseamshift}}{{{signed(payload['cohort_seam']['mean'], 4)}}}",
        rf"\newcommand{{\LADseamcells}}{{{payload['cohort_seam']['paired_cells']}}}",
    ]
    for r in rows:
        tag = r["domain"].split("_")[0]
        if r["recovered_fraction"] is not None:
            macros.append(
                rf"\newcommand{{\LAD{tag}recovered}}{{{round(100*r['recovered_fraction'])}\%}}"
            )
        macros.append(rf"\newcommand{{\LAD{tag}denom}}{{{fmt(r['redr_effect'])}}}")
        macros.append(rf"\newcommand{{\LAD{tag}freq}}{{{fmt(r['frequency'])}}}")
    (out / f"replay_mechanism_ladder_{args.stamp}_macros.tex").write_text(
        "\n".join(macros) + "\n", encoding="utf-8"
    )

    body = ["% Generated by ops/build_paper_replay_mechanism_ladder.py; do not hand edit."]
    for r in rows:
        rec = (rf"{round(100*r['recovered_fraction'])}\%"
               if r["recovered_fraction"] is not None else r"---")
        body.append(
            f"  {LABEL[r['domain']]} & {fmt(r['control'])} & {fmt(r['capacity_one'])}"
            f" & {fmt(r['frequency'])} & {fmt(r['uniform'])}"
            f" & {fmt(r['redr_effect'])} & {rec} \\\\"
        )
    body.append(r"  \midrule")
    body.append(
        f"  Mean & {fmt(mean['control'])} & {fmt(mean['capacity_one'])}"
        f" & {fmt(mean['frequency'])} & {fmt(mean['uniform'])}"
        f" & {fmt(mean['redr_effect'])} & \\\\"
    )
    body.append(r"  \bottomrule")
    (out / f"replay_mechanism_ladder_{args.stamp}_table_body.tex").write_text(
        "\n".join(body) + "\n", encoding="utf-8"
    )

    print(f"wrote replay_mechanism_ladder_{args.stamp}.{{json,_macros.tex,_table_body.tex}}")
    print(f"  control {fmt(mean['control'])} -> cap-1 {fmt(mean['capacity_one'])}"
          f" -> freq-wtd {fmt(mean['frequency'])} -> uniform {fmt(mean['uniform'])}")
    print(f"  monotonic: {payload['monotonic']}")
    for k, v in steps.items():
        print(f"    {k:32s} {signed(v)}  ({100*v/total:+.0f}% of the total effect)")
    print(f"  small Level-1 denominators (reported, not excluded): {small_base}")
    print("  per-domain recovered fraction (denominator in brackets):")
    for r in rows:
        rf_ = (f"{round(100*r['recovered_fraction'])}%"
               if r["recovered_fraction"] is not None else "n/a")
        l3 = r.get("level3_reference")
        extra = (f"   [L3 replay {l3['replay']:.3f} vs control {l3['control']:.3f}]"
                 if l3 else "")
        print(f"    {r['domain']:16s} {rf_:>5s}  [denom {fmt(r['redr_effect'])}]{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
