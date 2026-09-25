#!/usr/bin/env python3
"""Terminal endpoints for the E124 Qwen-7B Level-1 MaxRL / Re:Max comparison.

E124 lifts the Level-1 comparison to Qwen2.5-7B under its own plan, transaction
and supervisor. Each cell is a single seed (70): at roughly thirty-seven hours
per cell on the one A100 the cohort is pinned to, the five-seed design the
0.5B cohorts use is not reachable, so this cohort reports paired single-seed
endpoints and says so rather than implying replication it does not have.

Two quantities are emitted per cell, because at this scale they separate:

``pass8``
    Terminal ``any_correct_at_k`` averaged over the independently seeded
    sampled draws -- whether the policy still answers the task at all.
``pmd``
    Terminal pooled per-prompt PCMD over verified responses
    (``ops/mode_diversity.py``, via
    ``extract_diversity_comparator_mode_diversity.cell_pcmd``), the same
    estimator every other cell in the paper uses -- how broad the correct
    answers are.

The separation matters because one control does not survive to be measured on
breadth at all. On MathIR the MaxRL control's accuracy falls to zero partway
through training and stays there, so it ends with no verified response, and
PCMD over an empty set of correct answers is undefined rather than zero. That
cell is reported as undefined with its cause recorded, never coerced to a
number and never dropped: a control that stops answering is a stronger
statement about the scale than a missing row, and the accuracy column carries
it where the breadth column cannot.

Domains are discovered from the plan rather than fixed here. A domain is
emitted once both of its arms carry ``TRAINING_COMPLETE.json``, so re-running
this generator after further cells terminalize extends the table with no edit
to this file or to the appendix. Cell counts reach the paper only as generated
macros for the same reason.

Provenance is taken from each cell's ``TRAINING_COMPLETE.json``, whose
``terminal_attempt`` names the attempt that actually trained. Cells that were
requeued mid-run hold more than one attempt directory, and their metrics files
carry the overlapping steps twice; resolving by that record rather than by
recency or by metric value is what keeps a requeued cell honest.
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

LEDGER = ROOT / "var/artifacts/e124_qwen7b_three_level_jobs.json"
PLAN = ROOT / "var/artifacts/e124_qwen7b_three_level/plan.json"

SCHEMA = 1
LEVEL = 1
CONTROL = "maxrl"
TREATMENT = "replay_maxrl"

LABEL = {
    "graph_coloring": r"\texttt{Graph}",
    "countdown": r"\texttt{Countdown}",
    "python_factors": r"\texttt{Python}",
    "mathir": r"\texttt{MathIR}",
    "pantry_plan": r"\texttt{PantryPlan}",
}
TAG = {
    "graph_coloring": "graph",
    "countdown": "countdown",
    "python_factors": "python",
    "mathir": "mathir",
    "pantry_plan": "pantry",
}
UNDEF = "--"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_extractor():
    target = ROOT / "ops/extract_diversity_comparator_mode_diversity.py"
    spec = importlib.util.spec_from_file_location("_dc_extract", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, sha256(target)


def fmt(value: float | None, places: int = 3) -> str:
    if value is None:
        return UNDEF
    text = f"{value:.{places}f}"
    if text.startswith("0."):
        return text[1:]
    if text.startswith("-0."):
        return "-" + text[2:]
    return text


def signed(value: float | None, places: int = 3) -> str:
    if value is None:
        return UNDEF
    return ("+" if value >= 0 else "-") + fmt(abs(value), places)


def accuracy_trace(attempt: Path) -> dict[int, float]:
    """Mean sampled any_correct_at_k per evaluated step."""

    per_step: dict[int, list[float]] = {}
    path = attempt / "eval_mode_coverage_draws.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("draw_index") is None:
            continue
        value = record["metrics"].get("any_correct_at_k")
        if value is not None:
            per_step.setdefault(int(record["step"]), []).append(float(value))
    return {step: st.fmean(vals) for step, vals in per_step.items()}


def collapse_step(trace: dict[int, float]) -> int | None:
    """First evaluated step whose accuracy is zero and never recovers."""

    steps = sorted(trace)
    for index, step in enumerate(steps):
        if all(trace[later] == 0.0 for later in steps[index:]):
            return step if trace[step] == 0.0 else None
    return None


def measure(cell: dict, extractor) -> dict:
    run_dir = Path(cell["run_dir"])
    complete = json.loads(
        (run_dir / "TRAINING_COMPLETE.json").read_text(encoding="utf-8")
    )
    attempt = Path(complete["terminal_attempt"])
    endpoint = extractor.cell_pcmd(str(run_dir))
    trace = accuracy_trace(attempt)
    collapsed = collapse_step(trace)
    return {
        "cell_id": cell["cell_id"],
        "domain": cell["domain"],
        "arm": cell["arm"],
        "seed": cell["seed"],
        "terminal_step": endpoint["terminal_step"],
        "terminal_attempt": attempt.name,
        "prompts": endpoint["prompts"],
        "defined_prompts": endpoint["defined_prompts"],
        "pass8": endpoint["pass8"],
        "pmd": endpoint["pmd"],
        "reportable": endpoint["reportable"],
        "collapse_step": collapsed,
        "collapsed": collapsed is not None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/results")
    parser.add_argument("--stamp", default=dt.date.today().isoformat().replace("-", ""))
    args = parser.parse_args()

    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    plan = json.loads(PLAN.read_text(encoding="utf-8"))

    # The plan was amended in place after launch: level1_domain_correction
    # cleared OAT_ZERO_MODEBENCH_DOMAIN on the Level-1 cells that named a
    # domain, which moved the file's hash past the one the ledger pins and the
    # running jobs carry in their Slurm comment. That drift is legitimate and
    # self-describing, so it is recorded rather than refused -- but an
    # unexplained divergence still is refused, because then the plan on disk is
    # not the plan the cohort was launched from.
    plan_sha = sha256(PLAN)
    pinned_sha = ledger.get("plan_sha256")
    correction = plan.get("level1_domain_correction")
    if pinned_sha and pinned_sha != plan_sha and not correction:
        raise SystemExit(
            "E124 plan does not match the hash the ledger pins and records no "
            "amendment explaining the difference; refusing to measure against a "
            "plan the cohort was not launched from"
        )

    extractor, extract_sha = load_extractor()

    cells = {
        (c["domain"], c["arm"]): c
        for c in plan["cells"]
        if c["level"] == LEVEL
    }
    domains = sorted({domain for domain, _ in cells})

    rows: list[dict] = []
    pending: list[str] = []
    for domain in domains:
        pair = [cells.get((domain, CONTROL)), cells.get((domain, TREATMENT))]
        if not all(pair):
            continue
        if not all((Path(c["run_dir"]) / "TRAINING_COMPLETE.json").exists() for c in pair):
            pending.append(domain)
            continue
        control, treatment = (measure(c, extractor) for c in pair)
        pmd_gain = (
            None
            if control["pmd"] is None or treatment["pmd"] is None
            else treatment["pmd"] - control["pmd"]
        )
        rows.append(
            {
                "domain": domain,
                "control": control,
                "treatment": treatment,
                "pass8_gain": treatment["pass8"] - control["pass8"],
                "pmd_gain": pmd_gain,
                "pmd_defined": pmd_gain is not None,
                "control_collapsed": control["collapsed"],
                "treatment_collapsed": treatment["collapsed"],
            }
        )

    if not rows:
        raise SystemExit("no E124 Level-1 domain has both arms terminalized yet")

    steps = {r[a]["terminal_step"] for r in rows for a in ("control", "treatment")}
    if len(steps) != 1:
        raise SystemExit(f"E124 Level-1 cells end at differing steps: {sorted(steps)}")
    terminal_step = steps.pop()

    seeds = sorted({r[a]["seed"] for r in rows for a in ("control", "treatment")})
    defined = [r for r in rows if r["pmd_defined"]]
    collapsed_controls = [r["domain"] for r in rows if r["control_collapsed"]]
    collapsed_treatments = [r["domain"] for r in rows if r["treatment_collapsed"]]

    means = {
        "control_pass8": st.fmean(r["control"]["pass8"] for r in rows),
        "treatment_pass8": st.fmean(r["treatment"]["pass8"] for r in rows),
        "pass8_gain": st.fmean(r["pass8_gain"] for r in rows),
        "control_pmd": st.fmean(r["control"]["pmd"] for r in defined) if defined else None,
        "treatment_pmd": st.fmean(r["treatment"]["pmd"] for r in defined) if defined else None,
        "pmd_gain": st.fmean(r["pmd_gain"] for r in defined) if defined else None,
    }

    payload = {
        "schema": SCHEMA,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cohort": "e124",
        "model_size": ledger.get("model_size") or plan.get("model"),
        "level": LEVEL,
        "seeds": seeds,
        "seeds_per_cell": len(seeds),
        "estimator": "pooled per-prompt PCMD over verified responses, "
                     "ops/mode_diversity.py, via extract_diversity_comparator_"
                     "mode_diversity.cell_pcmd",
        "provenance": "terminal attempt from each cell's TRAINING_COMPLETE.json",
        "terminal_step": terminal_step,
        "target_steps": ledger.get("target_steps") or plan["cells"][0].get("target_steps"),
        "domains_reported": [r["domain"] for r in rows],
        "domains_pending": pending,
        "pmd_defined_domains": [r["domain"] for r in defined],
        "pmd_undefined_domains": [r["domain"] for r in rows if not r["pmd_defined"]],
        "control_collapse": {
            "domains": collapsed_controls,
            "steps": {
                r["domain"]: r["control"]["collapse_step"]
                for r in rows
                if r["control_collapsed"]
            },
            "reading": "the control's sampled accuracy reaches zero and does not "
                       "recover, so it ends with no verified response and its "
                       "PCMD is undefined rather than zero",
        },
        "treatment_collapse": {"domains": collapsed_treatments},
        "sources": {
            "ledger": {"path": str(LEDGER.relative_to(ROOT)), "sha256": sha256(LEDGER)},
            "plan": {
                "path": str(PLAN.relative_to(ROOT)),
                "sha256": plan_sha,
                "ledger_pinned_sha256": pinned_sha,
                "amended_after_launch": bool(pinned_sha and pinned_sha != plan_sha),
                "amendment": correction,
            },
            "extractor": {
                "path": "ops/extract_diversity_comparator_mode_diversity.py",
                "sha256": extract_sha,
            },
        },
        "domains": rows,
        "means": means,
    }

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / f"e124_qwen7b_level1_{args.stamp}.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )

    macros = [
        "% Generated by ops/build_paper_e124_qwen7b_level1.py; do not hand edit.",
        rf"\newcommand{{\Eiifourdomains}}{{{len(rows)}}}",
        rf"\newcommand{{\Eiifourcells}}{{{2 * len(rows)}}}",
        rf"\newcommand{{\Eiifourseeds}}{{{len(seeds)}}}",
        rf"\newcommand{{\Eiifoursteps}}{{{terminal_step}}}",
        rf"\newcommand{{\Eiifourctrlpass}}{{{fmt(means['control_pass8'])}}}",
        rf"\newcommand{{\Eiifourtrtpass}}{{{fmt(means['treatment_pass8'])}}}",
        rf"\newcommand{{\Eiifourpassgain}}{{{signed(means['pass8_gain'])}}}",
        rf"\newcommand{{\Eiifourctrlpmd}}{{{fmt(means['control_pmd'])}}}",
        rf"\newcommand{{\Eiifourtrtpmd}}{{{fmt(means['treatment_pmd'])}}}",
        rf"\newcommand{{\Eiifourpmdgain}}{{{signed(means['pmd_gain'])}}}",
        rf"\newcommand{{\Eiifourpmddomains}}{{{len(defined)}}}",
        rf"\newcommand{{\Eiifourcollapsed}}{{{len(collapsed_controls)}}}",
    ]
    if collapsed_controls:
        first = collapsed_controls[0]
        macros.append(
            rf"\newcommand{{\Eiifourcollapsedomain}}{{{LABEL[first]}}}"
        )
        macros.append(
            rf"\newcommand{{\Eiifourcollapsestep}}"
            rf"{{{payload['control_collapse']['steps'][first]}}}"
        )
    for row in rows:
        tag = TAG[row["domain"]]
        macros.append(rf"\newcommand{{\Eiifour{tag}ctrlpass}}{{{fmt(row['control']['pass8'])}}}")
        macros.append(rf"\newcommand{{\Eiifour{tag}trtpass}}{{{fmt(row['treatment']['pass8'])}}}")
        macros.append(rf"\newcommand{{\Eiifour{tag}ctrlpmd}}{{{fmt(row['control']['pmd'])}}}")
        macros.append(rf"\newcommand{{\Eiifour{tag}trtpmd}}{{{fmt(row['treatment']['pmd'])}}}")
        macros.append(rf"\newcommand{{\Eiifour{tag}pmdgain}}{{{signed(row['pmd_gain'])}}}")
    (out / f"e124_qwen7b_level1_{args.stamp}_macros.tex").write_text(
        "\n".join(macros) + "\n", encoding="utf-8"
    )

    body = ["% Generated by ops/build_paper_e124_qwen7b_level1.py; do not hand edit."]
    for row in rows:
        body.append(
            f"  {LABEL[row['domain']]}"
            f" & {fmt(row['control']['pass8'])} & {fmt(row['control']['pmd'])}"
            f" & {fmt(row['treatment']['pass8'])} & {fmt(row['treatment']['pmd'])}"
            f" & {signed(row['pass8_gain'])} & {signed(row['pmd_gain'])} \\\\"
        )
    body.append(r"  \midrule")
    body.append(
        f"  Mean & {fmt(means['control_pass8'])} & {fmt(means['control_pmd'])}"
        f" & {fmt(means['treatment_pass8'])} & {fmt(means['treatment_pmd'])}"
        f" & {signed(means['pass8_gain'])} & {signed(means['pmd_gain'])} \\\\"
    )
    body.append(r"  \bottomrule")
    (out / f"e124_qwen7b_level1_{args.stamp}_table_body.tex").write_text(
        "\n".join(body) + "\n", encoding="utf-8"
    )

    print(f"wrote e124_qwen7b_level1_{args.stamp}.{{json,_macros.tex,_table_body.tex}}")
    print(f"  domains={len(rows)} cells={2*len(rows)} seeds={seeds} "
          f"terminal_step={terminal_step}")
    print(f"  pass@8  MaxRL {fmt(means['control_pass8'])} -> Re:Max "
          f"{fmt(means['treatment_pass8'])}  ({signed(means['pass8_gain'])})")
    print(f"  PCMD    MaxRL {fmt(means['control_pmd'])} -> Re:Max "
          f"{fmt(means['treatment_pmd'])}  ({signed(means['pmd_gain'])}) "
          f"over {len(defined)} domain(s)")
    for row in rows:
        if row["control_collapsed"]:
            print(f"  control collapse: {row['domain']} at step "
                  f"{row['control']['collapse_step']}; PCMD undefined")
    if pending:
        print(f"  not yet paired: {', '.join(pending)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
