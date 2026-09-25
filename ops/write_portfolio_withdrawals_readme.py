#!/usr/bin/env python3
"""Regenerate the withdrawal artifact's README from its own bound records."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/modebench_portfolio_withdrawals_20260917'
CONTROLS = ROOT / 'artifacts/modebench_portfolio_withdrawal_controls_20260917'
ORDER = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')


def points(value):
    return f'{value * 100:.2f}'


def main():
    inputs = json.loads((BASE / 'inputs.json').read_text())
    results = json.loads((BASE / 'results.json').read_text())
    audit = json.loads((BASE / 'independent_audit.json').read_text())
    controls = json.loads((CONTROLS / 'results.json').read_text())
    head, protocol = results['headline'], results['protocol']
    checks, correlation = audit['checks'], results['pmd_agreement']['correlation']
    rows = []
    for domain in ORDER:
        best = head['per_domain'][domain]['4']['best']
        distinct = head['per_domain'][domain]['distinct4']
        rows.append(f"| {domain} | {inputs['withdrawal'][domain]} | "
                    f"{points(best['estimate'])} [{points(best['ci95'][0])}, {points(best['ci95'][1])}] | "
                    f"{distinct['estimate']:.3f} |")
    text = f"""# Portfolio survival when a task loses an option

Five-domain extension of the PantryPlan adaptation experiment
(`artifacts/modebench_inference_followups_20260911/pantry`). It adds no training
and no generation: it scores outcome keys the completed factorial already
produced, at the terminal endpoints the paper's PCMD curves report.

## Question

A portfolio of verified answers exists. The task then loses exactly one option.
Is any saved answer still valid?

| Domain | Withdrawal | Survival gain at four verified draws (pts, best arm) | Expected distinct outcomes at four draws (Re:Dr) |
| --- | --- | --- | --- |
{chr(10).join(rows)}

## Protocol

* Options are enumerated from the task specification in a fixed order, never
  from a generated answer.
* A withdrawal only removes answers from the original verified support, so that
  support certifies the revised task: feasible exactly when some original answer
  survives. Enumerating it reproduces the published certified mode count on all
  {inputs['certified_counts_reproduced']:,} prompt rows.
* Of {protocol['options']:,} options, {protocol['feasible_options']:,} leave the task feasible and
  {protocol['binding_options']:,} of those remove at least one answer. Both sets are reported.
* Survival is decided from the outcome key for Graph, Countdown and Python, so a
  verified answer outside the generator's census is still scored. MathIR and
  Pantry are decided by the census; no verified outcome of theirs falls outside
  it.
* Raw survival moves with correctness. Holding the verified-draw count fixed at
  1, 2 or 4 -- an exact expectation over subsets of the saved correct draws --
  removes that. Both, and expected distinct outcomes at the same budgets, are
  reported.

## Files

| File | Contents |
| --- | --- |
| `inputs.json` | Frozen protocol: prompt identity across all {inputs['cells']} cells, code hashes, budgets, bootstrap, design disclosure |
| `table.json.gz` | Per-prompt certified support and every option's survivors |
| `results.json` | Per-cell summaries, seed-paired contrasts, pooled bootstrap intervals, PCMD agreement |
| `independent_audit.json` | Re-derived supports, exhaustive subset check, key decidability, endpoint parity |

## Reproduction

```
ops/prepare_portfolio_withdrawals.py        # freeze prompts, supports, options
ops/analyze_portfolio_withdrawals.py        # score saved outcomes
ops/audit_portfolio_withdrawals.py          # independent re-derivation
ops/analyze_portfolio_withdrawal_controls.py  # decoding-grid baseline
ops/write_portfolio_withdrawals_readme.py   # this file
ops/build_paper_portfolio_withdrawals.py    # appendix, macros, paper record
```

## Findings

* Raw survival improves in {head['cells/binding/raw']['positive']} of {head['cells/binding/raw']['cells']} domain, cohort and arm comparisons.
* With the verified-draw count held fixed: {head['cells/binding/1']['positive']} improve at one draw,
  {head['cells/binding/2']['positive']} at two, {head['cells/binding/4']['positive']} of {head['cells/binding/4']['cells']} at four.
* {', '.join(head['broad_domains'])} are the domains whose
  portfolios are measurably broader at a fixed budget, and the ones whose
  survival gains are material. MathIR's portfolios are not broader at matched
  budget ({head['per_domain']['mathir']['distinct4']['estimate']:.3f} distinct outcomes), matching its near-zero PCMD gain.
* PCMD gain predicts the survival gain across {correlation['n']} cells
  (Pearson {correlation['pearson']:.2f}, Spearman {correlation['spearman']:.2f}).
* Two simultaneous withdrawals leave far less of the support standing, and the
  gain persists: {head['cells/pair/4']['positive']} of {head['cells/pair/4']['cells']} comparisons improve at four verified draws,
  against {head['cells/binding/4']['positive']} for one withdrawal. It grows in Graph and Python and
  compresses where supports are smallest.
* No decoding setting recovers what the control lacks. Over the whole E72 grid
  its survival moves by at most
  {max(c['control_grid_range']['4'][1] - c['control_grid_range']['4'][0] for c in controls['contrasts'])*100:.1f} points, while the replay arm at one fixed
  setting leads the control's best setting by
  {min(c['gap_vs_best']['4']['estimate'] for c in controls['contrasts'])*100:.1f}--{max(c['gap_vs_best']['4']['estimate'] for c in controls['contrasts'])*100:.1f} points in all five domains.

## Scope

A withdrawal is a single exclusion, not an arbitrary revision. Survival measures
whether a saved answer is still valid; recovery after a change -- extra calls,
their cost and the diversity-prompt control -- is measured in PantryPlan alone,
because that control needs new generation. Intervals resample prompts, not
training seeds.

## Design disclosure

Fixed before any arm was compared:
{'; '.join(protocol['design_disclosure']['fixed_before_any_contrast'])}.
Chosen with the raw contrast already visible:
{'; '.join(protocol['design_disclosure']['chosen_with_the_raw_contrast_visible'])}.
{protocol['design_disclosure']['mitigation']}

## Audit

{checks['independent_support']['checked']} supports re-derived by an independent route;
{checks['exhaustive_budget']['comparisons']:,} fixed-budget expectations matched against exhaustive subset
enumeration (max difference {checks['exhaustive_budget']['max_absolute_difference']:.1e});
{checks['key_membership']['keys']:,} verified outcomes checked for decidability, {checks['key_membership']['undecidable']} undecidable;
endpoint parity {checks['endpoint_parity']['cells']}/{checks['endpoint_parity']['curve_cells']} cells.
"""
    (BASE / 'README.md').write_text(text)
    print(f'Withdrawal README written: {len(text)} characters.')


if __name__ == '__main__':
    main()
