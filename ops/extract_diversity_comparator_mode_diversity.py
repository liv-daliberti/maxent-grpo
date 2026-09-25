#!/usr/bin/env python3
"""Terminal PCMD for the E126/E127/E128 diversity comparators.

The curve archive never covered these cohorts and the earlier comparator
extraction predates them, so their PCMD is read here directly from each cell's
terminal mode-coverage draws, with the same estimator every other cell uses
(``ops/mode_diversity.py``, pooled aggregation over a prompt's draws).

Three cohorts are emitted together because they are one paired set. ``gapo``
and ``setpo`` are the arms; ``e128_control`` is the matched Dr.GRPO control
they are differenced against, which is *not* the E78 control the older
comparators use. E78's runtime was retired and cannot be rebuilt, these cells
run on different hardware, and they take the corrected disjoint-draw
evaluator --- so the control had to be re-run rather than inherited. The
measured consequence is small and is reported alongside the arms: see
``control_shift_vs_e78`` in the output.

Absolute PCMD and the defined-prompt count are emitted per cell. The
paired-difference support bar is applied downstream by
``ops/build_pmd_retention_matrix.py``, which uses 20 rather than the whole-cell
30, so this extraction deliberately does not pre-filter on reportability.
"""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))

from mode_diversity import POOLED, cell_summary  # noqa: E402

CURVES = ROOT / 'paper/results/mode_diversity_curves.json'
OUT = ROOT / 'paper/results/mode_diversity_comparators_diversity_05b.json'
SCHEMA = 'paper-mode-diversity-comparators-v1'

#: ledger -> method key used by the retention matrix.
COHORTS = {
    'e126': ('e126_gapo_05b_jobs.json', 'gapo'),
    'e127': ('e127_setpo_05b_jobs.json', 'setpo'),
    'e128': ('e128_matched_control_05b_jobs.json', 'e128_control'),
}
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')
SEEDS = (43, 44, 45, 46, 47)


def terminal_draws(run_dir: str) -> tuple[int, list[dict]]:
    """Return the terminal step and its independently seeded sampled draws."""

    complete = Path(run_dir) / 'TRAINING_COMPLETE.json'
    payload = json.loads(complete.read_text(encoding='utf-8'))
    attempt = Path(payload['terminal_attempt'])
    records = [
        json.loads(line)
        for line in (attempt / 'eval_mode_coverage_draws.jsonl').open(encoding='utf-8')
    ]
    step = max(record['step'] for record in records)
    # draw_index None is the deterministic greedy trace, which is a different
    # evaluation kind and never contributes to a sampled breadth estimate.
    sampled = [
        record for record in records
        if record['step'] == step and record.get('draw_index') is not None
    ]
    return step, sampled


def cell_pcmd(run_dir: str) -> dict:
    step, sampled = terminal_draws(run_dir)
    by_prompt: dict[int, list[dict]] = {}
    for record in sampled:
        for prompt in record['prompts']:
            attempts = [
                {'canonical_key': key, 'verified': float(reward) > 0}
                for key, reward in zip(prompt['answer_keys'], prompt['rewards'])
            ]
            by_prompt.setdefault(int(prompt['prompt_index']), []).append(
                {'attempts': attempts}
            )
    summary = cell_summary(
        [{'draws': draws} for draws in by_prompt.values()], POOLED
    )
    return {
        'terminal_step': step,
        'draws': len(sampled),
        'prompts': summary['prompts'],
        'defined_prompts': summary['defined_prompts'],
        'pmd': summary['d_mode'],
        'reportable': summary['reportable'],
        'pass8': statistics.fmean(
            record['metrics']['any_correct_at_k'] for record in sampled
        ),
    }


def e78_control_pcmd() -> dict[tuple[str, int], float]:
    """Published E78 control terminal PCMD, for the control-shift check."""

    out: dict[tuple[str, int], float] = {}
    for curve in json.loads(CURVES.read_text(encoding='utf-8'))['curves']:
        if (curve['scale'] != 'qwen05b' or curve['level'] != 'level1'
                or curve['method'] != 'drgrpo' or int(curve['seed']) not in SEEDS):
            continue
        point = max(curve['points'], key=lambda p: p['step'])
        if point.get('reportable'):
            out[(curve['domain'], int(curve['seed']))] = point['pmd']
    return out


def main() -> int:
    cells: list[dict] = []
    control: dict[tuple[str, int], float] = {}
    for cohort, (ledger_name, method) in sorted(COHORTS.items()):
        ledger = json.loads(
            (ROOT / 'var/artifacts' / ledger_name).read_text(encoding='utf-8')
        )
        if not ledger.get('released'):
            raise SystemExit(f'{cohort} ledger is not released')
        for run in ledger['runs']:
            domain, seed = str(run['domain']), int(run['seed'])
            record = cell_pcmd(run['run_dir'])
            cells.append({
                'scale': 'qwen05b', 'method': method, 'domain': domain,
                'seed': seed, 'experiment': cohort,
                'run': Path(run['run_dir']).name, **record,
            })
            if method == 'e128_control' and record['pmd'] is not None:
                control[(domain, seed)] = record['pmd']

    expected = len(COHORTS) * len(DOMAINS) * len(SEEDS)
    if len(cells) != expected:
        raise SystemExit(f'expected {expected} cells, built {len(cells)}')

    # The one number a reader needs to accept these rows beside the others:
    # how far the re-run control sits from the published one it replaces.
    e78 = e78_control_pcmd()
    shifts = [control[key] - e78[key] for key in sorted(control) if key in e78]
    check = {
        'description': (
            'E128 matched control minus the published E78 Dr.GRPO control, '
            'terminal PCMD, paired by domain and seed'
        ),
        'paired_cells': len(shifts),
        'mean': statistics.fmean(shifts) if shifts else None,
        'min': min(shifts) if shifts else None,
        'max': max(shifts) if shifts else None,
    }

    OUT.write_text(json.dumps({
        'schema': SCHEMA,
        'description': (
            'terminal PCMD for the E126 GAPO, E127 SetPO and E128 matched '
            'control cells, differenced downstream against e128_control'
        ),
        'estimator': 'pooled per-prompt PCMD over verified responses, ops/mode_diversity.py',
        'control_shift_vs_e78': check,
        'cells': cells,
    }, indent=1) + '\n', encoding='utf-8')

    print(f'[pcmd] {len(cells)} cells -> {OUT}')
    if shifts:
        print(f'[pcmd] control shift vs E78: {check["mean"]:+.4f} '
              f'over {len(shifts)} paired cells '
              f'({check["min"]:+.3f} to {check["max"]:+.3f})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
