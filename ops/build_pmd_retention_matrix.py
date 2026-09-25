#!/usr/bin/env python3
"""Per-seed terminal PCMD deltas for the retention comparator matrix.

Panel A pairs each replay arm with the fresh objective it was added to, at every
scale. Panel B differences the 0.5B arms against matched Dr.GRPO, which is the
estimand the plate has always reported. Values come from the PCMD payloads rather
than from any re-measurement: the frozen curve archive for the five factorial
arms, and the comparator extraction for UCPO, sparse RLEP-Dr and the fixed
Semantic-MaxEnt arm, which the curve archive never covered.

A paired difference carries its own support rule. The whole-cell bar of 30
defined prompts protects a *reported PCMD value*; here both sides are measured on
the same prompts and only their difference is read, so the bar is 20. That
matters because the side that falls short is almost always the Dr.GRPO control,
whose definedness is depressed by the very concentration under study --- holding
the difference to the stricter bar would censor exactly the comparisons where
replay separates most. Cells where either side stays under 20 remain gaps.
"""
from __future__ import annotations

import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
CURVES = ROOT / 'paper/results/mode_diversity_curves.json'
COMPARATORS = ROOT / 'paper/results/mode_diversity_comparators_05b.json'
RLEP = ROOT / 'paper/results/mode_diversity_comparators_rlep.json'
#: E126 GAPO, E127 SetPO and the E128 control they are differenced against.
DIVERSITY = ROOT / 'paper/results/mode_diversity_comparators_diversity_05b.json'
OUT = ROOT / 'paper/results/mode_diversity_retention_matrix.json'
MACROS = ROOT / 'paper/results/pmd_retention_macros.tex'

DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')
SCALES = {'qwen05b': (43, 44, 45, 46, 47), 'falcon1b': (55, 56, 57, 58, 59),
          'qwen3b': (70, 71, 72, 73, 74),
          #: Level 2 runs the same Qwen2.5-0.5B seeds on a harder construction.
          'qwen05b_level2': (43, 44, 45, 46, 47)}
#: Defined-prompt bar for a paired difference; see the module docstring for why
#: it is not the whole-cell ``DEFAULT_MIN_DEFINED_PROMPTS`` of 30.
MIN_PAIRED_DEFINED_PROMPTS = 20
#: Each replay arm against the fresh objective it was added to. Re:Max was
#: missing here for no reason but order of construction: the curve archive
#: carries per-seed terminal PCMD for maxrl and replay_maxrl at every scale,
#: and without this panel the Re:Max tables could report breadth only through
#: distinct@8, the endpoint PCMD exists to replace.
PANEL_A = (('replay_drgrpo', 'drgrpo', 'Re:Dr'),
           ('replay_maxrl', 'maxrl', 'Re:Max'))
PANEL_B = (('before_training', 'Before training'),
           ('replay_drgrpo', 'Re:Dr'), ('replay_maxrl', 'Re:Max'),
           ('maxrl', 'MaxRL'), ('grpo', 'GRPO'),
           ('semantic_only', 'Fixed Semantic-MaxEnt'), ('ucpo', 'UCPO'), ('rlep', 'RLEP'),
           ('gapo', 'GAPO'), ('setpo', 'SetPO'))

#: Every panel B row differences against the E78-era Dr.GRPO control except
#: these two. E78's runtime was retired by the 2026-09-04 cleanup and cannot be
#: rebuilt, the cells run on different hardware, and they take the corrected
#: disjoint-draw evaluator, so their control was re-run as E128 rather than
#: inherited. The substitution is disclosed in the output and in the table's
#: own mark; ``control_shift_vs_e78`` in the source measures what it costs.
PANEL_B_CONTROL = {'gapo': 'e128_control', 'setpo': 'e128_control'}


def terminal(points):
    return max(points, key=lambda p: p['step'])


def load() -> tuple[dict, dict]:
    curve = {}
    for c in json.loads(CURVES.read_text())['curves']:
        if c['level'] not in ('level1', 'level2'):
            continue
        point = terminal(c['points'])
        # Level 2 is a separate construction, so it is keyed apart rather than
        # pooled: its scale key carries the level.
        scale_key = c['scale'] if c['level'] == 'level1' else c['scale'] + '_level2'
        curve[(scale_key, c['method'], c['domain'], int(c['seed']))] = (
            point.get('pmd'), int(point.get('defined_prompts') or 0))
        if c['method'] == 'drgrpo':
            # The initial reference is these same runs read at step 0.
            first = min(c['points'], key=lambda p: p['step'])
            curve[(scale_key, 'before_training', c['domain'], int(c['seed']))] = (
                first.get('pmd'), int(first.get('defined_prompts') or 0))
    extra: dict = {}
    for cell in json.loads(COMPARATORS.read_text())['cells']:
        method = cell['method']
        if method == 'semantic':
            # e83 is the arm without replay, which is the comparator the matrix
            # reports; e81/e85 add replay on top and are a different arm.
            method = 'semantic_only' if 'semantic_only' in cell['run'] else 'semantic_replay'
        key = (method, cell['domain'], cell['seed'])
        prior = extra.get(key)
        if prior is None or cell['terminal_step'] > prior['terminal_step']:
            extra[key] = cell
    for cell in json.loads(DIVERSITY.read_text())['cells']:
        extra[(cell['method'], cell['domain'], cell['seed'])] = cell
    for cell in json.loads(RLEP.read_text())['cells']:
        key = ('rlep', cell['domain'], cell['seed'])
        prior = extra.get(key)
        # Provenance, not the metric: the attempt that trained is the one that
        # reached the terminal step.
        if prior is None or cell['terminal_step'] > prior['terminal_step']:
            extra[key] = cell
    return curve, extra


def value(curve, extra, scale, method, domain, seed):
    if (scale, method, domain, seed) in curve:
        return curve[(scale, method, domain, seed)]
    cell = extra.get((method, domain, seed)) if scale == 'qwen05b' else None
    if cell is None:
        return (None, 0)
    return (cell['pmd'], int(cell.get('defined_prompts') or 0))


def _admissible(arm, ctl) -> bool:
    """Both sides define PCMD on at least the paired bar's worth of prompts."""
    return (arm[0] is not None and ctl[0] is not None
            and arm[1] >= MIN_PAIRED_DEFINED_PROMPTS
            and ctl[1] >= MIN_PAIRED_DEFINED_PROMPTS)


#: Two-sided 95% Student-t multipliers by degrees of freedom. A paired cell can
#: fall short of five admissible seeds, and the multiplier has to follow: t is
#: 12.706 at one degree of freedom against 2.776 at four, so pinning 2.776 for
#: every cell printed an interval up to 4.6 times too narrow on exactly the
#: cells that carry the least evidence.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776,
       5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def summarise(per_seed: dict) -> dict:
    values = list(per_seed.values())
    if not values:
        return {'n': 0, 'mean': None, 'student_t_95': None}
    mean = statistics.fmean(values)
    if len(values) < 2:
        return {'n': len(values), 'mean': mean, 'student_t_95': None}
    half = (T95[len(values) - 1] * statistics.stdev(values)
            / len(values) ** 0.5)
    return {'n': len(values), 'mean': mean,
            'student_t_95': [mean - half, mean + half]}


def main() -> None:
    curve, extra = load()
    panel_a: dict = {}
    for scale, seeds in SCALES.items():
        for method, control, label in PANEL_A:
            for domain in DOMAINS:
                per_seed = {}
                for seed in seeds:
                    arm = value(curve, extra, scale, method, domain, seed)
                    ctl = value(curve, extra, scale, control, domain, seed)
                    if _admissible(arm, ctl):
                        per_seed[str(seed)] = arm[0] - ctl[0]
                panel_a.setdefault(scale, {}).setdefault(label, {})[domain] = {
                    'per_seed': per_seed, **summarise(per_seed)}
    panel_b: dict = {}
    for method, label in PANEL_B:
        for domain in DOMAINS:
            per_seed = {}
            for seed in SCALES['qwen05b']:
                arm = value(curve, extra, 'qwen05b', method, domain, seed)
                ctl = value(curve, extra, 'qwen05b',
                            PANEL_B_CONTROL.get(method, 'drgrpo'), domain, seed)
                if _admissible(arm, ctl):
                    per_seed[str(seed)] = arm[0] - ctl[0]
            panel_b.setdefault(label, {})[domain] = {
                'per_seed': per_seed, **summarise(per_seed)}
    OUT.write_text(json.dumps({
        'schema': 'paper-pmd-retention-matrix-v1',
        'estimands': {
            'panel_a': 'replay arm minus the fresh objective it was added to, terminal PCMD',
            'panel_b': 'method minus matched Dr.GRPO, terminal PCMD, Qwen2.5-0.5B',
        },
        'panel_b_controls': {
            'default': 'drgrpo (E78)',
            **{m: f'{c} (E128)' for m, c in PANEL_B_CONTROL.items()},
        },
        'control_shift_vs_e78': json.loads(
            DIVERSITY.read_text())['control_shift_vs_e78'],
        'support': (f'a seed contributes only where both sides define PCMD on at '
                    f'least {MIN_PAIRED_DEFINED_PROMPTS} prompts'),
        'min_paired_defined_prompts': MIN_PAIRED_DEFINED_PROMPTS,
        'uncertainty': 'paired unadjusted two-sided 95% Student-t over seeds',
        'sources': {p.name: p.stat().st_size
                    for p in (CURVES, COMPARATORS, RLEP, DIVERSITY)},
        'panel_a': panel_a, 'panel_b': panel_b,
    }, indent=1) + '\n')
    # Counts move as the grid fills, so the prose quotes them through macros.
    # The body claim these macros feed is about Level 1 across the three scales,
    # so the population is fixed to that and does not widen when a panel is
    # added. Level 2 is a different construction and gets its own pair.
    def tally(scales):
        measurable = above = 0
        for scale in scales:
            for cells in panel_a.get(scale, {}).values():
                for cell in cells.values():
                    if cell['mean'] is None:
                        continue
                    measurable += 1
                    above += cell['mean'] > 1e-9
        return measurable, above

    level1 = [s for s in SCALES if not s.endswith('_level2')]
    measurable, above = tally(level1)
    l2_measurable, l2_above = tally([s for s in SCALES if s.endswith('_level2')])

    # Per-arm counts too: a sentence about one replay arm must not quote the
    # pooled figure, and hard-coding it is how the appendix came to claim 13 of
    # 14 for Re:Dr when every measurable Re:Dr cell was above its control.
    def arm_tally(label):
        measurable = above = 0
        for scale in level1:
            for domain, cell in panel_a[scale].get(label, {}).items():
                if cell['mean'] is None:
                    continue
                measurable += 1
                above += cell['mean'] > 1e-9
        return measurable, above

    dr_measurable, dr_above = arm_tally('Re:Dr')
    max_measurable, max_above = arm_tally('Re:Max')
    MACROS.write_text('\n'.join((
        '% Generated by build_pmd_retention_matrix.py; do not hand edit.',
        f'\\newcommand{{\\PMDblocks}}{{{measurable}}}',
        f'\\newcommand{{\\PMDabove}}{{{above}}}',
        f'\\newcommand{{\\PMDblocksLevelTwo}}{{{l2_measurable}}}',
        f'\\newcommand{{\\PMDaboveLevelTwo}}{{{l2_above}}}',
        f'\\newcommand{{\\PMDblocksReDr}}{{{dr_measurable}}}',
        f'\\newcommand{{\\PMDaboveReDr}}{{{dr_above}}}',
        f'\\newcommand{{\\PMDblocksReMax}}{{{max_measurable}}}',
        f'\\newcommand{{\\PMDaboveReMax}}{{{max_above}}}',
    )) + '\n')
    print(json.dumps({'event': 'built', 'output': str(OUT),
                      'measurable': measurable, 'above': above,
                      'level2_measurable': l2_measurable,
                      'level2_above': l2_above}))


if __name__ == '__main__':
    main()
