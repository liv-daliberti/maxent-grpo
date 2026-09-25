#!/usr/bin/env python3
"""Correctness and breadth along the level axis, for the Qwen2.5 family.

Two panels over the same cells: ``pass@8`` and PCMD, each averaged over all
five domains, one line per scale, with the hosted deployment for reference.
Both collapse the domain axis instead of the scale axis, which is what makes
the ladder's effect on a whole family readable in a single plot.

The breadth panel carries a support caveat the correctness panel does not, and
it is drawn rather than written.  PCMD exists only where a cell clears the
30-prompt bar; a point whose five domains all clear it is filled, and one
averaging whichever domains remain is hollow.  The macros this writes count
both, so the caption cannot drift from the picture.  The thinned points put the
two smallest scales highest at the hard levels, which tracks the support loss
behind the falling accuracy beside it rather than any gain in breadth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
import paper_style as style  # noqa: E402
from plot_paper_modebench_base_levels_appendix import scale_colors  # noqa: E402

PAYLOAD = ROOT / 'paper/results/mode_diversity_base_grid.json'
FRONTIER = ROOT / 'paper/results/mode_diversity_frontier_points.json'
OUT = ROOT / 'paper/figures/qwen_level_trends'
RECORD = ROOT / 'paper/figures/qwen_level_trends.json'
MACROS = ROOT / 'paper/results/qwen_level_trend_macros.tex'
# The frozen grid is also being scored on each domain's 384-row training split.
# Those rows are unseen by these off-the-shelf checkpoints, so they are simply
# more held-out prompts; pooling them raises how many prompts return the two
# verified responses PCMD needs. Only receipts that exist are pooled, so this
# figure improves as the sweep lands rather than waiting for all of it.
TRAIN_RECEIPTS = ROOT / 'artifacts/modebench_base_grid_trainsplit_20260918/receipts'

# Smallest first. Handing scale_colors exactly this list is what matches the
# main-body scale figure: the ramp is fitted to the log parameter range it is
# given, so a different list would recolour every line.
QWEN = ('05b', 'qwen15b', '3b', '7b', '14b', 'qwen32b', 'qwen72b')
LABELS = {'05b': '0.5B', 'qwen15b': '1.5B', '3b': '3B', '7b': '7B',
          '14b': '14B', 'qwen32b': '32B', 'qwen72b': '72B'}
LEVELS = ('level1', 'level2', 'level3', 'level4', 'level5')
FRONTIER_MODEL = 'GPT-5.6 Sol'
FRONTIER_INK = '#111111'


def _prompt_pmd(prompt):
    """PCMD for one prompt: the share of its verified pairs that differ."""
    keys = [a['canonical_key'] for g in prompt['draws'] for a in g['attempts']
            if a.get('verified') and a.get('canonical_key')]
    n = len(keys)
    if n < 2:
        return None
    counts = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    same = sum(c * (c - 1) for c in counts.values())
    return 1.0 - same / (n * (n - 1))


def train_cells():
    """Per-cell train-split statistics, for whichever receipts have landed."""
    out = {}
    if not TRAIN_RECEIPTS.is_dir():
        return out
    for path in sorted(TRAIN_RECEIPTS.glob('*.json')):
        record = json.loads(path.read_text())
        if record.get('split') != 'train':
            continue
        prompts = record['prompt_results']
        pmds = [v for v in (_prompt_pmd(p) for p in prompts) if v is not None]
        out[(record['model_label'], record['level'], record['domain'])] = {
            'prompts': len(prompts),
            'pass8': record['metrics']['pass8'],
            'defined_prompts': len(pmds),
            'pmd_sum': sum(pmds),
        }
    return out


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def series(cells, models, train=None):
    train = train or {}
    index = {(c['model'], c['level'], c['domain']): c for c in cells}
    # Pool the two splits per cell: pass@8 weights by prompts, PCMD is the
    # unweighted mean over every prompt that defines it in either split.
    for key, extra in train.items():
        cell = index.get(key)
        if cell is None:
            continue
        n_e, n_t = cell['prompts'], extra['prompts']
        pooled = dict(cell)
        pooled['prompts'] = n_e + n_t
        pooled['pass8'] = (cell['pass8'] * n_e + extra['pass8'] * n_t) / (n_e + n_t)
        d_e, d_t = cell['defined_prompts'], extra['defined_prompts']
        if d_e + d_t:
            total = (cell['pmd'] or 0.0) * d_e + extra['pmd_sum']
            pooled['pmd'] = total / (d_e + d_t)
            pooled['defined_prompts'] = d_e + d_t
            pooled['reportable'] = (d_e + d_t) >= 30
        pooled['pooled_splits'] = True
        index[key] = pooled
    domains = sorted({c['domain'] for c in cells})
    out = {}
    for model in models:
        levels = [l for l in LEVELS
                  if any((model, l, d) in index for d in domains)]
        if not levels:
            continue
        rows = [[index[(model, l, d)] for d in domains if (model, l, d) in index]
                for l in levels]
        # PCMD is defined in all but two cells of this family; what varies is
        # whether a cell clears the reporting bar. The mean takes every defined
        # cell so the domain mix cannot move with the level, and 'solid' records
        # whether the whole set behind a point cleared the bar.
        defined = [[c for c in r if c['pmd'] is not None] for r in rows]
        out[model] = {
            'levels': levels,
            'pass8': [_mean(c['pass8'] for c in r) for r in rows],
            'pmd': [_mean(c['pmd'] for c in r) if r else None for r in defined],
            'pmd_solid': [bool(r) and len(r) == len(domains)
                          and all(c['reportable'] for c in r) for r in defined],
            'domains': [len(r) for r in rows],
        }
    return out


def build(grid_payload, frontier_payload):
    qwen_cells = [dict(c, model=c['model_label']) for c in grid_payload['cells']
                  if c['model_label'] in QWEN]
    front_cells = [c for c in frontier_payload['cells']
                   if c['model'] == FRONTIER_MODEL]
    extra = train_cells()
    data = series(qwen_cells, QWEN, train=extra)
    data.update(series(front_cells, (FRONTIER_MODEL,)))
    data['_pooled_cells'] = len(extra)

    colors = dict(scale_colors(QWEN))
    colors[FRONTIER_MODEL] = FRONTIER_INK

    style.apply_rcparams()
    support = {'filled': 0, 'points': 0}
    for name, record in data.items():
        if not isinstance(record, dict) or 'pmd_solid' not in record:
            continue
        for full in record['pmd_solid']:
            support['points'] += 1
            support['filled'] += int(full)

    figure, axes = plt.subplots(1, 2, figsize=(3.41, 1.95), sharex=True)
    for axis, field, label in ((axes[0], 'pass8', r'$\mathtt{pass@8}$'),
                               (axes[1], 'pmd', r'$\mathtt{PCMD}$')):
        for model in tuple(QWEN) + (FRONTIER_MODEL,):
            record = data.get(model)
            if record is None:
                continue
            xs = [LEVELS.index(l) + 1 for l in record['levels']]
            ys = [float('nan') if v is None else v for v in record[field]]
            front = model == FRONTIER_MODEL
            axis.plot(xs, ys, color=colors[model],
                      linewidth=1.5 if front else 1.1,
                      linestyle='--' if front else '-',
                      zorder=3 if front else 2)
            size = 3.0 if front else 2.4
            solid = record['pmd_solid'] if field == 'pmd' else None
            for k, (x, y) in enumerate(zip(xs, ys)):
                if y != y:
                    continue
                full = True if solid is None else solid[k]
                axis.plot([x], [y], marker='o', markersize=size,
                          color=colors[model],
                          markerfacecolor=colors[model] if full else 'white',
                          markeredgecolor=colors[model] if not full else 'white',
                          markeredgewidth=.55 if not full else .45,
                          zorder=4 if front else 3)
        axis.set_xticks(list(range(1, len(LEVELS) + 1)),
                        [str(t) for t in range(1, len(LEVELS) + 1)])
        axis.set_xlim(.8, len(LEVELS) + .2)
        if field == 'pass8':
            axis.set_ylim(0, 1.0)
        # The shared pale blue panel wash, as the bank-balance plate uses it.
        axis.set_facecolor(style.PANEL)
        axis.set_xlabel('ModeBench level', fontsize=style.LABEL_FONT)
        axis.set_ylabel(label, fontsize=style.LABEL_FONT)
        axis.tick_params(labelsize=style.SMALL_FONT)
        style.style_axis(axis)
    data['_support'] = support

    handles = [Line2D([], [], color=colors[m], marker='o', markersize=2.8,
                      linewidth=1.1, label=LABELS[m]) for m in QWEN]
    handles.append(Line2D([], [], color=FRONTIER_INK, marker='o', markersize=3.2,
                          linewidth=1.5, linestyle='--', label=FRONTIER_MODEL))
    # One row: eight entries, so the spacing is tightened rather than wrapped.
    figure.legend(handles=handles, loc='lower center', ncol=len(handles),
                  frameon=False, fontsize=5.2,
                  handletextpad=.18, handlelength=.6, columnspacing=.38,
                  borderaxespad=0, bbox_to_anchor=(.5, .012))
    figure.tight_layout(rect=(0, .13, 1, 1), w_pad=1.1)
    return figure, data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    figure, data = build(json.loads(PAYLOAD.read_text()),
                         json.loads(FRONTIER.read_text()))
    figure.savefig(args.output.with_suffix('.pdf'))
    figure.savefig(args.output.with_suffix('.png'), dpi=300)
    RECORD.write_text(json.dumps(
        {'schema': 'qwen-level-trends-v3', 'metrics': ['pass8', 'pmd'],
         'frontier_model': FRONTIER_MODEL, 'series': data},
        indent=1, sort_keys=True) + '\n')
    # The caption states how many breadth points are fully supported. The grid
    # is still filling, so it spends macros rather than a written number.
    support = data['_support']
    MACROS.write_text(
        '% Generated by ops/plot_paper_qwen_level_trends.py; do not hand edit.\n'
        f'\\newcommand{{\\QLTsupported}}{{{support["filled"]}}}\n'
        f'\\newcommand{{\\QLTpoints}}{{{support["points"]}}}\n'
        f'\\newcommand{{\\QLTunsupported}}{{{support["points"] - support["filled"]}}}\n')
    print(json.dumps({'event': 'built', 'output': str(args.output),
                      'support': support, 'macros': str(MACROS)}))


if __name__ == '__main__':
    main()
