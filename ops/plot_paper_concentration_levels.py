#!/usr/bin/env python3
"""Final minus initial PCMD along the whole construction ladder.

Rows are domain by level, so the question the plate answers is whether training
narrows breadth the same way on constructions the policy never trained on. It is
deliberately a separate renderer from ``plot_paper_concentration_story.py``
rather than a mode of it: the two read different payloads, and a shared drawing
helper would bind one figure's record to the other's source hash.

Three states are drawn, and they mean different things:

* measured -- both checkpoints define PCMD, so the difference exists;
* initial below support -- the untrained checkpoint never returns two correct
  responses to one prompt, so there is no initial breadth to subtract and no
  amount of sampling fixes it;
* not measured -- the cell is outside this plate's measured set.  At present
  that is GRPO above Level 1, which was read at Level 1 only.

The third is a statement about scope and the second is a property of the task.
Printing them the same way would be the one mistake this plate must not make.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'ops') not in sys.path:
    sys.path.insert(0, str(ROOT / 'ops'))

DEFAULT_SOURCE = ROOT / 'paper/results/concentration_across_levels.json'
DEFAULT_OUTPUT = ROOT / 'paper/figures/concentration_levels'
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')
DOMAIN_LABELS = {'graph_coloring': 'Graph', 'countdown': 'Countdown',
                 'python_factors': 'Python', 'mathir': 'MathIR',
                 'pantry_plan': 'PantryPlan'}
LEVELS = ('level1', 'level2', 'level3', 'level4', 'level5')
METHODS = ('drgrpo', 'grpo', 'maxrl')
METHOD_LABELS = {'drgrpo': 'Dr.GRPO', 'grpo': 'GRPO', 'maxrl': 'MaxRL'}
FONT = 7.6
ROW_IN = .285
# Matched-geometry overrides; None keeps this plate's own layout.
HEIGHT_IN = None
XLIM = None
MARKER = 3.8
# Axes rectangle as a canvas fraction; shared with the story plate so the
# two align on a baseline and a top edge when printed side by side.
AXES_BOTTOM = None
AXES_HEIGHT = None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(record: dict, scales: tuple[str, ...] | None,
            combine_levels: bool = False) -> dict:
    """One entry per (domain, level, method), pooled over the drawn scales.

    With ``combine_levels`` the level dimension collapses: every level's seed
    deltas pool into one mark per domain and objective. The spread is then
    across levels *and* seeds rather than seeds alone, which is the honest
    reading of a mark that answers 'what does training do to this domain',
    without claiming the levels are replicates of each other.
    """
    out: dict[tuple[str, str, str], dict] = {}
    for block in record['blocks']:
        if scales and block['scale'] not in scales:
            continue
        key = ((block['domain'], 'all', block['method']) if combine_levels
               else (block['domain'], block['level'], block['method']))
        entry = out.setdefault(key, {'means': [], 'status': set(), 'n': 0,
                                     'provisional': False, 'ci95': [], 'blocks': 0,
                                     'pooled': []})
        entry['status'].add(block['status'])
        if block['status'] == 'measured' and block['summary']['mean'] is not None:
            entry['means'].append(block['summary']['mean'])
            entry['n'] += block['summary']['n']
            # Below the support bar the estimate stands but the precision does
            # not; the plate says so with the paper's hollow convention.
            entry['provisional'] = entry['provisional'] or block.get('provisional', False)
            entry['blocks'] += 1
            entry['pooled'].extend(block['summary'].get('values') or [])
            if block['summary'].get('ci95'):
                entry['ci95'].append(block['summary']['ci95'])
    for entry in out.values():
        entry['mean'] = (sum(entry['means']) / len(entry['means'])
                         if entry['means'] else None)
        # An interval is drawn only when one block backs the mark. Pooling two
        # scales averages their means, and averaging their intervals would
        # describe neither; the pooled plate shows position without precision.
        entry['interval'] = (entry['ci95'][0]
                             if entry['blocks'] == 1 and len(entry['ci95']) == 1
                             else None)
        if combine_levels and len(entry['pooled']) >= 2:
            import statistics as _st
            n = len(entry['pooled'])
            if n not in T95:
                raise RuntimeError(f'no t multiplier tabulated for n={n}')
            half = T95[n] * _st.stdev(entry['pooled']) / (n ** .5)
            entry['mean'] = _st.fmean(entry['pooled'])
            entry['interval'] = [entry['mean'] - half, entry['mean'] + half]
    return out


# Student-t 95% half-widths for n-1 degrees of freedom, as the builder uses.
# Gaps here used to fall back to the normal 1.96, which narrows the interval
# without saying so, so every n in range is tabulated and the rest raises.
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
       9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160,
       15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093,
       21: 2.086, 22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064, 26: 2.060,
       27: 2.056, 28: 2.052, 29: 2.045, 30: 2.045}


def build(source: Path, scales: tuple[str, ...] | None,
          gutter: float = .268,
          levels: tuple[str, ...] = LEVELS, width: float = 3.95,
          combine_levels: bool = False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import paper_style as style

    raw = source.read_bytes()
    record = json.loads(raw)
    if record.get('schema') != 'paper-concentration-across-levels-v1':
        raise ValueError('expected the across-levels concentration payload')
    data = collect(record, scales, combine_levels)
    if combine_levels:
        levels = ('all',)

    drawn = [e['mean'] for e in data.values() if e['mean'] is not None]
    for e in data.values():
        if e.get('interval'):
            drawn.extend(e['interval'])
    if not drawn:
        raise ValueError('no measured block to draw yet')
    # Each side of the window is sized from the data on that side. The positive
    # limit used to be .28 of the negative one, which is a rule keyed on how far
    # the collapse goes: on a plate where most arms narrow, the broadening side
    # got a window proportional to the narrowing, and the one genuine
    # broadening result -- Countdown at the upper levels -- was drawn outside
    # the axis. A rule that selects on magnitude, on a quantity whose magnitude
    # differs by direction, selects on direction.
    values = [value * 100 for value in drawn]
    decade = lambda edge: max(10, -(-int(abs(edge) * 1.06) // 10) * 10)
    low = -decade(min(min(values), 0.0))
    high = decade(max(max(values), 0.0))

    rows = len(DOMAINS) * len(levels)
    height = ROW_IN * rows + 0.86
    row_in = ROW_IN
    if HEIGHT_IN is not None:
        height = HEIGHT_IN
        row_in = (HEIGHT_IN - 0.86) / rows
    if XLIM is not None:
        low, high = XLIM
    colors = {'drgrpo': style.CONTROL, 'grpo': style.METHOD, 'maxrl': style.COMPARATOR}
    markers = {'drgrpo': 'o', 'grpo': 's', 'maxrl': 'D'}
    rc = {'font.family': 'DejaVu Sans', 'font.size': FONT, 'axes.labelsize': FONT,
          'xtick.labelsize': FONT, 'ytick.labelsize': FONT, 'text.color': style.INK,
          'axes.labelcolor': style.INK, 'xtick.color': style.MUTED,
          'ytick.color': style.INK, 'pdf.fonttype': 42, 'ps.fonttype': 42}

    with plt.rc_context(rc):
        fig = plt.figure(figsize=(width, round(height, 3)))
        axes_bottom = .50 / height if AXES_BOTTOM is None else AXES_BOTTOM
        axes_height = (row_in * rows / height if AXES_HEIGHT is None
                       else AXES_HEIGHT)
        ax = fig.add_axes((gutter, axes_bottom, .940 - gutter, axes_height))
        ax.set_xlim(low, high)
        ax.set_ylim(-.60, rows - .40)
        ax.set_xticks((low, 0, high), (f'−{abs(low):g}', '0', f'+{high:g}'))
        ax.set_yticks([])
        ax.axvspan(low, 0, color='#FBEFEF', zorder=0, lw=0)
        ax.axvspan(0, high, color='#EDF6EE', zorder=0, lw=0)
        ax.axvline(0, color=style.MUTED, linewidth=.65, zorder=1)
        for edge in range(1, len(DOMAINS)):
            ax.axhline(rows - 1 - edge * len(levels) + .5,
                       color='#5B6066', linewidth=1.1, zorder=3)

        for index, (domain, level) in enumerate((d, l) for d in DOMAINS for l in levels):
            y = rows - 1 - index
            states = set()
            for slot, method in enumerate(METHODS):
                entry = data.get((domain, level, method))
                offset = .26 - 2 * .26 * slot / (len(METHODS) - 1)
                if entry is None:
                    states.add('not measured')
                    continue
                if entry['mean'] is None:
                    states.add('initial below support'
                               if 'initial_below_support' in entry['status']
                               else 'not measured')
                    continue
                if entry['interval'] is not None:
                    ax.plot([entry['interval'][0] * 100, entry['interval'][1] * 100],
                            [y + offset, y + offset], color=colors[method],
                            linewidth=1.0, zorder=4, solid_capstyle='butt')
                ax.plot(entry['mean'] * 100, y + offset, marker=markers[method],
                        markersize=MARKER, markeredgewidth=.8,
                        markerfacecolor=('none' if entry['provisional']
                                         else colors[method]),
                        markeredgecolor=colors[method],
                        linestyle='none', zorder=5)
            if not any(data.get((domain, level, m), {}).get('mean') is not None
                       for m in METHODS):
                note = ('<2 eligible initial prompts' if 'initial below support' in states
                        else 'not measured')
                ax.text(.5, y, note, transform=ax.get_yaxis_transform(),
                        fontsize=FONT - 1.8, color=style.MUTED, ha='center',
                        va='center', style='italic', zorder=6)
            # With one level per domain the row *is* the domain, so it takes
            # the row label and the rotated side labels are dropped -- five
            # rotated names cannot fit beside five rows.
            ax.text(-.035, y,
                    level.replace('level', 'L') if len(levels) > 1
                    else DOMAIN_LABELS[domain],
                    transform=ax.get_yaxis_transform(), fontsize=FONT - .6,
                    ha='right', va='center', clip_on=False)

        for domain_index, domain in enumerate(DOMAINS if len(levels) > 1 else ()):
            centre = rows - 1 - domain_index * len(levels) - (len(levels) - 1) / 2
            fig.text(.040, axes_bottom + axes_height * (centre + .60) / (rows + .20),
                     DOMAIN_LABELS[domain], fontsize=FONT - .4, fontweight='bold',
                     rotation=90, ha='center', va='center')
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(style.GRID)
        ax.spines['bottom'].set_linewidth(.7)
        ax.tick_params(axis='x', length=2.4, width=.6, pad=2)

        present = [m for m in METHODS
                   if any(k[2] == m and e['mean'] is not None for k, e in data.items())]
        handles = [Line2D([], [], marker=markers[m], linestyle='none', markersize=MARKER,
                          color=colors[m], label=METHOD_LABELS[m]) for m in present]
        handles.append(Line2D([], [], marker='o', linestyle='none', markersize=MARKER,
                              markerfacecolor='none', markeredgecolor=style.MUTED,
                              markeredgewidth=.8, label='few eligible prompts'))
        fig.legend(handles=handles, loc='lower center',
                   bbox_to_anchor=(.630, axes_bottom + axes_height + .006),
                   ncol=len(handles), frameon=False, fontsize=FONT - 2.0,
                   handletextpad=.25, columnspacing=.62, borderaxespad=0)
        fig.text(.630, .150 / height,
                 'Δ PCMD (pp)\n← lower diversity    higher diversity →',
                 fontsize=FONT - 1.4, ha='center', va='center', linespacing=1.55)

        style.apply_domain_typography(fig)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        canvas = fig.bbox
        for text in fig.findobj(match=matplotlib.text.Text):
            if not text.get_visible() or not text.get_text():
                continue
            box = text.get_window_extent(renderer)
            if (box.x0 < canvas.x0 - 1 or box.x1 > canvas.x1 + 1
                    or box.y0 < canvas.y0 - 1 or box.y1 > canvas.y1 + 1):
                raise ValueError(f'text outside canvas: {text.get_text()!r}')

    measured = sum(1 for e in data.values() if e['mean'] is not None)
    metadata = {
        'schema': 'paper-concentration-levels-figure-v1',
        'source': {'path': str(source), 'sha256': hashlib.sha256(raw).hexdigest()},
        'renderer': {'path': str(Path(__file__).resolve().relative_to(ROOT)),
                     'sha256': _sha(Path(__file__))},
        'scales_pooled': list(scales) if scales else 'all',
        'rows': rows, 'cells_drawn': measured,
        'x_limits_percentage_points': [low, high],
        'transfer_levels': record['definition']['transfer_levels'],
    }
    return fig, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--scale', action='append', default=None)
    parser.add_argument('--level', action='append', default=None)
    parser.add_argument('--combine-levels', action='store_true',
                        help='pool every level into one mark per domain')
    parser.add_argument('--height', type=float, default=None)
    parser.add_argument('--xlow', type=float, default=None)
    parser.add_argument('--xhigh', type=float, default=None)
    parser.add_argument('--markersize', type=float, default=3.8)
    parser.add_argument('--axes-bottom', type=float, default=None)
    parser.add_argument('--axes-height', type=float, default=None)
    parser.add_argument('--gutter', type=float, default=.268,
                        help='left fraction reserved for row and domain labels; short row labels need less of it')
    parser.add_argument('--width', type=float, default=3.95,
                        help='canvas width in inches; a wrapped column needs its own')
    args = parser.parse_args()
    global HEIGHT_IN, XLIM, MARKER, AXES_BOTTOM, AXES_HEIGHT
    HEIGHT_IN = args.height
    XLIM = (args.xlow, args.xhigh) if args.xlow is not None else None
    MARKER = args.markersize
    AXES_BOTTOM = args.axes_bottom
    AXES_HEIGHT = args.axes_height
    import matplotlib.pyplot as plt
    fig, metadata = build(args.source, tuple(args.scale) if args.scale else None,
                          args.gutter,
                          tuple(args.level) if args.level else LEVELS, args.width,
                          args.combine_levels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix('.pdf'),
                metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(args.output.with_suffix('.png'), dpi=220)
    plt.close(fig)
    # Both shipped assets are bound by content, not merely named. A bare path
    # records nothing: the PNG is written and included in the release, and a
    # record that lists only the PDF's filename lets either file drift without
    # anything noticing.
    root = Path(__file__).resolve().parents[1]
    metadata['outputs'] = {
        extension: {
            'path': args.output.with_suffix('.' + extension).resolve()
                        .relative_to(root).as_posix(),
            'sha256': hashlib.sha256(
                args.output.with_suffix('.' + extension).read_bytes()).hexdigest(),
        }
        for extension in ('pdf', 'png')
    }
    metadata['source'] = {
        'path': Path(metadata['source']['path']).resolve().relative_to(root).as_posix(),
        'sha256': metadata['source']['sha256'],
    } if isinstance(metadata.get('source'), dict) else metadata.get('source')
    args.output.with_suffix('.json').write_text(
        json.dumps(metadata, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'event': 'built', 'cells_drawn': metadata['cells_drawn'],
                      'rows': metadata['rows']}, sort_keys=True))


if __name__ == '__main__':
    main()
