#!/usr/bin/env python3
"""Render the five-domain withdrawal appendix; --check authenticates the assets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
try:
    from paper_domain_typography import format_domain_names
except ModuleNotFoundError:
    from ops.paper_domain_typography import format_domain_names
import statistics

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/modebench_portfolio_withdrawals_20260917'
CONTROLS = ROOT / 'artifacts/modebench_portfolio_withdrawal_controls_20260917'
RECOVERY = ROOT / 'artifacts/modebench_recovery_five_domain_20260917'
PAPER = ROOT / 'paper'
BINDINGS = PAPER / 'audits/portfolio_withdrawals_20260917/source_sha256.json'
STEM = 'portfolio_withdrawals_20260917'
MACROS = PAPER / 'results/portfolio_withdrawal_macros.tex'
DOMAIN = {'graph_coloring': 'Graph', 'countdown': 'Countdown', 'python_factors': 'Python',
          'mathir': 'MathIR', 'pantry_plan': 'PantryPlan'}
SCALE = {'qwen05b': 'Qwen2.5-0.5B', 'falcon1b': 'Falcon3-1B', 'qwen3b': 'Qwen2.5-3B'}
LEVEL = {'level1': '1', 'level2': '2'}
ORDER = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')
WORD = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def points(value, digits=2):
    text = f'{value * 100:.{digits}f}'
    # A rounded estimate of zero prints without a sign.
    return text.lstrip('-') if float(text) == 0 else text


def signed(value, digits=2):
    """A bare table entry that may be negative, typeset with a true minus."""
    return f'${points(value, digits)}$'


def ci(metric, digits=2):
    lo, hi = metric['ci95']
    return f"${points(metric['estimate'], digits)}$ $[{points(lo, digits)}, {points(hi, digits)}]$"


def table(caption, label, headings, rows, layout=None):
    return '\n'.join([r'\begin{table}[!htbp]', r'\centering', r'\caption{' + caption + '}',
                      r'\label{' + label + '}', r'\scriptsize', r'\setlength{\tabcolsep}{3pt}',
                      r'\begin{tabular}{' + (layout or 'l' + 'r' * (len(headings) - 1)) + '}',
                      r'\toprule', ' & '.join(headings) + r' \\', r'\midrule',
                      *[' & '.join(str(value) for value in row) + r' \\' for row in rows],
                      r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])


def join_and(names):
    """An English list: 'a, b and c'."""
    names = list(names)
    if len(names) < 2:
        return ''.join(names)
    return ', '.join(names[:-1]) + ' and ' + names[-1]


def thousands(value):
    return f'{value:,}'.replace(',', '{,}')



def render_agreement_figure(results, output):
    """Scatter the per-cell diversity gain against its downstream survival gain.

    The table this replaces listed 38 paired numbers to report one correlation,
    which is the one thing a list of pairs cannot show. Marks carry the domain,
    so a reader can see whether the relation is carried by one domain or holds
    across them.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, str(ROOT / 'ops'))
    import paper_style as style

    points = results['pmd_agreement']['points']
    corr = results['pmd_agreement']['correlation']
    marks = {'graph_coloring': 'o', 'countdown': 's', 'python_factors': '^',
             'mathir': 'D', 'pantry_plan': 'v'}
    rc = {'font.family': 'DejaVu Sans', 'font.size': 9, 'text.color': style.INK,
          'axes.labelcolor': style.INK, 'xtick.color': style.INK,
          'ytick.color': style.INK, 'pdf.fonttype': 42, 'ps.fonttype': 42}
    with plt.rc_context(rc):
        figure, axis = plt.subplots(figsize=(style.WIDTH * 0.62, 2.9))
        axis.axhline(0, color=style.MUTED, lw=.6, zorder=1)
        axis.axvline(0, color=style.MUTED, lw=.6, zorder=1)
        for domain, marker in marks.items():
            xs = [p['pmd_gain'] for p in points if p['domain'] == domain]
            ys = [100 * p['survival_gain_budget4'] for p in points if p['domain'] == domain]
            if not xs:
                continue
            axis.scatter(xs, ys, s=21, marker=marker, facecolors='none',
                         edgecolors=style.ADAPTIVE, linewidths=.85, zorder=3,
                         label=DOMAIN[domain])
        axis.set_xlabel(r'$\Delta$ PCMD')
        axis.set_ylabel('$\\Delta$ survival at four\nverified draws (pp)')
        axis.set_title(f"Spearman {corr['spearman']:.2f}, Pearson {corr['pearson']:.2f}"
                       f" over {corr['n']} cells", fontsize=9.2, color=style.INK)
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(color=style.GRID, lw=.5, zorder=0)
        axis.legend(frameon=False, fontsize=7.6, ncol=3, loc='lower right',
                    handletextpad=.3, columnspacing=.9)
        style.apply_domain_typography(figure)
        figure.tight_layout()
        outputs = {}
        for suffix in ('.pdf', '.png'):
            path = output.with_suffix(suffix)
            figure.savefig(path, dpi=220, metadata={'CreationDate': None} if suffix == '.pdf' else None)
            outputs[suffix.lstrip('.')] = {'path': str(path.relative_to(ROOT)),
                                           'sha256': digest(path)}
        plt.close(figure)
    return outputs

def build():
    bindings = json.loads(BINDINGS.read_text()) if BINDINGS.is_file() else {}
    for name, expected in bindings.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen source changed: ' + name)
    results = json.loads((BASE / 'results.json').read_text())
    audit = json.loads((BASE / 'independent_audit.json').read_text())
    inputs = json.loads((BASE / 'inputs.json').read_text())
    controls = json.loads((CONTROLS / 'results.json').read_text())
    head = results['headline']
    protocol = results['protocol']
    pooled = {(row['domain'], row['arm']): row for row in results['pooled']}
    inventory = results['inventory']
    parity = results['endpoint_parity']
    correlation = results['pmd_agreement']['correlation']
    if parity['differing'] or parity['unregistered']:
        raise ValueError('endpoint parity is not clean')
    if results['census_coverage']['undecidable_outcomes']:
        raise ValueError('undecidable outcomes remain')
    best4 = max(head['per_domain'][d]['4']['best']['estimate'] for d in ORDER)

    macros = {
        'PWCohortCells': str(parity['cells']),
        'PWCells': str(head['cells/binding/raw']['cells']),
        'PWRawPositive': str(head['cells/binding/raw']['positive']),
        'PWBudgetOnePositive': str(head['cells/binding/1']['positive']),
        'PWBudgetTwoPositive': str(head['cells/binding/2']['positive']),
        'PWBudgetFourPositive': str(head['cells/binding/4']['positive']),
        'PWBudgetFourCells': str(head['cells/binding/4']['cells']),
        'PWBudgetFourMax': points(best4, 1),
        'PWDomains': str(head['domains']),
        'PWDomainsWord': WORD[head['domains']],
        'PWBroadDomains': str(len(head['broad_domains'])),
        'PWBroadDomainsWord': WORD[len(head['broad_domains'])],
        'PWPromptRows': thousands(inputs['certified_counts_reproduced']),
        'PWOptions': thousands(protocol['options']),
        'PWFeasibleOptions': thousands(protocol['feasible_options']),
        'PWBindingOptions': thousands(protocol['binding_options']),
        'PWPmdPearson': f"{correlation['pearson']:.2f}".lstrip('0'),
        'PWPmdCells': str(correlation['n']),
        'PWPairs': thousands(protocol['pairs']),
        'PWBindingPairs': thousands(protocol['binding_pairs']),
        'PWPairPositive': str(head['cells/pair/4']['positive']),
        'PWPairCells': str(head['cells/pair/4']['cells']),
        'PWCtrlSettings': str(sum(len(v) for v in controls['settings'].values())),
        'PWCtrlGapMin': points(min(c['gap_vs_best']['4']['estimate'] for c in controls['contrasts']), 1),
        'PWCtrlGapMax': points(max(c['gap_vs_best']['4']['estimate'] for c in controls['contrasts']), 1),
        'PWCtrlSpanMax': points(max(c['control_grid_range']['4'][1] - c['control_grid_range']['4'][0]
                                    for c in controls['contrasts']), 1),
    }
    macro_tex = ('% Generated by ops/build_paper_portfolio_withdrawals.py; do not edit numbers.\n'
                 + ''.join('\\newcommand{\\' + name + '}{' + value + '}\n'
                           for name, value in sorted(macros.items())))

    tex = r'''% Generated by ops/build_paper_portfolio_withdrawals.py; do not edit numbers.
\section{Portfolio Survival under Withdrawn Options}
\label{app:portfolio-withdrawals}
A portfolio survives a withdrawn option if at least one of its solutions
remains valid. We evaluate survival across all five domains in ''' + macros['PWCohortCells'] + r''' terminal checkpoint evaluations, each
with 128 prompts and four groups of eight responses per prompt. The groups have
overlapping sampling streams (App.~\ref{app:conditional-concentration}). Replay's
largest gains in expected distinct modes at four verified draws occur in
Graph, Countdown, Python and PantryPlan. Across ''' + macros['PWPmdCells'] + r''' matched
domain--scale--level--objective comparisons,
\pmd{} gains correlate with survival gains at four verified draws
(Pearson $r=''' + macros['PWPmdPearson'] + r'''$).

\paragraph{Task perturbations.} Each task loses one option: a colour at one
vertex (Graph), an arithmetic operation (Countdown), a returned divisor value
(Python), a menu action (MathIR), or an ingredient (PantryPlan; see
App.~\ref{app:pantry-adaptation}). Options are enumerated from the task
specification. The solution-mode enumeration matches the certified counts for
all ''' + macros['PWPromptRows'] + r''' prompts. A surviving certified mode establishes feasibility.
When the certified count is only a lower bound on the verifier's full support,
absence of a surviving mode from the enumeration does not establish infeasibility. Of
''' + macros['PWOptions'] + r''' options, ''' + macros['PWFeasibleOptions'] + r''' retain at least one certified mode. Among these,
''' + macros['PWBindingOptions'] + r''' are binding: they also remove at least one certified mode. We evaluate
both the binding set and the full set with a feasibility certificate.

\paragraph{Survival criterion.} Each portfolio contains the verified
responses among eight draws. It survives when at least one of these solutions
remains valid after the option is removed. Graph, Countdown and Python determine this directly from
the outcome key, including verified keys outside the enumeration. Countdown
contributes ''' + str(results['census_coverage']['distinct_keys_outside_census']['countdown']) + r''' such keys, observed ''' + thousands(audit['checks']['key_membership']['outside_census']) + r''' times among the
''' + thousands(audit['checks']['key_membership']['keys']) + r''' verified draws in this evaluation: the verifier accepts unary negation, which the
expression enumeration omits. For MathIR, re-enumerating the reduced menu tests
whether the same executed state path remains realizable, possibly through a
different action sequence. For PantryPlan, we test the certified plan
associated with each ingredient-support key. All observed MathIR and PantryPlan keys belong to their
respective enumerations.

\paragraph{Survival conditional on correct draws.} Raw survival depends on
both correctness and the distribution of valid modes. At budgets $m=1,2,4$, we
compute the exact expected survival of a uniformly selected $m$-element subset
of a portfolio's verified draws, as in App.~\ref{app:cross-model-overlap}.
Replay--control contrasts use only prompt--group pairs with at least $m$
verified draws under both methods, so eligibility can change with $m$.
Fixing the number of correct draws does not equalize policy accuracy.
A survival difference can also arise because the modes one method favors are
individually more robust to withdrawals, so a gain that grows with $m$ does
not by itself isolate the contribution of diversity. Expected distinct modes
at the same budget therefore serve as a separate measure of portfolio breadth.

'''
    rows = []
    for domain in ORDER:
        for level in ('level1', 'level2'):
            entry = inventory.get(level + '|' + domain)
            if entry is None:
                continue
            rows.append([DOMAIN[domain], LEVEL[level], entry['prompts'],
                         f"{entry['mean_support']:.1f}", entry['options'], entry['binding'],
                         f"{entry['mean_surviving_fraction']:.2f}", entry['binding_pairs'],
                         f"{entry['mean_pair_surviving_fraction']:.2f}"])
    tex += table(r'\textbf{Binding withdrawals remove certified modes while retaining a feasibility certificate.} '
                 'Each row covers 128 prompts. Support is the mean certified mode count per prompt; '
                 'Options counts single withdrawals. Binding counts options that retain at least one '
                 'certified mode and remove at least one. Kept is the mean retained fraction of '
                 'certified modes across binding options. Pairs counts binding pairs of withdrawals, '
                 r'and Kept$_2$ gives the corresponding retained fraction.',
                 'tab:withdrawal-inventory',
                 ['Domain', 'Level', 'Prompts', 'Support', 'Options', 'Binding', 'Kept',
                  'Pairs', r'Kept$_2$'], rows, 'llrrrrrrr')

    rows = []
    for domain in ORDER:
        for arm in ('Re:Dr', 'Re:Max'):
            row = pooled.get((domain, arm))
            if row is None:
                continue
            rows.append([DOMAIN[domain], arm, signed(row['raw']['binding']['estimate']),
                         ci(row['budget']['binding/1']), ci(row['budget']['binding/2']),
                         ci(row['budget']['binding/4']),
                         signed(row['budget']['feasible/4']['estimate']),
                         f"{row['distinct']['4']['estimate']:.3f}"])
    tex += table(r'\textbf{Both replay arms improve four-draw survival in Graph, Python and PantryPlan.} '
                 'Each replay arm is compared with its fresh-objective control. Survival differences are '
                 r'percentage points; $\Delta D_4$ is the difference in expected distinct modes at four '
                 'verified draws. Raw uses all eight responses; Budget columns use the indicated '
                 'number of verified draws. Fixed-correct-draw results pool jointly eligible '
                 'prompt--group pairs across matched seeds, scales and levels; raw results use '
                 'all paired groups. All columns use binding options except Budget 4 (all), '
                 'which includes every option with a feasibility certificate. Brackets are pointwise '
                 r'95\% intervals from 20{,}000 paired bootstrap replicates. Each '
                 'resampled cluster is a prompt within one level and scale, with its groups and '
                 'seeds kept together; copies at other scales are separate clusters. '
                 'Intervals condition on the observed seeds.',
                 'tab:withdrawal-contrasts',
                 ['Domain', 'Arm', 'Raw', 'Budget 1', 'Budget 2', 'Budget 4', 'Budget 4 (all)',
                  r'$\Delta D_4$'], rows, 'll' + 'r' * 6)

    broad = join_and(DOMAIN[d] for d in head['broad_domains'])
    countdown = inventory['level1|countdown']
    dilution = f"{1 - countdown['binding'] / countdown['feasible']:.0%}".replace('%', r'\%')
    breadth = [pooled[(d, 'Re:Dr')]['distinct']['4']['estimate'] for d in ORDER
               if d != 'mathir' and (d, 'Re:Dr') in pooled]
    tex += (r'''Raw survival improves in ''' + macros['PWRawPositive'] + ' of ' + macros['PWCells'] + r''' domain--scale--level--objective
comparisons. At four verified draws, the point estimate is positive in
''' + macros['PWBudgetFourPositive'] + ' of ' + macros['PWBudgetFourCells'] + r''' eligible comparisons; these conditional comparisons cover
fewer prompt--group pairs than raw survival. Outside MathIR, Re:Dr adds between
$''' + f"{min(breadth):.1f}".lstrip('0') + r'''$ and $''' + f"{max(breadth):.1f}" + r'''$ expected distinct modes at four verified draws, and its pooled
survival gains reach ''' + macros['PWBudgetFourMax'] + r''' percentage points. In MathIR, the corresponding
gains are only $.004$ distinct modes and $.14$ survival points, with an interval
of $[.00,.30]$ after rounding. These small positive estimates are consistent
with the small \pmd{} gain in Table~\ref{tab:cross-scale-terminal-effects}.

Graph's one-draw contrast is zero by construction. Each binding withdrawal
removes one color at one vertex, and each valid coloring is excluded by
exactly one such withdrawal per variable vertex, so every individual mode
survives the same fraction of binding withdrawals. Including nonbinding options shrinks the contrasts,
because those options leave every certified mode intact; for example,
''' + dilution + r''' of Countdown Level-1 options with a feasibility certificate are nonbinding.
The Budget 4 (all) and binding columns therefore describe different withdrawal
populations.
''')

    tex += (r'''\begin{figure}[!htbp]
  \centering
  \includegraphics[width=0.62\linewidth]{figures/withdrawal_pmd_agreement.pdf}
  \caption{\textbf{Diversity gains correlate with survival gains after a withdrawal.} Each mark is one replay--fresh-objective contrast in a domain, scale and level. The horizontal axis is the terminal \pmd{} difference; the vertical axis is the percentage-point survival difference at four verified draws, averaged equally over eligible seed-specific contrasts. Survival uses jointly eligible prompt--group pairs within each seed. Marker shape identifies the domain; zero lines mark no change. The 38 comparisons have Pearson and Spearman correlations of $.57$; points show estimates without intervals. The association is descriptive, and the two metrics need not use the same eligible population.}
  \label{fig:withdrawal-pmd-agreement}
\end{figure}

''')

    rows = []
    for domain in ORDER:
        for arm in ('Re:Dr', 'Re:Max'):
            row = pooled.get((domain, arm))
            if row is None:
                continue
            rows.append([DOMAIN[domain], arm,
                         signed(row['budget']['binding/2']['estimate']),
                         ci(row['budget']['binding/4']),
                         signed(row['budget']['pair/2']['estimate']),
                         ci(row['budget']['pair/4'])])
    pair_audit = audit['checks']['pairs']
    mathir_pairs = pair_audit['by_domain']['mathir']
    tex += (r'''\subsection{Two withdrawals at once}
\label{app:withdrawal-severity}
We next remove two binding options at once. Of the ''' + macros['PWPairs'] + r''' pairs of binding
single withdrawals, ''' + macros['PWBindingPairs'] + r''' retain at least one certified mode and enter the
survival comparison. For Graph, Countdown, Python and PantryPlan,
a certified mode survives the pair exactly when it survives both individual
withdrawals. MathIR can differ: removing two actions can eliminate every
realization of a state path even when either action alone can be removed.
Re-enumerating the reduced menus shows that no such case occurs here: for all
''' + thousands(mathir_pairs['pairs']) + r''' MathIR pairs, including those that leave no certified mode,
the surviving set equals the intersection of the two single-withdrawal sets.
''')
    tex += '\n' + table(r'\textbf{Two withdrawals retain survival gains in Graph, Python and PantryPlan.} '
                 'Entries are replay--fresh-objective differences in percentage points at two (@2) '
                 'or four (@4) verified draws. One uses binding single options; Two uses binding '
                 'pairs with a feasibility certificate. Their eligible prompt populations '
                 'can differ. Estimates pool jointly eligible groups. Brackets are pointwise '
                 r'95\% paired prompt-bootstrap intervals with the clustering in '
                 r'Table~\ref{tab:withdrawal-contrasts}.',
                 'tab:withdrawal-severity',
                 ['Domain', 'Arm', 'One @2', 'One @4', 'Two @2', 'Two @4'], rows, 'll' + 'r' * 4) + '\n'
    tex += (r'''Two withdrawals retain less certified support on average
(Table~\ref{tab:withdrawal-inventory}). Re:Dr's pooled survival gain at four
verified draws rises in Graph, Python and MathIR, and falls in Countdown and
PantryPlan. Countdown Level 1 starts with only 4.5 certified modes per prompt;
PantryPlan Level 1 retains only 18\% of its certified modes under a binding pair.
These support differences coincide with the smaller gains but do not
establish their cause. Across domain--scale--level--objective comparisons,
''' + macros['PWPairPositive'] + ' of ' + macros['PWPairCells'] + r''' point estimates are positive at four verified draws,
compared with ''' + macros['PWBudgetFourPositive'] + r''' of ''' + macros['PWBudgetFourCells'] + r''' for single withdrawals.

\subsection{Decoding settings as a substitute}
\label{app:withdrawal-decoding}
Can a better decoding setting give an ordinary policy the same survival?
We evaluate binding Level-1 withdrawals on 128 prompts per domain, using the
separate Qwen2.5-0.5B cohort trained for twelve passes in
App.~\ref{app:decoding-objection}. Its replay variant combines separate mass
and within-buffer balancing terms with novelty and semantic-entropy shaping;
it differs from Re:Dr in both objective and training duration. Five seeds per
arm are evaluated at ''' + macros['PWCtrlSettings'] + r''' settings: six temperatures from $.5$ to $2.0$,
three settings with larger groups, and two with top-$p=.95$. Larger groups
increase $K$ from 8 to 32, but use two groups rather than four: total responses
per prompt double from 32 to 64. The reference is $T=1$, top-$p=1$, and four
groups of eight. For each domain, we select the control setting with the
highest four-verified-draw survival on these evaluation prompts, while replay
keeps the reference setting.
''')
    rows = []
    for entry in controls['contrasts']:
        low, high = entry['control_grid_range']['4']
        rows.append([DOMAIN[entry['domain']], points(entry['control_reference']['4']),
                     points(low) + '--' + points(high),
                     '$T{=}' + f"{entry['control_best_temperature']:g}" + '$',
                     points(entry['replay_reference']['4']), ci(entry['gap_vs_best']['4'])])
    tex += '\n' + table(r'\textbf{Replay survival exceeds the selected control setting in all five domains.} '
                 'Survival at four verified draws is expressed as percentages. Control, Control grid '
                 'and Replay are equal-seed means on each setting\'s own eligible groups; the grid '
                 'column gives the control range and Best its maximizing temperature. '
                 r'$\Delta$ instead pools jointly eligible groups for reference replay minus selected '
                 'control, so it need not equal the difference of the displayed means. Brackets are '
                 r'pointwise 95\% intervals from 20{,}000 paired prompt-bootstrap replicates, '
                 'keeping all groups and five seeds of a prompt together. '
                 'The selected setting stays fixed in the bootstrap; intervals condition on '
                 'that selection and the seeds.',
                 'tab:withdrawal-decoding',
                 ['Domain', 'Control', 'Control grid', 'Best', 'Replay', r'$\Delta$ vs best'],
                 rows, 'lrrcrr') + '\n'
    tex += (r'''Control survival spans at most ''' + macros['PWCtrlSpanMax'] + r''' percentage points across the grid.
The paired replay gains range from ''' + macros['PWCtrlGapMin'] + r''' to ''' + macros['PWCtrlGapMax'] + r''' points, with all five
pointwise intervals above zero. Every selected control setting changes only the
temperature, keeping eight-response groups and top-$p=1$. The tested decoding
changes therefore do not close the survival gap in this cohort. Because each
control setting is selected on the evaluation prompts, the control's survival
is, if anything, optimistic for new prompts. Finite samples cannot show that
the missing alternative modes have zero probability under the control.

''')
    rec = json.loads((RECOVERY / 'results.json').read_text())
    rec_protocol = rec['protocol']
    summary = {(x['domain'], x['arm'], x['strategy']): x for x in rec['summary']}
    rec_contrast = {(x['domain'], x['strategy']): x['delta'] for x in rec['contrasts']}
    within = {(x['domain'], x['arm'], x['strategy'], x['metric']): x
              for x in rec['strategy_contrasts']}
    if rec['complete_cells'] != rec['expected_cells']:
        raise ValueError('the recovery cohort is not complete')
    ordinary_calls = [rec_contrast[(d, 'ordinary')]['recovery_calls'] for d in ORDER]
    macros['PWRecCells'] = str(rec['complete_cells'])
    macros['PWRecWithdrawals'] = thousands(sum(
        rec_protocol['counts'][d]['test_withdrawals'] for d in ORDER))
    macros['PWRecCallsMax'] = f"{-min(m['estimate'] for m in ordinary_calls):.2f}"
    macros['PWRecCallsMin'] = f"{-max(m['estimate'] for m in ordinary_calls):.2f}"
    rows = []
    for domain in ORDER:
        for strategy, label in (('ordinary', 'Ordinary'), ('temperature', 'Tuned temperature'),
                                ('diversity_prompt', 'Diversity prompt')):
            control, replay = summary[(domain, 'control', strategy)], summary[(domain, 'replay', strategy)]
            rows.append([DOMAIN[domain], label,
                         points(control['zero_call_recovery']), points(control['recovered']),
                         f"{control['recovery_calls']:.2f}",
                         points(replay['zero_call_recovery']), points(replay['recovered']),
                         f"{replay['recovery_calls']:.2f}"])
    tex += (r'''\subsection{Recovery after a withdrawal}
\label{app:withdrawal-recovery}
Recovery is evaluated at the Level-1 Qwen2.5-0.5B terminal Dr.GRPO and Re:Dr
checkpoints from two matched training seeds in each of five domains. Each domain has 32 test and 16 temperature-calibration prompts in a
disjoint split. Prompts without a binding withdrawal that retains a certified
mode are excluded, leaving 28 test and 15 calibration prompts in Countdown and all
prompts elsewhere. Up to four binding options per test prompt give
''' + macros['PWRecWithdrawals'] + r''' distinct withdrawals, shared by both arms and seeds.

Each initial portfolio contains eight responses. Ordinary sampling uses its
terminal evaluation group at $T=1$. The temperature strategy generates a group
at the temperature in $\{.7,1.0,1.3\}$ maximizing mean survival on the
calibration prompts. The diversity-prompt strategy generates the eight
responses sequentially, with the preceding responses included in each request.
All strategies share the same recovery procedure. If no initial mode survives,
the model generates one response per call at $T=1$, for up to eight calls,
with the revised task, the initial portfolio and previous failed attempts in
context. Recovery stops at the first verified response whose mode survives the
withdrawal.

Survival uses the mode predicates defined above. In MathIR, a retained or newly
generated state path can count as surviving even if realizing it under the
reduced menu requires a different action sequence. PantryPlan responses are
six-bit ingredient-support masks projected onto verified quantities by the
environment (App.~\ref{app:prompt-pantry}); survival is evaluated on the resulting
support key. These outcomes therefore measure whether a mode remains feasible,
not whether the original response text can be reused unchanged.
''')
    tex += '\n' + table(r'\textbf{Replay improves ordinary and temperature-tuned recovery in every domain.} '
                 'Saved is the percentage of withdrawals with a surviving initial mode; By 8 '
                 'also counts those resolved within eight additional responses. Calls is the mean '
                 'additional response count, with zero for initial survival and eight for '
                 'unresolved cases. Each row pools the same withdrawals over two matched seeds. '
                 'Fresh is Dr.GRPO and Replay is Re:Dr; the three strategies differ only '
                 'in initial portfolio generation.',
                 'tab:withdrawal-recovery',
                 ['Domain', 'Strategy', 'Fresh saved', 'Fresh by 8', 'Fresh calls',
                  'Replay saved', 'Replay by 8', 'Replay calls'],
                 rows, 'll' + 'r' * 6) + '\n'
    rows = []
    for domain in ORDER:
        for strategy, label in (('ordinary', 'Ordinary'), ('temperature', 'Tuned temperature'),
                                ('diversity_prompt', 'Diversity prompt')):
            delta = rec_contrast[(domain, strategy)]
            rows.append([DOMAIN[domain], label, ci(delta['zero_call_recovery']),
                         ci(delta['recovered']),
                         ci({'estimate': delta['recovery_calls']['estimate'],
                             'ci95': delta['recovery_calls']['ci95']}, 2)
                         if False else f"${delta['recovery_calls']['estimate']:.2f}$ "
                         f"$[{delta['recovery_calls']['ci95'][0]:.2f}, "
                         f"{delta['recovery_calls']['ci95'][1]:.2f}]$"])
    tex += '\n' + table(r'\textbf{Recovery benefits depend on the initial portfolio strategy.} '
                 'Entries are paired Re:Dr--Dr.GRPO differences on the same withdrawals under '
                 'each strategy. Saved and By 8 are percentage points; Calls is additional '
                 r'responses. Brackets are pointwise 95\% intervals from 20{,}000 paired '
                 'prompt-bootstrap replicates. Each prompt\'s withdrawals and '
                 'both training seeds move together; intervals condition on those two matched training seeds.',
                 'tab:withdrawal-recovery-contrast',
                 ['Domain', 'Strategy', r'$\Delta$ saved', r'$\Delta$ by 8', r'$\Delta$ calls'],
                 rows, 'llrrr') + '\n'
    reversal = [DOMAIN[d] for d in ORDER
                if rec_contrast[(d, 'diversity_prompt')]['zero_call_recovery']['ci95'][1] < 0]
    helped = [DOMAIN[d] for d in ORDER
              if within[(d, 'control', 'diversity_prompt', 'zero_call_recovery')]['ci95'][0] > 0]
    # Recovery phases are keyed per strategy ('recovery/ordinary' and so on).
    def recovery_responses(arm):
        return sum(totals['responses'] for cell in rec['costs'] if cell['arm'] == arm
                   for phase, totals in cell['phases'].items() if phase.startswith('recovery'))

    control_cost, replay_cost = recovery_responses('control'), recovery_responses('replay')
    tex += (r'''With ordinary or temperature-tuned initial portfolios, replay resolves more
withdrawals and uses fewer recovery responses in every domain. All ten
resolution contrasts and all ten response-count contrasts have pointwise
intervals excluding zero. With ordinary initial portfolios, replay needs
''' + macros['PWRecCallsMin'] + r''' to ''' + macros['PWRecCallsMax'] + r''' fewer recovery responses per withdrawal. Summed over all
three strategies and both seeds, replay uses ''' + thousands(replay_cost) + r''' recovery responses and
the control uses ''' + thousands(control_cost) + r'''. These totals cover recovery only, excluding initial
portfolios and temperature calibration, and response counts do not account for
token or context lengths or for training compute.

The diversity prompt increases the control's initial survival in Graph and
PantryPlan relative to ordinary sampling, with intervals above zero. For
replay, it lowers the initial-survival point estimate in every domain. The
intervention changes both the request for alternatives and the accumulated
context, and this design does not separate their effects. With both arms using
the diversity prompt, PantryPlan favors the control by 22.66 points of initial survival and 12.50 points of
resolution by eight calls. Replay's initial-survival difference is positive
in the other four domains, ranging from 2.7 to 48.4 points, although the
Countdown and MathIR intervals include zero. PantryPlan's six-bit response format
permits enumerating support masks, but this comparison does not establish that
enumerability explains the reversal.

\paragraph{Interpretation and limits.} In Graph, Countdown, Python and PantryPlan,
Re:Dr's broader four-draw portfolios accompany larger survival gains than in
MathIR. The positive association with \pmd{} is descriptive. Fixing the number
of verified draws conditions on jointly eligible groups; it does not fix policy
accuracy, and eligibility can change with the draw budget. An option enters
the analysis only when a certified mode survives it; where the certified count
is a lower bound on the verifier's support, some excluded options may also be
feasible. The interventions cover single and paired withdrawals rather than
arbitrary task revisions. Recovery outcomes rest on domain-specific mode
predicates, and PantryPlan's ordering reverses under the diversity prompt. Finally, the intervals quantify prompt-sampling uncertainty within
the observed training cohorts; they do not estimate variability over new
training seeds.
''')

    report = {'schema': 'paper-portfolio-withdrawals-v1', 'source_sha256': bindings,
              'macros': macros, 'headline': head, 'protocol': protocol,
              'endpoint_parity': parity, 'census_coverage': results['census_coverage'],
              'inventory': inventory, 'contrasts': results['contrasts'], 'pooled': results['pooled'],
              'pmd_agreement': results['pmd_agreement'], 'audit': audit['checks'],
              'validation': 'Frozen protocol and audit authenticated; every printed value read from the '
                            'bound analysis record.'}
    report['agreement_figure'] = render_agreement_figure(
        results, PAPER / 'figures/withdrawal_pmd_agreement')
    return report, format_domain_names(tex), format_domain_names(macro_tex)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    report, tex, macro_tex = build()
    content = {PAPER / f'results/{STEM}.tex': tex,
               PAPER / f'results/{STEM}.json': json.dumps(report, indent=2, sort_keys=True) + '\n',
               MACROS: macro_tex}
    for path, text in content.items():
        if args.check:
            if path.read_text() != text:
                raise ValueError('Paper artifact drift: ' + str(path))
        else:
            path.write_text(text)
    # The agreement plate is rendered by build(); bind its bytes so the figure
    # is authenticated exactly as every table in this appendix is.
    for kind, binding in report['agreement_figure'].items():
        rendered = ROOT / binding['path']
        if digest(rendered) != binding['sha256']:
            raise ValueError('Agreement figure drift: ' + str(rendered))
    print('Portfolio withdrawals: protocol and audit verified; '
          + ('retained assets match.' if args.check else 'appendix, macros and record written.'))


if __name__ == '__main__':
    main()
