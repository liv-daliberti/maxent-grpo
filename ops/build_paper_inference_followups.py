#!/usr/bin/env python3
"""Render the frozen inference follow-ups; --check authenticates retained assets."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
try:
    from paper_domain_typography import format_domain_names
except ModuleNotFoundError:
    from ops.paper_domain_typography import format_domain_names
import statistics

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/modebench_inference_followups_20260911'
PAPER = ROOT / 'paper'
BINDINGS = PAPER / 'audits/inference_followups_20260912/source_sha256.json'
STEM = 'inference_followups_20260912'
NAMES = {'DeepSeek-V4-Pro':'DeepSeek V4 Pro','FW-Kimi-K3':'Kimi K3',
         'claude-opus-4-8':'Opus 4.8','claude-opus-5':'Opus 5',
         'gpt-5.4':'GPT-5.4','gpt-5.6-sol':'GPT-5.6 Sol','grok-4.3':'Grok 4.3'}
STRATEGIES = {'ordinary':'Ordinary','temperature':'Tuned temperature','diversity_prompt':'Diversity prompt'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((BASE / name).read_text())


def checkpoint(label):
    if label.endswith('initial'):
        return 'Initial'
    return ('Re:Dr' if 'replay_' in label else 'Dr.GRPO') + ' (' + {'43':'a', '46':'b'}[label[-2:]] + ')'


def ci(metric, scale=1, digits=3):
    a, b = metric['ci95']
    return f"${metric['estimate']*scale:.{digits}f}$ $[{a*scale:.{digits}f}, {b*scale:.{digits}f}]$"


def table(caption, label, headings, rows, layout=None, *, column_groups=None):
    lines = [r'\begin{table}[!htbp]', r'\centering', r'\caption{'+caption+'}',
             r'\label{'+label+'}', r'\scriptsize', r'\setlength{\tabcolsep}{3pt}']
    groups = column_groups or (tuple(range(len(headings))),)
    for index, columns in enumerate(groups):
        if index:
            lines.append(r'\par\medskip')
        if column_groups:
            lines.extend([r'\textbf{'+chr(ord('A')+index)+r'}\par\smallskip',
                          r'\resizebox{\linewidth}{!}{%'])
        panel_layout = ('l'+'r'*(len(columns)-1)) if column_groups else (layout or 'l'+'r'*(len(headings)-1))
        lines.extend([r'\begin{tabular}{'+panel_layout+'}', r'\toprule',
                      ' & '.join(headings[i] for i in columns)+r' \\', r'\midrule'])
        # A None row is a rule between blocks, so a table that stacks several
        # groups does not need one table per group.
        lines.extend(r'\midrule' if row is None else
                     ' & '.join(str(row[i]) for i in columns)+r' \\' for row in rows)
        lines.extend([r'\bottomrule', r'\end{tabular}'])
        if column_groups:
            lines.append('}')
    return '\n' + '\n'.join(lines+[r'\end{table}', '', ''])


def build():
    bindings = json.loads(BINDINGS.read_text())
    for name, expected in bindings.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen source changed: '+name)
    o, l, p, saved = [read(name) for name in ('offline/results.json','local_coarse_v2/results.json',
                           'pantry/analysis_complete/results.json','pantry/saved_portfolios/results.json')]
    assert (o['responses'],o['prompt_count'],o['pairs']) == (107520,1920,21)
    assert p['status'] == 'complete' and len(p['models']) == 5
    assert l['all_pass_metrics_unchanged']
    # Independently reconstruct model means and stopping identities from the
    # preserved per-problem records (no repeated model calls or bootstrap fitting).
    for model in p['models'].values():
        for condition, rows in model['per_problem'].items():
            metrics = model['summary'][condition]
            for key, metric in metrics.items():
                vals = [r[key] for r in rows if r.get(key) is not None]
                assert len(vals) == metric['eligible_problems'], (condition,key)
                assert math.isclose(statistics.mean(vals),metric['estimate'],abs_tol=1e-9), (condition,key)
            for row in rows:
                curve = [row[f'recovery_at_{i}'] for i in range(9)]
                if curve[0] is None:
                    assert all(v is None for v in curve)
                    continue
                assert all(a <= b+1e-12 for a,b in zip(curve,curve[1:]))
                assert math.isclose(row['zero_call_survival'],curve[0],abs_tol=1e-12)
                assert math.isclose(row['unresolved'],1-curve[8],abs_tol=1e-12)
                assert math.isclose(row['capped_recovery_calls'],sum(1-v for v in curve[:8]),abs_tol=1e-10)
    pairs=o['cross_model']['strict']['fine']
    pair_rows=[]
    for pair, groups in pairs.items():
        m=groups['all']; coarse=o['cross_model']['strict']['coarse'][pair]['all']
        name=' + '.join(NAMES[n] for n in pair.split(' | '))
        # One row per pair carrying both readings. The two tables had the same
        # 21 pair names in the same order, so splitting them printed every name
        # twice to separate the average-constituent columns from the per-
        # constituent ones.
        pair_rows.append([name,ci(m['mix_minus_average_distinct8']),ci(coarse['mix_minus_average_distinct8']),
                          ci(m['matched_2_mix_minus_average']),m['matched_2_mix_minus_average']['eligible_prompts'],
                          ci(m['mix_minus_a_distinct8']),ci(m['mix_minus_b_distinct8']),
                          ci(m['mix_minus_average_pass8'],100,2)])
    local=[r for r in l['contrasts'] if r['seed']=='fixed_average' and r['grading']=='strict']
    assert len(local)==12
    main={'cross_model_gain_range':[min(g['all']['mix_minus_average_distinct8']['estimate'] for g in pairs.values()),
                                    max(g['all']['mix_minus_average_distinct8']['estimate'] for g in pairs.values())],
          'cross_model_both_positive':sum(all(g['all'][k]['estimate']>0 for k in ('mix_minus_a_distinct8','mix_minus_b_distinct8')) for g in pairs.values()),
          'cross_model_both_positive_ci':sum(all(g['all'][k]['ci95'][0]>0 for k in ('mix_minus_a_distinct8','mix_minus_b_distinct8')) for g in pairs.values()),
          'pantry_replay_minus_drgrpo':p['replay_minus_drgrpo']['fixed_two_seed_average'],
          'pantry_matched_correctness':saved['matched_observed_correctness'],
          'feasibility_counts':saved['feasibility_counts']}
    tex=r'''% Generated by ops/build_paper_inference_followups.py; do not edit numbers.
\section{Inference: Adaptation, Model Mixtures, and Coarser Keys}
\label{app:inference-followups}
We evaluate adaptation to changed requirements, complementarity between
models, and sensitivity to the definition of a solution mode. Model weights
remain fixed during inference. Pointwise 95\% percentile intervals use 20,000
paired whole-problem bootstrap replicates. Resampling preserves all
perturbations, model conditions, and training seeds for each problem;
hosted comparisons resample within each domain and level. These intervals
condition on the observed response pools and training seeds, so they do not
estimate variability across new response pools or training seeds. The 21
model-pair comparisons are dependent, and intervals are unadjusted for
multiple comparisons.

\subsection{PantryPlan portfolios under changed requirements}
\label{app:pantry-adaptation}
We evaluate the initial Qwen2.5-0.5B checkpoint and Level-2 Dr.GRPO and
Re:Dr checkpoints from two matched training seeds on 32 held-out problems.
We derive 192 single-ingredient outages from the task specifications,
independently of generated answers. Exhaustive search over the original
quantity grid finds valid plans for 181 revised tasks. These plans establish
feasibility and are withheld from the model. Five feasible additional dietary
restrictions are reported separately, as they cover only five source problems.

Each portfolio contains eight plans, which we test unchanged against the
revised requirements. If none remains valid, we allow up to eight sequential
recovery calls and stop at the first valid plan. Unresolved portfolios use
all eight calls. Each recovery prompt contains the revised task, initial
portfolio, previous recovery outputs, and deterministic verifier feedback.
Cumulative success includes portfolios that require no recovery. We average
over perturbations within each source problem, then weight problems equally.

The three sampling strategies use the same checkpoints, verifier, and
192-token response cap. Ordinary sampling uses temperature 1.0. Temperature
tuning selects from $\{0.7,1.0,1.3\}$ based on outage survival on 16 separate
development problems (384 responses per checkpoint). Diversity prompting
makes eight sequential calls, requesting a plan with a new ingredient support
and retaining the complete output history, including failures. All strategies
use the same number of initial responses, but their context lengths and total
token costs differ.

\begin{figure}[!htbp]
\centering
\includegraphics[width=\linewidth]{figures/pantry_adaptation_recovery_20260912.pdf}
\caption{\textbf{Re:Dr improves portfolio survival and success within eight
recovery calls.} Panels compare ordinary sampling, tuned temperature, and
diversity prompting on the same 181 feasible ingredient outages from 32
PantryPlan problems. The horizontal axis is the additional recovery-call budget;
the vertical axis includes success without recovery at zero calls. Curves
average outages within problems, then problems equally. Solid diamonds show
Re:Dr, dashed circles Dr.GRPO, and dotted squares the initial checkpoint;
trained curves average two matched seeds. Pointwise 95\% intervals for
paired differences appear in Table~\ref{tab:pantry-adaptation-contrasts}.}
\label{fig:pantry-adaptation-recovery}
\end{figure}
'''
    # Both perturbation families over the same checkpoints and strategies, in
    # one table. They had identical columns and differed only in which
    # perturbation the rows describe, so the caption was written twice.
    rows=[]
    for index,(kind, description) in enumerate([('outage','Outages'),
                                                ('diet','Dietary')]):
        if index: rows.append(None)
        first=True
        for label, model in sorted(p['models'].items(),key=lambda z:('initial' not in z[0],z[0])):
            for strategy, display in STRATEGIES.items():
                m=model['summary'][strategy+'/'+kind]
                rows.append([description if first else '',
                             checkpoint(label),display,*[f"{m[k]['estimate']*scale:.2f}" for k,scale in
                  [('zero_call_survival',100),('recovery_at_8',100),('capped_recovery_calls',1),('total_input_tokens',1),('total_output_tokens',1)]]])
                first=False
    tex+=table(r'\textbf{Re:Dr portfolios adapt better to ingredient outages across all three sampling strategies.} '
               'Eight initial plans face 181 feasible outages from 32 problems or five additional dietary restrictions from five problems. '
               'Survival is success without recovery; By 8 includes success within eight recovery calls (both percentages). '
               'Calls count unresolved cases as eight. Token totals count the initial portfolio once per revised task. '
               'Entries average perturbations within problems, then problems equally; dietary estimates use only five problems. Labels (a) and (b) denote the two matched training replicates.',
               'tab:pantry-adaptation',['Change','Checkpoint','Strategy','Survival','By 8','Calls','Input tokens','Output tokens'],rows,'lllrrrrr')
    rows=[]
    for strategy, display in STRATEGIES.items():
        m=p['replay_minus_drgrpo']['fixed_two_seed_average'][strategy+'/outage']
        rows.append([display,ci(m['zero_call_survival'],100,2),ci(m['recovery_at_8'],100,2),ci(m['capped_recovery_calls'],1,2)])
    tex+=table(r'\textbf{Re:Dr raises outage survival and reduces recovery calls.} '
               'Entries are Re:Dr minus Dr.GRPO, averaged over two matched seeds on 32 paired problems. '
               'Survival and success by eight calls are percentage-point differences; calls include unresolved cases. '
               'Brackets are pointwise 95\% paired problem-bootstrap intervals, conditional on the observed training seeds.',
               'tab:pantry-adaptation-contrasts',['Strategy',r'$\Delta$ survival',r'$\Delta$ by 8 calls',r'$\Delta$ calls'],rows)
    tex+=r'''With ordinary sampling, Re:Dr improves survival over Dr.GRPO by 23.83 percentage points
$[12.16,35.89]$, reduces the mean number of recovery calls, capped at eight, by 2.08
$[1.14,3.03]$, and improves success within eight recovery calls by 26.43 points
$[14.69,38.18]$. Mean input and output token totals fall by 2,539.27
$[1,618.01,3,468.86]$ and 427.40 $[386.83,466.30]$, respectively.
The initial portfolios also contain 1.234 more correct responses out of
eight $[.578,1.984]$. Restricting each paired-seed comparison to portfolios
with equal observed correct counts leaves a 1.05-percentage-point survival
gain $[0.00,3.53]$. We first average eligible seed comparisons within each
problem, then average the 19 problems with at least one eligible comparison.
This restriction changes both the problem and seed composition, and equal
observed counts do not establish equal underlying accuracy. The adaptation
gains therefore do not isolate diversity as their causal mechanism.

Per-perturbation token totals represent deploying one portfolio and recovering
from one change in requirements. Input counts include reused prefixes and
do not measure the compute saved by prefix caching. Table~\ref{tab:pantry-collection-cost}
counts each response once across temperature selection, initial portfolios
from tuned-temperature sampling and diversity prompting, and recovery under
all three strategies. It excludes ordinary initial-portfolio generation.
Neither inference cost summary includes training costs.
'''
    rows=[]
    for label, model in sorted(p['models'].items()):
        for kind,key in [('Calibration','calibration_cost'),('Inference total','actual_new_collection_cost')]:
            c=model[key]
            rows.append([checkpoint(label).replace(' 43', ' (a)').replace(' 46', ' (b)'),kind,model['chosen_temperature'],c['responses'],c['logical_input_tokens'],c['output_tokens'],f"{c['generation_wall_seconds']:.1f}"])
    tex+=table(r'\textbf{Inference costs include temperature selection and recovery.} '
               'Each checkpoint selects $T$ from 384 responses on 16 development problems. '
               'Inference totals include this calibration, initial portfolios for tuned-temperature and diversity prompting, '
               'and recovery under all three strategies on 32 evaluation problems. Ordinary initial-portfolio generation is excluded. '
               'Each response is counted once, including reused input prefixes. Seconds report generation time, excluding training, without hardware normalization. Labels (a) and (b) denote matched training replicates.',
               'tab:pantry-collection-cost',['Checkpoint','Scope',r'$T$','Responses','Input tokens','Output tokens','Seconds'],rows,'llrrrrr')
    tex+=r'''
\subsection{Complementarity across hosted models}
\label{app:cross-model-overlap}
Seven hosted models each produce eight responses to the same 1,920 prompts
(five domains, three levels, and 128 prompts per domain and level), totaling
107,520 responses. We use strict grading and the original Opus 5 Python
interface; the formatting-normalized hosted comparison uses a revised
Opus 5 interface. For each of the 21 model pairs, a mixed portfolio draws four
responses without replacement from each model's eight-response pool. We
compute its expected distinct-mode count exactly over all $\binom{8}{4}^2=4{,}900$
subset pairs, then compare with each constituent's complete eight-response
portfolio and their average. These are expectations within the finite
response pools, conditional on the outputs generated.

Every mixture increases the expected number of distinct solution modes
relative to its average constituent by $.115$--$.266$. Ten pairs improve
on both constituents in point estimates; nine have pointwise intervals
entirely above zero against both. On
prompts with at least four correct responses from each model, we also compare
a mixture of two correct draws per model with four correct draws from either
constituent. All pairwise gains over the average constituent remain positive
($.073$--$.177$ modes). Fixing the number of correct draws conditions on a
different eligible subset for each pair. It does not equate accuracy or the
number of attempts required to obtain those correct responses.
'''
    tex+=table(r'\textbf{Every model pair increases expected distinct modes over its average constituent.} '
               'A 4+4 mixture samples without replacement from the two eight-response pools; A and B name the constituents in order. '
               'Panel A compares with their average: fine and coarse keys use all 1,920 prompts, while the four-correct-draw comparison '
               'uses $n$ prompts with at least four correct responses per model (2+2 mixed versus four per constituent). '
               'Panel B compares fine-key distinct modes with each constituent and pass@8 with their average. '
               'Entries are distinct-mode differences except pass@8, in percentage points. Brackets are pointwise 95\% paired '
               'problem-bootstrap intervals, stratified by domain and level; comparisons share models and prompts.',
               'tab:cross-model-mixtures',
               ['Pair','Fine','Coarse','Four correct draws',r'$n$',
                r'$\Delta D$ vs A',r'$\Delta D$ vs B',r'$\Delta$ pass@8 vs mean'],
               pair_rows, column_groups=((0, 1, 2, 3, 4), (0, 5, 6, 7)))
    tex+=r'''Cross-model collision ranges from $.624$ to $.740$, compared with $.704$
to $.819$ for mean within-model collision. Each pair gives equal weight
to prompts with $\geq2$ correct responses from each model. Absence of
observed overlap does not establish disjoint supports. Gains relative to
the average constituent persist with coarser keys and formatting-normalized
grading. Equal response counts do not equate token use, compute, monetary
cost, or reasoning interfaces across providers. These comparisons of finite
response pools do not identify causal training mechanisms.

\subsection{Sensitivity to task-relevant equivalences}
\label{app:coarser-keys}
We merge solution keys according to task-relevant equivalences and recompute
diversity. Graph merges global color renamings that fix every anchored color;
vertex identities remain fixed. Python groups opposite members $d$ and $n/d$ of the
same unordered proper factor pair, retaining input-vector order. Countdown
flattens associative additions and multiplications and sorts their children,
preserving operands, multiplicity, and subtraction/division boundaries.
MathIR retains its simplified state trajectory, and PantryPlan retains
ingredient supports, whose differences can affect survival under outages. The same domain-specific
map applies to each model's correct responses. Correctness is unchanged;
deterministic merging cannot increase the observed distinct-mode count or
reduce correct-pair collision.

Uniform mass over fine keys generally induces unequal mass across coarse
groups. For Graph and Python, exhaustive enumeration determines both the
number of coarse groups and their sizes, allowing separate collision
references for uniform fine-key mass and uniform coarse-group mass.
Countdown's full verifier support is unknown, so these uniform references
cannot be computed for that domain.
'''
    rows=[]
    for label, groups in o['coarse_models']['strict'].items():
        m=groups['all']
        rows.append([NAMES[label],*[ci(m[k]) for k in ['fine_distinct8','coarse_distinct8','fine_collision','coarse_collision']]])
    tex+=table(r'\textbf{Coarser keys reduce observed mode counts and increase collision.} '
               'Strict grading uses eight responses per hosted model on 1,920 prompts. $D@8$ averages all prompts; '
               'collision averages prompts with at least two correct responses, with eligibility varying by model. '
               'Brackets are pointwise 95\% paired problem-bootstrap intervals, stratified by domain and level.',
               'tab:hosted-coarse-keys',['Model','Fine D@8','Coarse D@8','Fine collision','Coarse collision'],rows)
    rows=[]
    for r in local:
        rows.append([{'python_factors':'Python','mathir':'MathIR','pantry':'PantryPlan'}[r['domain']],r['level'],r['arm'],len(r['seeds']),
                     ci(r['metrics']['fine_distinct8']),ci(r['metrics']['coarse_distinct8'])])
    tex+=table(r'\textbf{Python diversity gains depend on the factor-pair equivalence and prompt wording.} '
               'Entries give Re:Dr minus Dr.GRPO in distinct modes from eight strictly graded responses. '
               'Level-2 trained Qwen2.5-0.5B checkpoints use 32 paired evaluation problems per level; means weight seeds equally. '
               'Brackets are pointwise 95\% paired problem-bootstrap intervals, conditional on those checkpoints. '
               'Fine and coarse keys coincide for MathIR and PantryPlan.',
               'tab:local-coarse-keys',['Domain','Level','Wording','Seeds',r'Fine $\Delta D@8$',r'Coarse $\Delta D@8$'],rows,'lrllrr')
    tex+=r'''With coarser Python keys, the Re:Dr-minus-Dr.GRPO gain in distinct modes
under original wording falls from $.419$ to zero at Level~3 and from
$.431$ to $.075$ at Level~2. Under neutral wording, the gains remain
$.556$ at Level~2 and $.500$ at Level~3. The original-wording effect can
therefore reflect opposite members of the same factor pair. Its interpretation
depends on both prompt wording and the definition of a solution mode;
neither divisor vectors nor factor-pair vectors identify different algorithms.

'''
    report={'schema':'paper-inference-followups-v1','source_sha256':bindings,'main_claims':main,
            'local_contrasts':local,'pantry_models':{k:{j:v[j] for j in ['summary','chosen_temperature','calibration_cost','actual_new_collection_cost']} for k,v in p['models'].items()},
            'cross_model':o['cross_model'],'coarse_models':o['coarse_models'],
            'validation':'Frozen sources authenticated; per-problem Pantry means, curve monotonicity and capped stopping identities independently reconstructed.'}
    return report,format_domain_names(tex),p


def plot(p):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from paper_style import CONTROL, METHOD, MUTED
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(7.35,2.55),sharey=True)
    lines=[]
    for ax,(strategy,title) in zip(axes,STRATEGIES.items()):
        for group,color,marker,style in [('Initial',MUTED,'s',':'),('Dr.GRPO',CONTROL,'o','--'),('Re:Dr',METHOD,'D','-')]:
            models=[m for label,m in p['models'].items() if checkpoint(label).split()[0]==group]
            y=[100*statistics.mean(m['summary'][strategy+'/outage'][f'recovery_at_{i}']['estimate'] for m in models) for i in range(9)]
            line,=ax.plot(range(9),y,color=color,marker=marker,linestyle=style,markersize=3,label=group)
            if ax is axes[0]: lines.append(line)
        ax.set_title(title);ax.set_xticks([0,2,4,6,8]);ax.set_ylim(0,60);ax.set_xlabel('Additional recovery calls');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Cumulative success (%)')
    fig.legend(handles=lines,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.02))
    fig.tight_layout(rect=(0,.1,1,1))
    for suffix in ('pdf','png'):
        fig.savefig(PAPER/f'figures/pantry_adaptation_recovery_20260912.{suffix}',dpi=180,bbox_inches='tight',metadata={'CreationDate':None} if suffix=='pdf' else None)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    report,tex,p=build()
    paths={s:PAPER/f'results/{STEM}.{s}' for s in ('json','tex')}
    if not args.check: plot(p)
    report['figure_sha256']={f'figures/pantry_adaptation_recovery_20260912.{s}':digest(PAPER/f'figures/pantry_adaptation_recovery_20260912.{s}') for s in ('pdf','png')}
    content={'tex':tex,'json':json.dumps(report,indent=2,sort_keys=True)+'\n'}
    for suffix,path in paths.items():
        if args.check:
            if path.read_text()!=content[suffix]: raise ValueError('Paper artifact drift: '+str(path))
        else: path.write_text(content[suffix])
    print('Inference follow-ups: sources and per-problem identities verified; '+('retained assets match.' if args.check else 'tables, figure and numerical record written.'))

if __name__=='__main__':main()
