#!/usr/bin/env python3
"""Render the frozen inference follow-ups; --check authenticates retained assets."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
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
    return ('ReplayDr.GRPO' if 'replay_' in label else 'Dr.GRPO') + ' ' + label[-2:]


def ci(metric, scale=1, digits=3):
    a, b = metric['ci95']
    return f"${metric['estimate']*scale:.{digits}f}$ $[{a*scale:.{digits}f}, {b*scale:.{digits}f}]$"


def table(caption, label, headings, rows, layout=None):
    return '\n'.join([r'\begin{table}[H]', r'\centering', r'\caption{'+caption+'}',
       r'\label{'+label+'}', r'\scriptsize', r'\setlength{\tabcolsep}{3pt}',
       r'\begin{tabular}{'+(layout or 'l'+'r'*(len(headings)-1))+'}', r'\toprule',
       ' & '.join(headings)+r' \\',r'\midrule',
       *[' & '.join(str(v) for v in row)+r' \\' for row in rows],
       r'\bottomrule',r'\end{tabular}',r'\end{table}',''])


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
    pair_rows=[]; comparisons=[]
    for pair, groups in pairs.items():
        m=groups['all']; coarse=o['cross_model']['strict']['coarse'][pair]['all']
        name=' + '.join(NAMES[n] for n in pair.split(' | '))
        pair_rows.append([name,ci(m['mix_minus_average_distinct8']),ci(coarse['mix_minus_average_distinct8']),
                          ci(m['matched_2_mix_minus_average']),m['matched_2_mix_minus_average']['eligible_prompts']])
        comparisons.append([name,ci(m['mix_minus_a_distinct8']),ci(m['mix_minus_b_distinct8']),
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
\clearpage
\section{Inference Follow-ups: Adaptation, Model Mixtures, and Coarser Keys}
\label{app:inference-followups}
These three analyses require no additional training. Their frozen protocols
precede the reported comparisons. Pointwise 95\% intervals use 20,000 paired
whole-problem bootstrap replicates (seed 20260911); all perturbations, model
arms and fixed training seeds move together within a problem. Intervals do
not estimate variation over a population of training seeds, and the 21 model
pairs are not independent replications. Machine-readable results retain every
registered domain, level, pair, wording and grading sensitivity.

\subsection{Pantry portfolios under changed requirements}
\label{app:pantry-adaptation}
We evaluate the initial Qwen2.5-0.5B checkpoint and Level-2 Dr.GRPO and
ReplayDr.GRPO checkpoints at fixed seeds 43 and 46 on 32 held-out problems.
All 192 single-ingredient outages come from task specifications independently
of generated answers; exhaustive search of the original quantity grid finds
181 feasible revised tasks. Verified feasibility witnesses are withheld from
the model. The five feasible additional dietary restrictions are reported
separately because only five problem clusters qualify.

Each portfolio contains eight plans. We revalidate saved plans unchanged,
then allow at most eight sequential recovery calls if none survives, stopping
at the first valid plan. Unresolved portfolios consume all eight calls.
The common recovery interface shows the revised task, initial portfolio,
previous recovery outputs and deterministic verifier feedback. Cumulative
success includes portfolios that needed no recovery. We average perturbations
within each source problem, then weight problems equally.

Controls retain each checkpoint, verifier and token cap. Temperature is chosen
from $\{0.7,1.0,1.3\}$ by outage survival on 16 separate development problems
(384 responses per checkpoint). Diversity prompting makes eight sequential
calls, requesting a new support while retaining the complete output history,
including failures. These strategies match initial response counts; their
context lengths and token costs differ.

\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{figures/pantry_adaptation_recovery_20260912.pdf}
\caption{\textbf{Recovery after independently chosen feasible Pantry outages.}
Zero additional calls measures survival of the eight saved plans. Subsequent
points include successful recovery within the displayed call budget. Trained
curves average fixed seeds 43 and 46; initial-model curves use one checkpoint.
All panels use the same 32 problems. Paired uncertainty appears in
Table~\ref{tab:pantry-adaptation-contrasts}; curves are descriptive means.}
\label{fig:pantry-adaptation-recovery}
\end{figure}
'''
    for kind, description in [('outage','ingredient outages (32 problem clusters)'),('diet','additional dietary restrictions (five problem clusters)')]:
        rows=[]
        for label, model in sorted(p['models'].items(),key=lambda z:('initial' not in z[0],z[0])):
            for strategy, display in STRATEGIES.items():
                m=model['summary'][strategy+'/'+kind]
                rows.append([checkpoint(label),display,*[f"{m[k]['estimate']*scale:.2f}" for k,scale in
                  [('zero_call_survival',100),('recovery_at_8',100),('capped_recovery_calls',1),('total_input_tokens',1),('total_output_tokens',1)]]])
        tex+=table('Eight initial plans and up to eight recovery calls under '+description+
                   '. Success columns are percentages. Calls include unresolved cases; token totals charge the initial portfolio once per revised task plus recovery.',
                   'tab:pantry-adaptation-'+kind,['Checkpoint','Strategy','Saved','By 8 calls','Calls','Input tokens','Output tokens'],rows,'llrrrrr')
    rows=[]
    for strategy, display in STRATEGIES.items():
        m=p['replay_minus_drgrpo']['fixed_two_seed_average'][strategy+'/outage']
        rows.append([display,ci(m['zero_call_survival'],100,2),ci(m['recovery_at_8'],100,2),ci(m['capped_recovery_calls'],1,2)])
    tex+=table('ReplayDr.GRPO minus Dr.GRPO under outages, averaging fixed seeds 43 and 46. Survival and final-success changes are percentage points. Intervals resample 32 paired problems.',
               'tab:pantry-adaptation-contrasts',['Strategy',r'$\Delta$ saved',r'$\Delta$ by 8 calls',r'$\Delta$ calls'],rows)
    tex+=r'''Ordinary sampling improves saved survival by 23.83 percentage points
$[12.16,35.89]$, reduces capped recovery burden by 2.08 calls
$[1.14,3.03]$, and improves success within eight extra calls by 26.43 points
$[14.69,38.18]$. Mean input and output token totals fall by 2,539.27
$[1,618.01,3,468.86]$ and 427.40 $[386.83,466.30]$, respectively.
However, original correctness also increases by 1.234 correct responses out
of eight $[.578,1.984]$. Conditioning descriptively on equal observed correct
counts leaves a 1.05-point survival gain $[0.00,3.53]$ on 19 eligible problems.
This demonstrates adaptation gains under the tested intervention; it does
not isolate diversity as their causal mechanism.

Logical input tokens include reused prefixes and do not measure cached
compute. Historical ordinary-sampling generation time is unavailable.
Per-perturbation totals represent deploying a portfolio once and recovering
from one update; they differ from actual collection totals, which count each
new response once. Training cost is outside this inference accounting.
'''
    rows=[]
    for label, model in sorted(p['models'].items()):
        for kind,key in [('Calibration','calibration_cost'),('New collection','actual_new_collection_cost')]:
            c=model[key]
            rows.append([checkpoint(label),kind,model['chosen_temperature'],c['responses'],c['logical_input_tokens'],c['output_tokens'],f"{c['generation_wall_seconds']:.1f}"])
    tex+=table('Selected temperatures and actual collection cost. Calibration is shown separately and is included in the new-collection totals; saved ordinary portfolios require no new initial generation. Seconds measure generation, not training or hardware-normalized compute.',
               'tab:pantry-collection-cost',['Checkpoint','Scope',r'$T$','Responses','Input tokens','Output tokens','Seconds'],rows,'llrrrrr')
    tex+=r'''
\subsection{Complementarity across hosted models}
\label{app:cross-model-overlap}
All seven original-protocol cohorts contribute 107,520 responses on the same
1,920 prompts (five domains, three levels, 128 prompts per cell). This primary
analysis uses strict grading and the original Opus 5 Python interface; it is
a different condition from the revised-Opus, normalized hosted display.
Every one of the 21 model pairs is included. We compare random 4+4 portfolios
with eight responses from each constituent, taking exact expectations over
subsets of the saved draws. An independent audit matches the formula against
all 4,900 subsets for each of 252 deterministic pair/prompt examples.

All pairs improve distinct outcomes over the average constituent by
$.115$--$.266$. Ten improve over both constituents in point estimates; nine
have positive pointwise intervals against both. At a fixed successful-response
budget, 2+2 correct draws beat the average four-correct-draw constituent for
all pairs ($.073$--$.177$ additional outcomes); eligibility varies by pair.
These conditional comparisons separate some correctness effects without
identifying unobserved support or a causal training mechanism.
'''
    tex+=table('Every 4+4 mixture versus its average eight-response constituent. Fine and coarse columns use all 1,920 prompts; correctness matching compares 2+2 correct responses with four from each constituent on the eligible subset. Entries are distinct-mode differences with pointwise 95\% intervals.',
               'tab:cross-model-mixtures',['Pair','Fine','Coarse','Correctness matched',r'$n$'],pair_rows)
    tex+=table('Mixtures compared separately with both eight-response constituents. A and B follow the displayed pair order. The last column is the pass@8 change versus the average constituent, in percentage points.',
               'tab:cross-model-constituents',['Pair',r'$\Delta D$ vs A',r'$\Delta D$ vs B',r'$\Delta$ pass@8 vs mean'],comparisons)
    tex+=r'''Shared preferences remain substantial: on joint prompts with at least two
correct draws per model, cross-model collision is $.624$--$.740$, compared
with $.704$--$.819$ for mean within-model collision. These are equal-prompt
averages, not pooled pair counts. The full record stratifies by domain and
level and provides exact-support references or bounds from verified support
witnesses. Absence of a shared observed key does not establish disjoint true
supports. All average-constituent gains remain positive under coarser keys and
the frozen normalization sensitivity. Equal response counts do not match
provider tokens, compute, monetary cost or reasoning interfaces.

\subsection{Sensitivity to task-relevant equivalences}
\label{app:coarser-keys}
Graph groups global color renamings that fix every anchored color; vertex
identities remain fixed. Python groups opposite members $d$ and $n/d$ of the
same unordered proper factor pair, retaining input-vector order. Countdown
flattens associative additions and multiplications and sorts their children,
preserving operands, multiplicity, and subtraction/division boundaries.
MathIR keeps its simplified state trajectory. Pantry keeps ingredient
supports, whose differences can matter under outages. Every map is frozen
from task semantics and applied consistently to authenticated correct keys.
The independent audit verifies all 6,574 unique successful hosted prompt/key
combinations. Correctness is unchanged; deterministic merging cannot increase
observed support or reduce correct-pair collision.

Uniform fine-key mass generally induces nonuniform coarse-group mass.
Graph/Python references are recomputed from exact groups or coarsened verified
witnesses; no unsupported numerical Countdown reference is retained. A uniform
distribution over coarse groups is a separate hypothetical policy.
'''
    rows=[]
    for label, groups in o['coarse_models']['strict'].items():
        m=groups['all']
        rows.append([NAMES[label],*[ci(m[k]) for k in ['fine_distinct8','coarse_distinct8','fine_collision','coarse_collision']]])
    tex+=table('Hosted coarse-key sensitivity under strict original-protocol grading. Distinct counts include all 1,920 prompts; collision conditions on at least two correct responses. Intervals are pointwise.',
               'tab:hosted-coarse-keys',['Model','Fine D@8','Coarse D@8','Fine collision','Coarse collision'],rows)
    rows=[]
    for r in local:
        rows.append([{'python_factors':'Python','mathir':'MathIR','pantry':'Pantry'}[r['domain']],r['level'],r['arm'],len(r['seeds']),
                     ci(r['metrics']['fine_distinct8']),ci(r['metrics']['coarse_distinct8'])])
    tex+=table('Local ReplayDr.GRPO minus Dr.GRPO endpoint probes: 32 problems per domain/level and fixed seed averages. Level-2 trained checkpoints are evaluated at Levels 2 and 3; this is not a reanalysis of every main-paper trajectory. MathIR/Pantry definitions are unchanged.',
               'tab:local-coarse-keys',['Domain','Level','Wording','Seeds',r'Fine $\Delta D@8$',r'Coarse $\Delta D@8$'],rows,'lrllrr')
    tex+=r'''The original-wording Python Level-3 gain falls from $.419$ to zero;
Level 2 falls from $.431$ to $.075$. Neutral-wording coarse gains remain
$.556$ at Level 2 and $.500$ at Level 3. Thus the original Python effect can
reflect opposite members of the same factor pair, and the conclusion depends
on wording and the meaning of output identity. Neither divisor vectors nor
factor-pair vectors identify different algorithms.

\paragraph{Reproduction.}
\path{ops/build_paper_inference_followups.py} checks frozen analysis hashes,
reconstructs Pantry means and stopping identities from per-problem records,
and renders these tables and curves. The accompanying
\path{results/inference_followups_20260912.json} binds the protocols, complete
analysis records, paper values and figure. Full domain/level, per-seed,
normalization, overlap and cost details remain in the bound analysis files.
'''
    report={'schema':'paper-inference-followups-v1','source_sha256':bindings,'main_claims':main,
            'local_contrasts':local,'pantry_models':{k:{j:v[j] for j in ['summary','chosen_temperature','calibration_cost','actual_new_collection_cost']} for k,v in p['models'].items()},
            'cross_model':o['cross_model'],'coarse_models':o['coarse_models'],
            'validation':'Frozen sources authenticated; per-problem Pantry means, curve monotonicity and capped stopping identities independently reconstructed.'}
    return report,tex,p


def plot(p):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from paper_style import CONTROL, METHOD, MUTED
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(7.35,2.55),sharey=True)
    lines=[]
    for ax,(strategy,title) in zip(axes,STRATEGIES.items()):
        for group,color,marker,style in [('Initial',MUTED,'s',':'),('Dr.GRPO',CONTROL,'o','--'),('ReplayDr.GRPO',METHOD,'D','-')]:
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
