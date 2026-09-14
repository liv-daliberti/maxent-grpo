#!/usr/bin/env python3
"""Render the completed two-temperature matched-subset analysis for the appendix."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/frontier_temperature_20260911/TEMPERATURE_ABLATION.json'
STEM='frontier_temperature_20260911'
MODELS={'grok-4.3':'Grok 4.3','FW-Kimi-K3':'Kimi K3'}

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def authenticate(value):
    count=0
    if isinstance(value,dict):
        if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str):
            path=Path(value['path']);path=path if path.is_absolute() else ROOT/path
            if digest(path)!=value['sha256']:raise ValueError(f'Stale temperature-analysis binding: {path}')
            count+=1
        for child in value.values():count+=authenticate(child)
    elif isinstance(value,list):
        for child in value:count+=authenticate(child)
    return count

def build_record(source=SOURCE):
    source=Path(source);analysis=json.loads(source.read_text())
    if analysis.get('status')!='complete' or set(analysis.get('models',{}))!=set(MODELS):
        raise ValueError('Both Grok and Kimi must have complete audited temperature pairs before paper export.')
    if analysis['bootstrap']['replicates']!=20000:raise ValueError('Unexpected temperature bootstrap contract.')
    authenticated=authenticate(analysis)
    models={}
    for model,block in analysis['models'].items():
        if not block.get('validation') or not all(block['validation'].values()):
            raise ValueError(f'{model}: incomplete independent temperature audit')
        for condition,temperature in [('t1p0',1.0),('t1p5',1.5)]:
            if block['conditions'][condition]['requested_temperature']!=temperature:
                raise ValueError(f'{model}: unexpected requested temperature')
        models[model]={'label':MODELS[model],'conditions':block['conditions'],
                       'validation':block['validation'],
                       'analyses':{grade:{'groups':data['groups'], 'totals':data['totals']} for grade,data in block['analyses'].items()}}
    return {'schema':'frontier-temperature-paper-v1','builder_sha256':digest(__file__),
            'source':{'path':str(source.resolve().relative_to(ROOT)),'sha256':digest(source)},
            'authenticated_file_bindings':authenticated,'bootstrap':analysis['bootstrap'],
            'selected_rows_sha256':analysis['selected_rows_sha256'],'collection_gate':analysis['collection_gate'],
            'definitions':analysis['definitions'],'limitations':analysis['limitations'],'models':models}

def number(item,metric,interval=False,signed=False):
    factor,digits=(1,3) if metric=='distinct8' else (100,2)
    value=item['estimate']
    if value is None:return '--'
    fmt=('+' if signed else '')+f'.{digits}f'
    result=format(value*factor,fmt)
    if interval:
        if item['ci95'] is None:return result+' [--]'
        if item.get('defined_replicates') != 20000:return result+r' [--]$^\dagger$'
        result+=' ['+', '.join(format(v*factor,fmt) for v in item['ci95'])+']'
    return result

def render(record):
    lines=[r'\subsection{A matched requested-temperature sensitivity test}',
           r'\label{app:hosted-temperature}',
           'A separate fixed subset contains eight prompts from each of the five domains',
           'at each of three levels: 120 prompts, with eight independent draws per prompt',
           'and temperature. Grok 4.3 and Kimi K3 each receive requested temperatures',
           '$T=1.0$ and $T=1.5$, giving 960 draws per condition and 3,840 in total.',
           'The two conditions share task rows and all non-temperature request settings;',
           'both temperatures are collected concurrently within each deployment.',
           'Every response is retained. This subset is a separate experiment and does not',
           'replace any full-cohort result in the main hosted display.', '',
           'Provider API contracts support the requested controls and the deployments accept',
           'the requests, but native responses do not report effective sampling temperature.',
           'DeepSeek is excluded from this contrast because its documented thinking-mode',
           'interface ignores the temperature setting. These contrasts therefore concern',
           'the tested requested settings, not arbitrary temperatures or all deployments.', '',
           'Tables average all five domains equally within each level; All equally averages',
           'all 15 domain--level cells. Confidence intervals use 20,000 paired whole-prompt',
           'bootstrap resamples stratified by domain and level, retaining each eight-draw',
           'group. They are pointwise and have no multiplicity adjustment. Correct-pair',
           'collision remains conditional on correct draws and can change its eligible',
           'prompt population; an undefined cell makes the corresponding macro undefined.', '']
    grok=record['models']['grok-4.3']['analyses']['strict']['groups']['five_domain_macro']['overall']
    kimi=record['models']['FW-Kimi-K3']['analyses']['strict']['groups']['five_domain_macro']['overall']
    kimi_counts=record['models']['FW-Kimi-K3']['analyses']['strict']['totals']
    sparse_replicates=[record['models']['FW-Kimi-K3']['analyses']['strict']['groups']['five_domain_macro']['levels'][level]['t1p5_minus_t1p0']['collision']['defined_replicates'] for level in ('1','2')]
    lines += [r'\paragraph{Observed sensitivity.}',
              'For Grok, the observed breadth change is '
              + number(grok['t1p5_minus_t1p0']['distinct8'],'distinct8',True,True)
              + ' modes, and the collision change is '
              + number(grok['t1p5_minus_t1p0']['collision'],'collision',True,True)
              + ' percentage points. These intervals do not show a clear broadening at the tested setting.',
              'The per-response accuracy change is '
              + number(grok['t1p5_minus_t1p0']['accuracy'],'accuracy',True,True)
              + ' percentage points. This does not establish invariance to temperature in general.', '',
              f'Kimi returns {kimi_counts["t1p5"]["truncated_responses"]:,}/{kimi_counts["t1p5"]["responses"]:,} token-limited outputs at $T=1.5$,',
              f'compared with {kimi_counts["t1p0"]["truncated_responses"]:,} at $T=1.0$. Strict accuracy falls from '
              + number(kimi['t1p0']['accuracy'],'accuracy') + r'\% to '
              + number(kimi['t1p5']['accuracy'],'accuracy') + r'\%, while raw breadth falls from '
              + number(kimi['t1p0']['distinct8'],'distinct8') + ' to '
              + number(kimi['t1p5']['distinct8'],'distinct8') + ' modes.',
              'This is a severe token-limit and accuracy failure, not a clean increase in',
              'correct-output concentration. The five-domain collision mean is undefined at',
              '$T=1.5$ because some cells have no eligible correct pairs. Neither model',
              'has provider-declared refusals in this experiment.', '',
              r'$\dagger$ marks an omitted interval when fewer than all 20,000 bootstrap',
              f'replicates define the contrast. For Kimi collision, only {sparse_replicates[0]:,} and {sparse_replicates[1]:,}',
              'replicates are defined at Levels 1 and 2; the raw conditional percentile',
              'summaries remain in the source record. We do not present them as ordinary',
              '95\% intervals. Undefined cells are never replaced by zero or silently omitted.', '']
    for grade,title in [('strict','Strict executable grades'),('normalized_secondary','Frozen formatting-normalized grades')]:
        lines += [r'\begin{table}[!htbp]',r'  \centering',r'  \setlength{\parfillskip}{0pt plus .20\linewidth}',
                  r'  \caption{\textbf{Requested-temperature sensitivity: '+title+'.}',
                  r'  A is per-response accuracy (\%), D is \texttt{distinct@8},',
                  r'  and C is correct-pair collision (\%). D differences are mode counts;',
                  '  A/C differences are percentage points. Brackets give paired pointwise',
                  r'  95\% intervals for the paired change in requested temperature.',
                  r'  Every fixed draw remains in its original denominator.}',r'  \small',
                  r'  \setlength{\tabcolsep}{5pt}',r'  \begin{tabular}{@{}llrrrr@{}}',r'    \toprule',
                  r'    Deployment & Level & Metric & $T=1.0$ & $T=1.5$ & Difference [95\% interval] \\',r'    \midrule']
        for index,(model,label) in enumerate(MODELS.items()):
            if index:lines.append(r'    \midrule')
            groups=record['models'][model]['analyses'][grade]['groups']['five_domain_macro']
            for level in ('1','2','3','All'):
                group=groups['overall'] if level=='All' else groups['levels'][level]
                for metric,short in [('accuracy','A'),('distinct8','D'),('collision','C')]:
                    cols=[label if level=='1' and short=='A' else '',level if short=='A' else '',short,
                          number(group['t1p0'][metric],metric),number(group['t1p5'][metric],metric),
                          number(group['t1p5_minus_t1p0'][metric],metric,True,True)]
                    lines.append('    '+' & '.join(cols)+r' \\')
        lines += [r'    \bottomrule',r'  \end{tabular}',r'\end{table}','']
    lines += ['Full domain--level estimates, counts, native outcomes and collision eligibility',
              'remain in the source-bound temperature analysis. Countdown has no finite-total-support',
              'uniform reference; finite-support sensitivity summaries use the other four domains.',
              'The paper record authenticates every included condition summary, native completion',
              'audit, frozen normalizer, paired analysis source and sampling-control review,',
              'with the complete fixed response cohort retained throughout.', '']
    return '\n'.join(lines)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output-directory',type=Path,default=ROOT/'paper/results');args=parser.parse_args()
    record=build_record(args.source);args.output_directory.mkdir(parents=True,exist_ok=True)
    (args.output_directory/(STEM+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    (args.output_directory/(STEM+'.tex')).write_text(render(record))
    print(json.dumps({'models':list(record['models']),'authenticated_bindings':record['authenticated_file_bindings']}))

if __name__=='__main__':main()
