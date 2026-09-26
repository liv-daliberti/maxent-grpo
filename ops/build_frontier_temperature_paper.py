#!/usr/bin/env python3
"""Render the completed two-temperature matched-subset analysis for the appendix."""
from pathlib import Path
import argparse
import hashlib
import json

try:
    from ops.paper_domain_typography import format_domain_names
except ModuleNotFoundError:
    from paper_domain_typography import format_domain_names

_DOMAIN_LANGUAGE_EXCEPTIONS = (
    "Python lambda", r"Python \texttt{lambda}", "Python modulo",
)

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
    # Typeset negative values with a true minus rather than a hyphen.
    signs=lambda text:text.replace('-','$-$')
    result=signs(format(value*factor,fmt))
    if interval:
        if item['ci95'] is None:return result+' [--]'
        if item.get('defined_replicates') != 20000:return result+r' [--]$^\dagger$'
        result+=' ['+', '.join(signs(format(v*factor,fmt)) for v in item['ci95'])+']'
    return result

def render(record):
    lines=[r'\subsection{Requested-temperature sensitivity}',
           r'\label{app:hosted-temperature}',
           'Grok 4.3 and Kimi K3 each receive requested temperatures $T=1.0$ and',
           '$T=1.5$ on the same 120 tasks: eight prompts in each of five domains and',
           'three levels. Eight responses from separate requests per prompt give 960',
           'responses per deployment--temperature condition and 3,840 in total.',
           'Within each deployment, the two conditions share all non-temperature',
           'request settings and run concurrently. Accuracy and mode counts include',
           'all responses, including verification failures and truncated outputs.', '',
           'The providers accept both requested temperature settings, but their responses',
           'do not report effective sampling temperatures. DeepSeek is outside this',
           'comparison because its thinking-mode interface ignores the temperature setting.',
           'The results concern these requested settings on this 120-task subset and',
           'do not establish a general temperature response for all hosted deployments.', '',
           'Each level summary averages five domains equally; All averages the 15',
           'domain--level cells equally. Collision pools correct pairs within each cell,',
           'so tasks with more correct responses receive more weight. Eligible tasks',
           'and pair weights can differ between temperatures; the contrast does not',
           'hold accuracy or the correct-pair population fixed. An undefined cell makes',
           'the corresponding average undefined. Pointwise 95\% intervals use 20,000',
           'paired whole-prompt bootstrap resamples, stratified by domain and level.',
           'Each resampled prompt includes all eight responses. Prompts are paired',
           'across conditions; output draws are not. Intervals have no multiplicity adjustment.', '']
    grok=record['models']['grok-4.3']['analyses']['strict']['groups']['five_domain_macro']['overall']
    kimi=record['models']['FW-Kimi-K3']['analyses']['strict']['groups']['five_domain_macro']['overall']
    kimi_counts=record['models']['FW-Kimi-K3']['analyses']['strict']['totals']
    sparse_replicates=[record['models']['FW-Kimi-K3']['analyses']['strict']['groups']['five_domain_macro']['levels'][level]['t1p5_minus_t1p0']['collision']['defined_replicates'] for level in ('1','2')]
    lines += [r'\paragraph{Observed sensitivity.}',
              'For Grok, the observed diversity change is '
              + number(grok['t1p5_minus_t1p0']['distinct8'],'distinct8',True,True)
              + ' modes, and the collision change is '
              + number(grok['t1p5_minus_t1p0']['collision'],'collision',True,True)
              + ' percentage points. Both intervals include zero, so neither shows a clear diversity gain.',
              'The per-response accuracy change is '
              + number(grok['t1p5_minus_t1p0']['accuracy'],'accuracy',True,True)
              + ' percentage points. These results do not establish invariance to temperature in general.', '',
              f'At $T=1.5$, {kimi_counts["t1p5"]["truncated_responses"]:,} of Kimi\'s {kimi_counts["t1p5"]["responses"]:,} outputs reach the token limit,',
              f'compared with {kimi_counts["t1p0"]["truncated_responses"]:,} at $T=1.0$. Strict accuracy falls from '
              + number(kimi['t1p0']['accuracy'],'accuracy') + r'\% to '
              + number(kimi['t1p5']['accuracy'],'accuracy') + r'\%, while raw diversity falls from '
              + number(kimi['t1p0']['distinct8'],'distinct8') + ' to '
              + number(kimi['t1p5']['distinct8'],'distinct8') + ' modes.',
              'The large increase in truncation accompanies losses in accuracy and distinct modes.',
              'These results do not isolate correct-output concentration at matched accuracy.',
              'The five-domain collision mean is undefined at',
              '$T=1.5$ because some cells have no eligible correct pairs. Neither model',
              'has provider-declared refusals in this experiment.', '',
              r'$\dagger$ marks a defined contrast whose interval is omitted because some',
              'bootstrap resamples have undefined cells. Five-domain averages require all',
              'five cell estimates; conditioning on only the estimable resamples would',
              'change the uncertainty calculation.', '']
    def measure(grade,model,level,metric):
        groups=record['models'][model]['analyses'][grade]['groups']['five_domain_macro']
        group=groups['overall'] if level=='All' else groups['levels'][level]
        return [number(group['t1p0'][metric],metric),number(group['t1p5'][metric],metric),
                number(group['t1p5_minus_t1p0'][metric],metric,True,True)]

    # The normalized table repeated the strict one almost entirely: the
    # normalizer rescues a response in a minority of these macro cells, and the
    # rest were printed twice to show that nothing changed. It now carries the
    # rows it changes, with the full grade retained in the source record.
    for grade in ('strict', 'normalized_secondary'):
        secondary=grade!='strict'
        body=[]
        for index,(model,label) in enumerate(MODELS.items()):
            block=[]
            for level in ('1','2','3','All'):
                for metric,short in [('accuracy','A'),('distinct8','D'),('collision','C')]:
                    values=measure(grade,model,level,metric)
                    if secondary and values==measure('strict',model,level,metric):
                        continue
                    block.append([label,level,short,*values])
            if not block: continue
            if body: body.append(None)
            for row in block: body.append(row)
        if secondary and not body:
            lines += ['Formatting normalization leaves every grade unchanged.', '']
            continue
        caption=(r'  \caption{\textbf{Higher requested temperature yields no clear Grok diversity gain and much lower Kimi accuracy.}'
                 r' Strict executable grades compare $T=1.5$ with $T=1.0$.'
                 if not secondary else
                 r'  \caption{\textbf{Formatting normalization leaves the temperature sensitivity largely unchanged.}'
                 r' Only rows differing from strict grading appear; all other rows are identical.')
        lines += [r'\begin{table}[!htbp]',r'  \centering',r'  \setlength{\parfillskip}{0pt plus .20\linewidth}',
                  caption,
                  '  Each condition contains eight responses on each of 120 tasks.',
                  '  Level summaries average five domains equally; All averages 15 domain--level cells.',
                  r'  A is per-response accuracy (\%); D is mean \texttt{distinct@8};',
                  r'  C is correct-pair collision (\%), pooling pairs within each cell.',
                  r'  Differences subtract $T=1.0$ from $T=1.5$: mode counts for D and percentage points for A/C.',
                  r'  Brackets give pointwise 95\% intervals from paired whole-prompt bootstrap resampling.',
                  r'  $\dagger$ marks an omitted interval when some resamples have undefined cells;',
                  r'  -- denotes an undefined estimate or omitted interval.}',r'  \small',
                  r'  \setlength{\tabcolsep}{5pt}',r'  \begin{tabular}{@{}llrrrr@{}}',r'    \toprule',
                  r'    Deployment & Level & Metric & $T=1.0$ & $T=1.5$ & Difference [95\% interval] \\',r'    \midrule']
        previous=None
        for row in body:
            if row is None:
                lines.append(r'    \midrule'); previous=None; continue
            label,level,short,*values=row
            cols=[label if label!=previous else '', level if short=='A' or secondary else '',
                  short,*values]
            lines.append('    '+' & '.join(cols)+r' \\')
            previous=label
        lines += [r'    \bottomrule',r'  \end{tabular}',r'\end{table}','']
    lines += ['Across the hosted comparisons, Countdown has no uniform reference because its',
              'total support is not certified, so uniform-reference comparisons use the other',
              'four domains. Cross-level differences',
              'also reflect different task populations and do not establish a common ordering',
              'of model difficulty. These inference-only results do not identify a training',
              'or parameter-scale cause of concentration.', '']
    return format_domain_names('\n'.join(lines), exclude_phrases=_DOMAIN_LANGUAGE_EXCEPTIONS)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output-directory',type=Path,default=ROOT/'paper/results');args=parser.parse_args()
    record=build_record(args.source);args.output_directory.mkdir(parents=True,exist_ok=True)
    (args.output_directory/(STEM+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    (args.output_directory/(STEM+'.tex')).write_text(render(record))
    print(json.dumps({'models':list(record['models']),'authenticated_bindings':record['authenticated_file_bindings']}))

if __name__=='__main__':main()
