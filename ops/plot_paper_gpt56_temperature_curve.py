#!/usr/bin/env python3
"""Source-bound pass@8/breadth curves for measured GPT-5.6 Sol temperatures.

All connected points use reasoning=none and the same complete fixed subset.
Historical medium-reasoning references are retained in source records.
"""
from pathlib import Path
import argparse
from copy import deepcopy
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/frontier_temperature_20260911/GPT56_PASS8_FRONTIER_EXPANDED480.json'
OUTPUT=ROOT/'paper/figures/gpt56_temperature_curve'
LEGACY_TEMPERATURES=('0.5','1.0','1.5','2.0')
LEVELS=('1','2','3')
COLORS=('#00509E','#C76A3A','#7B1FA2')
MARKERS=('o','s','^')
FIGSIZE=(6.6,3.1)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def relative(path):return str(Path(path).resolve().relative_to(ROOT))
def authenticate(value):
    count=0
    if isinstance(value,dict):
        if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str):
            path=Path(value['path']);path=path if path.is_absolute() else ROOT/path
            if sha(path)!=value['sha256']:raise ValueError(f'Stale GPT temperature binding: {path}')
            count+=1
        for child in value.values():count+=authenticate(child)
    elif isinstance(value,list):
        for child in value:count+=authenticate(child)
    return count

def build_record(source=SOURCE):
    source=Path(source);analysis=json.loads(source.read_text())
    expanded=analysis.get('schema')=='gpt56-pass8-temperature-frontier-expanded480-v1'
    prompts=480 if expanded else 120
    responses=prompts*8
    if (analysis.get('schema') not in ('gpt56-pass8-temperature-frontier-v1','gpt56-pass8-temperature-frontier-v2','gpt56-pass8-temperature-frontier-expanded480-v1') or analysis.get('status')!='complete'
            or analysis.get('model')!='gpt-5.6-sol' or analysis.get('reasoning_effort')!='none'):
        raise ValueError('The plotted GPT curve must be complete and use reasoning=none throughout.')
    legacy=analysis['schema']=='gpt56-pass8-temperature-frontier-v1'
    receipt_check='all_3840_native_receipts_authenticated' if legacy else 'all_native_receipts_authenticated'
    required={receipt_check,f'same_{prompts}_prompts_all_eight_slots',
              'only_temperature_varies_within_curve','all_returned_controls_match_requests',
              'source_summaries_reconstructed', 'pass8_computed_from_complete_prompt_groups'}
    if expanded:
        required.update(('expansion_plan_authenticated','original_4800_measurements_retained_unchanged'))
    if not all(analysis.get('validation',{}).get(key) is True for key in required):
        raise ValueError('The full fixed GPT temperature cohort has not passed its independent audit.')
    values=analysis.get('temperatures',[])
    if (not isinstance(values,list) or len(values)<2
            or any(type(t) not in (int,float) or not math.isfinite(t) or not 0<=t<=2 for t in values)
            or len(set(values))!=len(values)):
        raise ValueError('Invalid registered GPT temperature grid.')
    temperatures=tuple(str(float(t)) for t in sorted(values))
    if legacy and temperatures!=LEGACY_TEMPERATURES:
        raise ValueError('The original GPT curve requires its four registered temperatures.')
    if expanded and temperatures!=('0.0','0.5','1.0','1.5','2.0'):
        raise ValueError('The expanded GPT curve requires all five registered temperatures including zero.')
    if set(analysis.get('conditions',{}))!=set(temperatures):
        raise ValueError('Incomplete registered temperature conditions.')
    total_responses=responses*len(temperatures)
    if not legacy and (analysis.get('total_registered_responses')!=total_responses
            or analysis.get('responses_per_condition')!=responses or analysis.get('prompts_per_condition')!=prompts
            or analysis.get('draws_per_prompt')!=8
            or analysis['validation'].get('authenticated_native_receipt_count')!=total_responses):
        raise ValueError('Audited response counts differ from the registered GPT temperature grid.')
    bindings=authenticate(analysis)
    for grade in ('strict','normalized_secondary'):
        blocks=analysis['analyses'][grade]['temperatures']
        if set(blocks)!=set(temperatures):raise ValueError(f'Incomplete {grade} temperature grid.')
        if any(block['counts'].get('responses')!=responses or block['counts'].get('prompts')!=prompts
               for block in blocks.values()):
            raise ValueError(f'{grade}: every temperature must retain {prompts} prompts and all {responses} draws.')
        if analysis['analyses'][grade]['paired_endpoint_contrast'].get('comparison')!=f'T{temperatures[-1]}-T{temperatures[0]}':
            raise ValueError(f'{grade}: endpoint contrast does not match the measured temperature grid.')
    normalized=analysis['analyses']['normalized_secondary']['temperatures']
    points=[]
    for level in LEVELS:
        for temperature in temperatures:
            group=normalized[temperature]['groups']['five_domain_macro']['levels'][level]
            metrics={}
            for metric,maximum in [('pass8',1.),('distinct8',8.)]:
                item=group[metric];value=item['estimate']
                if (not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=maximum
                        or item.get('defined_replicates')!=20000 or not isinstance(item.get('ci95'),list)
                        or len(item['ci95'])!=2):
                    raise ValueError(f'{level}/{temperature}/{metric}: invalid complete-cohort metric.')
                metrics[metric]=deepcopy(item)
            points.append({'level':int(level),'temperature':float(temperature),'metrics':metrics,
                           'source_group':f'/analyses/normalized_secondary/temperatures/{temperature}/groups/five_domain_macro/levels/{level}'})
    if expanded:
        historical=analysis['historical_medium_reference_subset120']
        if historical.get('prompts_per_condition')!=120:
            raise ValueError('The historical medium reference covers only the original 120 prompts.')
        reference=historical['reference']
    else:
        reference=analysis['matched_medium_reference']
    if (reference['reasoning_effort']!='medium' or reference['requested_temperature'] is not None
            or reference['returned_temperature']!=1.0 or reference['connect_to_temperature_curve'] is not False):
        raise ValueError('The matched medium reference must remain separate from controlled none-reasoning curves.')
    return {'schema':('paper-gpt56-pass8-temperature-curve-expanded480-v1' if expanded else 'paper-gpt56-pass8-temperature-curve-v1' if legacy else 'paper-gpt56-pass8-temperature-curve-v2'),'source':{'path':relative(source),'sha256':sha(source)},
            'renderer':{'path':relative(__file__),'sha256':sha(__file__)},'authenticated_bindings':bindings,
            'model':'gpt-5.6-sol','reasoning_effort':'none','grading':'frozen formatting-normalized',
            'sampling':{'temperatures':[float(t) for t in temperatures],'prompts_per_domain_level':prompts//15,'domains':5,
                        'levels':[1,2,3],'draws_per_prompt':8,'responses_per_temperature':responses,'total_responses':total_responses},
            'display':{'x':'pass@8 (%)','y':'distinct correct modes per eight draws',
                       'aggregation':'Equal mean over all five domains within each level.',
                       'connections':'Straight segments in ascending requested temperature; no fit or Pareto claim.',
                       'points':points, 'overall_points':[{'temperature':float(t), 'metrics':{m:deepcopy(normalized[t]['groups']['five_domain_macro']['overall'][m]) for m in ('pass8','distinct8')}} for t in temperatures], 'figure_inches':list(FIGSIZE),'pointwise_intervals':'Retained in the source record and appendix; main plot shows point estimates.'},
            'conditions':analysis['conditions'],'bootstrap':analysis['bootstrap'],'validation':analysis['validation'],
            **({'historical_medium_reference_subset120':deepcopy(historical),
                'cohort_sensitivity':deepcopy(analysis['cohort_sensitivity'])} if expanded
               else {'matched_medium_reference':deepcopy(reference)}),
            'analyses':{grade:{'temperatures':{t:{'groups':block['groups'],'counts':block['counts']} for t,block in data['temperatures'].items()},
                               'paired_endpoint_contrast':data['paired_endpoint_contrast'],
                               **({'paired_high_temperature_contrast':data['paired_high_temperature_contrast']} if expanded else {})} for grade,data in analysis['analyses'].items()}}

def build_figure(record):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rc={'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'xtick.labelsize':8,
        'ytick.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42,'text.color':'#19324A',
        'axes.labelcolor':'#19324A','xtick.color':'#607487','ytick.color':'#607487'}
    with plt.rc_context(rc):
        fig,axes=plt.subplots(1,2,figsize=FIGSIZE,sharey=True,gridspec_kw={'width_ratios':[1,1.3]})
        fig.subplots_adjust(left=.09,right=.985,bottom=.18,top=.77,wspace=.20)
        def draw(ax,points,color,marker,offsets,label=None):
            points=sorted(points,key=lambda p:p['temperature'])
            xs=[p['metrics']['pass8']['estimate']*100 for p in points]
            ys=[p['metrics']['distinct8']['estimate'] for p in points]
            ax.plot(xs,ys,color=color,marker=marker,markersize=4.7,linewidth=1.1,label=label)
            for i,(p,x,y) in enumerate(zip(points,xs,ys)):
                offset=offsets.get(p['temperature'],((-15,14),(5,-14))[i%2])
                ax.annotate(f'{p["temperature"]:g}',(x,y),xytext=offset,textcoords='offset points',
                            color=color,fontsize=8,ha='left',va='center',
                            bbox={'boxstyle':'round,pad=.10','facecolor':'white','edgecolor':'none','alpha':.93},
                            arrowprops={'arrowstyle':'-','color':color,'linewidth':.5},zorder=5)
        draw(axes[0],record['display']['overall_points'],
             '#19324A','D',{0.0:(-18,-10),.5:(-8,-14),1.0:(-21,-3),1.5:(6,10),2.0:(-24,15)})
        offsets={1:{0.0:(-29,-17),.5:(-10,20),1.0:(-4,-14),1.5:(-23,-3),2.0:(5,12)},
                 2:{0.0:(-19,2),.5:(-10,-15),1.0:(-10,10),1.5:(3,-12),2.0:(-18,10)},
                 3:{0.0:(-18,13),.5:(-25,-4),1.0:(4,14),1.5:(-20,12),2.0:(7,6)}}
        if record['sampling']['prompts_per_domain_level']==32:
            offsets={1:{0.0:(5,-18),.5:(-10,20),1.0:(8,7),1.5:(-23,-3),2.0:(5,12)},
                     2:{0.0:(-19,2),.5:(-10,-15),1.0:(-19,8),1.5:(3,-12),2.0:(-10,23)},
                     3:{0.0:(-18,13),.5:(14,-25),1.0:(8,3),1.5:(-20,12),2.0:(7,6)}}
        for level,color,marker in zip((1,2,3),COLORS,MARKERS):
            draw(axes[1],[p for p in record['display']['points'] if p['level']==level],
                 color,marker,offsets[level],f'Level {level}')
        plotted=record['display']['points']+record['display']['overall_points']
        xmin=min(55,5*math.floor((min(p['metrics']['pass8']['estimate']*100 for p in plotted)-5)/5))
        ymin=max(0,min(.38,min(p['metrics']['distinct8']['estimate'] for p in plotted)-.12))
        ymax=max(2.04,max(p['metrics']['distinct8']['estimate'] for p in plotted)+.2)
        for ax,title in zip(axes,('(a) All domains and levels','(b) By benchmark level')):
            ax.set_title(title,loc='left',fontsize=9,pad=8,fontweight='bold')
            ax.set_xlim(max(-2,xmin),102);ax.set_ylim(ymin,ymax)
            ax.set_xticks([x for x in range(0,101,20) if x>=xmin]);ax.set_xlabel('pass@8 (%)')
            ax.grid(color='#D8E2EA',linewidth=.6,alpha=.75)
            for side in ('top','right'):ax.spines[side].set_visible(False)
            for side in ('bottom','left'):ax.spines[side].set_color('#AABAC7')
        axes[0].set_ylabel('Distinct correct modes / 8')
        axes[1].legend(loc='upper right',bbox_to_anchor=(1,1.40),ncol=3,frameon=False,
                       handlelength=1.1,columnspacing=.8,handletextpad=.4,fontsize=8)
        fig.text(.09,.98,'GPT-5.6 Sol',ha='left',va='top',fontsize=9,fontweight='bold')
        fig.text(.985,.88,'Reasoning: none',ha='right',
                 va='center',fontsize=8,color='#607487')
        prompt_count=record['sampling']['prompts_per_domain_level']*15
        fig.text(.09,.025,f'Frozen formatting normalization | {prompt_count} matched prompts, 8 draws each per temperature',
                 ha='left',va='center',fontsize=7,color='#607487')
    return fig

def render(record,output=OUTPUT):
    import matplotlib.pyplot as plt
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig=build_figure(record);outputs={}
    for suffix in ('.pdf','.png'):
        path=output.with_suffix(suffix)
        kwargs={'dpi':240} if suffix=='.png' else {'metadata':{'CreationDate':None,'ModDate':None}}
        fig.savefig(path,facecolor='white',**kwargs);outputs[suffix[1:]]={'path':relative(path),'sha256':sha(path)}
    plt.close(fig);result=deepcopy(record);result['outputs']=outputs
    output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');return result

def interval(item,metric):
    factor,digits=(1,3) if metric=='distinct8' else (100,1)
    if item['estimate'] is None:return '--'
    val=f"{item['estimate']*factor:.{digits}f}"
    ci=item.get('ci95')
    if ci is None or item.get('defined_replicates')!=20000:return val+' [--]'
    return val+' ['+', '.join(f'{x*factor:.{digits}f}' for x in ci)+']'

def render_appendix_legacy(record):
    temperatures=tuple(str(float(t)) for t in record['sampling']['temperatures'])
    first,last=temperatures[0],temperatures[-1]
    lines=[r'\subsection{GPT-5.6 Sol: temperature, pass@8 and verified modes}',r'\label{app:gpt56-temperature}',
        r'Figure~\ref{fig:gpt56-temperature-curve} plots empirical \texttt{pass@8}',
        r'against \texttt{distinct@8} for $T\in\{'+','.join(temperatures)+r'\}$, with reasoning',
        r'\texttt{none}. Each condition retains the same 120 held-out prompts',
        '(eight per domain--level cell), with eight stateless responses per prompt:',
        f"960 responses per temperature and {record['sampling']['total_responses']:,} in total. A prompt contributes one",
        r'to \texttt{pass@8} if at least one of its eight answers is verified correct,',
        r'and contributes its number of distinct correct canonical keys to \texttt{distinct@8}.',
        'Zero-success groups and all unsuccessful draws remain in the denominator.',
        r'We compute success directly from the saved groups, not as $1-(1-p)^8$',
        'from pooled per-response accuracy; task difficulty varies across prompts.', '',
        *([r'The $T=0$ condition was subsequently collected using the same frozen 120',
           'prompts, eight draws per prompt, and request settings except temperature.',
           'A zero-temperature request does not establish deterministic outputs or',
           'certain behavior on unseen prompts.', ''] if '0.0' in temperatures else []),
        'The deployment rejected non-default temperatures with medium reasoning.',
        r'The connected curves therefore use the supported reasoning \texttt{none}',
        'condition. The historical medium reference, retained only in the tables',
        'below and omitted from the figure, uses the same 120',
        r'prompts and eight draws; its requests omitted temperature and receipts returned $T=1$.',
        'Both reasoning setting and collection time differ from the plotted sweep.',
        'This does not test temperature robustness of the medium condition.', '',
        'Each level averages five domains equally; All averages all 15 cells.',
        'Pointwise 95 percent intervals use 20,000 paired whole-prompt bootstrap',
        'resamples stratified by domain and level, retaining all eight draws.',
        'There is no multiplicity adjustment. A zero-width empirical interval',
        'reflects constant observed groups, not certain success on unseen prompts.', '']
    for grade,title in [('strict','Strict executable grades'),('normalized_secondary','Frozen formatting-normalized grades')]:
        lines += [r'\begin{table}[!htbp]',r'\centering',r'\setlength{\parfillskip}{0pt plus .20\linewidth}',
            r'\caption{\textbf{GPT temperature frontier: '+title+r'.}',
            r'P is solved prompts (\%, \texttt{pass@8}); D is \texttt{distinct@8};',
            r'A is per-response accuracy (\%).',
            r'Brackets give pointwise 95\% intervals. M is the historical medium reference;',
            r'all temperatures use reasoning \texttt{none} and eight draws per prompt.}',
            r'\small',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{@{}llrrr@{}}',r'\toprule',
            r'$T$ & Level & P [95\% interval] & D [95\% interval] & A [95\% interval] \\',r'\midrule']
        for temperature in (*temperatures,'M'):
            groups=(record['matched_medium_reference']['analyses'][grade]['groups']['five_domain_macro'] if temperature=='M'
                    else record['analyses'][grade]['temperatures'][temperature]['groups']['five_domain_macro'])
            if temperature=='M':lines.append(r'\midrule')
            for level in (*LEVELS,'All'):
                group=groups['overall'] if level=='All' else groups['levels'][level]
                lines.append(' & '.join([temperature if level=='1' else '',level]+[interval(group[m],m) for m in ('pass8','distinct8','accuracy')])+r' \\')
        lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}','']
    normal=record['analyses']['normalized_secondary'];overall={t:b['groups']['five_domain_macro']['overall'] for t,b in normal['temperatures'].items()}
    best={metric:[t for t in temperatures if math.isclose(overall[t][metric]['estimate'],
          max(overall[u][metric]['estimate'] for u in temperatures),rel_tol=0,abs_tol=1e-12)]
          for metric in ('pass8','distinct8')}
    best_labels={metric:', '.join(f'$T={t}$' for t in winners) for metric,winners in best.items()}
    lines += [r'\paragraph{Observed temperature frontier.}',
        f'At $T={first}$ and $T={last}$, normalized '+r'\texttt{pass@8} is '
        +f"{overall[first]['pass8']['estimate']*100:.2f}"+r'\% and '+f"{overall[last]['pass8']['estimate']*100:.2f}"+r'\%, respectively,',
        r'while \texttt{distinct@8} is '+f"{overall[first]['distinct8']['estimate']:.3f}"+' and '+f"{overall[last]['distinct8']['estimate']:.3f}"+'.',
        f'Across these {len(temperatures)} measured temperatures, the highest observed aggregate',
        r'\texttt{pass@8} occurs at '+best_labels['pass8']+r' and the highest \texttt{distinct@8}',
        'occurs at '+best_labels['distinct8']+'. This finite exploratory sweep does not establish',
        'a population optimum or a continuous Pareto boundary, and level-specific curves differ.', '',
        r'Per-response accuracy changes from '+f"{overall[first]['accuracy']['estimate']*100:.2f}"+r'\% to '
        +f"{overall[last]['accuracy']['estimate']*100:.2f}"+r'\% between the endpoints.',
        r'That diagnostic is distinct from \texttt{pass@8}: success within eight',
        'draws and the reliability of an individual draw need not change together as temperature varies.', '']
    endpoint=normal['paired_endpoint_contrast']['groups']['five_domain_macro']['overall']
    lines += [f'The paired $T={last}$ minus $T={first}$ contrast is '
        +interval(endpoint['pass8'],'pass8')+r' \texttt{pass@8} percentage points and '
        +interval(endpoint['distinct8'],'distinct8')+r' distinct correct modes, with pointwise 95\% intervals.',
        'All prompt-level records, domain estimates, original per-response and',
        'collision analyses, and paired comparisons remain in the retained reports for both strict and normalized grading.', '',
        r'\paragraph{Retained outcomes.}']
    for t in temperatures:
        c=normal['temperatures'][t]['counts']
        lines.append(f'At $T={t}$, {c["truncated_responses"]} of {c["responses"]} outputs are token-limited and {c["native_refusals"]} are provider-declared refusals.')
    lines += ['These outcome counts retain every registered draw and are separate from',
        'the verifier-based correctness and canonical-key measurements above.', '']
    return '\n'.join(lines)

def render_appendix_expanded(record):
    temperatures=tuple(str(float(t)) for t in record['sampling']['temperatures'])
    first,last=temperatures[0],temperatures[-1]
    lines=[r'\subsection{GPT-5.6 Sol: temperature, pass@8 and verified modes}',r'\label{app:gpt56-temperature}',
        r'Figure~\ref{fig:gpt56-temperature-curve} plots empirical \texttt{pass@8}',
        r'against \texttt{distinct@8} for $T\in\{'+','.join(temperatures)+r'\}$, with reasoning',
        r'\texttt{none}. Each condition retains the same 480 held-out prompts',
        '(32 per domain--level cell), with eight stateless responses per prompt:',
        f"3,840 responses per temperature and {record['sampling']['total_responses']:,} in total. A prompt contributes one",
        r'to \texttt{pass@8} if at least one of its eight answers is verified correct,',
        r'and contributes its number of distinct correct canonical keys to \texttt{distinct@8}.',
        'Zero-success groups and all unsuccessful draws remain in the denominator.',
        r'We compute success directly from the saved groups, not as $1-(1-p)^8$',
        'from pooled per-response accuracy; task difficulty varies across prompts.', '',
        'The expanded cohort retains all 120 original prompts and their 4,800',
        'responses, adding 360 prompts and 14,400 responses. The extension follows',
        'the same outcome-independent row-hash ranking within each domain--level',
        'cell. Requests retain the frozen prompt bytes and verifier; only temperature',
        'varies in the sampling controls. New collection interleaves temperatures.',
        'Separate old-120 and new-360 analyses below assess cohort sensitivity;',
        'prompt composition and collection time differ between these cohorts.', '',
        'The deployment rejected non-default temperatures with medium reasoning',
        'and rejected temperature 2.5. These curves use the supported reasoning',
        r'\texttt{none} profile. Historical medium-reasoning measurements cover only',
        'the original 120 prompts and are omitted from the expanded-cohort tables',
        'and figure. This does not test temperature robustness of medium reasoning.',
        'A zero-temperature request does not establish deterministic outputs.', '',
        'Each level averages five domains equally; All averages all 15 cells.',
        'Pointwise 95 percent intervals use 20,000 paired whole-prompt bootstrap',
        'resamples stratified by domain and level, retaining all eight draws.',
        'There is no multiplicity adjustment. A zero-width empirical interval',
        'reflects constant observed groups, not certain success on unseen prompts.', '']
    for grade,title in [('strict','Strict executable grades'),('normalized_secondary','Frozen formatting-normalized grades')]:
        lines += [r'\begin{table}[!htbp]',r'\centering',r'\setlength{\parfillskip}{0pt plus .20\linewidth}',
            r'\caption{\textbf{GPT temperature frontier: '+title+r'.}',
            r'P is solved prompts (\%, \texttt{pass@8}); D is \texttt{distinct@8};',
            r'A is per-response accuracy (\%).',
            r'Brackets give pointwise 95\% intervals;',
            r'all temperatures use reasoning \texttt{none} and eight draws per prompt.}',
            r'\small',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{@{}llrrr@{}}',r'\toprule',
            r'$T$ & Level & P [95\% interval] & D [95\% interval] & A [95\% interval] \\',r'\midrule']
        for temperature in temperatures:
            groups=(record['matched_medium_reference']['analyses'][grade]['groups']['five_domain_macro'] if temperature=='M'
                    else record['analyses'][grade]['temperatures'][temperature]['groups']['five_domain_macro'])
            if temperature=='M':lines.append(r'\midrule')
            for level in (*LEVELS,'All'):
                group=groups['overall'] if level=='All' else groups['levels'][level]
                lines.append(' & '.join([temperature if level=='1' else '',level]+[interval(group[m],m) for m in ('pass8','distinct8','accuracy')])+r' \\')
        lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}','']
    normal=record['analyses']['normalized_secondary'];overall={t:b['groups']['five_domain_macro']['overall'] for t,b in normal['temperatures'].items()}
    best={metric:[t for t in temperatures if math.isclose(overall[t][metric]['estimate'],
          max(overall[u][metric]['estimate'] for u in temperatures),rel_tol=0,abs_tol=1e-12)]
          for metric in ('pass8','distinct8')}
    best_labels={metric:', '.join(f'$T={t}$' for t in winners) for metric,winners in best.items()}
    lines += [r'\paragraph{Observed temperature frontier.}',
        f'At $T={first}$ and $T={last}$, normalized '+r'\texttt{pass@8} is '
        +f"{overall[first]['pass8']['estimate']*100:.2f}"+r'\% and '+f"{overall[last]['pass8']['estimate']*100:.2f}"+r'\%, respectively,',
        r'while \texttt{distinct@8} is '+f"{overall[first]['distinct8']['estimate']:.3f}"+' and '+f"{overall[last]['distinct8']['estimate']:.3f}"+'.',
        f'Across these {len(temperatures)} measured temperatures, the highest observed aggregate',
        r'\texttt{pass@8} occurs at '+best_labels['pass8']+r' and the highest \texttt{distinct@8}',
        'occurs at '+best_labels['distinct8']+'. This finite exploratory sweep does not establish',
        'a population optimum or a continuous Pareto boundary, and level-specific curves differ.', '',
        r'Per-response accuracy changes from '+f"{overall[first]['accuracy']['estimate']*100:.2f}"+r'\% to '
        +f"{overall[last]['accuracy']['estimate']*100:.2f}"+r'\% between the endpoints.',
        r'That diagnostic is distinct from \texttt{pass@8}: success within eight',
        'draws and the reliability of an individual draw need not change together as temperature varies.', '']
    endpoint=normal['paired_endpoint_contrast']['groups']['five_domain_macro']['overall']
    lines += [f'The paired $T={last}$ minus $T={first}$ contrast is '
        +interval(endpoint['pass8'],'pass8')+r' \texttt{pass@8} percentage points and '
        +interval(endpoint['distinct8'],'distinct8')+r' distinct correct modes, with pointwise 95\% intervals.',
        'All prompt-level records, domain estimates, original per-response and',
        'collision analyses, and paired comparisons remain in the retained reports for both strict and normalized grading.', '',
        r'\paragraph{Retained outcomes.}']
    for t in temperatures:
        c=normal['temperatures'][t]['counts']
        lines.append(f'At $T={t}$, {c["truncated_responses"]} of {c["responses"]} outputs are token-limited and {c["native_refusals"]} are provider-declared refusals.')
    lines += ['These outcome counts retain every registered draw and are separate from',
        'the verifier-based correctness and canonical-key measurements above.', '']
    lines += render_cohort_sensitivity(record)
    return '\n'.join(lines)

def render_cohort_sensitivity(record):
    lines=[r'\paragraph{Sensitivity to the prompt expansion.}',
           'The original and additional prompt cohorts retain separate paired',
           'temperature comparisons. Differences between cohorts may reflect',
           'prompt composition or collection time; they are not causal time effects.', '']
    combined=record['analyses']['normalized_secondary']['paired_high_temperature_contrast']['groups']['five_domain_macro']['overall']
    lines.append(r'Across all 480 prompts, the paired $T=2$ minus $T=1.5$ contrast is '
                 +interval(combined['pass8'],'pass8')+r' \texttt{pass@8} percentage points and '
                 +interval(combined['distinct8'],'distinct8')+r' distinct modes (95\% intervals).')
    # Every expanded report must include the original and added cohorts. Exact
    # rows are rendered after report authentication, using the same metric units.
    for cohort,label in [('original_120','Original 120 prompts'),('additional_360','Additional 360 prompts')]:
        data=record['cohort_sensitivity'][cohort]
        analysis=data['analyses']['normalized_secondary']
        contrast=analysis['paired_high_temperature_contrast']['groups']['five_domain_macro']['overall']
        lines.append(label+r': the paired $T=2$ minus $T=1.5$ contrast is '
                     +interval(contrast['pass8'],'pass8')+r' \texttt{pass@8} percentage points and '
                     +interval(contrast['distinct8'],'distinct8')+r' distinct modes (95\% intervals).')
    return lines+['']

def render_appendix(record):
    if record['schema']=='paper-gpt56-pass8-temperature-curve-expanded480-v1':
        return render_appendix_expanded(record)
    return render_appendix_legacy(record)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args()
    result=render(build_record(args.source),args.output)
    (ROOT/'paper/results/gpt56_temperature_curve_20260911_appendix.tex').write_text(render_appendix(result))
    print(json.dumps({'figure':str(args.output),'points':len(result['display']['points']),'responses':result['sampling']['total_responses']}))

if __name__=='__main__':main()
