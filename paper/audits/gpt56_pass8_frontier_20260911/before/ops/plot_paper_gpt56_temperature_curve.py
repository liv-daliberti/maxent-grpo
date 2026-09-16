#!/usr/bin/env python3
"""Source-bound accuracy/breadth curves for four GPT-5.6 Sol temperatures.

All connected points use reasoning=none and the same complete fixed subset.
No original medium-reasoning reference is inserted into these curves.
"""
from pathlib import Path
import argparse
from copy import deepcopy
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/frontier_temperature_20260911/GPT56_TEMPERATURE_CURVE.json'
OUTPUT=ROOT/'paper/figures/gpt56_temperature_curve'
TEMPERATURES=('0.5','1.0','1.5','2.0')
LEVELS=('1','2','3')
COLORS=('#00509E','#C76A3A','#7B1FA2')
MARKERS=('o','s','^')
FIGSIZE=(6.0,3.25)

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
    if (analysis.get('schema')!='gpt56-none-temperature-curve-v1' or analysis.get('status')!='complete'
            or analysis.get('model')!='gpt-5.6-sol' or analysis.get('reasoning_effort')!='none'):
        raise ValueError('The plotted GPT curve must be complete and use reasoning=none throughout.')
    required={'all_3840_native_receipts_authenticated','same_120_prompts_all_eight_slots',
              'only_temperature_varies_within_curve','all_returned_controls_match_requests',
              'source_summaries_reconstructed'}
    if not all(analysis.get('validation',{}).get(key) is True for key in required):
        raise ValueError('The full fixed GPT temperature cohort has not passed its independent audit.')
    if {float(t) for t in analysis.get('temperatures',[])}!={.5,1.,1.5,2.}:
        raise ValueError('The GPT curve requires all four registered temperatures.')
    bindings=authenticate(analysis)
    normalized=analysis['analyses']['normalized_secondary']['temperatures']
    if set(normalized)!=set(TEMPERATURES):raise ValueError('Incomplete normalized temperature grid.')
    points=[]
    for level in LEVELS:
        for temperature in TEMPERATURES:
            group=normalized[temperature]['groups']['five_domain_macro']['levels'][level]
            metrics={}
            for metric,maximum in [('accuracy',1.),('distinct8',8.)]:
                item=group[metric];value=item['estimate']
                if (not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=maximum
                        or item.get('defined_replicates')!=20000 or not isinstance(item.get('ci95'),list)
                        or len(item['ci95'])!=2):
                    raise ValueError(f'{level}/{temperature}/{metric}: invalid complete-cohort metric.')
                metrics[metric]=deepcopy(item)
            points.append({'level':int(level),'temperature':float(temperature),'metrics':metrics,
                           'source_group':f'/analyses/normalized_secondary/temperatures/{temperature}/groups/five_domain_macro/levels/{level}'})
    reference=analysis['matched_medium_reference']
    if (reference['reasoning_effort']!='medium' or reference['requested_temperature'] is not None
            or reference['returned_temperature']!=1.0 or reference['connect_to_temperature_curve'] is not False):
        raise ValueError('The matched medium reference must remain separate from controlled none-reasoning curves.')
    reference_points=[]
    for level in LEVELS:
        group=reference['analyses']['normalized_secondary']['groups']['five_domain_macro']['levels'][level]
        reference_points.append({'level':int(level),'metrics':{m:deepcopy(group[m]) for m in ('accuracy','distinct8')},
                                 'source_group':f'/matched_medium_reference/analyses/normalized_secondary/groups/five_domain_macro/levels/{level}',
                                 'connected':False})
    return {'schema':'paper-gpt56-temperature-curve-v1','source':{'path':relative(source),'sha256':sha(source)},
            'renderer':{'path':relative(__file__),'sha256':sha(__file__)},'authenticated_bindings':bindings,
            'model':'gpt-5.6-sol','reasoning_effort':'none','grading':'frozen formatting-normalized',
            'sampling':{'temperatures':[.5,1.,1.5,2.],'prompts_per_domain_level':8,'domains':5,
                        'levels':[1,2,3],'draws_per_prompt':8,'responses_per_temperature':960,'total_responses':3840},
            'display':{'x':'per-response accuracy (%)','y':'distinct correct modes per eight draws',
                       'aggregation':'Equal mean over all five domains within each level.',
                       'connections':'Straight segments in ascending requested temperature; no fit or Pareto claim.',
                       'points':points,'figure_inches':list(FIGSIZE),'pointwise_intervals':'Retained in the source record and appendix; main plot shows point estimates.'},
            'conditions':analysis['conditions'],'bootstrap':analysis['bootstrap'],'validation':analysis['validation'],
            'matched_medium_reference':deepcopy(reference),'reference_points':reference_points,
            'analyses':{grade:{'temperatures':{t:{'groups':block['groups'],'counts':block['counts']} for t,block in data['temperatures'].items()},
                               'paired_endpoint_contrast':data['paired_endpoint_contrast']} for grade,data in analysis['analyses'].items()}}

def build_figure(record):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rc={'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'xtick.labelsize':9,
        'ytick.labelsize':9,'pdf.fonttype':42,'ps.fonttype':42,'text.color':'#19324A',
        'axes.labelcolor':'#19324A','xtick.color':'#607487','ytick.color':'#607487'}
    with plt.rc_context(rc):
        fig,ax=plt.subplots(figsize=FIGSIZE)
        fig.subplots_adjust(left=.13,right=.97,bottom=.18,top=.80)
        xs=[];ys=[]
        for level,color,marker in zip((1,2,3),COLORS,MARKERS):
            points=[p for p in record['display']['points'] if p['level']==level]
            points.sort(key=lambda p:p['temperature'])
            x=[p['metrics']['accuracy']['estimate']*100 for p in points]
            y=[p['metrics']['distinct8']['estimate'] for p in points]
            xs.extend(x);ys.extend(y)
            ax.plot(x,y,color=color,marker=marker,markersize=5,linewidth=1.2,label=f'Level {level}')
            offsets={1:[(10,-10),(10,8),(-24,-12),(-24,10)],
                     2:[(0,-12),(0,-21),(-17,-17),(-25,12)],
                     3:[(8,-6),(8,-20),(9,12),(-26,12)]}[level]
            for point,px,py,offset in zip(points,x,y,offsets):
                ax.annotate(f'{point["temperature"]:g}',(px,py),xytext=offset,textcoords='offset points',
                            color=color,fontsize=8.5,ha='left',va='center',
                            bbox={'boxstyle':'round,pad=.12','facecolor':'white','edgecolor':'none','alpha':.9},
                            arrowprops={'arrowstyle':'-','color':color,'linewidth':.55},zorder=5)
        for point,color,marker in zip(record['reference_points'],COLORS,MARKERS):
            px=point['metrics']['accuracy']['estimate']*100;py=point['metrics']['distinct8']['estimate']
            ax.scatter([px],[py],s=50,marker=marker,facecolors='white',edgecolors=color,linewidths=1.4,zorder=6)
            xs.append(px);ys.append(py)
        xpad=max((max(xs)-min(xs))*.10,1.5);ypad=max((max(ys)-min(ys))*.23,.12)
        ax.set_xlim(max(0,min(xs)-xpad),min(100,max(xs)+xpad))
        ax.set_ylim(max(0,min(ys)-ypad),max(ys)+ypad)
        ax.set_xlabel('Per-response accuracy (%)')
        ax.set_ylabel('Distinct correct modes / 8')
        ax.grid(color='#D8E2EA',linewidth=.6,alpha=.75)
        for side in ('top','right'):ax.spines[side].set_visible(False)
        for side in ('bottom','left'):ax.spines[side].set_color('#AABAC7')
        ax.legend(loc='upper left',bbox_to_anchor=(0,1.25),ncol=3,frameon=False,
                  handlelength=1.5,columnspacing=1.4,handletextpad=.5)
        fig.text(.97,.93,'GPT-5.6 Sol',ha='right',va='center',fontsize=9,fontweight='bold')
        fig.text(.97,.85,'Filled + lines: none | Open: historical medium',ha='right',va='center',fontsize=8,color='#607487')
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

def render_appendix(record):
    lines=[r'\subsection{GPT-5.6 Sol: a controlled temperature curve}',r'\label{app:gpt56-temperature}',
        r'Figure~\ref{fig:gpt56-temperature-curve} uses four separate fixed conditions at',
        r'$T\in\{0.5,1.0,1.5,2.0\}$, all with reasoning \texttt{none}. Each has',
        'the same 120 held-out prompts (eight from each domain--level cell) and eight',
        'stateless samples per prompt, giving 960 responses per temperature and 3,840',
        'new responses in total. Native receipts authenticate the requested and returned',
        'temperature and reasoning controls. Only temperature varies within this curve;',
        'all completed responses, including incorrect and token-limited ones, remain',
        'in the fixed denominator.', '',
        'The deployment rejects non-default temperatures with medium reasoning.',
        'The connected curve therefore uses a distinct, supported no-reasoning profile.',
        'It does not establish temperature robustness of the original medium-reasoning',
        'result. The unconnected reference reuses the same 120 prompts and all eight',
        'slots from the historical original cohort: its requests omitted temperature',
        r'and its native receipts report $T=1.0$, reasoning \texttt{medium}. It was',
        'collected earlier, so this is a descriptive reference rather than a randomized',
        'causal comparison of reasoning settings.', '',
        'Each level averages all five domains equally; All averages all 15 cells.',
        'Intervals use 20,000 whole-prompt bootstrap resamples stratified by domain',
        'and level, keeping all eight draws together. Temperature contrasts pair task',
        'rows, not individual generated responses. Intervals are pointwise, without',
        'multiplicity adjustment. Undefined collision cells make the macro undefined;',
        'no cell is dropped to manufacture a complete macro. An interval is omitted if',
        'any bootstrap replicate has undefined collision; the source record retains',
        'the exact number of defined replicates for every cell and macro.', '']
    for grade,title in [('strict','Strict executable grades'),('normalized_secondary','Frozen formatting-normalized grades')]:
        lines += [r'\begin{table}[!htbp]',r'\centering',r'\setlength{\parfillskip}{0pt plus .20\linewidth}',r'\caption{\textbf{GPT temperature curve: '+title+r'.}',
            r'A is per-response accuracy (\%), D is \texttt{distinct@8}, and C is correct-pair',
            r'collision (\%). Brackets give pointwise 95\% intervals. M denotes the separate',
            r'medium-reasoning reference. All numeric-temperature rows use the same reasoning \texttt{none} profile.',
            r'Every fixed draw remains in its original denominator.}',
            r'\small',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{@{}llrrr@{}}',r'\toprule',
            r'$T$ & Level & A [95\% interval] & D [95\% interval] & C [95\% interval] \\',r'\midrule']
        for temperature in (*TEMPERATURES,'M'):
            groups=(record['matched_medium_reference']['analyses'][grade]['groups']['five_domain_macro'] if temperature=='M'
                    else record['analyses'][grade]['temperatures'][temperature]['groups']['five_domain_macro'])
            if temperature=='M':lines.append(r'\midrule')
            for level in (*LEVELS,'All'):
                group=groups['overall'] if level=='All' else groups['levels'][level]
                lines.append(' & '.join([temperature if level=='1' else '',level]+[interval(group[m],m) for m in ('accuracy','distinct8','collision')])+r' \\')
        lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}','']
    lines += [r'\paragraph{Retained outcomes and tradeoff.}']
    for t in TEMPERATURES:
        counts=record['analyses']['normalized_secondary']['temperatures'][t]['counts']
        lines.append(f'At $T={t}$, {counts["truncated_responses"]} of {counts["responses"]} outputs are token-limited and {counts["native_refusals"]} are provider-declared refusals.')
    normal=record['analyses']['normalized_secondary'];start=normal['temperatures']['0.5']['groups']['five_domain_macro']['overall'];end=normal['temperatures']['2.0']['groups']['five_domain_macro']['overall']
    lines += ['Across all cells, normalized accuracy changes from '+f'{start["accuracy"]["estimate"]*100:.2f}'+r'\% to '+f'{end["accuracy"]["estimate"]*100:.2f}'+r'\%, and breadth changes from '+f'{start["distinct8"]["estimate"]:.3f}'+' to '+f'{end["distinct8"]["estimate"]:.3f}'+r' modes. This profile has substantially lower accuracy than the historical medium-reasoning reference.',
              'All per-domain estimates, returned control metadata, exact prompt identities,',
              'refusals, truncations, and paired contrasts remain in the source-bound report.', '']
    endpoint=normal['paired_endpoint_contrast']['groups']['five_domain_macro']['overall']
    lines += ['The paired $T=2.0$ minus $T=0.5$ endpoint contrast is '
              +interval(endpoint['distinct8'],'distinct8')+' modes and '
              +interval(endpoint['accuracy'],'accuracy')+r' accuracy percentage points, quantifying the measured tradeoff at reasoning \texttt{none} across the same fixed task subset.', '']
    return '\n'.join(lines)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args()
    result=render(build_record(args.source),args.output)
    (ROOT/'paper/results/gpt56_temperature_curve_20260911_appendix.tex').write_text(render_appendix(result))
    print(json.dumps({'figure':str(args.output),'points':len(result['display']['points']),'responses':3840}))

if __name__=='__main__':main()
