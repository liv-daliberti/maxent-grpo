#!/usr/bin/env python3
"""Plot all registered fresh concentration contrasts from a complete report."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SCALES=('qwen05b','falcon1b','qwen3b')
DOMAINS=('graph_coloring','pantry_plan')
SCALE_LABELS={'qwen05b':'Qwen2.5–0.5B','falcon1b':'Falcon3–1B','qwen3b':'Qwen2.5–3B'}
DOMAIN_LABELS={'graph_coloring':'Graph','pantry_plan':'PantryPlan'}
TRAINING=(('drgrpo_minus_initial','Dr.GRPO','#7755a2'),
          ('replay_drgrpo_minus_initial','Re:Dr.GRPO','#12867d'),
          ('maxrl_minus_initial','MaxRL','#cc7b22'),
          ('replay_maxrl_minus_initial','Re:MaxRL','#2673b4'))
REPLAY=(('replay_drgrpo_minus_drgrpo','Dr.GRPO + replay','#12867d'),
        ('replay_maxrl_minus_maxrl','MaxRL + replay','#2673b4'))

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def plot_report(report_path,output):
    report_path=Path(report_path).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('figure output exists; refusing to overwrite')
    report=json.loads(report_path.read_text());audit=report['completeness_audit']
    if report.get('status')!='complete' or audit.get('authenticated_tasks')!=150 or audit.get('authenticated_response_slots')!=1228800:
        raise ValueError('complete authenticated 150-task panel required')
    contrasts={(c['model_scale'],c['domain'],c['contrast']):c for c in report['contrasts']}
    expected={(s,d,key) for s in SCALES for d in DOMAINS for key,_,_ in TRAINING+REPLAY}
    if set(contrasts)!=expected or len(report['contrasts'])!=36:raise ValueError('exactly all36 registered contrasts required')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent,prefix='.fresh-plots-') as temporary:
        stage=Path(temporary)
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                             'axes.spines.right':False,'axes.titleweight':'bold','pdf.fonttype':42,'ps.fonttype':42})
        for name,rows,title,xlabel in (
            ('training_concentration',TRAINING,'Fresh evaluation: concentration before and after training','Change in correct-key collision (trained − initial)'),
            ('replay_concentration',REPLAY,'Fresh evaluation: the effect of adding replay','Change in correct-key collision (replay − control)')):
            fig,axes=plt.subplots(2,3,figsize=(11,6.8 if len(rows)==4 else 5.2),sharex=True,sharey=True)
            values=[0.0]
            for scale in SCALES:
                for domain in DOMAINS:
                    for key,_,_ in rows:
                        c=contrasts[scale,domain,key];effect=c['summary']['joint_population_effects']['collision']
                        values.extend(v for v in [effect['mean'],*(effect['ci95'] or [])] if v is not None)
                        values.extend(r['populations']['joint_R_ge_2']['delta']['collision'] for r in c['seeds'] if r['defined'])
            low,high=min(values),max(values);span=max(high-low,.2);limits=(low-.06*span,high+.08*span)
            for di,domain in enumerate(DOMAINS):
                for si,scale in enumerate(SCALES):
                    ax=axes[di,si];ax.axvline(0,color='#888888',lw=.8,zorder=1)
                    ax.grid(axis='x',color='#eeeeee',lw=.8);ax.set_axisbelow(True)
                    ax.set_title(f'{DOMAIN_LABELS[domain]} · {SCALE_LABELS[scale]}',fontsize=10,pad=12)
                    for ri,(key,label,color) in enumerate(rows):
                        y=len(rows)-ri-1;c=contrasts[scale,domain,key];summary=c['summary']
                        effect=summary['joint_population_effects']['collision'];mean=effect['mean'];ci=effect['ci95']
                        for offset,seed in zip(np.linspace(-.07,.07,5),c['seeds']):
                            value=seed['populations']['joint_R_ge_2']['delta']['collision']
                            if value is not None:ax.scatter(value,y+offset,s=14,color='#b9bdc2',zorder=2)
                        if mean is None:
                            ax.text(0,y,'undefined',fontsize=8,ha='center',va='center',color='#777777',bbox={'facecolor':'white','edgecolor':'none','pad':1.2})
                        else:
                            if ci is not None:ax.plot(ci,[y,y],lw=2.0,color=color,zorder=3)
                            ax.scatter(mean,y,s=42,facecolors=color if summary['n_defined']==5 else 'white',edgecolors=color,lw=1.4,zorder=4)
                        counts=list(summary['eligible_prompt_counts'].values())
                        ax.text(.98,y-.23,f"{summary['n_defined']}/5 seeds; {min(counts)}–{max(counts)} joint prompts",
                                transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=6.8,color='#555555')
                    ax.set_yticks(range(len(rows)),[r[1] for r in reversed(rows)])
                    ax.set_ylim(-.55,len(rows)-.45);ax.set_xlim(limits)
                    ax.tick_params(axis='y',length=0)
            fig.suptitle(title,fontsize=14,fontweight='bold',y=.98)
            fig.supxlabel(xlabel,fontsize=10,y=.115)
            fig.text(.5,.064,'Dots: equal-prompt means, then equal seed weights. Gray dots: individual seed contrasts. Bars: nominal 95% paired t intervals.',ha='center',fontsize=8)
            fig.text(.5,.035,'128 fixed prompts per domain; 64 fresh draws per policy/prompt. Joint eligibility requires at least two correct draws in both conditions.',ha='center',fontsize=8)
            fig.text(.5,.009,'Initial references are five sampling replicas of the same weights. Open dots indicate incomplete seed coverage; undefined contrasts are retained.',ha='center',fontsize=7.8)
            fig.subplots_adjust(left=.14,right=.985,bottom=.19,top=.875,hspace=.45,wspace=.24)
            for suffix in ('pdf','png'):fig.savefig(stage/f'{name}.{suffix}',dpi=220,bbox_inches='tight')
            plt.close(fig)
        shutil.copy2(__file__,stage/Path(__file__).name)
        manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'report':str(report_path),
                  'report_sha256':digest(report_path),'contrasts_plotted':36,'matplotlib':matplotlib.__version__,
                  'files':{p.name:digest(p) for p in stage.iterdir() if p.is_file()}}
        (stage/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
        stage.rename(output)
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();print(plot_report(a.report,a.output))
