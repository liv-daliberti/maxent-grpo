#!/usr/bin/env python3
"""Restore semantic factorial detail using the current strict endpoint admission."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ops/exp_scaling'))
from build_paper_core_terminal_endpoints import sampled_endpoint,sha256
from build_paper_inference_followups import table
from scipy.stats import t
OUT=ROOT/'paper/results/semantic_current_summary_20260912'
DOMAINS=['graph_coloring','countdown','python_factors','mathir','pantry_plan']
LABELS=['Graph','Countdown','Python','MathIR','Pantry']
LEDGERS={
 'Qwen2.5-0.5B':['e78_verified_replay_only_05b_jobs.json','e81_semantic_maxent_verified_replay_05b_jobs.json','e83_semantic_maxent_without_replay_05b_jobs.json','e85_pantry_semantic_repair_jobs.json'],
 'Falcon3-1B':['e79_falcon1b_aligned_verified_replay_jobs.json','e82_falcon_semantic_maxent_verified_replay_jobs.json','e86_falcon_semantic_maxent_without_replay_jobs.json','e85_pantry_semantic_repair_jobs.json'],
 'Qwen2.5-3B':['e80r1_qwen3b_aligned_verified_replay_jobs.json','e87_qwen3b_semantic_maxent_seed70_jobs.json']}

def build():
 sources={};audits=[];models={}
 for model,names in LEDGERS.items():
  runs={}
  for name in names:
   path=ROOT/'var/artifacts'/name;sources[str(path.relative_to(ROOT))]=sha256(path);ledger=json.loads(path.read_text())
   assert ledger['target_steps']==3072
   for run in ledger['runs']:
    if model=='Qwen2.5-3B' and run['seed']!=70:continue
    if name.startswith(('e81_','e82_','e83_')) and run['domain']=='pantry_plan':continue
    if name.startswith('e85_') and run.get('parent') not in (('e81','e83') if model=='Qwen2.5-0.5B' else ('e82',)):continue
    arm=('semantic_only' if run['parent']=='e83' else 'semantic') if name.startswith('e85_') else run['arm']
    key=(run['domain'],arm,run['seed']);assert key not in runs,key;runs[key]=Path(run['run_dir'])
  cells=[]
  for domain in DOMAINS:
   arms=['control','replay','semantic']+([] if model=='Qwen2.5-3B' else ['semantic_only'])
   seeds=[70] if model=='Qwen2.5-3B' else list(range(43,48)) if model=='Qwen2.5-0.5B' else list(range(55,60))
   endpoints={};eligible=[]
   for seed in seeds:
    endpoints[str(seed)]={arm:sampled_endpoint(runs[(domain,arm,seed)],step=3072,audit=audits) for arm in arms}
    if all(x is not None for x in endpoints[str(seed)].values()):eligible.append(seed)
   assert eligible==([55,56,57,58] if model=='Falcon3-1B' and domain=='countdown' else seeds)
   contrasts={}
   for contrast in (['with_replay'] if model=='Qwen2.5-3B' else ['without_replay','with_replay','interaction']):
    metrics={}
    for metric in ['pass8','distinct8','extra8']:
     values={}
     for seed in eligible:
      e=endpoints[str(seed)]
      def val(arm):return e[arm]['distinct8']-e[arm]['pass8'] if metric=='extra8' else e[arm][metric]
      with_replay=val('semantic')-val('replay')
      without=val('semantic_only')-val('control') if 'semantic_only' in e else None
      values[str(seed)]=with_replay if contrast=='with_replay' else without if contrast=='without_replay' else with_replay-without
     mean=statistics.mean(values.values());half=float(t.ppf(.975,len(values)-1))*statistics.stdev(values.values())/math.sqrt(len(values)) if len(values)>1 else None
     metrics[metric]={'mean':mean,'ci95':[mean-half,mean+half] if half is not None else None,'per_seed':values}
    contrasts[contrast]=metrics
   cells.append({'domain':domain,'seeds':eligible,'endpoints':endpoints,'contrasts':contrasts})
  models[model]=cells
  print('Strict semantic endpoint audit complete:',model,flush=True)
 for audit in audits:
  if audit['status']=='excluded':
   sources[audit['source_log']]=audit['source_log_sha256'];sources[audit['amendment']]=audit['amendment_sha256']
  for row in audit.get('sources',[]):sources[str(Path(row['path']).relative_to(ROOT))]=row['sha256']
 core_path=ROOT/'paper/results/core_terminal_endpoints.json';core=json.loads(core_path.read_text());sources[str(core_path.relative_to(ROOT))]=sha256(core_path)
 for model,cells in models.items():
  for cell in cells:
   for seed in cell['seeds']:
    for arm in ['control','replay']:
     assert cell['endpoints'][str(seed)][arm]==core['models'][model]['domains'][cell['domain']]['methods'][arm]['per_seed'][str(seed)]
 sources['paper/preregistration/e85_pantry_semantic_repair_20260809.md']=sha256(ROOT/'paper/preregistration/e85_pantry_semantic_repair_20260809.md')
 return {'schema':'paper-semantic-current-summary-v1','source_sha256':sources,'endpoint_audit':audits,'models':models,
         'selection':'Current strict exact four-draw endpoint policy; common four-arm seed intersection. Falcon Countdown seed 59 excluded throughout, including semantic-only contrast. Qwen3B seed70 is descriptive and lacks semantic-only arm.'}

def render(report):
 tex=r'''\clearpage
\section{Fixed Semantic-Entropy Factorials}
\label{app:semantic-current-factorial}
A fixed semantic-entropy bonus can be applied with or without verified replay.
Pantry semantic arms use the registered E85 repairs: original E81/E82/E83
Pantry runs received zero semantic advantage because of an outcome-key defect
and remain excluded. The historical Falcon factorial display had not applied
that repair; no value from its superseded Pantry column is reused here.
The matched four-arm studies compare its effect on Dr.GRPO, its effect on
Re:Dr, and the difference of these effects. This restores the full
factorial detail beyond the main semantic-only comparator. We reconstruct
exact terminal four-draw endpoints under the current source-admission policy:
identical repeated metrics are deduplicated, unapproved conflicting retries
abort, and the previously excluded Falcon Countdown replay seed 59 remains
excluded. All three contrasts in that cell therefore use the common four
seeds 55--58; other 0.5B/1B cells use five. No old five-seed Falcon Countdown
interval is reused. Tables report paired seed means and pointwise Student-$t$
95\% intervals, without pooling models or domains.

The Qwen2.5-3B extension has only fixed seed 70 and no semantic-only arm.
Its on-replay comparison is descriptive, has no interval and cannot identify
a factorial interaction. These are fixed-coefficient experiments, not a
sweep over entropy strengths. Their endpoints do not establish that semantic
entropy reproduces the mechanism of a bank that revisits verified keys.
'''
 for model in LEDGERS:
  cells=report['models'][model]
  rows=[]
  for cell,domain in zip(cells,LABELS):
   for contrast in ('without_replay','with_replay','interaction'):
    if contrast not in cell['contrasts']:continue
    m=cell['contrasts'][contrast]
    values=[]
    for key in ['pass8','distinct8','extra8']:
     x=m[key];value=f"${x['mean']:.3f}$"
     if x['ci95'] is not None:value+=f" $[{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]$"
     values.append(value)
    rows.append([domain,len(cell['seeds']),{'without_replay':'Without replay','with_replay':'On replay','interaction':'Interaction'}[contrast],*values])
  tex+=table(model+r': fixed semantic-entropy effects. Without replay is semantic-only minus Dr.GRPO; on replay is semantic-plus-replay minus Re:Dr; interaction subtracts the first effect from the second. Extra modes equal distinct@8 minus pass@8.',
             'tab:semantic-current-'+model.replace('.','').replace('-','').lower(),['Domain',r'$n$','Contrast',r'$\Delta$ pass@8',r'$\Delta$ distinct@8',r'$\Delta$ extra modes'],rows,'lr l rrr')
 tex+=r'''\path{results/semantic_current_summary_20260912.json} retains per-seed
endpoints, source hashes, exclusion evidence and all three metrics.
\path{ops/build_paper_semantic_current_summary.py} reconstructs these contrasts
with the same strict endpoint reader used by the current core comparison, including all documented exclusions.
'''
 return tex

def main():
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
 if args.check:
  report=json.loads(OUT.with_suffix('.json').read_text())
  for name,h in report['source_sha256'].items():assert sha256(ROOT/name)==h,name
  assert OUT.with_suffix('.tex').read_text()==render(report)
  print('Semantic summary source hashes and rendering match.');return
 report=build();OUT.with_suffix('.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');OUT.with_suffix('.tex').write_text(render(report))
 print('Wrote current-admission semantic summary.',flush=True)
if __name__=='__main__':main()
