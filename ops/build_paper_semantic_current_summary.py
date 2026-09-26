#!/usr/bin/env python3
"""Render fixed semantic-entropy effects with and without verified replay."""
import argparse
import json
import math
from pathlib import Path
try:
    from paper_domain_typography import format_domain_names
except ModuleNotFoundError:
    from ops.paper_domain_typography import format_domain_names
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ops/exp_scaling'))
from build_paper_core_terminal_endpoints import sampled_endpoint,sha256
from build_paper_inference_followups import table
from scipy.stats import t
OUT=ROOT/'paper/results/semantic_current_summary_20260912'
DOMAINS=['graph_coloring','countdown','python_factors','mathir','pantry_plan']
LABELS=['Graph','Countdown','Python','MathIR','PantryPlan']
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
 tex=r'''\section{Fixed Semantic-MaxEnt Comparisons}
\label{app:semantic-current-factorial}
A four-method factorial crosses replay with a fixed Semantic-MaxEnt bonus,
comparing Dr.GRPO and Re:Dr with and without the bonus across five domains
at Qwen2.5-0.5B and Falcon3-1B. All three contrasts within a model--domain
combination use the same training seeds: five in nine combinations and four
for Falcon Countdown.
Table~\ref{tab:semantic-current} reports terminal paired means after eight
training passes, with nominal pointwise 95\% Student-$t$ intervals using
$n-1$ degrees of freedom. These intervals describe training-seed variability
on the fixed evaluation prompts and are unadjusted across comparisons.

At Qwen2.5-3B, one training seed compares Re:Dr with and without the bonus across
all five domains. No uncertainty interval is available from a single seed.
The absence of a semantic-only comparison also prevents estimating the
interaction between replay and the bonus. These results characterize the
evaluated coefficients rather than an optimal entropy strength. Extra-mode
counts depend on correctness as well as diversity and do not measure
success-conditional solution diversity.
'''
 # One table for the three scales. They carried identical captions and column
 # sets, differing only in the model name, so splitting them restated the
 # contrast definitions once per scale.
 rows=[]
 for index,model in enumerate(LEDGERS):
  cells=report['models'][model]
  if index:rows.append(None)
  first=True
  for cell,domain in zip(cells,LABELS):
   for contrast in ('without_replay','with_replay','interaction'):
    if contrast not in cell['contrasts']:continue
    m=cell['contrasts'][contrast]
    values=[]
    for key in ['pass8','distinct8','extra8']:
     x=m[key];value=f"${x['mean']:.3f}$"
     if x['ci95'] is not None:value+=f" $[{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]$"
     values.append(value)
    rows.append([model if first else '',domain,len(cell['seeds']),{'without_replay':'Without replay','with_replay':'On replay','interaction':'Interaction'}[contrast],*values])
    first=False
 tex+=table(r'\textbf{Fixed Semantic-MaxEnt increases verified-mode counts on Qwen2.5-0.5B PantryPlan with and without replay.} Without replay is semantic-only minus Dr.GRPO; on replay is semantic-plus-replay minus Re:Dr; interaction subtracts the former effect from the latter. Extra modes equal \texttt{distinct@8} minus \texttt{pass@8}. Entries give paired means and nominal 95\% Student-$t$ intervals; the single-seed 3B effects have no intervals.',
            'tab:semantic-current',['Model','Domain',r'$n$','Contrast',r'$\Delta$ pass@8',r'$\Delta$ distinct@8',r'$\Delta$ extra modes'],rows,'llr l rrr').replace(r'\setlength{\tabcolsep}{3pt}', r'\setlength{\tabcolsep}{2.4pt}')
 return format_domain_names(tex)

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
