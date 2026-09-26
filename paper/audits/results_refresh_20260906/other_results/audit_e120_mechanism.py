from pathlib import Path
import json,math,concurrent.futures,datetime
ROOT=Path.cwd();OUT=ROOT/'paper/audits/results_refresh_20260906/other_results/e120_mechanism.json'
l=json.loads((ROOT/'var/artifacts/e120r1_frequency_weighted_replay_jobs.json').read_text())
required={'canonical_replay_key_weighting_frequency':1.,'canonical_replay_frequency_count_fresh_only':1.,'canonical_replay_frequency_count_from_replay':0.,'canonical_replay_frequency_count_from_proposals':0.,'canonical_replay_score_passes':2.,'canonical_replay_alpha_used':.1,'canonical_replay_objective_scale':1/16,'canonical_replay_reward_estimator_scale':15/16,'canonical_replay_global_groups_per_step':1.}
def audit(run):
 root=Path(run['run_dir']);c=json.loads((root/'TRAINING_COMPLETE.json').read_text());a=Path(c['terminal_attempt']).resolve();a.relative_to(root.resolve());p=a/'train_metrics.jsonl';n=0;nr=0;bad=[];steps=set()
 def finite(v):return isinstance(v,(float,int)) and math.isfinite(v)
 with p.open() as f:
  for line in f:
   if not line.strip():continue
   try:r=json.loads(line)
   except ValueError:bad.append({'line':n+1,'issue':'invalid JSON'});continue
   n+=1;d={k.split('/')[-1]:v for k,v in r.items()};steps.add(d.get('step'))
   if 'canonical_replay_key_weighting_frequency' not in d:continue
   nr+=1;issues=[]
   for k,v in required.items():
    if not finite(d.get(k)) or not math.isclose(d[k],v,rel_tol=0,abs_tol=1e-7):issues.append(k)
   modes=d.get('canonical_replay_actuator_modes');total=d.get('canonical_replay_target_weight_sum');weights={k.split('canonical_replay_target_weight_row_')[-1]:v for k,v in d.items() if k.startswith('canonical_replay_target_weight_row_')};counts={k.split('canonical_replay_fresh_count_row_')[-1]:v for k,v in d.items() if k.startswith('canonical_replay_fresh_count_row_')};keys={k.split('canonical_replay_outcome_fingerprint_row_')[-1]:v for k,v in d.items() if k.startswith('canonical_replay_outcome_fingerprint_row_')}
   if not weights or set(weights)!=set(counts) or set(weights)!=set(keys):issues.append('row_identity_alignment')
   if any(not finite(v) or v<=0 for v in weights.values()):issues.append('positive_finite_weights')
   if any(not finite(v) or v<1 for v in counts.values()):issues.append('positive_fresh_counts')
   if not finite(modes) or not finite(total) or abs(total-modes)>1e-5 or abs(sum(weights.values())-modes)>1e-5:issues.append('bank_budget')
   if d.get('canonical_replay_eligible_groups',0)>0 and d.get('canonical_replay_actuator_groups')!=1.:issues.append('one_selected_prompt')
   if issues and len(bad)<20:bad.append({'line':n,'step':d.get('step'),'issues':issues})
 return {'run_dir':str(root),'domain':run['domain'],'seed':run['seed'],'terminal_attempt':str(a),'metrics':str(p),'bytes':p.stat().st_size,'rows':n,'replay_rows':nr,'distinct_step_count':len(steps),'passed':nr>0 and not bad,'violations':bad}
runs=[r for r in l['runs'] if (Path(r['run_dir'])/'TRAINING_COMPLETE.json').exists()];out={'asof':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'All rows of terminal-attempt metrics only; prior attempts not audited. No efficacy fields inspected. This does not independently reconstruct fingerprint identities or replay losses from raw trajectories.','source_gate':l['source_gate'],'runs':[]}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
 for x in ex.map(audit,runs):
  out['runs'].append(x);OUT.write_text(json.dumps(out,indent=2)+'\n');print(x['seed'],x['domain'],x['rows'],x['replay_rows'],x['passed'],x['violations'][:1],flush=True)
out['complete']=True;out['passed']=all(x['passed'] for x in out['runs']);OUT.write_text(json.dumps(out,indent=2)+'\n')
