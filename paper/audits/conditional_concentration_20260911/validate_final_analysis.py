#!/usr/bin/env python3
"""Cross-check reported claims and artifacts against the frozen analysis."""
from pathlib import Path
import hashlib,json,runpy
ROOT=Path(__file__).resolve().parents[3]
AUDIT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=ROOT/'paper/results/conditional_concentration_20260911.json';r=json.loads(p.read_text())
assert r['schema']=='paper-conditional-concentration-v1' and len(r['blocks'])==135
assert sum(r['cache']['cells_by_method'].values())==475
assert sum(r['cache']['available_checkpoints'].values())==892
assert sha(ROOT/r['cache']['path'])==r['cache']['sha256']
assert r['analysis_code_sha256']==sha(ROOT/'ops/exp_scaling/analyze_paper_conditional_concentration.py')
assert not r['stream_source_audit']['unsupported_checkpoint_mappings']
assert r['stream_source_audit']['stream_counts_per_prompt_checkpoint']=={'11':892*128}
assert r['cohort_extension']['expected_primary_level1_pairs']=={'drgrpo':74,'maxrl':75}
for b in r['blocks']:
 for view in ('distinct_streams','orientation0','orientation1','naive_reused_streams_32'):
  s=b['summaries'][view]
  assert len(s['values'])==s['n']
  assert (s['ci95'] is not None)==(s['n']==5)
  assert (s['mean'] is not None)==(s['n']>0)
 for seed,v in b['per_seed'].items():
  for view in ('distinct_streams','orientation0','orientation1'):
   a=v[view]
   assert a['n_total']==128 and a['n_eligible']==len(a['eligible_ids'])
   assert sum(a['eligibility'].values())==128
   assert (a['delta']['collision'] is not None)==(a['n_eligible']>0)
   if view!='distinct_streams':assert a['stream_selection']['disjoint_in_every_prompt']
longitudinal=[b for b in r['blocks'] if b['kind']=='before_after' and b['level']=='level1' and b['method'] in ('drgrpo','grpo') and b['domain'] in ('graph_coloring','pantry_plan')]
replay=[b for b in r['blocks'] if b['kind']=='replay_effect' and b['level']=='level1' and b['method']=='drgrpo' and b['domain'] in ('graph_coloring','pantry_plan')]
assert len(longitudinal)==12 and all(b['summaries']['distinct_streams']['ci95'][0]>0 for b in longitudinal)
assert all(b['summaries']['distinct_streams']['original_metrics_on_eligible']['mean8']['mean']>0 for b in longitudinal)
assert sum(all(b['summaries'][v]['mean']>0 for v in ('orientation0','orientation1')) for b in longitudinal)==11
assert len(replay)==6 and all(b['summaries']['distinct_streams']['ci95'][1]<0 for b in replay)
assert all(all(b['summaries'][v]['mean']<0 for v in ('orientation0','orientation1')) for b in replay)
assert all(b['summaries']['distinct_streams']['original_metrics_on_eligible']['mean8']['mean']<0 for b in replay)
for b in r['blocks']:
 if b['kind']=='before_after' and b['level']=='level1' and b['method'] in ('drgrpo','grpo') and b['domain']=='python_factors':assert b['summaries']['distinct_streams']['n']==0
builder=runpy.run_path(str(ROOT/'ops/exp_scaling/build_paper_conditional_concentration_appendix.py'))
assert builder['render'](r)+'\n'==(ROOT/'paper/results/conditional_concentration_20260911_tables.tex').read_text()
receipt={'status':'pass','result_sha256':sha(p),'source_cells':475,'available_checkpoints':892,'reported_blocks':135,'longitudinal_graph_pantry_positive':12,'longitudinal_both_splits_positive':11,'dr_replay_graph_pantry_negative_both_splits':6,'artifact_bindings':{str(q.relative_to(ROOT)):sha(q) for q in (p,ROOT/'paper/results/conditional_concentration_20260911_appendix.tex',ROOT/'paper/results/conditional_concentration_20260911_tables.tex')}}
(AUDIT/'analysis_claim_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
