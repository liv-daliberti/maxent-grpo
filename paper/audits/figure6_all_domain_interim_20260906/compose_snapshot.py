"""Freeze selected draw records plus outcome-free complete availability."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[3]
AUDIT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'ops/exp_scaling'))
from build_paper_modebench_level_comparison import LEVELS,DOMAINS,METHODS,SEEDS,build_interim_comparison
paths={'level1':AUDIT/'level1_coverage_snapshot.json','level2':ROOT/'var/artifacts/e119_figure6_coverage_20260906/snapshot.json'}
raw={level:json.loads(path.read_text()) for level,path in paths.items()}
assert hashlib.sha256(paths['level2'].read_bytes()).hexdigest()=='6016f2464643a9352860ac0ffc7f1658587589e108b0b6e8af0e7c3e22f54c75'
index={}
availability=[]
for level,record in raw.items():
 for cell in record['cells']:
  method=cell.get('method',cell.get('arm'))
  key=(level,cell['domain'],method,int(cell['seed']))
  assert key not in index
  index[key]=cell
  availability.append(dict(level=level,domain=cell['domain'],method=method,seed=int(cell['seed']),
   complete_steps=cell['complete_steps'],invalid_or_conflicted_steps=cell['invalid_or_conflicted_steps'],
   source_files=cell['source_files'],run_dir=cell['run_dir']))
assert len(index)==200
chosen={}
for domain in DOMAINS:
 for seed in SEEDS:
  sets=[set(index[level,domain,method,seed]['complete_steps']) for level in LEVELS for method in METHODS]
  common=set.intersection(*sets)
  if common:chosen[domain,seed]=max(common)
assert all(any(domain==d for d,s in chosen) for domain in DOMAINS)

def normalized_rows(level,domain,method,seed,step):
 checkpoint=index[level,domain,method,seed]['complete_checkpoints'][str(step)]
 assert checkpoint['draw_count']==4
 rows=[]
 for draw in checkpoint['draws']:
  meta=draw['metadata']
  assert meta['prompt_count']==128 and meta['sample_count']==8 and meta['temperature']==1
  rows.append(dict(level=level,domain=domain,method=method,seed=seed,step=step,
   draw_index=draw['draw_index'],evaluation_kind=meta['evaluation_kind'],sample_count=meta['sample_count'],
   metrics=draw['metrics'],evaluation_metadata=meta,origins=draw['origins']))
 return rows

evaluations=[]
for (domain,seed),step in chosen.items():
 for level in LEVELS:
  for method in METHODS:evaluations.extend(normalized_rows(level,domain,method,seed,step))
terminal=[]
for (level,domain,method,seed),cell in index.items():
 if level=='level2' and 3072 in cell['complete_steps']:
  terminal.extend(normalized_rows(level,domain,method,seed,3072))
reference=json.loads((AUDIT/'before/paper/figures/modebench_level_admission.json').read_text())
snapshot=dict(schema='modebench-level-comparison-frozen-snapshot-v1',collected_at_utc=datetime.now(timezone.utc).isoformat(),
 selection_policy='Latest observed shared nonnegative checkpoint for every domain/seed across all8 series; actual valid step0 allowed based on coverage before inspecting effects. All5domains must have observations. No missing-value imputation, no substitution of admission estimates, no retry conflict selection.',
 input_snapshots={level:dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for level,path in paths.items()},
 collection_intervals={'level1':{'start':raw['level1']['started_at_utc'],'end':raw['level1']['finished_at_utc']},
                       'level2':{'start':raw['level2']['collection_started_utc'],'end':raw['level2']['collection_updated_utc']}},
 reference_figure=reference,availability=availability,evaluations=evaluations,level2_terminal_evaluations=terminal)
result=build_interim_comparison(evaluations,availability)
output=ROOT/'paper/results/modebench_level_comparison_snapshot.json'
output.write_text(json.dumps(snapshot,indent=2,sort_keys=True)+'\n')
(AUDIT/'selection_summary.json').write_text(json.dumps({key:result[key] for key in ['coverage_by_domain','eligible_domain_seed_cells','initial_only_domains','means','domain_means']},indent=2)+'\n')
print('Selected cell coverage:')
print(json.dumps(result['coverage_by_domain'],indent=2))
print('Level2 terminal cells:',len(terminal)//4)
print('Frozen snapshot:',output,output.stat().st_size,'bytes')
