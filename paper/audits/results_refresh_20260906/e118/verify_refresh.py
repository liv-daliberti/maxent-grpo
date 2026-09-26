from pathlib import Path
from collections import Counter
import datetime,json,math,statistics
ROOT=Path('/n/fs/similarity/maxent-grpo');OUT=Path(__file__).resolve().parent
path=Path('paper/figures/e118_all_scale_factorial_progress.json');before=json.loads((OUT/'before'/path).read_text());after=json.loads((ROOT/path).read_text());delta=[]
for scale,cells in after['cells'].items():
 for domain,cell in cells.items():
  old=before['cells'][scale][domain];seeds=cell['matched_seeds'];eff=cell['replay_maxrl_minus_maxrl'];assert set(eff['per_seed'])==set(map(str,seeds))
  for metric,s in eff['summaries'].items():
   v=[eff['per_seed'][str(seed)][metric] for seed in seeds];assert s['n']==len(seeds)
   if v:assert math.isclose(s['mean'],statistics.fmean(v),abs_tol=1e-12)
   assert ('student_t_95' in s)==(len(seeds)==5)
  delta.append({'scale':scale,'domain':domain,'before_seeds':old['matched_seeds'],'after_seeds':seeds,'new_seeds':sorted(set(seeds)-set(old['matched_seeds'])),'full_factorial_seeds':cell['method_seeds']['drgrpo'],'effects':eff['summaries']})
audit=after['endpoint_integrity_audit'];assert len(audit)==150;assert all(not r.get('conflicting_retry_selected') for r in audit)
summary={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'endpoint_statuses':dict(Counter(r['status'] for r in audit)),'complete_pair_blocks':sum(len(r['after_seeds'])==5 for r in delta),'total_terminal_pairs':sum(len(r['after_seeds']) for r in delta),'complete_four_arm_blocks':sum(len(r['full_factorial_seeds'])==5 for r in delta),'displayed_scale_cells_unchanged':all(before['cells'][s]==after['cells'][s] for s in ['qwen05b','falcon1b']),'main_average_values_unchanged':before['absolute_cross_domain_average']==after['absolute_cross_domain_average'],'delta':delta}
(OUT/'verified_delta.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
