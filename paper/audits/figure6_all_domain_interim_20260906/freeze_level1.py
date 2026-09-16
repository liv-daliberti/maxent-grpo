from pathlib import Path
import hashlib,json,sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'ops/exp_scaling'))
from snapshot_evaluation_coverage import read_cell
from build_paper_core_terminal_endpoints import approved_run_exclusion
sources=[ROOT/'var/artifacts/e78_verified_replay_only_05b_jobs.json',ROOT/'var/artifacts/e118_all_scales_maxrl_verified_replay_jobs.json']
payload={'schema':'modebench-level1-evaluation-coverage-v1','started_at_utc':datetime.now(timezone.utc).isoformat(),'sources':{},'cells':[]}
for path in sources:
 payload['sources'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
 ledger=json.loads(path.read_text())
 for run in ledger['runs']:
  if 'e118' in path.name and run['scale']!='qwen05b':continue
  method={'control':'drgrpo','replay':'replay_drgrpo'}.get(run['arm'],run['arm'])
  exclusion=approved_run_exclusion(Path(run['run_dir']))
  if exclusion:raise RuntimeError(f'unexpected Level1 source exclusion: {exclusion}')
  result=read_cell(run['run_dir'])
  result.update(level='level1',domain=run['domain'],method=method,seed=run['seed'],registered_job_id=run.get('job_id'),ledger=str(path.relative_to(ROOT)))
  payload['cells'].append(result)
  if len(payload['cells'])%10==0:print('Level1 scanned',len(payload['cells']),flush=True)
payload['finished_at_utc']=datetime.now(timezone.utc).isoformat()
output=Path(__file__).resolve().parent/'level1_coverage_snapshot.json'
output.write_text(json.dumps(payload,indent=2)+'\n')
print(output,flush=True)
print('Invalid/conflicted checkpoints:',[(r['domain'],r['method'],r['seed'],r['invalid_or_conflicted_steps']) for r in payload['cells'] if r['invalid_or_conflicted_steps']],flush=True)
