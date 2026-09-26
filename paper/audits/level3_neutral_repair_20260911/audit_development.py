"""Independent structural audit of the prospective neutral candidate pools."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from math import prod
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'ops')]
from datasets import load_from_disk
from oat_drgrpo.python_modebench_process import validate_python_factor_function_external
from modebench_independent_seeds import seed_schedule
ART=ROOT/'var/artifacts/modebench_level3_neutral_v1'
EXPECTED='48a5bdee7be17a16fa914b76e4f1db98c47a23b7e35e4d42c4a92ab63477b935'
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
def cases(row):return tuple(sorted(json.loads(row['answer'])['cases']))
assert digest(ART/'registration.json')==EXPECTED
plan=read(ART/'registration.json')
assert all(digest(p)==d for p,d in plan['files_sha256'].items())
ref=ROOT/'var/data/modebench_harder_v2_matched_r5/python_factors'
hist={s:Counter(r['answer_mode_count'] for r in load_from_disk(str(ref/s))['multi_answer']) for s in ('dev','eval')}
union=hist['dev']|hist['eval']
# This earlier independent inventory includes the original data and E117 reserves.
blocked={tuple(c) for c in read(ROOT/'artifacts/modebench_level3_neutral_calibration_20260911/excluded_cases.json')}
blocked|={tuple(c) for domain,c in read(ART/'excluded_identities.json') if domain=='python_factors'}
seen=set();request_blocks=set();report={};witnesses=0
# Warm the external worker without consuming model or calibration outcomes.
warm={'verifier':'python_factor_function','python_version':'factor-v1','cases':[6,8,10,12]}
validate_python_factor_function_external('lambda n:2',warm)
for tier in range(4):
 path=Path(plan['calibration_pools'][str(tier)]['path']);pool=rows(path)
 assert len(pool)==166 and Counter(r['answer_mode_count'] for r in pool)==union
 current={cases(r) for r in pool}
 assert len(current)==len(pool) and not current&(blocked|seen)
 low,high=plan['case_windows'][tier];a,b=plan['minimum_bands'][tier]
 for row in pool:
  spec=json.loads(row['answer']);cs=cases(row)
  ds=[[d for d in range(2,n) if n%d==0] for n in cs]
  assert len(cs)==len(set(cs))==4 and all(low<=n<=high for n in cs) and a<=min(cs)<=b
  assert all(len(d)>=2 and min(d)<=5 for d in ds)
  assert prod(map(len,ds))==row['answer_mode_count']==spec['num_modes']
  keys=set()
  for outputs in ([min(d) for d in ds],[max(d) for d in ds]):
   program='lambda n:'+''.join(f'{d} if n=={n} else ' for n,d in zip(cs[:-1],outputs[:-1]))+str(outputs[-1])
   val=validate_python_factor_function_external(program,spec)
   assert val is not None
   keys.add(val.canonical_key);witnesses+=1
  assert len(keys)==spec['num_externally_certified_modes']==2
  assert hashlib.sha256('\n'.join(sorted(keys)).encode()).hexdigest()==spec['certified_mode_key_sha256']
 schedule=seed_schedule('python_factors',[r['problem'] for r in pool],plan['development_labels'])
 blocks={v for group in schedule for v in group}
 assert len(blocks)==664 and not blocks&request_blocks
 request_blocks|=blocks;seen|=current
 report[str(tier)]={'rows':len(pool),'sha256':digest(path),'support_histogram':dict(sorted(union.items())),
                    'minimum_case_range':[min(min(c) for c in current),max(min(c) for c in current)],
                    'maximum_case_range':[min(max(c) for c in current),max(max(c) for c in current)]}
result={'status':'structural_development_audit_pass','registered_at':plan['created_at'],'audited_at':datetime.now(timezone.utc).isoformat(),
 'registration_sha256':EXPECTED,'source_pins_checked':len(plan['files_sha256']), 'historical_semantic_exclusions':len(blocked),
 'fresh_unique_problems':len(seen),'externally_reverified_witnesses':witnesses,'distinct_request_blocks':len(request_blocks),
 'exact_support_cells_per_tier':len(union),'pools':report,'model_success_or_difficulty_admission_claimed':False}
out=Path(__file__).with_name('development_structure.json')
with out.open('x') as h:json.dump(result,h,indent=2,sort_keys=True);h.write('\n')
print(json.dumps({k:v for k,v in result.items() if k!='pools'}))
