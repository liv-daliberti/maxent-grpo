"""Full source-to-standard-loader roundtrip and portable verifier checks."""
from __future__ import annotations
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
import pytest
from datasets import load_dataset,load_from_disk

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'var/artifacts/modebench_hf_release_20260911/package'
CONFIGS=[f'level{level}_{domain}' for level in (1,2,3) for domain in ('graph_coloring','countdown','python_factors','mathir','pantry_plan')]+['level1_graph_coloring_unique_answer']

@pytest.fixture(scope='session')
def manifest():return json.loads((PACKAGE/'MANIFEST.json').read_text())

@pytest.mark.parametrize('config',CONFIGS)
def test_standard_huggingface_loader_preserves_every_row(config,manifest,tmp_path):
 loaded=load_dataset(str(PACKAGE),config,cache_dir=str(tmp_path/'cache'),keep_in_memory=True)
 records=[s for s in manifest['splits'] if s['config_name']==config]
 assert set(loaded)=={s['split'] for s in records}
 for record in records:
  original=load_from_disk(str(ROOT/record['source_repository_path']))[record['original_datasetdict_split']]
  result=loaded[record['split']]
  assert list(result)==list(original)
  assert result.features.to_dict()==original.features.to_dict()
  assert len(result)==record['rows']
  raw=json.dumps(list(result),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
  assert hashlib.sha256(raw).hexdigest()==record['rows_sha256']


def test_original_auxiliary_and_split_roles_are_explicit(manifest):
 assert manifest['config_count']==16 and manifest['split_count']==42 and manifest['row_count']==9152
 primary=[s for s in manifest['splits'] if s['reported_primary_split']]
 assert sum(s['rows'] for s in primary)==9024
 assert not any(s['split']=='test' for s in manifest['splits'])
 assert [(s['config_name'],s['rows']) for s in primary if s['level']==1 and s['split']=='dev']==[('level1_pantry_plan',64)]
 aux=[s for s in manifest['splits'] if not s['reported_primary_split']]
 assert [(s['config_name'],s['split'],s['rows']) for s in aux]==[('level1_graph_coloring_unique_answer','eval',128)]


def test_portable_frozen_verifier_accepts_and_rejects_all_five_domains(tmp_path):
 copied=tmp_path/'code';shutil.copytree(PACKAGE/'code',copied)
 pantry=load_dataset('parquet',data_files=str(PACKAGE/'data/level1_pantry_plan/eval.parquet'),cache_dir=str(tmp_path/'cache'),keep_in_memory=True)['train'][0]
 (tmp_path/'pantry.json').write_text(pantry['answer'])
 script=r'''
import sys,json,itertools
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import oat_drgrpo.math_grader as grader
assert Path(grader.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
from oat_drgrpo.pantry_support_action import decode_pantry_support_mask
f=grader.validated_modebench_outcome_key
cases=[
 ({'verifier':'graph_coloring','n':4,'edges':[[1,2],[2,3],[3,4]],'partial_colors':[1,None,None,2]},'21','graph_coloring:1212'),
 ({'verifier':'countdown','numbers':[2,3,4],'target':14},'2+3*4','countdown:add(2,mul(3,4))'),
 ({'verifier':'python_factor_function','python_version':'factor-v1','cases':[6,10,15]},'lambda n: 2 if n % 2 == 0 else 3','python_factor:2,2,3'),
 ({'verifier':'mathir_action_menu','mathir_version':'linear-menu-v1','bindings':{'a':5,'b':2,'c':8,'d':2},'initial_lhs':'add(mul(a,x),b)','initial_rhs':'add(mul(d,x),c)','max_steps':4,'actions':{'A':'sub(b)','B':'sub(mul(d,x))','C':'div(sub(a,d))','D':'sub(add(mul(d,x),b))','E':'add(b)','F':'div(a)'},'support_is_open':False,'num_completions':5},'A;B;C',None),
]
for spec,response,expected in cases:
 reference=json.dumps(spec);key=f(response,reference)
 assert key is not None,(spec['verifier'],'valid response rejected')
 if expected is not None:assert key==expected
 assert f('not a valid answer',reference) is None
spec=json.loads(Path(sys.argv[2]).read_text());reference=json.dumps(spec)
for bits in itertools.product('01',repeat=6):
 response=decode_pantry_support_mask(''.join(bits),spec);key=f(response,reference)
 if key is not None:break
else:raise AssertionError('No feasible Pantry support in preserved eval row')
assert key.startswith('pantry:') or key.startswith('pantry_plan:')
assert f(decode_pantry_support_mask('000000',spec),reference) is None
print(json.dumps({'domains_verified':5,'portable_python_worker':True,'pantry_adapter_before_grader':True}))
'''
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None)
 done=subprocess.run([sys.executable,'-B','-c',script,str(copied),str(tmp_path/'pantry.json')],cwd=tmp_path,env=env,text=True,capture_output=True,timeout=45)
 assert done.returncode==0,done.stdout+done.stderr
 assert json.loads(done.stdout.strip())['domains_verified']==5
