from pathlib import Path
import json,os,subprocess,sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'var/artifacts/python_level3_cli_recovery_20260912'

@pytest.mark.parametrize('campaign',['e122','e124'])
def test_real_cli_and_domain_validator_accept_neutral_and_reject_mismatch(campaign):
    runtime=ART/'runtime_v2'/campaign
    code='''import tyro
from oat_drgrpo.args import ZeroMathArgs,validate_zero_math_args
name="qwen_level3_python_factors_neutral_v1"
a=tyro.cli(ZeroMathArgs,args=["--prompt-template",name,"--modebench-domain","python_factors","--modebench-syntax-profile","domain_legal_v1"])
validate_zero_math_args(a)
assert a.prompt_template==name
for domain,syntax in [("graph_coloring","none"),("python_factors","none"),("none","none")]:
 a.modebench_domain=domain;a.modebench_syntax_profile=syntax
 try:validate_zero_math_args(a)
 except ValueError:pass
 else:raise AssertionError("mismatched domain/syntax admitted")
print("Native CLI and validator passed")
'''
    env=dict(os.environ,PYTHONPATH=str(runtime/'src')+os.pathsep+os.environ.get('PYTHONPATH',''),PYTHONDONTWRITEBYTECODE='1')
    r=subprocess.run([sys.executable,'-B','-c',code],env=env,cwd='/tmp',text=True,capture_output=True,timeout=120)
    assert r.returncode==0,r.stdout+'\n'+r.stderr

@pytest.mark.parametrize('campaign',['e122','e124'])
def test_only_argument_admission_changes(campaign):
    from hashlib import sha256
    runtime=ART/'runtime_v2'/campaign
    d=json.loads((runtime/'CLI_AMENDMENT_IDENTITY.json').read_text())
    assert [k for k in d['parent_inventory'] if d['parent_inventory'][k]!=d['inventory_sha256'][k]]==['src/oat_drgrpo/args.py']
    for rel,sha in d['inventory_sha256'].items():assert sha256((runtime/rel).read_bytes()).hexdigest()==sha
