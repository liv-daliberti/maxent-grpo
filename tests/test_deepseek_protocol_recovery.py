"""Malformed native HTTP-200 responses remain protocol failures, not model draws."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import httpx
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ops'))
import evaluate_deepseek_protocol_recovery as runner
spec=importlib.util.spec_from_file_location('_native_runner_fixtures',ROOT/'tests/test_chat_frontier_modebench_runner.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

def setup(tmp_path,monkeypatch):
 monkeypatch.setattr(base,'runner',runner)
 data=base.fixture(tmp_path,monkeypatch,'DeepSeek-V4-Pro')
 target=tmp_path/'adapter_code/ops/evaluate_deepseek_protocol_recovery.py';target.parent.mkdir(parents=True);shutil.copyfile(runner.__file__,target)
 runner.atomic(tmp_path/'provider_protocol_adapter.json',{'version':'deepseek-provider-protocol-recovery-v1','source_path':str(target.relative_to(tmp_path)),'source_sha256':runner.file_sha(target),'original_manifest_sha256':runner.file_sha(tmp_path/'manifest.json')})
 return data

def bad_body():
 b=base.body('DeepSeek-V4-Pro',text='');b['usage']=None;b['choices'][0]['finish_reason']='';return b

def test_saved_malformed_receipt_retried_without_rewriting_evidence(tmp_path,monkeypatch):
 _,_,_,group,_,graded=setup(tmp_path,monkeypatch)
 receipt=base.raw(group,bad_body());path=tmp_path/receipt['relative_path'];runner.atomic(path,receipt);before=path.read_bytes()
 calls=base.install_client(monkeypatch,[httpx.Response(200,json=base.body('DeepSeek-V4-Pro'))])
 assert base.execute(tmp_path,'DeepSeek-V4-Pro')==0
 assert len(calls)==1 and graded==['correct'] and path.read_bytes()==before
 record=runner.read_jsonl(tmp_path/'samples.jsonl')[0]
 assert record['raw_receipt'].endswith('__02.json') and record['verified']
 assert json.loads(path.read_text())['response']['usage'] is None

def test_fresh_malformed_response_saved_and_retried_as_protocol_failure(tmp_path,monkeypatch):
 setup(tmp_path,monkeypatch);monkeypatch.setattr(runner,'retry_delay',lambda *args:0)
 calls=base.install_client(monkeypatch,[httpx.Response(200,json=bad_body()),httpx.Response(200,json=base.body('DeepSeek-V4-Pro'))])
 assert base.execute(tmp_path,'DeepSeek-V4-Pro',max_attempts=2)==0
 assert len(calls)==2 and len(runner.read_jsonl(tmp_path/'samples.jsonl'))==1
 error=runner.read_jsonl(tmp_path/'errors.jsonl')[0]
 assert error['http_status']==200 and error['error_type']=='ProviderProtocolError' and error['usage_unknown']
 assert error['response']['usage'] is None

@pytest.mark.parametrize('change',['model','answer','reasoning','stop','usage'])
def test_protocol_exception_is_narrow(change):
 b=bad_body()
 if change=='model':b['model']='grok-4.3'
 elif change=='answer':b['choices'][0]['message']['content']='answer'
 elif change=='reasoning':b['choices'][0]['message']['reasoning_content']=''
 elif change=='stop':b['choices'][0]['finish_reason']='other'
 else:b['usage']={'total_tokens':1}
 assert not runner.provider_protocol_failure({'http_status':200,'response':b})
