"""Offline pacing tests: all HTTP transports and clocks are local fixtures."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import json
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ops'))
import model_archive_commit_pacing as m

LIMIT=json.dumps({'error':'You have exceeded the rate limit for repository commits (256 per hour). Try again in about 1 hour.'})
class Clock:
 def __init__(self):self.value=1000.0;self.sleeps=[]
 def __call__(self):return self.value
 def sleep(self,seconds):self.sleeps.append(seconds);self.value+=seconds
class Response:
 def __init__(self,status=200,body=None,headers=None):
  self.status_code=status;self.text=json.dumps({'commitOid':'a'*40}) if body is None else body;self.headers=headers or {}
 def json(self):return json.loads(self.text)
@pytest.fixture
def pace(tmp_path):
 clock=Clock();evidence=tmp_path/'evidence.json';evidence.write_text('{}');pin={'path':str(evidence),'sha256':m.digest(evidence)}
 pacer=m.CommitPacer(tmp_path/'gate',clock=clock,sleep=clock.sleep);pacer.bootstrap(initial_cooldown_until=0,evidence_pin=pin)
 return SimpleNamespace(pacer=pacer,clock=clock,pin=pin)
def restart(f):return m.CommitPacer(f.pacer.directory,clock=f.clock,sleep=f.clock.sleep)

@pytest.mark.parametrize('status,body,classified',[(429,LIMIT,True),(503,LIMIT,False),(429,'{}',False),(429,'[]',False),(429,'not JSON',False),(429,json.dumps({'error':'rate limit for repository commits (256 per hour)'}),False),(429,json.dumps({'error':42}),False)])
def test_only_exact_granular_limit_is_retriable_after_cooldown(status,body,classified):assert m.granular_commit_limit(status,body) is classified

def test_intent_and_pending_state_precede_send_and_result_precedes_clear(pace,monkeypatch):
 saves=[];save=m.save
 def traced(path,value,**kwargs):saves.append((Path(path).name,dict(value)));return save(path,value,**kwargs)
 monkeypatch.setattr(m,'save',traced)
 def send():
  state=m.read(pace.pacer.directory/'state.json')
  assert state['pending_attempt']==0 and state['next_sequence']==1 and state['last_attempt_at']==pace.clock()
  assert (pace.pacer.directory/'attempts/000000.intent.json').is_file()
  assert not (pace.pacer.directory/'attempts/000000.result.json').exists()
  return Response()
 pace.pacer.transact('Archive fixture',send,'offline-placeholder')
 assert [name for name,value in saves]==['000000.intent.json','state.json','000000.result.json','state.json']
 assert saves[-1][1]['pending_attempt'] is None and saves[-2][1]['status']=='committed'

def test_twenty_second_spacing_survives_restart_and_counts_request_start(pace):
 starts=[]
 def send():starts.append(pace.clock());pace.clock.value+=3;return Response()
 pace.pacer.transact('Archive first',send,'offline-placeholder')
 restarted=restart(pace);restarted.bootstrap(initial_cooldown_until=0,evidence_pin=pace.pin)
 restarted.transact('Archive second',send,'offline-placeholder')
 assert starts==[1000.0,1020.0] and pace.clock.sleeps==[17.0]

def test_granular_cooldown_is_3605_seconds_from_response_and_records_no_send(pace):
 def limited():pace.clock.value+=7;return Response(429,LIMIT,{'Retry-After':'3600'})
 with pytest.raises(m.RepositoryCommitRateLimited):pace.pacer.transact('Archive limited',limited,'offline-placeholder')
 state=pace.pacer.snapshot();assert state['cooldown_until']==1007+3605 and state['pending_attempt'] is None
 sender=Mock(return_value=Response())
 with pytest.raises(m.RepositoryCommitCooldown):restart(pace).transact('Archive waiting',sender,'offline-placeholder')
 sender.assert_not_called();refusals=list((pace.pacer.directory/'not_sent').glob('*.json'));assert len(refusals)==1
 refusal=m.read(refusals[0]);assert refusal['status']=='deferred_before_http' and refusal['summary']=='Archive waiting' and refusal['last_sequence']==0
 assert restart(pace).snapshot()['next_sequence']==1
 pace.clock.value=state['cooldown_until'];restart(pace).transact('Archive after cooldown',sender,'offline-placeholder');assert sender.call_count==1

def test_uncertain_transport_blocks_restarted_sender(pace):
 with pytest.raises(TimeoutError):pace.pacer.transact('Archive uncertain',Mock(side_effect=TimeoutError()),'offline-placeholder')
 state=restart(pace).snapshot();assert state['blocked_reason']=='uncertain_request' and state['pending_attempt']==0
 assert m.read(pace.pacer.directory/'attempts/000000.result.json')['status']=='uncertain'
 send=Mock()
 with pytest.raises(m.CommitPacingUnsafe,match='unresolved_prior_commit_attempt'):restart(pace).transact('Archive retry',send,'offline-placeholder')
 send.assert_not_called()

@pytest.mark.parametrize('response,reason',[(Response(429,'{"error":"ordinary limit"}'),'unclassified_response'),(Response(503,'unavailable'),'unclassified_response'),(Response(403,'forbidden'),'unclassified_response'),(Response(200,'{}'),'unparseable_success'),(Response(200,'not json'),'unparseable_success')])
def test_unclassified_and_unparseable_responses_remain_blocked(pace,response,reason):
 with pytest.raises(m.CommitPacingUnsafe):pace.pacer.transact('Archive bad response',lambda:response,'offline-placeholder')
 state=restart(pace).snapshot();assert state['blocked_reason']==reason and state['pending_attempt']==0
 sender=Mock()
 with pytest.raises(m.CommitPacingUnsafe):restart(pace).transact('Archive cannot retry',sender,'offline-placeholder')
 sender.assert_not_called()

def test_crash_after_intent_before_pending_state_cannot_send_on_restart(pace,monkeypatch):
 real_save=m.save
 def crash(path,value,**kwargs):
  if Path(path).name=='state.json' and value.get('pending_attempt')==0:raise OSError('simulated crash before pending state')
  return real_save(path,value,**kwargs)
 sender=Mock(return_value=Response())
 with monkeypatch.context() as patcher:
  patcher.setattr(m,'save',crash)
  with pytest.raises(OSError):pace.pacer.transact('Archive interrupted',sender,'offline-placeholder')
 with pytest.raises(FileExistsError):restart(pace).transact('Archive interrupted',sender,'offline-placeholder')
 sender.assert_not_called()

def test_crash_after_result_before_state_clear_retains_pending_and_no_retry(pace,monkeypatch):
 real_save=m.save;sender=Mock(return_value=Response())
 def crash(path,value,**kwargs):
  if Path(path).name=='state.json' and value.get('pending_attempt') is None and value.get('next_sequence')==1:raise OSError('simulated crash before clear')
  return real_save(path,value,**kwargs)
 with monkeypatch.context() as patcher:
  patcher.setattr(m,'save',crash)
  with pytest.raises(OSError):pace.pacer.transact('Archive accepted but uncleared',sender,'offline-placeholder')
 assert m.read(pace.pacer.directory/'attempts/000000.result.json')['status']=='committed'
 with pytest.raises(m.CommitPacingUnsafe):restart(pace).transact('Archive accepted but uncleared',sender,'offline-placeholder')
 assert sender.call_count==1

@pytest.mark.parametrize('kind',['source','spacing','evidence'])
def test_restart_rejects_registration_policy_or_evidence_tamper(pace,kind):
 if kind=='source':
  path=pace.pacer.directory/'registration.json';v=m.read(path);v['source_sha256']='0'*64;m.save(path,v)
 elif kind=='spacing':
  path=pace.pacer.directory/'state.json';v=m.read(path);v['minimum_spacing_seconds']=0;m.save(path,v)
 else:Path(pace.pin['path']).write_text('changed')
 with pytest.raises(m.CommitPacingUnsafe):
  if kind=='evidence':restart(pace).bootstrap(initial_cooldown_until=0,evidence_pin=pace.pin)
  else:restart(pace).snapshot()

def test_timing_header_and_error_sanitization_is_bounded():
 secret='offline-placeholder';headers={'Authorization':secret,'Set-Cookie':'private','Retry-After':'3600','Date':'Fri, 11 Sep 2026 20:00:00 GMT','RateLimit':'"api";r=0;t=30','RateLimit-Policy':'hf_UNRELATED','retry-after':'https://example.org/private','DATE':'x'*513}
 result=m.safe_headers(headers,secret);assert set(result)=={'Retry-After','Date','RateLimit','RateLimit-Policy'}
 assert result['RateLimit-Policy']=='[credential omitted]'
 assert m.safe_headers({'Date':secret},secret)=={} and m.safe_headers({'Date':'line\nbreak'},secret)=={}
 text=m.sanitize(secret+' https://example.org/secret hf_EXAMPLE sk-EXAMPLE '+('x'*2000),secret)
 assert secret not in text and 'https://' not in text and 'hf_EXAMPLE' not in text and 'sk-EXAMPLE' not in text and len(text)<=1000

@pytest.fixture
def installed(monkeypatch,pace):
 backend={};transport=Mock(return_value=Response());contexts=[]
 class Session:
  def request(self,method,url,*args,**kwargs):return transport(method,url,*args,**kwargs)
 class Api:
  def create_commit(self,*args,**kwargs):
   contexts.append(getattr(m._TLS,'summary',None))
   return backend['factory']().request('POST','https://huggingface.co'+m.COMMIT_PATH,allow_redirects=True)
 def configure_http_backend(*,backend_factory):backend['factory']=backend_factory
 monkeypatch.setitem(sys.modules,'requests',SimpleNamespace(Session=Session))
 monkeypatch.setitem(sys.modules,'huggingface_hub',SimpleNamespace(HfApi=Api,configure_http_backend=configure_http_backend))
 monkeypatch.setattr(m._TLS,'summary',None,raising=False)
 m.install_pacing(pace.pacer,'offline-placeholder')
 return SimpleNamespace(api=Api,backend=backend,transport=transport,contexts=contexts)

def test_sdk_context_is_thread_local_and_commit_redirects_are_disabled(installed):
 kwargs={'repo_id':m.REPO,'repo_type':'model','commit_message':'Archive exact model','revision':'main'}
 installed.api().create_commit(**kwargs)
 assert installed.contexts==['Archive exact model'] and m._TLS.summary is None
 assert installed.transport.call_args.kwargs['allow_redirects'] is False

@pytest.mark.parametrize('change',[{'revision':'other'},{'create_pr':True},{'run_as_future':True},{'repo_id':'other/repo'},{'repo_type':'dataset'},{'commit_message':'unreviewed'}])
def test_unsupported_commit_modes_fail_before_transport(installed,change):
 kwargs={'repo_id':m.REPO,'repo_type':'model','commit_message':'Archive exact model',**change}
 with pytest.raises(m.CommitPacingUnsafe):installed.api().create_commit(**kwargs)
 installed.transport.assert_not_called()

def test_unmapped_post_context_and_query_cannot_bypass_pacer(installed):
 session=installed.backend['factory']()
 with pytest.raises(m.CommitPacingUnsafe,match='missing_exact_commit_context'):session.request('POST','https://huggingface.co'+m.COMMIT_PATH)
 with pytest.raises(m.CommitPacingUnsafe,match='unexpected_commit_query'):session.request('POST','https://huggingface.co'+m.COMMIT_PATH+'?x=1')
 installed.transport.assert_not_called()

def test_tls_restores_after_transport_failure_and_lfs_bypasses_commit_gate(installed):
 installed.transport.side_effect=TimeoutError()
 with pytest.raises(TimeoutError):installed.api().create_commit(repo_id=m.REPO,repo_type='model',commit_message='Archive fixture')
 assert m._TLS.summary is None
 installed.transport.side_effect=None
 installed.backend['factory']().request('PUT','https://example.org/lfs-object')
 assert installed.transport.call_args.args==('PUT','https://example.org/lfs-object')


def test_concurrent_pacer_instances_share_one_global_spacing_gate(pace):
 from concurrent.futures import ThreadPoolExecutor
 from threading import Barrier
 barrier=Barrier(3);starts=[]
 def worker(index):
  pacer=restart(pace);barrier.wait(timeout=5)
  def sender():starts.append(pace.clock());return Response()
  pacer.transact('Archive concurrent '+str(index),sender,'offline-placeholder')
 with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(worker,range(3)))
 assert starts==[1000.0,1020.0,1040.0]
 assert pace.pacer.snapshot()['next_sequence']==3
 assert len(list((pace.pacer.directory/'attempts').glob('*.result.json')))==3
