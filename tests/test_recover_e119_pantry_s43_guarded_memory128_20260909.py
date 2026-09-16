from pathlib import Path
import importlib,sys,subprocess
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ops/exp_scaling'))
x=importlib.import_module('recover_e119_pantry_s43_guarded_memory128_20260909')


def rec(state='RUNNING',memory='96G',restart=9,reason='None',priority=100):
 return f'JobState={state} MinMemoryNode={memory} Restarts={restart} Reason={reason} Priority={priority}'


@pytest.fixture
def setup(tmp_path,monkeypatch):
 monkeypatch.setattr(x,'ART',tmp_path)
 p=tmp_path/'31037827'/'plan.json';x.write(p,{'job_id':31037827,'before':rec(),'checkpoint':{'detail':{}}})
 current=[rec()];calls=[]
 monkeypatch.setattr(x,'show',lambda j:current[0]);monkeypatch.setattr(x,'audit',lambda p:None);monkeypatch.setattr(x,'sole_writer',lambda p:None)
 def profile(plan,r,memory=None,same_attempt=False):
  if memory:x.require(x.cli.field(r,'MinMemoryNode')==memory,'memory drift')
 monkeypatch.setattr(x,'profile',profile);monkeypatch.setattr(x.base,'archive',lambda *a:[])
 def command(a):
  calls.append(a)
  if a[1]=='requeuehold':current[0]=rec('PENDING',restart=10,reason='job_requeued_in_held_state',priority=0)
  if a[1]=='update':current[0]=rec('PENDING','128G',10,'job_requeued_in_held_state',0)
  if a[1]=='release':current[0]=rec('PENDING','128G',10,'Priority')
 monkeypatch.setattr(x,'command',command)
 return p,current,calls


def transaction(p,**kw):x.write(p.with_name('transaction.json'),{'plan_sha256':x.digest(p),'events':[],**kw})


def test_only_same_job_memory_changes(setup):
 p,current,calls=setup;x.apply(31037827)
 assert calls==[['scontrol','requeuehold','31037827'],['scontrol','update','JobId=31037827','MinMemoryNode=131072'],['scontrol','release','31037827']]
 x.apply(31037827);assert len(calls)==3


def test_uncertain_requeue_never_repeated(setup,monkeypatch):
 p,current,calls=setup
 def timeout(a):calls.append(a);raise subprocess.TimeoutExpired(a,45)
 monkeypatch.setattr(x,'command',timeout)
 with pytest.raises(subprocess.TimeoutExpired):x.apply(31037827)
 assert x.read(p.with_name('transaction.json'))['hold_intent']
 with pytest.raises(RuntimeError,match='owned requeue hold'):x.apply(31037827)
 assert len(calls)==1


@pytest.mark.parametrize('reason,restart',[('JobHeldUser',10),('JobHeldAdmin',10),('job_requeued_in_held_state',11)])
def test_only_exact_owned_hold_is_modified(setup,reason,restart):
 p,current,calls=setup;transaction(p,hold_intent=True);current[0]=rec('PENDING',restart=restart,reason=reason,priority=0)
 with pytest.raises(RuntimeError):x.apply(31037827)
 assert not calls


def test_lost_release_ack_reconciles_without_mutation(setup):
 p,current,calls=setup;transaction(p,hold_intent=True,memory_intent=True,release_intent=True);current[0]=rec('RUNNING','128G',10)
 x.apply(31037827);assert not calls and x.read(p.with_name('transaction.json'))['released']


def test_checkpoint_change_prevents_stop(setup,monkeypatch):
 p,current,calls=setup
 def changed(p):raise RuntimeError('checkpoint boundary changed')
 monkeypatch.setattr(x,'audit',changed)
 with pytest.raises(RuntimeError,match='boundary changed'):x.apply(31037827)
 assert not calls


def test_current_work_is_not_discarded_by_boundary_check(monkeypatch):
 monkeypatch.setattr(x.base,'timing_and_checkpoint',lambda *a:{'checkpoint_step':768,'current_step':780,'unsaved_steps':12})
 with pytest.raises(RuntimeError,match='preserve current work'):x.checkpoint(31037827,{})


def test_uncertain_memory_update_not_repeated(setup):
 p,current,calls=setup;transaction(p,hold_intent=True,memory_intent=True);current[0]=rec('PENDING',restart=10,reason='job_requeued_in_held_state',priority=0)
 with pytest.raises(RuntimeError,match='unacknowledged unchanged memory'):x.apply(31037827)
 assert not calls


def test_completing_after_requeue_stays_unmodified(setup):
 p,current,calls=setup;transaction(p,hold_intent=True);current[0]=rec('COMPLETING',restart=10,reason='job_requeued_in_held_state',priority=0)
 with pytest.raises(RuntimeError,match='poll and reconcile'):x.apply(31037827)
 assert not calls


def test_pressure_evidence_binds_exact_job_and_cgroup():
 import json,copy
 sample={'job_id':31037827,'cgroup_path':'/sys/fs/cgroup/system.slice/slurmstepd.scope/job_31037827','memory.high':96*2**30,'noncache_gib':99,'events':{'oom':0,'oom_kill':0}}
 proof={'job_id':31037827,'returncode':0,'stdout':json.dumps({'job_id':31037827,'high_events_delta':1,'before':sample,'after':sample})}
 assert x.validated_pressure(31037827,proof)['high_events_delta']==1
 with pytest.raises(RuntimeError,match='target'):x.validated_pressure(31048179,proof)
 wrong=copy.deepcopy(sample);wrong['cgroup_path']='wrong'
 proof['stdout']=json.dumps({'job_id':31037827,'high_events_delta':1,'before':sample,'after':wrong})
 with pytest.raises(RuntimeError,match='exact target'):x.validated_pressure(31037827,proof)


def test_scheduler_exports_must_match_ledger(monkeypatch):
 monkeypatch.setattr(x.base,'submitline',lambda r:'frozen submission')
 monkeypatch.setattr(x.base,'exports',lambda r:{'RUN_STAMP':'wrong','SAVE_PATH':'/other','OAT_ZERO_AUTO_RESUME':'1'})
 plan={'job_id':31037827,'before':'before','identity':{'run_stamp':'correct','run_dir':'/correct'}}
 with pytest.raises(RuntimeError,match='scheduler run/resume exports'):x.profile(plan,'JobId=31037827')


def test_replay_checkpoint_requires_bank_metadata():
 import pickle
 with pytest.raises(RuntimeError,match='replay bank state'):
  x.replay_metadata(31048179,'mp_rank_00_model_states.pt',pickle.dumps({'global_step':960}))
 x.replay_metadata(31048179,'mp_rank_00_model_states.pt',pickle.dumps({'online_canonical_bank_state':{'saved':True}}))
 x.replay_metadata(31037827,'mp_rank_00_model_states.pt',pickle.dumps({'global_step':768}))
