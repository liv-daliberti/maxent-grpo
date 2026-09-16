"""Real subprocess interruption, lock ownership, and conservative admission tests."""
from datetime import datetime,timezone,timedelta
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ops/exp_scaling'))
spec=importlib.util.spec_from_file_location('e122_two_release_test',ROOT/'ops/exp_scaling/release_e122_two_graph_cells_20260910.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def cells():
    return {'binding':{'frozen':'binding'},'jobs':[{'job_id':j,'cell':{'domain':'graph_coloring','arm':arm,'seed':43}} for j,arm in a.TARGETS.items()]}

def test_exact_frozen_graph_domain_and_paired_methods():
    campaign=cells();assert [j['job_id'] for j in a.selected(campaign)]==list(a.TARGETS)
    campaign['jobs'][0]['cell']['domain']='graph'
    with pytest.raises(RuntimeError,match='identity'):a.selected(campaign)

def test_atomic_publication_never_overwrites_existing_receipt(tmp_path):
    target=tmp_path/'receipt.json';a.atomic_new(target,{'first':True})
    with pytest.raises(FileExistsError):a.atomic_new(target,{'other':True})
    assert json.loads(target.read_text())=={'first':True}

def test_interrupted_fsync_never_exposes_partial_final_receipt(tmp_path):
    target=tmp_path/'receipt.json'
    code="import importlib.util,time; from pathlib import Path; s=importlib.util.spec_from_file_location('m',%r); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.os.fsync=lambda fd:time.sleep(60);m.atomic_new(Path(%r),{'a':'complete'})"%(str(a.SOURCE),str(target))
    p=subprocess.Popen(['/usr/bin/python3','-B','-c',code],close_fds=True)
    try:
        end=time.monotonic()+3
        while not list(tmp_path.glob('*.tmp')) and not list(tmp_path.glob('.*.tmp')) and time.monotonic()<end:time.sleep(.01)
        assert list(tmp_path.glob('.*.tmp')) and not target.exists()
    finally:p.kill();p.wait(timeout=2)
    assert not target.exists()

def test_blocked_child_cannot_extend_parent_flock_or_inherit_it(monkeypatch,tmp_path):
    art=tmp_path/'art';art.mkdir();journal=tmp_path/'journal';journal.mkdir();planpath=art/'plan.json';planpath.write_text('{}')
    fifo=tmp_path/'blocked_io';os.mkfifo(fifo);fdreport=tmp_path/'child_fds.json'
    helper=tmp_path/'worker.py';helper.write_text('''import json,os,sys,time
from pathlib import Path
mode=sys.argv[sys.argv.index('--worker')+1]
print(json.dumps({'event':'worker_ready'}),flush=True)
assert sys.stdin.buffer.readline()==b'GO\\n'
if mode=='dry-run':
    print(json.dumps({'event':'readonly_sequence_complete','elapsed_seconds':.01}),flush=True)
    sys.exit(0)
elif mode=='release':
    targets=[]
    for fd in Path('/proc/self/fd').iterdir():
        try:targets.append(str(fd.readlink()))
        except OSError:pass
    Path(%r).write_text(json.dumps(targets))
    with open(%r) as stream:stream.read()
else:
    sys.stdin.buffer.readline()
    print(json.dumps({'event':'worker_error','error':'no actual acknowledgement'}),flush=True)
    sys.exit(2)
'''%(str(fdreport),str(fifo)))
    campaign=cells()
    monkeypatch.setattr(a,'ART',art);monkeypatch.setattr(a,'JOURNAL',journal);monkeypatch.setattr(a,'PLAN',planpath)
    monkeypatch.setattr(a,'PROOF',planpath);monkeypatch.setattr(a,'SOURCE',helper);monkeypatch.setattr(a,'STORAGE_LOCK',tmp_path/'shared.lock')
    monkeypatch.setattr(a,'verify_plan',lambda:{'binding':campaign['binding']})
    monkeypatch.setattr(a,'authenticate_cached',lambda args,launcher:campaign)
    monkeypatch.setattr(a,'storage_proof',lambda *args,**kwargs:{})
    monkeypatch.setattr(a,'handoff_guard',lambda **kwargs:{})
    import e122_shared_storage_admission as shared_storage
    monkeypatch.setattr(shared_storage,'storage_report',lambda:{'allowed':True,'errors':[]})
    monkeypatch.setattr(a,'heartbeat',lambda:({'reserved_unfinished_slots':4,'observed_at':a.now()},0))
    monkeypatch.setattr(a,'PRIMARY_LIMIT',.25);monkeypatch.setattr(a,'RECOVERY_LIMIT',.2);monkeypatch.setattr(a,'TOTAL_LIMIT',1.0)
    started=time.monotonic()
    with pytest.raises(RuntimeError,match='did not fully reconcile'):a.apply(campaign=campaign,args=None,launcher=None)
    elapsed=time.monotonic()-started
    assert elapsed<2.5
    evidence=json.loads((art/'parent_execution_result.json').read_text());assert evidence['lock_elapsed_seconds']<1.25
    inherited=json.loads(fdreport.read_text());assert str(journal/'.lock') not in inherited and str(tmp_path/'shared.lock') not in inherited
    with (journal/'.lock').open('a') as probe:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not any(e.get('event')=='release_pair_complete' for e in evidence['events'])

def proof_fixture(monkeypatch,tmp_path,*,free,extra=()):
    proof={'status':'approved','observed_at':a.now(),'unknown_writers':[],'schema':'e122_shared_storage_admission_v1','e122_terminal_reserve_bytes':125*a.GIB,'e122_peak_reserve_bytes':96*a.GIB,'shared_headroom_bytes':64*a.GIB,'writer_job_ids':['1'],'external_peak_reserve_bytes':82*a.GIB}
    p=tmp_path/'proof.json';p.write_text(json.dumps(proof));monkeypatch.setattr(a,'queue_gpu_ids',lambda:(['1',*extra],''))
    monkeypatch.setattr(a.os,'statvfs',lambda path:SimpleNamespace(f_bavail=free,f_frsize=1))
    return p

def test_full_terminal_active_and_external_reserve_are_required(monkeypatch,tmp_path):
    required=(64+125+96+82)*a.GIB
    p=proof_fixture(monkeypatch,tmp_path,free=required-1)
    with pytest.raises(RuntimeError,match='combined reservations'):a.storage_proof(p)
    monkeypatch.setattr(a.os,'statvfs',lambda path:SimpleNamespace(f_bavail=required,f_frsize=1))
    assert a.storage_proof(p)['required_bytes']==required

def test_unknown_new_writer_blocks_even_when_disk_is_free(monkeypatch,tmp_path):
    p=proof_fixture(monkeypatch,tmp_path,free=10**15,extra=['unbudgeted'])
    with pytest.raises(RuntimeError,match='new GPU writer'):a.storage_proof(p)


@pytest.mark.parametrize('change',[None,{'returncode':None},{'returncode':1},{'job_id':'other'},{'intent_sha256':'other'}])
def test_reconciliation_never_manufactures_success_without_exact_process_ack(change):
    ack={'schema':'e122_level3_release_result_v1','job_id':'31158698','returncode':0,'error':None,'intent_sha256':'pinned','command':['scontrol','release','31158698']}
    candidate=None if change is None else {**ack,**change}
    with pytest.raises(RuntimeError,match='actual successful process acknowledgement'):a.actual_acknowledgement(candidate,'31158698','pinned')

def test_reconciliation_preserves_the_actual_process_receipt():
    ack={'schema':'e122_level3_release_result_v1','job_id':'31158698','returncode':0,'error':None,'intent_sha256':'pinned','command':['scontrol','release','31158698'],'stdout':'actual output','recorded_at':'actual time'}
    assert a.actual_acknowledgement(ack,'31158698','pinned') is ack


def test_quick_queue_classification_matches_canonical_slurm_held_spellings(monkeypatch):
    rows='1|e117-old|PENDING|job requeued in held state|gres/gpu:a100:1\n2|e117-old|PENDING|job_requeued_in_held_state|gres/gpu:a100:1\n3|held|PENDING|JobHeldUser|gres/gpu:1\n4|live|RUNNING|None|gres/gpu:1\n5|queued|PENDING|Priority|gres/gpu:1\n6|cpu|RUNNING|None|N/A\n'
    monkeypatch.setattr(a,'command',lambda argv:subprocess.CompletedProcess(argv,0,rows,''))
    assert a.queue_gpu_ids()[0]==['4','5']


@pytest.mark.parametrize('reason',['JobHeldUser','job_requeued_in_held_state','job requeued in held state'])
def test_owned_e124_hold_accepts_exact_user_and_requeue_display_forms(reason):
    record=f'JobState=PENDING Reason={reason} Dependency=(null)'
    assert a.scheduler_held('PENDING',a.scheduler_reason(record),allow_admin=False)
    assert not a.scheduler_held('RUNNING',reason,allow_admin=False)

def test_e124_pause_does_not_adopt_an_administrative_hold():
    assert not a.scheduler_held('PENDING','JobHeldAdmin',allow_admin=False)


def test_completed_heartbeat_accepts_six_second_snapshot_age_with_collision_margin():
    assert a.heartbeat_is_fresh(6)
    assert a.heartbeat_is_fresh(15)
    assert not a.heartbeat_is_fresh(15.001)
    assert a.heartbeat_is_fresh(16,after_open=True)
    assert not a.heartbeat_is_fresh(16.001,after_open=True)
    assert not a.heartbeat_is_fresh(-.001)
    assert 60-a.LOCK_OPEN_HEARTBEAT_LIMIT-a.TOTAL_LIMIT>=20


def heartbeat_fixture(monkeypatch,tmp_path,*,newline=True):
    publication=datetime.now(timezone.utc)-timedelta(seconds=2)
    snapshot=publication-timedelta(seconds=31.073)
    binding={'frozen':'binding'}
    payload={'schema':'e122_level3_release_status_v1','binding':binding,'observed_at':snapshot.isoformat(),'max_released_nonterminal':4,'issues':[],'unknown':[],'needs_operator_review_job_ids':[]}
    content=json.dumps(payload).encode()+(b'\n' if newline else b'')
    journal=tmp_path/'journal';directory=journal/'status';directory.mkdir(parents=True)
    log=directory/(publication.strftime('%Y%m%dT%H%M%S.%fZ')+'_'+'a'*32+'.json');log.write_bytes(content)
    plan=tmp_path/'plan.json';plan.write_text(json.dumps({'binding':binding}))
    monkeypatch.setattr(a,'JOURNAL',journal);monkeypatch.setattr(a,'PLAN',plan)
    return len(content),publication.isoformat()

def test_filename_publication_accepts_thirty_one_second_snapshot_lag(monkeypatch,tmp_path):
    metadata=heartbeat_fixture(monkeypatch,tmp_path)
    monkeypatch.setattr(a,'publication_stat',lambda path:metadata)
    record,age=a.heartbeat()
    assert a.heartbeat_is_fresh(age)
    assert record['_heartbeat_publication']['snapshot_to_publication_seconds']==31.073

def test_readonly_publication_rereads_when_append_races(monkeypatch,tmp_path):
    metadata=heartbeat_fixture(monkeypatch,tmp_path)
    replies=iter([(metadata[0]-1,metadata[1]),metadata,metadata,metadata])
    calls=[]
    def stat(path):calls.append(True);return next(replies)
    monkeypatch.setattr(a,'publication_stat',stat)
    assert a.heartbeat_is_fresh(a.heartbeat()[1])
    assert len(calls)==4

def test_partial_publication_cannot_authorize_parent_lock(monkeypatch,tmp_path):
    metadata=heartbeat_fixture(monkeypatch,tmp_path,newline=False)
    monkeypatch.setattr(a,'publication_stat',lambda path:metadata)
    with pytest.raises(RuntimeError,match='incomplete'):a.heartbeat()


def test_filename_stamp_is_not_shifted_by_delayed_nfs_file_mtime(monkeypatch,tmp_path):
    metadata=heartbeat_fixture(monkeypatch,tmp_path)
    delayed=(metadata[0],(datetime.fromisoformat(metadata[1])+timedelta(seconds=31)).isoformat())
    monkeypatch.setattr(a,'publication_stat',lambda path:delayed)
    record,age=a.heartbeat()
    assert a.heartbeat_is_fresh(age)
    assert record['_heartbeat_publication']['file_mtime']==delayed[1]


def test_gnu_stat_nanosecond_text_is_compared_without_python_version_date_parsing(monkeypatch):
    raw='32084|2026-09-10 21:54:35.967820000 -0400\n'
    monkeypatch.setattr(a,'command',lambda argv,timeout:subprocess.CompletedProcess(argv,0,raw,''))
    assert a.publication_stat(Path('/readonly/status.json'))==(32084,'2026-09-10 21:54:35.967820000 -0400')
