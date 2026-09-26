#!/usr/bin/env python3
"""Release one exact E122 Graph43 pair using a parent-owned, bounded flock."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'var/artifacts/e122_two_cell_push_20260910'
PLAN=ART/'plan.json'
PROOF=ART/'storage_admission.json'
PROTOCOL=ROOT/'paper/preregistration/e122_two_cell_push_20260910.md'
SOURCE=Path(__file__).resolve()
TEST=ROOT/'tests/test_e122_two_graph_release_20260910.py'
JOURNAL=ROOT/'var/artifacts/e122_level3_factorial/release_controller'
WATCHLOG=ROOT/'var/artifacts/e122_level3_factorial/release_controller.stdout'
WATCH_COMMAND=ROOT/'var/artifacts/e122_level3_factorial/root_watch_arm_result.json'
STORAGE_LOCK=ROOT/'var/artifacts/shared_storage_admission_20260911.lock'
HANDOFF=ROOT/'var/artifacts/e118_e122_storage_handoff_20260911'
OBSERVER_ID=31220594
E124_CPU_ID=31164037
TARGETS={'31158698':'drgrpo','31158699':'replay_drgrpo'}
OLD_NODES={'node205','node206','node207','node302'}
NEW_NODES=OLD_NODES|{'node105','node203','node204','node208'}
RESOURCE_PROOF=ROOT/'var/artifacts/capacity_audit_20260910/e119_graph40_e122_runtime_compare.json'
QUALIFICATION=ROOT/'var/artifacts/capacity_audit_20260910/e122_graph40_a5000_qualification.json'
POOL=','.join(sorted(NEW_NODES))
GIB=1024**3
TOTAL_LIMIT=24.0
PRIMARY_LIMIT=13.0
RECOVERY_LIMIT=8.0
HEARTBEAT_LIMIT=15.0
LOCK_OPEN_HEARTBEAT_LIMIT=16.0

def require(condition,message):
    if not condition:raise RuntimeError(message)

def now():return datetime.now(timezone.utc).isoformat()
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def emit(payload):print(json.dumps(payload,sort_keys=True),flush=True)

def atomic_new(path,payload):
    """Publish complete immutable JSON atomically; interrupted writes stay private."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.parent/f'.{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp'
    with temporary.open('x') as stream:
        json.dump(payload,stream,sort_keys=True,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.link(temporary,path)
    temporary.unlink()
    directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)

def command(argv,timeout=3):
    return subprocess.run(argv,capture_output=True,text=True,check=False,timeout=timeout)

def scheduler_held(state,reason,*,allow_admin=True):
    import e124_storage_admission as classification
    if reason=='job_requeued_in_held_state':reason='job requeued in held state'
    return state=='PENDING' and reason in classification.HELD_REASONS and (allow_admin or reason!='JobHeldAdmin')

def scheduler_reason(record):
    match=re.search(r'(?:^|\s)Reason=(.*?)(?=\s[A-Za-z][A-Za-z0-9_/]*=|$)',record)
    require(match is not None,'Scheduler reason is missing')
    return match.group(1)

def queue_gpu_ids():
    result=command(['squeue','--noheader','--user',str(os.getuid()),'--format=%i|%j|%T|%r|%b'])
    require(result.returncode==0,'Cannot refresh GPU writer identities')
    ids=[]
    for line in result.stdout.splitlines():
        fields=line.strip().split('|')
        require(len(fields)==5,'Malformed queue writer record')
        jid,name,state,reason,gres=fields
        if 'gpu' not in gres or scheduler_held(state,reason):continue
        require(jid.isdigit(),'Ambiguous GPU allocation identity')
        ids.append(jid)
    return sorted(ids),result.stdout

def storage_proof(path,*,allow_targets=(),max_age=180):
    proof=read(path)
    require(proof.get('status')=='approved','Aggregate storage admission is not approved')
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(proof['observed_at'])).total_seconds()
    require(0<=age<=max_age,'Aggregate storage evidence is stale')
    require(not proof.get('unknown_writers'),'Unmapped checkpoint writers block release')
    require(proof.get('schema')=='e122_shared_storage_admission_v1','Wrong storage admission schema')
    require(proof['e122_terminal_reserve_bytes']==125*GIB and proof['e122_peak_reserve_bytes']==96*GIB and proof['shared_headroom_bytes']==64*GIB,'Approved reserve profile changed')
    current,raw=queue_gpu_ids()
    allowed=set(map(str,proof['writer_job_ids']))|set(allow_targets)
    require(set(current)<=allowed,'A new GPU writer appeared after aggregate storage audit')
    external=int(proof['external_peak_reserve_bytes'])
    require(external>=0,'Invalid external checkpoint reserve')
    stat=os.statvfs(ROOT);free=stat.f_bavail*stat.f_frsize
    required=64*GIB+125*GIB+6*16*GIB+external
    require(free>=required,'Current shared space cannot honor the approved combined reservations')
    require(getattr(stat,'f_favail',10000)>=10000,'Shared inode headroom is insufficient')
    return {'proof_sha256':digest(path),'free_bytes':free,'required_bytes':required,
            'external_peak_reserve_bytes':external,'gpu_job_ids':current,'queue':raw,'observed_at':now()}

def handoff_guard(*,require_ready=True):
    import prioritize_e118_capacity_20260905 as base
    package=read(HANDOFF/'plan.json');tx=read(HANDOFF/'transaction.json');registration=read(HANDOFF/'supervisor.json')
    pin=digest(HANDOFF/'plan.json')
    require(tx['plan_sha256']==pin and registration['plan_sha256']==pin,'Storage handoff plan binding changed')
    require(tx.get('hold_scope_complete') is True and len(tx.get('jobs',{}))==6,'Initial six-hold storage handoff is incomplete')
    require(all(state.get('status') in ('owned_held','released') for state in tx['jobs'].values()),'Storage hold requires reconciliation')
    require(tx.get('cpu_job_id')==registration['job_id']==OBSERVER_ID,'Registered storage observer identity changed')
    state=tx.get('e124',{});item=package['e124_cpu']
    require(item['job_id']==E124_CPU_ID and state.get('status')=='owned_held' and not state.get('release_intent'),'E124 independent release coordinator is not in the owned pause')
    result=command(['scontrol','show','job','-dd','-o',str(E124_CPU_ID)],timeout=2)
    require(result.returncode==0,'E124 CPU pause cannot be observed');record=result.stdout
    require(base.submit_tokens(record)==item['submit_tokens'],'E124 paused CPU command changed')
    for key,value in item['resources'].items():
        try:actual=base.field(record,key)
        except RuntimeError:
            require(key=='Comment' and value is None,'E124 CPU resource disappeared');continue
        require(actual==value,'E124 paused CPU resource changed: '+key)
    require(scheduler_held(base.field(record,'JobState'),scheduler_reason(record),allow_admin=False) and base.field(record,'Priority')=='0','E124 CPU is not inactive and held')
    require(base.field(record,'RunTime')=='00:00:00' and base.field(record,'NodeList') in ('','(null)') and int(base.field(record,'Restarts'))==state['expected_restarts'],'E124 CPU hold allocation/restarts changed')
    if require_ready:
        ready=read(HANDOFF/'ready.json');status=read(HANDOFF/'status.json')
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(ready['at_utc'])).total_seconds()
        require(ready['job_id']==OBSERVER_ID and ready['plan_sha256']==pin and ready.get('singleton') and 0<=age<180,'Storage observer heartbeat is stale or differs')
        require(status.get('status')=='monitoring','Storage observer needs review')
        observed=command(['scontrol','show','job','-o',str(OBSERVER_ID)],timeout=2)
        require(observed.returncode==0 and base.field(observed.stdout,'JobState')=='RUNNING','Registered storage observer is not running')
    return {'handoff_plan_sha256':pin,'observer_job_id':OBSERVER_ID,'e124_cpu_job_id':E124_CPU_ID,'e124_pause_record':record,'checked_at':now()}

def selected(campaign):
    jobs=[j for j in campaign['jobs'] if j['job_id'] in TARGETS]
    require(len(jobs)==2,'Exact two Graph jobs missing')
    for job in jobs:
        cell=job['cell'];require((cell['domain'],cell['arm'],cell['seed'])==('graph_coloring',TARGETS[job['job_id']],43),'Graph cell identity changed')
    return jobs

def authenticate_cached(args,launcher):
    import control_e122_level3_release as c
    return c.load_campaign(args,launcher)

def verified_watch_interval():
    import control_e122_level3_release as c
    argv=read(WATCH_COMMAND)['command']
    parsed=c.parse_args(argv[argv.index('--')+1:])
    require(parsed.watch and parsed.interval_seconds==60,'Original watcher must sleep60seconds after each completed heartbeat')
    require(parsed.interval_seconds-LOCK_OPEN_HEARTBEAT_LIMIT-TOTAL_LIMIT>=20,'Insufficient legacy watcher collision margin')
    return parsed.interval_seconds

def heartbeat_is_fresh(age,*,after_open=False):
    return 0<=age<=(LOCK_OPEN_HEARTBEAT_LIMIT if after_open else HEARTBEAT_LIMIT)

def prepare(*,campaign,args,launcher):
    import control_e122_level3_release as c
    import recover_terminal_timeouts_20260908 as recovery
    import prioritize_e118_capacity_20260905 as base
    if PLAN.exists():
        plan=verify_plan()
        require(not any((ART/name).exists() for name in ('preparation_storage.json','preparation_request.json','preparation_rehearsal.json','execution.intent.json')),'Existing rehearsal or execution evidence requires reconciliation; do not repeat')
        campaign=authenticate_cached(args,launcher)
        require(campaign['binding']==plan['binding'],'Prepared campaign binding changed')
        emit({'event':'retrying_readonly_rehearsal_after_lock_contention','existing_plan_sha256':digest(PLAN)})
        return rehearse_prepared(campaign=campaign)
    campaign=authenticate_cached(args,launcher);jobs=selected(campaign);handoff=handoff_guard(require_ready=False)
    status=c.status(campaign,JOURNAL)
    require(status['reserved_unfinished_slots']<=4 and not status['issues'] and not status['unknown'] and not status['needs_operator_review_job_ids'],'Existing E122 controller needs reconciliation')
    require(all(j['job_id'] in status['staged_held_job_ids'] for j in jobs),'Selected Graph pair is no longer held')
    audits={j['job_id']:launcher.audit_held(int(j['job_id']),j['cell']) for j in jobs}
    writers=recovery.active_writers()
    for job in jobs:
        require(writers.get(str(Path(job['cell']['run_dir']).resolve()),set())<={int(job['job_id'])},'Unexpected same-cell writer')
    storage=storage_proof(PROOF)
    feasibility={}
    for job in jobs:
        trial=[x for x in job['cell']['command'] if x!='--hold' and not x.startswith(('--nodelist=','--mem='))]
        trial.insert(1,'--test-only');trial.insert(-1,'--nodelist='+POOL);trial.insert(-1,'--mem=40G')
        result=command(trial,timeout=10);require(result.returncode==0,'Scheduler rejected same-ID placement feasibility')
        feasibility[job['job_id']]={'stdout':result.stdout,'stderr':result.stderr}
    import e122_slurm_2511_compat as compatibility
    import e122_shared_storage_admission as shared_storage
    import amend_e122_countdown_peers_node208_20260910 as prior_route
    interval=verified_watch_interval()
    paths=[SOURCE,TEST,PROTOCOL,RESOURCE_PROOF,QUALIFICATION,WATCH_COMMAND,HANDOFF/'plan.json',HANDOFF/'supervisor.json',ROOT/'ops/exp_scaling/guard_e118_e122_storage_handoff_20260911.py',Path(shared_storage.__file__),Path(shared_storage.base.__file__),Path(compatibility.__file__),compatibility.TEST,Path(prior_route.__file__),Path(base.__file__),Path(recovery.__file__),Path(launcher.__file__),Path(c.__file__),launcher.PLAN,launcher.LEDGER,JOURNAL/'context.json']
    plan={'schema':'e122-two-graph43-release-v1','created_at':now(),'binding':campaign['binding'],'job_ids':list(TARGETS),
        'held_audits':audits,'storage_handoff':handoff,'source_pins':{str(p):digest(p) for p in paths},'storage_evidence':storage,
        'storage_proof_path':str(PROOF),'new_nodes':sorted(NEW_NODES),'feasibility':feasibility,
        'temporary_cap':6,'persistent_cap':4,'post_release_memory':'40G','parent_lock_seconds':TOTAL_LIMIT,
        'legacy_watch_interval_seconds':interval,'publication_age_limit_seconds':HEARTBEAT_LIMIT,'post_open_publication_age_limit_seconds':LOCK_OPEN_HEARTBEAT_LIMIT,
        'scientific_configuration_unchanged':True,'new_training_submissions':0}
    atomic_new(PLAN,plan);emit({'event':'two_cell_plan_prepared','plan':str(PLAN),'sha256':digest(PLAN),'job_ids':list(TARGETS)})
    rehearse_prepared(campaign=campaign)

def rehearse_prepared(*,campaign):
    import e122_shared_storage_admission as shared_storage
    with STORAGE_LOCK.open('a') as shared:
        fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=verify_plan();proof=shared_storage.storage_report()
        require(proof.get('allowed') is True and not proof.get('errors'),'Rehearsal budget is not admitted')
        proof_path=ART/'preparation_storage.json';atomic_new(proof_path,proof)
        packet={'plan':plan,'campaign':campaign,'plan_sha256':digest(PLAN),'storage_proof_path':str(proof_path),'storage_proof_sha256':digest(proof_path),'purpose':'readonly'}
        request=ART/'preparation_request.json';atomic_new(request,packet)
        child=spawn_worker(request,digest(request),'dry-run');events=[]
        try:
            require(collect(child,time.monotonic()+45,events)=='ready','Prepared worker initialization failed')
            events.clear();child.stdin.write(b'GO\n');child.stdin.flush()
            require(collect(child,time.monotonic()+10,events)=='exited' and child.returncode==0,'Prepared read-only sequence exceeded safety window')
            timing=next(e for e in events if e.get('event')=='readonly_sequence_complete')
            require(timing['elapsed_seconds']<=6.5,'Prepared read-only sequence lacks execution margin')
            atomic_new(ART/'preparation_rehearsal.json',{'at':now(),'events':events,'scheduler_mutations':False,'plan_sha256':digest(PLAN)})
            emit({'event':'prepared_rehearsal_passed','elapsed_seconds':timing['elapsed_seconds'],'scheduler_mutations':False})
        finally:stop_worker(child)

def verify_plan():
    plan=read(PLAN)
    require(plan['job_ids']==list(TARGETS) and plan['temporary_cap']==6 and plan['persistent_cap']==4,'Two-cell plan scope changed')
    require(plan['new_nodes']==sorted(NEW_NODES),'Unapproved placement pool')
    require(all(digest(p)==sha for p,sha in plan['source_pins'].items()),'Frozen amendment or campaign input changed')
    require(plan['legacy_watch_interval_seconds']==verified_watch_interval(),'Original watch interval changed')
    require(plan['publication_age_limit_seconds']==HEARTBEAT_LIMIT and plan['post_open_publication_age_limit_seconds']==LOCK_OPEN_HEARTBEAT_LIMIT,'Reviewed publication age limits changed')
    return plan

class HeartbeatPublicationPending(RuntimeError):
    pass

def publication_stat(path):
    result=command(['stat','--cached=never','--format=%s|%y',str(path)],timeout=2)
    require(result.returncode==0,'Cannot refresh watcher publication metadata from server')
    fields=result.stdout.strip().split('|',1)
    require(len(fields)==2 and fields[0].isdigit(),'Malformed watcher publication metadata')
    require(re.fullmatch(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{9} [+-]\d{4}',fields[1]) is not None,'Malformed raw watcher file mtime')
    return int(fields[0]),fields[1]

def heartbeat():
    directory=JOURNAL/'status'
    fresh=command(['stat','--cached=never','--format=%Y',str(directory)],timeout=2)
    require(fresh.returncode==0,'Cannot refresh immutable status directory from server')
    paths=sorted(directory.glob('*.json'))
    require(paths,'Original watcher has no immutable status publication')
    path=paths[-1]
    require(re.fullmatch(r'\d{8}T\d{6}\.\d{6}Z_[a-f0-9]{32}\.json',path.name) is not None,'Unexpected immutable status filename')
    publication=datetime.strptime(path.name.split('_',1)[0],'%Y%m%dT%H%M%S.%fZ').replace(tzinfo=timezone.utc)
    for _ in range(3):
        before=publication_stat(path);length=before[0]
        with path.open('rb') as stream:published=stream.read(length)
        after=publication_stat(path)
        if before!=after:continue  # A publisher write raced this read-only observation.
        if len(published)!=length or not published.endswith(b'\n'):
            raise HeartbeatPublicationPending('Immutable status publication is incomplete')
        data=json.loads(published);snapshot=datetime.fromisoformat(data['observed_at'])
        require(snapshot.timestamp()<=publication.timestamp()+.001,'Scheduler snapshot postdates publication stamp')
        require(data.get('schema')=='e122_level3_release_status_v1' and data['binding']==read(PLAN)['binding'],'Original heartbeat campaign binding differs')
        require(data['max_released_nonterminal']==4 and not data['issues'] and not data['unknown'] and not data['needs_operator_review_job_ids'],'Original controller is not healthy')
        age=(datetime.now(timezone.utc)-publication).total_seconds()
        data['_heartbeat_publication']={'path':str(path),'filename_stamp':publication.isoformat(),'file_mtime':after[1],'size':length,'snapshot_to_publication_seconds':(publication-snapshot).total_seconds()}
        return data,age
    raise HeartbeatPublicationPending('Immutable status changed repeatedly during read-only observation')

def spawn_worker(request,request_sha,mode):
    return subprocess.Popen(['/usr/bin/python3','-B',str(SOURCE),'--worker',mode,'--request',str(request),'--request-sha256',request_sha],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=False,close_fds=True,start_new_session=True)

def collect(process,deadline,events):
    """Bound waiting using only local nonblocking pipe IO and the monotonic clock."""
    buffers={process.stdout:b'',process.stderr:b''}
    while time.monotonic()<deadline:
        streams=[s for s in buffers if not s.closed]
        if streams:
            ready,_,_=select.select(streams,[],[],min(.1,max(0,deadline-time.monotonic())))
            for stream in ready:
                chunk=os.read(stream.fileno(),65536)
                if not chunk:stream.close();continue
                buffers[stream]+=chunk
                while b'\n' in buffers[stream]:
                    line,buffers[stream]=buffers[stream].split(b'\n',1)
                    if stream is process.stdout:
                        try:events.append(json.loads(line))
                        except ValueError:events.append({'event':'invalid_worker_output','raw':line.decode(errors='replace')})
                    else:events.append({'event':'worker_stderr','text':line.decode(errors='replace')})
        if events and events[-1].get('event')=='worker_ready':return 'ready'
        if process.poll() is not None:
            # Give already-readable final lines one more iteration before return.
            if not any(not s.closed for s in buffers):return 'exited'
    return 'timeout'

def stop_worker(process):
    if process.poll() is None:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    # No wait()/communicate() here: a killed process in uninterruptible NFS IO
    # may remain briefly alive, but never inherited the parent's flock.

def apply(*,campaign,args,launcher):
    with STORAGE_LOCK.open('a') as shared:
        fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return apply_under_storage_lock(campaign=campaign,args=args,launcher=launcher)

def apply_under_storage_lock(*,campaign,args,launcher):
    import control_e122_level3_release as c
    plan=verify_plan();require(not (ART/'execution.intent.json').exists(),'Two-cell attempt already claimed; reconcile rather than retry')
    campaign=authenticate_cached(args,launcher);require(campaign['binding']==plan['binding'],'Campaign binding changed')
    selected(campaign);handoff_guard();storage_proof(PROOF)
    import e122_shared_storage_admission as shared_storage
    fresh_storage=shared_storage.storage_report()
    require(fresh_storage.get('allowed') is True and not fresh_storage.get('errors'),'Fresh shared budget no longer admits the pair')
    fresh_path=ART/'storage_at_final_admission.json';atomic_new(fresh_path,fresh_storage)
    storage_proof(fresh_path)
    packet={'plan':plan,'campaign':campaign,'plan_sha256':digest(PLAN),'storage_proof_path':str(fresh_path),'storage_proof_sha256':digest(fresh_path),'purpose':'apply'}
    request=ART/'worker_request.json';atomic_new(request,packet);request_sha=digest(request)
    dry=spawn_worker(request,request_sha,'dry-run');dry_events=[]
    try:
        require(collect(dry,time.monotonic()+45,dry_events)=='ready','Read-only worker initialization failed')
        dry_events.clear();dry.stdin.write(b'GO\n');dry.stdin.flush()
        require(collect(dry,time.monotonic()+10,dry_events)=='exited' and dry.returncode==0,'Read-only worker sequence exceeded safety window')
        measured=next(e for e in dry_events if e.get('event')=='readonly_sequence_complete')
        require(measured['elapsed_seconds']<=6.5,'Read-only critical sequence lacks execution deadline margin')
        atomic_new(ART/'readonly_sequence_timing.json',{'at':now(),'events':dry_events})
        emit({'event':'readonly_worker_sequence_passed','elapsed_seconds':measured['elapsed_seconds']})
    finally:stop_worker(dry)
    worker=spawn_worker(request,request_sha,'release');events=[]
    try:
        require(collect(worker,time.monotonic()+45,events)=='ready','Release worker did not finish read-only initialization')
        events.clear()
        wait_end=time.monotonic()+90
        while True:
            try:hb,age=heartbeat()
            except HeartbeatPublicationPending:
                require(time.monotonic()<wait_end,'No complete original publication; leave held jobs untouched')
                time.sleep(.2);continue
            if heartbeat_is_fresh(age):break
            require(time.monotonic()<wait_end,'No fresh original heartbeat; leave held jobs untouched')
            time.sleep(.2)
        require(hb['reserved_unfinished_slots']<=4,'Legacy released slots changed before burst')
        heartbeat_seen=time.monotonic()
        # Open before locking; FD is non-inheritable, and both Popen calls use
        # close_fds=True. Parent critical work never reads campaign files.
        with (JOURNAL/'.lock').open('a') as lock:
            os.set_inheritable(lock.fileno(),False)
            require(heartbeat_is_fresh(age+time.monotonic()-heartbeat_seen,after_open=True),'Lock file open exceeded fresh heartbeat window; no mutation')
            while True:
                require(heartbeat_is_fresh(age+time.monotonic()-heartbeat_seen,after_open=True),'Publisher did not finish within the verified heartbeat window')
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(.05)
            acquired=time.monotonic();hard_end=acquired+TOTAL_LIMIT
            try:
                require(heartbeat_is_fresh(age+acquired-heartbeat_seen,after_open=True),'Lock acquisition exceeded verified publication window; no mutation')
                worker.stdin.write(b'GO\n');worker.stdin.flush()
                outcome=collect(worker,min(hard_end,acquired+PRIMARY_LIMIT),events)
                if outcome!='exited' or worker.returncode!=0:
                    stop_worker(worker)
                    # Retry no scheduler mutation. A separate bounded child can
                    # finish missing receipts only from positive scheduler proof.
                    rescue=spawn_worker(request,request_sha,'reconcile')
                    rescue_events=[]
                    try:
                        ready=collect(rescue,min(hard_end,time.monotonic()+3),rescue_events)
                        if ready=='ready':
                            rescue_events.clear()
                            acknowledgements=[e['result'] for e in events if e.get('event')=='scheduler_acknowledgement']
                            rescue.stdin.write(b'GO\n'+json.dumps({'acknowledgements':acknowledgements}).encode()+b'\n');rescue.stdin.flush()
                            collect(rescue,min(hard_end,time.monotonic()+RECOVERY_LIMIT),rescue_events)
                        events.extend(rescue_events)
                    finally:stop_worker(rescue)
            finally:
                stop_worker(worker)
                fcntl.flock(lock,fcntl.LOCK_UN)
        elapsed=time.monotonic()-acquired
        atomic_new(ART/'parent_execution_result.json',{'at':now(),'lock_elapsed_seconds':elapsed,'events':events,'original_heartbeat':hb['observed_at'],'completed_heartbeat_publication':hb.get('_heartbeat_publication')})
        require(any(e.get('event') in ('release_pair_complete','reconciliation_complete') for e in events),'Bounded release did not fully reconcile; inspect consumed intents, never repeat blindly')
        # Route mutation stays outside the legacy controller lock and only
        # follows positive release ownership. Original held routes were audited.
        route_pair(campaign)
        emit({'event':'two_cell_release_complete','job_ids':list(TARGETS),'lock_elapsed_seconds':elapsed,'original_controller_cap':4})
    finally:stop_worker(worker)

def actual_acknowledgement(ack,jid,intent_sha):
    require(ack is not None and ack.get('schema')=='e122_level3_release_result_v1'
            and ack.get('job_id')==jid and ack.get('returncode')==0 and not ack.get('error')
            and ack.get('intent_sha256')==intent_sha and ack.get('command')==['scontrol','release',jid],
            'No actual successful process acknowledgement; preserve consumed intent for operator review')
    return ack

def worker(mode,request,expected):
    global PROOF
    require(digest(request)==expected,'Worker request pin changed');packet=read(request)
    PROOF=Path(packet['storage_proof_path'])
    require(PROOF in (ART/'storage_at_final_admission.json',ART/'preparation_storage.json'),'Unexpected worker storage proof path')
    require(mode=='dry-run' or packet.get('purpose')=='apply','Read-only preparation cannot authorize release')
    import control_e122_level3_release as c
    import launch_e122_level3_factorial as launcher
    import e122_slurm_2511_compat as compat
    watch=read(ROOT/'var/artifacts/e122_level3_factorial/root_watch_arm_result.json')['command']
    compat.install_compat(launcher,watch[watch.index('--compat-source-sha256')+1],watch[watch.index('--compat-test-sha256')+1])
    plan=verify_plan();require(packet['plan']==plan and packet['plan_sha256']==digest(PLAN),'Worker campaign plan differs')
    require(packet['storage_proof_sha256']==digest(PROOF),'Storage proof changed during worker launch')
    campaign=packet['campaign'];require(campaign['binding']==plan['binding'],'Worker campaign binding differs')
    jobs=selected(campaign);c.MAX_ACTIVE=6;c.command=lambda argv:command(argv,timeout=2)
    emit({'event':'worker_ready','mode':mode})
    require(sys.stdin.buffer.readline()==b'GO\n','Worker never received its owned-lock authorization')
    if mode=='reconcile':
        acknowledgements=json.loads(sys.stdin.buffer.readline())['acknowledgements']
        ack_by_job={r['job_id']:r for r in acknowledgements}
        for job in jobs:
            jid=job['job_id'];intent=JOURNAL/'jobs'/f'{jid}.intent.json';result=intent.with_name(f'{jid}.result.json')
            if not intent.exists():continue
            if result.exists():continue
            owned=read(intent);require(owned['binding']==campaign['binding'] and owned['job_id']==jid and owned.get('operational_amendment_sha256')==digest(PLAN),'Unowned release intent')
            ack=actual_acknowledgement(ack_by_job.get(jid),jid,digest(intent))
            observed=c.scheduler_snapshot([jid])['jobs'][jid]
            require(not observed['unknown'] and not observed['held'],'Actual acknowledgement conflicts with current scheduler state')
            atomic_new(result,{**ack,'recovered_actual_process_acknowledgement':True,'recovery_at':now(),'scheduler_evidence':observed})
        status=c.status(campaign,JOURNAL)
        require(not status['issues'] and not status['unknown'] and not status['needs_operator_review_job_ids'],'Reconciliation requires operator review')
        require(all(j['job_id'] in status['released_nonterminal_job_ids'] or j['job_id'] in status['endpoint_evidence'] for j in jobs),'Pair only partially released; no automatic repeat')
        emit({'event':'reconciliation_complete','released':list(TARGETS)});return
    if mode=='dry-run':
        started=time.monotonic();storage_proof(PROOF);c.status(campaign,JOURNAL)
        for job in jobs:
            before=c.status(campaign,JOURNAL)
            require(job['job_id'] in before['staged_held_job_ids'] and not before['issues'] and not before['unknown'],'Read-only held status changed')
            handoff_guard(require_ready=False);storage_proof(PROOF,allow_targets=TARGETS)
            raw=command(['scontrol','show','job','-dd','-o',job['job_id']],timeout=2)
            require(raw.returncode==0,'Read-only exact held readback failed')
            launcher.audit_held_record(raw.stdout,int(job['job_id']),job['cell'])
        c.status(campaign,JOURNAL)
        emit({'event':'readonly_sequence_complete','elapsed_seconds':time.monotonic()-started,'job_ids':list(TARGETS)});return
    atomic_new(ART/'execution.intent.json',{'at':now(),'plan_sha256':digest(PLAN),'job_ids':list(TARGETS),'source_sha256':digest(SOURCE)})
    storage_proof(PROOF)
    for job in jobs:
        jid=job['job_id'];before=c.status(campaign,JOURNAL)
        require(not before['issues'] and not before['unknown'] and not before['needs_operator_review_job_ids'],'E122 status requires review')
        require(before['reserved_unfinished_slots']<6 and before['storage']['can_release'],'Original E122 storage/cap blocks release')
        require(jid in before['staged_held_job_ids'],'Graph job is no longer staged held')
        handoff_guard(require_ready=False);storage_proof(PROOF,allow_targets=TARGETS)
        before['shared_storage_admission']={'path':str(PROOF),'sha256':digest(PROOF)}
        record=command(['scontrol','show','job','-dd','-o',jid],timeout=2);require(record.returncode==0,'Exact held readback failed')
        held=launcher.audit_held_record(record.stdout,int(jid),job['cell'])
        intent=JOURNAL/'jobs'/f'{jid}.intent.json';argv=['scontrol','release',jid]
        atomic_new(intent,{'schema':'e122_level3_release_intent_v1','created_at':now(),'binding':campaign['binding'],'job_id':jid,'cell':job['cell'],'held_scheduler_record':held,'command':argv,'decision':before,'operational_amendment_sha256':digest(PLAN),'operational_amendment_plan':str(PLAN)})
        result={'schema':'e122_level3_release_result_v1','job_id':jid,'intent_sha256':digest(intent),'command':argv}
        try:
            answer=command(argv,timeout=2);result.update(returncode=answer.returncode,stdout=answer.stdout,stderr=answer.stderr,error=None)
        except Exception as exc:result.update(returncode=None,error=f'{type(exc).__name__}: {exc}')
        result['recorded_at']=now();emit({'event':'scheduler_acknowledgement','result':result})
        atomic_new(JOURNAL/'jobs'/f'{jid}.result.json',result)
        require(result['returncode']==0 and not result.get('error'),'Uncertain release; consume intent and stop')
    after=c.status(campaign,JOURNAL)
    require(not after['issues'] and not after['unknown'] and not after['needs_operator_review_job_ids'],'Post-release controller requires review')
    emit({'event':'release_pair_complete','released':list(TARGETS),'reserved_unfinished_slots':after['reserved_unfinished_slots']})

def route_pair(campaign):
    import prioritize_e118_capacity_20260905 as base
    import amend_e122_countdown_peers_node208_20260910 as prior
    for job in selected(campaign):
        jid=job['job_id'];before=base.show(int(jid))
        state=base.field(before,'JobState')
        if state!='PENDING':
            emit({'event':'preserved_existing_allocation','job_id':jid,'state':state});continue
        require(base.field(before,'Priority')!='0' and base.field(before,'Reason') not in ('JobHeldUser','JobHeldAdmin'),'Preserve an existing graph hold')
        nodes=set(command(['scontrol','show','hostnames',base.field(before,'ReqNodeList')]).stdout.split())
        require(nodes==OLD_NODES,'Unexpected existing Graph route')
        for nodename in ('node105','node203','node204'):
            node=command(['scontrol','show','node','-o',nodename]);require(node.returncode==0,'Cannot inspect '+nodename)
            require('gpu:a5000:' in base.field(node.stdout,'Gres'),'Approved A5000 hardware changed: '+nodename)
        require(base.field(before,'MinMemoryNode')=='64G','Original Graph64GiB request changed')
        argv=['scontrol','update',f'JobId={jid}',f'ReqNodeList={POOL}','MinMemoryNode=40G'];intent=ART/f'{jid}.route.intent.json'
        atomic_new(intent,{'job_id':jid,'before':before,'command':argv,'at':now()})
        current=base.show(int(jid))
        if base.field(current,'JobState')!='PENDING':
            atomic_new(ART/f'{jid}.route.preserved_started.json',{'job_id':jid,'at':now(),'record':current,'scheduler_update_attempted':False})
            emit({'event':'preserved_started_before_resource_update','job_id':jid});continue
        require(base.submit_tokens(current)==base.submit_tokens(before) and all(base.field(current,k)==base.field(before,k) for k in prior.PRESERVE),'Pending resource request changed before update')
        result=command(argv,timeout=5);atomic_new(ART/f'{jid}.route.result.json',{'job_id':jid,'intent_sha256':digest(intent),'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'at':now()})
        after=base.show(int(jid));require(base.submit_tokens(before)==base.submit_tokens(after),'Frozen SubmitLine changed')
        if result.returncode!=0 and base.field(after,'JobState') in ('RUNNING','CONFIGURING'):
            require(all(base.field(after,k)==base.field(before,k) for k in prior.PRESERVE),'Rejected pending-only update changed running resources')
            atomic_new(ART/f'{jid}.route.preserved_started.json',{'job_id':jid,'at':now(),'record':after,'scheduler_update_attempted':True,'scheduler_rejected_running_memory_update':True})
            emit({'event':'preserved_start_racing_resource_update','job_id':jid});continue
        require(all(base.field(before,k)==base.field(after,k) for k in prior.PRESERVE if k not in ('MinMemoryNode','ReqTRES')),'Scientific/runtime/resource scheduler field changed')
        require(base.field(after,'MinMemoryNode')=='40G','Unapproved memory change')
        before_tres=dict(x.split('=',1) for x in base.field(before,'ReqTRES').split(','))
        after_tres=dict(x.split('=',1) for x in base.field(after,'ReqTRES').split(','))
        require(before_tres.get('mem')=='64G' and after_tres.get('mem')=='40G','Memory TRES change differs')
        require({k:v for k,v in before_tres.items() if k not in ('mem','billing')}=={k:v for k,v in after_tres.items() if k not in ('mem','billing')},'Unapproved CPU/GPU/node TRES change')
        require(int(after_tres.get('billing',0))<=int(before_tres.get('billing',0)),'Derived billing unexpectedly increased')
        nodes=set(command(['scontrol','show','hostnames',base.field(after,'ReqNodeList')]).stdout.split())
        require(result.returncode==0 and nodes==NEW_NODES,'Qualified A5000 pool and40GiB amendment not acknowledged')
        atomic_new(ART/f'{jid}.route.readback.json',{'job_id':jid,'at':now(),'record':after})
        emit({'event':'qualified_graph40_route_verified','job_id':jid,'state':base.field(after,'JobState')})

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',choices=('release','reconcile','dry-run'),required=True);parser.add_argument('--request',type=Path,required=True);parser.add_argument('--request-sha256',required=True);a=parser.parse_args()
    try:worker(a.worker,a.request,a.request_sha256)
    except Exception as exc:emit({'event':'worker_error','error':f'{type(exc).__name__}: {exc}'});raise
