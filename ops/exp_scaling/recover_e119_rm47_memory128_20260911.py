#!/usr/bin/env python3
"""Prepare-only by default; explicit phases implement one guarded RAM repair."""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickletools
import re
import subprocess
import time
import zipfile

import amend_e119_max44_node208_20260910 as h
import recover_e119_memory_pressure_20260905 as memory

ROOT = h.ROOT
TARGET = 31163710
OLD_CPU = 31194078
ART = ROOT / 'var/artifacts/gpu_progress_check_20260911/rm47_memory128'
PLAN, TX = ART / 'plan.json', ART / 'transaction.json'
PROTOCOL = ROOT / 'paper/preregistration/e119_rm47_memory128_20260911.md'
OLD_ART = ROOT / 'var/artifacts/e119_pending208_completion_20260910/guard'
OLD_PLAN, OLD_TX = OLD_ART / 'plan.json', OLD_ART / 'transaction.json'
OLD_LOCK = ROOT / 'var/artifacts/e119_healthy_completion_guard_20260909/singleton.lock'
NEW = ART / 'guard'
NODES = 'node205,node207,node208,node302'
g, b, recovery = h.guard, h.b, h.recovery
require, atomic = g.require, g.atomic
for key in ('ROOT', 'TARGET', 'OLD_CPU', 'ART', 'PLAN', 'TX', 'PROTOCOL', 'OLD_ART', 'OLD_PLAN', 'OLD_TX', 'OLD_LOCK', 'NEW', 'NODES'):
    setattr(h, key, globals()[key])
h.__file__ = str(Path(__file__).resolve())


def sha(path):
    return recovery.digest(path)


def load():
    plan, tx = h.load()
    old = json.loads(OLD_PLAN.read_text())
    g.before_deadline(json.loads(OLD_TX.read_text()))
    require(all(sha(path) == digest for path, digest in old['helper_sha256'].items()), 'Inherited guard helper drift')
    require(sha(g.PRIMARY) == old['primary_sha256'], 'Original scientific ledger drift')
    for path, digest in plan['resume_source_pins'].items():
        require(sha(path) == digest, 'Resume source changed: ' + path)
    return plan, tx


def state(item, *, amended, held, same_attempt=False):
    record = g.show(TARGET)
    expected = copy.deepcopy(item)
    if amended:
        expected['resources']['MinMemoryNode'] = '128G'
    g.stable(expected, record)
    plan = json.loads(PLAN.read_text())
    if held:
        require(b.field(record, 'JobState') == 'PENDING' and b.field(record, 'Priority') == '0'
                and b.field(record, 'Reason') == 'job_requeued_in_held_state', 'Exact owned requeue hold missing')
        require(int(b.field(record, 'Restarts')) == int(plan['target_restarts']) + 1, 'Unexpected restart count')
    if same_attempt:
        for key in ('JobState', 'NodeList', 'StartTime', 'Restarts'):
            require(b.field(record, key) == b.field(plan['target_before'], key), 'Target attempt changed: ' + key)
    g.writer_check(item)
    return record


# The reused activation path validates our owned requeue hold and memory profile.
h.target_record = state


def partial_manifest(path):
    path = Path(path)
    require(path.is_dir() and not path.is_symlink() and path.resolve() == path, 'Quarantine path must be an exact real directory')
    files = sorted(path.iterdir())
    require(len(files) == 2 and all(p.is_file() and not p.is_symlink() and p.suffix == '.pt' for p in files), 'Unexpected partial checkpoint contents')
    return {p.name: {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns, 'inode': p.stat().st_ino} for p in files}


def checkpoint_files(path):
    result = {}
    for p in sorted(Path(path).glob('*.pt')):
        s = p.stat()
        with zipfile.ZipFile(p) as z:
            info = next(i for i in z.infolist() if i.filename.endswith('/data.pkl'))
            require(info.file_size < 4 * 1024**2, 'Unbounded checkpoint metadata')
            data = z.read(info)
        if p.name.endswith('_model_states.pt'):
            require(any(arg == 'online_canonical_bank_state' for _, arg, _ in pickletools.genops(data)), 'Replay bank missing')
        result[p.name] = {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns,
                          'inode': s.st_ino, 'metadata_sha256': hashlib.sha256(data).hexdigest()}
    return result


def checkpoint(plan, *, quarantined=False):
    run = plan['item']['identity']
    d = memory.timing_and_checkpoint(TARGET, run)
    step = plan['resume_step']
    require(d['checkpoint_step'] == step and d['checkpoint'] == plan['checkpoint']['checkpoint'],
            'Selected checkpoint changed; prepare a new review rather than rolling back newer work')
    require(d['saved_counter_validation']['saved_counters'] == dict.fromkeys(
        ('global_steps', 'global_step', 'prompt_batches_consumed_total'), step), 'Saved counters disagree')
    require(checkpoint_files(d['checkpoint']) == plan['checkpoint_files'], 'Durable checkpoint shard changed')
    require(d['current_step'] == plan['checkpoint']['current_step'], 'Training advanced; preserve new work')
    expected = set() if quarantined else set(plan['checkpoint']['rejected_checkpoints'])
    require(set(d['rejected_checkpoints']) == expected, 'Partial checkpoint set changed')
    if expected:
        require(step == 768 and expected == {plan['partial_path']}, 'Unreviewed partial checkpoint')
        require('BadZipFile' in ' '.join(d['rejected_checkpoints'][plan['partial_path']]), 'Partial failure changed')
    log = Path(plan['item']['resources']['StdOut'])
    with log.open('rb') as stream:
        stream.seek(max(0, log.stat().st_size - 200000)); tail = stream.read().decode(errors='replace')
    learned = [int(x) for x in re.findall(r'post-learning start step=(\d+)', tail)]
    require(learned and max(learned) <= step + plan['allow_repeat_updates'], 'Unlogged optimizer progress exceeds reviewed rollback')
    ops = Path(plan['resume_ops'])
    selected = b.command(['/usr/bin/python3', str(ops / 'validate_deepspeed_checkpoint.py'),
                          '--select-under', run['run_dir']]).stdout.strip()
    require(selected == d['checkpoint'], 'Frozen runtime auto-resume selects a different checkpoint')
    require(not recovery.complete(Path(run['run_dir'])), 'Target completed; do not restart')
    return d


def pressure():
    code = """from pathlib import Path
import json,time
p=Path('/sys/fs/cgroup/system.slice/slurmstepd.scope/job_31163710')
rows=[]
for i in range(2):
 s=dict(line.split() for line in (p/'memory.stat').read_text().splitlines())
 e=dict(line.split() for line in (p/'memory.events').read_text().splitlines())
 rows.append({'at':time.time(),'memory.current':int((p/'memory.current').read_text()),'memory.high':int((p/'memory.high').read_text()),'noncache_bytes':sum(int(s.get(k,0)) for k in ['anon','shmem','kernel']),'events':{k:int(v) for k,v in e.items()}})
 if i==0:time.sleep(5)
print(json.dumps(rows))"""
    p = subprocess.run(['timeout','-k','3s','25s','srun',f'--jobid={TARGET}','--overlap','--exact',
        '--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=0','--gres=none','/usr/bin/python3','-c',code],
        capture_output=True,text=True,check=True,timeout=32)
    rows = json.loads(p.stdout)
    for row in rows:
        require(row['memory.high'] == 116 * 2**30 and 116 * 2**30 < row['noncache_bytes'] < 128 * 2**30,
                'Pressure or reviewed128GiB headroom changed')
        require(row['events']['oom'] == row['events']['oom_kill'] == 0, 'OOM requires fresh review')
    require(rows[1]['events']['high'] > rows[0]['events']['high'], 'No ongoing memory throttling')
    return rows


def cpu_command():
    return ['sbatch','--parsable','--hold','--job-name=e119-rm47-memory-guard','--account=mltheory',
        '--partition=lowprio','--nodelist=node915,node917','--nodes=1','--ntasks=1','--cpus-per-task=2',
        '--mem=2G','--gres=none','--time=1-01:10:00','--requeue','--export=NONE',f'--chdir={ROOT}',
        f'--output={NEW}/supervisor-%j.out',f'--error={NEW}/supervisor-%j.err',
        '--comment=e119-rm47-memory128-20260911',str(NEW/'supervisor.slurm')]


def prepare(args):
    require(not PLAN.exists() and not TX.exists(), 'Existing immutable preparation; inspect it first')
    old = json.loads(OLD_PLAN.read_text())
    require(len(old['rows']) == 10, 'Inherited guard scope changed')
    item = copy.deepcopy(next(r for r in old['rows'] if r['job_id'] == TARGET))
    require((item['identity']['domain'],item['identity']['arm'],item['identity']['seed']) == ('pantry_plan','replay_maxrl',47), 'Wrong scientific cell')
    before = g.show(TARGET); g.stable(item,before); g.writer_check(item)
    require(b.field(before,'JobState') == 'RUNNING' and b.field(before,'MinMemoryNode') == '116G', 'Expected running116GiB target missing')
    require(b.field(before,'UserId').endswith(f'({os.getuid()})'), 'Target owner differs')
    d = memory.timing_and_checkpoint(TARGET,item['identity'])
    step = args.resume_step
    require(d['checkpoint_step'] == step, 'Requested checkpoint not latest structurally complete; preserve newer work')
    partial = str(Path(item['identity']['run_dir']) / f'debug_job{TARGET}/checkpoints/step_00864')
    if step == 768:
        require(args.allow_repeat_updates == 96 and d['current_step'] == 863 and set(d['rejected_checkpoints']) == {partial},
                'Step768 fallback requires explicit96-update repeat and exact incomplete864 boundary')
    else:
        require(args.allow_repeat_updates == 0 and d['current_step'] in (863,864) and not d['rejected_checkpoints'],
                'Zero-loss864 recovery requires current boundary and no partial checkpoint')
    mapping = g.ledger_mapping(); require(mapping[TARGET]['identity'] == item['identity'], 'Ledger identity differs')
    cpu = g.show(OLD_CPU); registration = json.loads((OLD_ART/'supervisor.json').read_text())
    require(registration['job_id'] == OLD_CPU and registration['plan_sha256'] == sha(OLD_PLAN), 'Old CPU registration differs')
    require(b.submit_tokens(cpu) == registration['submit_tokens'] and all(b.field(cpu,k) == v for k,v in registration['resources'].items()), 'Old CPU identity/resources differ')
    require(b.field(cpu,'JobState') == 'RUNNING', 'Old guard inactive; inspect ownership before handoff')
    env = b.exports(item['submit_tokens']); ops = Path(env['OAT_ZERO_OPS_SNAPSHOT_ROOT']); src = Path(env['OAT_ZERO_SOURCE_ROOT'])
    require(env.get('OAT_ZERO_AUTO_RESUME') == '1' and not env.get('OAT_ZERO_RESUME_DIR') and not env.get('OAT_ZERO_INITIAL_RESUME_DIR'), 'Resume environment differs')
    pins = [ops/'run_experiment.sh',ops/'train.sh',ops/'validate_deepspeed_checkpoint.py',
            src/'oat_drgrpo/learner/run.py', ROOT/'var/seed_paper_eval/paper310/lib/python3.10/site-packages/oat/utils/deepspeed.py']
    plan = {'schema':'e119-rm47-memory128-guard-handoff-v1','created_at_utc':b.now(),'job_id':TARGET,'item':item,
        'target_before':before,'target_restarts':b.field(before,'Restarts'),'resume_step':step,'allow_repeat_updates':args.allow_repeat_updates,
        'checkpoint':d,'checkpoint_files':checkpoint_files(d['checkpoint']),'partial_path':partial if step == 768 else None,
        'quarantine_path':str(ART/'quarantine/step_00864') if step == 768 else None,
        'resume_ops':str(ops),'resume_source_pins':{str(p):sha(p) for p in pins},
        'ledger_row_before':mapping[TARGET]['row'],'old_cpu_job_id':OLD_CPU,'old_cpu_submit_tokens':b.submit_tokens(cpu),
        'old_cpu_resources':{k:b.field(cpu,k) for k in h.CPU_FIELDS},'old_cpu_restarts':b.field(cpu,'Restarts'),
        'old_plan_sha256':sha(OLD_PLAN),'controller_sha256':sha(__file__),'protocol_sha256':sha(PROTOCOL),
        'cpu_submit_tokens':cpu_command(),'scheduler_mutations':False}
    g.before_deadline(h.final_old_state(plan)); checkpoint(plan); plan['pressure'] = pressure()
    require(all(sha(path) == digest for path,digest in old['helper_sha256'].items()), 'Inherited guard helper drift')
    require(g.runtime.runtime_fingerprints(old['rows']) == old['runtime_fingerprints'], 'Inherited scientific runtime drift')
    # Preserve all scientific runtime fingerprints and all ten rows; edit one RAM field.
    successor = copy.deepcopy(old)
    next(r for r in successor['rows'] if r['job_id'] == TARGET)['resources']['MinMemoryNode'] = '128G'
    for path in (Path(__file__).resolve(),PROTOCOL): successor['helper_sha256'][str(path)] = sha(path)
    NEW.mkdir(parents=True,exist_ok=True)
    (NEW/'supervisor.slurm').write_text('#!/bin/bash\nset -euo pipefail\nexport PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n'+f'cd {ROOT}\nexec /usr/bin/python3 -u {Path(__file__).resolve()} watch\n')
    atomic(NEW/'plan.json',successor)
    proposal = {'command':cpu_command(),'script_sha256':sha(NEW/'supervisor.slurm'),'scheduler_mutations':False}
    atomic(NEW/'cpu_submission.json',proposal)
    plan.update(new_guard_plan_sha256=sha(NEW/'plan.json'),cpu_script_sha256=proposal['script_sha256'])
    atomic(PLAN,plan)
    state(item,amended=False,held=False,same_attempt=True); checkpoint(plan)
    print(json.dumps({'prepared':str(PLAN),'resume_step':step,'repeat_updates_bound':args.allow_repeat_updates,'scheduler_mutations':False}))


def check():
    plan,tx = load(); require(not tx.get('hold_intent'), 'Recovery already started; inspect transaction')
    state(plan['item'],amended=False,held=False,same_attempt=True); checkpoint(plan); h.old_cpu(plan,running=True)
    h.final_old_state(plan)
    print(json.dumps({'valid':True,'resume_step':plan['resume_step'],'scheduler_mutations':False}))


def submit():
    plan,tx = load(); check(); require(not tx.get('cpu_submission_intent'), 'Uncertain prior CPU submission; never repeat')
    tx['cpu_submission_intent'] = True; h.save(tx,'Persisted held successor CPU submission intent')
    receipt = b.command(plan['cpu_submit_tokens']).stdout.strip()
    require(re.fullmatch(r'\d+(;\S+)?',receipt), 'Uncertain CPU receipt; reconcile without repeating')
    tx.update(new_cpu_job_id=int(receipt.split(';')[0]),cpu_submission_receipt=receipt); h.save(tx,'Recorded successor CPU ID')
    h.new_cpu(plan,tx['new_cpu_job_id'],held=True)
    print(json.dumps({'held_cpu':tx['new_cpu_job_id']}))


def route():
    plan,tx = load(); item = plan['item']; h.new_cpu(plan,tx['new_cpu_job_id'],held=True)
    require(not tx.get('guard_state_copied'), 'Handoff already staged; use activate or inspect transaction')
    with g.LEDGER_LOCK.open('a+') as ledger:
        fcntl.flock(ledger,fcntl.LOCK_EX)
        if not tx.get('hold_intent'):
            state(item,amended=False,held=False,same_attempt=True); checkpoint(plan); h.old_cpu(plan,running=True); h.final_old_state(plan)
            tx['fresh_pressure'] = pressure()
            archive = ART/'before_stop'; archive.mkdir(exist_ok=True)
            tx['archives'] = memory.archive(TARGET,g.show(TARGET),plan['checkpoint'],archive)
            checkpoint(plan); state(item,amended=False,held=False,same_attempt=True)
            g.before_deadline(h.final_old_state(plan))
            tx['hold_intent'] = True; h.save(tx,'Persisted one target requeuehold intent; never repeat uncertain call')
            b.command(['scontrol','requeuehold',str(TARGET)])
        until=time.monotonic()+45
        while b.field(g.show(TARGET),'JobState') != 'PENDING':
            require(time.monotonic()<until,'Target requeue not acknowledged; inspect without repeating'); time.sleep(.5)
        actual_memory = b.field(g.show(TARGET),'MinMemoryNode')
        require(actual_memory in ('116G','128G') and (actual_memory == '116G' or tx.get('memory_intent')), 'Unexpected memory edit')
        state(item,amended=actual_memory == '128G',held=True)
        if not tx.get('old_cpu_stop_requested'):
            h.old_cpu(plan,running=True); h.final_old_state(plan)
            tx['old_cpu_stop_requested'] = True; h.save(tx,'Stopping only registered old CPU at ledger boundary')
            b.command(['scancel',str(OLD_CPU)])
        with OLD_LOCK.open('a+') as oldlock:
            until=time.monotonic()+45
            while True:
                if OLD_CPU not in b.queue():
                    try: fcntl.flock(oldlock,fcntl.LOCK_EX|fcntl.LOCK_NB); break
                    except BlockingIOError: pass
                require(time.monotonic()<until,'Old CPU stop not acknowledged; retain target hold'); time.sleep(.5)
            previous=h.final_old_state(plan)
            if not tx.get('old_final_tx_sha256'):
                tx['old_final_tx_sha256']=sha(OLD_TX); atomic(ART/'old_guard_transaction.final.json',previous)
                h.save(tx,'Frozen old guard transaction with unchanged retry histories/deadline')
            require(sha(OLD_TX)==tx['old_final_tx_sha256'],'Old guard changed after stop')
            # Preserve incomplete864 outside the run tree, after its writer stopped.
            if plan['partial_path'] and not tx.get('quarantined'):
                source,dest=Path(plan['partial_path']),Path(plan['quarantine_path'])
                if tx.get('quarantine_intent'):
                    require(not source.exists() and dest.is_dir(),'Uncertain quarantine; inspect rather than repeat')
                else:
                    checkpoint(plan); require(not dest.exists(),'Quarantine destination exists')
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    require(source.resolve() == source and dest.parent.resolve() == dest.parent and not source.is_symlink(), 'Quarantine path escape')
                    require(source.stat().st_dev == dest.parent.stat().st_dev, 'Quarantine rename must stay on same filesystem')
                    tx['partial_before_quarantine']=partial_manifest(source)
                    tx['quarantine_intent']=True; h.save(tx,'Persisted incomplete864 preservation by atomic directory rename')
                    source.rename(dest)
                    for directory in (source.parent,dest.parent):
                        fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
                        try: os.fsync(fd)
                        finally: os.close(fd)
                require(partial_manifest(dest) == tx['partial_before_quarantine'], 'Preserved checkpoint manifest differs')
                tx['quarantined']=True; h.save(tx,'Preserved incomplete864 outside auto-resume discovery tree')
            checkpoint(plan,quarantined=bool(tx.get('quarantined')))
            current=g.show(TARGET); mem=b.field(current,'MinMemoryNode')
            if mem=='116G':
                require(not tx.get('memory_intent'),'Uncertain memory update; inspect rather than repeat')
                state(item,amended=False,held=True); tx['memory_intent']=True; h.save(tx,'Persisted target-only116GiB to128GiB edit')
                b.command(['scontrol','update',f'JobId={TARGET}','MinMemoryNode=131072'])
            else: require(mem=='128G' and tx.get('memory_intent'),'Unexpected memory profile')
            tx['memory_updated']=True; state(item,amended=True,held=True)
            data=json.loads(g.CONTINUATIONS.read_text()); rows=[r for r in data['continuations'] if r['continuation_job_id']==TARGET]
            require(len(rows)==1 and len(data['continuations'])==75,'Ledger cardinality changed')
            desired=copy.deepcopy(plan['ledger_row_before']); desired['actual_scheduler_profile']['MinMemoryNode']='128G'
            for key in ('ReqTRES','Restarts'):
                desired['actual_scheduler_profile'][key]=b.field(g.show(TARGET),key)
            desired['memory128_guard_amendment']=str(TX)
            if rows[0]!=desired:
                require(rows[0]==plan['ledger_row_before'],'Target ledger row changed independently')
                atomic(ART/'continuation_ledger.before.json',data); rows[0].clear(); rows[0].update(desired)
                data.setdefault('repair_history',[]).append({'at':b.now(),'audit':str(TX),'job_id':TARGET,'only_scheduler_change':'116GiB to128GiB; same job/science/nodes','resume_step':plan['resume_step'],'repeat_updates_bound':plan['allow_repeat_updates']})
                tx['ledger_commit_intent']=True; h.save(tx,'Persisted target-only RAM provenance amendment'); atomic(g.CONTINUATIONS,data)
            tx['ledger_updated']=True
            copied=copy.deepcopy(previous); copied['plan_sha256']=plan['new_guard_plan_sha256']
            # This RAM repair does not consume or reset the TIMEOUT retry counters.
            if (NEW/'transaction.json').exists(): require(json.loads((NEW/'transaction.json').read_text())==copied,'Staged retry state differs')
            else: atomic(NEW/'transaction.json',copied)
            tx.update(guard_state_copied=True,status='target_and_successor_cpu_held'); h.save(tx,'Copied all ten guard states and original deadline; no GPU release yet')
    print(json.dumps({'staged':True,'next':'activate','resume_step':plan['resume_step']}))


def release():
    plan,tx=load()
    with g.LEDGER_LOCK.open('a+') as ledger:
        fcntl.flock(ledger,fcntl.LOCK_EX); h.ready(plan,tx)
        if not tx.get('target_release_requested'):
            state(plan['item'],amended=True,held=True); checkpoint(plan,quarantined=bool(tx.get('quarantined')))
            g.before_deadline(json.loads((NEW/'transaction.json').read_text()))
            tx['target_release_requested']=True; h.save(tx,'Healthy successor guard verified; releasing target-only owned hold')
            b.command(['scontrol','release',str(TARGET)])
        record=state(plan['item'],amended=True,held=False)
        require(b.field(record,'JobState') in ('PENDING','RUNNING','CONFIGURING') and b.field(record,'Priority')!='0','Uncertain release; inspect without repeating')
        require(int(b.field(record,'Restarts'))==int(plan['target_restarts'])+1,'Unexpected release restart count')
        tx.update(target_released=True,status='released',target_after=record); h.save(tx,'Target released with preserved optimizer/replay checkpoint and128GiB RAM')
    print(json.dumps({'released':TARGET,'resume_step':plan['resume_step'],'state':b.field(record,'JobState')}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('prepare','check','submit','route','activate','release','watch'))
    p.add_argument('--resume-step',type=int,choices=(768,864),default=864)
    p.add_argument('--allow-repeat-updates',type=int,choices=(0,96),default=0)
    a=p.parse_args(); g.install_bounded_scheduler(); ART.mkdir(parents=True,exist_ok=True)
    if a.phase=='watch': load(); h.watch(); return
    with (ART/'amendment.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.phase=='prepare':
            with g.LEDGER_LOCK.open('a+') as ledger:
                fcntl.flock(ledger,fcntl.LOCK_EX); prepare(a)
        elif a.phase=='activate': load(); h.activate()
        else: globals()[a.phase]()


if __name__=='__main__': main()
