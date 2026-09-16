#!/usr/bin/env python3
"""Queue one existing unfinished E118 continuation after job31073906."""
from pathlib import Path
import argparse, fcntl, hashlib, json, re, sys
import prioritize_e118_capacity_20260905 as base
sys.path.insert(0, str(base.ROOT / 'ops'))
from recover_e119_health_20260905 import checkpoint

ART = base.ROOT / 'var/artifacts/e118_node302_successor_20260906'
PROTOCOL = base.ROOT / 'paper/preregistration/e118_node302_successor_20260906.md'
OLD, PREDECESSOR = 31048108, 31073906
DEPENDENCY = f'afterany:{PREDECESSOR}'
base.AUDIT = ART / 'transaction.json'
base.PROTOCOL = PROTOCOL
LOCK = base.ROOT / 'var/artifacts/e118_ledger_promotion.lock'

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def independent():
    rows = base.command(['squeue', '-h', '-o', '%i|%E']).stdout.splitlines()
    assert not [r for r in rows if re.search(r'(?<!\d)'+str(OLD)+r'(?!\d)', r.split('|',1)[1])], 'Old pending job has dependants'

def predecessor_terminal():
    r=base.command(['scontrol','show','job','-o',str(PREDECESSOR)],check=False)
    if r.returncode == 0 and r.stdout.strip():
        return base.field(r.stdout,'JobState') in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED'}
    rows=base.command(['sacct','-n','-X','-P','-j',str(PREDECESSOR),'--format=JobIDRaw,State']).stdout.splitlines()
    states=[r.split('|')[1].split()[0] for r in rows if r.split('|')[0]==str(PREDECESSOR)]
    return len(states)==1 and states[0] in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED'}

def valid_dependency(record):
    dep=base.field(record,'Dependency')
    assert dep == DEPENDENCY or dep.startswith(DEPENDENCY+'(') or (dep=='(null)' and predecessor_terminal()), dep

def prepare():
    assert not base.AUDIT.exists(), 'Existing transaction requires apply/reconciliation'
    ART.mkdir(parents=True,exist_ok=True)
    raw=base.LEDGER.read_bytes(); source=json.loads(raw)
    row=next(r for r in source['runs'] if int(r['job_id'])==OLD)
    assert (row['domain'],row['arm'],int(row['seed']))==('graph_coloring','replay_maxrl',70)
    record=base.show(OLD)
    assert base.field(record,'JobState')=='PENDING' and int(base.field(record,'Priority'))>0
    assert base.field(record,'Dependency')=='(null)'
    assert not (Path(row['run_dir'])/'TRAINING_COMPLETE.json').exists()
    independent()
    checked=checkpoint(row)
    assert checked['step']==1728 and not checked['rejected']
    original=base.submit_tokens(record)
    cmd=base.placed(original,'node302',OLD)
    assert not any(t.startswith('--dependency') for t in cmd)
    cmd.insert(-1,'--dependency='+DEPENDENCY)
    item={k:row[k] for k in base.IDENTITY}
    item.update(old_job_id=OLD,before_record=record,lane='node302',original_command=original,command=cmd,new_job_id=None,resume_checkpoint=checked['path'],checkpoint_validation=checked,launcher_sha256=digest(original[-1]))
    test=base.command([cmd[0],'--test-only',*[t for t in cmd[1:] if t!='--hold']])
    PROTOCOL.write_text('''# E118 node302 successor — September 6, 2026

The user requested other Qwen-3B E118 training to start when job31073906
finishes. The five cited Python failures already have successful terminal
successors; no completed Python cell is restarted. Select existing pending
Graph Re:Max seed70 (job31048108) by highest valid pending checkpoint
progress, breaking the tie by original job ID, without inspecting efficacy
outcomes. Resume its valid model and optimizer checkpoint at step1728.

Replace this pending cs placement with mltheory/node302, queued afterany31073906.
Preserve the same registered cell, run directory, frozen launcher and all
scientific/runtime settings: one GPU,16 CPUs,128GiB,12-hour walltime and automatic
resume. Do not stop a running allocation. Retain the standard excluded-node list.
Hold both old and new pending jobs while checking identity, checkpoint counters,
launcher hash and dependency. Promote source and aggregate ledgers under their
lock before retiring the old pending job and releasing its successor. Record
all scheduler records and lineage in var/artifacts/e118_node302_successor_20260906/.
''')
    audit=dict(schema='e118-node302-successor-v1',created_at=base.now(),protocol=str(PROTOCOL),source_ledger=str(base.LEDGER),original_ledger_sha256=base.sha(raw),status='planned',scheduler_only=True,same_scientific_cells=True,same_run_directories=True,treatment_changed=False,outcomes_inspected=False,replacements=[item],cs_placement_updates=[],events=[],dependency=DEPENDENCY,controller_sha256=digest(__file__),helper_sha256=digest(base.__file__),protocol_sha256=digest(PROTOCOL),sbatch_test_only={'stdout':test.stdout,'stderr':test.stderr})
    (ART/'source.before.json').write_bytes(raw)
    base.save(audit,'prepared one existing E118 continuation behind31073906')
    print(json.dumps({'prepared':True,'old_job':OLD,'resume_step':1728,'dependency':DEPENDENCY,'sbatch_test_only':test.stderr.strip()}))

def apply():
    audit=json.loads(base.AUDIT.read_text())
    assert digest(__file__)==audit['controller_sha256']
    assert digest(base.__file__)==audit['helper_sha256']
    assert digest(PROTOCOL)==audit['protocol_sha256']
    item=audit['replacements'][0]
    assert digest(item['original_command'][-1])==item['launcher_sha256']
    if audit.get('status')=='complete':
        print(json.dumps({'already_complete':True,'new_job':item['new_job_id']})); return
    if not audit.get('ledger_committed'):
        assert not (Path(item['run_dir'])/'TRAINING_COMPLETE.json').exists()
        independent()
        assert checkpoint(item)==item['checkpoint_validation']
    original_audit=base.audit_record
    def checked_record(record, entry, *, held):
        original_audit(record,entry,held=held)
        valid_dependency(record)
        assert digest(base.field(record,'Command'))==entry['launcher_sha256']
    base.audit_record=checked_record
    try:
        base.apply(audit)
        after=base.show(item['new_job_id'])
        checked_record(after,item,held=False)
        assert int(base.field(after,'Priority'))>0
        item['final_scheduler_record']=after
        base.save(audit,'confirmed released node302 successor and predecessor dependency')
        print(json.dumps({'new_job':item['new_job_id'],'state':base.field(after,'JobState'),'dependency':base.field(after,'Dependency'),'node':base.field(after,'ReqNodeList'),'resume_step':item['checkpoint_validation']['step']}))
    except BaseException as exc:
        audit['status']='interrupted'; audit['error']=repr(exc)
        base.save(audit,'stopped for reconciliation; preserve recorded transaction state')
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','apply'))
    args=parser.parse_args()
    with LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        globals()[args.phase]()
