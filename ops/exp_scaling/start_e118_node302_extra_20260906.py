#!/usr/bin/env python3
"""Start one existing unfinished E118 continuation in available node302 capacity."""
from pathlib import Path
import argparse, fcntl, hashlib, json, re, sys
import prioritize_e118_capacity_20260905 as base
sys.path.insert(0, str(base.ROOT / 'ops'))
from recover_e119_health_20260905 import checkpoint

ART = base.ROOT / 'var/artifacts/e118_node302_extra_20260906'
PROTOCOL = base.ROOT / 'paper/preregistration/e118_node302_extra_20260906.md'
OLD = 31048109
DEPENDENCY = '(null)'
base.AUDIT = ART / 'transaction.json'
base.PROTOCOL = PROTOCOL
LOCK = base.ROOT / 'var/artifacts/e118_ledger_promotion.lock'

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def independent():
    rows = base.command(['squeue', '-h', '-o', '%i|%E']).stdout.splitlines()
    assert not [r for r in rows if re.search(r'(?<!\d)'+str(OLD)+r'(?!\d)', r.split('|',1)[1])], 'Old pending job has dependants'

def valid_dependency(record):
    assert base.field(record,'Dependency') == '(null)', 'Unexpected dependency'

def available_capacity():
    record=base.command(['scontrol','show','node','-o','node302']).stdout
    assert int(base.field(record,'RealMemory'))-int(base.field(record,'AllocMem')) >= 128*1024, 'Insufficient full-request memory capacity'
    assert int(base.field(record,'CPUEfctv'))-int(base.field(record,'CPUAlloc')) >= 16, 'Insufficient CPU capacity'
    total=dict(x.split('=',1) for x in base.field(record,'CfgTRES').split(','))
    used=dict(x.split('=',1) for x in base.field(record,'AllocTRES').split(','))
    assert int(total['gres/gpu'])-int(used.get('gres/gpu',0)) >= 1, 'No available GPU'
    return record

def prepare():
    assert not base.AUDIT.exists(), 'Existing transaction requires apply/reconciliation'
    ART.mkdir(parents=True,exist_ok=True)
    raw=base.LEDGER.read_bytes(); source=json.loads(raw)
    row=next(r for r in source['runs'] if int(r['job_id'])==OLD)
    assert (row['domain'],row['arm'],int(row['seed']))==('graph_coloring','maxrl',71)
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
    node_before=available_capacity()
    item={k:row[k] for k in base.IDENTITY}
    item.update(old_job_id=OLD,before_record=record,lane='node302',original_command=original,command=cmd,new_job_id=None,resume_checkpoint=checked['path'],checkpoint_validation=checked,launcher_sha256=digest(original[-1]))
    test=base.command([cmd[0],'--test-only',*[t for t in cmd[1:] if t!='--hold']])
    PROTOCOL.write_text('# E118 additional node302 allocation — September 6, 2026\n\nThe user explicitly requested another E118 job running on node302. Select\nexisting pending Qwen-3B Graph MaxRL seed71 (job31048109), the most advanced\nvalid unfinished pending continuation, without inspecting efficacy outcomes.\nResume its validated model and optimizer checkpoint at step1728. Node302 has\ncapacity for the full unchanged request:1GPU,16CPUs,128GiB and12-hour walltime.\n\nReplace only this pending cs allocation with mltheory/node302 placement, with\nno predecessor dependency. Preserve the frozen launcher, complete scientific\nand runtime exports, registered run directory, automatic resume and excluded\nnode list; add the established explicit repository-root exports. No running\njob is interrupted. Under the ledger-promotion lock, hold and validate both\npending allocations, promote source and aggregate ledgers, retire the old\npending allocation, then release the replacement. Check checkpoint counters,\nlauncher hash, unchanged non-root exports and no unexpected dependency before\nrelease. Record all before/after state in\nvar/artifacts/e118_node302_extra_20260906/.\n')
    audit=dict(schema='e118-node302-extra-v1',created_at=base.now(),protocol=str(PROTOCOL),source_ledger=str(base.LEDGER),original_ledger_sha256=base.sha(raw),status='planned',scheduler_only=True,same_scientific_cells=True,same_run_directories=True,treatment_changed=False,outcomes_inspected=False,replacements=[item],cs_placement_updates=[],events=[],dependency=DEPENDENCY,node_before=node_before,controller_sha256=digest(__file__),helper_sha256=digest(base.__file__),protocol_sha256=digest(PROTOCOL),sbatch_test_only={'stdout':test.stdout,'stderr':test.stderr})
    (ART/'source.before.json').write_bytes(raw)
    base.save(audit,'prepared one existing E118 continuation for available node302 capacity')
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
        base.save(audit,'confirmed released additional node302 continuation with no dependency')
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
