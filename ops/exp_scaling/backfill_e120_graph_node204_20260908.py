#!/usr/bin/env python3
"""Prepare or apply one reviewed E120 Graph continuation on node204."""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
import re
from pathlib import Path
import sys

import campaign_stats as campaign
import prioritize_e118_capacity_20260905 as base
import backfill_e120_node105_20260905 as prior
from recover_e119_health_20260905 import checkpoint

ROOT = base.ROOT
ART = ROOT / 'var/artifacts/additional_campaign_capacity_20260908/e120_graph_node204'
PLAN = ART / 'plan.json'
TX = ART / 'transaction.json'
PROTOCOL = ART / 'protocol.md'
OLD, ORIGINAL, CHILD = 31048517, 31033708, 31048519
EXPECTED_OLD_DEP = 'afterany:31033705(unfulfilled)'
EXPECTED_CHILD_DEP = f'afterany:{OLD}(unfulfilled)'
RATIO = 'OAT_ZERO_VLLM_GPU_RATIO'
FIELDS = ('domain', 'model_key', 'seed', 'run_dir', 'run_stamp')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    base.atomic(path, value)


def command(parts):
    return base.command(parts)


def no_completion(run):
    run = Path(run)
    markers = [p for p in [run/'TRAINING_COMPLETE.json', *run.glob('debug_job*/TRAINING_COMPLETE.json')] if p.exists()]
    if markers:
        raise RuntimeError(f'completion marker exists: {markers}')


def checkpoint_file_identity(cp):
    return {str(p): {'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'inode':p.stat().st_ino}
            for p in Path(cp['path']).glob('*.pt')}


def live_records():
    result = {}
    for jid in base.queue():
        try:
            rec = base.show(jid)
        except Exception:
            if jid not in base.queue():
                continue
            raise
        result[jid] = rec
    return result


def writers(run, records):
    found = []
    for jid, rec in records.items():
        if 'SAVE_PATH=' not in rec:
            continue
        env = base.exports(base.submit_tokens(rec))
        if env.get('SAVE_PATH') and Path(env['SAVE_PATH']).resolve() == Path(run).resolve():
            found.append(jid)
    return sorted(found)


def incoming(job):
    output = command(['squeue', '-h', '-o', '%i|%T|%E']).stdout
    return {int(line.split('|')[0]):line.split('|')[2] for line in output.splitlines()
            if line.split('|')[0].isdigit() and re.search(r':'+str(job)+r'(?:\D|$)', line.split('|')[2])}


def capacity():
    rec = command(['scontrol','show','node','-o','node204']).stdout
    field = lambda key: base.field(rec,key)
    configured = dict(x.split('=',1) for x in field('CfgTRES').split(','))
    allocated = dict(x.split('=',1) for x in field('AllocTRES').split(','))
    assert 'gpu:a5000:' in field('Gres')
    assert not any(x in field('State') for x in ('DOWN','DRAIN','FAIL'))
    assert 'lowprio' in field('Partitions').split(',')
    assert int(field('RealMemory'))-int(field('AllocMem')) >= 128*1024
    assert int(field('CPUEfctv'))-int(field('CPUAlloc')) >= 16
    assert int(configured['gres/gpu'])-int(allocated.get('gres/gpu',0)) >= 1
    return rec


def new_command(original):
    env = base.exports(original)
    assert env[RATIO] == '0.25'
    env[RATIO] = '0.40'
    updates = {'--account':'allcs', '--partition':'lowprio', '--nodelist':'node204',
               '--gres':'gpu:a5000:1', '--mem':'128G', '--cpus-per-task':'16',
               '--time':'3-00:00:00', '--nice':'200', '--nodes':'1', '--ntasks':'1',
               '--ntasks-per-node':'1', '--exclude':prior.EXCLUSION,
               '--output':str(ROOT/'slurm-%j.out'), '--error':str(ROOT/'slurm-%j.out'),
               '--comment':f'e120-graph-node204-20260908-old{OLD}',
               '--export':'ALL,'+','.join(f'{k}={v}' for k,v in env.items())}
    skip = set(updates) | {'--hold','--dependency','--begin'}
    result = [v for v in original[:-1] if v.split('=',1)[0] not in skip]
    result += [f'{k}={v}' for k,v in updates.items()] + ['--hold',original[-1]]
    before, after = base.exports(original), base.exports(result)
    assert {k:(before.get(k),after.get(k)) for k in before.keys()|after.keys() if before.get(k)!=after.get(k)} == {RATIO:('0.25','0.40')}
    assert '--requeue' in result
    return result


def old_guard(plan, *, held=False):
    rec = base.show(OLD)
    assert base.field(rec,'JobState') == 'PENDING'
    assert base.field(rec,'Dependency') == EXPECTED_OLD_DEP
    assert base.submit_tokens(rec) == plan['original_command']
    assert base.field(rec,'ExcNodeList') == prior.EXCLUSION
    if held:
        assert base.field(rec,'Reason') == 'JobHeldUser' and base.field(rec,'Priority') == '0'
    return rec


def child_guard(plan, dependency, *, held=False):
    rec = base.show(CHILD)
    assert base.field(rec,'JobState') == 'PENDING'
    assert base.field(rec,'Dependency') == dependency
    assert base.submit_tokens(rec) == base.submit_tokens(plan['child_before'])
    assert base.field(rec,'ExcNodeList') == prior.EXCLUSION
    assert base.field(rec,'ReqNodeList') == 'node302'
    if held:
        assert base.field(rec,'Reason') == 'JobHeldUser' and base.field(rec,'Priority') == '0'
    return rec


def new_guard(plan):
    rec = base.show(plan['new_job_id'])
    expected = {'JobState':'PENDING','Reason':'JobHeldUser','Priority':'0','Dependency':'(null)',
                'Account':'allcs','Partition':'lowprio','ReqNodeList':'node204','MinMemoryNode':'128G',
                'NumCPUs':'16','TimeLimit':'3-00:00:00','Nice':'200','ExcNodeList':prior.EXCLUSION,
                'Requeue':'1','Restarts':'0','RunTime':'00:00:00','TresPerNode':'gres/gpu:a5000:1'}
    assert all(base.field(rec,k)==v for k,v in expected.items())
    tokens=base.submit_tokens(rec)
    assert tokens[-1] == plan['original_command'][-1]
    assert base.exports(tokens) == base.exports(plan['command'])
    assert base.field(rec,'StdOut') == str(ROOT/f"slurm-{plan['new_job_id']}.out")
    assert 'pvl' not in rec.lower()
    return rec


def prepare():
    if PLAN.exists() or TX.exists():
        raise RuntimeError('existing plan/transaction: inspect, do not overwrite')
    prior.patch_valid()
    main=json.loads(campaign.E120_LEDGER.read_text())
    mapping=campaign.e120_continuation_jobs(campaign.E120_LEDGER)
    assert len(main['runs']) == 45 and mapping[ORIGINAL] == OLD
    row=next(r for r in main['runs'] if r['job_id']==ORIGINAL)
    assert (row['model_key'],row['domain'],row['seed']) == ('qwen3b','graph_coloring',73)
    rec=base.show(OLD); original=base.submit_tokens(rec); env=base.exports(original)
    assert env['SAVE_PATH']==row['run_dir'] and env['RUN_STAMP']==row['run_stamp']
    assert env['OAT_ZERO_AUTO_RESUME']=='1' and env['OAT_ZERO_ONLINE_CANONICAL_REPLAY_KEY_WEIGHTING']=='fresh_frequency'
    assert env['OAT_ZERO_ENABLE_FLASH_ATTN']=='0'
    no_completion(row['run_dir'])
    cp=checkpoint(row); assert cp['step']==576
    records=live_records(); assert writers(row['run_dir'],records)==[OLD]
    assert incoming(OLD)=={CHILD:EXPECTED_CHILD_DEP}
    child=base.show(CHILD)
    assert base.field(child,'Priority')!='0' and base.field(child,'Reason')=='Dependency'
    runtime=Path(env['OAT_ZERO_OPS_SNAPSHOT_ROOT'])
    paths=[Path(original[-1]),runtime/'slurm/train_node302.slurm',runtime/'run_experiment.sh',runtime/'repo_env.sh']
    paths += [ROOT/'var/artifacts/e120_runtime_recovery_20260905/runtime-amendment.json']
    cmd=new_command(original)
    plan={'schema':'e120-graph-node204-capacity-v1','created_at_utc':base.now(),
          'original_job_id':ORIGINAL,'old_job_id':OLD,'child_job_id':CHILD,'run':row,
          'original_command':original,'command':cmd,'old_before':rec,'child_before':child,
          'child_was_unheld':True,'checkpoint':cp,'checkpoint_file_identity':checkpoint_file_identity(cp),'node204_before':capacity(),
          'scientific_outcomes_inspected':False,'same_scientific_cell':True,
          'scientific_export_diff':{},'operational_export_diff':{RATIO:['0.25','0.40']},
          'primary_sha256':digest(campaign.E120_LEDGER),'continuations_sha256':digest(campaign.E120_CONTINUATIONS),
          'runtime_sha256':{str(p):digest(p) for p in paths},
          'controller_sha256':digest(__file__),'protocol_sha256':digest(PROTOCOL)}
    old_guard(plan);child_guard(plan,EXPECTED_CHILD_DEP)
    check=command([cmd[0],'--test-only',*cmd[1:]])
    plan['sbatch_test_only']={'returncode':check.returncode,'stdout':check.stdout,'stderr':check.stderr}
    save(PLAN,plan)
    (ART/'primary.before.json').write_bytes(campaign.E120_LEDGER.read_bytes())
    (ART/'continuations.before.json').write_bytes(campaign.E120_CONTINUATIONS.read_bytes())
    print(json.dumps({'prepared':True,'old':OLD,'checkpoint':cp['step'],'child':CHILD,'sbatch_test_only':plan['sbatch_test_only']},indent=2))


def apply():
    if TX.exists():
        raise RuntimeError('transaction exists; reconcile recorded IDs/flags rather than retry')
    plan=json.loads(PLAN.read_text())
    assert digest(__file__)==plan['controller_sha256'] and digest(PROTOCOL)==plan['protocol_sha256']
    assert digest(campaign.E120_LEDGER)==plan['primary_sha256']
    assert digest(campaign.E120_CONTINUATIONS)==plan['continuations_sha256']
    assert all(digest(p)==v for p,v in plan['runtime_sha256'].items())
    prior.patch_valid();capacity();old_guard(plan);child_guard(plan,EXPECTED_CHILD_DEP)
    no_completion(plan['run']['run_dir']);assert checkpoint(plan['run'])==plan['checkpoint']
    assert checkpoint_file_identity(plan['checkpoint'])==plan['checkpoint_file_identity']
    assert writers(plan['run']['run_dir'],live_records())==[OLD]
    assert incoming(OLD)=={CHILD:EXPECTED_CHILD_DEP}
    tx=copy.deepcopy(plan);tx.update(status='applying',events=[])
    def event(message):
        tx['events'].append({'at':base.now(),'message':message});save(TX,tx)
    event('Validated unchanged recipe, checkpoint, ledger, runtime, child dependency and live capacity')
    try:
        tx['holding_child']=True;event('Holding child before any replacement work')
        command(['scontrol','hold',str(CHILD)]);child_guard(tx,EXPECTED_CHILD_DEP,held=True)
        tx['child_held']=True;event('Child held with original afterany dependency intact')
        command(['scontrol','hold',str(OLD)]);old_guard(tx,held=True)
        tx['old_held']=True;event('Old pending writer held')
        tx['submission_uncertain']=True;event('Submitting exactly one held continuation')
        response=command(tx['command']).stdout.strip();tx['new_job_id']=int(response.split(';')[0])
        tx['submission_uncertain']=False;event('Captured new held job ID before auditing')
        tx['new_held_record']=new_guard(tx)
        assert writers(tx['run']['run_dir'],live_records())==sorted([OLD,tx['new_job_id']])
        no_completion(tx['run']['run_dir']);assert checkpoint(tx['run'])==tx['checkpoint']
        assert checkpoint_file_identity(tx['checkpoint'])==tx['checkpoint_file_identity']
        assert incoming(OLD)=={CHILD:EXPECTED_CHILD_DEP}
        newdep=f"afterany:{tx['new_job_id']}"
        tx['retargeting_child']=True;event('Retargeting held child to replacement before retiring old job')
        command(['scontrol','update',f'JobId={CHILD}',f'Dependency={newdep}'])
        tx['child_after_retarget']=child_guard(tx,newdep+'(unfulfilled)',held=True)
        assert incoming(OLD)=={}
        tx['child_retargeted']=True;event('Child now waits for replacement completion')
        assert digest(campaign.E120_CONTINUATIONS)==plan['continuations_sha256']
        ledger=json.loads(campaign.E120_CONTINUATIONS.read_text())
        row=next(r for r in ledger['continuations'] if r['original_job_id']==ORIGINAL)
        assert row['continuation_job_id']==OLD
        row.setdefault('intermediate_job_ids',[]).append(OLD)
        row.update(continuation_job_id=tx['new_job_id'],released=False,
                   repair_kind='graph_a5000_capacity_20260908',resume_checkpoint=576,
                   placement_amendment=str(PROTOCOL),held_scheduler_record=tx['new_held_record'],
                   runtime_changes={RATIO:{'before':'0.25','after':'0.40'}},
                   optimizer_update_changed=False,treatment_changed=False,scientific_command_equal=True,
                   new_placement={'account':'allcs','partition':'lowprio','node':'node204','gpu':'a5000','cpus':16,'memory':'128G','nice':200})
        childrow=next(r for r in ledger['continuations'] if r['continuation_job_id']==CHILD)
        childrow['scheduler_dependency']=newdep
        childrow['dependency_amendment']=str(PROTOCOL)
        ledger.setdefault('placement_amendments',[]).append(str(PROTOCOL))
        ledger['scheduler_only']=False
        ledger['runtime_allocation_only']=True
        save(campaign.E120_CONTINUATIONS,ledger)
        assert digest(campaign.E120_LEDGER)==plan['primary_sha256']
        mapping=campaign.e120_continuation_jobs(campaign.E120_LEDGER)
        assert len(mapping)==9 and mapping[ORIGINAL]==tx['new_job_id']
        tx['ledger_committed']=True;event('Promoted held successor and child dependency; 45-cell primary unchanged')
        old_guard(tx,held=True);new_guard(tx);child_guard(tx,newdep+'(unfulfilled)',held=True)
        assert incoming(OLD)=={}
        tx['cancelling_old']=True;event('Retiring superseded held pending job')
        command(['scancel',str(OLD)])
        assert base.field(base.show(OLD),'JobState')=='CANCELLED'
        tx['old_cancelled']=True;event('Old writer cancelled after child retarget and lineage promotion')
        assert writers(tx['run']['run_dir'],live_records())==[tx['new_job_id']]
        no_completion(tx['run']['run_dir']);assert checkpoint(tx['run'])==tx['checkpoint']
        tx['releasing_child']=True;event('Restoring original child unheld state')
        command(['scontrol','release',str(CHILD)])
        tx['child_release_record']=child_guard(tx,newdep+'(unfulfilled)')
        assert base.field(tx['child_release_record'],'Priority')!='0'
        tx['child_released']=True;event('Child remains pending on replacement with original lane resources')
        assert checkpoint_file_identity(tx['checkpoint'])==tx['checkpoint_file_identity']
        assert all(digest(p)==v for p,v in tx['runtime_sha256'].items())
        mapping=campaign.e120_continuation_jobs(campaign.E120_LEDGER)
        assert len(mapping)==9 and mapping[ORIGINAL]==tx['new_job_id']
        tx['releasing_new']=True;event('Releasing registered successor')
        command(['scontrol','release',str(tx['new_job_id'])])
        tx['release_record']=base.show(tx['new_job_id'])
        assert base.field(tx['release_record'],'JobState') in ('PENDING','RUNNING','CONFIGURING')
        assert base.field(tx['release_record'],'Priority')!='0'
        row['released']=True;save(campaign.E120_CONTINUATIONS,ledger)
        assert digest(campaign.E120_LEDGER)==plan['primary_sha256']
        tx['released']=True;tx['status']='released';tx['continuations_after_sha256']=digest(campaign.E120_CONTINUATIONS)
        event('Released exactly one E120 Graph successor; scientific cell and child meaning preserved')
        print(json.dumps({'released':tx['new_job_id'],'child':CHILD,'dependency':newdep},indent=2))
    except BaseException as exc:
        tx['status']='stopped_for_reconciliation';tx['error']=repr(exc)
        event('Stopped fail closed; inspect transaction flags before any further mutation')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prepare','apply'))
    args=parser.parse_args();ART.mkdir(parents=True,exist_ok=True)
    with (ROOT/'var/artifacts/e120_ledger_promotion.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);globals()[args.phase]()
