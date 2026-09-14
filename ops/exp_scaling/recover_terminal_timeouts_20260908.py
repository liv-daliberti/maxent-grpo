#!/usr/bin/env python3
"""Resume ten existing timed-out science cells with preserved frozen recipes."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import prioritize_e118_capacity_20260905 as base
import campaign_stats as campaign
sys.path.insert(0, str(base.ROOT / 'ops'))
from validate_deepspeed_checkpoint import select_latest_checkpoint, validate_checkpoint

ROOT = base.ROOT
ART = ROOT / 'var/artifacts/campaign_recovery_20260908'
PLAN = ART / 'plan.json'
TX = ART / 'transaction.json'
E118 = base.LEDGER
E119 = campaign.E119_CONTINUATIONS
TARGETS = {
    'e118': [31073910, 31048118, 31048119, 31048120, 31048121, 31048122, 31073911, 31073909],
    'e119': [31048184, 31048189],
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state(job):
    output = base.command(['sacct', '-n', '-X', '-P', '-j', str(job), '-o', 'JobID,State']).stdout
    states = [row.split('|')[1].split()[0].rstrip('+') for row in output.splitlines()
              if row.split('|')[0] == str(job)]
    if len(states) != 1:
        raise RuntimeError(f'ambiguous accounting {job}: {states}')
    return states[0]


def complete(run):
    paths = [run / 'TRAINING_COMPLETE.json', *run.glob('debug_job*/TRAINING_COMPLETE.json')]
    return any(p.exists() and int(json.loads(p.read_text()).get('terminal_step', 0)) >= 3072 for p in paths)


def active_writers():
    writers = {}
    for job in base.queue():
        try:
            current = base.show(job)
        except Exception:
            if job not in base.queue():
                continue
            raise
        if base.field(current, 'JobState') not in {'RUNNING','PENDING','CONFIGURING','COMPLETING','SUSPENDED'}:
            continue
        # Only jobs with an exported SAVE_PATH can write these frozen science runs.
        if 'SAVE_PATH=' not in current:
            continue
        env = base.exports(base.submit_tokens(current))
        if env.get('SAVE_PATH'):
            writers.setdefault(str(Path(env['SAVE_PATH']).resolve()), set()).add(job)
    return writers


def no_other_writer(item, writers):
    allowed = {item['new_job_id']} if item.get('new_job_id') else set()
    actual = writers.get(str(Path(item['identity']['run_dir']).resolve()), set())
    if actual - allowed:
        raise RuntimeError(f'unexpected live writer for {item["old_job_id"]}: {actual - allowed}')


def record(job, rows):
    return next(row for row in rows if int(row.get('job_id', row.get('continuation_job_id'))) == job)


def command(original, job, dependency=None):
    env = base.exports(original)
    env.update(base.ROOT_EXPORTS)
    updates = {
        '--account': 'mltheory', '--partition': 'mltheory', '--nodelist': 'node302',
        '--gres': 'gpu:a100:1', '--mem': '128G', '--time': '3-00:00:00',
        '--nodes': '1', '--ntasks': '1', '--ntasks-per-node': '1', '--nice': '200',
        '--exclude': base.PVL,
        '--output': str(ROOT / 'var/artifacts/logs/%x-%j.out'),
        '--error': str(ROOT / 'var/artifacts/logs/%x-%j.err'),
        '--comment': f'campaign-recovery-20260908-old{job}',
        '--export': 'ALL,' + ','.join(f'{k}={v}' for k, v in env.items()),
    }
    if dependency:
        updates['--dependency'] = dependency
    skip = set(updates) | {'--dependency', '--hold', '--begin'}
    result = [x for x in original[:-1] if x.split('=', 1)[0] not in skip]
    result += [f'{k}={v}' for k, v in updates.items()]
    result += ['--hold', original[-1]]
    assert base.nonroot_exports(result) == base.nonroot_exports(original)
    return result


def prepare():
    if PLAN.exists() or TX.exists():
        raise RuntimeError('existing plan/transaction: inspect it rather than overwriting')
    ART.mkdir(exist_ok=True)
    old_records = json.loads((ROOT / 'var/artifacts/e119_pending_memory96_20260906/plan.json').read_text())['entries']
    rows = []
    writers = active_writers()
    for cohort, path, key in [('e118', E118, 'runs'), ('e119', E119, 'continuations')]:
        data = json.loads(path.read_text())
        (ART / (path.name + '.before')).write_bytes(path.read_bytes())
        for job in TARGETS[cohort]:
            row = record(job, data[key])
            if state(job) != 'TIMEOUT' or job in base.queue():
                raise RuntimeError(f'job is no longer an inactive timeout: {job}')
            run = Path(row['run_dir'])
            if complete(run):
                raise RuntimeError(f'completed cell must not be restarted: {job}')
            checkpoint, rejected = select_latest_checkpoint(run)
            if checkpoint is None or validate_checkpoint(checkpoint):
                raise RuntimeError(f'no valid optimizer checkpoint for {job}')
            archived = row['held_scheduler_record'] if cohort == 'e118' else next(x['before'] for x in old_records if x['job_id'] == job)
            original = base.submit_tokens(archived)
            env = base.exports(original)
            assert env['SAVE_PATH'] == row['run_dir'] and env['RUN_STAMP'] == row['run_stamp']
            assert env['OAT_ZERO_AUTO_RESUME'] == '1'
            assert env['OAT_ZERO_SOURCE_ROOT'] == str(Path(original[-1]).parents[2] / 'src')
            rows.append(dict(cohort=cohort, old_job_id=job, identity={k:row[k] for k in base.IDENTITY},
                             original_command=original, frozen_launcher_sha256=digest(original[-1]),
                             checkpoint=str(checkpoint), checkpoint_step=int(checkpoint.name[5:]),
                             rejected_checkpoints=rejected, new_job_id=None, released=False))
    for item in rows:
        no_other_writer(item, writers)
    # Preserve one node302 lane for this recovery while two lanes finish E120.
    # Interleave E119 with E118 rather than leave either family at the end.
    order = [31073910,31048189,31048118,31048184,31048119,31048120,31048121,31048122,31073911,31073909]
    rows.sort(key=lambda x:order.index(x['old_job_id']))
    plan = dict(schema='terminal-timeout-recovery-20260908-v1', created_at_utc=base.now(),
                authorization='User explicitly requested repairing failed E118/E119/E120 cells and using node302/node105.',
                scheduler_only=True, scientific_exports_unchanged=True, existing_cells_only=True,
                source_sha256={str(p):digest(p) for p in (E118,E119)}, rows=rows,
                scheduling='One afterany node302 lane; E120 owns two other lanes. Existing live jobs are untouched. 72h allocations avoid repeating 12h/36h timeouts; same eight-pass training target. Host RAM 128GiB follows existing memory-pressure repairs; 80GB A100 satisfies Pantry GPU requirement.',
                controller_sha256=digest(__file__))
    base.atomic(PLAN, plan)
    print(json.dumps({k:v for k,v in plan.items() if k != 'rows'}, indent=2))
    print([(r['cohort'],r['old_job_id'],r['checkpoint_step']) for r in rows])


def held_audit(item):
    rec = base.show(item['new_job_id'])
    expected = {'JobState':'PENDING','Reason':'JobHeldUser','Account':'mltheory',
                'Partition':'mltheory','ReqNodeList':'node302','MinMemoryNode':'128G',
                'TimeLimit':'3-00:00:00','Nice':'200','ExcNodeList':base.PVL,'Requeue':'1'}
    for key,value in expected.items():
        if base.field(rec,key) != value:
            raise RuntimeError(f'held job mismatch {key}: {base.field(rec,key)} != {value}')
    assert 'gres/gpu=1' in base.field(rec,'ReqTRES').split(',')
    assert base.field(rec,'TresPerNode') == 'gres/gpu:a100:1'
    tokens=base.submit_tokens(rec)
    assert base.nonroot_exports(tokens)==base.nonroot_exports(item['original_command'])
    assert tokens[-1]==item['original_command'][-1]
    assert all(base.exports(tokens).get(k)==v for k,v in base.ROOT_EXPORTS.items())
    dep=item.get('dependency')
    assert not dep or base.field(rec,'Dependency').startswith(dep)
    return rec


def apply(resume=False):
    if TX.exists() and not resume:
        raise RuntimeError('transaction exists; reconcile recorded submissions, do not retry blindly')
    tx=json.loads((TX if resume else PLAN).read_text())
    if resume and tx.get('ledger_committed'):
        raise RuntimeError('already committed: inspect release records rather than repeat promotion')
    assert tx['controller_sha256']==digest(__file__)
    assert all(digest(p)==sha for p,sha in tx['source_sha256'].items())
    tx.update(status='applying')
    tx.setdefault('started_at_utc',base.now())
    tx.setdefault('events',[])
    def event(message):
        tx['events'].append(dict(at=base.now(),message=message));base.atomic(TX,tx)
    previous=None
    writers=active_writers()
    for item in tx['rows']:
        no_other_writer(item,writers)
    try:
        for item in tx['rows']:
            old=item['old_job_id']
            assert state(old)=='TIMEOUT' and old not in base.queue()
            assert not complete(Path(item['identity']['run_dir']))
            assert not validate_checkpoint(Path(item['checkpoint']))
            assert digest(item['original_command'][-1])==item['frozen_launcher_sha256']
            item['dependency']=f'afterany:{previous}' if previous else None
            item['command']=command(item['original_command'],old,item['dependency'])
            if item.get('submission_uncertain'):
                raise RuntimeError('uncertain submission: reconcile scheduler comment before retry')
            if item['new_job_id'] is None:
                item['submission_uncertain']=True;event(f'submitting held continuation for {old}')
                submitted=base.command(item['command']).stdout.strip()
                item['new_job_id']=int(submitted.split(';')[0]);item['submission_uncertain']=False
                event(f'submitted held continuation {item["new_job_id"]}')
            else:
                event(f'reusing recorded held continuation {item["new_job_id"]}; no duplicate submission')
            item['held_scheduler_record']=held_audit(item)
            previous=item['new_job_id'];event(f'verified frozen recipe and placement for {previous}')
        with (ROOT/'var/artifacts/e118_ledger_promotion.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            assert all(digest(p)==sha for p,sha in tx['source_sha256'].items())
            writers=active_writers()
            for item in tx['rows']:
                no_other_writer(item,writers)
                assert not complete(Path(item['identity']['run_dir']))
            for cohort,path,key in [('e118',E118,'runs'),('e119',E119,'continuations')]:
                data=json.loads(path.read_text())
                for item in tx['rows']:
                    if item['cohort']!=cohort:continue
                    old=item['old_job_id'];row=record(old,data[key])
                    assert all(row[k]==v for k,v in item['identity'].items())
                    assert state(old)=='TIMEOUT' and old not in base.queue()
                    held_audit(item)
                    history='previous_job_ids' if cohort=='e118' else 'previous_continuation_job_ids'
                    row[history]=[*row.get(history,[]),old]
                    row['job_id' if cohort=='e118' else 'continuation_job_id']=item['new_job_id']
                    row.update(held_scheduler_record=item['held_scheduler_record'],repair_audit=str(TX),
                               scheduler_dependency=item['dependency'])
                base.atomic(path,data);event(f'committed {cohort} authoritative continuation ledger')
            base.command(['python3',str(ROOT/'ops/exp_scaling/build_e118_aggregate_ledger.py')])
            tx['ledger_committed']=True;event('regenerated 150-cell E118 aggregate')
        writers=active_writers()
        for item in tx['rows']:
            no_other_writer(item,writers)
            assert not complete(Path(item['identity']['run_dir']))
            base.command(['scontrol','release',str(item['new_job_id'])]);item['released']=True
            item['release_record']=base.show(item['new_job_id']);event(f'released {item["new_job_id"]}')
        tx['status']='released';event('all ten valid-checkpoint continuations released')
        print([(r['old_job_id'],r['new_job_id']) for r in tx['rows']])
    except BaseException as exc:
        tx.update(status='stopped',error=repr(exc));event('stopped with auditable transaction; no blind retry');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','apply','resume']);args=parser.parse_args()
    prepare() if args.action=='prepare' else apply(resume=args.action=='resume')
