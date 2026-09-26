#!/usr/bin/env python3
"""Strictly advancing hourly timeout retries for three existing cells lost to repeated preemption."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
import math
import pickletools
import zipfile
from pathlib import Path
import time

import campaign_stats as campaign
import prioritize_e118_capacity_20260905 as base
import recover_terminal_timeouts_20260908 as recovery
import backfill_preempted_hourly_20260909 as capacity

ROOT = base.ROOT
ART = ROOT / 'var/artifacts/preemption_hourly_timeout_guard_20260909'
PLAN = ART / 'plan.json'
TX = ART / 'transaction.json'
PROTOCOL = ROOT / 'paper/preregistration/preemption_hourly_timeout_guard_20260909.md'
LOCK = ROOT / 'var/artifacts/preemption_hourly_timeout_guard_20260909.lock'
LEDGER_LOCK = ROOT / 'var/artifacts/e118_ledger_promotion.lock'
DEPLOYMENT = ROOT / 'var/artifacts/campaign_preemption_hourly_backfill_20260909/transaction.json'
PRESERVE = ('UserId', 'Account', 'Partition', 'ReqNodeList', 'ExcNodeList',
            'MinMemoryNode', 'NumCPUs', 'NumNodes', 'NumTasks', 'TresPerNode',
            'Requeue', 'Nice', 'QOS', 'Dependency', 'Features', 'WorkDir', 'TimeLimit')
ACTIVE = {'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def save(tx, message):
    tx['updated_at_utc'] = base.now()
    tx.setdefault('events', []).append(dict(at=tx['updated_at_utc'], event=message))
    base.atomic(TX, tx)
    print(json.dumps(dict(at=tx['updated_at_utc'], event=message)), flush=True)


EXPECTED = {
    31158505: ('e118', 'mathir', 'maxrl', 73, 960, '116G'),
    31158506: ('e118', 'mathir', 'replay_maxrl', 73, 960, '116G'),
    31158507: ('e120', 'graph_coloring', None, 74, 576, '128G'),
}


def mapping(job_id=None):
    deployment = json.loads(DEPLOYMENT.read_text())
    require(deployment['schema'] == 'campaign-preemption-hourly-backfill-v1', 'Unexpected deployment schema')
    items = deployment['items']
    require(len(items) == 3 and {r['old_job_id'] for r in items} == set(EXPECTED), 'Unexpected hourly cohort')
    result = {}
    for item in items:
        jid = item['new_job_id']
        require(isinstance(jid, int), 'Replacement has not been staged')
        if job_id is not None and jid != job_id:
            continue
        cohort, domain, arm, seed, floor, memory = EXPECTED[item['old_job_id']]
        require((item['cohort'], item['domain'], item['arm'], item['seed']) ==
                (cohort, domain, arm, seed), 'Unexpected scientific cell')
        require(item['checkpoint']['step'] == floor, 'Initial durable floor changed')
        restored = item.get('fallback', {}).get('status') == 'complete'
        capacity.current_identity(item, item['old_job_id'] if restored else jid)
        identity = {k: item[k] for k in base.IDENTITY}
        result[jid] = dict(cohort=cohort, identity=identity, initial_checkpoint_step=floor,
                           memory=memory, long_route_command=item['long_route_command'],
                           dormant_old_job_id=item['old_job_id'], fallback_complete=restored)
    require(len(result) == (3 if job_id is None else 1), 'Missing or duplicate hourly IDs')
    return result


SCALAR_KEYS = ('global_steps','global_step','policy_sgd_step','prompt_batches_consumed_total')
MEMO_OPS = {'BINPUT','LONG_BINPUT','PUT','MEMOIZE'}
NUMBER_OPS = {'BININT','BININT1','BININT2','INT','LONG','LONG1','LONG4','BINFLOAT','FLOAT'}


def metadata_scalars(raw, keys):
    values = {key:[] for key in keys}; pending = None; bank = False
    memo = {}; top_string = None
    string_ops = {'UNICODE','BINUNICODE','SHORT_BINUNICODE','BINUNICODE8','STRING'}
    get_ops = {'GET','BINGET','LONG_BINGET'}
    for op,arg,_ in pickletools.genops(raw):
        if op.name in MEMO_OPS:
            index = len(memo) if op.name == 'MEMOIZE' else int(arg)
            memo[index] = top_string
            continue
        semantic_string = arg if op.name in string_ops and isinstance(arg,str) else memo.get(int(arg)) if op.name in get_ops else None
        if pending is not None:
            require(op.name in NUMBER_OPS and isinstance(arg,(int,float)) and not isinstance(arg,bool) and math.isfinite(arg), f'Invalid scalar counter {pending}')
            values[pending].append(arg); pending = None
        top_string = semantic_string
        if semantic_string == 'online_canonical_bank_state':
            bank = True
        if semantic_string in values:
            pending = semantic_string
    require(pending is None, 'Truncated scalar counter metadata')
    return values, bank


def archive_metadata(path):
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith('/data.pkl') or n == 'data.pkl']
        require(len(names) == 1, 'Expected exactly one pickle metadata member')
        require(archive.getinfo(names[0]).file_size <= 16*1024*1024, 'Checkpoint metadata exceeds bounded inspection budget')
        return archive.read(names[0])


def checked_checkpoint(path):
    path = Path(path)
    require(not recovery.validate_checkpoint(path), 'Structurally invalid model/optimizer checkpoint')
    step = int(path.name.removeprefix('step_'))
    require(0 < step < 3072 and step % 48 == 0, 'Checkpoint is outside the registered unfinished 48-step grid')
    files = []; model_records = []; optimizer_records = []
    for archive in sorted(path.glob('*.pt')):
        raw = archive_metadata(archive)
        if archive.name.endswith('_model_states.pt'):
            counters, bank = metadata_scalars(raw, SCALAR_KEYS)
            require(all(counters[k] and all(v == step for v in counters[k]) for k in SCALAR_KEYS), f'Model/prompt counters disagree with checkpoint tag: {counters}')
            require(bank, 'Saved replay bank state is missing')
            model_records.append(counters)
        elif archive.name.endswith('_optim_states.pt'):
            counters,_ = metadata_scalars(raw, ('step',))
            require(counters['step'] and all(v == step for v in counters['step']), f'Optimizer counters disagree with checkpoint tag: {counters}')
            optimizer_records.append(counters)
        stat = archive.stat()
        files.append(dict(name=archive.name, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
    require(model_records and optimizer_records, 'Missing model or optimizer counter records')
    return dict(path=str(path),step=step,model_counters=model_records,optimizer_counters=optimizer_records,files=files)


def require_progress(item, job_state, detail):
    floor = max(item['initial_checkpoint_step'], job_state.get('last_resume_step', item['initial_checkpoint_step']))
    require(detail['step'] > floor, f'No durable progress: checkpoint {detail["step"]} is not newer than resume floor {floor}; preserve long-route fallback for coordinated review')


def stable(item, record):
    require(base.submit_tokens(record) == item['submit_tokens'], 'Frozen SubmitLine changed')
    require(recovery.digest(item['submit_tokens'][-1]) == item['launcher_sha256'], 'Frozen launcher changed')
    for field in PRESERVE:
        current, frozen = base.field(record, field), item['resources'][field]
        if field == 'NumNodes':
            require(current in {'1', '1-1'} and frozen in {'1', '1-1'}, 'Guarded allocation is not exactly one node')
        else:
            require(current == frozen, f'Guarded resource changed: {field}')


def show(job):
    result = base.command(['scontrol', 'show', 'job', '-dd', '-o', str(job)], check=False)
    require(result.returncode == 0 and result.stdout.strip(), f'Job {job} no longer has a controller record; manual continuation required')
    return result.stdout


def checkpoint_and_writer(item):
    run = Path(item['identity']['run_dir'])
    require(not recovery.complete(run), 'Training already has a terminal receipt; do not restart')
    checkpoint,rejected = recovery.select_latest_checkpoint(run)
    require(checkpoint is not None, 'No saved model/optimizer checkpoint')
    detail = checked_checkpoint(checkpoint)
    writers = recovery.active_writers().get(str(run.resolve()),set())
    allowed = {item['job_id']}
    old_id = item['dormant_old_job_id']
    if old_id in writers:
        deployment = json.loads(DEPLOYMENT.read_text())
        deployed = next(r for r in deployment['items'] if r['new_job_id'] == item['job_id'])
        require(deployed.get('old_fallback_held') and deployed['old_job_id'] == old_id, 'Unapproved dormant fallback writer')
        capacity.old_guard(deployed,held=True)
        allowed.add(old_id)
    require(writers <= allowed, f'Unexpected active or pending writer: {writers-allowed}')
    return dict(detail,rejected=rejected)


def own_hold(item, record, action):
    stable(item, record)
    require(base.field(record, 'JobState') == 'PENDING', 'Owned hold is not pending')
    require(base.field(record, 'Reason') == 'job_requeued_in_held_state', 'Expected requeue-owned hold missing')
    require(int(base.field(record, 'Restarts')) == action['before_restarts'] + 1, 'Expected exactly one requeue restart increment')


def prepare():
    require(not PLAN.exists() and not TX.exists(), 'Existing plan/transaction must not be overwritten')
    ART.mkdir(parents=True,exist_ok=True)
    identities = mapping(); rows = []
    deployment = json.loads(DEPLOYMENT.read_text())
    for job,ident in identities.items():
        record = show(job)
        require(base.field(record,'JobState') in {'PENDING','RUNNING'}, 'Replacement is not queued or running')
        tokens = base.submit_tokens(record); env = base.exports(tokens)
        require(env['SAVE_PATH'] == ident['identity']['run_dir'] and env['RUN_STAMP'] == ident['identity']['run_stamp'], 'Run exports changed')
        require(env['OAT_ZERO_AUTO_RESUME'] == '1' and env['OAT_ZERO_RESUME_STEPS'] == '48' and env['OAT_ZERO_VLLM_GPU_RATIO'] == '0.40', 'Expected runtime settings missing')
        for key,value in {'Account':'mltheory','Partition':'all','TimeLimit':'01:00:00','MinMemoryNode':ident['memory'],'NumCPUs':'16','TresPerNode':'gres/gpu:a5000:1','Requeue':'1'}.items():
            require(base.field(record,key) == value, f'Hourly allocation drift: {key}')
        original = next(i for i in deployment['items'] if i['new_job_id'] == job)
        initial = checked_checkpoint(original['checkpoint']['path'])
        require(initial['step'] == ident['initial_checkpoint_step'], 'Initial durable progress floor changed')
        rows.append(dict(ident,job_id=job,before=record,submit_tokens=tokens,
            launcher_sha256=recovery.digest(tokens[-1]),resources={f:base.field(record,f) for f in PRESERVE},
            initial_restarts=int(base.field(record,'Restarts')),initial_checkpoint=initial,max_requeues=24))
    plan = dict(schema='preemption-hourly-timeout-guard-20260909-v1',created_at_utc=base.now(),
        authorization='User requested bounded recovery of broken workloads and accelerated completion; root approved this exact hourly route.',
        protocol=str(PROTOCOL),protocol_sha256=recovery.digest(PROTOCOL),controller_sha256=recovery.digest(__file__),
        helper_sha256={str(Path(m.__file__).resolve()):recovery.digest(m.__file__) for m in (base,recovery,campaign,capacity,capacity.prior)},
        rows=rows,poll_seconds=30,maximum_watch_hours=24,existing_cells_only=True,scheduler_only=True,
        fallback_policy='Capacity-owned atomic activation of exact dormant long fallback; never submit a new science job.',
        outcomes_inspected=False)
    base.atomic(PLAN,plan)
    print(json.dumps(dict(plan=str(PLAN),job_ids=sorted(identities),max_requeues=24,scheduler_mutations=False)),flush=True)


def retry_deadline_reached(tx, now=None):
    return bool(tx.get('deadline_utc')) and (now or datetime.now(timezone.utc)) >= datetime.fromisoformat(tx['deadline_utc'])


def fallback_result(tx,item,job_state,reason,apply):
    if not apply:
        return dict(job_id=item['job_id'],status='would_activate_long_fallback',reason=reason,scheduler_mutations=False)
    job_state.update(status='fallback_in_progress',fallback_reason=reason)
    tx['jobs'][str(item['job_id'])] = job_state
    save(tx,f"{item['job_id']}: durable coordinated fallback intent: {reason}")
    try:
        capacity.fallback(item['job_id'],reason)
    except RuntimeError as error:
        if str(error) != 'hourly allocation started; defer fallback until inactive':
            raise
        record=show(item['job_id']);stable(item,record)
        require(base.field(record,'JobState') in ACTIVE,'Fallback race did not leave an active allocation')
        job_state['status']='monitoring';save(tx,f"{item['job_id']}: deadline hold raced allocation; preserve running job until inactive")
        return dict(job_id=item['job_id'],status='monitoring_final_allocation')
    job_state.update(status='fallback_complete')
    save(tx,f"{item['job_id']}: dormant original long allocation restored; hourly retries ended")
    return dict(job_id=item['job_id'],status='fallback_complete',reason=reason)


def finish_release(tx, item, record, action, job_state):
    expected = action['before_restarts'] + 1
    require(int(base.field(record, 'Restarts')) == expected, 'Restart count changed during release')
    action.update(after=record, released=True)
    job_state.update(status='monitoring', last_resume_step=action['checkpoint_before_release']['step'])
    save(tx, f"{item['job_id']}: released same-ID checkpoint retry {len(job_state['attempts'])}/{item['max_requeues']}; original allocation limit retained")


def require_committed_before_deadline(tx,action):
    require(action.get('hold_intent') and action.get('requested_at_utc'),'Missing durable pre-deadline requeue intent')
    requested=datetime.fromisoformat(action['requested_at_utc'])
    deadline=datetime.fromisoformat(tx['deadline_utc'])
    require(requested < deadline,'Retry was not committed before the deadline')
    # Allow the already committed transition to finish, not another requeue.
    # Keeping the release within five minutes leaves a full hour of cleanup.
    require(datetime.now(timezone.utc) <= deadline+timedelta(minutes=5),'Committed transition exceeds the bounded final-allocation grace')


def deadline_hold_handoff(tx, item, action, job_state):
    """Normalize only our proven requeue hold, then use capacity deadline fallback."""
    require(action.get('hold_intent') and action.get('requested_at_utc'), 'Missing persisted requeue intent')
    require(datetime.fromisoformat(action['requested_at_utc']) < datetime.fromisoformat(tx['deadline_utc']),
            'Deadline handoff requires a pre-deadline committed action')
    record = show(item['job_id']); stable(item, record)
    require(base.field(record, 'JobState') == 'PENDING', 'Deadline handoff cannot preempt an allocation')
    require(int(base.field(record, 'Restarts')) == action['before_restarts'] + 1,
            'Deadline handoff requires exact owned restart increment')
    reason = base.field(record, 'Reason')
    require(reason == 'job_requeued_in_held_state' or
            (action.get('deadline_handoff_requested') and reason == 'JobHeldUser'),
            'Deadline handoff cannot claim an unrelated user/admin hold')
    identities = mapping(item['job_id'])
    require(identities[item['job_id']]['identity'] == item['identity'], 'Deadline handoff identity changed')
    detail = checkpoint_and_writer(item); require_progress(item, job_state, detail)
    require(detail['step'] >= action['checkpoint_before']['step'], 'Deadline handoff checkpoint regressed')
    deployment = capacity.load_transaction()
    deployed = next(i for i in deployment['items'] if i['new_job_id'] == item['job_id'])
    capacity.current_identity(deployed, item['job_id'])
    capacity.old_guard(deployed, held=True)
    action['deadline_handoff_requested'] = True
    job_state['status'] = 'deadline_hold_handoff'
    tx['jobs'][str(item['job_id'])] = job_state
    save(tx, f"{item['job_id']}: persisted exact-owned-hold deadline normalization intent")
    if reason != 'JobHeldUser':
        base.command(['scontrol', 'hold', str(item['job_id'])])
    normalized = show(item['job_id']); stable(item, normalized)
    require(base.field(normalized, 'JobState') == 'PENDING'
            and base.field(normalized, 'Reason') == 'JobHeldUser'
            and base.field(normalized, 'Priority') == '0'
            and int(base.field(normalized, 'Restarts')) == action['before_restarts'] + 1,
            'Exact normalized deadline hold missing')
    deployment = capacity.load_transaction()
    deployed = next(i for i in deployment['items'] if i['new_job_id'] == item['job_id'])
    f = deployed.setdefault('fallback', {'reason': 'deadline', 'status': 'preparing'})
    require(f['reason'] == 'deadline' and f['status'] == 'preparing', 'Conflicting capacity fallback transaction')
    f['deadline_hold_requested'] = True
    f['guard_deadline_owned_hold'] = {
        'guard_transaction': str(TX), 'before_restarts': action['before_restarts'],
        'expected_restarts': action['before_restarts'] + 1, 'requested_at_utc': action['requested_at_utc'],
        'checkpoint': detail, 'normalized_record': normalized, 'at_utc': base.now()}
    capacity.event(deployment, 'Verified guard-owned deadline hold handed to capacity fallback')
    return fallback_result(tx, item, job_state, 'deadline', True)


def deadline_grace_expired(tx):
    return bool(tx.get('deadline_utc')) and datetime.now(timezone.utc) > datetime.fromisoformat(tx['deadline_utc']) + timedelta(minutes=5)


def reconcile_action(tx, item, action, job_state):
    job = item['job_id']
    record = show(job)
    stable(item, record)
    state, reason = base.field(record, 'JobState'), base.field(record, 'Reason')
    if action.get('release_requested') and state in {'RUNNING', 'CONFIGURING', 'COMPLETING', 'PENDING', 'COMPLETED'}:
        if reason not in {'JobHeldUser', 'JobHeldAdmin', 'job_requeued_in_held_state'}:
            finish_release(tx, item, record, action, job_state)
            return
    if action.get('deadline_handoff_requested') or deadline_grace_expired(tx):
        return deadline_hold_handoff(tx, item, action, job_state)
    own_hold(item, record, action)
    if retry_deadline_reached(tx):
        require_committed_before_deadline(tx,action)
    action['own_hold'] = True
    detail = checkpoint_and_writer(item)
    require(detail['step'] >= action['checkpoint_before']['step'], 'Checkpoint regressed after holding')
    require_progress(item, job_state, detail)
    action['checkpoint_before_release'] = detail
    action['held_before_release'] = record
    action['release_requested'] = True
    save(tx, f'{job}: audited owned hold, unchanged recipe/resources and valid advancing checkpoint; releasing')
    if deadline_grace_expired(tx):
        return deadline_hold_handoff(tx, item, action, job_state)
    if retry_deadline_reached(tx):
        require_committed_before_deadline(tx,action)
    base.command(['scontrol', 'release', str(job)])
    record = show(job)
    stable(item, record)
    require(base.field(record, 'JobState') in {'PENDING', 'RUNNING', 'CONFIGURING'}, 'Unexpected post-release state')
    require(base.field(record, 'Reason') not in {'JobHeldUser', 'JobHeldAdmin', 'job_requeued_in_held_state'}, 'Hold remained after release')
    finish_release(tx, item, record, action, job_state)


def observe_one(tx, item, *, apply):
    job = item['job_id']; key = str(job)
    job_state = tx['jobs'].get(key,dict(status='monitoring',attempts=[]))
    if job_state['status'] in {'completed','manual_stop','fallback_complete'}:
        return dict(job_id=job,status=job_state['status'])
    if job_state['status'] == 'fallback_in_progress':
        return fallback_result(tx,item,job_state,job_state['fallback_reason'],apply)
    identities = mapping(job)
    require(identities[job]['identity'] == item['identity'], 'Guarded cell identity changed')
    if identities[job].get('fallback_complete'):
        job_state['status']='fallback_complete';tx['jobs'][key]=job_state
        return dict(job_id=job,status='fallback_complete')
    run = Path(item['identity']['run_dir'])
    if recovery.complete(run):
        if apply:
            job_state['retirement_in_progress'] = True; tx['jobs'][key] = job_state
            save(tx, f'{job}: persisted dormant predecessor retirement intent')
            capacity.retire_completed(job)
            job_state.pop('retirement_in_progress', None)
            job_state['status']='completed';tx['jobs'][key]=job_state
            save(tx,f'{job}: terminal receipt validated; exact dormant long hold retired')
        return dict(job_id=job,status='completed')
    unfinished=[a for a in job_state['attempts'] if not a.get('released')]
    require(len(unfinished)<=1,'Multiple unfinished retry transactions')
    if unfinished:
        if apply:
            reconcile_action(tx,item,unfinished[0],job_state)
            return dict(job_id=job,status=job_state['status'],retries=len(job_state['attempts']))
        return dict(job_id=job,status='action_requires_reconciliation',scheduler_mutations=False)
    record=show(job);stable(item,record)
    state,reason=base.field(record,'JobState'),base.field(record,'Reason')
    if state in ACTIVE:
        return dict(job_id=job,status='monitoring_final_allocation' if retry_deadline_reached(tx) else 'monitoring',
                    state=state,reason=reason,retries=len(job_state['attempts']),runtime=base.field(record,'RunTime'))
    if state=='PENDING':
        if retry_deadline_reached(tx):
            return fallback_result(tx,item,job_state,'deadline',apply)
        return dict(job_id=job,status='monitoring',state=state,reason=reason,retries=len(job_state['attempts']),runtime=base.field(record,'RunTime'))
    require(state=='TIMEOUT',f'{job} entered {state}; manual inspection required, no blind restart')
    require(job not in base.queue() and recovery.state(job)=='TIMEOUT','TIMEOUT is not inactive and accounted')
    if retry_deadline_reached(tx):
        return fallback_result(tx,item,job_state,'deadline',apply)
    if len(job_state['attempts'])>=item['max_requeues']:
        return fallback_result(tx,item,job_state,'requeue_cap',apply)
    detail=checkpoint_and_writer(item)
    floor=max(item['initial_checkpoint_step'],job_state.get('last_resume_step',item['initial_checkpoint_step']))
    if detail['step']<=floor:
        return fallback_result(tx,item,job_state,'no_progress',apply)
    require_progress(item,job_state,detail)
    if not apply:
        return dict(job_id=job,status='would_requeue_same_id',checkpoint=detail,retry_number=len(job_state['attempts'])+1,scheduler_mutations=False)
    current=show(job);stable(item,current)
    require(base.field(current,'JobState')=='TIMEOUT' and base.field(current,'Restarts')==base.field(record,'Restarts'),'Attempt/state changed before requeuehold')
    require(job not in base.queue() and recovery.state(job)=='TIMEOUT','Timed-out job became active')
    detail=checkpoint_and_writer(item);require_progress(item,job_state,detail)
    if retry_deadline_reached(tx):
        return fallback_result(tx,item,job_state,'deadline',apply)
    action=dict(before=current,before_restarts=int(base.field(current,'Restarts')),checkpoint_before=detail,hold_intent=True,requested_at_utc=base.now(),hold_command_returned=False,released=False)
    job_state['attempts'].append(action);job_state['status']='holding';tx['jobs'][key]=job_state
    save(tx,f'{job}: durable requeuehold intent for bounded advancing retry; never repeat an uncertain call')
    if retry_deadline_reached(tx):
        job_state['attempts'].pop();job_state['status']='monitoring'
        return fallback_result(tx,item,job_state,'deadline',apply)
    base.command(['scontrol','requeuehold',str(job)])
    action['hold_command_returned']=True
    save(tx,f'{job}: requeuehold returned; audit owned hold and counters')
    reconcile_action(tx,item,action,job_state)
    return dict(job_id=job,status=job_state['status'],retries=len(job_state['attempts']))


def handle_transition_error(tx, item, error):
    """Bounded reconciliation of already committed actions; never new requeue."""
    job_state = tx['jobs'].setdefault(str(item['job_id']), dict(attempts=[]))
    pending = (job_state.get('status') == 'fallback_in_progress'
               or job_state.get('retirement_in_progress')
               or any(not a.get('released') for a in job_state['attempts']))
    job_state['transition_errors'] = job_state.get('transition_errors', 0) + 1
    within_cleanup = (bool(tx.get('cleanup_deadline_utc'))
                      and datetime.now(timezone.utc) < datetime.fromisoformat(tx['cleanup_deadline_utc']))
    if pending and job_state['transition_errors'] < 8 and within_cleanup:
        job_state['last_transition_error'] = repr(error)
        save(tx, f"{item['job_id']}: persisted transition needs reconciliation: {error}")
        return dict(job_id=item['job_id'], status='reconciliation_pending', error=repr(error))
    job_state.update(status='manual_stop', error=repr(error), long_route_command=item['long_route_command'],
                     dormant_old_job_id=item['dormant_old_job_id'])
    save(tx, f"{item['job_id']}: manual stop: {error}; no blind retry/release")
    return dict(job_id=item['job_id'], status='manual_stop', error=repr(error))


def run(*, watch, apply):
    plan = json.loads(PLAN.read_text())
    require(plan['controller_sha256'] == recovery.digest(__file__), 'Guard changed after preparation')
    require(plan['protocol_sha256'] == recovery.digest(PROTOCOL), 'Guard protocol changed')
    require(all(recovery.digest(p) == h for p, h in plan['helper_sha256'].items()), 'Guard helper changed')
    tx = json.loads(TX.read_text()) if TX.exists() else dict(schema=plan['schema'], plan_sha256=recovery.digest(PLAN), jobs={}, events=[])
    require(tx['plan_sha256'] == recovery.digest(PLAN), 'Guard plan changed')
    if apply and not tx.get('deadline_utc'):
        tx['started_at_utc'] = base.now()
        tx['deadline_utc'] = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
        tx['cleanup_deadline_utc']=(datetime.fromisoformat(tx['deadline_utc'])+timedelta(minutes=65)).isoformat()
        save(tx, 'Started absolute24-hour retry deadline plus65-minute final-allocation cleanup window')
    while True:
        if apply and datetime.now(timezone.utc)>=datetime.fromisoformat(tx['cleanup_deadline_utc']):
            save(tx,'Final cleanup deadline reached; no unbounded guard lifetime')
            return
        results = []
        for item in plan['rows']:
            try:
                with LEDGER_LOCK.open('a+') as ledger_lock:
                    fcntl.flock(ledger_lock, fcntl.LOCK_EX)
                    results.append(observe_one(tx, item, apply=apply))
            except Exception as error:
                result = (handle_transition_error(tx, item, error) if apply else
                          dict(job_id=item['job_id'], status='manual_stop', error=repr(error)))
                results.append(result)
        print(json.dumps(dict(at=base.now(), read_only=not apply, jobs=results)), flush=True)
        if not watch or all(r['status'] in {'completed', 'manual_stop', 'fallback_complete'} for r in results):
            return
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'once', 'watch'))
    parser.add_argument('--apply', action='store_true', help='Permit bounded same-ID timeout retries; default is read-only')
    args = parser.parse_args()
    require(args.phase != 'prepare' or not args.apply, 'prepare never applies scheduler changes')
    require(args.phase != 'watch' or args.apply, 'watch requires explicit --apply; use once for read-only inspection')
    with LOCK.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another guard instance holds the singleton lock')
        if args.phase == 'prepare':
            with LEDGER_LOCK.open('a+') as ledger_lock:
                fcntl.flock(ledger_lock, fcntl.LOCK_EX)
                prepare()
        else:
            run(watch=args.phase == 'watch', apply=args.apply)
