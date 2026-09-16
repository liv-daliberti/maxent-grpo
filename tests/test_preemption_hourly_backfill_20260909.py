"""Verify mixed authoritative mappings and frozen scheduler continuations offline."""
import copy
import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops/exp_scaling'))
c = importlib.import_module('backfill_preempted_hourly_20260909')


@pytest.fixture
def ledgers(tmp_path, monkeypatch):
    for name in ('SOURCE','AGGREGATE','CONTINUATIONS','E120_PRIMARY'):
        monkeypatch.setattr(c,name,tmp_path/(name+'.json'))
    monkeypatch.setattr(c,'MUTABLE',(c.SOURCE,c.AGGREGATE,c.CONTINUATIONS))
    monkeypatch.setattr(c.prior.campaign,'E120_LEDGER',c.E120_PRIMARY)
    monkeypatch.setattr(c.prior.campaign,'E120_CONTINUATIONS',c.CONTINUATIONS)
    monkeypatch.setattr(c,'ART',tmp_path)
    monkeypatch.setattr(c,'TX',tmp_path/'transaction.json')
    source=[]
    for i in range(50):
        source.append(dict(job_id=100+i,domain='mathir',arm='maxrl' if i%2==0 else 'replay_maxrl',seed=i,
                           run_dir=f'/run/{i}',run_stamp=f'run_{i}',previous_job_ids=[]))
    aggregate=[dict(r,scale='qwen3b') for r in source]
    aggregate += [dict(r,job_id=1000+j*100+i,run_dir=f'/other/{j}/{i}',run_stamp=f'other_{j}_{i}',scale=f'other{j}')
                  for j in range(2) for i,r in enumerate(source)]
    continuation=[dict(continuation_job_id=500,original_job_id=400,domain='graph_coloring',seed=74,model_key='qwen3b',
                       run_dir='/graph',run_stamp='graph',intermediate_job_ids=[450],released=True)]
    c.SOURCE.write_text(json.dumps({'runs':source}));c.AGGREGATE.write_text(json.dumps({'runs':aggregate}))
    continuation[0]['runtime_changes']={'OAT_ZERO_VLLM_GPU_RATIO':{'before':'0.25','after':'0.40'}}
    continuation[0].update(optimizer_update_changed=False,treatment_changed=False)
    continuation.append(dict(continuation[0],continuation_job_id=501,original_job_id=401,run_dir='/other_graph',run_stamp='other_graph',seed=70))
    c.E120_PRIMARY.write_text(json.dumps({'runs':[dict(row,job_id=row['original_job_id']) for row in continuation],'immutable_fixture_marker':True}))
    payload=dict(schema='e120r1_scheduler_continuation_jobs_v1',original_ledger=str(c.E120_PRIMARY),original_ledger_sha256=c.prior.digest(c.E120_PRIMARY),
                 same_scientific_cells=True,same_run_directories=True,optimizer_update_changed=False,treatment_changed=False,pvl_excluded=True,
                 released=True,installed=True,outcomes_inspected=False,scheduler_only=False,runtime_allocation_only=True,continuations=continuation)
    c.CONTINUATIONS.write_text(json.dumps(payload))
    items=[]
    for cohort,row,new in [('e118',source[0],200),('e118',source[1],201),('e120',continuation[0],600)]:
        old=row['job_id'] if cohort=='e118' else row['continuation_job_id']
        items.append(dict(cohort=cohort,domain=row['domain'],arm=row.get('arm'),seed=row['seed'],run_dir=row['run_dir'],
                          run_stamp=row['run_stamp'],row_before=copy.deepcopy(row),old_job_id=old,new_job_id=new,
                          checkpoint={'step':960 if cohort=='e118' else 576},new_held_record='HELD',
                          before='MinMemoryNode=116G Nice=200' if cohort=='e118' else 'MinMemoryNode=128G Nice=100',
                          long_route_command=['old-frozen-launcher']))
    return items


def values():
    return {p:json.loads(p.read_text()) for p in c.MUTABLE}


@pytest.mark.parametrize('memory',['116G','128G'])
def test_command_changes_only_recovery_export(memory):
    env={'OAT_ZERO_RESUME_STEPS':'192','OAT_ZERO_VLLM_GPU_RATIO':'0.40','OAT_ZERO_AUTO_RESUME':'1',
         'MODEL':'frozen-model','SEED':'73','SAVE_PATH':'/registered','EVAL_EVERY':'192','TRAIN_STEPS':'3072'}
    original=['sbatch','--partition=lowprio','--account=mltheory','--mem='+memory,'--cpus-per-task=16',
              '--time=3-00:00:00','--export=ALL,'+','.join(f'{k}={v}' for k,v in env.items()),'/frozen/train.slurm']
    item={'original_command':original,'before':f'MinMemoryNode={memory} Nice=200','old_job_id':12,'comment':'owned-test'}
    command=c.build_command(item,short=True)
    assert c.b.exports(command)==dict(env,OAT_ZERO_RESUME_STEPS='48')
    assert command[-1]==original[-1] and '--mem='+memory in command
    assert '--time=1:00:00' in command and '--partition=all' in command
    assert '--nodelist=node105,node202,node203,node204' in command and '--hold' in command
    assert c.b.exports(c.build_command(item,short=False))==env


def test_mixed_staging_and_reconciliation_preserve_primary(ledgers):
    tx={'items':ledgers,'mutable_before_sha256':{str(p):c.prior.digest(p) for p in c.MUTABLE},
        'e120_primary_sha256':c.prior.digest(c.E120_PRIMARY),'events':[]}
    c.stage_ledgers(tx)
    assert tx['ledgers_committed']
    for item in ledgers:c.current_identity(item,item['new_job_id'])
    assert json.loads(c.E120_PRIMARY.read_text())['immutable_fixture_marker'] is True
    assert json.loads(c.CONTINUATIONS.read_text())['continuations'][0]['released'] is False
    hashes={p:c.prior.digest(p) for p in c.MUTABLE}
    c.stage_ledgers(tx)
    assert hashes=={p:c.prior.digest(p) for p in c.MUTABLE}


def test_graph_fallback_maps_only_continuation(ledgers):
    graph=ledgers[-1];v=values();source_before=copy.deepcopy(v[c.SOURCE]);aggregate_before=copy.deepcopy(v[c.AGGREGATE])
    c.remap_item(v,graph,graph['new_job_id'],'NEWHELD')
    c.remap_item(v,graph,graph['old_job_id'],'OLDHELD',fallback=True)
    assert v[c.SOURCE]==source_before and v[c.AGGREGATE]==aggregate_before
    row=v[c.CONTINUATIONS]['continuations'][0]
    assert row['continuation_job_id']==500 and row['original_job_id']==400
    assert row['intermediate_job_ids']==[450,600] and row['released'] is False
    assert row['new_placement']['memory']=='128G' and row['new_placement']['partition']=='lowprio'
    assert row['runtime_changes']=={'OAT_ZERO_VLLM_GPU_RATIO':{'before':'0.25','after':'0.40'}}
    assert row['hourly_backfill_runtime_changes']=={'OAT_ZERO_RESUME_STEPS':{'before':'48','after':'192'}}
    c.checked_images({c.CONTINUATIONS:v[c.CONTINUATIONS]})


def test_mathir_fallback_preserves_graph_and_peer(ledgers):
    first,peer,graph=ledgers;v=values();graph_before=copy.deepcopy(v[c.CONTINUATIONS])
    for item in ledgers:c.remap_item(v,item,item['new_job_id'],'HELD')
    graph_after=copy.deepcopy(v[c.CONTINUATIONS]);peer_after=copy.deepcopy(v[c.SOURCE]['runs'][1])
    c.remap_item(v,first,first['old_job_id'],'OLDHELD',fallback=True)
    assert v[c.CONTINUATIONS]==graph_after and v[c.SOURCE]['runs'][1]==peer_after
    assert v[c.SOURCE]['runs'][0]['previous_job_ids']==[first['new_job_id']]
    c.checked_images(v)


def test_identity_rejects_wrong_graph_effective_id(ledgers):
    graph=ledgers[-1];c.current_identity(graph,graph['old_job_id'])
    with pytest.raises(AssertionError):c.current_identity(graph,graph['new_job_id'])


def test_identity_rejects_divergent_source_aggregate(ledgers):
    value=json.loads(c.AGGREGATE.read_text());value['runs'][0]['job_id']=99999;c.AGGREGATE.write_text(json.dumps(value))
    with pytest.raises(AssertionError):c.current_identity(ledgers[0],ledgers[0]['old_job_id'])


def test_graph_release_ack_leaves_primary_and_e118_unchanged(ledgers):
    graph=ledgers[-1];v=values();c.remap_item(v,graph,graph['new_job_id'],'HELD');c.CONTINUATIONS.write_text(json.dumps(v[c.CONTINUATIONS]))
    before={p:c.prior.digest(p) for p in (c.SOURCE,c.AGGREGATE,c.E120_PRIMARY)}
    tx={'events':[],'e120_primary_sha256':c.prior.digest(c.E120_PRIMARY)}
    c.mark_e120_released(tx,graph,graph['new_job_id'])
    assert json.loads(c.CONTINUATIONS.read_text())['continuations'][0]['released'] is True
    assert before=={p:c.prior.digest(p) for p in before}
    c.mark_e120_released(tx,graph,graph['new_job_id'])


def fallback_fixture(ledgers, monkeypatch, index):
    item=ledgers[index]
    value=values()
    c.remap_item(value,item,item['new_job_id'],'HELD')
    for path,rows in value.items():path.write_text(json.dumps(rows))
    item['before'] += ' Account=mltheory NumCPUs=16 NumTasks=1 CPUs/Task=16 Requeue=1 ExcNodeList=excluded WorkDir=/work TresPerNode=gres/gpu:a5000:1'
    item.update(old_fallback_held=True,released=True,original_command=['sbatch','/frozen.slurm'])
    tx={'items':ledgers,'events':[],'e120_primary_sha256':c.prior.digest(c.E120_PRIMARY)}
    monkeypatch.setattr(c,'load_transaction',lambda:tx)
    monkeypatch.setattr(c,'old_guard',lambda item,held: 'OWNED HELD')
    monkeypatch.setattr(c.b,'queue',lambda:{item['old_job_id']})
    monkeypatch.setattr(c.prior.recovery,'state',lambda job:'TIMEOUT')
    monkeypatch.setattr(c.prior.recovery,'complete',lambda run:False)
    monkeypatch.setattr(c.prior.recovery,'active_writers',lambda:{item['run_dir']:{item['old_job_id']}})
    monkeypatch.setattr(c.prior,'checkpoint',lambda run:dict(item['checkpoint'],path='/valid/cp'))
    validation=importlib.import_module('guard_mathir_hourly_timeouts_20260909')
    monkeypatch.setattr(validation,'checked_checkpoint',lambda path:{'step':item['checkpoint']['step']})
    monkeypatch.setattr(c,'nodes',lambda record:c.LONG_NODES)
    monkeypatch.setattr(c.b,'submit_tokens',lambda record:item['original_command'])
    after='JobState=PENDING Priority=999 Reason=Priority TimeLimit=3-00:00:00 Partition=lowprio '+item['before']
    monkeypatch.setattr(c.b,'show',lambda job:after)
    mutations=[]
    def command(argv,*args,**kwargs):
        assert argv==['scontrol','release',str(item['old_job_id'])]
        # The exact authoritative mapping must already be restored at release.
        c.current_identity(item,item['old_job_id'])
        mutations.append(argv)
    monkeypatch.setattr(c.b,'command',command)
    return item,tx,mutations


def test_end_to_end_graph_fallback_restores_before_release(ledgers,monkeypatch):
    item,tx,mutations=fallback_fixture(ledgers,monkeypatch,2)
    immutable={p:p.read_bytes() for p in (c.SOURCE,c.AGGREGATE,c.E120_PRIMARY)}
    c.fallback(item['new_job_id'],'no_progress')
    assert len(mutations)==1 and item['fallback']['status']=='complete' and not item['old_fallback_held']
    row=json.loads(c.CONTINUATIONS.read_text())['continuations'][0]
    assert row['continuation_job_id']==item['old_job_id'] and row['released'] is True
    assert all(p.read_bytes()==content for p,content in immutable.items())
    c.fallback(item['new_job_id'],'no_progress')
    assert len(mutations)==1


def test_promotion_recovers_after_first_ledger_write(ledgers,monkeypatch):
    tx={'items':ledgers,'mutable_before_sha256':{str(p):c.prior.digest(p) for p in c.MUTABLE},
        'e120_primary_sha256':c.prior.digest(c.E120_PRIMARY),'events':[]}
    real_save=c.save;failed=[]
    def interrupted(path,value):
        real_save(path,value)
        if Path(path)==c.SOURCE and not failed:
            failed.append(True);raise RuntimeError('simulated lost acknowledgement after SOURCE replacement')
    monkeypatch.setattr(c,'save',interrupted)
    with pytest.raises(RuntimeError,match='lost acknowledgement'):c.stage_ledgers(tx)
    assert json.loads(c.SOURCE.read_text())['runs'][0]['job_id']==ledgers[0]['new_job_id']
    assert json.loads(c.AGGREGATE.read_text())['runs'][0]['job_id']==ledgers[0]['old_job_id']
    c.stage_ledgers(tx)
    for item in ledgers:c.current_identity(item,item['new_job_id'])
    assert tx['ledgers_committed'] and json.loads(c.E120_PRIMARY.read_text())['immutable_fixture_marker'] is True


def test_fallback_recovers_after_first_ledger_write(ledgers,monkeypatch):
    item,tx,mutations=fallback_fixture(ledgers,monkeypatch,0)
    before_graph=c.CONTINUATIONS.read_bytes();before_primary=c.E120_PRIMARY.read_bytes()
    real_save=c.save;failed=[]
    def interrupted(path,value):
        real_save(path,value)
        if Path(path)==c.SOURCE and not failed:
            failed.append(True);raise RuntimeError('simulated lost acknowledgement after SOURCE replacement')
    monkeypatch.setattr(c,'save',interrupted)
    with pytest.raises(RuntimeError,match='lost acknowledgement'):c.fallback(item['new_job_id'],'no_progress')
    assert not mutations
    assert json.loads(c.SOURCE.read_text())['runs'][0]['job_id']==item['old_job_id']
    assert json.loads(c.AGGREGATE.read_text())['runs'][0]['job_id']==item['new_job_id']
    c.fallback(item['new_job_id'],'no_progress')
    assert len(mutations)==1 and item['fallback']['status']=='complete'
    assert c.CONTINUATIONS.read_bytes()==before_graph and c.E120_PRIMARY.read_bytes()==before_primary


def test_real_resolver_preserves_all_e120_mappings_through_promotion_and_fallback(ledgers,monkeypatch):
    campaign=c.prior.campaign
    monkeypatch.setattr(campaign,'E120_LEDGER',c.E120_PRIMARY)
    monkeypatch.setattr(campaign,'E120_CONTINUATIONS',c.CONTINUATIONS)
    baseline=campaign.e120_continuation_jobs(c.E120_PRIMARY)
    assert baseline=={400:500,401:501}
    primary_before=c.E120_PRIMARY.read_bytes();graph=ledgers[-1]
    tx={'items':ledgers,'mutable_before_sha256':{str(p):c.prior.digest(p) for p in c.MUTABLE},
        'e120_primary_sha256':c.prior.digest(c.E120_PRIMARY),'events':[]}
    c.stage_ledgers(tx)
    assert campaign.e120_continuation_jobs(c.E120_PRIMARY)=={400:600,401:501}
    c.mark_e120_released(tx,graph,graph['new_job_id'])
    assert campaign.e120_continuation_jobs(c.E120_PRIMARY)=={400:600,401:501}
    value={c.CONTINUATIONS:json.loads(c.CONTINUATIONS.read_text())}
    c.remap_item(value,graph,graph['old_job_id'],'OLD HELD',fallback=True)
    c.checked_images(value);c.CONTINUATIONS.write_text(json.dumps(value[c.CONTINUATIONS]))
    assert campaign.e120_continuation_jobs(c.E120_PRIMARY)==baseline
    c.mark_e120_released(tx,graph,graph['old_job_id'])
    assert campaign.e120_continuation_jobs(c.E120_PRIMARY)==baseline
    assert c.E120_PRIMARY.read_bytes()==primary_before
