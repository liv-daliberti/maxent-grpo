"""Protect fresh64 slot identity, unchanged controls, and preflight gating."""
import importlib.util,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ops'))
import prepare_modebench_discovery_hosted as prep
import run_modebench_discovery_hosted as runner

def pool(arm='original'):
    rows=[];prompts=[];items=[]
    for level in (2,3):
        for domain in ('python_factors','mathir','pantry_plan'):
            for idx in range(16):
                row={'level':level,'domain':domain,'row_index':idx,'problem':f'problem{idx}','answer':'unseen verifier spec'}
                messages=[{'role':'system','content':arm},{'role':'user','content':row['problem']}]
                payload={'model':'test-model','input':messages,'reasoning':{'effort':'medium'},'max_output_tokens':8192,'store':False}
                rows.append(row);prompts.append({**row,'arm':arm,'messages':messages})
                items.append({**row,'sample_index':0,'sample_id':f'L{level}_{domain}_{idx:03d}_0','row_sha256':prep.sha_object(row),'request':payload,'request_sha256':prep.sha_object(payload)})
    return rows,prompts,items

def test_fresh64_slots_keep_exact_payload_and_are_deterministic():
    rows,prompts,items=pool();out=prep.expand(items,rows,prompts,'original')
    assert len(out)==6144
    refs={prep.identity(x):x for x in items}
    groups={}
    for item in out:
        groups.setdefault(prep.identity(item),[]).append(item['sample_index'])
        assert item['request']==refs[prep.identity(item)]['request']
        assert item['group_id']==item['sample_id']
        assert item['choice_index']==0
        assert item['experiment_condition']=='sampling_budget_ablation_v1'
    assert all(sorted(v)==list(range(64)) for v in groups.values())
    assert out==prep.expand(list(reversed(items)),list(reversed(rows)),list(reversed(prompts)),'original')

def test_wording_arms_share_slot_order():
    r,p,x=pool('original');original=prep.expand(x,r,p,'original')
    r,p,x=pool('neutral');neutral=prep.expand(x,r,p,'neutral')
    assert [x['sample_id'] for x in original]==[x['sample_id'] for x in neutral]
    for a,b in zip(original,neutral):
        assert a['request']['input'][1]==b['request']['input'][1]
        assert {k:v for k,v in a['request'].items() if k!='input'}=={k:v for k,v in b['request'].items() if k!='input'}

@pytest.mark.parametrize('change',['problem','digest','messages'])
def test_modified_source_fails_closed(change):
    r,p,x=pool()
    if change=='problem':r[0]['problem']='changed'
    elif change=='digest':x[0]['request_sha256']='0'*64
    else:p[0]['messages']=[{'role':'user','content':'different problem'}]
    with pytest.raises(AssertionError):prep.expand(x,r,p,'original')

def test_bound_artifact_change_rejected(tmp_path):
    p=tmp_path/'rows.jsonl';p.write_text('original')
    m={'artifact_sha256':{'rows.jsonl':prep.sha_file(p)},'code_sha256':{}}
    prep.authenticate(tmp_path,m);p.write_text('changed')
    with pytest.raises(ValueError):prep.authenticate(tmp_path,m)

def fake_cohort(tmp_path,name='cohort',arm='original'):
    out=tmp_path/name;out.mkdir()
    (out/'manifest.json').write_text(json.dumps({'protocol':'responses'}))
    return {'run_dir':str(out),'model':'test-model','arm':arm}

def test_missing_credential_cannot_start_collector(monkeypatch,tmp_path):
    entry=fake_cohort(tmp_path)
    monkeypatch.setattr(runner,'inventory',lambda b:[entry])
    monkeypatch.setattr(runner,'authenticated_completed',lambda p:[])
    monkeypatch.delenv('AZURE_OPENAI_API_KEY',raising=False)
    monkeypatch.setattr(runner.subprocess,'Popen',lambda *a,**kw:pytest.fail('No process may launch without credential'))
    with pytest.raises(ValueError,match='Supply AZURE_OPENAI_API_KEY'):runner.run('preflight',tmp_path)

def test_full_requires_preflight_before_any_process(monkeypatch,tmp_path):
    entry=fake_cohort(tmp_path)
    monkeypatch.setattr(runner,'inventory',lambda b:[entry])
    monkeypatch.setattr(runner,'credential',lambda p:pytest.fail('Preflight failure must precede credential access'))
    monkeypatch.setattr(runner,'authenticated_completed',lambda p:[])
    monkeypatch.setattr(runner.subprocess,'Popen',lambda *a,**kw:pytest.fail('No full run without authenticated preflight'))
    with pytest.raises(ValueError,match='preflight'):runner.run('full',tmp_path)

def test_provider_identity_reuse_across_arms_rejected(monkeypatch,tmp_path):
    entries=[fake_cohort(tmp_path,'original','original'),fake_cohort(tmp_path,'neutral','neutral')]
    monkeypatch.setattr(runner,'authenticated_completed',lambda p:[{'sample_id':Path(p).name,'provider_sample_identity':['duplicate',0]}])
    with pytest.raises(ValueError,match='Duplicate provider sample across'):runner.authenticated_inventory(entries)

def test_distinct_provider_samples_accepted(monkeypatch,tmp_path):
    entries=[fake_cohort(tmp_path,'original','original'),fake_cohort(tmp_path,'neutral','neutral')]
    monkeypatch.setattr(runner,'authenticated_completed',lambda p:[{'sample_id':'slot0','provider_sample_identity':[Path(p).name,0]}])
    assert sum(len(v) for v in runner.authenticated_inventory(entries).values())==2

def test_preflight_recovery_needs_no_credential_or_network(monkeypatch,tmp_path):
    entry=fake_cohort(tmp_path)
    evidence=[{'sample_id':'slot0','provider_sample_identity':['retained',0]}]
    monkeypatch.setattr(runner,'inventory',lambda b:[entry])
    monkeypatch.setattr(runner,'authenticated_completed',lambda p:evidence)
    monkeypatch.setattr(runner,'credential',lambda p:pytest.fail('Recovery must not read a secret'))
    monkeypatch.setattr(runner.subprocess,'Popen',lambda *a,**kw:pytest.fail('Recovery must not request a new sample'))
    runner.run('preflight',tmp_path)
    marker=json.loads((Path(entry['run_dir'])/'discovery_preflight.json').read_text())
    assert marker['evidence']==evidence and marker['retained_in_production']

def test_completed_full_recovery_needs_no_credential_or_network(monkeypatch,tmp_path):
    entry=fake_cohort(tmp_path)
    evidence=[{'sample_id':f'slot{i}','provider_sample_identity':[f'retained{i}',0]} for i in range(6144)]
    monkeypatch.setattr(runner,'inventory',lambda b:[entry])
    monkeypatch.setattr(runner,'authenticated_completed',lambda p:evidence)
    monkeypatch.setattr(runner,'credential',lambda p:pytest.fail('Completed recovery must not read a secret'))
    monkeypatch.setattr(runner.subprocess,'Popen',lambda *a,**kw:pytest.fail('Completed recovery must not launch HTTP'))
    runner.run('full',tmp_path)
    marker=json.loads((Path(entry['run_dir'])/'discovery_preflight.json').read_text())
    assert marker['authenticated_sample_count']==6144
