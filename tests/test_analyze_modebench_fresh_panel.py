"""Authenticate synthetic committed collection receipts without sampling hardware."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ops'))
import analyze_modebench_fresh_panel as panel
spec=importlib.util.spec_from_file_location('fresh_collector_test_fixtures',Path(__file__).with_name('test_modebench_fresh_concentration.py'))
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)


@pytest.fixture
def completed():
    case=fixtures.FreshConcentrationTests();case.setUp()
    try:
        case.plan['scheduler']={'attention_backend':'XFORMERS'}
        case.save_plan()
        case.collect(fixtures.FakeEngine())
        yield case
    finally:
        case.doCleanups()


def authenticate(case):
    return panel.authenticate_task(case.plan_path,case.plan,case.task,case.rows)


def test_valid_panel_reconstructs_every_raw_slot_and_correct_key(completed):
    r=authenticate(completed)
    assert r['draws']==8192 and len(r['prompts'])==128
    assert all(p['draws']==p['correct_draws']==64 for p in r['prompts'])
    assert all(p['distinct_all']==3 for p in r['prompts'])


def test_changed_flat_response_is_rejected_even_with_result_hash_updated(completed):
    output=Path(completed.plan['output_root'])/completed.task['task_id']
    path=output/'responses.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['canonical_key']='forged-mode'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result=json.loads((output/'result.json').read_text());result['responses_sha256']=panel.collector.file_sha(path)
    (output/'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError,match='flat responses'):authenticate(completed)


def test_missing_committed_child_cannot_be_silently_dropped(completed):
    output=Path(completed.plan['output_root'])/completed.task['task_id']
    path=next(output.glob('batch_b*.json'));batch=json.loads(path.read_text())
    batch['requests'][0]['attempts'].pop();batch['requests_sha256']=panel.collector.sha(batch['requests'])
    path.write_text(json.dumps(batch))
    with pytest.raises(ValueError,match='Missing child'):authenticate(completed)


def test_repeated_child_stream_fails_before_any_summary(completed):
    output=Path(completed.plan['output_root'])/completed.task['task_id']
    path=next(output.glob('batch_b*.json'));batch=json.loads(path.read_text())
    batch['requests'][0]['attempts'][1]['child_sampling_seed']=batch['requests'][0]['attempts'][0]['child_sampling_seed']
    batch['requests_sha256']=panel.collector.sha(batch['requests']);path.write_text(json.dumps(batch))
    with pytest.raises(ValueError,match='child index or RNG'):authenticate(completed)


def test_runtime_versions_are_authenticated(completed):
    output=Path(completed.plan['output_root'])/completed.task['task_id']
    path=output/'runtime.json';runtime=json.loads(path.read_text());runtime['runtime']['versions']['vllm']='0.9'
    path.write_text(json.dumps(runtime))
    with pytest.raises(ValueError,match='Runtime identity'):authenticate(completed)


def full_plan(tmp_path):
    tasks=[]
    for scale in ('qwen05b','falcon1b','qwen3b'):
        for domain in ('graph_coloring','pantry_plan'):
            for method in ('initial',*panel.METHODS):
                for seed in range(55,60):
                    tasks.append({'task_id':f'{scale}-{domain}-{method}-{seed}','model_scale':scale,'domain':domain,
                                  'method':method,'training_seed':None if method=='initial' else seed,'checkpoint_stage':'initial' if method=='initial' else 'terminal',
                                  'eval_replica_id':seed if method=='initial' else None,
                                  'files':[{'name':'model.safetensors','sha256':'same-fixed-weights'}]})
    return {'tasks':tasks,'output_root':str(tmp_path/'outputs')}


def test_inventory_requires_complete_six_blocks_four_methods_five_seeds(tmp_path):
    plan=full_plan(tmp_path);index,groups=panel.validate_panel_inventory(plan)
    assert len(index)==150 and len(groups)==6
    for mutated in (dict(plan,tasks=plan['tasks'][:-1]),dict(plan,tasks=plan['tasks'][:-1]+[plan['tasks'][0]])):
        with pytest.raises(ValueError):panel.validate_panel_inventory(mutated)
    changed=copy.deepcopy(plan);changed['tasks'][0]['files'][0]['sha256']='different-initial-model'
    with pytest.raises(ValueError,match='identical initial'):panel.validate_panel_inventory(changed)


def test_initial_pairing_requires_explicit_index(tmp_path):
    task=full_plan(tmp_path)['tasks'][0];task['eval_replica_id']='ambiguous-index'
    with pytest.raises(ValueError,match='explicit'):panel.seed_index(task)
    task['eval_replica_id']=55
    assert panel.seed_index(task)==55


def test_audit_preserves_every_absent_task_and_refuses_completion(tmp_path,monkeypatch):
    plan=full_plan(tmp_path);path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    monkeypatch.setattr(panel.collector,'validate_plan',lambda plan,templates=None:{t['task_id']:[] for t in plan['tasks']})
    audit,*_=panel.audit_panel(path)
    assert audit['status']=='incomplete'
    assert audit['expected_tasks']==150 and audit['authenticated_tasks']==0
    assert audit['expected_response_slots']==1228800 and audit['authenticated_response_slots']==0
    assert len(audit['tasks'])==150 and {t['status'] for t in audit['tasks']}=={'not_started'}


def test_consistently_fingerprinted_wrong_attention_backend_is_rejected(completed):
    output=Path(completed.plan['output_root'])/completed.task['task_id']
    path=output/'runtime.json';runtime=json.loads(path.read_text())
    runtime['runtime']['environment']['VLLM_ATTENTION_BACKEND']='FLASH_ATTN'
    runtime['runtime_sha256']=panel.collector.runtime_fingerprint(runtime['runtime'])
    path.write_text(json.dumps(runtime))
    with pytest.raises(ValueError,match='attention backend'):authenticate(completed)


def test_attempt_environment_must_match_runtime_stable_fields(completed):
    output=Path(completed.plan['output_root'])/completed.task['task_id']
    path=next((output/'attempts').glob('*.json'));attempt=json.loads(path.read_text())
    attempt['environment']['VLLM_ATTENTION_BACKEND']='FLASH_ATTN';path.write_text(json.dumps(attempt))
    with pytest.raises(ValueError,match='stable environment'):authenticate(completed)
