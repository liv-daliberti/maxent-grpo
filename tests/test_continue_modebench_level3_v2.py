"""Synthetic continuation failure guards; no jobs, models, or scientific writes."""
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import pytest

SOURCE=Path(__file__).resolve().parents[1]/'ops/exp_scaling/continue_modebench_level3_v2.py'
SPEC=importlib.util.spec_from_file_location('continue_v2_under_test',SOURCE)
continuation=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(continuation)


@pytest.fixture
def driver(tmp_path):
    with ExitStack() as stack:
        stack.enter_context(patch.object(continuation,'HERE',tmp_path))
        stack.enter_context(patch.object(continuation,'RESULT',tmp_path/'terminal.json'))
        value=continuation.Driver('sealed')
        value.verify=Mock()
        yield value


def test_atomic_records_cannot_overwrite_existing_evidence(tmp_path):
    path=tmp_path/'record.json';continuation.atomic_new(path,{'value':'old'})
    with pytest.raises(FileExistsError):continuation.atomic_new(path,{'value':'new'})
    assert continuation.read(path)=={'value':'old'}


def test_default_never_constructs_action_driver_or_touches_scheduler():
    with patch.object(continuation,'Driver') as driver,patch.object(continuation,'prepare') as prepare,\
         patch.object(continuation.subprocess,'run',side_effect=AssertionError('scheduler action')):
        assert continuation.main([])==0
    driver.assert_not_called();prepare.assert_not_called()


def test_fit_ownership_is_not_accepted_in_readonly_mode():
    with pytest.raises(ValueError,match='fit ownership requires explicit'):
        continuation.main(['--own-pantry-fit'])


def test_coordinator_ownership_is_exclusive(tmp_path):
    with patch.object(continuation,'HERE',tmp_path),continuation.coordinator_lock():
        with pytest.raises(BlockingIOError):
            with continuation.coordinator_lock():pass


def test_phase_verifies_sources_before_writing_intent_or_running_action(driver):
    action=Mock();driver.verify.side_effect=ValueError('source changed')
    with pytest.raises(ValueError,match='source changed'):driver.phase('pantry_fit',{},action)
    action.assert_not_called()
    assert not list(continuation.HERE.glob('*_intent.json'))


def test_completed_phase_is_idempotent_only_for_exact_inputs(driver):
    path=continuation.HERE/'product';path.write_text('registered')
    action=Mock(return_value=({'done':True},continuation.pins([path])))
    assert driver.phase('pantry_fit',{'input':'same'},action)=={'done':True}
    assert driver.phase('pantry_fit',{'input':'same'},action)=={'done':True}
    assert action.call_count==1
    with pytest.raises(ValueError,match='phase inputs changed'):
        driver.phase('pantry_fit',{'input':'different'},action)
    assert action.call_count==1


def test_published_phase_output_cannot_change_on_resume(driver):
    path=continuation.HERE/'product';path.write_text('old')
    driver.phase('pantry_fit',{},lambda:({},continuation.pins([path])))
    path.write_text('changed')
    with pytest.raises(ValueError,match='bytes changed'):driver.journal()


def test_unmatched_intent_never_replays_even_if_child_output_exists(driver):
    continuation.atomic_new(continuation.HERE/'finalize_intent.json',{'seal_sha256':'sealed'})
    (continuation.HERE/'complete_child_output').write_text('published before interruption')
    action=Mock()
    with pytest.raises(ValueError,match='ambiguous interrupted phase'):driver.phase('finalize',{},action)
    action.assert_not_called()


def test_orphan_result_never_adopted(driver):
    continuation.atomic_new(continuation.HERE/'finalize_result.json',{'status':'complete'})
    with pytest.raises(ValueError,match='ambiguous interrupted phase'):driver.journal()


def test_exception_after_child_publication_is_recorded_and_never_replayed(driver):
    path=continuation.HERE/'partial-output'
    def action():
        path.write_text('preserve this')
        raise RuntimeError('child interrupted')
    with pytest.raises(RuntimeError):driver.phase('finalize',{},action)
    assert path.read_text()=='preserve this'
    assert continuation.read(continuation.HERE/'finalize_result.json')['status']=='failed'
    retry=Mock()
    with pytest.raises(ValueError,match='failed phase'):driver.phase('finalize',{},retry)
    retry.assert_not_called()


def test_source_change_after_action_records_failure(driver):
    driver.verify.side_effect=[None,ValueError('source changed after action')]
    action=Mock(return_value=({},{}))
    with pytest.raises(ValueError,match='source changed after action'):driver.phase('finalize',{},action)
    assert action.call_count==1
    assert continuation.read(continuation.HERE/'finalize_result.json')['status']=='failed'


def test_changed_intent_breaks_result_binding(driver):
    driver.phase('pantry_fit',{},lambda:({},{}))
    path=continuation.HERE/'pantry_fit_intent.json'
    value=continuation.read(path);value['inputs']={'different':True};path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='phase ownership changed'):driver.journal()


def test_other_owner_result_is_rejected(driver):
    driver.phase('pantry_fit',{},lambda:({},{}))
    path=continuation.HERE/'pantry_fit_result.json'
    value=continuation.read(path);value['seal_sha256']='other';path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='phase ownership changed'):driver.journal()


def test_unregistered_phase_cannot_execute(driver):
    action=Mock()
    with pytest.raises(ValueError,match='unregistered'):driver.phase('alternate_seed',{},action)
    action.assert_not_called()


def tick_fixture(driver,tmp_path):
    recipes={domain:tmp_path/(domain+'.json') for domain in continuation.DOMAINS}
    receipts=[tmp_path/('receipt'+str(i)) for i in range(5)]
    completion=SimpleNamespace(scheduler_records=Mock(return_value={}),completion_status=Mock(return_value=('pending',[],list(continuation.NEW_IDS))))
    stack=ExitStack()
    stack.enter_context(patch.object(continuation,'RECIPES',recipes))
    stack.enter_context(patch.object(continuation,'PANTRY_RECEIPTS',receipts))
    stack.enter_context(patch.object(continuation,'module',return_value=completion))
    driver.phase=Mock(side_effect=AssertionError('no action phase expected'))
    return stack,recipes,receipts,completion


def test_existing_mathir_and_python_fit_owners_are_waited_for(driver,tmp_path):
    stack,recipes,_,_=tick_fixture(driver,tmp_path)
    with stack:
        for domain in ('countdown','graph_coloring','pantry'):
            continuation.atomic_new(recipes[domain],{'development_fit_pass':True})
        result=driver.tick()
    assert result['status']=='waiting_for_registered_recipes'
    assert result['missing_domains']==['mathir','python_factors']
    driver.phase.assert_not_called()


def test_any_failed_registered_recipe_stops_without_another_fit(driver,tmp_path):
    stack,recipes,_,_=tick_fixture(driver,tmp_path)
    with stack:
        continuation.atomic_new(recipes['python_factors'],{'development_fit_pass':False})
        driver.pantry_fit=Mock()
        result=driver.tick()
    assert result['status']=='needs_calibration_revision' and result['domain']=='python_factors'
    driver.phase.assert_not_called();driver.pantry_fit.assert_not_called()


def test_all_pantry_receipts_still_require_explicit_fit_owner(driver,tmp_path):
    stack,_,receipts,_=tick_fixture(driver,tmp_path)
    with stack:
        for path in receipts:path.write_text('{}')
        assert driver.tick()['status']=='waiting_for_explicit_pantry_fit_ownership'
    driver.phase.assert_not_called()


def test_failed_recovery_execution_stops_before_any_recipe_action(driver,tmp_path):
    stack,_,_,completion=tick_fixture(driver,tmp_path)
    completion.completion_status.return_value=('failed',['31149154'],[])
    with stack:result=driver.tick()
    assert result['status']=='needs_execution_review'
    driver.phase.assert_not_called()


def test_external_pantry_recipe_appearing_during_validation_is_not_overwritten(driver,tmp_path):
    recipe=tmp_path/'pantry.json';receipts=[tmp_path/f'receipt{i}' for i in range(5)]
    for path in receipts:path.write_text('{}')
    auditor=SimpleNamespace(validate_receipt=lambda *args,**kwargs:None)
    driver.own_pantry_fit=True
    original_phase=driver.phase
    def phase(name,inputs,action):
        recipe.write_text('{"external_owner":true}')
        return original_phase(name,inputs,action)
    driver.phase=phase;driver.command=Mock()
    with patch.object(continuation,'RECIPES',{'pantry':recipe}),patch.object(continuation,'PANTRY_RECEIPTS',receipts),\
         patch.object(continuation,'module',return_value=auditor):
        with pytest.raises(ValueError,match='another owner'):driver.pantry_fit()
    driver.command.assert_not_called()
    assert continuation.read(recipe)=={'external_owner':True}


def test_prepare_cannot_overwrite_an_inherited_scientific_pin(tmp_path):
    inherited={'files_sha256':{str(continuation.FITTER):'original-frozen-hash'},'directory_files':{},'models':{}}
    def read(path):
        if path==continuation.RECOVERY_SEAL:return inherited
        if path==continuation.READINESS:return {'files_sha256':{}}
        return {'jobs':[]}
    def digest(path):
        if path==continuation.RECOVERY_SEAL:return continuation.RECOVERY_SHA
        if path==continuation.RECOVERY_LEDGER:return continuation.LEDGER_SHA
        if path==continuation.READINESS:return continuation.READINESS_SHA
        if path==continuation.MATHIR_WATCHER:return continuation.MATHIR_WATCHER_SHA
        return 'changed-current-hash'
    with patch.object(continuation,'SEAL',tmp_path/'seal'),patch.object(continuation,'RESULT',tmp_path/'result'),\
         patch.object(continuation,'HERE',tmp_path),patch.object(continuation,'read',side_effect=read),\
         patch.object(continuation,'digest',side_effect=digest),patch.object(continuation,'verify_pins'),\
         patch.object(continuation,'module',return_value=SimpleNamespace(load_recipe=lambda *args:None)):
        with pytest.raises(ValueError,match='conflicting source pin'):
            continuation.prepare()
    assert not (tmp_path/'seal').exists()


@pytest.mark.parametrize('value',['0','-1','123\n456','123;cluster\n456','123 extra','  ','123;','123;cluster extra'])
def test_scheduler_identity_parser_rejects_ambiguous_or_invalid_response(value):
    with pytest.raises(ValueError,match='ambiguous scheduler'):continuation.parse_job_id(value)


def test_scheduler_identity_parser_accepts_one_positive_id_and_optional_cluster():
    assert continuation.parse_job_id('123\n')=='123'
    assert continuation.parse_job_id('123;cluster-1\n')=='123'


def test_models_and_engine_must_match_both_frozen_identities():
    models={'05b':{'path':'/small','vllm_version':'registered'},'3b':{'path':'/large','vllm_version':'registered'}}
    evaluator=SimpleNamespace(model_identity=lambda path,label:{'path':str(path)})
    with patch.object(continuation,'module',return_value=evaluator),\
         patch.object(continuation.importlib.metadata,'version',return_value='registered'):
        continuation.verify_models(models)
        with pytest.raises(ValueError,match='both frozen models'):continuation.verify_models({'3b':models['3b']})
    with patch.object(continuation,'module',return_value=evaluator),\
         patch.object(continuation.importlib.metadata,'version',return_value='changed'):
        with pytest.raises(ValueError,match='model or engine changed'):continuation.verify_models(models)


def recipe_fixture(tmp_path):
    recipes={domain:tmp_path/(domain+'.json') for domain in continuation.DOMAINS}
    root=continuation.ROOT/'var/results/modebench_level3_v2'
    records={}
    for domain,path in recipes.items():
        path.write_text('{}')
        candidate={'graph_coloring':'graph_v7','python_factors':'python_v5'}.get(domain,domain)
        records[domain]={'provenance':{'baseline_receipt_path':str(root/f'calibration_05b_{domain}.json'),
            'pools':{str(tier):{'receipt_path':str(root/f'calibration_3b_{candidate}_d{tier}.json')} for tier in range(4)}}}
        if candidate!=domain:records[domain]['candidate_revision']={'name':candidate}
    return recipes,records


def test_every_accepted_recipe_is_exactly_refit_before_acceptance(driver,tmp_path):
    recipes,records=recipe_fixture(tmp_path)
    load=Mock(side_effect=lambda path,domain:records[domain])
    with patch.object(continuation,'RECIPES',recipes),\
         patch.object(continuation,'module',return_value=SimpleNamespace(load_recipe=load)):
        result=driver.authenticate_recipes()
    assert load.call_count==5
    assert {call.args[1] for call in load.call_args_list}==set(continuation.DOMAINS)
    assert result==continuation.pins(recipes.values())


def test_revised_recipe_cannot_use_old_python_candidate_receipts(driver,tmp_path):
    recipes,records=recipe_fixture(tmp_path)
    records['python_factors']['provenance']['pools']['0']['receipt_path']=str(continuation.ROOT/'var/results/modebench_level3_v2/calibration_3b_python_factors_d0.json')
    with patch.object(continuation,'RECIPES',recipes),\
         patch.object(continuation,'module',return_value=SimpleNamespace(load_recipe=lambda path,domain:records[domain])):
        with pytest.raises(ValueError,match='different registered development receipts'):driver.authenticate_recipes()


def test_failed_exact_refit_stops_before_accepting_all_recipes(driver,tmp_path):
    recipes,_=recipe_fixture(tmp_path)
    with patch.object(continuation,'RECIPES',recipes),\
         patch.object(continuation,'module',return_value=SimpleNamespace(load_recipe=Mock(side_effect=ValueError('does not reproduce exactly')))):
        with pytest.raises(ValueError,match='does not reproduce exactly'):driver.authenticate_recipes()
    assert not list(continuation.HERE.glob('*_intent.json'))


def test_completed_recovery_with_missing_receipt_stops_before_waiting_for_recipes(driver,tmp_path):
    stack,_,_,completion=tick_fixture(driver,tmp_path)
    completion.completion_status.return_value=('complete',[],[])
    missing=tmp_path/'missing_registered_receipt.json'
    ledger=tmp_path/'recovery_ledger.json'
    continuation.atomic_new(ledger,{'jobs':[{'output':str(missing)}]})
    driver.authenticate_recipes=Mock(side_effect=AssertionError('recipe authentication must not start'))
    driver.pantry_fit=Mock(side_effect=AssertionError('fitting must not start'))
    with stack,patch.object(continuation,'RECOVERY_LEDGER',ledger):
        result=driver.tick()
    assert result['status']=='needs_execution_review'
    assert result['missing_receipts']==[str(missing)]
    driver.phase.assert_not_called()
    driver.authenticate_recipes.assert_not_called()
    driver.pantry_fit.assert_not_called()
