"""Synthetic continuation failure guards; no jobs, models, or scientific writes."""
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import pytest

SOURCE=Path(__file__).resolve().parents[1]/'ops/exp_scaling/continue_modebench_level3_python_v6.py'
SPEC=importlib.util.spec_from_file_location('continue_python_v6_under_test',SOURCE)
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
    with pytest.raises(SystemExit) as error:
        continuation.main(['--own-pantry-fit'])
    assert error.value.code == 2


def test_coordinator_ownership_is_exclusive(tmp_path):
    with patch.object(continuation,'HERE',tmp_path),continuation.coordinator_lock():
        with pytest.raises(BlockingIOError):
            with continuation.coordinator_lock():pass


def test_phase_verifies_sources_before_writing_intent_or_running_action(driver):
    action=Mock();driver.verify.side_effect=ValueError('source changed')
    with pytest.raises(ValueError,match='source changed'):driver.phase('accepted_recipes',{},action)
    action.assert_not_called()
    assert not list(continuation.HERE.glob('*_intent.json'))


def test_completed_phase_is_idempotent_only_for_exact_inputs(driver):
    path=continuation.HERE/'product';path.write_text('registered')
    action=Mock(return_value=({'done':True},continuation.pins([path])))
    assert driver.phase('accepted_recipes',{'input':'same'},action)=={'done':True}
    assert driver.phase('accepted_recipes',{'input':'same'},action)=={'done':True}
    assert action.call_count==1
    with pytest.raises(ValueError,match='phase inputs changed'):
        driver.phase('accepted_recipes',{'input':'different'},action)
    assert action.call_count==1


def test_published_phase_output_cannot_change_on_resume(driver):
    path=continuation.HERE/'product';path.write_text('old')
    driver.phase('accepted_recipes',{},lambda:({},continuation.pins([path])))
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
    driver.phase('accepted_recipes',{},lambda:({},{}))
    path=continuation.HERE/'accepted_recipes_intent.json'
    value=continuation.read(path);value['inputs']={'different':True};path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='phase ownership changed'):driver.journal()


def test_other_owner_result_is_rejected(driver):
    driver.phase('accepted_recipes',{},lambda:({},{}))
    path=continuation.HERE/'accepted_recipes_result.json'
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
    stack.enter_context(patch.object(continuation,'V6_AUDIT',tmp_path/'v6_audit.json'))
    stack.enter_context(patch.object(continuation,'V6_FIT_FAILURE',tmp_path/'v6_fit_failure.json'))
    stack.enter_context(patch.object(continuation,'authenticate_python_v6_submissions',return_value=({'job_ids':['v6id0','v6id1','v6id2','v6id3'],'outputs':[]},{})))
    stack.enter_context(patch.object(continuation,'module',return_value=completion))
    driver.phase=Mock(side_effect=AssertionError('no action phase expected'))
    return stack,recipes,receipts,completion


def test_four_accepted_domains_wait_only_for_python_v6_owner(driver,tmp_path):
    stack,recipes,_,_=tick_fixture(driver,tmp_path)
    with stack:
        for domain in ('countdown','graph_coloring','mathir','pantry'):
            continuation.atomic_new(recipes[domain],{'development_fit_pass':True})
        result=driver.tick()
    assert result['status']=='waiting_for_registered_recipes'
    assert result['missing_domains']==['python_factors']
    driver.phase.assert_not_called()


def test_any_failed_registered_recipe_stops_without_another_fit(driver,tmp_path):
    stack,recipes,_,_=tick_fixture(driver,tmp_path)
    with stack:
        continuation.atomic_new(recipes['python_factors'],{'development_fit_pass':False})
        result=driver.tick()
    assert result['status']=='needs_calibration_revision' and result['domain']=='python_factors'
    driver.phase.assert_not_called()




def test_failed_recovery_execution_stops_before_any_recipe_action(driver,tmp_path):
    stack,_,_,completion=tick_fixture(driver,tmp_path)
    completion.completion_status.return_value=('failed',['31149154'],[])
    with stack:result=driver.tick()
    assert result['status']=='needs_execution_review'
    driver.phase.assert_not_called()




def test_prepare_cannot_overwrite_an_inherited_scientific_pin(tmp_path):
    inherited={'files_sha256':{str(continuation.FITTER):'original-frozen-hash'},'directory_files':{},'models':{}}
    def read(path):
        if path==continuation.V6_SEAL:return inherited
        if path in (continuation.READINESS,continuation.INTEGRATION_READINESS):return {'files_sha256':{}}
        return {'jobs':[]}
    def digest(path):
        if path==continuation.RECOVERY_SEAL:return continuation.RECOVERY_SHA
        if path==continuation.RECOVERY_LEDGER:return continuation.LEDGER_SHA
        if path==continuation.READINESS:return continuation.READINESS_SHA
        if path==continuation.MATHIR_WATCHER:return continuation.MATHIR_WATCHER_SHA
        if path==continuation.V6_SEAL:return continuation.V6_SHA
        if path==continuation.INTEGRATION_READINESS:return continuation.INTEGRATION_SHA
        return 'changed-current-hash'
    with patch.object(continuation,'SEAL',tmp_path/'seal'),patch.object(continuation,'RESULT',tmp_path/'result'),\
         patch.object(continuation,'HERE',tmp_path),patch.object(continuation,'read',side_effect=read),\
         patch.object(continuation,'digest',side_effect=digest),patch.object(continuation,'verify_pins'),\
         patch.object(continuation,'authenticate_python_v6_submissions',return_value=({},{})),\
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
        candidate={'graph_coloring':'graph_v7','python_factors':'python_v6'}.get(domain,domain)
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
    with stack,patch.object(continuation,'RECOVERY_LEDGER',ledger):
        result=driver.tick()
    assert result['status']=='waiting_for_completed_recovery_receipt_visibility'
    assert result['missing_receipts']==[str(missing)]
    driver.phase.assert_not_called()
    driver.authenticate_recipes.assert_not_called()


def test_python_v6_failed_execution_stops_before_recipe_or_audit(driver,tmp_path):
    stack,_,_,completion=tick_fixture(driver,tmp_path)
    completion.completion_status.side_effect=[('pending',[],['old']),('failed',['v6id2'],[])]
    with stack:result=driver.tick()
    assert result['status']=='needs_execution_review'
    assert result['failed_python_v6_job_ids']==['v6id2']
    assert completion.scheduler_records.call_args_list[-1].args==(['v6id0','v6id1','v6id2','v6id3'],)
    driver.phase.assert_not_called()


def test_completed_python_v6_receipt_visibility_lag_waits_same_paths(driver,tmp_path):
    stack,_,_,completion=tick_fixture(driver,tmp_path)
    completion.completion_status.side_effect=[('pending',[],['old']),('complete',[],[])]
    missing=tmp_path/'v6_output.json'
    with stack,patch.object(continuation,'authenticate_python_v6_submissions',return_value=({'job_ids':['a','b','c','d'],'outputs':[str(missing)]},{})):
        result=driver.tick()
    assert result['status']=='waiting_for_python_v6_receipt_visibility'
    assert result['missing_receipts']==[str(missing)]
    driver.phase.assert_not_called()
    assert not continuation.RESULT.exists()


def test_python_v6_fit_failure_journal_stops_without_restart(driver,tmp_path):
    stack,_,_,_=tick_fixture(driver,tmp_path)
    with stack:
        continuation.atomic_new(continuation.V6_FIT_FAILURE,{'preserved':'failed fit execution'})
        result=driver.tick()
    assert result['status']=='needs_execution_review'
    assert result['python_v6_fit_failure']==str(tmp_path/'v6_fit_failure.json')
    driver.phase.assert_not_called()


def test_five_passing_recipes_wait_for_python_independent_proof_before_any_phase(driver,tmp_path):
    stack,recipes,_,_=tick_fixture(driver,tmp_path)
    driver.authenticate_recipes=Mock(side_effect=AssertionError('must wait for proof'))
    with stack:
        for path in recipes.values():continuation.atomic_new(path,{'development_fit_pass':True})
        result=driver.tick()
    assert result['status']=='waiting_for_python_v6_independent_completed_development_audit'
    driver.phase.assert_not_called()
    driver.authenticate_recipes.assert_not_called()


def test_failed_python_independent_proof_prevents_mapping_and_dataset(driver,tmp_path):
    stack,recipes,_,completion=tick_fixture(driver,tmp_path)
    validator=SimpleNamespace(validate_completed_development_audit=Mock(return_value={'status':'failed_development','files_sha256':{}}))
    driver.phase=continuation.Driver.phase.__get__(driver)
    driver.authenticate_recipes=Mock(side_effect=AssertionError('proof must pass first'))
    with stack,patch.object(continuation,'module',side_effect=lambda path,name:validator if path==continuation.V6_AUDITOR else completion):
        for path in recipes.values():continuation.atomic_new(path,{'development_fit_pass':True})
        continuation.atomic_new(continuation.V6_AUDIT,{'finished':True})
        with pytest.raises(ValueError,match='independent development audit did not pass'):driver.tick()
    driver.authenticate_recipes.assert_not_called()
    assert not (tmp_path/'accepted_recipes_intent.json').exists()
    assert continuation.read(tmp_path/'python_v6_development_attestation_result.json')['status']=='failed'


def test_current_scope_has_no_pantry_fit_or_legacy_namespace():
    configuration=continuation.fixed_configuration()
    assert set(configuration['external_fit_owners'])=={'python_factors'}
    assert 'pantry_receipts' not in configuration
    assert 'pantry_fit' not in continuation.PHASES
    assert continuation.HERE.name=='continuation_python_v6'
    assert continuation.MAPPING.parent==continuation.HERE
    assert continuation.CONFIRMATION.name=='confirmation_python_v6'
    assert not hasattr(continuation.Driver,'pantry_fit')


@pytest.fixture
def development_submission_chain(tmp_path,monkeypatch):
    campaign=tmp_path/'v6';campaign.mkdir()
    launcher_path=campaign/'launch.py';launcher_path.write_text('# sealed launcher')
    seal=campaign/'implementation_seal.json'
    continuation.atomic_new(seal,{'files_sha256':{str(launcher_path):continuation.digest(launcher_path)}})
    expected=continuation.digest(seal)
    plan_path=campaign/'protocol.json'
    jobs=[]
    for index in range(4):
        task=campaign/f'task_{index}.json';task.write_text('{}')
        jobs.append({'name':f'3b_python_v6_d{index}','tasks':str(task),'output':str(campaign/f'output{index}.json')})
    continuation.atomic_new(plan_path,{'jobs':jobs})
    claim=campaign/'development_execution_claim.json'
    continuation.atomic_new(claim,{'jobs':4,'candidate_revision':'python_v6','seal_sha256':expected,'protocol_sha256':continuation.digest(plan_path)})
    launcher=SimpleNamespace(authenticate_saved_seal=Mock(),check_static=Mock(),command_for=lambda index,job,sha:['sbatch',str(index),job['name'],sha])
    for index,job in enumerate(jobs):
        command=launcher.command_for(index,job,expected)
        continuation.atomic_new(campaign/f'submission_{index:02d}_intent.json',{'command':command,'cell':job['name'],'task_sha256':continuation.digest(job['tasks']),'seal_sha256':expected})
        continuation.atomic_new(campaign/f'submission_{index:02d}_result.json',{'command':command,'cell':job['name'],'returncode':0,'stdout':str(4000+index)})
    recovery=tmp_path/'recovery.json';continuation.atomic_new(recovery,{'jobs':[{'old_job_id':'3000'}]})
    for name,value in [('V6_SEAL',seal),('V6_SHA',expected),('V6_LAUNCHER',launcher_path),('V6_CLAIM',claim),('RECOVERY_LEDGER',recovery)]:monkeypatch.setattr(continuation,name,value)
    monkeypatch.setattr(continuation,'module',lambda *args:launcher)
    return campaign,launcher


def test_four_development_submission_chains_bind_actual_ids_and_files(development_submission_chain):
    campaign,launcher=development_submission_chain
    ledger=campaign/'development_jobs.json';ledger.write_text('{"fixed":true}')
    result,files=continuation.authenticate_python_v6_submissions()
    assert result['job_ids']==['4000','4001','4002','4003']
    assert len(result['outputs'])==4
    assert len(files)==11 and str(ledger) in files
    launcher.authenticate_saved_seal.assert_called_once_with(continuation.V6_SHA)
    launcher.check_static.assert_called_once()


@pytest.mark.parametrize('mutation', ['claim_revision','claim_seal','failed_result','ambiguous_id','result_command','intent_task','duplicate_id','old_id','missing_result','launcher_bytes'])
def test_changed_or_ambiguous_python_submission_chain_fails_closed(development_submission_chain,mutation):
    campaign,_=development_submission_chain
    intent=campaign/'submission_00_intent.json';result=campaign/'submission_00_result.json'
    target=result
    if mutation.startswith('claim_'):target=continuation.V6_CLAIM
    elif mutation=='intent_task':target=intent
    value=continuation.read(target)
    if mutation=='claim_revision':value['candidate_revision']='python_v5'
    elif mutation=='claim_seal':value['seal_sha256']='changed'
    elif mutation=='failed_result':value['returncode']=1
    elif mutation=='ambiguous_id':value['stdout']='4000\n4001'
    elif mutation=='result_command':value['command']=['sbatch','different']
    elif mutation=='intent_task':value['task_sha256']='changed'
    elif mutation=='duplicate_id':value['stdout']='4001'
    elif mutation=='old_id':value['stdout']=continuation.NEW_IDS[0]
    elif mutation=='missing_result':result.unlink()
    else:continuation.V6_LAUNCHER.write_text('# changed')
    if mutation not in ('missing_result','launcher_bytes'):target.write_text(json.dumps(value))
    with pytest.raises((ValueError,FileNotFoundError)):
        continuation.authenticate_python_v6_submissions()


def test_new_confirmation_namespace_authenticates_all_ten_and_rejects_legacy_prefix(tmp_path,monkeypatch):
    runtime_source=Path(__file__).with_name('test_continue_modebench_level3_confirmation_runtime.py')
    spec=importlib.util.spec_from_file_location('python_v6_runtime_fixture',runtime_source)
    runtime_fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime_fixture)
    # Reuse the frozen ten-worker evidence fixture, targeting this new driver.
    monkeypatch.setattr(runtime_fixture,'driver',continuation)
    execution=runtime_fixture.execution.__wrapped__(tmp_path,monkeypatch)
    for cell in execution.cells:
        for runtime in (cell.claim['runtime'],cell.event['runtime']):
            runtime['scheduler']['JobName']='mb-l3-v2-confirm-pyv6-'+cell.job['name']
        runtime_fixture.write(cell.claim_path,cell.claim)
        cell.outpath.write_text(json.dumps(cell.event)+'\n'+cell.engine)
    proof,files=continuation.authenticate_confirmation_execution(execution.plan,execution.ids,execution.accounts,'fixed-seal')
    assert proof['all_ten_workers_authenticated'] is True and len(files)==30
    cell=execution.cells[9]
    cell.claim['runtime']['scheduler']['JobName']='mb-l3-v2-confirm-'+cell.job['name']
    runtime_fixture.write(cell.claim_path,cell.claim)
    with pytest.raises(ValueError,match='actual scheduler identity differs'):
        continuation.authenticate_confirmation_execution(execution.plan,execution.ids,execution.accounts,'fixed-seal')
