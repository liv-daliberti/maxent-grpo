"""Synthetic continuation ownership, completion and fixed-target phase guards."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'ops/exp_scaling/continue_modebench_level3_v3.py'
spec = importlib.util.spec_from_file_location('v3_continuation_test', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
PIN = 'a' * 64
REG = 'b' * 64
SCIENCE = 'c' * 64


@pytest.fixture
def env(tmp_path, monkeypatch):
    campaign = tmp_path / 'campaign'
    here = campaign / 'continuation'
    here.mkdir(parents=True)
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    for key, value in {'CAMPAIGN': campaign, 'REGISTRATION': campaign / 'registration.json',
                       'RESULTS': tmp_path / 'results', 'DATASET': tmp_path / 'dataset'}.items():
        monkeypatch.setattr(m.common, key, value)
    for key, value in {'HERE': here, 'SEAL': here / 'seal.json', 'RESULT': here / 'result.json',
                       'DEV_SEAL': campaign / 'development/seal.json',
                       'DEV_AUDIT': campaign / 'development/completed_development_audit.json',
                       'CONFIRMATION': campaign / 'confirmation',
                       'REPORT': campaign / 'confirmation/confirmation_report.json'}.items():
        monkeypatch.setattr(m, key, value)
    monkeypatch.setattr(m, 'RECIPES', {d: campaign / 'recipes' / (d + '.json') for d in m.common.REVISED})
    driver = m.Driver(PIN)
    driver.verify = Mock(return_value={'registration_sha256': REG, 'scientific_seal_sha256': SCIENCE})
    driver.command_binding = lambda: driver.verify()
    return SimpleNamespace(root=tmp_path, driver=driver)


def action(evidence=None, files=None, trees=None):
    return Mock(return_value=(evidence or {'value': 1}, files or {}, trees or {}))


def advance_to(driver, target):
    for name in m.PHASES:
        if name == target:
            return
        driver.phase(name, {'name': name}, action())


def test_contract_preserves_fixed_targets_retained_domains_and_exact_counts():
    config = m.configuration()
    assert config['development_jobs'] == 8 and config['development_original_grader_attempts'] == 39424
    assert config['confirmation_jobs'] == 2 and config['confirmation_original_grader_attempts'] == 8192
    assert config['retained_domains'] == ['countdown', 'mathir', 'pantry']
    assert config['revised_domains'] == ['graph_coloring', 'python_factors']
    assert config['historical_level1_role'] == 'frozen_benchmark_reference'
    assert config['all_five_fresh_same_round'] is config['treatment_training_started'] is False
    assert m.PHASES == ('development_completed', 'fit_graph_coloring', 'fit_python_factors',
                        'development_audit', 'finalize', 'confirmation_prepare', 'confirmation_submit', 'confirmation_audit')


def test_phase_reuses_only_authenticated_identical_completed_action(env):
    run = action({'done': True})
    expected = env.driver.phase(m.PHASES[0], {'jobs': ['1']}, run)
    assert env.driver.phase(m.PHASES[0], {'jobs': ['1']}, run) == expected
    run.assert_called_once()
    with pytest.raises(ValueError, match='inputs changed'):
        env.driver.phase(m.PHASES[0], {'jobs': ['2']}, run)
    run.assert_called_once()


@pytest.mark.parametrize('failure', [ValueError('failed'), KeyboardInterrupt('interrupted'), SystemExit(7)])
def test_action_failure_is_preserved_and_never_retried(env, failure):
    run = Mock(side_effect=failure)
    with pytest.raises(type(failure)):
        env.driver.phase(m.PHASES[0], {}, run)
    result = m.read(m.HERE / (m.PHASES[0] + '_result.json'))
    assert result['status'] == 'failed' and type(failure).__name__ in result['error']
    with pytest.raises(ValueError, match='failed phase'):
        env.driver.phase(m.PHASES[0], {}, run)
    run.assert_called_once()


@pytest.mark.parametrize('kind', ['intent', 'result'])
def test_interrupted_or_orphan_phase_requires_review(env, kind):
    m.atomic_new(m.HERE / (m.PHASES[0] + '_' + kind + '.json'), {})
    run = action()
    with pytest.raises(ValueError, match='ambiguous interrupted'):
        env.driver.phase(m.PHASES[0], {}, run)
    run.assert_not_called()


def test_out_of_order_phase_cannot_run(env):
    run = action()
    with pytest.raises(ValueError, match='out of order'):
        env.driver.phase('finalize', {}, run)
    run.assert_not_called()


@pytest.mark.parametrize('mutation', ['file', 'tree', 'intent', 'result'])
def test_completed_phase_evidence_remains_immutable(env, mutation):
    directory = env.root / 'frozen'
    directory.mkdir()
    item = directory / 'evidence.json'
    item.write_text('evidence')
    env.driver.phase(m.PHASES[0], {}, action(files=m.pins([item]), trees={str(directory): [str(item)]}))
    if mutation == 'file':
        item.write_text('changed')
    elif mutation == 'tree':
        (directory / 'extra').write_text('added')
    elif mutation == 'intent':
        path = m.HERE / (m.PHASES[0] + '_intent.json')
        record = m.read(path); record['inputs'] = {'changed': True}; path.write_text(json.dumps(record))
    else:
        path = m.HERE / (m.PHASES[0] + '_result.json')
        record = m.read(path); record['seal_sha256'] = REG; path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        env.driver.journal()


def test_durable_command_failure_keeps_stdout_stderr_and_ownership(env, monkeypatch):
    def failed(command, **kwargs):
        kwargs['stdout'].write('visible output')
        kwargs['stderr'].write('visible error')
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(m.subprocess, 'run', failed)
    with pytest.raises(ValueError, match='returned 7'):
        env.driver.command(m.PHASES[0], ['/sealed/python', '-B', '/sealed/action.py'])
    assert (m.HERE / (m.PHASES[0] + '.stdout')).read_text() == 'visible output'
    assert (m.HERE / (m.PHASES[0] + '.stderr')).read_text() == 'visible error'
    assert m.read(m.HERE / (m.PHASES[0] + '_command.json'))['seal_sha256'] == PIN
    with pytest.raises(ValueError):
        env.driver.command(m.PHASES[0], ['/sealed/python'])


def test_process_lock_has_one_owner(env):
    with m.coordinator_lock():
        with pytest.raises(BlockingIOError):
            with m.coordinator_lock():
                pytest.fail('two owners acquired the same lock')


@pytest.mark.parametrize('status', ['needs_execution_review', 'needs_calibration_revision',
                                  'matched_fixed_reference', 'outside_fixed_reference_tolerance'])
def test_terminal_record_never_replaced(env, status):
    result = env.driver.finish(status)
    assert result['status'] == status
    with pytest.raises(ValueError):
        env.driver.finish(status)
    assert m.read(m.RESULT) == result


@pytest.mark.parametrize('kind', ['recipe', 'fit_intent', 'audit', 'dataset', 'report', 'seal', 'prepare_intent',
                                  'confirmation_claim', 'confirmation_result', 'phase_intent'])
def test_additive_prepare_refuses_preexisting_future_work(env, kind):
    recipe = next(iter(m.RECIPES.values()))
    paths = {'recipe': recipe, 'fit_intent': recipe.with_suffix('.fit_intent.json'), 'audit': m.DEV_AUDIT,
             'dataset': m.common.DATASET, 'report': m.REPORT, 'seal': m.SEAL,
             'prepare_intent': m.HERE / 'prepare_intent.json',
             'confirmation_claim': m.CONFIRMATION / 'confirmation_execution_claim.json',
             'confirmation_result': m.CONFIRMATION / 'submission_00_result.json',
             'phase_intent': m.HERE / 'development_completed_intent.json'}
    path = paths[kind]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{}')
    with pytest.raises(ValueError):
        m.no_future_work()


def seal_fixture(env, monkeypatch):
    source = env.root / 'driver.py'; source.write_text('reviewed source')
    test = env.root / 'test_driver.py'; test.write_text('reviewed tests')
    monkeypatch.setattr(m, 'SOURCE', source)
    monkeypatch.setattr(m, 'TEST', test)
    monkeypatch.setattr(m, 'IMPLEMENTATION', (source, test))
    m.common.REGISTRATION.write_text('{}')
    m.DEV_SEAL.parent.mkdir(parents=True)
    m.DEV_SEAL.write_text('{}')
    inherited = env.root / 'old_input'; inherited.write_text('unchanged inherited input')
    files = m.pins([inherited])
    registration = {'files_sha256': files, 'directory_files': {}}
    scientific = {'files_sha256': files, 'directory_files': {}, 'models': {'05b': {}, '3b': {}},
                  'registration_sha256': REG}
    jobs = []
    submission_files = {}
    for index in range(8):
        output = m.common.RESULTS / ('development_' + str(index) + '.json')
        intent = m.DEV_SEAL.parent / ('submission_' + str(index) + '_intent.json')
        intent.write_text(json.dumps({'job_id': str(index + 1)}))
        submission_files.update(m.pins([intent]))
        jobs.append({'job_id': str(index + 1), 'job': {'output': str(output)}})
    submissions = {'jobs': jobs, 'job_ids': [str(i + 1) for i in range(8)], 'files_sha256': submission_files}
    launcher = SimpleNamespace(authenticate_saved_seal=lambda pin: deepcopy(scientific),
                               authenticate_submissions=lambda pin: deepcopy(submissions))
    monkeypatch.setattr(m.common, 'validate_registration', lambda path, pin: deepcopy(registration))
    monkeypatch.setattr(m, 'module', lambda path, name: launcher)
    monkeypatch.setattr(m, 'local_dependency_sources', lambda paths: {})
    return SimpleNamespace(source=source, test=test, inherited=inherited, scientific=scientific,
                           submissions=submissions, launcher=launcher)


def test_additive_prepare_authenticates_actual_eight_chains_and_is_action_free(env, monkeypatch):
    state = seal_fixture(env, monkeypatch)
    never = Mock(side_effect=AssertionError('scientific or scheduler action'))
    monkeypatch.setattr(m.subprocess, 'run', never)
    result = m.prepare(REG, SCIENCE)
    assert result['status'] == 'prepared_not_armed' and result['development_job_ids'] == list(map(str, range(1, 9)))
    seal = m.verify_seal(result['seal_sha256'])
    assert seal['new_candidate_outcomes_loaded'] is seal['new_actions_started'] is False
    assert str(m.SEAL) not in seal['files_sha256']
    for path, value in state.submissions['files_sha256'].items():
        assert seal['files_sha256'][path] == value
    assert not any(path.exists() for path in m.RECIPES.values())
    never.assert_not_called()


@pytest.mark.parametrize('omission', ['source', 'submission', 'inherited', 'registration', 'dev_seal'])
def test_additive_authenticator_requires_complete_source_and_actual_chain_closure(env, monkeypatch, omission):
    state = seal_fixture(env, monkeypatch)
    m.prepare(REG, SCIENCE)
    saved = m.read(m.SEAL)
    path = {'source': str(state.source), 'submission': next(iter(state.submissions['files_sha256'])),
            'inherited': str(state.inherited), 'registration': str(m.common.REGISTRATION), 'dev_seal': str(m.DEV_SEAL)}[omission]
    del saved['files_sha256'][path]
    m.SEAL.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        m.verify_seal(m.digest(m.SEAL))


def test_missing_future_source_stops_preparation_before_registration_reads(env, monkeypatch):
    monkeypatch.setattr(m, 'IMPLEMENTATION', (env.root / 'missing_future.py',))
    never = Mock(side_effect=AssertionError('premature authentication'))
    monkeypatch.setattr(m.common, 'validate_registration', never)
    with pytest.raises(ValueError, match='future implementation is not ready'):
        m.prepare(REG, SCIENCE)
    never.assert_not_called()


@pytest.mark.parametrize('state', ['RUNNING', 'COMPLETED', 'FAILED'])
def test_actual_completion_waits_for_same_receipt_paths_and_preserves_execution_failure(env, monkeypatch, state):
    output = env.root / 'receipt.json'
    submissions = {'job_ids': ['12'], 'jobs': [{'job': {'output': str(output)}}]}
    accounts = {'12': {'State': state}}
    helper = SimpleNamespace(scheduler_records=lambda ids: accounts,
        completion_status=lambda records: ('failed', ['12'], []) if state == 'FAILED' else
            ('pending', [], ['12']) if state == 'RUNNING' else ('complete', [], []))
    monkeypatch.setattr(m, 'module', lambda *args: helper)
    result, _ = env.driver.observe('development', submissions)
    if state == 'FAILED':
        assert result['status'] == 'needs_execution_review' and m.RESULT.is_file()
    elif state == 'RUNNING':
        assert result['status'] == 'waiting_for_development'
    else:
        assert result['status'] == 'waiting_for_development_receipt_visibility'
        output.write_text('{}')
        assert env.driver.observe('development', submissions)[0] is None


@pytest.mark.parametrize('kind', ['recipe', 'intent'])
def test_fit_cannot_adopt_or_recompute_existing_recipe(env, kind):
    domain = m.common.REVISED[0]
    path = m.RECIPES[domain]
    path = path if kind == 'recipe' else path.with_suffix('.fit_intent.json')
    path.parent.mkdir(parents=True); path.write_text('{}')
    env.driver.command = Mock(side_effect=AssertionError('duplicate fit'))
    with pytest.raises(ValueError, match='another fit'):
        env.driver.fit(domain, REG)
    env.driver.command.assert_not_called()


def test_default_cli_reads_no_outcomes_and_takes_no_actions(env, monkeypatch):
    never = Mock(side_effect=AssertionError('default invocation took action'))
    monkeypatch.setattr(m, 'read', never)
    monkeypatch.setattr(m.subprocess, 'run', never)
    monkeypatch.setattr(m, 'prepare', never)
    assert m.main([]) == 0
    never.assert_not_called()


@pytest.mark.parametrize('argv', [['--prepare'], ['--prepare', '--registration-sha256', REG],
    ['--advance'], ['--watch'], ['--prepare', '--registration-sha256', REG, '--scientific-seal-sha256', SCIENCE, '--seal-sha256', PIN],
    ['--advance', '--seal-sha256', PIN, '--registration-sha256', REG]])
def test_action_cli_requires_exact_external_hash_roles(env, monkeypatch, argv):
    never = Mock(side_effect=AssertionError('unauthorized action'))
    monkeypatch.setattr(m, 'prepare', never)
    monkeypatch.setattr(m.Driver, 'tick', never)
    with pytest.raises(ValueError):
        m.main(argv)
    never.assert_not_called()


def pipeline_fixture(env, monkeypatch, *, development_pass=True, confirmation_match=True, conf_visible=True):
    jobs = []
    for index in range(8):
        path = env.root / f'dev_receipt_{index}.json'; path.write_text('{}')
        jobs.append({'job_id': str(index + 1), 'job': {'output': str(path)}})
    development = {'jobs': jobs, 'job_ids': [j['job_id'] for j in jobs]}
    seal = {'registration_sha256': REG, 'scientific_seal_sha256': SCIENCE,
            'development_submissions': development}
    env.driver.verify.return_value = seal
    conf_jobs = [{'job_id': str(index + 10), 'job': {'output': str(env.root / f'conf_receipt_{index}.json')}}
                 for index in range(2)]
    conf = {'jobs': conf_jobs, 'job_ids': [j['job_id'] for j in conf_jobs],
            'seal_sha256': 'd' * 64, 'registration_sha256': REG, 'files_sha256': {}}
    calls = []
    def command(name, argv):
        calls.append((name, list(map(str, argv))))
        m.atomic_new(m.HERE / (name + '_command.json'), {'phase': name, 'seal_sha256': PIN,
                     'cwd': str(m.ROOT), 'command': list(map(str, argv))})
        (m.HERE / (name + '.stdout')).write_text('synthetic durable output')
        (m.HERE / (name + '.stderr')).write_text('')
        if name.startswith('fit_'):
            domain = name[4:]
            path = m.RECIPES[domain]
            m.atomic_new(path, {'development_fit_pass': development_pass, 'domain': domain})
            m.atomic_new(path.with_suffix('.fit_intent.json'), {'domain': domain})
        elif name == 'development_audit':
            m.atomic_new(m.DEV_AUDIT, {'schema': 'dev_schema', 'jobs': 8, 'rows': 1232, 'attempts': 39424,
                'status': 'passed_development' if development_pass else 'needs_calibration_revision',
                'all_revised_development_gates_pass': development_pass,
                'all_attempts_regraded_with_original_grader': True,
                'every_attempt_including_failures_and_full_canonical_keys_compared': True,
                'scientific_seal_sha256': SCIENCE, 'registration_sha256': REG,
                'files_sha256': m.pins(m.RECIPES.values()), 'directory_files': {}})
        elif name == 'finalize':
            m.common.DATASET.mkdir()
            (m.common.DATASET / 'identity.json').write_text('{}')
        elif name == 'confirmation_prepare':
            m.atomic_new(m.CONFIRMATION / 'seal.json', {'stage': 'confirmation'})
        elif name == 'confirmation_submit':
            m.atomic_new(m.CONFIRMATION / 'confirmation_execution_claim.json', {'jobs': 2})
            conf['files_sha256'] = m.pins([m.CONFIRMATION / 'confirmation_execution_claim.json'])
            if conf_visible:
                for item in conf_jobs:
                    Path(item['job']['output']).write_text('{}')
        elif name == 'confirmation_audit':
            m.atomic_new(m.REPORT, {'status': 'matched_fixed_reference' if confirmation_match else 'outside_fixed_reference_tolerance',
                                   'confirmation_match_verified': confirmation_match})
        else:
            pytest.fail('unexpected command: ' + name)
    env.driver.command = command
    completion = SimpleNamespace(scheduler_records=lambda ids: {i: {'State': 'COMPLETED', 'ExitCode': '0:0'} for i in ids},
                                 completion_status=lambda accounts: ('complete', [], []))
    def proof(path, status):
        result = {'status': status, 'path': str(path), 'sha256': m.digest(path),
                  'files_sha256': m.pins([path]), 'directory_files': {}}
        if path == m.REPORT:
            result['confirmation_match_verified'] = confirmation_match
        return result
    auditor = SimpleNamespace(DEV_SCHEMA='dev_schema', validate_recipe_gates=lambda recipe: None,
        validate_completed_development_audit=lambda path: proof(path, 'passed_development'),
        validate_confirmation_report=lambda path: proof(path, 'matched_fixed_reference' if confirmation_match else 'outside_fixed_reference_tolerance'))
    def dataset(registration_sha256):
        path = m.common.DATASET / 'identity.json'
        return {'identity_metadata': {'rows': 3200, 'retained_domains': list(m.common.RETAINED)},
                'files_sha256': m.pins([path]), 'directory_files': {str(m.common.DATASET): [str(path)]}}
    finalizer = SimpleNamespace(authenticate_dataset=dataset)
    launcher = SimpleNamespace(SEAL=m.CONFIRMATION / 'seal.json', HERE=m.CONFIRMATION,
        CLAIM=m.CONFIRMATION / 'confirmation_execution_claim.json',
        authenticate_saved_seal=lambda pin: {'files_sha256': {}, 'directory_files': {}},
        authenticate_submissions=lambda pin: deepcopy(conf))
    def load(path, name):
        return {m.COMPLETION: completion, m.AUDITOR: auditor, m.FINALIZER: finalizer, m.CONF_LAUNCHER: launcher}[path]
    monkeypatch.setattr(m, 'module', load)
    return SimpleNamespace(calls=calls, conf_jobs=conf_jobs, seal=seal)


def test_complete_pipeline_fits_exactly_two_and_submits_exactly_two_without_baseline(env, monkeypatch):
    state = pipeline_fixture(env, monkeypatch)
    result = env.driver.tick()
    assert result['status'] == 'matched_fixed_reference' and result['confirmation_match_verified'] is True
    names = [name for name, argv in state.calls]
    assert names == list(m.PHASES[1:])
    fits = [argv for name, argv in state.calls if name.startswith('fit_')]
    assert len(fits) == 2 and all('--scores' not in argv and '--baseline' not in argv for argv in fits)
    preparation = next(argv for name, argv in state.calls if name == 'confirmation_prepare')
    assert preparation[preparation.index('--execution-seal-sha256') + 1] == PIN
    assert env.driver.tick() == result
    assert [name for name, argv in state.calls] == names


def test_failed_development_replays_all_completed_evidence_then_stops_without_finalizing(env, monkeypatch):
    state = pipeline_fixture(env, monkeypatch, development_pass=False)
    result = env.driver.tick()
    assert result['status'] == 'needs_calibration_revision'
    assert [name for name, argv in state.calls] == ['fit_graph_coloring', 'fit_python_factors', 'development_audit']
    assert all(m.read(path)['development_fit_pass'] is False for path in m.RECIPES.values())
    assert not m.common.DATASET.exists() and not m.CONFIRMATION.exists()
    assert env.driver.tick() == result
    assert len(state.calls) == 3


def test_failed_confirmation_gate_remains_a_complete_numerical_outcome(env, monkeypatch):
    state = pipeline_fixture(env, monkeypatch, confirmation_match=False)
    result = env.driver.tick()
    assert result['status'] == 'outside_fixed_reference_tolerance'
    assert result['confirmation_match_verified'] is False
    assert m.REPORT.is_file() and len(state.calls) == 7
    assert env.driver.tick() == result and len(state.calls) == 7


def test_completed_confirmation_visibility_wait_does_not_resubmit_or_regrade_early(env, monkeypatch):
    state = pipeline_fixture(env, monkeypatch, conf_visible=False)
    result = env.driver.tick()
    assert result['status'] == 'waiting_for_confirmation_receipt_visibility'
    assert [name for name, argv in state.calls][-1] == 'confirmation_submit'
    assert not m.REPORT.exists()
    for item in state.conf_jobs:
        Path(item['job']['output']).write_text('{}')
    assert env.driver.tick()['status'] == 'matched_fixed_reference'
    assert [name for name, argv in state.calls] == list(m.PHASES[1:])


@pytest.mark.parametrize('mutation', ['argv', 'cwd', 'phase', 'seal', 'missing_log_pin'])
def test_saved_child_command_semantics_cannot_change_even_if_phase_repins_it(env, monkeypatch, mutation):
    pipeline_fixture(env, monkeypatch)
    assert env.driver.tick()['status'] == 'matched_fixed_reference'
    name = 'fit_graph_coloring'
    command_path = m.HERE / (name + '_command.json')
    result_path = m.HERE / (name + '_result.json')
    command = m.read(command_path)
    if mutation == 'argv':
        command['command'].extend(['--scores', 'alternate1', 'alternate2', 'alternate3', 'alternate4'])
    elif mutation == 'cwd':
        command['cwd'] = '/unregistered/working/directory'
    elif mutation == 'phase':
        command['phase'] = 'fit_python_factors'
    elif mutation == 'seal':
        command['seal_sha256'] = REG
    result = m.read(result_path)
    if mutation == 'missing_log_pin':
        del result['files_sha256'][str(m.HERE / (name + '.stderr'))]
    else:
        command_path.write_text(json.dumps(command))
        result['files_sha256'][str(command_path)] = m.digest(command_path)
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='completed child command differs|omits command or durable logs'):
        env.driver.journal()


def test_every_registered_child_uses_explicit_no_bytecode_python(env, monkeypatch):
    state = pipeline_fixture(env, monkeypatch)
    assert env.driver.tick()['status'] == 'matched_fixed_reference'
    assert all(argv[:2] == [str(m.PYTHON), '-B'] for name, argv in state.calls)
    assert m.sys.dont_write_bytecode is True
