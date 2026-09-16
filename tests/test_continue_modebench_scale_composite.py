"""Composite release transitions, tested without model calls or scheduler writes.

The fixture uses real immutable manifests and release publication. Only costly
scientific replay/generation and submitted array execution are substituted.
"""
from datetime import datetime, timedelta
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops/exp_scaling'))
import continue_modebench_scale_composite as controller


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')
    return path


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    parent, release, artifacts, parent_artifacts = [tmp_path / name for name in
                                                   ('parent', 'release', 'artifacts', 'parent_artifacts')]
    events = []
    state = SimpleNamespace(parent=parent, release=release, artifacts=artifacts,
                            parent_artifacts=parent_artifacts, events=events,
                            missing=set(), failed={('level4', 'python_factors'), ('level5', 'mathir')},
                            revision_failed=set(), audit_failed=set(), replay_counts={},
                            complete_dev=True, complete_confirmation=True, bindings={}, roots={})
    protocol = {'models': {label: {'path': str(tmp_path / label)} for label in controller.LEVELS.values()},
                'targets': {domain: {'pass1': .2, 'pass8': .6} for domain in controller.DOMAINS},
                'histograms': {domain: {'fixture': domain} for domain in controller.DOMAINS},
                'tolerances': {'pass1': .04, 'pass8': .08}, 'files_sha256': {}}
    write(parent / 'protocol.json', protocol)
    state.protocol = protocol
    recipes = {}
    revision_protocols = {}
    for level in controller.LEVELS:
        state.bindings[level] = {}
        for domain in controller.DOMAINS:
            key = level, domain
            recipe = {'development_fit_pass': key not in state.failed, 'input_sha256': {}}
            recipes[key] = recipe
            paths = controller.domain_paths(parent, level, domain)
            write(paths['recipe'], recipe)
            for receipt in controller.development_receipts(parent, level, domain):
                write(receipt, {'fixture_original_development': True})
            if key not in state.failed:
                continue
            root = tmp_path / ('revision_' + level + '_' + domain)
            state.roots[key] = root
            state.bindings[level][domain] = str(root)
            value = {**protocol, 'level': level, 'domain': domain, 'parent_root': str(parent),
                     'parent_protocol_sha256': controller.file_sha(parent / 'protocol.json'),
                     'parent_recipe_sha256': controller.file_sha(paths['recipe'])}
            write(root / 'protocol.json', value)
            write(root / 'protocol.sha256.json', {'sha256': controller.file_sha(root / 'protocol.json')})
            write(controller.domain_paths(root, level, domain)['recipe'], {'development_fit_pass': True, 'input_sha256': {}})
            revision_protocols[root] = value
    state.recipes = recipes
    write(artifacts / 'revision_bindings.json',
          {'schema': 'modebench_scale_composite_revision_bindings_v1', 'revisions': state.bindings})
    monkeypatch.setattr(controller, 'PARENT_ARTIFACTS', parent_artifacts)
    monkeypatch.setattr(controller.original, 'authenticate', lambda path: controller.read(path))
    monkeypatch.setattr(controller.revision, 'authenticate', lambda root: controller.read(Path(root) / 'protocol.json'))

    def parent_recipe(parent_root, level, domain, value, *, advance):
        events.append(('parent_fit', level, domain, advance))
        return None if (level, domain) in state.missing else recipes[level, domain]

    monkeypatch.setattr(controller.previous, 'recipe', parent_recipe)
    monkeypatch.setattr(controller.original_fit, 'fit_domain', lambda root, level, domain, publish: recipes[level, domain])
    real_bind = controller.bind_level

    def bind(*args):
        events.append(('bind', args[2], dict(args[3])))
        return real_bind(*args)

    monkeypatch.setattr(controller, 'bind_level', bind)

    def freeze(root, level, domain, kind):
        events.append(('freeze_' + kind, level, domain))
        directory = controller.domain_paths(root, level, domain)['dataset']
        directory.mkdir(parents=True)
        for split, (count, subset) in controller.SPLITS.items():
            write(directory / split / 'fixture.json', {'split': split, 'rows': count})
        write(directory / 'identity.json', {'fixture': True})

    monkeypatch.setattr(controller.original, 'freeze_dataset',
                        lambda root, level, domain: freeze(root, level, domain, 'carry'))

    def revision_paths(root):
        p = controller.read(Path(root) / 'protocol.json')
        return controller.domain_paths(root, p['level'], p['domain'])

    monkeypatch.setattr(controller.revision, 'paths', revision_paths)

    def pools(root):
        p = controller.read(Path(root) / 'protocol.json')
        events.append(('revision_pools', p['level'], p['domain']))
        revision_paths(root)['pools'].mkdir(parents=True)

    monkeypatch.setattr(controller.revision, 'materialize_pools', pools)

    def revision_freeze(root):
        p = controller.read(Path(root) / 'protocol.json')
        freeze(root, p['level'], p['domain'], 'revision')

    monkeypatch.setattr(controller.revision, 'freeze_dataset', revision_freeze)

    def identity(root, level, domain, kind):
        assert controller.domain_paths(root, level, domain)['dataset'].is_dir()
        return {'splits': {split: {'rows': count, 'subset': subset} for split, (count, subset) in controller.SPLITS.items()}}

    monkeypatch.setattr(controller, 'frozen_identity', identity)
    monkeypatch.setattr(controller, 'verify_dataset', lambda root, level, domain: {'files_sha256': {}})
    monkeypatch.setattr(controller, 'validate_revision_pool_exclusions',
                        lambda root, level, domain, carried: {}, raising=False)

    def cell(root, level, domain, phase, kind):
        return {'id': level + '_' + domain + '_' + phase, 'level': level, 'domain': domain,
                'phase': phase, 'source_kind': kind, 'source_root': str(root), 'tasks': []}, {}

    monkeypatch.setattr(controller, 'carried_inputs',
                        lambda root, level, domain: cell(root, level, domain, 'eval', 'campaign_v1'))

    def revision_inputs(root, phase):
        p = controller.read(Path(root) / 'protocol.json')
        return cell(root, p['level'], p['domain'], phase, 'domain_revision_v1')

    monkeypatch.setattr(controller.revision, 'launch_inputs', revision_inputs)

    def stage(parent_root, artifacts_root, name, cells, pins, *, advance, required_pins=None):
        events.append(('stage', name, tuple(cell['id'] for cell in cells)))
        path = write(artifacts_root / name / 'plan.json', {'fixture_stage': name})
        if name == 'revision_development' and state.complete_dev:
            for item in cells:
                for receipt in controller.development_receipts(item['source_root'], item['level'], item['domain']):
                    write(receipt, {'fixture_revision_development': True})
        if name == 'confirmation' and state.complete_confirmation:
            for item in cells:
                write(controller.domain_paths(item['source_root'], item['level'], item['domain'])['receipt'],
                      {'fixture_heldout_receipt': True})
        return path, None

    state.fake_stage = stage
    monkeypatch.setattr(controller, 'submitted_stage', stage)
    monkeypatch.setattr(controller, 'observe_array', lambda path: None)

    def revision_fit(root, *, advance):
        p = controller.read(Path(root) / 'protocol.json')
        key = p['level'], p['domain']
        events.append(('revision_fit', *key))
        return {'development_fit_pass': key not in state.revision_failed}

    monkeypatch.setattr(controller, 'saved_revision_recipe', revision_fit)

    def audit(parent_root, level, domain, source):
        events.append(('audit', level, domain))
        value = {'difficulty_matched': (level, domain) not in state.audit_failed,
                 'original_grader_replayed_attempts': state.replay_counts.get((level, domain), 4096)}
        write(controller.domain_paths(source['source_root'], level, domain)['audit'], value)
        return value

    monkeypatch.setattr(controller, 'audit_source', audit)
    state.run = lambda advance=True: controller._sweep(parent, release, artifacts, advance=advance)
    return state


def test_incomplete_any_original_domain_blocks_every_later_stage(campaign):
    missing = campaign.parent / 'level5/results/development/pantry/difficulty_3.json'
    missing.unlink()
    result = campaign.run()
    assert result == {'status': 'waiting_original_development', 'missing': ['level5_pantry']}
    assert {event[0] for event in campaign.events} == {'parent_fit'}
    assert not campaign.release.exists()


def test_original_fits_all_ten_before_binding_and_all_carries_freeze_before_revision_pools(campaign):
    campaign.complete_dev = False
    result = campaign.run()
    assert result['status'] == 'waiting_revision_development'
    first_bind = next(index for index, event in enumerate(campaign.events) if event[0] == 'bind')
    assert {(event[1], event[2]) for event in campaign.events[:first_bind]} == {
        (level, domain) for level in controller.LEVELS for domain in controller.DOMAINS}
    freezes = [index for index, event in enumerate(campaign.events) if event[0] == 'freeze_carry']
    pools = [index for index, event in enumerate(campaign.events) if event[0] == 'revision_pools']
    assert len(freezes) == 8 and len(pools) == 2 and max(freezes) < min(pools)
    assert [event[1] for event in campaign.events if event[0] == 'stage'] == ['revision_development']
    for level in controller.LEVELS:
        manifest = controller.read(campaign.release / level / 'source_manifest.json')
        for domain, source in manifest['sources'].items():
            failed = (level, domain) in campaign.failed
            assert source['source_kind'] == ('domain_revision_v1' if failed else 'campaign_v1')
            assert Path(source['source_root']) == (campaign.roots[level, domain] if failed else campaign.parent)


def test_unmapped_failed_domain_waits_without_automatic_revision_selection(campaign):
    campaign.bindings['level5'] = {}
    write(campaign.artifacts / 'revision_bindings.json',
          {'schema': 'modebench_scale_composite_revision_bindings_v1', 'revisions': campaign.bindings})
    result = campaign.run()
    assert result == {'status': 'needs_source_manifest', 'failed_domains': {'level5': ['mathir']}}
    assert not any(event[0].startswith('freeze') or event[0] in {'revision_pools', 'stage'} for event in campaign.events)
    assert not (campaign.release / 'level5/source_manifest.json').exists()


def test_bind_requires_all_ten_original_recipes(campaign):
    controller.domain_paths(campaign.parent, 'level5', 'pantry')['recipe'].unlink()
    with pytest.raises(ValueError, match='all ten original development fits'):
        controller.bind_level(campaign.parent, campaign.release, 'level4', campaign.bindings['level4'])


@pytest.mark.parametrize('mapping', [{}, {'python_factors': 'revision', 'pantry': 'revision'}])
def test_binding_requires_exactly_failed_original_domains(campaign, mapping):
    with pytest.raises(ValueError, match='cover exactly the failed'):
        controller.bind_level(campaign.parent, campaign.release, 'level4', mapping)


@pytest.mark.parametrize('field', ['level', 'domain', 'parent_protocol_sha256', 'parent_recipe_sha256', 'models', 'tolerances'])
def test_revision_must_preserve_registered_parent_domain(campaign, field):
    root = campaign.roots['level4', 'python_factors']
    value = controller.read(root / 'protocol.json')
    value[field] = 'altered'
    write(root / 'protocol.json', value)
    with pytest.raises(ValueError, match='differs from selected parent domain'):
        controller.bind_level(campaign.parent, campaign.release, 'level4', campaign.bindings['level4'])


@pytest.mark.parametrize('observed', ['receipt', 'batches'])
def test_binding_rejects_observed_heldout_outcomes(campaign, observed):
    receipt = controller.domain_paths(campaign.parent, 'level4', 'countdown')['receipt']
    write(receipt if observed == 'receipt' else Path(str(receipt) + '.batches'), {})
    with pytest.raises(ValueError, match='unobserved heldout'):
        controller.bind_level(campaign.parent, campaign.release, 'level4', campaign.bindings['level4'])


@pytest.mark.parametrize('changed', ['manifest', 'parent_protocol', 'recipe', 'revision_protocol'])
def test_source_manifest_rejects_changed_immutable_inputs(campaign, changed):
    controller.bind_level(campaign.parent, campaign.release, 'level4', campaign.bindings['level4'])
    paths = {'manifest': campaign.release / 'level4/source_manifest.json',
             'parent_protocol': campaign.parent / 'protocol.json',
             'recipe': controller.domain_paths(campaign.parent, 'level4', 'countdown')['recipe'],
             'revision_protocol': campaign.roots['level4', 'python_factors'] / 'protocol.json'}
    path = paths[changed]
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='changed'):
        controller.source_manifest(campaign.parent, campaign.release, 'level4')


def test_failed_revision_fit_stops_before_freeze_confirmation_and_publication(campaign):
    campaign.revision_failed.add(('level5', 'mathir'))
    result = campaign.run()
    assert result == {'status': 'needs_new_development_revision', 'failed_domains': ['level5_mathir']}
    assert not any(event[0] == 'freeze_revision' or event[:2] == ('stage', 'confirmation') for event in campaign.events)
    assert not list(campaign.release.rglob('admission.json'))


@pytest.mark.parametrize('kind', ['development', 'freeze'])
def test_partial_generation_requires_reconciliation_and_never_restarts(campaign, kind):
    root = campaign.roots['level4', 'python_factors']
    write(root / 'exclusions' / (kind + '.json'), {'started': True})
    result = campaign.run()
    assert result == {'status': 'generation_needs_reconciliation', 'source_root': str(root)}
    blocked = 'revision_pools' if kind == 'development' else 'freeze_revision'
    assert not any(event[0] == blocked for event in campaign.events)
    assert not list(campaign.release.rglob('admission.json'))


def test_heldout_failure_blocks_affected_level_and_root_admission(campaign):
    campaign.audit_failed.add(('level5', 'pantry'))
    result = campaign.run()
    assert result['status'] == 'heldout_confirmation_failed'
    assert result['levels']['level5']['failed_domains'] == ['pantry']
    assert not (campaign.release / 'level5/admission.json').exists()
    assert not (campaign.release / 'admission.json').exists()
    assert (campaign.release / 'level4/admission.json').is_file()


@pytest.mark.parametrize('attempts', [0, 4095, 4097])
def test_every_domain_requires_all_4096_original_grader_replays(campaign, attempts):
    campaign.replay_counts['level4', 'pantry'] = attempts
    with pytest.raises(ValueError, match='all five regraded heldout domains'):
        campaign.run()
    assert not list(campaign.release.rglob('admission.json'))


def test_complete_release_has_both_levels_all_five_domains_and_train_test_pointers(campaign):
    result = campaign.run()
    assert result['status'] == 'admitted'
    assert [event[1] for event in campaign.events if event[0] == 'stage'] == ['revision_development', 'confirmation']
    assert len([event for event in campaign.events if event[0] == 'audit']) == 10
    root = controller.read(campaign.release / 'admission.json')
    assert set(root['levels']) == set(controller.LEVELS)
    for level in controller.LEVELS:
        admission = controller.read(campaign.release / level / 'admission.json')
        assert set(admission['domains']) == set(controller.DOMAINS)
        assert admission['test_split'] == 'eval' and admission['treatment_training_started'] is False
        for domain, source in admission['domains'].items():
            assert source['splits']['train']['rows'] == 384
            assert source['splits']['eval']['rows'] == 128
            link = campaign.release / level / 'dataset' / domain
            assert link.is_symlink() and link.resolve() == controller.domain_paths(source['source_root'], level, domain)['dataset']
    first = (campaign.release / 'admission.json').read_bytes()
    campaign.events.clear()
    assert campaign.run()['status'] == 'admitted'
    assert not any(event[0].startswith('freeze') or event[0] == 'revision_pools' for event in campaign.events)
    assert (campaign.release / 'admission.json').read_bytes() == first


def test_read_only_reports_original_fit_without_creating_manifests(campaign):
    campaign.missing.add(('level4', 'countdown'))
    result = campaign.run(advance=False)
    assert result['status'] == 'ready_original_fit'
    assert not campaign.release.exists()


def test_activation_rejects_live_predecessor(tmp_path, monkeypatch):
    parent, release, artifacts = [tmp_path / name for name in ('parent', 'release', 'artifacts')]
    source = write(tmp_path / 'source.json', {'sealed': True})
    monkeypatch.setattr(controller, 'activation_sources', lambda *args: [source])
    monkeypatch.setattr(controller, 'old_controller_running', lambda: True)
    calls = []
    monkeypatch.setattr(controller.previous, 'runtime_amendment', lambda *args: calls.append('runtime'))
    value = {'schema': controller.ACTIVATION_SCHEMA, 'parent_root': str(parent), 'release_root': str(release),
             'artifacts_root': str(artifacts), 'parent_array_job_id': controller.PARENT_ARRAY_JOB_ID,
             'previous_controller_pid': controller.PREVIOUS_CONTROLLER_PID, 'previous_controller_stopped': True,
             'host': controller.os.uname().nodename, 'source_selection_policy': controller.POLICY,
             'runtime_profile': controller.launch.RUNTIME_PROFILE, 'files_sha256': controller.pin_files([source])}
    write(artifacts / 'controller_activation.json', value)
    with pytest.raises(ValueError, match='previous controller is still running'):
        controller.validate_activation(parent, release, artifacts)
    assert calls == []


def array_fixture(tmp_path, states):
    directory = tmp_path / 'stage'
    cells = []
    for index in range(len(states)):
        tasks = write(directory / f'tasks_{index}.json', [{'output': str(directory / f'receipt_{index}.json')}])
        cells.append({'tasks': str(tasks)})
    path = write(directory / 'plan.json', {'cells': cells})
    write(directory / 'submission_result.json', {'status': 'submitted', 'array_job_id': 73})
    return path


@pytest.mark.parametrize('state', ['FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED'])
def test_scheduler_queries_expanded_array_ids_and_reports_exact_failed_element(tmp_path, monkeypatch, state):
    path = array_fixture(tmp_path, ['RUNNING', state])
    calls = []

    def query(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=f'73_0|RUNNING|Unknown\n73_1|{state}+|Unknown\n', stderr='')

    monkeypatch.setattr(controller.subprocess, 'run', query)
    result = controller.observe_array(path)
    assert '--array' in calls[0][0] and calls[0][1]['timeout'] == 30
    assert result == {'status': 'execution_failed', 'jobs': [{'job': '73_1', 'state': state}]}


@pytest.mark.parametrize('seconds,failed', [(30, False), (180, True)])
def test_completed_array_waits_for_receipt_visibility_grace(tmp_path, monkeypatch, seconds, failed):
    path = array_fixture(tmp_path, ['COMPLETED'])
    ended = (datetime.now() - timedelta(seconds=seconds)).isoformat()
    monkeypatch.setattr(controller.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=f'73_0|COMPLETED|{ended}\n', stderr=''))
    result = controller.observe_array(path)
    assert bool(result) is failed
    if failed:
        assert result['jobs'][0]['error'] == 'missing_receipts_after_visibility_grace'


def test_transient_scheduler_timeout_is_visible_and_does_not_restart(tmp_path, monkeypatch):
    path = array_fixture(tmp_path, ['RUNNING'])

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('sacct', 30)

    monkeypatch.setattr(controller.subprocess, 'run', timeout)
    result = controller.observe_array(path)
    assert result['status'] == 'scheduler_query_unavailable'
    assert result['array_job_id'] == 73 and result['error'] == 'TimeoutExpired'


def test_failed_element_with_complete_receipt_is_not_restarted(tmp_path, monkeypatch):
    path = array_fixture(tmp_path, ['TIMEOUT'])
    write(path.parent / 'receipt_0.json', {'completed_before_job_timeout': True})
    monkeypatch.setattr(controller.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout='73_0|TIMEOUT|Unknown\n', stderr=''))
    assert controller.observe_array(path) is None


@pytest.fixture
def submission(tmp_path, monkeypatch):
    parent, artifacts, parent_artifacts = [tmp_path / name for name in ('parent', 'artifacts', 'parent_artifacts')]
    monkeypatch.setattr(controller, 'PARENT_ARTIFACTS', parent_artifacts)
    write(parent / 'protocol.json', {'models': {'7b': {'path': '/fixture/model'}}})
    monkeypatch.setattr(controller.original, 'authenticate', controller.read)
    events = []
    scientific = write(tmp_path / 'scientific.json', {'fixed': True})
    pins = controller.pin_files([scientific])
    cell = {'id': 'level4_countdown', 'level': 'level4', 'domain': 'countdown', 'phase': 'dev',
            'source_kind': 'campaign_v1', 'source_root': str(parent), 'tasks': [{'output': str(tmp_path / 'output.json')}]}

    def record(directory, job):
        path = write(directory / 'plan.json', {'schema': controller.launch.SCHEMA, 'cells': [], 'phase': 'dev'})
        write(directory / 'submission_intent.json', {'plan_sha256': controller.file_sha(path)})
        write(directory / 'submission_result.json', {'status': 'submitted', 'array_job_id': job})

    record(parent_artifacts / 'development', controller.PARENT_ARRAY_JOB_ID)

    def prepare(directory, cells, models, *, dependency_ids, pins):
        events.append(('prepare', tuple(dependency_ids)))
        plan_cells = []
        for index, item in enumerate(cells):
            tasks = write(directory / f'tasks_{index}.json', item['tasks'])
            plan_cells.append({**item, 'tasks': str(tasks)})
        write(directory / 'plan.json', {'schema': controller.launch.SCHEMA, 'cells': plan_cells,
              'dependency_ids': list(dependency_ids), 'scientific_inputs_sha256': pins})

    def submit(path):
        events.append(('submit', str(path)))
        write(path.parent / 'submission_intent.json', {'plan_sha256': controller.file_sha(path)})
        write(path.parent / 'submission_result.json', {'status': 'submitted', 'array_job_id': 800 + len(events)})

    monkeypatch.setattr(controller.launch, 'prepare', prepare)
    monkeypatch.setattr(controller.launch, 'verify', controller.read)
    monkeypatch.setattr(controller.launch, 'submit', submit)
    return SimpleNamespace(parent=parent, artifacts=artifacts, parent_artifacts=parent_artifacts,
                           events=events, pins=pins, cells=[cell], record=record)


def test_stage_submits_once_and_depends_on_parent_and_every_prior_array(submission):
    s = submission
    s.record(s.artifacts / 'prior_a', 111)
    s.record(s.artifacts / 'prior_b', 222)
    args = s.parent, s.artifacts, 'confirmation', s.cells, s.pins
    path, status = controller.submitted_stage(*args, advance=True)
    assert status is None
    assert s.events[0] == ('prepare', (111, 222, controller.PARENT_ARRAY_JOB_ID))
    assert len([event for event in s.events if event[0] == 'submit']) == 1
    assert controller.submitted_stage(*args, advance=True) == (path, None)
    assert len(s.events) == 2


@pytest.mark.parametrize('own', [True, False])
def test_unresolved_submission_intent_prevents_any_prepare_or_submit(submission, own):
    s = submission
    directory = s.artifacts / ('confirmation' if own else 'prior')
    write(directory / 'plan.json', {'schema': controller.launch.SCHEMA, 'cells': []})
    write(directory / 'submission_intent.json', {'plan_sha256': controller.file_sha(directory / 'plan.json')})
    path, status = controller.submitted_stage(s.parent, s.artifacts, 'confirmation', s.cells, s.pins, advance=True)
    assert status['status'] == 'submission_needs_reconciliation'
    assert s.events == []


def test_auxiliary_capacity_smoke_is_not_a_scientific_dependency(submission):
    s = submission
    write(s.parent_artifacts / 'capacity_smoke/plan.json', {'schema': 'synthetic_capacity_smoke_v1'})
    write(s.parent_artifacts / 'capacity_smoke/submission_intent.json', {'scope': 'synthetic_runtime_only'})
    write(s.parent_artifacts / 'capacity_smoke/submission_result.json', {'status': 'submitted', 'job_id': 41})
    assert controller.prior_submissions(s.artifacts) == [controller.PARENT_ARRAY_JOB_ID]


def test_prepared_plan_cannot_omit_newly_observed_prior_array_dependency(submission):
    s = submission
    controller.launch.prepare(s.artifacts / 'confirmation', s.cells, {}, dependency_ids=[controller.PARENT_ARRAY_JOB_ID], pins=s.pins)
    s.record(s.artifacts / 'later_prior', 555)
    with pytest.raises(ValueError, match='dependencies omit earlier arrays'):
        controller.submitted_stage(s.parent, s.artifacts, 'confirmation', s.cells, s.pins, advance=True)
    assert not any(event[0] == 'submit' for event in s.events)


def test_saved_stage_pin_cannot_change(submission):
    s = submission
    controller.launch.prepare(s.artifacts / 'confirmation', s.cells, {}, dependency_ids=[controller.PARENT_ARRAY_JOB_ID], pins=s.pins)
    changed = {source: 'different' for source in s.pins}
    with pytest.raises(ValueError, match='scientific inputs'):
        controller.submitted_stage(s.parent, s.artifacts, 'confirmation', s.cells, changed, advance=True)
    assert not any(event[0] == 'submit' for event in s.events)


def test_watch_retries_transient_scheduler_query_then_reports_terminal_failure(monkeypatch, capsys):
    results = iter([{'status': 'scheduler_query_unavailable', 'array_job_id': 73},
                    {'status': 'waiting_confirmation'},
                    {'status': 'execution_failed', 'jobs': [{'job': '73_1', 'state': 'TIMEOUT'}]}])
    sleeps = []
    monkeypatch.setattr(controller, 'sweep', lambda *args, **kwargs: next(results))
    monkeypatch.setattr(controller.time, 'sleep', sleeps.append)
    result = controller.main(['--watch', '--interval', '1'])
    assert result['status'] == 'execution_failed'
    assert sleeps == [1, 1]
    assert [json.loads(line)['status'] for line in capsys.readouterr().out.splitlines()] == [
        'scheduler_query_unavailable', 'waiting_confirmation', 'execution_failed']


def test_saved_revision_fit_must_reproduce_before_reuse(tmp_path, monkeypatch):
    recipe = write(tmp_path / 'recipe.json', {'development_fit_pass': True, 'selected': 'first'})
    monkeypatch.setattr(controller.revision, 'paths', lambda root: {'recipe': recipe})
    calls = []

    def fit(root, *, publish):
        calls.append(publish)
        return {'development_fit_pass': True, 'selected': 'changed'}

    monkeypatch.setattr(controller.revision, 'fit_domain', fit)
    with pytest.raises(ValueError, match='saved revision recipe changed'):
        controller.saved_revision_recipe(tmp_path, advance=True)
    assert calls == [False]


def test_recipe_selection_must_reproduce_before_binding(campaign, monkeypatch):
    monkeypatch.setattr(controller.original_fit, 'fit_domain', lambda *args, **kwargs:
                        {'development_fit_pass': True, 'input_sha256': {}, 'altered': True})
    with pytest.raises(ValueError, match='original recipe is not reproducible'):
        controller.bind_level(campaign.parent, campaign.release, 'level4', campaign.bindings['level4'])


def test_prepared_stage_cannot_omit_any_required_scientific_pin(submission):
    s = submission
    controller.launch.prepare(s.artifacts / 'confirmation', s.cells, {},
                              dependency_ids=[controller.PARENT_ARRAY_JOB_ID], pins={})
    with pytest.raises(ValueError, match='scientific inputs'):
        controller.submitted_stage(s.parent, s.artifacts, 'confirmation', s.cells, s.pins,
                                   advance=True, required_pins=s.pins)
    assert not any(event[0] == 'submit' for event in s.events)


def test_new_optional_current_source_observer_pin_does_not_replan_sealed_stage(submission, tmp_path):
    s = submission
    controller.launch.prepare(s.artifacts / 'confirmation', s.cells, {},
                              dependency_ids=[controller.PARENT_ARRAY_JOB_ID], pins=s.pins)
    observer = write(tmp_path / 'new_observer.json', {'new_independent_source': True})
    observed_pins = {**s.pins, **controller.pin_files([observer])}
    path, status = controller.submitted_stage(s.parent, s.artifacts, 'confirmation', s.cells, observed_pins,
                                              advance=True, required_pins=s.pins)
    assert status is None
    assert len([event for event in s.events if event[0] == 'prepare']) == 1
    assert len([event for event in s.events if event[0] == 'submit']) == 1
    assert str(observer) not in controller.read(path)['scientific_inputs_sha256']


@pytest.fixture
def exclusion_snapshot(tmp_path, monkeypatch):
    # Tiny structural snapshots use real hashes and semantic identity encoding;
    # only Arrow deserialization is replaced with a deterministic row mapping.
    root, carry = tmp_path / 'revision', tmp_path / 'carry'
    level, domain = 'level4', 'countdown'
    p = {'level': level, 'domain': domain, 'parent_root': str(carry)}
    protocol_path = write(root / 'protocol.json', p)
    q = controller.domain_paths(root, level, domain)
    monkeypatch.setattr(controller.revision, 'paths', lambda source: q)
    monkeypatch.setattr(controller.revision, 'authenticate', lambda source: controller.read(protocol_path))
    frozen = controller.domain_paths(carry, 'level5', domain)['dataset']
    rows_by_path = {}
    arrow_paths = []
    all_rows = []
    for index, (split, (count, subset)) in enumerate(controller.SPLITS.items()):
        rows = [{'problem': 'Carried countdown fixture ' + split,
                 'answer': json.dumps({'numbers': [2, 3, 4, 5], 'target': 20 + index})}]
        rows_by_path[frozen / split] = rows
        all_rows.extend(rows)
        arrow_paths.append(write(frozen / split / 'data.arrow', {'rows': rows}))
        write(frozen / (split + '.jsonl'), rows[0])
    monkeypatch.setattr(controller, 'load_rows', lambda path, subset: rows_by_path[Path(path)])
    monkeypatch.setattr(controller.revision, 'load_rows', lambda path, subset: rows_by_path[Path(path)])
    snapshot_path = root / 'exclusions/development.json'
    snapshot = {'schema': 'modebench_scale_revision_exclusions_v1', 'stage': 'development',
                'protocol_sha256': controller.file_sha(protocol_path),
                'identities': sorted(controller.revision.identity_set(domain, all_rows), key=controller.sha),
                'prompt_sha256': sorted(controller.sha(row['problem']) for row in all_rows),
                'files_sha256': controller.pin_files([protocol_path, *arrow_paths])}
    write(snapshot_path, snapshot)
    certificate_path = q['pools'] / 'identity.json'
    certificate = {'schema': 'modebench_scale_revision_pools_v1', 'level': level, 'domain': domain,
                   'protocol_sha256': controller.file_sha(protocol_path), 'exclusions': str(snapshot_path),
                   'exclusions_sha256': controller.file_sha(snapshot_path)}
    write(certificate_path, certificate)
    carried = [('level5', domain, {'source_kind': 'campaign_v1', 'source_root': str(carry)}),
               ('level5', 'pantry', {'source_kind': 'campaign_v1', 'source_root': str(tmp_path / 'unrelated')})]
    return SimpleNamespace(root=root, level=level, domain=domain, carried=carried,
                           snapshot=snapshot_path, certificate=certificate_path, arrow=arrow_paths,
                           validate=lambda: controller.validate_revision_pool_exclusions(root, level, domain, carried))


def test_revision_pool_snapshot_covers_every_carried_same_domain_split(exclusion_snapshot):
    s = exclusion_snapshot
    pins = s.validate()
    assert pins[str(s.snapshot)] == controller.file_sha(s.snapshot)
    assert pins[str(s.certificate)] == controller.file_sha(s.certificate)
    assert all(pins[str(path)] == controller.file_sha(path) for path in s.arrow)


@pytest.mark.parametrize('changed', ['certificate_protocol', 'snapshot_hash', 'snapshot_stage', 'snapshot_protocol',
                                     'missing_arrow_pin', 'changed_arrow', 'missing_identity', 'missing_prompt',
                                     'precarry_snapshot'])
def test_stale_or_precarried_revision_pool_snapshot_is_rejected(exclusion_snapshot, changed):
    s = exclusion_snapshot
    snapshot, certificate = controller.read(s.snapshot), controller.read(s.certificate)
    if changed == 'certificate_protocol':
        certificate['protocol_sha256'] = 'different'
    elif changed == 'snapshot_hash':
        s.snapshot.write_text(s.snapshot.read_text() + ' ')
    elif changed == 'snapshot_stage':
        snapshot['stage'] = 'freeze'
    elif changed == 'snapshot_protocol':
        snapshot['protocol_sha256'] = 'different'
    elif changed == 'missing_arrow_pin':
        snapshot['files_sha256'].pop(str(s.arrow[0]))
    elif changed == 'changed_arrow':
        s.arrow[0].write_text('changed after snapshot')
    elif changed == 'missing_identity':
        snapshot['identities'].pop()
    elif changed == 'missing_prompt':
        snapshot['prompt_sha256'].pop()
    elif changed == 'precarry_snapshot':
        snapshot['files_sha256'] = {}
        snapshot['identities'] = []
        snapshot['prompt_sha256'] = []
    if changed not in {'snapshot_hash', 'changed_arrow'}:
        write(s.snapshot, snapshot)
        certificate['exclusions_sha256'] = controller.file_sha(s.snapshot)
    write(s.certificate, certificate)
    with pytest.raises(ValueError):
        s.validate()
