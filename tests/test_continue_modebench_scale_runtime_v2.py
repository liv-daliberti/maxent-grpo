"""Runtime v2 preserves scientific gates and requires prospective capacity evidence."""
import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops/exp_scaling'))
import continue_modebench_scale_runtime_v2 as controller


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    data, artifacts = tmp_path / 'data', tmp_path / 'artifacts'
    source = tmp_path / 'generator.py'
    source.write_text('registered generator')
    target = {'pass1': .25, 'pass8': .5}
    protocol = {'schema': 'modebench_scale_protocol_v1',
                'files_sha256': {str(source): controller.file_sha(source)},
                'targets': {domain: {'metrics': target} for domain in controller.DOMAINS},
                'models': {label: {'path': str(tmp_path / label)} for label in controller.LEVELS.values()}}
    write(data / 'protocol.json', protocol)
    write(artifacts / 'development/plan.json', {
        'phase': 'dev', 'data_root': str(data), 'concurrency': 1,
        'cells': [{'id': level + '_' + domain} for level in controller.LEVELS for domain in controller.DOMAINS]})
    write(artifacts / 'development/submission_result.json', {'status': 'submitted', 'array_job_id': 500})
    calls = {'fit': [], 'freeze': [], 'prepare': [], 'submit': [], 'confirm': []}
    failures = {'dev': set(), 'eval': set()}
    monkeypatch.setattr(controller.launch, 'verify', lambda path: controller.read(path))
    monkeypatch.setattr(controller.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout='', stderr=''))
    def fit(data, level, domain):
        calls['fit'].append((level, domain))
        value = {'schema': controller.fit.SCHEMA, 'level': level, 'domain': domain,
                 'development_fit_pass': (level, domain) not in failures['dev'],
                 'protocol_sha256': controller.file_sha(data / 'protocol.json'),
                 'target': protocol['targets'][domain], 'fitter_sha256': controller.file_sha(controller.fit.__file__),
                 'input_sha256': {str(p): controller.file_sha(p)
                                  for p in controller.receipt_paths(data, level, 'dev') if p.parent.name == domain}}
        controller.atomic_new(data / level / 'recipes' / (domain + '.json'), value)
        return value
    monkeypatch.setattr(controller.fit, 'fit_domain', fit)
    def freeze(data, level, domain):
        calls['freeze'].append((level, domain))
        base = data / level / 'dataset' / domain
        splits = {}
        for split, (count, _) in controller.SPLITS.items():
            rows = [{'problem': f'{level}/{domain}/{split}/{i}', 'answer': '{}'} for i in range(count)]
            base.mkdir(parents=True, exist_ok=True)
            (base / (split + '.jsonl')).write_text(''.join(json.dumps(row) + '\n' for row in rows))
            (base / split).mkdir()
            splits[split] = {'rows': count, 'rows_sha256': controller.sha(rows)}
        write(base / 'identity.json', {'schema': 'modebench_scale_frozen_domain_v1',
              'status': 'frozen_pending_heldout_confirmation', 'level': level, 'domain': domain,
              'protocol_sha256': controller.file_sha(data / 'protocol.json'),
              'recipe_sha256': controller.file_sha(data / level / 'recipes' / (domain + '.json')), 'splits': splits})
    monkeypatch.setattr(controller.materialize, 'freeze_dataset', freeze)
    monkeypatch.setattr(controller, 'load_rows', lambda path, subset:
                        [json.loads(line) for line in path.with_suffix('.jsonl').read_text().splitlines()])
    def prepare(inputs, path, models, *, phase, concurrency, data_root, levels, dependency_ids):
        calls['prepare'].append(levels[0])
        assert inputs is None and phase == 'eval' and concurrency == 1
        assert models == {label: model['path'] for label, model in protocol['models'].items()}
        write(path / 'plan.json', {'phase': phase, 'data_root': str(data_root), 'dependency_ids': list(dependency_ids), 'concurrency': concurrency,
              'cells': [{'id': levels[0] + '_' + domain} for domain in controller.DOMAINS]})
    monkeypatch.setattr(controller.launch, 'prepare', prepare)
    def submit(path):
        calls['submit'].append(path.parent.name)
        controller.atomic_new(path.parent / 'submission_intent.json', {'plan_sha256': controller.file_sha(path)})
        controller.atomic_new(path.parent / 'submission_result.json', {'status': 'submitted', 'array_job_id': 100})
    monkeypatch.setattr(controller.launch, 'submit', submit)
    def confirm(data, level, domain):
        calls['confirm'].append((level, domain))
        path = data / level / 'results/confirmation' / (domain + '.json')
        receipt = controller.read(path)
        delta = {metric: receipt['metrics'][metric] - target[metric] for metric in target}
        gates = {metric: abs(delta[metric]) <= controller.fit.TOLERANCES[metric] for metric in target}
        value = {'schema': 'modebench_scale_confirmation_v1', 'level': level, 'domain': domain,
                 'target': target, 'metrics': receipt['metrics'], 'differences': delta, 'gates': gates,
                 'difficulty_matched': all(gates.values()), 'original_grader_replayed_attempts': 128 * 32,
                 'receipt_sha256': controller.file_sha(path),
                 'recipe_sha256': controller.file_sha(data / level / 'recipes' / (domain + '.json')),
                 'dataset_identity_sha256': controller.file_sha(data / level / 'dataset' / domain / 'identity.json')}
        controller.atomic_new(data / level / 'confirmation' / (domain + '.json'), value)
        return value
    monkeypatch.setattr(controller.fit, 'confirm_domain', confirm)
    def receipts(level, phase):
        for path in controller.receipt_paths(data, level, phase):
            domain = path.parent.name if phase == 'dev' else path.stem
            metrics = dict(target)
            if phase == 'eval' and (level, domain) in failures['eval']:
                metrics['pass1'] = 0.
            write(path, {'schema': 'modebench-scale-calibration-independent-v1', 'status': 'complete',
                  'level': level, 'domain': domain, 'split': phase, 'model_label': controller.LEVELS[level],
                  'identity_sha256': controller.sha([level, domain, phase, str(path)]), 'metrics': metrics})
    monkeypatch.setattr(controller, 'REGISTERED_PROTOCOL_SHA256', controller.file_sha(data / 'protocol.json'))
    smoke = {}
    for label in controller.LEVELS.values():
        path = artifacts / ('fake_runtime_smoke_' + label + '.json')
        write(path, {'status': 'pass', 'model_label': label, 'runtime_profile': controller.runtime_profile()})
        smoke[label] = {'path': str(path), 'sha256': controller.file_sha(path)}
    write(artifacts / 'runtime_amendment.json', {
        'schema': controller.AMENDMENT_SCHEMA, 'data_root': str(data), 'artifacts_root': str(artifacts),
        'protocol_sha256': controller.file_sha(data / 'protocol.json'),
        'original_controller_sha256': controller.ORIGINAL_CONTROLLER_SHA256,
        'scientific_protocol_unchanged': True, 'model_identities_unchanged': True,
        'candidate_outcomes_observed_before_replacement': False,
        'former_array_job_id': controller.FORMER_ARRAY_JOB_ID,
        'replacement_reason': 'Synthetic test fixture; no Slurm calls or GPU inference.',
        'runtime_profile': controller.runtime_profile(), 'runtime_smoke_receipts': smoke,
        'files_sha256': {str(path): controller.file_sha(path) for path in controller.amendment_sources(data)}})
    return SimpleNamespace(data=data, artifacts=artifacts, calls=calls, failures=failures,
                           receipts=receipts, protocol=protocol, source=source)


def sweep(campaign, advance=True):
    return controller.sweep(campaign.data, campaign.artifacts, advance=advance)


def test_default_status_is_read_only_and_waits_for_every_development_tier(campaign):
    before = set(campaign.artifacts.rglob('*'))
    result = sweep(campaign, advance=False)
    assert result['level4'] == {'status': 'waiting_development', 'complete': 0, 'expected': 20}
    assert set(campaign.artifacts.rglob('*')) == before
    campaign.receipts('level4', 'dev')
    missing = controller.receipt_paths(campaign.data, 'level4', 'dev')[-1]
    missing.unlink()
    assert sweep(campaign)['level4']['complete'] == 19
    assert not any(campaign.calls.values())
    campaign.receipts('level4', 'dev')
    assert sweep(campaign, advance=False)['level4']['status'] == 'ready_to_fit'
    assert not any(campaign.calls.values())


def test_complete_campaign_fits_freezes_submits_audits_and_releases_once(campaign):
    for level in controller.LEVELS:
        campaign.receipts(level, 'dev')
    assert all(value['status'] == 'waiting_confirmation' for value in sweep(campaign).values())
    assert len(campaign.calls['fit']) == len(campaign.calls['freeze']) == 10
    assert len(campaign.calls['submit']) == 2
    assert sweep(campaign)['level4']['status'] == 'waiting_confirmation'
    assert len(campaign.calls['fit']) == 10 and len(campaign.calls['submit']) == 2
    for level in controller.LEVELS:
        campaign.receipts(level, 'eval')
    assert sweep(campaign, advance=False)['level4']['status'] == 'ready_to_audit'
    assert not campaign.calls['confirm']
    assert all(value['status'] == 'admitted' for value in sweep(campaign).values())
    assert len(campaign.calls['confirm']) == 10
    assert controller.read(campaign.data / 'admission.json')['difficulty_matched'] is True
    before = {path: path.read_bytes() for path in campaign.data.rglob('admission.json')}
    assert sweep(campaign)['level5']['status'] == 'admitted'
    assert len(campaign.calls['confirm']) == 10
    assert all(path.read_bytes() == body for path, body in before.items())
    for level in controller.LEVELS:
        text = (campaign.data / level / 'README.md').read_text()
        assert all(f'dataset/{domain}/train' in text and f'dataset/{domain}/eval' in text for domain in controller.DOMAINS)


def test_failed_development_saves_all_five_recipes_and_never_freezes_or_submits(campaign):
    campaign.failures['dev'].add(('level4', 'countdown'))
    campaign.receipts('level4', 'dev')
    result = sweep(campaign)['level4']
    assert result == {'status': 'needs_new_development_revision', 'failed_domains': ['countdown']}
    assert len(campaign.calls['fit']) == 5
    assert not campaign.calls['freeze'] and not campaign.calls['submit']
    assert sweep(campaign)['level4'] == result
    assert len(campaign.calls['fit']) == 5
    assert not list(campaign.data.rglob('admission.json'))


def test_partial_existing_recipes_are_validated_and_only_missing_ones_are_fitted(campaign):
    campaign.receipts('level4', 'dev')
    controller.fit.fit_domain(campaign.data, 'level4', 'countdown')
    sweep(campaign)
    assert campaign.calls['fit'].count(('level4', 'countdown')) == 1
    assert len(campaign.calls['fit']) == 5


def test_failed_heldout_keeps_failure_and_never_publishes_admission(campaign):
    campaign.receipts('level4', 'dev')
    sweep(campaign)
    campaign.failures['eval'].add(('level4', 'pantry'))
    campaign.receipts('level4', 'eval')
    expected = {'status': 'heldout_confirmation_failed', 'failed_domains': ['pantry']}
    assert sweep(campaign)['level4'] == expected
    assert len(campaign.calls['confirm']) == 5
    assert sweep(campaign, advance=False)['level4'] == expected
    assert sweep(campaign)['level4'] == expected
    assert len(campaign.calls['confirm']) == 5
    assert not list(campaign.data.rglob('admission.json'))


def test_ambiguous_submission_intent_prevents_any_retry(campaign, monkeypatch):
    campaign.receipts('level4', 'dev')
    def interrupted(path):
        campaign.calls['submit'].append(path.parent.name)
        controller.atomic_new(path.parent / 'submission_intent.json', {'plan_sha256': controller.file_sha(path)})
        raise RuntimeError('interrupted after sbatch')
    monkeypatch.setattr(controller.launch, 'submit', interrupted)
    with pytest.raises(RuntimeError, match='interrupted'):
        sweep(campaign)
    assert sweep(campaign)['level4']['status'] == 'submission_needs_reconciliation'
    assert len(campaign.calls['submit']) == 1


def test_duplicate_receipt_identity_is_rejected_before_fitting(campaign):
    campaign.receipts('level4', 'dev')
    paths = controller.receipt_paths(campaign.data, 'level4', 'dev')
    value = controller.read(paths[1])
    value['identity_sha256'] = controller.read(paths[0])['identity_sha256']
    write(paths[1], value)
    with pytest.raises(ValueError, match='duplicate receipt'):
        sweep(campaign)
    assert not campaign.calls['fit']


def test_changed_recipe_inputs_and_frozen_rows_are_rejected_on_resume(campaign):
    campaign.receipts('level4', 'dev')
    sweep(campaign)
    path = controller.receipt_paths(campaign.data, 'level4', 'dev')[0]
    value = controller.read(path)
    value['extra'] = 'changed'
    write(path, value)
    with pytest.raises(ValueError, match='recorded input changed'):
        sweep(campaign)
    value.pop('extra')
    write(path, value)
    rows_path = campaign.data / 'level4/dataset/countdown/train.jsonl'
    rows_path.write_text(rows_path.read_text().replace('countdown', 'changed', 1))
    with pytest.raises(ValueError, match='frozen dataset rows changed'):
        sweep(campaign)


def test_changed_confirmation_target_cannot_be_reused_or_released(campaign):
    campaign.receipts('level4', 'dev')
    sweep(campaign)
    campaign.failures['eval'].add(('level4', 'pantry'))
    campaign.receipts('level4', 'eval')
    sweep(campaign)
    path = campaign.data / 'level4/confirmation/pantry.json'
    value = controller.read(path)
    value['target'] = dict(value['metrics'])
    value['differences'] = {'pass1': 0, 'pass8': 0}
    value['gates'] = {'pass1': True, 'pass8': True}
    value['difficulty_matched'] = True
    write(path, value)
    with pytest.raises(ValueError, match='audit inputs changed'):
        sweep(campaign)
    assert not list(campaign.data.rglob('admission.json'))


def test_release_helper_requires_all_heldout_gates(campaign):
    campaign.receipts('level4', 'dev')
    sweep(campaign)
    campaign.failures['eval'].add(('level4', 'countdown'))
    campaign.receipts('level4', 'eval')
    sweep(campaign)
    with pytest.raises(ValueError, match='all five held-out domains'):
        controller.publish_level(campaign.data, 'level4', campaign.protocol)
    assert not list(campaign.data.rglob('admission.json'))


def test_controller_lock_and_arm_identity_refuse_concurrent_or_changed_controller(campaign):
    with controller.controller_lock(campaign.artifacts):
        with pytest.raises(BlockingIOError):
            sweep(campaign)
    sweep(campaign)
    path = campaign.artifacts / 'controller_identity.json'
    seal = controller.read(path)
    seal['files_sha256'][str(Path(controller.__file__))] = 'changed'
    write(path, seal)
    with pytest.raises(ValueError, match='changed after arm'):
        sweep(campaign)


def test_failed_slurm_array_elements_stop_only_the_affected_level(campaign, monkeypatch):
    write(campaign.artifacts / 'development/submission_result.json', {'status': 'submitted', 'array_job_id': 500})
    monkeypatch.setattr(controller.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout='500_0|TIMEOUT\n500_5|RUNNING\n', stderr=''))
    result = sweep(campaign)
    assert result['level4'] == {'status': 'execution_failed', 'jobs': [{'job': '500_0', 'state': 'TIMEOUT'}]}
    assert result['level5']['status'] == 'waiting_development'
    assert not any(campaign.calls.values())


def test_failed_confirmation_job_stops_without_resubmission(campaign, monkeypatch):
    campaign.receipts('level4', 'dev')
    sweep(campaign)
    monkeypatch.setattr(controller.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout='100_3|OUT_OF_MEMORY\n', stderr=''))
    assert sweep(campaign)['level4']['status'] == 'execution_failed'
    assert len(campaign.calls['submit']) == 1


def test_existing_audits_resume_once_and_missing_readme_is_repaired(campaign):
    campaign.receipts('level4', 'dev')
    sweep(campaign)
    campaign.receipts('level4', 'eval')
    controller.fit.confirm_domain(campaign.data, 'level4', 'countdown')
    assert sweep(campaign)['level4']['status'] == 'admitted'
    assert campaign.calls['confirm'].count(('level4', 'countdown')) == 1
    readme = campaign.data / 'level4/README.md'
    expected = readme.read_bytes()
    readme.unlink()
    assert sweep(campaign)['level4']['status'] == 'admitted'
    assert readme.read_bytes() == expected
    assert len(campaign.calls['confirm']) == 5


def test_aggregate_array_failure_is_not_attributed_to_running_level(campaign, monkeypatch):
    monkeypatch.setattr(controller.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout='500|FAILED|Unknown\n500_0|TIMEOUT|Unknown\n500_5|RUNNING|Unknown\n', stderr=''))
    result = sweep(campaign)
    assert result['level4']['status'] == 'execution_failed'
    assert result['level5']['status'] == 'waiting_development'


@pytest.mark.parametrize(('age', 'status'), [(60, 'waiting_development'), (180, 'execution_failed')])
def test_completed_jobs_without_receipts_have_visibility_grace(campaign, monkeypatch, age, status):
    ended = (datetime.now() - timedelta(seconds=age)).isoformat()
    monkeypatch.setattr(controller.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=f'500_0|COMPLETED|{ended}\n', stderr=''))
    assert sweep(campaign)['level4']['status'] == status


def test_completed_cell_with_receipts_does_not_fail_waiting_sibling_cells(campaign, monkeypatch):
    campaign.receipts('level4', 'dev')
    for path in controller.receipt_paths(campaign.data, 'level4', 'dev')[4:]:
        path.unlink()
    ended = (datetime.now() - timedelta(minutes=5)).isoformat()
    monkeypatch.setattr(controller.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=f'500_0|COMPLETED|{ended}\n', stderr=''))
    assert sweep(campaign)['level4']['status'] == 'waiting_development'


def test_confirmation_arrays_depend_on_development_and_prior_arrays_for_global_cap(campaign):
    for level in controller.LEVELS:
        campaign.receipts(level, 'dev')
    sweep(campaign)
    first = controller.read(campaign.artifacts / 'confirmation_level4/plan.json')
    second = controller.read(campaign.artifacts / 'confirmation_level5/plan.json')
    assert first['dependency_ids'] == [500]
    assert second['dependency_ids'] == [100, 500]
    assert len(campaign.calls['submit']) == 2


def test_unresolved_prior_confirmation_blocks_another_array(campaign):
    campaign.receipts('level5', 'dev')
    write(campaign.artifacts / 'confirmation_level4/submission_intent.json', {'unknown_job_id': True})
    assert sweep(campaign)['level5']['status'] == 'submission_needs_reconciliation'
    assert not campaign.calls['prepare'] and not campaign.calls['submit']


def test_sealed_original_controller_is_unchanged():
    assert controller.file_sha(controller.ORIGINAL_CONTROLLER) == controller.ORIGINAL_CONTROLLER_SHA256


def test_missing_prospective_amendment_rejects_advance_but_status_remains_read_only(campaign):
    (campaign.artifacts / 'runtime_amendment.json').unlink()
    assert sweep(campaign, advance=False)['level4']['status'] == 'waiting_development'
    with pytest.raises(ValueError, match='prospective runtime amendment required'):
        sweep(campaign)
    assert not (campaign.artifacts / 'controller_identity.json').exists()
    assert not any(campaign.calls.values())


@pytest.mark.parametrize(('field', 'value'), [
    ('scientific_protocol_unchanged', False), ('model_identities_unchanged', False),
    ('candidate_outcomes_observed_before_replacement', True), ('protocol_sha256', 'wrong'),
    ('original_controller_sha256', 'wrong'), ('former_array_job_id', 123),
    ('replacement_reason', ''), ('runtime_profile', {'concurrency': 2})])
def test_nonprospective_or_changed_runtime_amendment_is_rejected(campaign, field, value):
    path = campaign.artifacts / 'runtime_amendment.json'
    amendment = controller.read(path)
    amendment[field] = value
    write(path, amendment)
    with pytest.raises(ValueError, match='runtime amendment identity'):
        sweep(campaign)
    assert not any(campaign.calls.values())


def test_amendment_pins_cannot_omit_new_controller_source(campaign):
    path = campaign.artifacts / 'runtime_amendment.json'
    amendment = controller.read(path)
    amendment['files_sha256'].pop(str(Path(controller.__file__).resolve()))
    write(path, amendment)
    with pytest.raises(ValueError, match='omits required source pins'):
        sweep(campaign)


def test_each_model_requires_its_own_passing_capacity_receipt(campaign):
    path = campaign.artifacts / 'runtime_amendment.json'
    amendment = controller.read(path)
    amendment['runtime_smoke_receipts'].pop('14b')
    write(path, amendment)
    with pytest.raises(ValueError, match='both model scales'):
        sweep(campaign)


def test_smoke_result_and_profile_are_authenticated_before_arm(campaign):
    path = campaign.artifacts / 'runtime_amendment.json'
    amendment = controller.read(path)
    receipt_path = Path(amendment['runtime_smoke_receipts']['14b']['path'])
    receipt = controller.read(receipt_path)
    receipt['status'] = 'fail'
    write(receipt_path, receipt)
    with pytest.raises(ValueError, match='runtime smoke receipt changed'):
        sweep(campaign)
    amendment['runtime_smoke_receipts']['14b']['sha256'] = controller.file_sha(receipt_path)
    write(path, amendment)
    with pytest.raises(ValueError, match='runtime smoke receipt did not pass'):
        sweep(campaign)
    assert not (campaign.artifacts / 'controller_identity.json').exists()


def test_amendment_is_pinned_in_controller_seal(campaign):
    sweep(campaign)
    path = campaign.artifacts / 'runtime_amendment.json'
    seal = controller.read(campaign.artifacts / 'controller_identity.json')
    assert seal['schema'] == 'modebench_scale_controller_runtime_v2'
    assert seal['files_sha256'][str(path)] == controller.file_sha(path)
    amendment = controller.read(path)
    amendment['replacement_reason'] += ' Edited after arming.'
    write(path, amendment)
    with pytest.raises(ValueError, match='changed after arm'):
        sweep(campaign)


def test_development_concurrency_cannot_exceed_owner_capacity(campaign):
    path = campaign.artifacts / 'development/plan.json'
    plan = controller.read(path)
    plan['concurrency'] = 2
    write(path, plan)
    with pytest.raises(ValueError, match='wrong development plan'):
        sweep(campaign)
    assert not any(campaign.calls.values())


def test_confirmation_concurrency_cannot_exceed_owner_capacity(campaign):
    campaign.receipts('level4', 'dev')
    sweep(campaign)
    path = campaign.artifacts / 'confirmation_level4/plan.json'
    plan = controller.read(path)
    plan['concurrency'] = 2
    write(path, plan)
    with pytest.raises(ValueError, match='wrong confirmation plan'):
        sweep(campaign)
    assert len(campaign.calls['submit']) == 1


def test_failure_queries_expand_array_job_ids_instead_of_raw_allocation_ids(campaign, monkeypatch):
    calls = []
    def accounting(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stderr='', stdout=(
            '500_0|CANCELLED by 363432|2026-09-11T04:00:00\n'
            '500_1|PENDING|Unknown\n500_5|RUNNING|Unknown\n'))
    monkeypatch.setattr(controller.subprocess, 'run', accounting)
    result = sweep(campaign)
    assert result['level4'] == {'status': 'execution_failed',
                                'jobs': [{'job': '500_0', 'state': 'CANCELLED'}]}
    assert result['level5']['status'] == 'waiting_development'
    assert all(command == ['sacct', '-n', '-X', '-P', '--array', '-j', '500',
                           '--format=JobID,State,End'] for command, _ in calls)
    assert all(kwargs == dict(capture_output=True, text=True, check=False, timeout=30)
               for _, kwargs in calls)
    assert not any(campaign.calls.values())


@pytest.mark.parametrize('failure', ['timeout', 'nonzero', 'oserror'])
def test_scheduler_query_failures_are_visible_nonterminal_and_do_not_retry_jobs(campaign, monkeypatch, failure):
    def unavailable(command, **kwargs):
        if failure == 'timeout':
            raise controller.subprocess.TimeoutExpired(command, 30)
        if failure == 'oserror':
            raise OSError('temporary scheduler transport failure')
        return SimpleNamespace(returncode=1, stdout='', stderr='slurmdbd is temporarily unavailable')
    monkeypatch.setattr(controller.subprocess, 'run', unavailable)
    result = sweep(campaign)
    assert all(value['status'] == 'scheduler_query_unavailable' for value in result.values())
    assert all(value['array_job_id'] == 500 and value['detail'] for value in result.values())
    assert result['level4']['error'] == {'timeout': 'TimeoutExpired', 'nonzero': 'sacct_nonzero_exit',
                                        'oserror': 'OSError'}[failure]
    assert 'scheduler_query_unavailable' not in controller.TERMINAL
    assert not any(campaign.calls.values())
    assert not list(campaign.data.rglob('admission.json'))
    monkeypatch.setattr(controller.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout='500_0|PENDING|Unknown\n', stderr=''))
    assert sweep(campaign)['level4']['status'] == 'waiting_development'


def test_watch_survives_accounting_timeout_and_reports_recovery(campaign, monkeypatch, capsys):
    calls = {'accounting': 0, 'sleep': 0}
    def accounting(command, **kwargs):
        calls['accounting'] += 1
        if calls['accounting'] <= 2:
            raise controller.subprocess.TimeoutExpired(command, 30)
        return SimpleNamespace(returncode=0, stdout='500_0|PENDING|Unknown\n500_5|PENDING|Unknown\n', stderr='')
    def sleep(interval):
        assert interval == 1
        calls['sleep'] += 1
        if calls['sleep'] == 2:
            raise KeyboardInterrupt('end bounded watch test')
    monkeypatch.setattr(controller.subprocess, 'run', accounting)
    monkeypatch.setattr(controller.time, 'sleep', sleep)
    with pytest.raises(KeyboardInterrupt, match='bounded watch test'):
        controller.main(['--data-root', str(campaign.data), '--artifacts-root', str(campaign.artifacts),
                         '--watch', '--advance', '--interval', '1'])
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(output) == 2
    assert output[0]['level4']['status'] == 'scheduler_query_unavailable'
    assert output[1]['level4']['status'] == 'waiting_development'
    assert calls['sleep'] == 2 and not any(campaign.calls.values())
