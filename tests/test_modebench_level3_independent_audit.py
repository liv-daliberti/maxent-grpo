"""The independent audit rejects legacy evidence and altered sampled metrics."""
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from test_modebench_level3_independent_evaluator import LLM, evaluate, make_task

PATH = Path(__file__).resolve().parents[1] / "ops/audit_modebench_level3_independent_match.py"
SPEC = spec_from_file_location("independent_audit_test", PATH)
audit = module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_new_receipt_passes_seed_and_metric_validation(tmp_path):
    receipt = evaluate(LLM(), make_task(tmp_path))
    values = audit.validate_receipt(receipt, domain="graph_coloring", role="candidate", development=True)
    assert len(values) == 3
    assert all(value["pass1"] == 3 / 8 and value["pass8"] == 1 for value in values)


def test_correct_seed_metadata_does_not_hide_false_metrics(tmp_path):
    receipt = evaluate(LLM(), make_task(tmp_path))
    receipt["prompt_results"][0]["draws"][0]["pass1"] = 1
    with pytest.raises(ValueError, match="metric mismatch"):
        audit.validate_receipt(receipt, domain="graph_coloring", role="candidate", development=True)


def test_legacy_receipt_schema_cannot_qualify(tmp_path):
    receipt = evaluate(LLM(), make_task(tmp_path))
    receipt["schema"] = "modebench-level3-frozen-calibration-v1"
    with pytest.raises(ValueError, match="independent v2 complete receipt"):
        audit.validate_receipt(receipt, domain="graph_coloring", role="candidate", development=True)


def test_changed_effective_child_stream_cannot_qualify(tmp_path):
    receipt = deepcopy(evaluate(LLM(), make_task(tmp_path)))
    draw = receipt["prompt_results"][0]["draws"][0]
    draw["child_seeds"] = [draw["child_seeds"][0] + 8, *draw["child_seeds"][1:]]
    with pytest.raises(ValueError, match="draw RNG metadata"):
        audit.validate_receipt(receipt, domain="graph_coloring", role="candidate", development=True)


def test_confirmation_seed_labels_cannot_be_chosen_afterward(tmp_path):
    with pytest.raises(ValueError, match="prospective registration"):
        audit.audit_pairs([], dataset_root=tmp_path,
                          expected_eval_seeds=[6329010, 6329011, 6329012, 6329013])


@pytest.mark.parametrize('field', ['row_sha256', 'spec_sha256'])
def test_rehashed_row_identity_cannot_differ_from_source(tmp_path, field):
    receipt = evaluate(LLM(), make_task(tmp_path))
    receipt['prompt_results'][0][field] = '0' * 64
    with pytest.raises(ValueError, match='recorded row/spec/prompt differs'):
        audit.validate_receipt(receipt, domain='graph_coloring', role='candidate', development=True)


def test_changed_row_metadata_cannot_differ_from_source(tmp_path):
    receipt = evaluate(LLM(), make_task(tmp_path))
    receipt['prompt_results'][0]['row_metadata']['candidate_id'] = 'another'
    with pytest.raises(ValueError, match='row metadata differs'):
        audit.validate_receipt(receipt, domain='graph_coloring', role='candidate', development=True)


def test_boolean_metric_is_not_accepted_as_numeric_one(tmp_path):
    receipt = evaluate(LLM(), make_task(tmp_path))
    receipt['prompt_results'][0]['draws'][0]['pass8'] = True
    with pytest.raises(ValueError, match='metric mismatch'):
        audit.validate_receipt(receipt, domain='graph_coloring', role='candidate', development=True)


def publication_metadata(root):
    return {'schema': 'modebench_level3_capability_matched_splits_independent_v2',
            'status': 'structural_checks_pass', 'decision': 'pending_confirmation',
            'split_sizes': {'train': 384, 'dev': 128, 'eval': 128}, 'generation_seed_offset': 1_000_000,
            'information_boundary': {'recipe_frozen_before_eval_generation': True,
                                     'evaluation_model_outcomes_loaded': False, 'treatment_training_started': False},
            'domains': {domain: {split: {'rows': count, 'rows_sha256': '0' * 64,
                                        'path': str(root / domain / split),
                                        'dataset_split': 'train' if split == 'train' else 'multi_answer',
                                        'checks': {key: True for key in audit.required_structural_checks(split)}}
                                for split, count in [('train', 384), ('dev', 128), ('eval', 128)]}
                        for domain in audit.DOMAINS}}


def test_complete_named_structural_certificate_is_accepted(tmp_path):
    audit.validate_publication_metadata(publication_metadata(tmp_path), tmp_path)


@pytest.mark.parametrize('domain', audit.DOMAINS)
@pytest.mark.parametrize('split', ['train', 'dev', 'eval'])
def test_no_domain_or_split_can_omit_required_exclusion_check(tmp_path, domain, split):
    identity = publication_metadata(tmp_path)
    del identity['domains'][domain][split]['checks']['historical_and_cross_split_disjointness']
    with pytest.raises(ValueError, match='required structural checks'):
        audit.validate_publication_metadata(identity, tmp_path)


@pytest.mark.parametrize('field', ['recipe_frozen_before_eval_generation', 'evaluation_model_outcomes_loaded', 'treatment_training_started'])
def test_unsafe_publication_boundary_is_rejected(tmp_path, field):
    identity = publication_metadata(tmp_path)
    identity['information_boundary'][field] = not identity['information_boundary'][field]
    with pytest.raises(ValueError, match='information boundary'):
        audit.validate_publication_metadata(identity, tmp_path)


def test_legacy_structural_publication_cannot_qualify(tmp_path):
    identity = publication_metadata(tmp_path)
    identity['schema'] = 'modebench_level3_capability_matched_splits_v1'
    with pytest.raises(ValueError, match='all-five independent structural'):
        audit.validate_publication_metadata(identity, tmp_path)


def test_unscored_training_rows_must_match_their_frozen_hash(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import datasets
    identity = publication_metadata(tmp_path)
    first = audit.DOMAINS[0]
    original_rows = [{'id': index} for index in range(384)]
    identity['domains'][first]['train']['rows_sha256'] = audit.materializer_row_hash(original_rows)
    changed = deepcopy(original_rows)
    changed[0]['id'] = 'tampered'
    monkeypatch.setattr(datasets, 'load_from_disk', lambda path: {'train': changed})
    monkeypatch.setattr(audit, 'finalizer_module', lambda: SimpleNamespace())
    with pytest.raises(ValueError, match='actual rows differ from structural certificate'):
        audit.verify_all_split_sources(tmp_path, identity)


def test_a_nonempty_subset_of_fixed_control_pins_is_rejected(tmp_path, monkeypatch):
    from types import SimpleNamespace
    pinned = {'control.arrow': 'a' * 64}
    bundle = {'prior_and_fixed_control_files_sha256': pinned}
    identity = deepcopy(bundle)
    monkeypatch.setattr(audit, 'finalizer_module', lambda: SimpleNamespace(
        baseline_and_prior_pins=lambda: {**pinned, 'omitted.arrow': 'b' * 64}))
    with pytest.raises(ValueError, match='complete file inventory'):
        audit.verify_fixed_control_provenance(bundle, identity)


def test_valid_hash_cannot_hide_an_invalid_amendment(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import json
    source = tmp_path / 'finalizer.py'
    source.write_text('frozen finalizer')
    amendment = tmp_path / 'amendment.json'
    amendment.write_text('{}')
    controls = {name: str(tmp_path / name) for name in audit.DOMAINS}
    bundle = {'prior_and_fixed_control_files_sha256': {'control.arrow': 'a' * 64},
              'generation_seed_offset': 1_000_000, 'development_inputs': {},
              'finalizer_source_sha256': audit.file_sha(source), 'baseline_confirmation_paths': controls,
              'prospective_amendment_path': str(amendment), 'prospective_amendment_sha256': audit.file_sha(amendment)}
    def reject(path):
        assert path == amendment
        raise ValueError('strict registered amendment rejection')
    finalizer = SimpleNamespace(__file__=str(source),
        baseline_and_prior_pins=lambda: bundle['prior_and_fixed_control_files_sha256'],
        verify_development_inputs_unchanged=lambda _: None,
        validate_protocol_amendment=reject)
    monkeypatch.setattr(audit, 'finalizer_module', lambda: finalizer)
    monkeypatch.setattr(audit, 'level1_eval_path', lambda name: tmp_path / name)
    with pytest.raises(ValueError, match='strict registered amendment rejection'):
        audit.verify_fixed_control_provenance(bundle, deepcopy(bundle))


def test_current_publication_is_skipped_without_erasing_other_historical_collision(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import datasets
    data = tmp_path / 'var/data'
    current = data / 'modebench_level3_current'
    other = data / 'modebench_harder_old'
    prior = data / 'modebench_level3_prior'
    for root in (current, other, prior):
        path = root / 'mathir' / 'train'
        path.mkdir(parents=True)
        (path / 'dataset_dict.json').write_text('{}')
    opened = []
    def load(path):
        opened.append(path)
        return {'train': [{'id': 'shared_with_current'}]}
    finalizer = SimpleNamespace(PRIOR_REVISION=prior, existing_ids=lambda _: {'original'},
                               identity_set=lambda _, rows: {row['id'] for row in rows})
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    monkeypatch.setattr(datasets, 'load_from_disk', load)
    assert audit.protected_other_history(finalizer, 'mathir', current) == {'original', 'shared_with_current'}
    assert opened == [str(other / 'mathir' / 'train')]


def test_registered_amendment_validator_accepts_actual_prospective_artifact():
    path = Path(__file__).resolve().parents[1] / 'var/artifacts/modebench_level3_v2/confirmation_control_amendment.json'
    amendment = audit.finalizer_module().validate_protocol_amendment(path)
    assert amendment['baseline_confirmation_paths'] == {domain: str(audit.level1_eval_path(domain)) for domain in audit.DOMAINS}


def test_full_saved_split_reader_revalidates_all_five_existing_publication_rows(monkeypatch):
    """Exercise real serialization/support contracts; this never admits v1 as v2.

    The legacy publication supplies a stable CPU fixture. Prior-revision
    exclusion is isolated here because those fixture rows *are* that revision;
    separate tests enforce the v2 schema and protected-identity boundary.
    """
    import json
    root = Path(__file__).resolve().parents[1] / 'var/data/modebench_level3_matched_v1'
    identity = json.loads((root / 'identity.json').read_text())
    finalizer = audit.finalizer_module()
    monkeypatch.setattr(finalizer, 'authenticated_exclusion_rows',
                        lambda _: ({split: [] for split in ('train', 'dev', 'eval')}, []))
    monkeypatch.setattr(audit, 'protected_other_history', lambda *args: set())
    audit.verify_all_split_sources(root, identity)


def test_complete_current_development_chain_includes_rejected_and_revised_ranges():
    inventory = audit.authenticated_development_sources()
    assert len(inventory['manifest']) == 33
    assert len(inventory['blocks']) == 16640
    names = {record['name'] for record in inventory['manifest']}
    assert {f'3b_graph_coloring_d{tier}' for tier in range(4)} <= names
    assert {f'3b_python_factors_d{tier}' for tier in range(4)} <= names
    assert {f'3b_graph_v7_d{tier}' for tier in range(4)} <= names
    assert {f'3b_python_v5_d{tier}' for tier in range(4)} <= names


@pytest.mark.parametrize('change', ['latest_hash', 'source_omission', 'combined_count'])
def test_complete_development_inventory_rejects_changed_chain(tmp_path, monkeypatch, change):
    import json
    if change == 'latest_hash':
        monkeypatch.setattr(audit, 'LATEST_DEVELOPMENT_SEAL_SHA256', '0' * 64)
    else:
        sealed = json.loads(audit.LATEST_DEVELOPMENT_SEAL.read_text())
        if change == 'source_omission':
            sealed['sources'].pop()
        else:
            sealed['combined_distinct_request_blocks'] -= 1
        temporary = tmp_path / 'latest.json'
        temporary.write_text(json.dumps(sealed))
        monkeypatch.setattr(audit, 'LATEST_DEVELOPMENT_SEAL', temporary)
        monkeypatch.setattr(audit, 'LATEST_DEVELOPMENT_SEAL_SHA256', audit.file_sha(temporary))
    with pytest.raises(ValueError, match='seal changed|source inventory differs|combined block count differs'):
        audit.authenticated_development_sources()


@pytest.fixture
def sealed_confirmation(tmp_path, monkeypatch):
    """Synthetic immutable metadata and real deterministic schedules; no outcomes."""
    import json
    from types import SimpleNamespace
    import evaluate_modebench_level3_independent as evaluator
    campaign = tmp_path / 'campaign'
    folder = campaign / 'confirmation'
    folder.mkdir(parents=True)
    dataset = tmp_path / 'final_dataset'
    dataset.mkdir()
    bundle = dataset / 'frozen_recipes.json'
    bundle.write_text('{}')
    latest_path = campaign / 'latest_development.json'
    latest_path.write_text('{}')
    monkeypatch.setattr(audit, 'CAMPAIGN', campaign)
    monkeypatch.setattr(audit, 'LATEST_DEVELOPMENT_SEAL', latest_path)
    monkeypatch.setattr(audit, 'LATEST_DEVELOPMENT_SEAL_SHA256', audit.file_sha(latest_path))
    models = {label: {'path': f'/registered/{label}', 'label': label} for label in ('05b', '3b')}
    inherited_source = campaign / 'inherited.py'
    inherited_source.write_text('fixed inherited implementation')
    development = {'manifest': [{'name': f'prior_{i}'} for i in range(3)],
                   'protocols': [{'path': '/prior/protocol', 'sha256': 'd' * 64}],
                   'blocks': set(range(0, 96, 8)), 'latest': {'files_sha256': {str(inherited_source): audit.file_sha(inherited_source)}, 'models': models},
                   'latest_sha256': audit.file_sha(latest_path)}
    monkeypatch.setattr(audit, 'authenticated_development_sources', lambda: development)
    amendment_path = folder / 'prospective_execution_amendment.json'
    amendment_path.write_bytes((audit.ROOT / 'var/artifacts/modebench_level3_v2/confirmation/prospective_execution_amendment.json').read_bytes())
    execution_file = campaign / 'recovery_complete.json'
    execution_file.write_text('authenticated completed recovery')
    recovery = {'path': str(execution_file), 'sha256': audit.file_sha(execution_file),
                'files_sha256': {str(execution_file): audit.file_sha(execution_file)},
                'recovery_seal_path': '/fixed/recovery/seal.json', 'recovery_seal_sha256': 'c' * 64,
                'scientific_seal_sha256': 'd' * 64, 'jobs': 11, 'attempts': 43008}
    monkeypatch.setattr(audit, 'validate_completion_attestation', lambda: recovery)
    plan = {'dataset_root': str(dataset), 'domains': list(audit.DOMAINS),
            'execution': {'partition': 'all', 'qos': 'normal', 'preempt_mode': 'OFF'},
            'prospective_execution_amendment_path': str(amendment_path),
            'prospective_execution_amendment_sha256': audit.EXECUTION_AMENDMENT_SHA256,
            'confirmation_draw_labels': list(audit.DEFAULT_CONFIRMATION_SEEDS),
            'jobs': [], 'immutable_inputs_sha256': {}}
    identity = {'frozen_recipe_bundle_sha256': audit.file_sha(bundle), 'domains': {}}
    baseline_paths, loaded, sealed_sources, files, trees = {}, {}, [], {}, {}
    for domain in audit.DOMAINS:
        identity['domains'][domain] = {}
        for label in ('05b', '3b'):
            source = tmp_path / 'controls' / domain if label == '05b' else dataset / domain / 'eval'
            source.mkdir(parents=True)
            rows = [{'problem': f'{domain}/{label}/row{index}'} for index in range(128)]
            rowfile = source / 'rows.json'
            rowfile.write_text(json.dumps(rows))
            source_identity = {'kind': 'saved_dataset', 'path': str(source), 'row_limit': 0, 'row_offset': 0,
                               'selected_rows': 128, 'total_rows': 128, 'rows_sha256': evaluator.sha(rows),
                               'all_rows_sha256': evaluator.sha(rows)}
            loaded[str(source)] = (rows, source_identity)
            if label == '05b':
                baseline_paths[domain] = source
            else:
                identity['domains'][domain]['eval'] = {'rows_sha256': audit.materializer_row_hash(rows)}
            task = {'domain': domain, 'level': 'level1' if label == '05b' else 'level3', 'split': 'eval',
                    'dataset': str(source), 'interface': evaluator.INTERFACE,
                    'seeds': list(audit.DEFAULT_CONFIRMATION_SEEDS), 'batch_size': 8,
                    'row_offset': 0, 'row_limit': 0, 'output': str(tmp_path / f'{label}_{domain}.json')}
            task_path = folder / f'{label}_{domain}_tasks.json'
            task_path.write_text(json.dumps([task]))
            plan['immutable_inputs_sha256'][str(task_path)] = audit.file_sha(task_path)
            plan['jobs'].append({'name': f'{label}_{domain}', 'domain': domain, 'model_label': label,
                                 'tasks': str(task_path), 'output': task['output']})
            schedule = evaluator.schedule_record(domain, rows, task['seeds'])
            sealed_sources.append({'name': f'{label}_{domain}', 'domain': domain, 'model_label': label,
                                   'identity': source_identity, 'seed_schedule_sha256': evaluator.sha(schedule),
                                   'distinct_request_blocks': 512, 'distinct_child_seeds': 4096})
            trees[str(source)] = [str(rowfile)]
            files[str(rowfile)] = audit.file_sha(rowfile)
    monkeypatch.setattr(evaluator, 'load_rows', lambda task: loaded[task['dataset']])
    monkeypatch.setattr(audit, 'level1_eval_path', lambda domain: baseline_paths[domain])
    plan_path = folder / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    files.update(plan['immutable_inputs_sha256'])
    files.update(recovery['files_sha256'])
    files[str(amendment_path)] = audit.file_sha(amendment_path)
    for path in (bundle, latest_path, plan_path, inherited_source, Path(audit.__file__).resolve()):
        files[str(path)] = audit.file_sha(path)
    trees[str(dataset)] = sorted(str(path) for path in dataset.rglob('*') if path.is_file())
    seal = {'schema': 'modebench_level3_independent_confirmation_input_seal_v2',
            'development_execution_recovery': {key: value for key, value in recovery.items() if key != 'files_sha256'},
            'execution': dict(plan['execution']),
            'prospective_execution_amendment_path': str(amendment_path),
            'prospective_execution_amendment_sha256': audit.EXECUTION_AMENDMENT_SHA256,
            'plan': str(plan_path), 'plan_sha256': audit.file_sha(plan_path),
            'inherited_latest_development_seal': str(latest_path),
            'inherited_latest_development_seal_sha256': audit.file_sha(latest_path),
            'confirmation_outcomes_loaded': False, 'treatment_training_started': False,
            'fresh_heldout_claim_applies_to_level3_only': True, 'level1_controls_are_untouched': False,
            'confirmation_draw_labels': list(audit.DEFAULT_CONFIRMATION_SEEDS),
            'files_sha256': files, 'directory_files': trees, 'models': models,
            'development_sources': development['manifest'], 'development_protocols': development['protocols'],
            'development_request_blocks_checked_disjoint': len(development['blocks']),
            'frozen_recipe_bundle_sha256': audit.file_sha(bundle), 'amendment_sha256': 'a' * 64,
            'sources': sealed_sources, 'distinct_request_blocks': 5120, 'distinct_child_seeds': 40960}
    seal_path = folder / 'seal.json'
    claim_path = campaign / 'confirmation_execution_claim.json'
    def resign():
        seal_path.write_text(json.dumps(seal))
        claim_path.write_text(json.dumps({'seal': str(seal_path), 'seal_sha256': audit.file_sha(seal_path),
            'plan': str(plan_path), 'plan_sha256': audit.file_sha(plan_path),
            'amendment_sha256': seal['amendment_sha256'], 'jobs': 10}))
    resign()
    return SimpleNamespace(seal=seal, seal_path=seal_path, claim_path=claim_path, plan=plan,
                           plan_path=plan_path, dataset=dataset, identity=identity, loaded=loaded,
                           development=development, recovery=recovery, resign=resign)


def test_full_prospective_confirmation_inventory_is_independently_reproduced(sealed_confirmation):
    fixture = sealed_confirmation
    evidence = audit.validate_confirmation_seal(fixture.dataset, fixture.identity)
    assert evidence['confirmation_sources'] == 10
    assert evidence['confirmation_request_blocks'] == 5120
    assert evidence['confirmation_child_seeds'] == 40960
    assert evidence['development_request_blocks'] == 12


@pytest.mark.parametrize('change', ['source_omitted', 'protocol_omitted', 'count_retyped', 'count_changed'])
def test_confirmation_seal_cannot_omit_or_retype_prior_sampling(sealed_confirmation, change):
    fixture = sealed_confirmation
    if change == 'source_omitted':
        fixture.seal['development_sources'] = fixture.seal['development_sources'][:-1]
    elif change == 'protocol_omitted':
        fixture.seal['development_protocols'] = []
    else:
        fixture.seal['development_request_blocks_checked_disjoint'] = 12.0 if change == 'count_retyped' else 8
    fixture.resign()
    with pytest.raises(ValueError, match='omits or changes registered development schedules'):
        audit.validate_confirmation_seal(fixture.dataset, fixture.identity)


@pytest.mark.parametrize('change', ['missing_cell', 'schedule_hash', 'child_count', 'source_identity'])
def test_confirmation_seal_cannot_rebind_a_saved_source_schedule(sealed_confirmation, change):
    fixture = sealed_confirmation
    if change == 'missing_cell':
        fixture.seal['sources'].pop()
    elif change == 'schedule_hash':
        fixture.seal['sources'][0]['seed_schedule_sha256'] = '0' * 64
    elif change == 'child_count':
        fixture.seal['sources'][0]['distinct_child_seeds'] = 4095
    else:
        fixture.seal['sources'][0]['identity'] = {'path': '/changed'}
    fixture.resign()
    with pytest.raises(ValueError, match='sealed source schedules or full coverage differ'):
        audit.validate_confirmation_seal(fixture.dataset, fixture.identity)


@pytest.mark.parametrize('collision', ['other_domain_development', 'other_confirmation'])
def test_global_rng_collisions_are_rejected_even_outside_fitted_domain(sealed_confirmation, monkeypatch, collision):
    import evaluate_modebench_level3_independent as evaluator
    fixture = sealed_confirmation
    if collision == 'other_domain_development':
        first = fixture.plan['jobs'][0]
        rows, _ = fixture.loaded[fixture.seal['sources'][0]['identity']['path']]
        base = evaluator.schedule_record(first['domain'], rows, list(audit.DEFAULT_CONFIRMATION_SEEDS))['request_seeds'][0][0]
        fixture.development['blocks'].remove(0)
        fixture.development['blocks'].add(base)
    else:
        original = evaluator.schedule_record
        first = fixture.plan['jobs'][0]
        first_rows, _ = fixture.loaded[fixture.seal['sources'][0]['identity']['path']]
        repeated = original(first['domain'], first_rows, list(audit.DEFAULT_CONFIRMATION_SEEDS))
        def cross_domain_collision(domain, rows, seeds):
            schedule = original(domain, rows, seeds)
            if domain != first['domain']:
                schedule['request_seeds'][0][0] = repeated['request_seeds'][0][0]
            return schedule
        monkeypatch.setattr(evaluator, 'schedule_record', cross_domain_collision)
    with pytest.raises(ValueError, match='overlap development or another confirmation source'):
        audit.validate_confirmation_seal(fixture.dataset, fixture.identity)


@pytest.mark.parametrize('change', ['claim', 'source_file', 'auditor_pin', 'latest_pin', 'latest_source_pin', 'dataset_inventory'])
def test_prospective_source_and_execution_authentication_cannot_be_bypassed(sealed_confirmation, change):
    import json
    fixture = sealed_confirmation
    if change == 'claim':
        fixture.claim_path.write_text('{}')
    elif change == 'source_file':
        source = Path(fixture.seal['sources'][0]['identity']['path']) / 'rows.json'
        source.write_text('modified')
    else:
        if change == 'auditor_pin':
            fixture.seal['files_sha256'].pop(str(Path(audit.__file__).resolve()))
        elif change == 'latest_pin':
            fixture.seal['files_sha256'].pop(str(audit.LATEST_DEVELOPMENT_SEAL))
        elif change == 'latest_source_pin':
            fixture.seal['files_sha256'].pop(next(iter(fixture.development['latest']['files_sha256'])))
        else:
            fixture.seal['directory_files'].pop(str(fixture.dataset))
        fixture.resign()
    with pytest.raises(ValueError):
        audit.validate_confirmation_seal(fixture.dataset, fixture.identity)


@pytest.mark.parametrize('change', ['output', 'source', 'schedule', 'model'])
def test_actual_receipt_must_match_the_prospective_cell(sealed_confirmation, change):
    fixture = sealed_confirmation
    evidence = audit.validate_confirmation_seal(fixture.dataset, fixture.identity)
    job = fixture.plan['jobs'][0]
    expected = evidence['sources'][job['name']]
    receipt = {'identity': {'source': deepcopy(expected['identity']),
                            'seed_schedule_sha256': expected['seed_schedule_sha256'],
                            'model': deepcopy(evidence['models']['05b'])}}
    path = job['output']
    if change == 'output':
        path += '.other'
    elif change == 'source':
        receipt['identity']['source']['path'] += '/other'
    elif change == 'schedule':
        receipt['identity']['seed_schedule_sha256'] = '0' * 64
    else:
        receipt['identity']['model']['path'] += '/other'
    with pytest.raises(ValueError, match='differs from prospective confirmation'):
        audit.verify_receipt_matches_confirmation_seal(receipt, path, evidence, job['domain'], 'baseline')


@pytest.mark.parametrize('change', ['missing_metadata', 'changed_metadata', 'missing_chain_pin', 'changed_chain_pin', 'lowprio', 'missing_amendment'])
def test_confirmation_requires_separate_completed_recovery_chain(sealed_confirmation, change):
    fixture = sealed_confirmation
    if change == 'missing_metadata':
        fixture.seal.pop('development_execution_recovery')
    elif change == 'changed_metadata':
        fixture.seal['development_execution_recovery']['attempts'] = 40960
    elif change == 'missing_chain_pin':
        fixture.seal['files_sha256'].pop(fixture.recovery['path'])
    elif change == 'changed_chain_pin':
        fixture.seal['files_sha256'][fixture.recovery['path']] = '0' * 64
    elif change == 'lowprio':
        fixture.plan['execution']['partition'] = 'lowprio'
    else:
        fixture.plan.pop('prospective_execution_amendment_sha256')
    with pytest.raises(ValueError, match='recovery|amendment'):
        audit.validate_recovery_execution_binding(fixture.seal, fixture.plan)
