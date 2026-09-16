"""Scratch-only fit records and frozen process simulation; no actual science."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(0o600)
    path.write_text(json.dumps(value, sort_keys=True))


def change(path, **updates):
    value = json.loads(Path(path).read_text())
    value.update(updates)
    write(path, value)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('scratch_level5_ready', ROOT/'artifacts/continue_modebench_scale_level5_ready_fits_20260912.py')
    a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a)
    state, view = tmp_path/'state', tmp_path/'view.json'
    write(view, {'scratch': 'view'})
    source = tmp_path/'scientific.json'
    write(source, {'unchanged': True})
    lock = tmp_path/'controller.lock'
    lock.write_text('')
    canonical = tmp_path/'canonical'
    canonical.write_text('neutral')
    for key, value in {'STATE': state, 'VIEW': view, 'PLAN': tmp_path/'original/plan.json',
                       'MANIFEST': tmp_path/'level5/source_manifest.json', 'REVIEW': tmp_path/'review.json',
                       'FIRST_REVIEW': tmp_path/'first_review.json'}.items():
        monkeypatch.setattr(a, key, value)
    monkeypatch.setattr(a.first, 'LOCK', lock)
    monkeypatch.setattr(a.first, 'LOCK_INODE', lock.stat().st_ino)
    cells, sources, certificates, receipts, roots = [None]*5, {}, [], [], []
    for index, domain, raw in a.SCOPE:
        root = tmp_path/domain
        roots.append(root)
        tasks_path = tmp_path/f'tasks{index}.json'
        tasks = []
        for tier in range(4):
            output = root/'level5/results/development'/domain/f'difficulty_{tier}.json'
            write(output, {'level': 'level5', 'domain': domain, 'split': 'dev', 'status': 'complete'})
            tasks.append({'output': str(output)})
            receipts.append(output)
        write(tasks_path, tasks)
        cells[index] = {'level': 'level5', 'domain': domain, 'source_root': str(root), 'tasks': str(tasks_path)}
        sources[domain] = {'source_kind': 'domain_revision_v1', 'source_root': str(root)}
    for domain in ('countdown', 'pantry', 'python_factors'):
        sources[domain] = {'source_kind': 'domain_revision_v1', 'source_root': str(tmp_path/domain)}
    write(a.PLAN, {'cells': cells})
    monkeypatch.setattr(a.BASE, 'REVISED_PLAN_SHA', a.sha(a.PLAN))
    write(a.MANIFEST, {'level': 'level5', 'model_label': '14b', 'sources': sources,
                       'files_sha256': {str(source): a.sha(source)}})
    monkeypatch.setattr(a.BASE, 'SOURCE_PINS', {a.MANIFEST: a.sha(a.MANIFEST)})
    monkeypatch.setattr(a.BASE, 'predecessor_inputs', lambda **kwargs: {str(source): a.sha(source)})
    for index, domain, raw in a.SCOPE:
        certificate = a.PLAN.parent/'execution_reconciliations'/str(index)/'reconciliation.json'
        write(certificate, {'schema': 'modebench_scale_completed_stage_cell_reconciliation_v1',
            'status': 'verified_scientific_outputs_with_failed_execution', 'array_job_id': 31254520,
            'array_index': index, 'job_id_raw': raw, 'job_id': '31254520_'+str(index), 'state': 'FAILED',
            'exit_code': '1:0', 'scheduler_success': False, 'scientific_outputs_complete': True,
            'exit_cause': 'unknown', 'completed_receipts': 4, 'stage_plan_sha256': a.BASE.REVISED_PLAN_SHA,
            'files_sha256': {str(source): a.sha(source)}})
        certificates.append(certificate)
    write(a.FIRST_REVIEW, {'files_sha256': {str(source): a.sha(source)}})
    monkeypatch.setattr(a, 'SEALED', {p: a.sha(p) for p in (source, a.PLAN, a.MANIFEST, a.FIRST_REVIEW)})
    write(a.REVIEW, {'schema': 'modebench_scale_level5_ready_fits_independent_review_v1', 'status': 'reviewed',
        'files_sha256': {str(p): a.sha(p) for p in (a.SOURCE, a.TESTS, *a.SEALED)}})
    calls, verified, exclusions, guards, held, active = [], [], [], [], [], []
    failures, exceptions = set(), set()
    def paths(root):
        return {'recipe': root/'level5/recipes'/(root.name+'.json'), 'dataset': root/'level5/dataset',
                'confirmation': root/'level5/results/confirmation.json'}
    def exclusion(root, level, domain, carried):
        assert level == 'level5' and domain in ('graph_coloring', 'mathir')
        exclusions.append(domain)
        return {str(source): a.sha(source)}
    def verify(stage, index):
        assert stage == 'revision_development' and index in (3, 4)
        verified.append(index)
        return a.read(a.PLAN.parent/'execution_reconciliations'/str(index)/'reconciliation.json')
    def fit(root):
        domain = root.name
        assert held == [8] and (state/'fits'/domain/'claim.json').is_file()
        assert a.read(state/'fits'/domain/'claim.json')['entrypoint'] == str(a.REVISION)+':fit_domain(publish=True)'
        calls.append(domain)
        if domain in exceptions:
            raise ValueError('scratch fitter exception')
        result = {'level': 'level5', 'domain': domain, 'development_fit_pass': domain not in failures,
                  'input_sha256': {str(source): a.sha(source)}}
        write(paths(root)['recipe'], result)
        return result
    def authenticate(path):
        assert path == view and canonical.read_text() == 'neutral'
    def load(path, name):
        if path == a.CORE:
            return SimpleNamespace(revision=SimpleNamespace(paths=paths), validate_revision_pool_exclusions=exclusion)
        if path == a.REVISION:
            return SimpleNamespace(fit_domain=fit)
        if path == a.RUNNER:
            return SimpleNamespace(authenticate=authenticate)
        assert path == a.BASE.CELL_HELPERS['v2']
        return SimpleNamespace(verify_reconciliation=verify)
    monkeypatch.setattr(a, 'load_module', load)
    owner = {'pid': 12345, 'start_ticks': '5678', 'state': 'R', 'uid': a.os.getuid(),
             'command': [str(a.PYTHON), '-B', str(a.SOURCE), 'fit']}
    def identity(pid):
        if active and pid != 12345:
            command = active[0][active[0].index('--')+1:]
            command.insert(1, '-B')
            return {**owner, 'pid': 12346, 'start_ticks': '5679', 'command': command}
        return deepcopy(owner)
    monkeypatch.setattr(a, 'process_identity', identity)
    @contextmanager
    def fence():
        assert not held
        held.append(8)
        try:
            yield 8
        finally:
            held.clear()
    monkeypatch.setattr(a, 'lifetime_fence', fence)
    def assert_fence(fd):
        a.require(fd == 8 and held == [8], 'continuous fence lost')
        guards.append(fd)
    monkeypatch.setattr(a, 'assert_fence', assert_fence)
    def host_guard(*, guest):
        a.require(canonical.read_text() == ('old' if guest else 'neutral'), 'source view changed')
    monkeypatch.setattr(a.first, 'host_guard', host_guard)
    dispatch = {'code': 0, 'drop_result': False, 'drift': None}
    def run(command, **kwargs):
        assert kwargs['pass_fds'] == (8,) and held == [8]
        assert kwargs['env']['OPENBLAS_NUM_THREADS'] == '1'
        active.append(command)
        canonical.write_text('old')
        try:
            if dispatch['drift']:
                change(state/'action/runtime.json', **dispatch['drift'])
            a.guest(state, 8, a.sha(a.REVIEW))
        finally:
            canonical.write_text('neutral')
            active.clear()
        if dispatch['drop_result']:
            (state/'action/result.json').unlink()
        return SimpleNamespace(returncode=dispatch['code'])
    monkeypatch.setattr(a, 'subprocess', SimpleNamespace(run=run))
    return SimpleNamespace(a=a, state=state, source=source, roots=roots, certificates=certificates, receipts=receipts,
        calls=calls, verified=verified, exclusions=exclusions, held=held, guards=guards, failures=failures,
        exceptions=exceptions, dispatch=dispatch, owner=owner, canonical=canonical,
        run=lambda: a.run(state, a.sha(a.REVIEW)))


def test_real_registration_composition_and_two_claimed_fits(fixture):
    f = fixture
    value = f.run()
    assert f.calls == ['graph_coloring', 'mathir']
    assert f.verified == [3, 4] and f.exclusions == f.calls
    assert value['status'] == 'two_level5_development_gates_passed'
    assert f.a.read(f.state/'registration.json')['completed_receipts'] == 8
    assert f.a.verify_existing(f.state) == value and f.calls == ['graph_coloring', 'mathir']
    assert f.guards and set(f.guards) == {8} and not f.held and f.canonical.read_text() == 'neutral'


def test_failed_first_gates_are_saved_and_second_fixed_fit_still_runs(fixture):
    f = fixture
    f.failures.add('graph_coloring')
    result = f.run()
    assert f.calls == ['graph_coloring', 'mathir'] and result['failed_domains'] == ['graph_coloring']
    assert result['status'] == 'needs_new_development_revision' and result['new_grader_invocations'] == 0
    assert not result['confirmation_performed'] and not result['publication_performed']
    assert f.a.verify_existing(f.state) == result


def test_exception_preserves_first_recipe_and_second_claim_without_retry(fixture):
    f = fixture
    f.exceptions.add('mathir')
    with pytest.raises(ValueError, match='scratch fitter exception'):
        f.run()
    assert (f.state/'fits/graph_coloring/result.json').exists()
    assert (f.state/'fits/mathir/claim.json').exists() and not (f.state/'fits/mathir/result.json').exists()
    assert (f.state/'action/failure.json').exists()
    pins = f.a.read(f.state/'action/failure.json')['files_sha256']
    for p in (f.state/'registration.json', f.state/'fits/graph_coloring/result.json',
              f.state/'fits/mathir/claim.json', f.roots[0]/'level5/recipes/graph_coloring.json'):
        assert pins[str(p)] == f.a.sha(p)
    with pytest.raises(ValueError, match='already attempted'):
        f.run()
    assert f.calls == ['graph_coloring', 'mathir']


def test_duplicate_action_never_repeats_native_fit(fixture):
    f = fixture
    f.run()
    with pytest.raises(ValueError, match='already attempted'):
        f.run()
    assert len(f.calls) == 2


@pytest.mark.parametrize('code', [1, 143, -15])
def test_actual_nonzero_outer_exit_is_retained_even_with_all_outputs(fixture, code):
    f = fixture
    f.dispatch['code'] = code
    with pytest.raises(ValueError, match='explicit reconciliation'):
        f.run()
    assert len(f.calls) == 2 and f.a.read(f.state/'action/exit.json')['returncode'] == code
    assert (f.state/'action/result.json').exists() and (f.state/'action/failure.json').exists()
    with pytest.raises(ValueError, match='explicit reconciliation'):
        f.a.verify_existing(f.state)


@pytest.mark.parametrize('field,value', [('state', 'COMPLETED'), ('exit_code', '143:0'),
    ('scheduler_success', True), ('completed_receipts', 3), ('array_index', 2),
    ('job_id_raw', 'wrong'), ('scientific_outputs_complete', False), ('exit_cause', 'assumed')])
def test_exact_original_failed_cell_policy_before_any_fit(fixture, field, value):
    f = fixture
    change(f.certificates[0], **{field: value})
    with pytest.raises(ValueError, match='audited original'):
        f.run()
    assert not f.calls


@pytest.mark.parametrize('kind', ['receipt_missing', 'receipt_wrong_scope', 'recipe_exists', 'dataset_exists', 'holdout_exists'])
def test_preexisting_or_wrong_scientific_inputs_block_registration(fixture, kind):
    f = fixture
    if kind == 'receipt_missing':
        f.receipts[0].unlink()
    elif kind == 'receipt_wrong_scope':
        change(f.receipts[0], level='level4')
    elif kind == 'recipe_exists':
        write(f.roots[0]/'level5/recipes/graph_coloring.json', {'existing': True})
    elif kind == 'dataset_exists':
        (f.roots[0]/'level5/dataset').mkdir(parents=True)
    else:
        write(f.roots[0]/'level5/results/confirmation.json', {'observed': True})
    with pytest.raises((ValueError, FileNotFoundError)):
        f.run()
    assert not f.calls


@pytest.mark.parametrize('drift', [{'host': 'wash.cs.princeton.edu'}, {'start_ticks': 'changed'}, {'fence_fd': 9},
    {'source_sha256': 'changed'}, {'view_manifest_sha256': 'changed'}, {'command': ['other']}, {'lock_inode': 0}])
def test_continuous_truthful_owner_and_source_view_checked_before_guest_work(fixture, drift):
    f = fixture
    f.dispatch['drift'] = drift
    with pytest.raises(ValueError, match='owner/view/fence'):
        f.run()
    assert not f.calls


@pytest.mark.parametrize('kind', ['registration_scope', 'receipt_pin', 'claim', 'recipe', 'guest_command',
    'outer_exit', 'intent', 'aggregate', 'source_drift', 'owner_command', 'view_manifest', 'lock_inode', 'lock_device'])
def test_static_verification_rejects_changed_records_without_any_fit(fixture, kind):
    f = fixture
    f.run()
    if kind in ('registration_scope', 'receipt_pin'):
        value = f.a.read(f.state/'registration.json')
        if kind == 'registration_scope':
            value['revisions'][0]['source_root'] = '/wrong'
        else:
            value['files_sha256'].pop(str(f.receipts[0]))
        write(f.state/'registration.json', value)
        write(f.state/'registration.sha256.json', {'sha256': f.a.sha(f.state/'registration.json')})
    elif kind == 'claim':
        change(f.state/'fits/graph_coloring/claim.json', entrypoint='other fitter')
    elif kind == 'recipe':
        change(f.roots[0]/'level5/recipes/graph_coloring.json', development_fit_pass=False)
    elif kind == 'guest_command':
        change(f.state/'action/guest_runtime.json', command=['wrong'])
    elif kind == 'outer_exit':
        change(f.state/'action/exit.json', returncode=143)
    elif kind == 'intent':
        change(f.state/'action/intent.json', command=['wrong'])
    elif kind == 'aggregate':
        change(f.state/'action/result.json', status='admitted')
    elif kind == 'owner_command':
        change(f.state/'action/runtime.json', command=['other owner'])
    elif kind == 'view_manifest':
        change(f.state/'action/runtime.json', view_manifest_sha256='changed')
    elif kind == 'lock_inode':
        change(f.state/'action/runtime.json', lock_inode=0)
    elif kind == 'lock_device':
        change(f.state/'action/runtime.json', lock_device=-1)
    else:
        f.source.write_text('changed')
    if kind in ('owner_command', 'view_manifest', 'lock_inode', 'lock_device'):
        digest = f.a.sha(f.state/'action/runtime.json')
        change(f.state/'action/intent.json', runtime_sha256=digest)
        change(f.state/'action/guest_runtime.json', outer_runtime_sha256=digest)
    with pytest.raises(ValueError):
        f.a.verify_existing(f.state)
    assert len(f.calls) == 2


def test_missing_guest_result_preserves_outputs_without_retry(fixture):
    f = fixture
    f.dispatch['drop_result'] = True
    with pytest.raises(FileNotFoundError):
        f.run()
    assert (f.state/'action/failure.json').exists() and len(f.calls) == 2


def test_explicit_review_and_no_future_result_cycle(fixture):
    f = fixture
    with pytest.raises(ValueError, match='explicit reviewed'):
        f.a.run(f.state, 'wrong')
    value = f.a.read(f.a.REVIEW)
    value['files_sha256'][str(f.state/'action/result.json')] = 'future'
    write(f.a.REVIEW, value)
    with pytest.raises(ValueError, match='future-result cycles'):
        f.run()
    assert not f.calls


def test_first_review_transitive_dependency_must_remain_in_new_review(fixture):
    f = fixture
    dependency = f.state.parent/'first_transitive_dependency.py'
    dependency.write_text('historical implementation')
    value = f.a.read(f.a.FIRST_REVIEW)
    value['files_sha256'][str(dependency)] = f.a.sha(dependency)
    write(f.a.FIRST_REVIEW, value)
    f.a.SEALED[f.a.FIRST_REVIEW] = f.a.sha(f.a.FIRST_REVIEW)
    value = f.a.read(f.a.REVIEW)
    value['files_sha256'][str(f.a.FIRST_REVIEW)] = f.a.sha(f.a.FIRST_REVIEW)
    write(f.a.REVIEW, value)
    with pytest.raises(ValueError, match='source/tests/sealed primitives'):
        f.run()
    assert not f.calls
