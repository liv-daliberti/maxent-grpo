"""Exercise archive retirement on temporary files only, without Hub or Slurm."""
from contextlib import contextmanager, nullcontext
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('archive_completed_models_under_test', ROOT / 'ops/archive_completed_models.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
COMMIT = 'a' * 40


class MetadataApi:
    """Return exact immutable metadata for the fixture's uploaded files."""
    token = 'fake-token-for-temporary-tests'

    def __init__(self, manifest):
        self.manifest = manifest
        self.calls = []

    def get_paths_info(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs['revision'] == COMMIT
        return [SimpleNamespace(path=f['path_in_repo'], size=f['size'],
                    lfs=SimpleNamespace(sha256=f['sha256'], size=f['size']),
                    blob_id=f['git_blob_sha1'])
                for f in self.manifest['files'] if f['path_in_repo'] in kwargs['paths']]


@pytest.fixture
def case(tmp_path, monkeypatch):
    data = tmp_path / 'var/data'
    run = data / 'xdr_fixture_e119_pantry_maxrl_s43'
    attempt = run / 'debug_job123'
    export = attempt / 'saved_models/step_03073'
    export.mkdir(parents=True)
    weights = {'model-00001-of-00002.safetensors': b'first trained shard' * 17,
               'model-00002-of-00002.safetensors': b'second trained shard' * 19}
    metadata = {'config.json': b'{"model_type":"qwen2"}\n',
                'tokenizer_config.json': b'{"tokenizer_class":"AutoTokenizer"}\n',
                'tokenizer.json': b'{"version":"1.0"}\n',
                'model.safetensors.index.json': b'{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}\n'}
    for name, content in {**weights, **metadata}.items():
        (export / name).write_bytes(content)
    (attempt / 'train_metrics.jsonl').write_text('{"trainer/global_step":3072}\n')
    (attempt / 'eval_mode_coverage_draws.jsonl').write_text('{"step":3072}\n')
    receipt = run / 'TRAINING_COMPLETE.json'
    receipt.write_text(json.dumps({'schema': 'oat_zero_training_complete_v1',
        'terminal_step': 3073, 'terminal_attempt': str(attempt),
        'terminal_export': str(export)}) + '\n')
    ledger = tmp_path / 'ledger.json'
    ledger.write_text('{"runs":[{"job_id":123}]}\n')
    out = tmp_path / 'archive'
    state = out / 'models/fixture'
    state.mkdir(parents=True)
    monkeypatch.setattr(a, 'ROOT', tmp_path)
    monkeypatch.setattr(a, 'DATA_ROOT', data)
    monkeypatch.setattr(a, 'SHARED_LOCK', tmp_path / 'storage.lock')
    monkeypatch.setattr(a, 'storage_lock', lambda: nullcontext())
    monkeypatch.setattr(a, 'queue_snapshot', lambda: [])
    prefix = 'experiments/E119/Qwen2.5-0.5B-Instruct/pantry_plan/maxrl/seed-43/step-03073'
    record = {'audited_candidate': True, 'endpoint_status': 'admitted',
        'run_dir': str(run), 'run_stamp': 'e119_pantry_maxrl_s43',
        'terminal_export': str(export), 'receipt_path': str(receipt),
        'receipt_sha256': a.digest(receipt), 'terminal_step': 3073,
        'source_experiment': 'e119', 'model_key': 'qwen05b',
        'source_arm': 'maxrl', 'domain': 'pantry_plan', 'seed': 43,
        'campaign': ['e119'], 'all_related_job_ids': [123, 456],
        'large_weight_files': list(weights), 'repo_prefix': prefix,
        'archive_id': 'fixture', 'terminal_bytes': sum(len(x) for x in {**weights, **metadata}.values())}
    manifest = a.verify.build_manifest(export, prefix)
    remote = a.verify.verify_remote_manifest(MetadataApi(manifest), 'owner/archive', COMMIT, manifest)
    proof = {'schema': 'verified-model-archive-v1', 'repo_id': 'owner/archive',
        'commit_sha': COMMIT, 'remote_metadata': remote,
        'remote_bytes': {'status': 'verified', 'commit_sha': COMMIT, 'repo_id': 'owner/archive',
            'files': [{'path': f['path_in_repo'], 'bytes': f['size'], 'sha256': f['sha256']}
                      for f in manifest['files'] if f['relative_path'] in weights]}}
    plan = {'repo_id': 'owner/archive', 'ledger_pins': {str(ledger): a.digest(ledger)},
            'output_dir': str(out), 'models': [record], 'private': False}
    a.save(state / 'manifest.json', manifest)
    proof['manifest_sha256'] = a.digest(state / 'manifest.json')
    a.save(state / 'verification.json', proof)
    a.save(state / 'upload.json', {'commit_sha': COMMIT, 'repo_id': plan['repo_id']})
    return SimpleNamespace(run=run, attempt=attempt, export=export, receipt=receipt,
        ledger=ledger, state=state, plan=plan, record=record, manifest=manifest,
        proof=proof, weights=weights, metadata=metadata)


def retirement(case):
    return a.retire_weights(case.plan, case.record, case.manifest, case.proof, case.state)


def assert_weights_present(case):
    for name, content in case.weights.items():
        assert (case.export / name).read_bytes() == content
    assert not (case.state / 'retired.json').exists()


@pytest.mark.parametrize('gate', ['remote_metadata', 'remote_bytes'])
@pytest.mark.parametrize('status', ['failed', 'pending', None])
def test_no_deletion_before_both_remote_gates(case, gate, status):
    case.proof[gate]['status'] = status
    with pytest.raises(RuntimeError): retirement(case)
    assert_weights_present(case)
    assert not (case.state / 'retirement_intent.json').exists()


@pytest.mark.parametrize('location', ['commit_sha', 'remote_metadata', 'remote_bytes'])
def test_commit_binding_mismatch_prevents_deletion(case, location):
    if location == 'commit_sha': case.proof[location] = 'main'
    else: case.proof[location]['commit_sha'] = 'b' * 40
    with pytest.raises(RuntimeError): retirement(case)
    assert_weights_present(case)


@pytest.mark.parametrize('change', ['path', 'sha256', 'bytes', 'missing'])
def test_downloaded_byte_proof_covers_exact_weight_paths(case, change):
    files = case.proof['remote_bytes']['files']
    if change == 'missing': files.pop()
    elif change == 'path': files[0]['path'] = 'another/export/model.safetensors'
    elif change == 'sha256': files[0]['sha256'] = 'f' * 64
    else: files[0]['bytes'] += 1
    with pytest.raises(RuntimeError): retirement(case)
    assert_weights_present(case)


@pytest.mark.parametrize('change', ['proof_repo', 'metadata_repo', 'metadata_type', 'metadata_omission', 'metadata_wrong_path', 'prefix'])
def test_metadata_and_repository_identity_must_match_full_export(case, change):
    if change == 'proof_repo': case.proof['repo_id'] = 'other/archive'
    elif change == 'metadata_repo': case.proof['remote_metadata']['repo_id'] = 'other/archive'
    elif change == 'metadata_type': case.proof['remote_metadata']['repo_type'] = 'dataset'
    elif change == 'metadata_omission': case.proof['remote_metadata']['verified_files'].pop()
    elif change == 'metadata_wrong_path': case.proof['remote_metadata']['verified_files'][0]['path'] = 'other/config.json'
    else: case.record['repo_prefix'] = 'other/export'
    with pytest.raises(RuntimeError): retirement(case)
    assert_weights_present(case)


@pytest.mark.parametrize('kind', ['receipt_content', 'ledger_content', 'weight_content', 'weight_replacement', 'new_file', 'metadata_content'])
def test_changed_local_inputs_prevent_retirement(case, kind):
    if kind == 'receipt_content': case.receipt.write_text('{}\n')
    elif kind == 'ledger_content': case.ledger.write_text('{"changed":true}\n')
    elif kind == 'weight_content': (case.export / next(iter(case.weights))).write_bytes(b'changed')
    elif kind == 'weight_replacement':
        p = case.export / next(iter(case.weights)); raw = p.read_bytes(); p.unlink(); p.write_bytes(raw)
    elif kind == 'new_file': (case.export / 'new-metadata.json').write_text('{}')
    else: (case.export / 'config.json').write_text('{"changed":true}\n')
    with pytest.raises((RuntimeError, a.verify.ArchiveVerificationError)): retirement(case)
    assert all((case.export / name).exists() for name in case.weights)
    assert not (case.state / 'retirement_intent.json').exists()


@pytest.mark.parametrize('kind', ['wrong_receipt_path', 'wrong_schema', 'wrong_attempt'])
def test_receipt_must_belong_to_exact_canonical_completed_run(case, tmp_path, kind):
    payload = json.loads(case.receipt.read_text())
    if kind == 'wrong_receipt_path':
        other = tmp_path / 'other_receipt.json'; other.write_text(case.receipt.read_text())
        case.record['receipt_path'] = str(other)
    else:
        if kind == 'wrong_schema': payload['schema'] = 'unrelated_receipt'
        else: payload['terminal_attempt'] = str(tmp_path / 'another-attempt')
        case.receipt.write_text(json.dumps(payload))
        case.record['receipt_sha256'] = a.digest(case.receipt)
    with pytest.raises(RuntimeError): retirement(case)
    assert_weights_present(case)


@pytest.mark.parametrize('state,reason', [('RUNNING', 'None'), ('PENDING', 'Priority'), ('PENDING', 'JobHeldUser'), ('PENDING', 'JobHeldAdmin')])
def test_any_related_queued_job_blocks_even_deliberate_holds(case, monkeypatch, state, reason):
    monkeypatch.setattr(a, 'queue_snapshot', lambda: [{'job_id': 456, 'job_state': [state], 'state_reason': reason}])
    with pytest.raises(RuntimeError, match='queued|consumer|active'): retirement(case)
    assert_weights_present(case)


@pytest.mark.parametrize('field,value_kind', [('submit_line', 'weight'), ('command', 'export'), ('current_working_directory', 'run'), ('name', 'stamp')])
def test_new_job_id_with_model_path_or_run_stamp_blocks_cleanup(case, monkeypatch, field, value_kind):
    value = {'weight': '--pretrain=' + str(case.export / next(iter(case.weights))),
             'export': 'evaluate --model ' + str(case.export),
             'run': str(case.run), 'stamp': 'eval-' + case.record['run_stamp']}[value_kind]
    monkeypatch.setattr(a, 'queue_snapshot', lambda: [{'job_id': 999, field: value}])
    with pytest.raises(RuntimeError, match='consumer|queued'): retirement(case)
    assert_weights_present(case)


def test_scheduler_failure_cannot_enable_deletion(case, monkeypatch):
    def failed(): raise TimeoutError('mock Slurm unavailable')
    monkeypatch.setattr(a, 'queue_snapshot', failed)
    with pytest.raises(TimeoutError): retirement(case)
    assert_weights_present(case)


def test_second_ledger_check_catches_change_while_waiting_for_lock(case, monkeypatch):
    @contextmanager
    def changed():
        case.ledger.write_text('{"new_successor":999}')
        yield
    monkeypatch.setattr(a, 'storage_lock', changed)
    with pytest.raises(RuntimeError, match='mapping|ledger'): retirement(case)
    assert_weights_present(case)


def test_weight_replacement_after_hashing_is_rejected_inside_lock(case, monkeypatch):
    @contextmanager
    def changed():
        p = case.export / next(iter(case.weights)); raw = p.read_bytes(); p.unlink(); p.write_bytes(raw)
        yield
    monkeypatch.setattr(a, 'storage_lock', changed)
    with pytest.raises(RuntimeError, match='changed'): retirement(case)
    assert_weights_present(case)


def test_weights_only_retirement_preserves_all_local_research_metadata(case):
    receipt_before = case.receipt.read_bytes()
    retained = {p: p.read_bytes() for p in [case.receipt, case.ledger, case.attempt / 'train_metrics.jsonl',
                case.attempt / 'eval_mode_coverage_draws.jsonl', *[case.export / n for n in case.metadata]]}
    result = retirement(case)
    assert result['status'] == 'retired'
    assert result['removed_bytes'] == sum(map(len, case.weights.values()))
    assert {r['relative_path'] for r in result['removed_files']} == set(case.weights)
    assert all(not (case.export / name).exists() for name in case.weights)
    assert case.run.is_dir() and case.attempt.is_dir() and case.export.is_dir()
    assert all(p.read_bytes() == content for p, content in retained.items())
    assert case.receipt.read_bytes() == receipt_before
    receipt = json.loads((case.run / 'MODEL_ARCHIVE.json').read_text())
    assert receipt['commit_sha'] == COMMIT and receipt['repo_id'] == case.plan['repo_id']
    assert receipt['manifest_sha256'] == a.digest(case.state / 'manifest.json')
    assert (case.state / 'retirement_intent.json').exists()
    assert len(json.loads((case.state / 'retirement_progress.json').read_text())['removed']) == 2


def test_metadata_name_can_never_enter_weight_deletion_allowlist(case):
    case.record['large_weight_files'].append('config.json')
    with pytest.raises(RuntimeError, match='weights|allowlist'): retirement(case)
    assert_weights_present(case)


def test_keep_local_archive_does_not_call_retirement(case, monkeypatch):
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(CommitOperationAdd=lambda **kw: SimpleNamespace(**kw)))
    a.save(case.state / 'upload.json', {'commit_sha': COMMIT, 'repo_id': case.plan['repo_id']})
    def forbidden(*args, **kwargs): raise AssertionError('keep-local attempted deletion')
    monkeypatch.setattr(a, 'retire_weights', forbidden)
    result = a.archive_one(MetadataApi(case.manifest), case.plan, case.record, keep_local=True)
    assert result['commit_sha'] == COMMIT
    assert_weights_present(case)


def test_stream_remote_weights_hashes_real_mock_download_bytes(case, monkeypatch):
    urls = []
    def url(repo, filename, **kwargs):
        assert repo == case.plan['repo_id'] and kwargs['revision'] == COMMIT
        urls.append(filename)
        return filename
    class Response:
        def __init__(self, raw): self.raw = raw
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, chunk_size):
            for i in range(0, len(self.raw), 5): yield self.raw[i:i+5]
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(hf_hub_url=url))
    monkeypatch.setitem(sys.modules, 'requests', SimpleNamespace(get=lambda path, **kwargs: Response(case.weights[Path(path).name])))
    result = a.stream_remote_weights(MetadataApi(case.manifest), case.plan['repo_id'], COMMIT, case.manifest, set(case.weights))
    assert result['status'] == 'verified' and len(urls) == 2
    assert {x['sha256'] for x in result['files']} == {hashlib.sha256(x).hexdigest() for x in case.weights.values()}


def test_corrupt_download_never_becomes_verified(case, monkeypatch):
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(hf_hub_url=lambda *args, **kwargs: 'mock-url'))
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, chunk_size): yield b'corrupted remote bytes'
    monkeypatch.setitem(sys.modules, 'requests', SimpleNamespace(get=lambda *args, **kwargs: Response()))
    with pytest.raises(RuntimeError, match='differs|exceeds'):
        a.stream_remote_weights(MetadataApi(case.manifest), case.plan['repo_id'], COMMIT, case.manifest, set(case.weights))
    assert_weights_present(case)


def test_archive_stops_before_download_or_retirement_when_remote_metadata_fails(case, monkeypatch):
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(CommitOperationAdd=lambda **kw: SimpleNamespace(**kw)))
    (case.state / 'verification.json').unlink()
    api = MetadataApi(case.manifest)
    api.get_paths_info = lambda **kwargs: []
    def forbidden(*args, **kwargs): raise AssertionError('metadata failure reached a later stage')
    monkeypatch.setattr(a, 'stream_remote_weights', forbidden)
    monkeypatch.setattr(a, 'retire_weights', forbidden)
    with pytest.raises(RuntimeError, match='metadata'):
        a.archive_one(api, case.plan, case.record, keep_local=False)
    assert_weights_present(case)
    assert not (case.state / 'verification.json').exists()


def test_archive_stops_before_retirement_when_download_fails(case, monkeypatch):
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(CommitOperationAdd=lambda **kw: SimpleNamespace(**kw)))
    (case.state / 'verification.json').unlink()
    def failed(*args, **kwargs): raise RuntimeError('downloaded remote weight differs')
    def forbidden(*args, **kwargs): raise AssertionError('download failure reached retirement')
    monkeypatch.setattr(a, 'stream_remote_weights', failed)
    monkeypatch.setattr(a, 'retire_weights', forbidden)
    with pytest.raises(RuntimeError, match='downloaded'):
        a.archive_one(MetadataApi(case.manifest), case.plan, case.record, keep_local=False)
    assert_weights_present(case)
    assert not (case.state / 'verification.json').exists()


@pytest.mark.parametrize('change', ['repo', 'commit'])
def test_saved_upload_identity_must_match_verification_before_retirement(case, change):
    upload = {'repo_id': case.plan['repo_id'], 'commit_sha': COMMIT}
    upload['repo_id' if change == 'repo' else 'commit_sha'] = 'another/repo' if change == 'repo' else 'b' * 40
    a.save(case.state / 'upload.json', upload)
    with pytest.raises(RuntimeError, match='upload|identity'): retirement(case)
    assert_weights_present(case)


def test_changed_stored_manifest_invalidates_previous_remote_proof(case):
    stored = copy.deepcopy(case.manifest)
    stored['unexpected_change'] = True
    a.save(case.state / 'manifest.json', stored)
    with pytest.raises(RuntimeError, match='manifest'): retirement(case)
    assert_weights_present(case)


def interrupt_after_first_unlink(case, monkeypatch):
    original = a.save
    def fail_before_progress(path, value):
        if Path(path).name == 'retirement_progress.json':
            raise RuntimeError('simulated interruption after unlink before progress receipt')
        return original(path, value)
    with monkeypatch.context() as scoped:
        scoped.setattr(a, 'save', fail_before_progress)
        with pytest.raises(RuntimeError, match='simulated interruption'):
            retirement(case)
    assert (case.state / 'retirement_intent.json').is_file()
    assert not (case.state / 'retired.json').exists()
    assert sum((case.export / name).exists() for name in case.weights) == 1


def test_interrupted_unlink_recovers_under_same_verified_intent(case, monkeypatch):
    retained = {case.export / n: content for n, content in case.metadata.items()}
    interrupt_after_first_unlink(case, monkeypatch)
    result = retirement(case)
    assert result['status'] == 'retired'
    assert len(result['removed_files']) == len(case.weights)
    assert result['removed_bytes'] == sum(map(len, case.weights.values()))
    assert all(not (case.export / name).exists() for name in case.weights)
    assert all(p.read_bytes() == content for p, content in retained.items())


@pytest.mark.parametrize('key', ['repo_id', 'commit_sha', 'repo_prefix', 'manifest_sha256', 'retired_files'])
def test_interrupted_retirement_rejects_different_prior_intent(case, monkeypatch, key):
    interrupt_after_first_unlink(case, monkeypatch)
    p = case.state / 'retirement_intent.json'
    previous = json.loads(p.read_text())
    previous[key] = ['unrelated.safetensors'] if key == 'retired_files' else 'unrelated'
    a.save(p, previous)
    before = {name: (case.export / name).exists() for name in case.weights}
    with pytest.raises(RuntimeError, match='intent'): retirement(case)
    assert before == {name: (case.export / name).exists() for name in case.weights}


def test_missing_weight_without_owned_intent_never_retires_other_shard(case):
    missing = next(iter(case.weights))
    (case.export / missing).unlink()
    with pytest.raises(RuntimeError, match='missing|intent'): retirement(case)
    assert all((case.export / name).exists() for name in case.weights if name != missing)


def test_consumer_arriving_before_partial_recovery_preserves_remaining_weight(case, monkeypatch):
    interrupt_after_first_unlink(case, monkeypatch)
    monkeypatch.setattr(a, 'queue_snapshot', lambda: [{'job_id': 999, 'submit_line': 'eval --pretrain=' + str(case.export)}])
    with pytest.raises(RuntimeError, match='consumer'): retirement(case)
    assert sum((case.export / name).exists() for name in case.weights) == 1


def test_partial_recovery_never_excuses_missing_retained_metadata(case, monkeypatch):
    interrupt_after_first_unlink(case, monkeypatch)
    (case.export / 'tokenizer.json').unlink()
    with pytest.raises(RuntimeError, match='changed'): retirement(case)
    assert sum((case.export / name).exists() for name in case.weights) == 1


@pytest.fixture
def worker_run(case, monkeypatch):
    output = Path(case.plan['output_dir'])
    models = [{**case.record, 'archive_id': f'fixture-{i}', 'repo_prefix': f'model-{i}'} for i in range(7)]
    plan = {**case.plan, 'models': models}
    path = output / 'plan.json'
    a.save(path, plan)
    path.with_suffix('.sha256').write_text(a.digest(path) + '\n')
    token = output / 'fake-token'
    token.write_text('only-a-mocked-test-token')
    api = SimpleNamespace(model_info=lambda repo_id: SimpleNamespace(private=False))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(HfApi=lambda **kwargs: api))
    monkeypatch.setattr(a, 'publish_catalog', lambda *args: None)
    monkeypatch.setattr(a, 'catalog', lambda p: {'models': []})
    return SimpleNamespace(plan=plan, path=path, token=token, output=output, models=models)


@pytest.mark.parametrize('workers', [1, 2, 3, 4, 5, 6])
def test_worker_pool_never_exceeds_authorized_concurrency(worker_run, monkeypatch, workers):
    import threading
    import time
    called, active, peak = [], 0, 0
    lock = threading.Lock()
    def archive(api, plan, record, *, keep_local):
        nonlocal active, peak
        assert keep_local is True
        with lock:
            called.append(record['archive_id']); active += 1; peak = max(peak, active)
        time.sleep(.02)
        with lock: active -= 1
        return {'status': 'verified'}
    monkeypatch.setattr(a, 'archive_one', archive)
    a.run(worker_run.path, worker_run.token, limit=0, keep_local=True, workers=workers)
    assert sorted(called) == sorted(r['archive_id'] for r in worker_run.models)
    assert 1 <= peak <= workers and active == 0
    assert json.loads((worker_run.output / 'status.json').read_text())['status'] == 'batch_finished'


def test_worker_limit_restricts_total_models_before_batch_submission(worker_run, monkeypatch):
    called = []
    monkeypatch.setattr(a, 'archive_one', lambda api, plan, record, **kwargs: called.append(record['archive_id']))
    a.run(worker_run.path, worker_run.token, limit=2, keep_local=True, workers=3)
    assert sorted(called) == ['fixture-0', 'fixture-1']


@pytest.mark.parametrize('workers', [0, 7, -1, True, 1.5, '2'])
def test_invalid_worker_limits_never_start_archive(worker_run, monkeypatch, workers):
    def forbidden(*args, **kwargs): raise AssertionError('invalid worker count reached archive')
    monkeypatch.setattr(a, 'archive_one', forbidden)
    with pytest.raises(RuntimeError, match='workers'):
        a.run(worker_run.path, worker_run.token, limit=0, keep_local=True, workers=workers)
    assert not (worker_run.output / 'status.json').exists()


def test_observed_worker_failure_stops_refilling_and_drains_inflight(worker_run, monkeypatch):
    import threading
    barrier = threading.Barrier(3)
    failure_observed = threading.Event()
    monkeypatch.setattr(a, 'emit', lambda event, **fields: failure_observed.set() if event == 'stopped_model' else None)
    started, finished = [], []
    lock = threading.Lock()
    def archive(api, plan, record, **kwargs):
        with lock: started.append(record['archive_id'])
        barrier.wait(timeout=3)
        if record['archive_id'] == 'fixture-0':
            raise RuntimeError('simulated model verification failure')
        assert failure_observed.wait(3), 'remaining workers must drain after failure is observed'
        with lock: finished.append(record['archive_id'])
        return {'status': 'verified'}
    monkeypatch.setattr(a, 'archive_one', archive)
    with pytest.raises(SystemExit) as error:
        a.run(worker_run.path, worker_run.token, limit=0, keep_local=True, workers=3)
    assert error.value.code == 1
    assert sorted(started) == ['fixture-0', 'fixture-1', 'fixture-2']
    assert sorted(finished) == ['fixture-1', 'fixture-2']
    status = json.loads((worker_run.output / 'status.json').read_text())
    assert status['status'] == 'stopped' and len(status['failures']) == 1


def test_catalog_publication_failure_stops_refilling(worker_run, monkeypatch):
    called = []
    monkeypatch.setattr(a, 'archive_one', lambda api, plan, record, **kwargs: called.append(record['archive_id']))
    def failed(*args): raise RuntimeError('simulated catalog failure')
    monkeypatch.setattr(a, 'publish_catalog', failed)
    with pytest.raises(SystemExit):
        a.run(worker_run.path, worker_run.token, limit=0, keep_local=True, workers=2)
    assert sorted(called) == ['fixture-0', 'fixture-1']


def test_verified_keep_local_and_already_retired_models_are_not_resubmitted(worker_run, monkeypatch):
    for i, name in [(0, 'verification.json'), (1, 'retired.json')]:
        a.save(worker_run.output / 'models' / f'fixture-{i}' / name, {'existing': True, 'removed_bytes': 0})
    called = []
    monkeypatch.setattr(a, 'archive_one', lambda api, plan, record, **kwargs: called.append(record['archive_id']))
    a.run(worker_run.path, worker_run.token, limit=2, keep_local=True, workers=3)
    assert sorted(called) == ['fixture-2', 'fixture-3']


@pytest.fixture
def large_download(case, monkeypatch, tmp_path):
    import shutil
    scratch = tmp_path / 'bounded-downloads'
    scratch.mkdir()
    preserved = scratch / 'unrelated-local-evidence.txt'
    preserved.write_text('must remain')
    monkeypatch.setattr(a, 'DOWNLOAD_ROOT', scratch)
    monkeypatch.setattr(shutil, 'disk_usage', lambda path: SimpleNamespace(free=3 * 1024**3))
    item = next(f for f in case.manifest['files'] if f['relative_path'] in case.weights)
    return SimpleNamespace(item=item, content=case.weights[item['relative_path']], scratch=scratch, preserved=preserved)


def mocked_large_sdk(case, large_download, monkeypatch, *, mutation=None):
    calls = []
    def download(**kwargs):
        assert kwargs['repo_id'] == case.plan['repo_id']
        assert kwargs['revision'] == COMMIT and kwargs['repo_type'] == 'model'
        assert kwargs['filename'] == large_download.item['path_in_repo']
        folder = Path(kwargs['local_dir'])
        assert folder.parent == large_download.scratch
        target = folder / kwargs['filename']
        target.parent.mkdir(parents=True, exist_ok=True)
        calls.append((folder, target))
        raw = large_download.content
        if mutation == 'size': raw += b'x'
        if mutation == 'hash': raw = b'x' * len(raw)
        if mutation == 'outside': return str(case.export / large_download.item['relative_path'])
        if mutation == 'symlink':
            target.symlink_to(case.export / large_download.item['relative_path'])
            return str(target)
        target.write_bytes(raw)
        if mutation == 'exception':
            (folder / 'sdk-partial.tmp').write_bytes(b'partial download')
            raise RuntimeError('mock SDK failed after partial scratch write')
        return str(target)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(hf_hub_download=download))
    return calls


def assert_only_owned_scratch_was_cleaned(case, large_download, calls):
    assert large_download.preserved.read_text() == 'must remain'
    assert all(not folder.exists() for folder, target in calls)
    assert not list(large_download.scratch.glob('verification-*'))
    assert_weights_present(case)


def test_large_download_hashes_full_pinned_bytes_then_cleans_scratch(case, large_download, monkeypatch):
    calls = mocked_large_sdk(case, large_download, monkeypatch)
    result = a.verify_large_remote_weight(MetadataApi(case.manifest), case.plan['repo_id'], COMMIT, large_download.item)
    assert result == {'path': large_download.item['path_in_repo'], 'bytes': len(large_download.content),
                      'sha256': hashlib.sha256(large_download.content).hexdigest()}
    assert len(calls) == 1
    assert_only_owned_scratch_was_cleaned(case, large_download, calls)


@pytest.mark.parametrize('mutation', ['size', 'hash', 'exception', 'outside', 'symlink'])
def test_failed_large_download_cleans_owned_scratch_but_never_originals(case, large_download, monkeypatch, mutation):
    calls = mocked_large_sdk(case, large_download, monkeypatch, mutation=mutation)
    with pytest.raises(RuntimeError):
        a.verify_large_remote_weight(MetadataApi(case.manifest), case.plan['repo_id'], COMMIT, large_download.item)
    assert_only_owned_scratch_was_cleaned(case, large_download, calls)


def test_large_download_refuses_insufficient_free_space_before_sdk_call(case, large_download, monkeypatch):
    import shutil
    calls = mocked_large_sdk(case, large_download, monkeypatch)
    monkeypatch.setattr(shutil, 'disk_usage', lambda path: SimpleNamespace(free=large_download.item['size'] + 2 * 1024**3 - 1))
    with pytest.raises(RuntimeError, match='space'):
        a.verify_large_remote_weight(MetadataApi(case.manifest), case.plan['repo_id'], COMMIT, large_download.item)
    assert not calls
    assert_only_owned_scratch_was_cleaned(case, large_download, calls)


def test_parallel_large_verification_materializes_one_weight_at_a_time(case, large_download, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time
    active, peak = 0, 0
    called = []
    guard = threading.Lock()
    def download(**kwargs):
        nonlocal active, peak
        folder = Path(kwargs['local_dir'])
        with guard:
            active += 1; peak = max(peak, active); called.append(folder)
        target = folder / kwargs['filename']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(large_download.content)
        time.sleep(.02)
        with guard: active -= 1
        return str(target)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(hf_hub_download=download))
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: a.verify_large_remote_weight(MetadataApi(case.manifest), case.plan['repo_id'], COMMIT, large_download.item), range(3)))
    assert peak == 1 and active == 0
    assert len(set(called)) == 3 and all(not p.exists() for p in called)
    assert all(r['sha256'] == large_download.item['sha256'] for r in results)
    assert_weights_present(case)


def test_large_weights_take_bounded_download_path_instead_of_http_stream(case, monkeypatch):
    item = copy.deepcopy(next(f for f in case.manifest['files'] if f['relative_path'] in case.weights))
    item['size'] = 64 * 1024**2
    manifest = {'files': [item]}
    calls = []
    def large(api, repo, commit, current):
        calls.append(current)
        return {'path': current['path_in_repo'], 'bytes': current['size'], 'sha256': current['sha256']}
    def forbidden(*args, **kwargs): raise AssertionError('large weight used serial HTTP path')
    monkeypatch.setattr(a, 'verify_large_remote_weight', large)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(hf_hub_url=forbidden))
    monkeypatch.setitem(sys.modules, 'requests', SimpleNamespace(get=forbidden))
    result = a.stream_remote_weights(MetadataApi(case.manifest), case.plan['repo_id'], COMMIT, manifest, {item['relative_path']})
    assert len(calls) == 1 and result['status'] == 'verified'
    assert_weights_present(case)


def test_rolling_worker_starts_replacement_before_slow_initial_worker_finishes(worker_run, monkeypatch):
    import threading
    replacement_started = threading.Event()
    events = []
    guard = threading.Lock()
    def archive(api, plan, record, **kwargs):
        key = record['archive_id']
        with guard: events.append(('start', key))
        if key == 'fixture-0':
            assert replacement_started.wait(3), 'batch barrier prevented replacement work'
        elif key == 'fixture-2':
            replacement_started.set()
        with guard: events.append(('finish', key))
        return {'status': 'verified'}
    monkeypatch.setattr(a, 'archive_one', archive)
    a.run(worker_run.path, worker_run.token, limit=0, keep_local=True, workers=2)
    assert events.index(('start', 'fixture-2')) < events.index(('finish', 'fixture-0'))
    assert len([event for event in events if event[0] == 'finish']) == 7


def test_all_initial_completed_futures_are_checked_before_any_refill(worker_run, monkeypatch):
    import concurrent.futures as futures
    submitted = []
    class Executor:
        def __init__(self, max_workers): assert max_workers == 3
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, fn, api, plan, record, **kwargs):
            submitted.append(record['archive_id'])
            future = futures.Future()
            if record['archive_id'] == 'fixture-1': future.set_exception(RuntimeError('second completed result failed'))
            else: future.set_result({'status': 'verified'})
            return future
    # Deterministically present a success before a failure in the same completed
    # set. Refilling after each successful result would wrongly submit fixture-3.
    def ready(current, return_when):
        assert return_when is futures.FIRST_COMPLETED
        assert all(f.done() for f in current)
        return list(current), []
    monkeypatch.setattr(futures, 'ThreadPoolExecutor', Executor)
    monkeypatch.setattr(futures, 'wait', ready)
    with pytest.raises(SystemExit):
        a.run(worker_run.path, worker_run.token, limit=0, keep_local=True, workers=3)
    assert submitted == ['fixture-0', 'fixture-1', 'fixture-2']


def test_failure_completing_during_catalog_publication_is_checked_before_refill(worker_run, monkeypatch):
    import concurrent.futures as futures
    submitted, outstanding = [], []
    class Executor:
        def __init__(self, max_workers): assert max_workers == 2
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, fn, api, plan, record, **kwargs):
            submitted.append(record['archive_id'])
            future = futures.Future()
            if record['archive_id'] == 'fixture-0': future.set_result({'status': 'verified'})
            outstanding.append(future)
            return future
    def ready(current, return_when):
        return {current[0]}, {current[1]}
    def publishing(api, plan):
        # A different model fails while the coordinator is publishing the first.
        outstanding[1].set_exception(RuntimeError('late failure already completed before refill'))
    monkeypatch.setattr(futures, 'ThreadPoolExecutor', Executor)
    monkeypatch.setattr(futures, 'wait', ready)
    monkeypatch.setattr(a, 'publish_catalog', publishing)
    with pytest.raises(SystemExit):
        a.run(worker_run.path, worker_run.token, limit=0, keep_local=True, workers=2)
    assert submitted == ['fixture-0', 'fixture-1']


class CatalogHttpError(Exception):
    def __init__(self, status=429, headers=None):
        super().__init__('https://signed.invalid/?token=DO_NOT_LOG')
        self.response = SimpleNamespace(status_code=status, headers=headers or {})


@pytest.mark.parametrize('headers,expected', [
    ({'Retry-After': '15'}, 17),
    ({'retry-after': '0'}, 2),
    ({'RateLimit': '"api";r=0;t=34'}, 36),
    ({'RateLimit': '"api";r=0;t=34, "commits";r=0;t=240'}, 242),
    ({'Retry-After': '15', 'RateLimit': '"api";r=0;t=240'}, 242),
    ({'Retry-After': 'invalid'}, 300),
    ({'Retry-After': 'NaN'}, 300),
    ({'Retry-After': 'inf'}, 300),
    ({}, 300),
])
def test_catalog_retry_honors_server_reset_or_conservative_fallback(headers, expected):
    assert a.catalog_retry_delay(CatalogHttpError(headers=headers)) == expected


def test_catalog_retry_accepts_http_date(monkeypatch):
    monkeypatch.setattr(a.time, 'time', lambda: 0)
    error = CatalogHttpError(headers={'Retry-After': 'Thu, 01 Jan 1970 00:02:00 GMT'})
    assert a.catalog_retry_delay(error) == 122


def test_catalog_429_retries_metadata_only_and_redacts_diagnostics(monkeypatch, capsys):
    calls, sleeps = [], []
    headers = {'Date': 'Fri, 11 Sep 2026 15:57:46 GMT',
               'RateLimit': '"api";r=0;t=34',
               'RateLimit-Policy': '"fixed window";"api";q=2500;w=300',
               'Authorization': 'DO_NOT_LOG', 'X-Signed-URL': 'https://signed.invalid/'}
    def publish(api, plan):
        calls.append(plan)
        if len(calls) == 1: raise CatalogHttpError(headers=headers)
        return 'published'
    monkeypatch.setattr(a, 'publish_catalog', publish)
    monkeypatch.setattr(a.time, 'sleep', sleeps.append)
    monkeypatch.setattr(a, 'archive_one', lambda *args, **kwargs: pytest.fail('metadata retry called a model operation'))
    assert a.publish_catalog_with_retry(object(), {'fixture': True}) == 'published'
    assert len(calls) == 2 and sleeps == [30, 6]
    event = json.loads(capsys.readouterr().out)
    assert event['event'] == 'catalog_rate_limited' and event['wait_seconds'] == 36
    assert set(event['headers']) == {'Date', 'RateLimit', 'RateLimit-Policy'}
    assert 'DO_NOT_LOG' not in json.dumps(event) and 'signed.invalid' not in json.dumps(event)


def test_catalog_429_exhaustion_is_bounded(monkeypatch):
    calls, sleeps = [], []
    error = CatalogHttpError()
    def publish(*args): calls.append(1); raise error
    monkeypatch.setattr(a, 'publish_catalog', publish)
    monkeypatch.setattr(a.time, 'sleep', sleeps.append)
    with pytest.raises(CatalogHttpError) as caught:
        a.publish_catalog_with_retry(object(), {})
    assert caught.value is error and len(calls) == 4
    assert len(sleeps) == 30 and sum(sleeps) == 900 and max(sleeps) <= 30


def test_catalog_excessive_server_wait_stops_without_early_retry(monkeypatch):
    calls, sleeps = [], []
    def publish(*args): calls.append(1); raise CatalogHttpError(headers={'Retry-After': '3600'})
    monkeypatch.setattr(a, 'publish_catalog', publish)
    monkeypatch.setattr(a.time, 'sleep', sleeps.append)
    with pytest.raises(CatalogHttpError): a.publish_catalog_with_retry(object(), {})
    assert len(calls) == 1 and sleeps == []


@pytest.mark.parametrize('status', [403, 500, 503])
def test_non429_catalog_errors_are_never_retried(monkeypatch, status):
    calls, sleeps = [], []
    def publish(*args): calls.append(1); raise CatalogHttpError(status=status)
    monkeypatch.setattr(a, 'publish_catalog', publish)
    monkeypatch.setattr(a.time, 'sleep', sleeps.append)
    with pytest.raises(CatalogHttpError): a.publish_catalog_with_retry(object(), {})
    assert len(calls) == 1 and sleeps == []


def test_catalog_cadence_coalesces_completions_and_flushes_at_clean_finish(worker_run, monkeypatch):
    clock = [0]
    completed, published = [], []
    times = [0, 30, 119, 120, 121, 239, 250]
    def archive(api, plan, record, **kwargs):
        clock[0] = times[len(completed)]
        completed.append(record['archive_id'])
    monkeypatch.setattr(a, 'archive_one', archive)
    monkeypatch.setattr(a.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(a, 'publish_catalog', lambda *args: published.append((clock[0], len(completed))))
    a.run(worker_run.path, worker_run.token, limit=0, keep_local=True, workers=1)
    assert published == [(0, 1), (120, 4), (250, 7)]
    assert json.loads((worker_run.output / 'status.json').read_text())['status'] == 'batch_finished'


def test_catalog_flush_publishes_subinterval_tail(worker_run, monkeypatch):
    clock = [0]
    completed, published = [], []
    def archive(api, plan, record, **kwargs):
        clock[0] = len(completed) * 10
        completed.append(record['archive_id'])
    monkeypatch.setattr(a, 'archive_one', archive)
    monkeypatch.setattr(a.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(a, 'publish_catalog', lambda *args: published.append((clock[0], len(completed))))
    a.run(worker_run.path, worker_run.token, limit=0, keep_local=True, workers=1)
    assert published == [(0, 1), (60, 7)]


def test_catalog_empty_resume_flushes_without_resubmitting_models(worker_run, monkeypatch):
    for record in worker_run.models:
        a.save(worker_run.output / 'models' / record['archive_id'] / 'retired.json', {'removed_bytes': 1})
    calls = []
    monkeypatch.setattr(a, 'archive_one', lambda *args, **kwargs: pytest.fail('retired model resubmitted'))
    monkeypatch.setattr(a, 'publish_catalog', lambda *args: calls.append('catalog'))
    a.run(worker_run.path, worker_run.token, limit=0, keep_local=False, workers=6)
    assert calls == ['catalog']
    assert json.loads((worker_run.output / 'status.json').read_text())['retired_models'] == 7


def test_catalog_final_flush_failure_is_reported_as_metadata_not_model(worker_run, monkeypatch):
    calls, completed = [], []
    monkeypatch.setattr(a, 'archive_one', lambda api, plan, record, **kwargs: completed.append(record['archive_id']))
    monkeypatch.setattr(a.time, 'monotonic', lambda: 0)
    def publish(*args):
        calls.append(1)
        if len(calls) == 2: raise CatalogHttpError(status=403)
    monkeypatch.setattr(a, 'publish_catalog', publish)
    with pytest.raises(SystemExit) as caught:
        a.run(worker_run.path, worker_run.token, limit=0, keep_local=True, workers=1)
    assert caught.value.code == 1 and len(completed) == 7
    status = json.loads((worker_run.output / 'status.json').read_text())
    assert status['status'] == 'stopped' and status['failures'] == [
        {'model': None, 'phase': 'catalog', 'error_type': 'CatalogHttpError', 'http_status': 403}]
