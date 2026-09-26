"""Real scratch manifests/hard links; costly scientific replay is substituted.

These tests prove publication/guard behavior, not production calibration or NFS
support. Production inputs are never staged or published by this test module.
"""
import importlib.util
import json
import os
from pathlib import Path

import pytest

from test_continue_modebench_scale_composite import campaign, write

PATH = Path(__file__).resolve().parents[1] / 'artifacts/stage_modebench_scale_source_binding_resume_20260912.py'
spec = importlib.util.spec_from_file_location('staged_binding_helper', PATH)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
controller = helper.composite


@pytest.fixture
def staging(campaign, monkeypatch):
    state = campaign
    state.artifacts.mkdir(exist_ok=True)
    monkeypatch.setattr(helper, 'PARENT', state.parent)
    monkeypatch.setattr(helper, 'RELEASE', state.release)
    monkeypatch.setattr(helper, 'ARTIFACTS', state.artifacts)
    proposals = write(state.artifacts / 'revision_bindings.json',
                      {'schema': 'modebench_scale_composite_revision_bindings_v1',
                       'revisions': {level: {} for level in controller.LEVELS}})
    sources = [state.parent / 'protocol.json', Path(controller.__file__).resolve(), proposals]
    monkeypatch.setattr(controller, 'activation_sources', lambda *args: sources)
    monkeypatch.setattr(controller.previous, 'runtime_amendment', lambda *args: None)
    activation = write(state.artifacts / 'controller_activation.json',
                       {'schema': controller.ACTIVATION_SCHEMA, 'parent_root': str(state.parent),
                        'release_root': str(state.release), 'artifacts_root': str(state.artifacts),
                        'parent_array_job_id': controller.PARENT_ARRAY_JOB_ID,
                        'previous_controller_pid': controller.PREVIOUS_CONTROLLER_PID,
                        'previous_controller_stopped': True, 'host': helper.ACTIVATION_HOST,
                        'source_selection_policy': controller.POLICY,
                        'runtime_profile': controller.launch.RUNTIME_PROFILE,
                        'files_sha256': controller.pin_files(sources)})
    monkeypatch.setattr(helper, 'ACTIVATION_SHA256', helper.file_sha(activation))
    monkeypatch.setattr(helper, 'BINDINGS_SHA256', helper.file_sha(proposals))

    def forbidden(*args, **kwargs):
        raise AssertionError('controller activation/lock or production work must not be called')

    monkeypatch.setattr(controller, 'validate_activation', forbidden)
    monkeypatch.setattr(controller, 'controller_lock', forbidden)
    monkeypatch.setattr(controller.original, 'freeze_dataset', forbidden)
    monkeypatch.setattr(controller.revision, 'materialize_pools', forbidden)
    monkeypatch.setattr(controller.launch, 'submit', forbidden)
    state.fit_calls = []

    def fit(root, level, domain, *, publish):
        assert publish is False
        state.fit_calls.append((level, domain))
        return state.recipes[level, domain]

    monkeypatch.setattr(controller.original_fit, 'fit_domain', fit)
    state.mapping_path = write(state.artifacts / 'explicit_maps.json', state.bindings)
    state.stage_root = state.release.parent / '.modebench_scale_source_stage_scratch'
    state.stage_report = state.artifacts / 'stage_report.json'
    state.publish_report = state.artifacts / 'publish_report.json'
    return state


def run_stage(state):
    return helper.stage(state.stage_root, state.mapping_path, helper.file_sha(state.mapping_path), state.stage_report)


def run_publish(state):
    return helper.publish(state.stage_report, helper.file_sha(state.stage_report), state.publish_report)


def repin_activation(state, monkeypatch):
    path = state.artifacts / 'controller_activation.json'
    value = helper.read(path)
    value['files_sha256'] = controller.pin_files(controller.activation_sources())
    write(path, value)
    monkeypatch.setattr(helper, 'ACTIVATION_SHA256', helper.file_sha(path))
    monkeypatch.setattr(helper, 'BINDINGS_SHA256', helper.file_sha(state.artifacts / 'revision_bindings.json'))


def test_stage_then_real_ordered_publication(staging, monkeypatch):
    state = staging
    staged = run_stage(state)
    assert len(state.fit_calls) == 10
    assert not state.release.exists()
    assert len(helper.tree_files(state.stage_root)) == 4
    assert staged['local_action_host'] == os.uname().nodename
    assert staged['recorded_activation_host'] == 'soak.cs.princeton.edu'
    assert 'not invoked' in staged['activation_verification']
    real_link = os.link
    observations = []

    def observe(source, destination, **kwargs):
        result = real_link(source, destination, **kwargs)
        destination = Path(destination)
        if destination.is_relative_to(state.release):
            ready = [level for level in controller.LEVELS
                     if (state.release / level / 'source_manifest.json').is_file()]
            observations.append((destination.name, ready))
            if len(ready) < 2:
                # Real controller early-return prevents any freezes/submissions.
                assert controller._sweep(state.parent, state.release, state.artifacts, advance=True)['status'] == 'needs_source_manifest'
        return result

    monkeypatch.setattr(helper.os, 'link', observe)
    published = run_publish(state)
    assert observations == [('source_manifest.sha256.json', []), ('source_manifest.sha256.json', []),
                            ('source_manifest.json', ['level4']), ('source_manifest.json', ['level4', 'level5'])]
    assert published['status'] == 'published'
    assert len(state.fit_calls) == 10
    assert not (state.artifacts / 'controller.lock').exists()
    for relative, digest in staged['stage_files_sha256'].items():
        source, destination = state.stage_root / relative, state.release / relative
        assert helper.file_sha(destination) == digest
        assert source.stat().st_ino == destination.stat().st_ino


@pytest.mark.parametrize('kind', ['file', 'empty_directory', 'dangling_symlink'])
def test_existing_canonical_destination_never_overwritten(staging, kind):
    state = staging
    run_stage(state)
    if kind == 'file':
        state.release.write_text('keep')
    elif kind == 'empty_directory':
        state.release.mkdir()
    else:
        state.release.symlink_to(state.release.parent / 'missing')
    with pytest.raises(ValueError, match='absent|canonical'):
        run_publish(state)
    assert os.path.lexists(state.release)
    if kind == 'file':
        assert state.release.read_text() == 'keep'
    assert not state.publish_report.exists()


@pytest.mark.parametrize('missing', ['recipe', 'receipt'])
def test_missing_original_evidence_rejected_before_stage(staging, missing):
    state = staging
    if missing == 'recipe':
        path = controller.domain_paths(state.parent, 'level5', 'pantry')['recipe']
    else:
        path = controller.development_receipts(state.parent, 'level5', 'pantry')[3]
    path.unlink()
    with pytest.raises(ValueError, match='original recipes|full-domain'):
        run_stage(state)
    assert not state.stage_root.exists()


@pytest.mark.parametrize('kind', ['missing_level', 'extra_domain', 'missing_failed'])
def test_exact_explicit_maps_required(staging, kind):
    mappings = helper.read(staging.mapping_path)
    if kind == 'missing_level':
        del mappings['level5']
    elif kind == 'extra_domain':
        mappings['level4']['pantry'] = str(staging.parent)
    else:
        mappings['level5'] = {}
    write(staging.mapping_path, mappings)
    with pytest.raises(ValueError, match='exactly'):
        run_stage(staging)
    assert not staging.stage_root.exists()


@pytest.mark.parametrize('level', ['level4', 'level5'])
def test_sealed_proposal_complete_for_either_level_is_rejected(staging, monkeypatch, level):
    path = staging.artifacts / 'revision_bindings.json'
    value = helper.read(path)
    value['revisions'][level] = staging.bindings[level]
    write(path, value)
    repin_activation(staging, monkeypatch)
    with pytest.raises(ValueError, match='BOTH levels'):
        run_stage(staging)
    assert not staging.stage_root.exists()


@pytest.mark.parametrize('root_kind', ['parent', 'selected_revision'])
@pytest.mark.parametrize('heldout_kind', ['receipt', 'batches', 'dangling_receipt'])
def test_observed_heldout_rejected_at_publication(staging, root_kind, heldout_kind):
    run_stage(staging)
    level, domain = 'level5', 'mathir'
    root = staging.parent if root_kind == 'parent' else Path(staging.bindings[level][domain])
    receipt = controller.domain_paths(root, level, domain)['receipt']
    receipt.parent.mkdir(parents=True)
    if heldout_kind == 'receipt':
        write(receipt, {})
    elif heldout_kind == 'batches':
        Path(str(receipt) + '.batches').mkdir()
    else:
        receipt.symlink_to(receipt.parent / 'missing')
    with pytest.raises(ValueError, match='absent'):
        run_publish(staging)
    assert not staging.release.exists()


@pytest.mark.parametrize('drift', ['recipe', 'mapping', 'activation', 'manifest', 'manifest_sidecar'])
def test_input_drift_rejected(staging, drift):
    run_stage(staging)
    path = {'recipe': controller.domain_paths(staging.parent, 'level5', 'pantry')['recipe'],
            'mapping': staging.mapping_path,
            'activation': staging.artifacts / 'controller_activation.json',
            'manifest': staging.stage_root / 'level4/source_manifest.json',
            'manifest_sidecar': staging.stage_root / 'level4/source_manifest.sha256.json'}[drift]
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='changed|pins|tree'):
        run_publish(staging)
    assert not staging.release.exists()


@pytest.mark.parametrize('damage', ['missing', 'extra', 'symlink'])
def test_incomplete_or_unexpected_staged_tree_rejected(staging, damage):
    run_stage(staging)
    sidecar = staging.stage_root / 'level5/source_manifest.sha256.json'
    if damage == 'missing':
        sidecar.unlink()
    elif damage == 'extra':
        (staging.stage_root / 'extra').write_text('unexpected')
    else:
        content = sidecar.read_bytes()
        sidecar.unlink()
        target = staging.artifacts / 'elsewhere.json'
        target.write_bytes(content)
        sidecar.symlink_to(target)
    with pytest.raises(ValueError, match='stage|regular|directories'):
        run_publish(staging)
    assert not staging.release.exists()


@pytest.mark.parametrize('interrupt_before_link', [1, 2, 3, 4])
def test_interrupted_publication_stays_safe_and_requires_reconciliation(staging, monkeypatch, interrupt_before_link):
    run_stage(staging)
    real_link = os.link
    count = 0

    def interrupt(source, destination, **kwargs):
        nonlocal count
        if Path(destination).is_relative_to(staging.release):
            count += 1
            if count == interrupt_before_link:
                raise OSError('injected interruption')
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(helper.os, 'link', interrupt)
    with pytest.raises(OSError, match='injected'):
        run_publish(staging)
    assert staging.release.is_dir()
    assert staging.publish_report.with_name('publish_report.intent.json').is_file()
    assert not staging.publish_report.exists()
    assert controller._sweep(staging.parent, staging.release, staging.artifacts, advance=True)['status'] == 'needs_source_manifest'
    with pytest.raises(ValueError, match='absent'):
        run_publish(staging)


def test_postcommit_watcher_activity_does_not_invalidate_publication(staging, monkeypatch):
    run_stage(staging)
    real_link = os.link

    def immediate_watcher(source, destination, **kwargs):
        result = real_link(source, destination, **kwargs)
        if Path(destination) == staging.release / 'level5/source_manifest.json':
            write(staging.artifacts / 'source_set_identity.json', {'watcher': 'advanced'})
            write(staging.release / 'downstream.json', {'watcher': 'advanced'})
            write(controller.domain_paths(staging.parent, 'level4', 'pantry')['receipt'], {'watcher': 'advanced'})
        return result

    monkeypatch.setattr(helper.os, 'link', immediate_watcher)
    assert run_publish(staging)['status'] == 'published'


def test_action_lock_is_nonblocking_and_independent(staging):
    with helper.action_lock():
        with pytest.raises(BlockingIOError):
            with helper.action_lock():
                pytest.fail('second action acquired lock')
    assert not (staging.artifacts / 'controller.lock').exists()


def test_staging_failure_never_creates_canonical_root(staging, monkeypatch):
    original_bind = controller.bind_level

    def fail_second(parent, stage_root, level, mapping):
        if level == 'level5':
            raise RuntimeError('reproduction failed')
        return original_bind(parent, stage_root, level, mapping)

    monkeypatch.setattr(controller, 'bind_level', fail_second)
    with pytest.raises(RuntimeError, match='reproduction'):
        run_stage(staging)
    assert not staging.release.exists()
    assert not staging.stage_report.exists()
    assert (staging.stage_root / 'level4/source_manifest.json').is_file()


def test_recorded_activation_host_checked_without_current_host_spoof(staging, monkeypatch):
    path = staging.artifacts / 'controller_activation.json'
    value = helper.read(path)
    value['host'] = 'different-host'
    write(path, value)
    monkeypatch.setattr(helper, 'ACTIVATION_SHA256', helper.file_sha(path))
    with pytest.raises(ValueError, match='recorded host'):
        run_stage(staging)


@pytest.mark.parametrize('kind', ['file', 'empty_directory', 'dangling_symlink'])
def test_real_hardlink_primitive_does_not_clobber_any_existing_path(tmp_path, kind):
    source, destination = tmp_path / 'source', tmp_path / 'destination'
    source.write_text('staged')
    if kind == 'file':
        destination.write_text('keep')
    elif kind == 'empty_directory':
        destination.mkdir()
    else:
        destination.symlink_to(tmp_path / 'missing')
    with pytest.raises(FileExistsError):
        os.link(source, destination, follow_symlinks=False)
    assert source.read_text() == 'staged'
    assert os.path.lexists(destination)
    if kind == 'file':
        assert destination.read_text() == 'keep'


@pytest.mark.parametrize('phase', ['stage', 'publish'])
def test_source_set_identity_forbids_prepublication_actions(staging, phase):
    if phase == 'publish':
        run_stage(staging)
    write(staging.artifacts / 'source_set_identity.json', {'existing': True})
    with pytest.raises(ValueError, match='absent'):
        (run_stage if phase == 'stage' else run_publish)(staging)
    assert not staging.release.exists()
