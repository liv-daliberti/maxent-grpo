"""Offline completion must skip active runs and resume without rewriting evidence."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path

import pytest
from ops import complete_gpt56_all_levels32_analysis as completion


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def cohort(tmp_path, *, complete=True):
    out = tmp_path / 'run'
    out.mkdir()
    for name in ('.runner.lock', '.all_levels_orchestrator.lock'):
        (out / name).touch()
    write(out / 'status.json', {'complete': complete, 'completed_samples': 8192 if complete else 1,
                              'expected_samples': 8192, 'failed_groups_this_session': 0})
    manifest = {'collection_runs': {'L1_graph_coloring': {'run_dir': str(out), 'new_samples': 8192}}}
    return out, manifest


def no_calls(*args):
    pytest.fail('Incomplete or locked evidence must never launch grading/auditing')


def fake_grading(out):
    write(out / 'samples.jsonl', {'sample': 'raw'})
    write(out / 'discovery_hosted_grades.jsonl', {'strict': 'checked', 'normalization': 'checked'})
    for name in ('frontier_modebench_contract.py', 'frontier_modebench_normalization.py'):
        path = out / 'code/ops' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('frozen code')
    write(out / 'all_domain_grading_audit.json', {
        'status': 'complete', 'responses': 8192, 'api_calls': 0,
        'samples': completion.binding(out / 'samples.jsonl'),
        'grades': completion.binding(out / 'discovery_hosted_grades.jsonl'),
        'grader': completion.binding(out / 'code/ops/frontier_modebench_contract.py'),
        'normalizer': completion.binding(out / 'code/ops/frontier_modebench_normalization.py'),
        'grader_driver': completion.binding(completion.ROOT / 'ops/analyze_gpt56_all_levels_discovery.py')})


def test_partial_collection_never_starts_semantic_grading(tmp_path):
    out, manifest = cohort(tmp_path, complete=False)
    states = completion.grade_completed(manifest, tmp_path, runner=no_calls)
    assert states == {'L1_graph_coloring': 'waiting_for_collection'}
    assert not (out / 'discovery_hosted_grades.jsonl').exists()


@pytest.mark.parametrize('name', ['.runner.lock', '.all_levels_orchestrator.lock'])
def test_complete_status_is_insufficient_while_collection_lock_is_held(tmp_path, name):
    out, manifest = cohort(tmp_path)
    with (out / name).open() as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        states = completion.grade_completed(manifest, tmp_path, runner=no_calls)
    assert states['L1_graph_coloring'] == 'waiting_for_collection_locks'


def test_missing_lock_does_not_create_false_evidence_of_completion(tmp_path):
    out, manifest = cohort(tmp_path)
    (out / '.runner.lock').unlink()
    states = completion.grade_completed(manifest, tmp_path, runner=no_calls)
    assert states['L1_graph_coloring'] == 'waiting_for_collection_locks'
    assert not (out / '.runner.lock').exists()


def test_resume_reuses_completed_grades_and_preserves_their_bytes_and_mtimes(tmp_path):
    out, manifest = cohort(tmp_path)
    calls = []
    def grade(command, log):
        calls.append(command)
        assert command[-2:] == ['--grade-run', str(out)]
        # Both collector locks remain held during the grading subprocess.
        with completion.available_locks([out / '.runner.lock']) as ready:
            assert not ready
        fake_grading(out)
    assert completion.grade_completed(manifest, tmp_path, runner=grade)['L1_graph_coloring'] == 'graded_complete'
    files = [out / 'all_domain_grading_audit.json', out / 'discovery_hosted_grades.jsonl']
    before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in files]
    assert completion.grade_completed(manifest, tmp_path, runner=no_calls)['L1_graph_coloring'] == 'reused_authenticated_grades'
    assert len(calls) == 1 and [(p.read_bytes(), p.stat().st_mtime_ns) for p in files] == before


def test_interrupted_unaudited_grade_file_is_preserved_before_retry(tmp_path):
    out, manifest = cohort(tmp_path)
    partial = out / 'discovery_hosted_grades.jsonl'
    partial.write_text('{"incomplete":')
    def grade(command, log):
        assert not partial.exists()
        assert [p.read_text() for p in out.glob('discovery_hosted_grades.unaudited-*.jsonl')] == ['{"incomplete":']
        fake_grading(out)
    completion.grade_completed(manifest, tmp_path, runner=grade)
    assert completion.grading_complete(out)


def test_changed_cached_evidence_fails_closed_and_is_not_regraded(tmp_path):
    out, manifest = cohort(tmp_path)
    fake_grading(out)
    (out / 'samples.jsonl').write_text('changed raw response')
    with pytest.raises(ValueError, match='Changed bound artifact'):
        completion.grade_completed(manifest, tmp_path, runner=no_calls)


def test_final_audit_never_starts_for_a_partial_grid(tmp_path):
    out, manifest = cohort(tmp_path)
    result = completion.finalize_if_ready(tmp_path, manifest, {}, {'L1_graph_coloring': 'graded_complete'}, tmp_path, 4, runner=no_calls)
    assert result == 'waiting_for_all_fifteen_graded_cohorts'


def test_final_auditor_is_called_after_orchestrator_releases_probe_locks(tmp_path, monkeypatch):
    entries = {name: {'run_dir': str(tmp_path / name)} for name in completion.EXPECTED_COHORTS}
    manifest = {'collection_runs': entries}
    states = dict.fromkeys(entries, 'graded_complete')
    held = {'value': False}
    @contextmanager
    def locks(paths):
        held['value'] = True
        yield True
        held['value'] = False
    monkeypatch.setattr(completion, 'available_locks', locks)
    monkeypatch.setattr(completion, 'collection_locks', lambda *args: [])
    monkeypatch.setattr(completion, 'complete_status', lambda *args: True)
    monkeypatch.setattr(completion, 'authenticated_existing_audit', lambda *args: False)
    class AuditReached(Exception):
        pass
    def runner(command, log):
        assert str(completion.AUDITOR) in command and held['value'] is False
        raise AuditReached
    with pytest.raises(AuditReached):
        completion.finalize_if_ready(tmp_path, manifest, {}, states, tmp_path, 4, runner=runner)


def test_interrupted_json_output_is_preserved_for_offline_regeneration(tmp_path):
    path = tmp_path / 'analysis.json'
    path.write_text('{"status":')
    assert completion.existing_json(path) is None
    assert not path.exists()
    assert [p.read_text() for p in tmp_path.glob('analysis.json.incomplete-*')] == ['{"status":']


def test_interrupted_grading_audit_is_preserved_before_grade_retry(tmp_path):
    out, manifest = cohort(tmp_path)
    fake_grading(out)
    (out / 'all_domain_grading_audit.json').write_text('{"status":')
    def grade(command, log):
        assert [p.read_text() for p in out.glob('all_domain_grading_audit.json.incomplete-*')] == ['{"status":']
        assert len(list(out.glob('discovery_hosted_grades.unaudited-*.jsonl'))) == 1
        fake_grading(out)
    assert completion.grade_completed(manifest, tmp_path, runner=grade)['L1_graph_coloring'] == 'graded_complete'
