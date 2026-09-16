"""Only complete paired deployments may receive empirical reasoning scores."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reasoning_summary', ROOT / 'ops/summarize_hosted_reasoning_off.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def cohort():
    rows = [{'level': level, 'domain': domain, 'row_index': index}
            for level in m.LEVELS for domain in m.DOMAINS for index in range(32)]
    samples = [{**row, 'sample_index': draw, 'sample_id': f"{row['level']}/{row['domain']}/{row['row_index']}/{draw}",
                'verified': row['row_index'] == 0, 'canonical_key': str(draw % 2) if row['row_index'] == 0 else None}
               for row in rows for draw in range(8)]
    return rows, samples


def test_empirical_success_and_mode_means_include_unsolved_prompts(cohort):
    rows, samples = cohort
    result = m.summarize_cohort(rows, samples)
    assert len(result['prompts']) == 480
    for level in result['levels'].values():
        assert level['counts'] == {'prompts': 160, 'responses': 1280, 'prompts_with_correct': 5,
                                   'distinct_correct_modes': 10, 'correct_responses': 40}
        assert level['metrics'] == {'pass8': 1 / 32, 'distinct8': 2 / 32}
        assert level['metrics']['pass8'] != pytest.approx(1 - (1 - 40 / 1280) ** 8)


@pytest.mark.parametrize('tamper', ['missing_draw', 'duplicate_draw', 'missing_domain', 'nonfirst_row', 'missing_mode'])
def test_whole_model_admission_rejects_partial_or_invalid_grid(cohort, tamper):
    rows, samples = cohort
    if tamper == 'missing_draw':
        samples.pop()
    elif tamper == 'duplicate_draw':
        samples[-1] = deepcopy(samples[0])
    elif tamper == 'missing_domain':
        rows = [r for r in rows if r['domain'] != 'graph_coloring']
    elif tamper == 'nonfirst_row':
        rows[-1]['row_index'] = 32
    else:
        samples[0]['canonical_key'] = None
    with pytest.raises(ValueError):
        m.summarize_cohort(rows, samples)


def test_paired_differences_preserve_each_condition(cohort):
    rows, on = cohort
    off = deepcopy(on)
    for sample in off:
        if sample['row_index'] == 1:
            sample.update(verified=True, canonical_key='third-mode')
    result = m.paired_metrics(rows, on, off)
    assert result['on']['levels']['1']['metrics']['pass8'] == 1 / 32
    assert result['off']['levels']['1']['metrics']['pass8'] == 2 / 32
    assert result['off_minus_on']['1'] == {'pass8': 1 / 32, 'distinct8': 1 / 32}


def test_incomplete_collection_produces_no_metrics(tmp_path):
    state = m.collection_state(tmp_path)
    assert state['status'] == 'pending_collection'
    assert state['scores_admitted'] is False
    assert 'metrics' not in state


def test_control_violation_blocks_entire_model_even_with_complete_inventory(tmp_path, cohort):
    rows, samples = cohort
    receipts = tmp_path / 'sample_receipts'
    receipts.mkdir()
    # Gate before grading/file aggregation even if all registered receipt names exist.
    for index in range(3840):
        (receipts / f'{index}.json').touch()
    (tmp_path / 'provider_control_violation.json').write_text(json.dumps({'reason': 'Provider returned thinking tokens'}))
    state = m.collection_state(tmp_path)
    assert state['status'] == 'blocked_control_violation'
    assert state['terminal_sample_receipts'] == 3840
    assert state['scores_admitted'] is False
    assert 'metrics' not in state


def test_amended_registry_resolves_deepseek_child_and_binds_original(tmp_path):
    original = {'slug': 'deepseek', 'model': 'DeepSeek-V4-Pro', 'run_directory': 'original', 'manifest_sha256': 'old'}
    replacement = {**original, 'run_directory': 'child', 'manifest_sha256': 'new'}
    registry_path = tmp_path / 'experiment.json'
    registry_path.write_text(json.dumps({'runs': [original]}))
    amendment = {'original_experiment_sha256': m.file_sha(registry_path), 'replacements': [
        {'slug': 'deepseek', 'original_entry': original, 'replacement_entry': replacement}]}
    (tmp_path / 'deployment_amendments.json').write_text(json.dumps(amendment))
    assert m.resolved_registry(tmp_path)['runs'] == [replacement]
    amendment['original_experiment_sha256'] = 'changed'
    (tmp_path / 'deployment_amendments.json').write_text(json.dumps(amendment))
    with pytest.raises(ValueError, match='different experiment'):
        m.resolved_registry(tmp_path)


def test_finalizer_stops_when_collectors_exit_despite_incomplete_models(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'active_collectors', lambda base: [])
    assert m.wait_for_collectors(tmp_path, max_wait_seconds=1) == 'collectors_stopped'
    status = json.loads((tmp_path / 'reasoning_comparison_finalizer_status.json').read_text())
    assert status['api_calls'] == 0


def test_finalizer_deadline_is_bounded(tmp_path, monkeypatch):
    times = iter([0, 2])
    monkeypatch.setattr(m.time, 'monotonic', lambda: next(times))
    monkeypatch.setattr(m, 'active_collectors', lambda base: [{'pid': 1}])
    assert m.wait_for_collectors(tmp_path, max_wait_seconds=1) == 'wait_deadline_reached'


def test_finalizer_ignores_its_own_live_launch_marker(tmp_path, monkeypatch):
    import os
    import sys
    marker = {'pid': os.getpid(), 'command': [sys.executable, str(ROOT / 'ops/summarize_hosted_reasoning_off.py'), '--wait-for-collectors']}
    (tmp_path / 'reasoning_comparison_finalizer_launch.json').write_text(json.dumps(marker))
    (tmp_path / 'control_monitor_launch.json').write_text(json.dumps({**marker,
        'command': [sys.executable, str(ROOT / 'ops/watch_hosted_reasoning_off_controls.py')]}))
    monkeypatch.setattr(m, 'resolved_registry', lambda base: {'runs': []})
    assert m.active_collectors(tmp_path) == []
