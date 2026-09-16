"""Regression tests for explicit partial evidence and conflicting curve points."""
from pathlib import Path
import importlib.util
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'ops/exp_scaling' / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


matrix = load_module('partial_matrix', 'plot_paper_experiment1_composite.py')
curves = load_module('partial_curves', 'plot_paper_aligned_domain_strips.py')


def cell(seeds, offset=0.0):
    per_seed = {str(seed): {'pass8': seed + offset, 'distinct8': 2 * seed + offset}
                for seed in seeds}
    return {'n': len(per_seed), 'per_seed': per_seed, 'summaries': matrix._summary(per_seed)}


def test_partial_mean_has_no_five_seed_interval():
    summary = cell([55, 56, 57, 58])['summaries']
    assert summary['pass8']['mean'] == 56.5
    assert 'student_t_95' not in summary['pass8']
    assert 'student_t_95' in cell([55, 56, 57, 58, 59])['summaries']['pass8']


def test_macro_restricts_every_domain_to_common_admissible_seeds():
    cells = {domain: cell([55, 56, 57, 58, 59], i)
             for i, domain in enumerate(matrix.DOMAINS)}
    cells['countdown'] = cell([55, 56, 57, 58], 1)
    # A large valid fifth observation elsewhere must not contaminate a paired macro.
    cells['graph_coloring']['per_seed']['59']['pass8'] = 1000000.0
    result = matrix._macro_average(cells)
    assert result['seeds'] == [55, 56, 57, 58]
    assert result['summaries']['pass8']['mean'] == 58.5
    assert 'student_t_95' not in result['summaries']['pass8']


def test_incomplete_block_cannot_enter_complete_block_sign_test():
    cells = {model: {domain: cell([1, 2, 3, 4, 5]) for domain in matrix.DOMAINS}
             for model in matrix.MODELS}
    cells['Falcon3-1B']['countdown'] = cell([1, 2, 3, 4], -100)
    result = matrix._omnibus_sign_tests(cells)
    assert result['tests']['pass8']['n_nonzero'] == 14
    assert result['tests']['pass8']['negative'] == 0
    assert result['incomplete_blocks'] == [{'model': 'Falcon3-1B', 'domain': 'countdown', 'n': 4}]


def trajectory_fixture(tmp_path, conflict_step):
    log = tmp_path / 'debug_job1/eval_mode_coverage_draws.jsonl'
    log.parent.mkdir()
    rows = []
    for step in [0, 192, 3072]:
        for draw in range(4):
            rows.append({'evaluation_kind': curves.RLEP_EVALUATION_KIND,
                         'step': step, 'draw_index': draw,
                         'metrics': {'any_correct_at_k': .4, 'distinct_correct_modes_at_k': .6}})
    rows.append({'evaluation_kind': curves.RLEP_EVALUATION_KIND,
                 'step': conflict_step, 'draw_index': 0,
                 'metrics': {'any_correct_at_k': .5, 'distinct_correct_modes_at_k': .8}})
    log.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return {'run_dir': str(tmp_path), 'domain': 'pantry_plan', 'seed': 43, 'job_id': 1}


def test_conflicted_initial_checkpoint_is_omitted_without_selecting_a_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(curves, "ROOT", tmp_path)
    result = curves.rlep_run_curve(trajectory_fixture(tmp_path, 0))
    assert set(result['curve']) == {192, 3072}
    assert result['curve'][3072] == {'pass8': .4, 'distinct8': .6}
    assert result['prefix_sources'][0]['excluded_conflicting_checkpoints'] == [0]
    assert result['prefix_sources'][0]['outcome_value_selected_for_conflicts'] is False


def test_unknown_conflicted_terminal_checkpoint_still_aborts(tmp_path, monkeypatch):
    monkeypatch.setattr(curves, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match='conflicting terminal'):
        curves.rlep_run_curve(trajectory_fixture(tmp_path, 3072))


def recovery_fixture(tmp_path, *, current_exists=True):
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    protocol = tmp_path / 'recovery.md'
    protocol.write_text('Replace the failed job and retain its run directory.\n')
    run = {'run_dir': str(run_dir), 'domain': 'pantry_plan', 'seed': 43,
           'job_id': 2, 'replaced_job_ids': [1],
           'repair_history': [{'old_job_id': 1, 'new_job_id': 2,
                               'protocol': str(protocol)}]}
    for job_id, value in [(1, .9), (2, .4)]:
        if job_id == 2 and not current_exists:
            continue
        log = run_dir / f'debug_job{job_id}/eval_mode_coverage_draws.jsonl'
        log.parent.mkdir()
        rows = [{'evaluation_kind': curves.RLEP_EVALUATION_KIND,
                 'step': step, 'draw_index': draw,
                 'metrics': {'any_correct_at_k': value,
                             'distinct_correct_modes_at_k': value + .2}}
                for step in [0, 192, 3072] for draw in range(4)]
        log.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return run


def test_registered_recovery_retains_initial_point_and_excludes_old_outcomes(tmp_path, monkeypatch):
    monkeypatch.setattr(curves, 'ROOT', tmp_path)
    run = recovery_fixture(tmp_path)
    result = curves.rlep_run_curve(run)
    assert set(result['curve']) == {0, 192, 3072}
    assert result['curve'][0]['pass8'] == .4
    assert result['curve'][3072]['pass8'] == .4
    assert len(result['prefix_sources']) == 1
    source = result['prefix_sources'][0]
    assert 'debug_job2/' in source['path']
    binding = source['source_binding']
    assert binding['registered_job_id'] == 2
    assert binding['superseded_job_ids'] == [1]
    assert binding['recovery_history'][0]['protocol_sha256'] == curves.sha256(tmp_path / 'recovery.md')
    assert binding['excluded_sources'][0]['job_id'] == 1
    assert binding['excluded_sources'][0]['outcome_value_selected'] is False
    assert 'excluded_conflicting_checkpoints' not in source


def test_missing_replacement_never_falls_back_to_superseded_outcomes(tmp_path, monkeypatch):
    monkeypatch.setattr(curves, 'ROOT', tmp_path)
    assert curves.rlep_run_curve(recovery_fixture(tmp_path, current_exists=False)) is None


def test_unregistered_source_aborts_even_when_current_job_is_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(curves, 'ROOT', tmp_path)
    run = recovery_fixture(tmp_path)
    unknown = Path(run['run_dir']) / 'debug_job3/eval_mode_coverage_draws.jsonl'
    unknown.parent.mkdir()
    unknown.write_text('{}\n')
    with pytest.raises(RuntimeError, match='unregistered evaluation source'):
        curves.rlep_run_curve(run)


def test_missing_recovery_amendment_aborts(tmp_path, monkeypatch):
    monkeypatch.setattr(curves, 'ROOT', tmp_path)
    run = recovery_fixture(tmp_path)
    (tmp_path / 'recovery.md').unlink()
    with pytest.raises(FileNotFoundError, match='recovery amendment missing'):
        curves.rlep_run_curve(run)


def test_terminal_summary_uses_registered_replacement_for_all_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(curves, 'ROOT', tmp_path)
    run = recovery_fixture(tmp_path)
    marker = Path(run['run_dir']) / 'TRAINING_COMPLETE.json'
    marker.write_text(json.dumps({'terminal_step': 3072}))
    summary, prefixes, markers = curves._terminal_method_summary(
        {('pantry_plan', 43): run}, domain='pantry_plan', seeds=[43],
        target=3072, train_rows=384,
    )
    assert summary['n'] == 1
    assert summary['summaries']['0']['pass8']['mean'] == .4
    assert summary['summaries']['3072']['pass8']['per_seed'] == {'43': .4}
    assert markers == [marker]
    assert prefixes[0]['source_binding']['registered_job_id'] == 2
