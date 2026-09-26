"""Mutation guards for the calibration orchestration entry point."""
from argparse import Namespace
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import advance_modebench_level3 as advance


def campaign(tmp_path, active=True):
    args = Namespace(advance=active, fit_only=False, results_root=tmp_path / 'results',
                     recipes_root=tmp_path / 'recipes', output_root=tmp_path / 'dataset',
                     journal_root=tmp_path / 'journals')
    return advance.Campaign(args)


def test_finalization_waits_for_all_five_recipes_without_writes(tmp_path):
    run = campaign(tmp_path)
    run.finalize({'graph_coloring': tmp_path / 'graph.json'})
    assert run.report['finalization']['status'] == 'waiting_for_five_passing_recipes'
    assert not list(tmp_path.iterdir())
    assert not run.report['produced_artifacts']


def test_existing_output_is_never_passed_to_a_mutating_command(tmp_path, monkeypatch):
    output = tmp_path / 'immutable.json'
    output.write_text('original bytes')
    monkeypatch.setattr(advance.subprocess, 'run', lambda *a, **k: pytest.fail('must not run a command'))
    with pytest.raises(ValueError, match='fresh output required'):
        campaign(tmp_path).run_command('fit', [sys.executable, 'irrelevant.py'], output)
    assert output.read_text() == 'original bytes'


def test_stale_existing_merge_is_preserved_even_with_advance(tmp_path, monkeypatch):
    import merge_modebench_level3_receipts as merger
    output = tmp_path / 'merged.json'
    output.write_text('old merge bytes')
    expected = {'identity': {'seeds': advance.SEEDS}, 'identity_sha256': 'expected'}
    actual = {'identity': {'seeds': advance.SEEDS}, 'identity_sha256': 'different'}
    monkeypatch.setattr(merger, 'merge_receipts', lambda *a, **k: expected)
    monkeypatch.setattr(advance, 'receipt', lambda *a, **k: (actual, [], None))
    run = campaign(tmp_path)
    monkeypatch.setattr(run, 'run_command', lambda *a, **k: pytest.fail('must not overwrite stale merge'))
    result = run.merge('graph_coloring', 'level1', '05b', [tmp_path / 'a', tmp_path / 'b'], output)
    assert result['status'] == 'blocked_unchanged'
    assert 'stale or unexpected' in result['detail']
    assert output.read_text() == 'old merge bytes'
    assert not run.report['produced_artifacts']


def test_stale_existing_dataset_is_preserved(tmp_path, monkeypatch):
    run = campaign(tmp_path)
    run.args.output_root.mkdir()
    sentinel = run.args.output_root / 'identity.json'
    sentinel.write_text('untouched')
    def reject(*args):
        raise ValueError('stale recipe binding')
    monkeypatch.setattr(advance, 'validate_dataset', reject)
    monkeypatch.setattr(run, 'run_command', lambda *a, **k: pytest.fail('must not replace dataset'))
    run.finalize({domain: tmp_path / f'{domain}.json' for domain in advance.DOMAINS})
    assert run.report['finalization']['status'] == 'blocked_unchanged'
    assert sentinel.read_text() == 'untouched'
