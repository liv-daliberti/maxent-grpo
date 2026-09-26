#!/usr/bin/env python3
"""Validate this dated figure refresh against its frozen census and prior figures."""
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import hashlib
import json
import runpy

ROOT = Path(__file__).resolve().parents[4]
AUDIT = ROOT / 'paper/audits/results_refresh_20260910'
HERE = AUDIT / 'figure_refresh'
CENSUS = AUDIT / 'latest_endpoints.json'
EXPECTED_CENSUS_SHA = 'a27e6762a3f05d6f3fa6b5f6659cf400d4a929a67db892c69f2f6c96c424a8e3'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sha(CENSUS) == EXPECTED_CENSUS_SHA, 'dated census changed'
    census = read(CENSUS)
    fig5_path = ROOT / 'paper/figures/e118_all_scale_factorial_progress.json'
    fig6_path = ROOT / 'paper/figures/modebench_level_admission.json'
    snap_path = ROOT / 'paper/results/modebench_level_comparison_snapshot.json'
    fig5, fig6, snapshot = map(read, (fig5_path, fig6_path, snap_path))
    old5 = read(HERE / 'before/paper/figures/e118_all_scale_factorial_progress.json')
    old6 = read(HERE / 'before/paper/figures/modebench_level_admission.json')
    old_snapshot = read(HERE / 'before/paper/results/modebench_level_comparison_snapshot.json')
    assert fig5['endpoint_snapshot']['sha256'] == EXPECTED_CENSUS_SHA
    assert len(fig5['endpoint_integrity_audit']) == 150
    assert fig5['absolute_cross_domain_average'] == old5['absolute_cross_domain_average'], 'displayed Figure 5 completed tracks changed'
    assert fig5['cross_domain_average'] == old5['cross_domain_average']
    assert snapshot['reference_figure'] == old_snapshot['reference_figure']
    assert fig6['admission_rows'] == old6['admission_rows']
    assert fig6['partial_treatment'] == old6['partial_treatment']
    assert fig6['sources'][str(snap_path.resolve())] == sha(snap_path)
    assert len(snapshot['availability']) == 200
    level_analysis = runpy.run_path(str(ROOT / 'ops/exp_scaling/build_paper_modebench_level_comparison.py'))
    expected = level_analysis['build_interim_comparison'](snapshot['evaluations'], snapshot['availability'])
    assert fig6['interim_comparison'] == expected
    assert fig6['terminal_progress_by_domain'] == level_analysis['build_terminal_progress'](snapshot['level2_terminal_evaluations'])
    # The all-checkpoint snapshot is independently collected on the same date.
    # Its experiment registrations and admitted terminal values must agree with
    # the canonical terminal census used for all new paper campaign tables.
    for campaign in ('e118', 'e119'):
        ledger = census['campaigns'][campaign]['ledger']
        frozen = AUDIT / 'figure6' / Path(ledger).name
        assert sha(frozen) == census['campaigns'][campaign]['ledger_sha256'], f'{campaign} registration differs'
        if campaign == 'e118':
            assert fig5['source_sha256'][ledger] == census['campaigns'][campaign]['ledger_sha256'], 'Figure 5 registration differs from census'
    for level, meta in snapshot['input_snapshots'].items():
        assert sha(ROOT / meta['path']) == meta['sha256'], f'{level} coverage snapshot changed'
    terminal = defaultdict(list)
    for row in snapshot['level2_terminal_evaluations']:
        assert row['step'] == 3072 and row['sample_count'] == 8
        terminal[row['domain'], row['method'], row['seed']].append(row)
    admitted = {(r['domain'], r['arm'], r['seed']): r for r in census['campaigns']['e119']['rows'] if r['endpoint'] is not None}
    assert set(terminal) == set(admitted), 'Level 2 terminal coverage differs from canonical census'
    for key, draws in terminal.items():
        assert len(draws) == 4 and {d['draw_index'] for d in draws} == {0, 1, 2, 3}
        for field, metric in (('pass8', 'any_correct_at_k'), ('distinct8', 'distinct_correct_modes_at_k')):
            mean = sum(d['metrics'][metric] for d in draws) / 4
            assert abs(mean - admitted[key]['endpoint'][field]) < 1e-12, (key, field)
    assert sha(ROOT / 'paper/results/e118_qwen3b_python_terminal_table_body.tex') == sha(HERE / 'before/paper/results/e118_qwen3b_python_terminal_table_body.tex'), 'unchanged completed Python table unexpectedly changed'
    current_paths = [ROOT / 'paper/figures' / f'{stem}.{suffix}'
                     for stem in ('e118_all_scale_factorial_progress', 'e118_scale_extensions_appendix', 'modebench_level_admission')
                     for suffix in ('pdf', 'png', 'json')]
    dated_snapshot = AUDIT / 'figure6/modebench_level_comparison_snapshot.json'
    raw_snapshot = snap_path.read_bytes()
    if dated_snapshot.exists():
        assert dated_snapshot.read_bytes() == raw_snapshot, 'retained dated comparison snapshot differs'
    else:
        dated_snapshot.write_bytes(raw_snapshot)
    receipt = {
        'schema': 'paper-figure-refresh-validation-v1',
        'validated_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_census': {'path': str(CENSUS.relative_to(ROOT)), 'sha256': sha(CENSUS)},
        'comparison_snapshot': {'path': str(dated_snapshot.relative_to(ROOT)), 'sha256': sha(dated_snapshot)},
        'comparison_collection_intervals': snapshot['collection_intervals'],
        'admission_reference_unchanged': True,
        'figure5_displayed_completed_tracks_unchanged': True,
        'level2_terminal_endpoints_equal_census': len(admitted),
        'figure5_old_matched_seeds': {scale: {d: row['matched_seeds'] for d, row in domains.items()} for scale, domains in old5['cells'].items()},
        'figure5_new_matched_seeds': {scale: {d: row['matched_seeds'] for d, row in domains.items()} for scale, domains in fig5['cells'].items()},
        'figure6_previous_coverage': old6['interim_comparison']['coverage_by_domain'],
        'figure6_current_coverage': expected['coverage_by_domain'],
        'figure6_previous_means': old6['interim_comparison']['means'],
        'figure6_current_means': expected['means'],
        'figure6_eligible_domain_seed_cells': expected['eligible_domain_seed_cells'],
        'figure6_initial_only_domains': expected['initial_only_domains'],
        'outputs_sha256': {str(p.relative_to(ROOT)): sha(p) for p in current_paths + [snap_path]},
        'prior_png_equal': {p.stem: sha(p) == sha(HERE / 'before' / p.relative_to(ROOT)) for p in current_paths if p.suffix == '.png'},
        'unchanged_qwen3b_python_table': True,
    }
    (HERE / 'validation.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: receipt[k] for k in ('level2_terminal_endpoints_equal_census', 'figure6_eligible_domain_seed_cells', 'figure6_initial_only_domains', 'prior_png_equal')}, indent=2))


if __name__ == '__main__':
    main()
