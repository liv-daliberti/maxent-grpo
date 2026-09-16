"""Add the completed E80-R1 reference row without refreshing E118 progress."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import shutil

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    'figure5_plotter', ROOT / 'ops/exp_scaling/plot_paper_e118_all_scale_progress.py',
)
plotter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plotter)
source = AUDIT / 'before/paper/figures/e118_all_scale_factorial_progress.json'
original = json.loads(source.read_text())
record = copy.deepcopy(original)
plotter.add_qwen3b_main_figure_track(record)
record['absolute_cross_domain_average'] = plotter.absolute_cross_domain_averages(record)
for scale in ('qwen05b', 'falcon1b'):
    assert record['cells'][scale] == original['cells'][scale]
    assert record['absolute_cross_domain_average'][scale] == original['absolute_cross_domain_average'][scale]
for domain in plotter.DOMAINS:
    before = original['cells']['qwen3b'][domain]
    after = record['cells']['qwen3b'][domain]
    assert after['matched_seeds'] == before['matched_seeds']
    assert after['replay_maxrl_minus_maxrl'] == before['replay_maxrl_minus_maxrl']
    for method in ('maxrl', 'replay_maxrl'):
        assert after['methods'][method] == before['methods'][method]
        assert after['method_seeds'][method] == before['method_seeds'][method]
assert record['cross_domain_average'] == original['cross_domain_average']
assert record['endpoint_integrity_audit'] == original['endpoint_integrity_audit']
assert record['incomplete_scales'] == ['qwen3b']
plotter.render_main_figure(record, record['absolute_cross_domain_average'])
for path in (plotter.OUT.with_suffix('.json'), plotter.APPENDIX_OUT.with_suffix('.json')):
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
for name in ('e118_all_scale_factorial_progress.pdf', 'e118_all_scale_factorial_progress.json',
             'e118_scale_extensions_appendix.json'):
    shutil.copy2(ROOT / 'paper/figures' / name, ROOT / 'paper/mathai2026/figures' / name)
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
summary = {
    'baseline_record_sha256': digest(source),
    'new_record_sha256': digest(plotter.OUT.with_suffix('.json')),
    'existing_model_cells_and_average_arrays_unchanged': True,
    'all_maxrl_cells_and_average_arrays_unchanged': True,
    'existing_falcon_seed_exclusion_unchanged': True,
    'qwen3b_drgrpo_seeds': [70, 71, 72, 73, 74],
    'qwen3b_reference_source': record['qwen3b_reference_source'],
    'qwen3b_averages': record['absolute_cross_domain_average']['qwen3b'],
    'qwen3b_maxrl_cross_domain_aggregate': False,
    'source_sha256': record['source_sha256'],
}
(AUDIT / 'validation.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({metric: {method: row['mean'] for method, row in methods.items()}
                 for metric, methods in summary['qwen3b_averages'].items()}, indent=2))
