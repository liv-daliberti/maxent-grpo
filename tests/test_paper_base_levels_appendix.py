"""Keep the published 60-cell appendix tied to observed independent K=8 groups."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil

from matplotlib.colors import to_rgba
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def renderer(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'ops'))
    spec = importlib.util.spec_from_file_location(
        'paper_base_levels_appendix_test', ROOT / 'ops/plot_paper_modebench_base_levels_appendix.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def published_record():
    return json.loads((ROOT / 'paper/figures/modebench_base_levels_appendix.json').read_text())


def test_published_record_reauthenticates_all_60_native_receipts(renderer, published_record):
    renderer.validate_record(published_record)
    assert published_record['sampling']['observed_samples'] == 245760
    assert published_record['validation']['frozen_validator_used_in_isolated_interpreter']
    assert published_record['validation']['complete_60_cell_scope']
    assert not published_record['validation']['complete_100_cell_grid']


def test_coordinates_equal_independently_counted_canonical_sets(published_record):
    for point in published_record['points']:
        receipt = json.loads((ROOT / point['receipt']['path']).read_text())
        successes, modes, groups = 0, 0, 0
        for prompt in receipt['prompt_results']:
            for draw in prompt['draws']:
                keys = {json.dumps(attempt['canonical_key'], sort_keys=True)
                        for attempt in draw['attempts'] if attempt['verified']}
                successes += bool(keys)
                modes += len(keys)
                groups += 1
        assert groups == 512  # Four separate groups of eight, never pooled into 32.
        assert point['metrics'] == {'pass8': successes / groups, 'distinct8': modes / groups}


def test_complete_scope_excludes_unadmitted_levels_without_zero_values(renderer, published_record):
    actual = {renderer.base_grid.key(point) for point in published_record['points']}
    omitted = {renderer.base_grid.key(cell) for cell in published_record['omitted_cells']}
    assert actual == renderer.EXPECTED
    assert omitted == renderer.OMITTED
    assert len(actual) == 60 and len(omitted) == 40
    assert not published_record['missing_cells']
    assert published_record['status'] == 'complete'
    assert published_record['required_cells'] == published_record['complete_cells'] == 60
    assert all(cell['reason'] == 'awaiting_dataset_admission' and 'metrics' not in cell
               for cell in published_record['omitted_cells'])
    assert all(value == {'complete': 15, 'required': 15}
               for value in published_record['coverage_by_model'].values())


def test_five_domain_axes_preserve_coordinates_markers_and_domain_colors(renderer, published_record):
    figure = renderer.build_figure(published_record)
    try:
        assert len(figure.axes) == 5
        assert [[text.get_text() for text in legend.get_texts()] for legend in figure.legends] == [
            ['Level 1', 'Level 2', 'Level 3'], ['0.5B', '3B', '7B', '14B']]
        for axis, domain in zip(figure.axes, renderer.DOMAINS):
            measured = [point for point in published_record['points'] if point['domain'] == domain]
            assert len(axis.collections) == len(measured) == 12
            assert axis.get_facecolor() == to_rgba(renderer.DOMAIN_BACKGROUNDS[domain])
            assert not axis.lines and not axis.patches
            assert axis.get_xlim() == (0, 1)
            assert axis.get_ylim() == (0, 1.2)
            displayed = sorted(tuple(row) for collection in axis.collections
                               for row in collection.get_offsets().tolist())
            expected = sorted((point['metrics']['pass8'], point['metrics']['distinct8'])
                              for point in measured)
            assert displayed == expected
            assert all(not len(collection.get_facecolors()) for collection in axis.collections)
            assert all(y <= axis.get_ylim()[1] for _, y in expected)
        assert any('Levels 4–5' in text.get_text() for text in figure.texts)
    finally:
        renderer.plt.close(figure)


@pytest.mark.parametrize('change', ['drop_cell', 'duplicate_cell', 'add_level4'])
def test_plot_refuses_incomplete_or_expanded_scopes(renderer, published_record, change):
    altered = deepcopy(published_record)
    if change == 'drop_cell':
        altered['points'].pop()
    elif change == 'duplicate_cell':
        altered['points'][-1] = deepcopy(altered['points'][0])
    else:
        altered['points'][0]['level'] = 'level4'
    with pytest.raises(ValueError, match='complete 60-cell'):
        renderer.build_figure(altered)


def test_rebound_receipt_with_corrupted_summary_fails_native_authentication(renderer, tmp_path):
    source = json.loads(renderer.SOURCE.read_text())
    binding = source['receipts'][0]
    receipt = json.loads((ROOT / binding['path']).read_text())
    receipt['metrics']['distinct8'] += .01
    changed_receipt = tmp_path / 'changed_receipt.json'
    changed_receipt.write_text(json.dumps(receipt))
    binding.update(renderer.base_grid.binding(changed_receipt))
    changed_source = tmp_path / 'changed_source.json'
    changed_source.write_text(json.dumps(source))
    with pytest.raises(ValueError, match='summary metrics differ from saved prompts'):
        renderer.build_record(changed_source)


@pytest.mark.parametrize('suffix', ['.pdf', '.png'])
def test_identical_output_cannot_replace_the_canonical_asset(renderer, published_record, tmp_path, suffix):
    altered = deepcopy(published_record)
    original = renderer.OUT.with_suffix(suffix).resolve()
    substitute = tmp_path / original.name
    shutil.copyfile(original, substitute)
    altered['outputs'][str(substitute)] = altered['outputs'].pop(str(original))
    with pytest.raises(ValueError, match='published PDF and PNG bindings'):
        renderer.validate_record(altered)


def test_rebound_identical_source_cannot_replace_the_canonical_source(renderer, published_record, tmp_path):
    substitute = tmp_path / 'substituted_source.json'
    shutil.copyfile(renderer.SOURCE, substitute)
    altered = deepcopy(published_record)
    altered['source'] = renderer.base_grid.binding(substitute)
    with pytest.raises(ValueError, match='unexpected frozen source'):
        renderer.validate_record(altered)
