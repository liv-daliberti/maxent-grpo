"""Authenticate the single-model level scatter and exclude fabricated cells."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def renderer(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'ops'))
    spec = importlib.util.spec_from_file_location(
        'paper_level_construction_test', ROOT / 'ops/plot_paper_modebench_level_construction.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def published_record():
    return json.loads((ROOT / 'paper/figures/modebench_level_construction.json').read_text())


def test_published_record_reauthenticates_native_collection(renderer, published_record):
    renderer.validate_record(published_record)
    assert published_record['sampling']['observed_samples'] == 61440
    assert published_record['validation']['frozen_validator_used_in_isolated_interpreter']


def test_coordinates_equal_independently_counted_canonical_sets(published_record):
    for point in published_record['points']:
        receipt_path = Path(point['receipt']['path'])
        receipt = json.loads((ROOT / receipt_path).read_text())
        successes, modes, groups = 0, 0, 0
        for prompt in receipt['prompt_results']:
            for draw in prompt['draws']:
                keys = {json.dumps(attempt['canonical_key'], sort_keys=True)
                        for attempt in draw['attempts'] if attempt['verified']}
                successes += bool(keys)
                modes += len(keys)
                groups += 1
        assert groups == 512  # 128 prompts, four separate K=8 groups; never pooled K=32.
        assert point['metrics'] == {'pass8': successes / groups, 'distinct8': modes / groups}


def test_one_model_scope_contains_all_three_levels(renderer, published_record):
    expected = {('7b', level, domain) for level in ('level1', 'level2', 'level3')
                for domain in renderer.DOMAINS}
    actual = {renderer.base_grid.key(point) for point in published_record['points']}
    assert actual == expected == renderer.EXPECTED
    assert not published_record['missing_cells']
    assert published_record['excluded_levels'] == ['level4', 'level5']
    assert published_record['required_cells'] == 15
    assert published_record['complete_cells'] == 15
    assert published_record['status'] == 'complete'
    assert published_record['validation']['complete_15_cell_scope']


def test_five_domain_axes_plot_only_measured_coordinates(renderer, published_record):
    figure = renderer.build_figure(published_record)
    try:
        assert len(figure.axes) == 5
        assert len(figure.legends[0].get_texts()) == 3
        for axis, domain in zip(figure.axes, renderer.DOMAINS):
            measured = [point for point in published_record['points'] if point['domain'] == domain]
            assert len(axis.collections) == len(measured)
            assert not axis.lines and not axis.patches  # No connections or admission band.
            assert axis.get_xlim() == (0, 1)
            assert axis.get_ylim() == (0, 1.2)
            assert all(0 <= p['metrics']['distinct8'] <= axis.get_ylim()[1] for p in measured)
            displayed = sorted(tuple(row) for collection in axis.collections
                               for row in collection.get_offsets().tolist())
            expected = sorted((point['metrics']['pass8'], point['metrics']['distinct8'])
                              for point in measured)
            assert displayed == expected
            assert all(len(collection.get_facecolors()) == 1
                       and collection.get_facecolors()[0][3] == 1
                       for collection in axis.collections)
            assert not any('pending' in text.get_text() for text in axis.texts)
    finally:
        renderer.plt.close(figure)


@pytest.mark.parametrize('metric', ['pass8', 'distinct8'])
def test_coordinate_drift_is_rejected(renderer, published_record, metric):
    altered = deepcopy(published_record)
    altered['points'][0]['metrics'][metric] += .01
    with pytest.raises(ValueError, match='differs from native frozen-base evidence'):
        renderer.validate_record(altered)


def test_rebound_receipt_with_corrupted_summary_fails_native_recomputation(renderer, tmp_path):
    source = json.loads(renderer.SOURCE.read_text())
    binding = source['receipts'][0]
    receipt = json.loads((ROOT / binding['path']).read_text())
    receipt['metrics']['pass8'] += .01
    changed_receipt = tmp_path / 'changed_receipt.json'
    changed_receipt.write_text(json.dumps(receipt))
    binding.update(renderer.base_grid.binding(changed_receipt))
    changed_source = tmp_path / 'changed_source.json'
    changed_source.write_text(json.dumps(source))
    with pytest.raises(ValueError, match='summary metrics differ from saved prompts'):
        renderer.build_record(changed_source)


def test_another_model_is_rejected_before_scoping(renderer, tmp_path):
    source = json.loads(renderer.SOURCE.read_text())
    source['receipts'][0]['model_label'] = '14b'
    changed_source = tmp_path / 'other_model.json'
    changed_source.write_text(json.dumps(source))
    with pytest.raises(ValueError, match='only frozen Qwen-7B receipts'):
        renderer.build_record(changed_source)


@pytest.mark.parametrize('suffix', ['.pdf', '.png'])
def test_identical_output_at_another_path_cannot_replace_the_published_asset(
        renderer, published_record, tmp_path, suffix):
    altered = deepcopy(published_record)
    original = renderer.OUT.with_suffix(suffix).resolve()
    substitute = tmp_path / original.name
    shutil.copyfile(original, substitute)
    bound_hash = altered['outputs'].pop(str(original))
    assert renderer.digest(substitute) == bound_hash
    altered['outputs'][str(substitute)] = bound_hash
    with pytest.raises(ValueError, match='requires its published PDF and PNG bindings'):
        renderer.validate_record(altered)


def test_rebound_identical_source_cannot_replace_the_canonical_source(renderer, published_record, tmp_path):
    substitute = tmp_path / 'substituted_source.json'
    shutil.copyfile(renderer.SOURCE, substitute)
    altered = deepcopy(published_record)
    altered['source'] = renderer.base_grid.binding(substitute)
    with pytest.raises(ValueError, match='unexpected frozen source'):
        renderer.validate_record(altered)


def test_changed_rendered_bytes_in_custom_output_are_rejected(renderer, published_record, tmp_path):
    output = tmp_path / 'construction'
    altered = deepcopy(published_record)
    altered['outputs'] = {}
    for suffix in ('.pdf', '.png'):
        destination = output.with_suffix(suffix)
        shutil.copyfile(renderer.OUT.with_suffix(suffix), destination)
        altered['outputs'][str(destination)] = renderer.digest(destination)
    with output.with_suffix('.pdf').open('ab') as handle:
        handle.write(b'changed artwork')
    with pytest.raises(ValueError, match='construction figure output changed'):
        renderer.validate_record(altered, output=output)
