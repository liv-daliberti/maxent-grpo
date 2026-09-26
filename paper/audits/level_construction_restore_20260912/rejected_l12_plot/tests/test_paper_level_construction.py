"""Construction artwork must retain its measured admission evidence and assets."""
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


def test_published_construction_record_validates(renderer, published_record):
    renderer.validate_record(published_record)


@pytest.mark.parametrize('metric', ['level1_pass8', 'level2_pass8'])
def test_changed_development_measurement_is_rejected(renderer, published_record, metric):
    altered = deepcopy(published_record)
    altered['admission_rows'][0][metric] += .01
    with pytest.raises(ValueError, match='differs from frozen admission evidence'):
        renderer.validate_record(altered)


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


def test_fully_rebound_copy_cannot_substitute_the_frozen_snapshot(
        renderer, published_record, tmp_path):
    substitute = tmp_path / 'substituted_snapshot.json'
    shutil.copyfile(renderer.SNAPSHOT, substitute)
    altered = renderer.build_record(json.loads(substitute.read_text()), substitute)
    altered['outputs'] = deepcopy(published_record['outputs'])
    with pytest.raises(ValueError, match='unexpected frozen snapshot'):
        renderer.validate_record(altered)


def test_custom_build_still_rejects_changed_rendered_bytes(renderer, published_record, tmp_path):
    output = tmp_path / 'construction'
    altered = deepcopy(published_record)
    altered['outputs'] = {}
    for suffix in ('.pdf', '.png'):
        destination = output.with_suffix(suffix)
        shutil.copyfile(renderer.OUT.with_suffix(suffix), destination)
        altered['outputs'][str(destination)] = renderer.digest(destination)
    renderer.validate_record(altered, output=output)
    with output.with_suffix('.pdf').open('ab') as handle:
        handle.write(b'changed artwork')
    with pytest.raises(ValueError, match='construction figure output changed'):
        renderer.validate_record(altered, output=output)
