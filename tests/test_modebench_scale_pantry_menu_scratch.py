"""Focused structural checks; native generation/certification is in scratch reports."""
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/modebench_scale_pantry_menu_feasibility_20260912'
spec = importlib.util.spec_from_file_location('pantry_menu_scratch_under_test', BASE / 'scratch_candidate.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def test_only_two_declared_ast_subtrees_change(monkeypatch):
    captured = []
    real_compile = compile
    def capture(source, filename, mode, *args, **kwargs):
        if isinstance(source, ast.Module) and filename.endswith(':two_explicit_ast_changes'):
            captured.append(copy.deepcopy(source))
        return real_compile(source, filename, mode, *args, **kwargs)
    monkeypatch.setattr(c, 'compile', capture, raising=False)
    c.compile_candidate()
    assert len(captured) == 1
    changed = captured[0]
    original = ast.parse(inspect.getsource(c.bridge._candidate))
    old_menu = next(node.value for node in ast.walk(original) if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'menu_size' for t in node.targets))
    old_available = next(node.values[i] for node in ast.walk(original) if isinstance(node, ast.Dict)
        for i, key in enumerate(node.keys) if isinstance(key, ast.Constant) and key.value == 'available_g')
    for node in ast.walk(changed):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'menu_size' for t in node.targets):
            assert ast.unparse(node.value) == '_MENU_SIZES[tier]'
            node.value = copy.deepcopy(old_menu)
        if isinstance(node, ast.Dict):
            for i, key in enumerate(node.keys):
                if isinstance(key, ast.Constant) and key.value == 'available_g':
                    assert ast.unparse(node.values[i]) == '_AVAILABILITY[tier][0] if len(_AVAILABILITY[tier]) == 1 else step * (minimum_steps + rng.choice((2, 3, 4)))'
                    node.values[i] = copy.deepcopy(old_available)
    assert ast.dump(changed, include_attributes=False) == ast.dump(original, include_attributes=False)


def test_original_globals_and_native_enumeration_are_unchanged():
    before = dict(vars(c.bridge))
    made = c.compile_candidate()
    assert dict(vars(c.bridge)) == before
    for name in ('_pantry_allocations', '_pantry_admitted', '_decimal_units',
                 'validate_pantry_plan', 'pantry_prompt', 'SODIUM_CAPS', 'TOTAL_SCALE'):
        assert made.__globals__[name] is before[name]
    assert '_MENU_SIZES' not in vars(c.bridge)
    assert '_AVAILABILITY' not in vars(c.bridge)
    assert hashlib.sha256(Path(c.bridge.__file__).read_bytes()).hexdigest() == c.BASE_SHA


@pytest.mark.parametrize('menus,available', [
    ((7, 8), c.AVAILABILITY),
    ((7, 7, 8, 9), c.AVAILABILITY),
    (c.MENU_SIZES, ((100,),) * 3),
    (c.MENU_SIZES, ((100,), (100,), (100,), (100, 150))),
])
def test_unsupported_structural_law_rejected(menus, available):
    with pytest.raises(AssertionError):
        c.compile_candidate(menus, available)


@pytest.mark.parametrize('phase,expected', [('cost', 16), ('reachability', 436)])
def test_native_scratch_fixture_scope_and_fingerprints(phase, expected):
    report = json.loads((BASE / phase / 'report.json').read_text())
    execution = json.loads((BASE / f'{phase}.execution.json').read_text())
    assert report['status'] == 'all_requested_scratch_quotas_certified'
    assert report['input_pins_unchanged'] and execution['returncode'] == 0
    assert report['original_profile_equivalence'] == {'proposals': 12, 'exact_rows_and_rng_states_equal': True}
    assert report['model_calls'] == report['model_output_grader_calls'] == report['production_rows'] == 0
    identities = set()
    for fixture in report['fixtures']:
        path = Path(fixture['path'])
        assert path.is_relative_to(BASE)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == fixture['sha256']
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == fixture['rows']
        for row in rows:
            tier = row['scale_candidate_tier']; answer = json.loads(row['answer'])
            assert row['scratch_only'] is True and row['scratch_profile'] == c.PROFILES[tier]
            assert len(answer['ingredients']) == c.MENU_SIZES[tier]
            assert all(i['available_g'] in c.AVAILABILITY[tier]
                and i['min_if_used_g'] == 50 and i['step_g'] == 25 for i in answer['ingredients'])
            assert answer['min_ingredients'] == 2 and answer['max_ingredients'] == 4
            assert answer['certified_mode_count'] == row['answer_mode_count']
            answer.pop('instance_id')
            identity = hashlib.sha256(json.dumps(answer, sort_keys=True,
                separators=(',', ':'), allow_nan=False).encode('ascii')).hexdigest()
            assert identity == row['instance_fingerprint'] and identity not in identities
            identities.add(identity)
    assert len(identities) == expected
    assert all(cell['accepted'] == cell['requested'] for cell in report['cells'])
    assert sum(cell['independent_original_verifier_witnesses'] for cell in report['cells']) == sum(
        cell['accepted'] * cell['support'] for cell in report['cells'])


def test_all_required_joint_cells_covered_in_every_profile():
    protocol = json.loads((ROOT / 'var/data/modebench_scale_v1/protocol.json').read_text())
    expected = {tuple(cell['cell']) for cells in protocol['histograms']['pantry'].values() for cell in cells}
    assert len(expected) == 109
    report = json.loads((BASE / 'reachability/report.json').read_text())
    for tier in range(4):
        actual = {(cell['support'], cell['family']) for cell in report['cells'] if cell['tier'] == tier}
        assert actual == expected
