"""Support certificates follow the native L1 surface and frozen cohort selection."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from ops.gpt56_all_levels_discovery_support import (
    DOMAINS, audit_selection, certify_surface_row, python_witnesses,
)


def digest(keys):
    return hashlib.sha256('\n'.join(sorted(keys)).encode()).hexdigest()


def row(domain, spec):
    return {'level': 1, 'domain': domain, 'row_index': 3,
            'problem': 'Frozen input', 'answer': json.dumps(spec)}


def test_selection_rejects_changed_prompt_and_outcome_dependent_replacement():
    source = [{'level': 1, 'domain': d, 'row_index': i, 'problem': str(i)}
              for d in DOMAINS for i in range(128)]
    selected = []
    for d in DOMAINS:
        selected.extend(sorted((r for r in source if r['domain'] == d), key=lambda r: (
            hashlib.sha256(f"20260911\0{r['level']}\0{r['domain']}\0{r['row_index']}".encode()).hexdigest(),
            (r['level'], r['domain'], r['row_index'])))[:16])
    audit_selection(selected, source)
    altered = [dict(r) for r in selected]
    altered[0]['problem'] = 'Changed wording'
    with pytest.raises(ValueError, match='differs from current native source'):
        audit_selection(altered, source)
    replacement = next(r for r in source if r['domain'] == DOMAINS[0] and r not in selected)
    with pytest.raises(ValueError, match='identities differ'):
        audit_selection([replacement] + selected[1:], source)


def test_python_certifies_two_realizable_vectors_not_metadata_cartesian_total():
    keys = ['python_factor:2,3,2', 'python_factor:4,5,6']
    spec = {'cases': [8, 15, 12], 'num_modes': 12,
            'num_externally_certified_modes': 2, 'certified_mode_key_sha256': digest(keys)}
    witnesses = python_witnesses(spec)
    assert sorted(witnesses) == keys
    for key, boxed in witnesses.items():
        # Only helper-created fixed expressions execute in this unit test.
        function = eval(boxed[len('\\boxed{'):-1], {'__builtins__': {}})
        assert 'python_factor:' + ','.join(str(function(n)) for n in spec['cases']) == key
    def grade(level, domain, r, text):
        key = next(k for k, value in witnesses.items() if value == text)
        return {'verified': True, 'canonical_key': key, 'graded_text': text}
    ref = certify_surface_row(row('python_factors', spec), grade, None)
    assert ref['support_count'] == 2
    assert ref['support_kind'] == 'certified_lower_bound'
    spec['certified_mode_key_sha256'] = 'tampered'
    with pytest.raises(ValueError, match='disagree with source certificate'):
        python_witnesses(spec)


def test_pantry_uses_every_native_unboxed_mask_and_retains_bound_label():
    supports = {'001100': 'a+b', '101010': 'a+c+d'}
    spec = {'certified_mode_count': 2, 'certified_support_sha256': digest(supports.values())}
    seen = []
    def grade(level, domain, r, text):
        assert level == 1 and domain == 'pantry_plan'
        assert len(text) == 6 and set(text) <= {'0', '1'}
        seen.append(text)
        key = 'pantry_plan:pantry-v1:' + supports[text] if text in supports else None
        return {'verified': key is not None, 'canonical_key': key, 'graded_text': 'projected allocation'}
    ref = certify_surface_row(row('pantry_plan', spec), grade, None)
    assert set(seen) == {f'{n:06b}' for n in range(64)}
    assert len(seen) == 66  # full enumeration plus independent witness recheck
    assert ref['support_count'] == 2 and ref['support_kind'] == 'certified_lower_bound'
    assert {w['text'] for w in ref['witnesses'].values()} == set(supports)


def test_mathir_revalidates_enumerated_witnesses_through_hosted_adapter():
    key = 'mathir:path-one'
    spec = {'valid_mode_count': 1, 'valid_mode_key_sha256': digest([key])}
    def enumerate_routes(spec):
        return [SimpleNamespace(canonical_key=key, action_ids=('A', 'C'))]
    def grade(level, domain, r, text):
        assert text == '\\boxed{A;C}'
        return {'verified': False, 'canonical_key': None, 'graded_text': text}
    with pytest.raises(ValueError, match='rejects witness'):
        certify_surface_row(row('mathir', spec), grade, enumerate_routes)
