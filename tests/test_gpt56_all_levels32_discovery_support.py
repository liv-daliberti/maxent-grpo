"""Selection is nested and outcome independent; witnesses use hosted surfaces."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest
from ops import gpt56_all_levels32_discovery_support as support


def selection():
    source = [{'level': level, 'domain': domain, 'row_index': i, 'problem': str(i)}
              for level in support.LEVELS for domain in support.DOMAINS for i in range(128)]
    new, combined = [], []
    for level in support.LEVELS:
        for domain in support.DOMAINS:
            ranked = sorted((r for r in source if support.identity(r)[:2] == (level, domain)), key=lambda r: (
                hashlib.sha256(f"20260911\0{level}\0{domain}\0{r['row_index']}".encode()).hexdigest(), support.identity(r)))
            new.extend(ranked[16:32])
            combined.extend(ranked[:32])
    return new, combined, source


def test_nested_selection_preserves_first16_adds_next16_in_every_cell():
    new, combined, source = selection()
    support.audit_selection(new, combined, source)
    assert len(new) == 240 and len(combined) == 480
    assert len({support.identity(r) for r in combined} - {support.identity(r) for r in new}) == 240
    support.audit_selection(list(reversed(new)), list(reversed(combined)), list(reversed(source)))


@pytest.mark.parametrize('mutation,match', [('outcome_replacement', 'ranks17'), ('changed_prompt', 'source bytes'),
                                         ('missing_cell', '240 unique'), ('missing_candidate', '128 distinct')])
def test_certificate_rejects_selection_or_row_revision_changes(mutation, match):
    new, combined, source = deepcopy(selection())
    if mutation == 'outcome_replacement':
        replacement = next(r for r in combined if r not in new)
        new[0] = replacement
    elif mutation == 'changed_prompt':
        # Break sharing from deepcopy so only the selected copy is altered.
        new[0] = {**new[0], 'problem': 'different wording'}
    elif mutation == 'missing_cell':
        new = new[16:]
    elif mutation == 'missing_candidate':
        source = source[1:]
    with pytest.raises(ValueError, match=match):
        support.audit_selection(new, combined, source)


@pytest.mark.parametrize('level,width', [(2, 6), (3, 7)])
def test_higher_level_pantry_certificates_cover_full_ingredient_inventory(level, width):
    ingredients = [chr(ord('a') + i) for i in range(width)]
    valid = {('a', 'b'), ('a', ingredients[-1])}
    digest = hashlib.sha256('\n'.join(sorted('+'.join(ids) for ids in valid)).encode()).hexdigest()
    row = {'level': level, 'domain': 'pantry_plan', 'row_index': 3,
           'answer': json.dumps({'certified_mode_count': 2, 'certified_support_sha256': digest,
                                 'ingredients': [{'id': i} for i in ingredients],
                                 'min_ingredients': 2, 'max_ingredients': 2})}
    projected, seen = [], []
    def project(selected, spec):
        projected.append(selected)
        return SimpleNamespace(allocations_g=[(i, 50) for i in selected]) if selected in valid else None
    def grade(level_arg, domain, source, text):
        assert level_arg == level and text.startswith('\\boxed{')
        seen.append(text)
        allocation = text[len('\\boxed{'):-1]
        ids = tuple(piece.split('=')[0] for piece in allocation.split(';'))
        key = 'pantry_plan:pantry-v1:' + '+'.join(ids) if ids in valid else None
        return {'verified': key is not None, 'canonical_key': key, 'graded_text': text}
    ref = support.certify_pantry_allocation(row, grade, project)
    assert len(projected) == width * (width - 1) // 2 and len(seen) == 4
    assert ('a', ingredients[-1]) in projected
    assert ref['support_count'] == 2 and ref['support_kind'] == 'certified_lower_bound'
    assert ref['row_sha256'] == support.object_sha(row)
    assert all(w['text'].startswith('\\boxed{') and '=50' in w['text'] for w in ref['witnesses'].values())
    altered = json.loads(row['answer'])
    altered['certified_support_sha256'] = 'incorrect'
    row['answer'] = json.dumps(altered)
    with pytest.raises(ValueError, match='digest differs'):
        support.certify_pantry_allocation(row, grade, project)
