"""Guard fresh response accounting, fixed selection, and native L1 interfaces."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
from ops import prepare_gpt56_all_levels512_discovery as stage


def inputs():
    rows = [json.loads(line) for line in (stage.SOURCE / 'rows.jsonl').read_text().splitlines()]
    requests = {stage.identity(r): r for r in
                (json.loads(line) for line in (stage.SOURCE / 'requests.jsonl').read_text().splitlines())
                if r['sample_index'] == 0}
    return rows, requests


def test_target_accounts_for_every_existing_and_new_slot():
    old = new = 0
    for level in stage.LEVELS:
        for domain in stage.DOMAINS:
            start = stage.draw_start(level, domain)
            retained = set(range(start))
            fresh = set(range(start, 512))
            assert retained.isdisjoint(fresh)
            assert retained | fresh == set(range(512))
            old += 16 * len(retained)
            new += 16 * len(fresh)
    assert old == 20480 and new == 102400 and old + new == 122880


def test_selection_does_not_depend_on_outcomes_specs_or_input_order():
    rows, _ = inputs()
    selected, _ = stage.selected_cell(rows, 1, 'graph_coloring')
    altered = deepcopy(rows)
    for row in altered:
        row['answer'] = 'changed outcome specification'
        row['metadata'] = {'answer_mode_count': -100}
        row['problem'] = 'different content must not change SHA identity ranking'
    changed, _ = stage.selected_cell(list(reversed(altered)), 1, 'graph_coloring')
    assert [stage.identity(r) for r in selected] == [stage.identity(r) for r in changed]
    with pytest.raises(ValueError, match='all128'):
        stage.selected_cell(rows[1:], 1, 'graph_coloring')


@pytest.mark.parametrize('domain', stage.DOMAINS)
def test_new_l1_slots_preserve_each_native_interface_exactly(domain):
    rows, templates = inputs()
    selected, _ = stage.selected_cell(rows, 1, domain)
    # One representative row exercises all512 new indices while checking exact
    # original payload bytes, including L1 Pantry's transformed mask interface.
    row = selected[0]
    reference = templates[stage.identity(row)]
    expanded = stage.expand([row], templates, 0)
    assert len(expanded) == 512 and {r['sample_index'] for r in expanded} == set(range(512))
    assert len({r['sample_id'] for r in expanded}) == 512
    for item in expanded:
        assert item['request'] == reference['request']
        assert item['request_sha256'] == reference['request_sha256']
        assert item['row_sha256'] == reference['row_sha256']
        assert item['choice_index'] == 0 and item['group_id'] == item['sample_id']
        assert item['level'] == 1 and item['domain'] == domain
    assert 'temperature' not in reference['request'] and 'top_p' not in reference['request']
    assert reference['request']['reasoning'] == {'effort': 'medium'}


def test_original_frozen_templates_match_all_new_l1_messages():
    script = r'''
from pathlib import Path
import json, sys
root=Path(sys.argv[1]);source=root/'artifacts/frontier_modebench_gpt56sol_20260911'
code=root/'artifacts/modebench_discovery_curves_20260911/hosted/gpt56sol/original/code'
sys.path[:0]=[str(code/'ops'),str(code/'src')]
from frontier_modebench_contract import make_messages,profile_metadata
rows={(r['level'],r['domain'],r['row_index']):r for r in
      (json.loads(l) for l in (source/'rows.jsonl').read_text().splitlines())}
checked=0
for line in (source/'requests.jsonl').read_text().splitlines():
 r=json.loads(line)
 if r['level']==1 and r['sample_index']==0:
  row=rows[(r['level'],r['domain'],r['row_index'])]
  assert make_messages(1,r['domain'],row)==r['request']['input']
  checked+=1
assert checked==640
p=profile_metadata(1,'pantry_plan')
assert p['response_surface']=='six_bit_ingredient_support' and p['trusted_quantity_projection'] is True
print('640 original L1 prompt interfaces match frozen native templates')
'''
    result = subprocess.run([sys.executable, '-c', script, str(stage.ROOT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_schedule_prioritizes_service_work_and_covers_levels_early():
    work = {f'L{level}_{domain}': level * 1000 + j
            for level in stage.LEVELS for j, domain in enumerate(stage.DOMAINS)}
    order = stage.cohort_order(work)
    assert len(order) == len(set(order)) == 15
    assert {name[1] for name in order[:3]} == {'1', '2', '3'}
    assert all(name.endswith('pantry_plan') for name in order[:3])
    remaining = [work[name] for name in order[3:]]
    assert remaining == sorted(remaining, reverse=True)
