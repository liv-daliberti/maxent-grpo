"""Native echoed controls must match the registered no-reasoning curve."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops'))
import audit_gpt56_temperature_controls as controls


@pytest.mark.parametrize('change', [None, 'returned_temperature', 'returned_effort', 'missing_effort',
                                  'requested_temperature', 'requested_effort'])
def test_native_controls_require_exact_requested_echo(tmp_path, monkeypatch, change):
    manifest = {'experiment_condition': 'gpt56_none_temperature_curve_v1',
                'reasoning_effort': 'none', 'temperature': 1.5}
    expected, records, raw = {}, [], {}
    for index in range(960):
        sid = str(index)
        relative = f'raw_responses/{sid}__01.json'
        expected[sid] = {'request': {'temperature': 1.5, 'reasoning': {'effort': 'none'}}}
        records.append({'sample_id': sid, 'raw_receipt': relative})
        raw[relative] = {'response': {'temperature': 1.5, 'reasoning': {'effort': 'none'},
                                       'top_p': 0.98, 'model': 'gpt-5.6-sol-snapshot'}}
    first = raw['raw_responses/0__01.json']['response']
    if change == 'returned_temperature': first['temperature'] = 1.0
    elif change == 'returned_effort': first['reasoning']['effort'] = 'medium'
    elif change == 'missing_effort': first.pop('reasoning')
    elif change == 'requested_temperature': expected['0']['request']['temperature'] = 2.0
    elif change == 'requested_effort': expected['0']['request']['reasoning']['effort'] = 'medium'
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    (tmp_path / 'samples.jsonl').write_text(''.join(json.dumps(record) + '\n' for record in records))
    inventory = {'directory': tmp_path, 'manifest': manifest, 'expected': expected}
    monkeypatch.setattr(controls, 'load_inventory', lambda *args, **kwargs: inventory)
    monkeypatch.setattr(controls, 'validate_native_records', lambda *args, **kwargs: raw)
    if change:
        with pytest.raises(ValueError): controls.audit(tmp_path)
    else:
        result = controls.audit(tmp_path)
        assert result['responses'] == 960
        assert result['returned_temperature_counts'] == {'1.5': 960}
        assert result['returned_reasoning_effort_counts'] == {'none': 960}
