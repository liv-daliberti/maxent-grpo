"""Do not repeat unknown retry attempts or accept changed saved requests."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops'))
import retry_opus5_python_until_valid as retry


@pytest.mark.parametrize('change', [None, 'unknown_start', 'duplicate_start', 'missing_start',
                                  'payload', 'request_digest', 'timestamp', 'unregistered_slot'])
def test_retry_resume_authenticates_every_physical_attempt(tmp_path, monkeypatch, change):
    monkeypatch.setattr(retry, 'OUTPUT', tmp_path)
    (tmp_path / 'raw_responses').mkdir()
    payload = {'model': 'claude-opus-5', 'messages': [{'role': 'user', 'content': 'frozen'}]}
    item = {'sample_id': 'a', 'request': payload, 'request_sha256': retry.sha(payload)}
    event = {'event': 'request_started', 'sample_id': 'a', 'attempt': 2,
             'at_utc': '2026-09-11T17:00:00+00:00', 'request_sha256': item['request_sha256']}
    receipt = {'sample_id': 'a', 'attempt': 2, 'relative_path': 'raw_responses/a__02.json',
               'request': payload, 'request_sha256': item['request_sha256'],
               'started_at_utc': event['at_utc'], 'http_status': 500}
    events = [event]
    if change == 'duplicate_start': events.append(dict(event))
    elif change == 'missing_start': events.clear()
    elif change == 'payload': receipt['request'] = {'different': True}
    elif change == 'request_digest': receipt['request_sha256'] = 'changed'
    elif change == 'timestamp': receipt['started_at_utc'] = '2026-09-11T18:00:00+00:00'
    elif change == 'unregistered_slot': events[0]['sample_id'] = 'b'
    (tmp_path / 'events.jsonl').write_text(''.join(json.dumps(record) + '\n' for record in events))
    if change != 'unknown_start': (tmp_path / 'raw_responses/a__02.json').write_text(json.dumps(receipt))
    slots = [{'sample_id': 'a', 'initial_accepted': False}]
    if change:
        with pytest.raises(ValueError): retry.validate_existing_attempts({'a': item}, slots)
    else:
        assert retry.validate_existing_attempts({'a': item}, slots) == {('a', 2)}
