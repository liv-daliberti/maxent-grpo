"""An incomplete saved run is live only while its recorded collector still exists."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
spec = importlib.util.spec_from_file_location('reasoning_runner', ROOT / 'ops/run_hosted_reasoning_off.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fake_collector(tmp_path, actual_command=None, state='S'):
    run = tmp_path / 'run'
    run.mkdir()
    command = ['/python', '/collector.py', '--output', str(run)]
    (run / 'active_collection.json').write_text(json.dumps({'pid': 42, 'command': command}))
    proc = tmp_path / 'proc'
    process = proc / '42'
    process.mkdir(parents=True)
    (process / 'cmdline').write_bytes(b'\0'.join(s.encode() for s in (actual_command or command)) + b'\0')
    (process / 'stat').write_text(f'42 (python) {state} 1 2 3')
    return run, proc


def test_saved_collector_command_is_live(tmp_path):
    run, proc = fake_collector(tmp_path)
    assert m.collector_running(run, proc)


@pytest.mark.parametrize('failure', ['exited', 'pid_reused', 'zombie', 'malformed_marker'])
def test_dead_or_unrelated_process_is_not_collecting(tmp_path, failure):
    run, proc = fake_collector(tmp_path, actual_command=['/python', '/unrelated.py'] if failure == 'pid_reused' else None,
                               state='Z' if failure == 'zombie' else 'S')
    if failure == 'exited':
        (proc / '42' / 'cmdline').unlink()
    elif failure == 'malformed_marker':
        (run / 'active_collection.json').write_text('{')
    assert not m.collector_running(run, proc)


def test_status_preserves_completion_and_provider_block(tmp_path, monkeypatch):
    entries = []
    for name, completed in [('complete', 3840), ('blocked', 183), ('stopped', 3839), ('live', 12)]:
        run = tmp_path / name
        run.mkdir()
        (run / 'status.json').write_text(json.dumps({'complete': completed == 3840, 'completed_samples': completed}))
        entries.append({'slug': name, 'model': name, 'run_directory': str(run)})
    blocked = 'collection_paused_control_not_reliably_honored'
    (tmp_path / 'blocked' / 'provider_control_violation.json').write_text(json.dumps({'status': blocked}))
    monkeypatch.setattr(m, 'registry_for', lambda base: {'runs': entries})
    monkeypatch.setattr(m, 'collector_running', lambda run: run.name == 'live')
    result = m.status(tmp_path)
    assert [r['collection_state'] for r in result['runs']] == ['complete', blocked, 'stopped_incomplete', 'collecting']
    assert [r['admissible_for_final_comparison'] for r in result['runs']] == [True, False, False, False]
    assert result['completed'] == 3840 + 183 + 3839 + 12
