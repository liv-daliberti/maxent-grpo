#!/usr/bin/env python3
"""Collect the user-requested T=0 extension using the unchanged frozen cohort."""
import getpass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'ops/evaluate_frontier_modebench.py').is_file())
sys.path.insert(0, str(ROOT / 'ops'))
from evaluate_frontier_modebench import atomic, file_sha, now

BASE = ROOT / 'artifacts/frontier_temperature_20260911'
RUN = BASE / 'gpt56_none_t0p0'
PROBE = BASE / 'gpt56_zero_control_probe'


def main():
    key = os.environ.get('AZURE_OPENAI_API_KEY') or getpass.getpass('Azure API key (hidden): ')
    if not key:
        raise ValueError('Missing API credential')
    env = os.environ.copy()
    env['AZURE_OPENAI_API_KEY'] = key
    frozen = BASE / 'collection_code/run_gpt56_zero_extension.py'
    if frozen.exists() and file_sha(frozen) != file_sha(__file__):
        raise ValueError('Frozen extension collector differs')
    if not frozen.exists():
        shutil.copyfile(__file__, frozen)
    logs = BASE / 'collection_logs' / ('gpt56_zero_' + now().replace(':', '-'))
    logs.mkdir(parents=True)

    def execute(name, args):
        command = [sys.executable, *map(str, args)]
        log = logs / (name + '.log')
        start = now()
        with log.open('w') as handle:
            result = subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=handle, stderr=subprocess.STDOUT)
        receipt = {'stage': name, 'condition': RUN.name, 'started_at_utc': start,
                   'completed_at_utc': now(), 'exit_code': result.returncode,
                   'command': command, 'log': str(log.relative_to(BASE)), 'log_sha256': file_sha(log)}
        print(json.dumps(receipt), flush=True)
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {log}')
        return receipt

    execute('capability', [ROOT / 'ops/probe_gpt56_temperature.py', '--output', PROBE,
                          '--temperatures', '0.0', '--reasoning-efforts', 'medium', 'none'])
    results = json.loads((PROBE / 'probe_results.json').read_text())
    accepted = [r for r in results['results'] if r['reasoning_effort'] == 'none'
                and r['requested_temperature'] == 0.0 and r['http_status'] == 200
                and r['returned_temperature'] == 0.0 and r['returned_reasoning_effort'] == 'none']
    if len(accepted) != 1:
        raise ValueError('The requested zero-temperature condition is not supported')
    execute('prepare', [ROOT / 'ops/prepare_gpt56_temperature_curve.py', '--temperature', '0.0', '--output', RUN])
    runner = RUN / 'code/ops/evaluate_frontier_modebench.py'
    if not (RUN / 'preflight_result.json').exists():
        receipt = execute('preflight', [runner, 'run', '--output', RUN, '--workers', '1',
                          '--max-new', '1', '--max-attempts', '2', '--request-timeout', '600'])
        status = json.loads((RUN / 'status.json').read_text())
        receipt['terminal_samples'] = status['completed_samples']
        atomic(RUN / 'preflight_result.json', receipt)
    preflight = json.loads((RUN / 'preflight_result.json').read_text())
    if preflight['exit_code'] != 0 or preflight['terminal_samples'] != 1:
        raise ValueError('Expected one retained preflight sample')
    request = json.loads((RUN / 'requests.jsonl').read_text().splitlines()[0])
    sample = json.loads((RUN / 'sample_receipts' / (request['sample_id'] + '.json')).read_text())
    raw_path = RUN / sample['raw_receipt']
    raw = json.loads(raw_path.read_text())
    if (raw['http_status'] != 200 or raw['response'].get('temperature') != 0.0
            or raw['response'].get('reasoning', {}).get('effort') != 'none'
            or sample['request_sha256'] != request['request_sha256']):
        raise ValueError('Preflight does not authenticate exact requested controls')
    gate_path = BASE / 'gpt56_zero_extension_collection_gate.json'
    if not gate_path.exists():
        condition = {'condition': RUN.name, 'temperature': 0.0, 'reasoning_effort': 'none',
                     'registered_responses': 960, 'preflight_result_sha256': file_sha(RUN / 'preflight_result.json'),
                     'preflight_receipt': str(raw_path.relative_to(BASE)), 'preflight_receipt_sha256': file_sha(raw_path)}
        for name in ('manifest', 'requests', 'rows'):
            condition[name + '_sha256'] = file_sha(RUN / (name + ('.json' if name == 'manifest' else '.jsonl')))
        sources = ['gpt56_curve_collection_gate.json', 'gpt56_requested_grid_probe/probe_results.json',
                   'gpt56_zero_control_probe/probe_results.json', 'collection_code/run_gpt56_zero_extension.py']
        gate = {'schema': 'frontier-gpt56-temperature-extension-gate-v1',
                'status': 'authorized_supported_controls', 'created_at_utc': now(),
                'model': 'gpt-5.6-sol', 'reasoning_effort': 'none', 'temperatures': [0.0],
                'total_registered_responses': 960, 'responses_per_condition': 960,
                'prompts_per_condition': 120, 'draws_per_prompt': 8, 'domains': 5, 'levels': [1, 2, 3],
                'authorization': {'scope': 'User requested adding temperature 0 to Figure 8 using the same 120 prompts and eight draws.',
                                  'unsupported_conditions': 'Fresh probes reject 2.5 and medium reasoning with nondefault temperatures.'},
                'conditions': [condition], 'evidence': {name: file_sha(BASE / name) for name in sources}}
        atomic(gate_path, gate)
    status = json.loads((RUN / 'status.json').read_text())
    if not status.get('complete'):
        receipt = execute('full', [runner, 'run', '--output', RUN, '--workers', '64', '--request-timeout', '600'])
        status = json.loads((RUN / 'status.json').read_text())
        receipt['terminal_samples'] = status['completed_samples']
        atomic(RUN / 'collection_result.json', receipt)
    if not status.get('complete') or status['completed_samples'] != 960:
        raise ValueError('Zero-temperature collection is incomplete')
    print(json.dumps({'status': 'complete', 'temperature': 0.0, 'responses': 960}), flush=True)


if __name__ == '__main__':
    main()
