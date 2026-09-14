#!/usr/bin/env python3
"""Authenticate all completed fresh64 local receipts and scheduler inventory."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/modebench_discovery_curves_20260911'
LOCAL = BASE / 'local'
sys.path.insert(0, str(LOCAL / 'code/ops'))
from evaluate_modebench_sampling_budget_local import (SCHEMA, SETTINGS, completion_records, file_sha,
    ordered_items, problem_seed, read_jsonl, sha, validate_batch, validate_plan)
from evaluate_modebench_level3 import atomic_new


def audit():
    plan_path = LOCAL / 'plan.json'
    plan = json.loads(plan_path.read_text())
    rows, pairs = validate_plan(plan)
    plan_sha = file_sha(plan_path)
    checkpoints, total, batches = [], 0, 0
    for index, checkpoint in enumerate(plan['checkpoints']):
        directory = LOCAL / 'results' / checkpoint['label']
        result_path = directory / 'result.json'
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text())
        identity = {'schema': SCHEMA, 'plan_sha256': plan_sha, 'checkpoint': checkpoint, 'settings': SETTINGS}
        assert result['identity'] == identity and result['identity_sha256'] == sha(identity)
        assert result['status'] == 'complete'
        raw_path = directory / 'responses.jsonl'
        assert result['responses_path'] == str(raw_path) and result['responses_sha256'] == file_sha(raw_path)
        selected = sorted(k for k in rows if checkpoint['domain'] is None or k[1] == checkpoint['domain'])
        rebuilt, prompt_results, expected_batch_paths = [], [], []
        for block in range(8):
            ordered = ordered_items(selected, block)
            for start in range(0, len(ordered), 8):
                path = directory / f'batch_b{block:02d}_{start:04d}.json'
                expected_batch_paths.append(path)
                batch = json.loads(path.read_text())
                items = ordered[start:start+8]
                validate_batch(batch, sha(identity), block, start, items, pairs, rows)
                prompt_results.extend(batch['records'])
                for (key, arm), record in zip(items, batch['records']):
                    rebuilt.extend(completion_records(checkpoint, rows[key], pairs[key][arm], record, block))
        assert set(directory.glob('batch_*.json')) == set(expected_batch_paths)
        assert result['prompt_results'] == prompt_results
        raw = read_jsonl(raw_path)
        assert raw == rebuilt and len(raw) == checkpoint['expected_draws'] == result['draws']
        observed = {(r['level'], r['domain'], r['row_index'], r['arm'], r['draw_index']) for r in raw}
        expected = {(*key, arm, draw) for key in selected for arm in ('original', 'neutral') for draw in range(64)}
        assert observed == expected and len(raw) == len(expected)
        for r in raw:
            key = r['level'], r['domain'], r['row_index']
            assert r['draw_block'] == r['draw_index'] // 8 and r['block_draw_index'] == r['draw_index'] % 8
            assert r['sampling_seed'] == problem_seed(key, r['draw_block'])
            assert r['child_sampling_seed'] == r['sampling_seed'] + r['block_draw_index']
            assert r['row_sha256'] == sha(rows[key])
            assert r['messages_sha256'] == pairs[key][r['arm']]['messages_sha256']
            assert r['token_count'] <= 192 and r['verified'] == (r['canonical_key'] is not None)
        runtime = json.loads((directory / 'runtime.json').read_text())
        assert runtime['identity_sha256'] == sha(identity)
        checkpoints.append({'checkpoint_index': index, 'checkpoint': checkpoint['label'], 'draws': len(raw),
            'result_sha256': file_sha(result_path), 'responses_sha256': file_sha(raw_path),
            'generation_batch_count': len(expected_batch_paths), 'runtime_sha256': file_sha(directory / 'runtime.json')})
        total += len(raw)
        batches += len(expected_batch_paths)
    return plan, {'schema': 'modebench-discovery-curves-local-integrity-v1',
        'created_at': datetime.now(timezone.utc).isoformat(), 'plan_sha256': plan_sha,
        'checkpoints': len(checkpoints), 'draws': total, 'generation_batches': batches,
        'checkpoints_audit': checkpoints, 'all_source_input_pins_verified': True,
        'raw_draw_inventory_exact': True, 'all_n64_seed_blocks_verified': True,
        'all_token_counts_within192': True, 'all_final_exports_match_generation_batches': True,
        'original_rows_and_arm_prompt_hashes_verified': True, 'prior_draws_reused': 0,
        'verifier_replay': 'Independent analysis controller authenticates/regrades separately; this audit does not replace it.'}


def scheduler(job_id):
    result = subprocess.run(['sacct', '-j', job_id, '--parsable2', '--noheader',
        '--format=JobID,State,ExitCode,Start,End,ElapsedRaw,AllocTRES'], text=True, capture_output=True, check=True)
    records = []
    for line in result.stdout.splitlines():
        parts = line.split('|')
        if len(parts) < 7 or not parts[0].startswith(job_id + '_') or not parts[0][len(job_id)+1:].isdigit():
            continue
        records.append(dict(zip(('job_id', 'state', 'exit_code', 'start', 'end', 'elapsed_seconds', 'allocated_tres'), parts[:7])))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    previous = None
    while True:
        plan, report = audit()
        job_id = str(json.loads((LOCAL / 'submission_result.json').read_text())['job_id'])
        records = scheduler(job_id)
        failed = [r for r in records if r['state'] in ('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL')]
        if failed:
            atomic_new(LOCAL / ('runtime_failure_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json'), {'failed_jobs': failed})
            raise RuntimeError('Owned evaluation task failed; inspect immutable failure receipt')
        state = (report['checkpoints'], report['draws'], tuple(sorted(Counter(r['state'] for r in records).items())))
        if state != previous:
            print(json.dumps({'event': 'progress', 'created_at': report['created_at'], 'checkpoints': report['checkpoints'],
                              'draws': report['draws'], 'scheduler_states': dict(Counter(r['state'] for r in records))}), flush=True)
            previous = state
        if report['checkpoints'] == 25 and len(records) == 25 and all(r['state'] == 'COMPLETED' and r['exit_code'] == '0:0' for r in records):
            assert report['draws'] == 110592 and report['generation_batches'] == 1728
            assert {int(r['job_id'].split('_')[1]) for r in records} == set(range(25))
            events = [(datetime.fromisoformat(r[t]), delta) for r in records for t, delta in [('start', 1), ('end', -1)]]
            peak = running = 0
            for _, delta in sorted(events):
                running += delta
                peak = max(peak, running)
            assert peak <= 8 and running == 0
            assert all('gres/gpu=1' in r['allocated_tres'] and 'gres/gpu:a5000=1' in r['allocated_tres'] for r in records)
            report.update(status='complete', scheduler_records=records, peak_concurrent_production_gpus=peak,
                          total_production_gpu_seconds=sum(int(r['elapsed_seconds']) for r in records),
                          audit_source_sha256=file_sha(Path(__file__)))
            final = LOCAL / 'completion_integrity_audit.json'
            atomic_new(final, report)
            atomic_new(LOCAL / 'COMPLETE.json', {'status': 'complete', 'created_at': report['created_at'],
                'plan_sha256': report['plan_sha256'], 'completion_audit_sha256': file_sha(final),
                'checkpoints': 25, 'draws': 110592, 'all_jobs_completed_zero_exit': True})
            print(json.dumps({'event': 'complete', 'audit_sha256': file_sha(final), 'draws': 110592,
                              'production_gpu_seconds': report['total_production_gpu_seconds']}), flush=True)
            return
        if not args.watch:
            return
        time.sleep(45)


if __name__ == '__main__':
    main()
