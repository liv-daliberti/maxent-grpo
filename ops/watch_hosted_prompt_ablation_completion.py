#!/usr/bin/env python3
"""Read-only collection monitoring and isolated frozen grading of complete arms."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/modebench_prompt_ablation_20260911'
ANALYZER = BASE / 'analysis_code_editorial_v2/ops/analyze_modebench_prompt_ablation.py'
PINS = {
    BASE / 'manifest.json': '0d0149905d29cf846c97cbfd2f8f23e5117cf837af78a5b2eb1c17f69d951f93',
    BASE / 'hosted_analysis_runs.json': 'dae01a61cee160576a94646b562f47ef58ce088bc6cfef33af79c48dd9f352da',
    ANALYZER: '2be926e99e98446512f4a14077a2e175d4b5bde0fada2933323f1bf7eaebb891',
    ANALYZER.parent / 'audit_hosted_modebench_completion.py': 'b2fdb2262d9829de42f4f161ba95d68c3b6bf7919194798f0a96406d19f9ea60',
}
OUT = BASE / 'hosted_completion_20260911'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_new(p, value):
    with p.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def emit(value):
    print(json.dumps({'at_utc': now(), **value}), flush=True)


def main():
    for path, expected in PINS.items():
        assert sha(path) == expected, str(path)
    entries = read(BASE / 'hosted_analysis_runs.json')['runs']
    OUT.mkdir(exist_ok=True)
    intent = OUT / 'monitor_intent.json'
    if not intent.exists():
        write_new(intent, {'created_at_utc': now(), 'api_calls': 0,
                          'source_sha256': sha(Path(__file__)),
                          'pins': {str(p): d for p, d in PINS.items()},
                          'action': 'Monitor saved status; run frozen isolated grading only after each successful complete collection marker; authenticate final hosted inventory.'})
    graded = set()
    while True:
        progress = []
        for entry in entries:
            run = Path(entry['run_dir'])
            ident = entry['model_id'] + '_' + entry['arm']
            status = read(run / 'status.json') if (run / 'status.json').exists() else {}
            elapsed = status.get('elapsed_seconds_this_session', 0)
            new = status.get('new_samples_this_session', 0)
            count = status.get('completed_samples', 0)
            errors = {}
            if (run / 'errors.jsonl').exists():
                for line in (run / 'errors.jsonl').read_text().splitlines():
                    record = json.loads(line)
                    key = str(record.get('http_status', record.get('error_type')))
                    errors[key] = errors.get(key, 0) + 1
            progress.append({'cohort': ident, 'saved_samples': count,
                             'elapsed_seconds': elapsed,
                             'estimated_remaining_seconds': (1536-count)*elapsed/new if new else None,
                             'transport_attempt_errors': errors,
                             'failed_groups': status.get('failed_groups_this_session', 0)})
            marker_path = run / 'collection_result.json'
            if ident not in graded and marker_path.exists():
                marker = read(marker_path)
                assert marker['exit_code'] == 0, f'Collection did not finish successfully: {ident}'
                assert marker['terminal_samples'] == 1536 and status['complete']
                assert sha(ANALYZER) == PINS[ANALYZER]
                log_path = OUT / (ident + '_grading.log')
                emit({'event': 'grading_started', 'cohort': ident})
                with log_path.open('x') as log:
                    p = subprocess.run([sys.executable, '-I', '-B', str(ANALYZER),
                                        '--base', str(BASE), '--grade-hosted', str(run)],
                                       cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
                assert p.returncode == 0, f'Frozen grading failed: {log_path}'
                audit = read(run / 'prompt_ablation_grading_audit.json')
                assert audit['status'] == 'complete' and audit['records'] == 1536
                graded.add(ident)
                emit({'event': 'grading_complete', 'cohort': ident,
                      'strict_changed_records': audit['strict_changed_records'],
                      'normalization_only_successes': audit['normalized_verified']-audit['strict_verified']})
        emit({'event': 'progress', 'cohorts_graded': len(graded), 'cohorts': progress})
        if len(graded) == 6:
            break
        time.sleep(45)
    inventory_path = OUT / 'hosted_inventory.json'
    with (OUT / 'inventory.log').open('x') as log:
        p = subprocess.run([sys.executable, '-I', '-B', str(ANALYZER),
                            '--base', str(BASE), '--scope', 'hosted', '--inventory',
                            '--output', str(inventory_path)], cwd=ROOT,
                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    assert p.returncode == 0, 'Final frozen inventory failed'
    reports = []
    for entry in entries:
        run = Path(entry['run_dir'])
        audit = read(run / 'prompt_ablation_grading_audit.json')
        reports.append({'model_id': entry['model_id'], 'arm': entry['arm'],
                        'records': audit['records'],
                        'strict_changed_records': audit['strict_changed_records'],
                        'normalization_only_successes': audit['normalized_verified']-audit['strict_verified'],
                        'physical_attempts': audit['physical_attempts'],
                        'attempt_status_counts': audit['attempt_status_counts'],
                        'sha256': {name: sha(run/name) for name in
                            ('manifest.json', 'samples.jsonl', 'collection_result.json',
                             'prompt_ablation_grades.jsonl', 'prompt_ablation_grading_audit.json',
                             'completion_audit.json', 'evidence_file_sha256.json')}})
    assert sum(r['records'] for r in reports) == 9216
    complete = OUT / 'COMPLETE.json'
    write_new(complete, {'status': 'complete', 'created_at_utc': now(), 'cohorts': 6,
                         'records': 9216, 'grading_api_calls': 0,
                         'monitor_source_sha256': sha(Path(__file__)),
                         'hosted_inventory_sha256': sha(inventory_path), 'reports': reports})
    emit({'event': 'complete', 'path': str(complete), 'sha256': sha(complete)})


if __name__ == '__main__':
    main()
