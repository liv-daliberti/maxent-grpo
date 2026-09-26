#!/usr/bin/env python3
"""Wait for completed saved native runs, audit, and summarize without API calls."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / 'artifacts/frontier_models_comparison_20260911'
SLUGS = ('kimi_k3', 'gpt54', 'grok43', 'deepseek_v4_pro', 'claude_opus48')
EVIDENCE = ('manifest.json', 'rows.jsonl', 'datasets.json', 'requests.jsonl', 'http_requests.jsonl', 'samples.jsonl')


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())
    temporary.replace(path)


def evidence_hashes(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    return {name: digest(directory / name) for name in EVIDENCE
            if name != 'http_requests.jsonl' or name in manifest['artifact_sha256']}


def verify_assets(directory):
    for name in ('secondary_rule_provenance.json', 'postprocessing_source_manifest.json'):
        record = json.loads((directory / name).read_text())
        hashes = record.get('copied_asset_sha256', record.get('source_sha256', {}))
        for relative, expected in hashes.items():
            if digest(directory / relative) != expected:
                raise ValueError('Frozen analysis asset changed: ' + relative)


def valid_completed_marker(directory):
    path = directory / 'analysis_complete.json'
    if not path.exists():
        return False
    marker = json.loads(path.read_text())
    if marker.get('status') != 'complete' or marker['evidence_sha256'] != evidence_hashes(directory):
        raise ValueError('Existing completed analysis is stale relative to the saved primary evidence')
    for name, expected in marker['output_sha256'].items():
        if digest(directory / name) != expected:
            raise ValueError('Completed analysis output changed: ' + name)
    verify_assets(directory)
    return True


def process(directory):
    directory = Path(directory)
    with (directory / '.postprocess.lock').open('w') as analysis_lock:
        fcntl.flock(analysis_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (directory / '.runner.lock').open('a') as collector_lock:
            fcntl.flock(collector_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            status = json.loads((directory / 'status.json').read_text())
            if not status.get('complete') or status.get('completed_samples') != 15360:
                raise ValueError('Refusing a final analysis of an incomplete run')
            verify_assets(directory)
            if valid_completed_marker(directory):
                return json.loads((directory / 'analysis_complete.json').read_text())
            before = evidence_hashes(directory)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            logs = directory / 'analysis_logs' / stamp
            logs.mkdir(parents=True, exist_ok=False)
            completion_auditor = directory / 'analysis_code/ops/audit_hosted_modebench_completion.py'
            completion_adapter = directory / 'completion_audit_adapter.json'
            if completion_adapter.exists():
                adapter = json.loads(completion_adapter.read_text())
                if adapter.get('schema') != 'hosted-completion-audit-adapter-v1':
                    raise ValueError('Unsupported completion audit adapter')
                for relative, expected_digest in adapter['source_sha256'].items():
                    candidate = (directory / relative).resolve()
                    if not candidate.is_relative_to(directory.resolve()) or digest(candidate) != expected_digest:
                        raise ValueError('Frozen completion adapter source changed: ' + relative)
                if digest(directory / adapter['registration']) != adapter['registration_sha256']:
                    raise ValueError('Frozen interrupted-attempt registration changed')
                if adapter['adapter'] not in adapter['source_sha256']:
                    raise ValueError('Unfrozen completion adapter')
                completion_auditor = directory / adapter['adapter']
            steps = [
                ('completion_audit', [sys.executable, str(completion_auditor), str(directory)]),
                ('provider_outcomes', [sys.executable, str(directory / 'analysis_code/ops/audit_hosted_provider_outcomes.py'), str(directory)]),
                ('primary_python_audit', [sys.executable, str(directory / 'analysis_code/ops/audit_hosted_modebench_python.py'), '--output-root', str(directory)]),
                ('summary_with_normalization', [sys.executable, str(directory / 'secondary_code/ops/summarize_frontier_modebench.py'),
                    '--input-dir', str(directory), '--primary-samples', 'audited_primary_samples.jsonl',
                    '--with-normalization', '--bootstrap-replicates', '2000', '--seed', '20260911']),
            ]
            receipts = []
            for step, command in steps:
                started = time.monotonic()
                receipt = {'step': step, 'command': command, 'started_at_utc': now(), 'api_calls': 0}
                atomic(directory / 'analysis_status.json', {'status': 'running', **receipt})
                print(json.dumps({'directory': str(directory), **receipt}), flush=True)
                log = logs / (step + '.log')
                with log.open('w') as handle:
                    result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
                receipt.update(exit_code=result.returncode, completed_at_utc=now(), elapsed_seconds=time.monotonic() - started,
                               log=str(log.relative_to(directory)), log_sha256=digest(log))
                receipts.append(receipt)
                atomic(logs / 'steps.json', receipts)
                if result.returncode:
                    raise RuntimeError(f'{step} failed with exit code {result.returncode}; inspect {log}')
            if before != evidence_hashes(directory):
                raise ValueError('Primary evidence changed during offline postprocessing')
            completion = json.loads((directory / 'completion_audit.json').read_text())
            primary = json.loads((directory / 'primary_python_regrade_audit.json').read_text())
            provider = json.loads((directory / 'provider_outcomes.json').read_text())
            if provider.get('status') != 'complete' or provider.get('responses') != 15360:
                raise ValueError('Provider outcome audit did not complete')
            summary = json.loads((directory / 'summary.json').read_text())
            if completion['status'] != 'pass' or not primary.get('full_sampling_complete') or primary['status'] != 'complete_for_snapshot':
                raise ValueError('Completion or serial Python audit did not pass')
            if summary['status'] != 'complete' or summary['received_responses'] != 15360 or summary['missing_responses'] != 0 or summary['complete_prompts'] != 1920:
                raise ValueError('Summary is incomplete or stale')
            if summary['primary_samples_sha256'] != digest(directory / 'audited_primary_samples.jsonl'):
                raise ValueError('Summary references an outdated primary sidecar')
            if summary['normalized_secondary']['cache_sha256'] != digest(directory / 'normalized_samples.jsonl'):
                raise ValueError('Summary references an outdated secondary cache')
            rendered_path = directory / 'report.md'
            original_render = rendered_path.read_bytes()
            preserved = directory / 'report.frozen_renderer.md'
            if preserved.exists() and preserved.read_bytes() != original_render:
                raise ValueError('Prior unannotated renderer report differs; preserve/review the prior analysis before replacing it')
            preserved.write_bytes(original_render)
            annotation = (
                'The formatting rules were derived from the first 15 responses of the earlier GPT-5.6 Sol run and were fixed before collecting this model’s test responses. References below to the first fifteen responses describe that earlier rule derivation. Completed HTTP responses can include provider-declared refusals; see [native provider outcomes](provider_outcomes.md) and [normalization rule provenance](SECONDARY_RULE_PROVENANCE.md). This provenance annotation changes no statistics.\n\n'
            )
            rendered_path.write_text(annotation + original_render.decode())
            receipts.append({'step': 'report_provenance_annotation', 'completed_at_utc': now(), 'api_calls': 0,
                             'original_report': preserved.name, 'original_report_sha256': digest(preserved),
                             'annotated_report_sha256': digest(rendered_path), 'metric_changes': False})
            atomic(logs / 'steps.json', receipts)
            outputs = ['completion_audit.json', 'provider_outcomes.json', 'provider_outcomes.md', 'provider_outcome_samples.jsonl',
                       'primary_python_regrade_audit.json', 'audited_primary_samples.jsonl',
                       'normalized_samples.jsonl', 'summary.json', 'report.md', 'report.frozen_renderer.md', 'modebench_frontier.png',
                       'modebench_frontier.pdf', 'modebench_frontier_normalized.png', 'modebench_frontier_normalized.pdf',
                       'secondary_rule_provenance.json', 'SECONDARY_RULE_PROVENANCE.md']
            if completion_adapter.exists():
                outputs.extend(['completion_audit_adapter.json', adapter['registration']])
                outputs.extend(adapter['source_sha256'])
            marker = {'schema': 'hosted-native-modebench-postprocessing-v1', 'status': 'complete', 'api_calls': 0,
                'completed_at_utc': now(), 'directory': str(directory), 'model': json.loads((directory / 'manifest.json').read_text())['model'],
                'received_responses': 15360, 'complete_prompts': 1920, 'evidence_sha256': before,
                'output_sha256': {name: digest(directory / name) for name in outputs}, 'steps': receipts,
                'normalization_provenance': 'Exact prior-GPT-5.6 Sol rules; copied initial15 audit describes that source experiment only.',
                'primary_python_corrected_records': primary.get('corrected_samples'),
                'normalized_additional_verified': summary['normalized_secondary']['additional_verified_responses'],
                'provider_outcome_totals': provider['totals'],
                'runner_pid': os.getpid()}
            atomic(directory / 'analysis_complete.json', marker)
            atomic(directory / 'analysis_status.json', {'status': 'complete', 'completed_at_utc': now(),
                                                       'marker': 'analysis_complete.json', 'marker_sha256': digest(directory / 'analysis_complete.json')})
            return marker


def collection_complete_and_idle(directory):
    status_path = directory / 'status.json'
    if not status_path.exists():
        return False
    status = json.loads(status_path.read_text())
    if not status.get('complete') or status.get('completed_samples') != 15360:
        return False
    with (directory / '.runner.lock').open('a') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true', help='Wait until every selected model is complete, then analyze it.')
    parser.add_argument('--slug', choices=SLUGS, action='append')
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.workers not in (1, 2, 3, 4):
        parser.error('Invalid poll interval or postprocessing concurrency')
    slugs = args.slug or list(SLUGS)
    state = {slug: {'status': 'waiting'} for slug in slugs}
    submitted = {}
    with (COMPARISON / '.native_postprocessor.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            while True:
                for slug in slugs:
                    directory = ROOT / 'artifacts' / f'frontier_modebench_{slug}_20260911'
                    if slug in submitted:
                        future = submitted[slug]
                        if future.done() and state[slug]['status'] == 'running':
                            try:
                                marker = future.result()
                                state[slug] = {'status': 'complete', 'completed_at_utc': now(), 'model': marker['model'],
                                               'analysis_complete': str(directory / 'analysis_complete.json')}
                            except Exception as error:
                                state[slug] = {'status': 'failed', 'at_utc': now(), 'error_type': type(error).__name__, 'error': str(error)}
                                atomic(directory / 'analysis_status.json', state[slug])
                            print(json.dumps({'slug': slug, **state[slug]}), flush=True)
                    elif collection_complete_and_idle(directory):
                        submitted[slug] = executor.submit(process, directory)
                        state[slug] = {'status': 'running', 'submitted_at_utc': now()}
                    else:
                        status_path = directory / 'status.json'
                        if status_path.exists():
                            collection = json.loads(status_path.read_text())
                            state[slug] = {'status': 'waiting', 'completed_samples': collection['completed_samples'],
                                           'collection_updated_at_utc': collection['updated_at_utc']}
                atomic(COMPARISON / 'native_postprocessing_status.json', {'updated_at_utc': now(), 'pid': os.getpid(), 'api_calls': 0, 'runs': state})
                if all(s['status'] in ('complete', 'failed') for s in state.values()):
                    return 1 if any(s['status'] == 'failed' for s in state.values()) else 0
                if not args.watch and all(f.done() for f in submitted.values()):
                    return 1 if any(s['status'] == 'failed' for s in state.values()) else 0
                time.sleep(args.poll_seconds)


if __name__ == '__main__':
    raise SystemExit(main())
