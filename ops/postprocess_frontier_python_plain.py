#!/usr/bin/env python3
"""Audit the full separate Python prompt condition without making model calls."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
from evaluate_claude_modebench import atomic, file_sha, now

DEFAULT = ROOT / 'artifacts/frontier_modebench_claude_opus5_python_plain_20260911'
REFERENCE = ROOT / 'artifacts/frontier_modebench_claude_opus5_20260911'


def freeze_assets(directory):
    path = directory / 'postprocessing_source_manifest.json'
    if path.exists():
        for name, value in json.loads(path.read_text())['source_sha256'].items():
            if file_sha(directory / name) != value:
                raise ValueError('Frozen analysis asset changed: ' + name)
        return
    files = {}
    for source in (REFERENCE / 'secondary_code').rglob('*.py'):
        files[Path('secondary_code') / source.relative_to(REFERENCE / 'secondary_code')] = source
    for name in ('audit_hosted_modebench_completion.py', 'audit_hosted_modebench_python.py',
                 'audit_hosted_provider_outcomes.py', 'postprocess_frontier_python_plain.py'):
        files[Path('analysis_code/ops') / name] = ROOT / 'ops' / name
    files[Path('secondary_initial15_audit.json')] = REFERENCE / 'secondary_initial15_audit.json'
    for relative, source in files.items():
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and file_sha(destination) != file_sha(source):
            raise ValueError('Existing unfrozen analysis asset differs: ' + str(relative))
        shutil.copyfile(source, destination)
    provenance = {
        'schema': 'frontier-python-plain-normalization-provenance-v1',
        'source_original_opus5_run': str(REFERENCE),
        'source_manifest_sha256': file_sha(REFERENCE / 'manifest.json'),
        'normalizer_sha256': file_sha(directory / 'secondary_code/ops/frontier_modebench_normalization.py'),
        'interpretation': 'Exact prior GPT-5.6 Sol formatting rules; initial15 audit refers to that earlier derivation, not this prompt condition.',
    }
    atomic(directory / 'secondary_rule_provenance.json', provenance)
    atomic(path, {'frozen_at_utc': now(), 'source_sha256': {
        str(relative): file_sha(directory / relative) for relative in files}, 'api_calls': 0})


def process(directory):
    directory = directory.resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('experiment_condition') != 'python_plain_user_no_system_v1' or manifest['request_count'] != 3072:
        raise ValueError('Expected the separately frozen full Python condition')
    freeze_assets(directory)
    with (directory / '.postprocess.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            path = directory / 'status.json'
            status = json.loads(path.read_text()) if path.exists() else {}
            if status.get('complete') and status.get('completed_samples') == 3072:
                break
            print(json.dumps({'status': 'waiting_for_collection', 'completed': status.get('completed_samples', 0), 'at_utc': now()}), flush=True)
            time.sleep(15)
        with (directory / '.runner.lock').open('a') as collector_lock:
            fcntl.flock(collector_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            marker_path = directory / 'analysis_complete.json'
            if marker_path.exists():
                marker = json.loads(marker_path.read_text())
                for name, value in marker['output_sha256'].items():
                    if file_sha(directory / name) != value:
                        raise ValueError('Completed analysis changed: ' + name)
                return marker
            evidence = {name: file_sha(directory / name) for name in
                        ('manifest.json', 'rows.jsonl', 'requests.jsonl', 'samples.jsonl', 'events.jsonl', 'status.json')}
            logs = directory / 'analysis_logs' / now().replace(':', '-')
            logs.mkdir(parents=True)
            audit_code = directory / 'analysis_code/ops'
            call_audit = 'import sys;sys.path.insert(0,sys.argv[1]);from {module} import audit;audit(sys.argv[2],expected_samples=3072)'
            steps = [
                ('completion', [sys.executable, '-c', call_audit.format(module='audit_hosted_modebench_completion'), str(audit_code), str(directory)]),
                ('provider_outcomes', [sys.executable, '-c', call_audit.format(module='audit_hosted_provider_outcomes'), str(audit_code), str(directory)]),
                ('python_audit', [sys.executable, str(audit_code / 'audit_hosted_modebench_python.py'), '--output-root', str(directory), '--expected-samples', '3072']),
                ('summary', [sys.executable, str(directory / 'secondary_code/ops/summarize_frontier_modebench.py'), '--input-dir', str(directory),
                             '--primary-samples', 'audited_primary_samples.jsonl', '--with-normalization', '--bootstrap-replicates', '2000', '--seed', '20260911']),
            ]
            receipts = []
            for name, command in steps:
                receipt = {'step': name, 'command': command, 'started_at_utc': now(), 'api_calls': 0}
                atomic(directory / 'analysis_status.json', {'status': 'running', **receipt})
                print(json.dumps(receipt), flush=True)
                log = logs / (name + '.log')
                with log.open('w') as handle:
                    result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                receipt.update(exit_code=result.returncode, completed_at_utc=now(), log=str(log.relative_to(directory)), log_sha256=file_sha(log))
                receipts.append(receipt)
                atomic(logs / 'steps.json', receipts)
                if result.returncode:
                    raise RuntimeError('Offline analysis failed: ' + str(log))
            for name, digest in evidence.items():
                if file_sha(directory / name) != digest:
                    raise ValueError('Primary evidence changed during analysis: ' + name)
            summary = json.loads((directory / 'summary.json').read_text())
            provider = json.loads((directory / 'provider_outcomes.json').read_text())
            primary = json.loads((directory / 'primary_python_regrade_audit.json').read_text())
            if (summary['status'] != 'complete' or summary['received_responses'] != 3072
                    or summary['complete_prompts'] != 384 or summary['missing_responses'] != 0
                    or provider['responses'] != 3072 or provider['status'] != 'complete'
                    or primary['status'] != 'complete_for_snapshot' or not primary['full_sampling_complete']):
                raise ValueError('Final analysis did not authenticate the entire condition')
            if (summary['primary_samples_sha256'] != file_sha(directory / 'audited_primary_samples.jsonl')
                    or summary['normalized_secondary']['cache_sha256'] != file_sha(directory / 'normalized_samples.jsonl')):
                raise ValueError('Final summary inputs differ from authenticated sidecars')
            rendered = (directory / 'report.md').read_text()
            (directory / 'report.frozen_renderer.md').write_text(rendered)
            annotation = (
                'This is a separate changed-prompt condition for Python only: 384 original held-out prompts and 3,072 new stateless calls. '
                'It must not be inserted into the original five-domain model cohort. Any macro below covers Python alone. '
                'The original system message was omitted and the user prompt was rewritten; mathematical constraints, boxed-lambda output and frozen verifier are unchanged. '
                'The formatting rules were derived from the first 15 responses of the earlier GPT-5.6 Sol run and frozen before this condition. '
                'All original and new refusals remain recorded; see [provider outcomes](provider_outcomes.md).\n\n'
            )
            (directory / 'report.md').write_text(annotation + rendered)
            outputs = ['completion_audit.json', 'evidence_file_sha256.json', 'provider_outcomes.json', 'provider_outcome_samples.jsonl',
                       'primary_python_regrade_audit.json', 'audited_primary_samples.jsonl', 'normalized_samples.jsonl', 'summary.json',
                       'report.md', 'report.frozen_renderer.md', 'secondary_rule_provenance.json', 'postprocessing_source_manifest.json']
            marker = {'schema': 'frontier-python-plain-analysis-v1', 'status': 'complete', 'completed_at_utc': now(),
                      'model': manifest['model'], 'condition': manifest['experiment_condition'], 'received_responses': 3072,
                      'complete_prompts': 384, 'api_calls': 0, 'evidence_sha256': evidence,
                      'output_sha256': {name: file_sha(directory / name) for name in outputs}, 'steps': receipts,
                      'provider_outcome_totals': provider['totals'], 'original_prompt_cohort': False}
            atomic(marker_path, marker)
            atomic(directory / 'analysis_status.json', {'status': 'complete', 'completed_at_utc': now(), 'marker': marker_path.name})
            print(json.dumps({'status': 'complete', 'condition': manifest['experiment_condition'], 'responses': 3072, 'provider_outcomes': provider['totals']}), flush=True)
            return marker


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    process(parser.parse_args().output)
