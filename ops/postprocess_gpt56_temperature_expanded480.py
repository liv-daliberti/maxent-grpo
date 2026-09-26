#!/usr/bin/env python3
"""Audit one completed 360-prompt expansion arm without making API calls."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / '.git').exists())
SOURCE_OPS = Path(__file__).resolve().parent
sys.path[:0] = [str(SOURCE_OPS), str(ROOT / 'ops')]
import audit_hosted_modebench_completion as native

EXPECTED = 2880
PROMPTS = 360

def now():
    return datetime.now(timezone.utc).isoformat()

def validate_manifest(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if (manifest.get('expansion_view_schema') != 'gpt56-temperature-expansion-arm-view-v1' or
            manifest.get('schema') != 'frontier-modebench-responses-v1' or
            manifest.get('model') != 'gpt-5.6-sol' or manifest.get('reasoning_effort') != 'none' or
            manifest.get('temperature') not in (0.0, .5, 1., 1.5, 2.) or
            manifest.get('prompt_count') != PROMPTS or manifest.get('request_count') != EXPECTED or
            manifest.get('sample_count') != 8):
        raise ValueError('Require a registered complete additional-360 temperature arm')
    return manifest

def freeze_assets(directory):
    manifest = validate_manifest(directory)
    reference = Path(manifest['reference_run']).resolve()
    if native.file_sha(reference / 'manifest.json') != manifest['reference_manifest_sha256']:
        raise ValueError('Original reference manifest changed')
    marker = directory / 'postprocessing_source_manifest.json'
    if marker.exists():
        for name, digest in json.loads(marker.read_text())['source_sha256'].items():
            if native.file_sha(directory / name) != digest:
                raise ValueError('Frozen expanded postprocessing source changed: ' + name)
        return
    files = {Path('secondary_code') / source.relative_to(reference / 'secondary_code'): source
             for source in (reference / 'secondary_code').rglob('*.py')}
    for name in ('audit_hosted_modebench_completion.py', 'audit_hosted_modebench_python.py',
                 'audit_hosted_provider_outcomes.py', 'postprocess_gpt56_temperature_expanded480.py',
                 'complete_gpt56_prompt_expansion_views.py'):
        files[Path('analysis_code/ops') / name] = SOURCE_OPS / name
    files[Path('secondary_initial15_audit.json')] = reference / 'secondary_initial15_audit.json'
    for relative, source in files.items():
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and native.file_sha(destination) != native.file_sha(source):
            raise ValueError('Existing expanded analysis source differs: ' + str(relative))
        if not destination.exists():
            shutil.copyfile(source, destination)
    native.atomic(directory / 'secondary_rule_provenance.json', {
        'schema': 'frontier-temperature-normalization-provenance-v1',
        'source_original_model_run': str(reference),
        'source_manifest_sha256': native.file_sha(reference / 'manifest.json'),
        'normalizer_sha256': native.file_sha(directory / 'secondary_code/ops/frontier_modebench_normalization.py'),
        'interpretation': 'Exact original GPT-5.6 Sol formatting rules, frozen before the expanded collection.'})
    native.atomic(marker, {'schema': 'gpt56-expanded480-postprocessing-source-v1',
        'frozen_at_utc': now(), 'api_calls': 0,
        'source_sha256': {str(p): native.file_sha(directory / p) for p in files}})

def audit_controls(directory):
    manifest = validate_manifest(directory)
    inventory = native.load_inventory(directory, expected_samples=EXPECTED)
    records = native.read_jsonl(directory / 'samples.jsonl')
    if len(records) != EXPECTED or len({r['sample_id'] for r in records}) != EXPECTED:
        raise ValueError('Expanded arm is incomplete or duplicated')
    bodies = native.validate_native_records(inventory, records)
    counts = {name: Counter() for name in ('temperature', 'reasoning_effort', 'top_p', 'model', 'served_model')}
    for record in records:
        request = inventory['expected'][record['sample_id']]['request']
        raw = bodies[record['raw_receipt']]
        body = raw['response']
        if (request.get('temperature') != manifest['temperature'] or
                request.get('reasoning', {}).get('effort') != 'none' or
                body.get('temperature') != manifest['temperature'] or
                body.get('reasoning', {}).get('effort') != 'none'):
            raise ValueError('Native expanded temperature or reasoning control mismatch')
        for name, value in (('temperature', body.get('temperature')),
                            ('reasoning_effort', body.get('reasoning', {}).get('effort')),
                            ('top_p', body.get('top_p')), ('model', body.get('model')),
                            ('served_model', raw.get('headers', {}).get('x-ms-served-model', 'not_returned'))):
            counts[name][str(value)] += 1
    report = {'schema': 'gpt56-temperature-native-control-audit-expanded480-v1', 'status': 'pass',
        'responses': EXPECTED, 'api_calls': 0, 'audited_at_utc': now(),
        'manifest_sha256': native.file_sha(directory / 'manifest.json'),
        'samples_sha256': native.file_sha(directory / 'samples.jsonl'),
        'auditor_sha256': native.file_sha(__file__),
        'requested_temperature': manifest['temperature'], 'requested_reasoning_effort': 'none',
        **{'returned_' + name + '_counts': dict(count) for name, count in counts.items()}}
    native.atomic(directory / 'native_control_audit.json', report)
    return report

def process(directory):
    directory = Path(directory).resolve()
    manifest = validate_manifest(directory)
    freeze_assets(directory)
    with (directory / '.postprocess.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker_path = directory / 'analysis_complete.json'
        if marker_path.exists():
            marker = json.loads(marker_path.read_text())
            for name, digest in {**marker['evidence_sha256'], **marker['output_sha256']}.items():
                if native.file_sha(directory / name) != digest:
                    raise ValueError('Completed expanded analysis changed: ' + name)
            return marker
        status = json.loads((directory / 'status.json').read_text())
        if not status.get('complete') or status.get('completed_samples') != EXPECTED:
            raise ValueError('Additional-360 arm is not fully collected')
        evidence = {name: native.file_sha(directory / name) for name in
            ('manifest.json', 'rows.jsonl', 'requests.jsonl', 'samples.jsonl', 'events.jsonl',
             'status.json', 'parent_collection_binding.json', 'transport_history_binding.json')}
        logs = directory / 'analysis_logs' / now().replace(':', '-')
        logs.mkdir(parents=True)
        audit_code = directory / 'analysis_code/ops'
        call_audit = 'import sys;sys.path.insert(0,sys.argv[1]);from {module} import audit;audit(sys.argv[2],expected_samples=2880)'
        steps = [
            ('completion', [sys.executable, '-c', call_audit.format(module='audit_hosted_modebench_completion'), str(audit_code), str(directory)]),
            ('native_controls', [sys.executable, str(audit_code / 'postprocess_gpt56_temperature_expanded480.py'), '--output', str(directory), '--audit-controls']),
            ('provider_outcomes', [sys.executable, '-c', call_audit.format(module='audit_hosted_provider_outcomes'), str(audit_code), str(directory)]),
            ('python_audit', [sys.executable, str(audit_code / 'audit_hosted_modebench_python.py'), '--output-root', str(directory), '--expected-samples', str(EXPECTED)]),
            ('summary', [sys.executable, str(directory / 'secondary_code/ops/summarize_frontier_modebench.py'), '--input-dir', str(directory),
                         '--primary-samples', 'audited_primary_samples.jsonl', '--with-normalization', '--bootstrap-replicates', '2000', '--seed', '20260911']),
        ]
        receipts = []
        for name, command in steps:
            receipt = {'step': name, 'command': command, 'started_at_utc': now(), 'api_calls': 0}
            native.atomic(directory / 'analysis_status.json', {'status': 'running', **receipt})
            print(json.dumps(receipt), flush=True)
            log = logs / (name + '.log')
            with log.open('w') as handle:
                result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            receipt.update(exit_code=result.returncode, completed_at_utc=now(), log=str(log.relative_to(directory)), log_sha256=native.file_sha(log))
            receipts.append(receipt)
            native.atomic(logs / 'steps.json', receipts)
            if result.returncode:
                raise RuntimeError('Expanded offline analysis failed: ' + str(log))
        for name, digest in evidence.items():
            if native.file_sha(directory / name) != digest:
                raise ValueError('Primary expanded evidence changed: ' + name)
        summary = json.loads((directory / 'summary.json').read_text())
        if (summary['status'] != 'complete' or summary['received_responses'] != EXPECTED or
                summary['complete_prompts'] != PROMPTS or summary['missing_responses'] != 0 or
                len(summary['cells']) != 15 or any(c['complete_prompts'] != 24 for c in summary['cells'].values())):
            raise ValueError('Final expanded summary is incomplete or has unequal cells')
        outputs = ('native_control_audit.json', 'completion_audit.json', 'evidence_file_sha256.json',
                   'provider_outcomes.json', 'provider_outcome_samples.jsonl', 'primary_python_regrade_audit.json',
                   'audited_primary_samples.jsonl', 'normalized_samples.jsonl', 'summary.json',
                   'secondary_rule_provenance.json', 'postprocessing_source_manifest.json')
        marker = {'schema': 'gpt56-none-temperature-expanded480-arm-analysis-v1', 'status': 'complete',
            'completed_at_utc': now(), 'model': manifest['model'], 'temperature': manifest['temperature'],
            'reasoning_effort': 'none', 'received_responses': EXPECTED, 'complete_prompts': PROMPTS,
            'api_calls': 0, 'evidence_sha256': evidence,
            'output_sha256': {name: native.file_sha(directory / name) for name in outputs}, 'steps': receipts}
        native.atomic(marker_path, marker)
        native.atomic(directory / 'analysis_status.json', {'status': 'complete', 'completed_at_utc': now(), 'marker': marker_path.name})
        return marker

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--audit-controls', action='store_true')
    args = parser.parse_args()
    directory = args.output.resolve()
    if args.prepare_only:
        freeze_assets(directory)
    elif args.audit_controls:
        print(json.dumps(audit_controls(directory)))
    else:
        print(json.dumps(process(directory)))
