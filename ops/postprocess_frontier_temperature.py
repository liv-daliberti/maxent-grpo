#!/usr/bin/env python3
"""Audit separate matched temperature conditions without making model calls."""
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



def freeze_assets(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    reference = Path(manifest['reference_run'])
    if file_sha(reference / 'manifest.json') != manifest['reference_manifest_sha256']:
        raise ValueError('Original reference manifest changed')
    path = directory / 'postprocessing_source_manifest.json'
    if path.exists():
        for name, value in json.loads(path.read_text())['source_sha256'].items():
            if file_sha(directory / name) != value:
                raise ValueError('Frozen analysis asset changed: ' + name)
        return
    files = {}
    for source in (reference / 'secondary_code').rglob('*.py'):
        files[Path('secondary_code') / source.relative_to(reference / 'secondary_code')] = source
    for name in ('audit_hosted_modebench_completion.py', 'audit_hosted_modebench_python.py',
                 'audit_hosted_provider_outcomes.py', 'postprocess_frontier_temperature.py'):
        files[Path('analysis_code/ops') / name] = ROOT / 'ops' / name
    files[Path('secondary_initial15_audit.json')] = reference / 'secondary_initial15_audit.json'
    for relative, source in files.items():
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and file_sha(destination) != file_sha(source):
            raise ValueError('Existing unfrozen analysis asset differs: ' + str(relative))
        shutil.copyfile(source, destination)
    provenance = {
        'schema': 'frontier-temperature-normalization-provenance-v1',
        'source_original_model_run': str(reference),
        'source_manifest_sha256': file_sha(reference / 'manifest.json'),
        'normalizer_sha256': file_sha(directory / 'secondary_code/ops/frontier_modebench_normalization.py'),
        'interpretation': 'Exact prior GPT-5.6 Sol formatting rules; initial15 audit refers to that earlier derivation, not this prompt condition.',
    }
    atomic(directory / 'secondary_rule_provenance.json', provenance)
    atomic(path, {'frozen_at_utc': now(), 'source_sha256': {
        str(relative): file_sha(directory / relative) for relative in files}, 'api_calls': 0})


def process(directory):
    directory = directory.resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    if (manifest.get('experiment_condition') != 'temperature_ablation_v1' or manifest['request_count'] != 960
            or manifest.get('prompt_count') != 120 or manifest.get('temperature') not in (1.0, 1.5)
            or manifest.get('original_prompt_cohort') is not False):
        raise ValueError('Expected a separately frozen matched temperature condition')
    freeze_assets(directory)
    with (directory / '.postprocess.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            path = directory / 'status.json'
            status = json.loads(path.read_text()) if path.exists() else {}
            if status.get('complete') and status.get('completed_samples') == 960:
                with (directory / '.runner.lock').open('a') as readiness_lock:
                    try:
                        fcntl.flock(readiness_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        time.sleep(2)
                        continue
                break
            print(json.dumps({'status': 'waiting_for_collection', 'completed': status.get('completed_samples', 0), 'at_utc': now()}), flush=True)
            time.sleep(15)
        with (directory / '.runner.lock').open('a') as collector_lock:
            fcntl.flock(collector_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            marker_path = directory / 'analysis_complete.json'
            if marker_path.exists():
                marker = json.loads(marker_path.read_text())
                for name, value in {**marker['evidence_sha256'], **marker['output_sha256']}.items():
                    if file_sha(directory / name) != value:
                        raise ValueError('Completed analysis changed: ' + name)
                return marker
            evidence = {name: file_sha(directory / name) for name in
                        ('manifest.json', 'rows.jsonl', 'requests.jsonl', 'http_requests.jsonl', 'samples.jsonl', 'events.jsonl', 'status.json', 'temperature_condition.json')}
            logs = directory / 'analysis_logs' / now().replace(':', '-')
            logs.mkdir(parents=True)
            audit_code = directory / 'analysis_code/ops'
            call_audit = 'import sys;sys.path.insert(0,sys.argv[1]);from {module} import audit;audit(sys.argv[2],expected_samples=960)'
            steps = [
                ('completion', [sys.executable, '-c', call_audit.format(module='audit_hosted_modebench_completion'), str(audit_code), str(directory)]),
                ('provider_outcomes', [sys.executable, '-c', call_audit.format(module='audit_hosted_provider_outcomes'), str(audit_code), str(directory)]),
                ('python_audit', [sys.executable, str(audit_code / 'audit_hosted_modebench_python.py'), '--output-root', str(directory), '--expected-samples', '960']),
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
            if (summary['status'] != 'complete' or summary['received_responses'] != 960
                    or summary['complete_prompts'] != 120 or summary['missing_responses'] != 0
                    or provider['responses'] != 960 or provider['status'] != 'complete'
                    or primary['status'] != 'complete_for_snapshot' or not primary['full_sampling_complete']):
                raise ValueError('Final analysis did not authenticate the entire condition')
            if (summary['primary_samples_sha256'] != file_sha(directory / 'audited_primary_samples.jsonl')
                    or summary['normalized_secondary']['cache_sha256'] != file_sha(directory / 'normalized_samples.jsonl')):
                raise ValueError('Final summary inputs differ from authenticated sidecars')
            rendered = (directory / 'report.md').read_text()
            (directory / 'report.frozen_renderer.md').write_text(rendered)
            annotation = (
                f"This is a separate small temperature condition: {manifest['model']}, requested T={manifest['temperature']:.1f}, "
                '120 held-out prompts (eight per domain and level) and 960 new stateless calls. '
                'It must not be inserted into the original full model cohort. Selection uses only frozen row identities, never outcomes. '
                'Original prompt bytes, mathematical constraints, frozen verifiers, reasoning settings and other sampling fields are unchanged. '
                'API acceptance does not establish that a provider applies temperature; consult the saved provider-control preflight and documentation. '
                'The formatting rules were derived from the first 15 responses of the earlier GPT-5.6 Sol run and frozen before these conditions. '
                'Refusals, truncations and invalid answers remain recorded. Eight prompts per cell is exploratory; intervals resample whole prompts. '
                'See [provider outcomes](provider_outcomes.md).\n\n'
            )
            (directory / 'report.md').write_text(annotation + rendered)
            completion = json.loads((directory / 'completion_audit.json').read_text())
            if completion.get('status') != 'pass' or completion.get('saved_samples') != 960:
                raise ValueError('Completion audit did not pass for the entire condition')
            if len(summary['cells']) != 15 or any(cell['complete_prompts'] != 8 for cell in summary['cells'].values()):
                raise ValueError('Every domain-level cell must contain exactly eight complete prompts')
            outputs = ['completion_audit.json', 'evidence_file_sha256.json', 'provider_outcomes.json', 'provider_outcome_samples.jsonl',
                       'primary_python_regrade_audit.json', 'audited_primary_samples.jsonl', 'normalized_samples.jsonl', 'summary.json',
                       'report.md', 'report.frozen_renderer.md', 'secondary_rule_provenance.json', 'postprocessing_source_manifest.json']
            marker = {'schema': 'frontier-temperature-analysis-v1', 'status': 'complete', 'completed_at_utc': now(),
                      'model': manifest['model'], 'condition': manifest['experiment_condition'], 'temperature': manifest['temperature'], 'received_responses': 960,
                      'complete_prompts': 120, 'api_calls': 0, 'evidence_sha256': evidence,
                      'output_sha256': {name: file_sha(directory / name) for name in outputs}, 'steps': receipts,
                      'provider_outcome_totals': provider['totals'], 'original_prompt_cohort': False}
            atomic(marker_path, marker)
            atomic(directory / 'analysis_status.json', {'status': 'complete', 'completed_at_utc': now(), 'marker': marker_path.name})
            print(json.dumps({'status': 'complete', 'condition': manifest['experiment_condition'], 'responses': 960, 'provider_outcomes': provider['totals']}), flush=True)
            return marker


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        freeze_assets(args.output.resolve())
    else:
        process(args.output)
