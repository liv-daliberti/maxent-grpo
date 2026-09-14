#!/usr/bin/env python3
"""Bind each expanded temperature view to the parent's full transport history.

Additive completion for the sealed collector's views; no native receipt or
sealed collection manifest is modified. Run after collection, before analysis.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
sys.path.insert(0, str(ROOT / 'ops'))
import run_gpt56_prompt_expansion as collection_runner
import prepare_gpt56_prompt_expansion as collection_preparer
from run_gpt56_prompt_expansion import (DEFAULT_OUTPUT, TEMPERATURES, atomic, audit_attempts,
    file_sha, load_completed, read_lines, require, sha, slug, validate_inventory)


def immutable_jsonl(path, records):
    """Only create a view log, or authenticate a byte-identical previous export."""
    path = Path(path)
    payload = ''.join(json.dumps(record, sort_keys=True, allow_nan=False) + '\n' for record in records)
    if path.exists():
        require(path.read_text() == payload, 'Existing transport view log differs: ' + str(path))
    else:
        with path.open('x') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())


def verify_collector_bindings(output):
    """Refuse interpretation through any dependency differing from the seal."""
    output = Path(output).resolve()
    manifest = json.loads((output / 'manifest.json').read_text())
    bindings = {}
    modules = [('ops/run_gpt56_prompt_expansion.py', collection_runner),
               ('ops/prepare_gpt56_prompt_expansion.py', collection_preparer),
               ('ops/evaluate_frontier_modebench.py', collection_runner.native)]
    for relative, module in modules:
        expected = manifest['code_sha256'][relative]
        require(file_sha(module.__file__) == expected, 'Loaded collection dependency differs from sealed bytes: ' + relative)
        require(file_sha(output / 'code' / relative) == expected, 'Sealed collection dependency changed: ' + relative)
        bindings[relative] = expected
    return bindings


def complete_views(output=DEFAULT_OUTPUT):
    output = Path(output).resolve()
    collector_bindings = verify_collector_bindings(output)
    manifest, rows, requests = validate_inventory(output)
    completed = load_completed(output, requests)
    require(len(completed) == 14400, 'Transport views require a complete parent collection')
    history = audit_attempts(output, requests)
    parent_logs = {name: read_lines(output / name) for name in ('events.jsonl', 'errors.jsonl', 'runner_errors.jsonl') if (output / name).exists()}
    results = []
    for temperature in TEMPERATURES:
        view = output / 'arms' / slug(temperature)
        view_manifest = json.loads((view / 'manifest.json').read_text())
        require(view_manifest.get('expansion_view_schema') == 'gpt56-temperature-expansion-arm-view-v1', 'Unexpected arm view schema')
        require(view_manifest['temperature'] == temperature and view_manifest['request_count'] == 2880, 'Arm scope differs')
        arm = [item for item in requests if item['temperature_condition'] == temperature]
        require(read_lines(view / 'requests.jsonl') == arm, 'Arm requests differ from exact parent subsequence')
        for name, digest in view_manifest['artifact_sha256'].items():
            require(file_sha(view / name) == digest, 'Sealed arm view artifact differs')
        ids = {item['sample_id'] for item in arm}
        logs = {}
        for name, records in parent_logs.items():
            selected = [record for record in records if record.get('sample_id') in ids]
            immutable_jsonl(view / name, selected)
            logs[name] = {'parent_sha256': file_sha(output / name), 'view_sha256': file_sha(view / name), 'records': len(selected)}
        attempts = []
        for item in arm:
            for raw in sorted(history[item['sample_id']], key=lambda record: record['attempt']):
                relative = Path(raw['relative_path'])
                source, destination = output / relative, view / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    require(file_sha(destination) == file_sha(source), 'Existing raw attempt view differs')
                else:
                    os.link(source, destination)
                attempts.append({'sample_id': item['sample_id'], 'attempt': raw['attempt'],
                                 'path': str(relative), 'sha256': file_sha(source)})
        require({str(path.relative_to(view)) for path in (view / 'raw_responses').glob('*.json')} == {record['path'] for record in attempts}, 'Unexpected raw attempt in arm view')
        binding = {'schema': 'gpt56-temperature-expansion-transport-view-v1',
                   'parent_collection': str(output), 'parent_manifest_sha256': file_sha(output / 'manifest.json'),
                   'view_manifest_sha256': file_sha(view / 'manifest.json'), 'temperature': temperature,
                   'logical_samples': len(arm), 'raw_attempts': len(attempts), 'logs': logs,
                   'attempts': attempts, 'attempts_sha256': sha(attempts), 'exporter_sha256': file_sha(__file__),
                   'collector_dependency_sha256': collector_bindings}
        path = view / 'transport_history_binding.json'
        if path.exists():
            require(json.loads(path.read_text()) == binding, 'Transport view binding differs')
        else:
            atomic(path, binding)
        audit_attempts(view, arm)
        results.append({'temperature': temperature, 'logical_samples': len(arm), 'raw_attempts': len(attempts),
                        'transport_history_binding_sha256': file_sha(path)})
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps({'status': 'complete', 'views': complete_views(args.output)}, indent=2))
