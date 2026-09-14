#!/usr/bin/env python3
"""Freeze and run the authorized Sol five-domain discovery follow-up.

The existing 96-prompt, 64-draw original-wording cohort stays immutable.
Collect 64 fresh draws for Graph/Countdown and a separate fixed 64-draw
extension for the same 32 Pantry prompts, yielding 128 Pantry draws.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/modebench_discovery_five_domains_sol_20260912'
SOURCE = ROOT / 'artifacts/frontier_modebench_gpt56sol_20260911'
PRIOR = ROOT / 'artifacts/modebench_discovery_curves_20260911/hosted/gpt56sol/original'
RUNS = {'graph_countdown': 'hosted/gpt56sol/original',
        'pantry_extension': 'hosted/gpt56sol/pantry_extension'}
CONDITION = 'sol_five_domain_discovery_followup_20260912_v1'
MODEL = 'gpt-5.6-sol'


def now():
    return datetime.now(timezone.utc).isoformat()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def binding(path):
    return {'path': str(Path(path).resolve()), 'sha256': digest(path)}


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, value):
    path = Path(path)
    require(not path.exists(), 'Refusing to replace existing artifact: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def write_rows(path, values):
    path = Path(path)
    require(not path.exists(), 'Refusing to replace existing artifact: ' + str(path))
    path.write_text(''.join(json.dumps(v, sort_keys=True, allow_nan=False) + '\n' for v in values))


def identity(row):
    return row['level'], row['domain'], row['row_index']


def selection_hash(row):
    return hashlib.sha256(f"20260911\0{row['level']}\0{row['domain']}\0{row['row_index']}".encode()).hexdigest()


def authenticate(path, manifest):
    for name, sha in manifest['artifact_sha256'].items():
        require(digest(path / name) == sha, 'Changed frozen artifact: ' + str(path / name))
    for name, sha in manifest.get('code_sha256', {}).items():
        require(digest(path / 'code' / name) == sha, 'Changed frozen code: ' + name)


def prepare(base=BASE):
    base = Path(base).resolve()
    if (base / 'manifest.json').exists():
        authenticate_base(base)
        return json.loads((base / 'manifest.json').read_text())
    require(not base.exists() or not any(base.iterdir()), 'Use a fresh output directory')
    source_manifest = json.loads((SOURCE / 'manifest.json').read_text())
    prior_manifest = json.loads((PRIOR / 'manifest.json').read_text())
    authenticate(SOURCE, source_manifest)
    authenticate(PRIOR, prior_manifest)
    source_rows = rows(SOURCE / 'rows.jsonl')
    source_requests = {identity(r): r for r in rows(SOURCE / 'requests.jsonl') if r['sample_index'] == 0}
    prior_rows = rows(PRIOR / 'rows.jsonl')
    prior_requests = {identity(r): r for r in rows(PRIOR / 'requests.jsonl') if r['sample_index'] == 0}
    selected, ledger = [], []
    for level in (2, 3):
        for domain in ('graph_coloring', 'countdown'):
            candidates = [r for r in source_rows if (r['level'], r['domain']) == (level, domain)]
            require(len(candidates) == 128, 'Expected 128 source prompts in every new cell')
            for rank, row in enumerate(sorted(candidates, key=lambda r: (selection_hash(r), identity(r))), 1):
                ledger.append({'level': level, 'domain': domain, 'row_index': row['row_index'],
                               'selection_sha256': selection_hash(row), 'rank_in_cell': rank,
                               'selected': rank <= 16, 'row_sha256': object_digest(row)})
                if rank <= 16:
                    selected.append(row)
    selected.sort(key=identity)
    pantry = sorted([r for r in prior_rows if r['domain'] == 'pantry_plan'], key=identity)
    require(len(selected) == 64 and len(pantry) == 32, 'Incorrect fixed cohort sizes')
    profile = json.loads((PRIOR / 'model_profile.json').read_text())
    expected_parameters = {'max_output_tokens': 8192, 'reasoning': {'effort': 'medium'}, 'store': False}
    require(profile['model'] == MODEL and profile['protocol'] == 'responses'
            and profile['request_parameters'] == expected_parameters, 'Unexpected Sol request controls')
    base.mkdir(parents=True, exist_ok=True)
    write(base / 'selection.json', {'schema': 'sol-five-domain-selection-v1',
          'source_rows': binding(SOURCE / 'rows.jsonl'), 'seed': 20260911,
          'rule': 'First 16 per L2/L3 domain by SHA256(seed NUL level NUL domain NUL row_index); ties by identity.',
          'selection_uses_outcomes': False, 'candidates': ledger,
          'prior_96_prompt_rows': binding(PRIOR / 'rows.jsonl'),
          'pantry_extension_rule': 'All existing 32 Pantry prompts, unchanged.'})
    write(base / 'protocol.json', {'schema': CONDITION,
          'authorization': 'User explicitly requested 64 draws on all five domains and more draws where needed.',
          'domains': ['graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan'],
          'levels': [2, 3], 'prompts_per_cell': 16, 'unique_prompts': 160,
          'wording': 'original', 'neutral_graph_countdown_control': False,
          'new_graph_countdown_draws_per_prompt': 64, 'new_graph_countdown_requests': 4096,
          'pantry_total_draws_per_prompt': 128, 'new_pantry_draw_indices': [64, 127], 'new_pantry_requests': 2048,
          'reuse': 'Prior 96 complete n64 original-wording pools for Python, MathIR, Pantry; no prior n8 responses.',
          'timing': 'New domain follow-up collected after the original three-domain cohort; no claim of simultaneous collection.',
          'pantry_extension_basis': 'Prior Pantry late gains remain positive; exploratory fixed extension to 128, frozen before new responses.',
          'stopping': 'Complete every frozen request independent of correctness or mode discovery; no replacement or outcome stopping.',
          'curves': 'Exact without-replacement rarefaction at powers of two up to each observed domain budget; retain ordered-prefix sensitivity.',
          'grading': 'Frozen strict verifier is primary; identical frozen normalization is separately labeled sensitivity.',
          'uncertainty': '20000 whole-problem bootstrap replicates, stratified by level within each domain; seed 20260911.',
          'interpretation': 'Report late gains and certified available-mode references; no extrapolation or claim that unseen modes are absent.'})
    collection_runs = {}
    for name, chosen, templates, start in [('graph_countdown', selected, source_requests, 0),
                                          ('pantry_extension', pantry, prior_requests, 64)]:
        out = base / RUNS[name]
        out.mkdir(parents=True)
        shutil.copytree(PRIOR / 'code', out / 'code')
        shutil.copyfile(PRIOR / 'datasets.json', out / 'datasets.json')
        shutil.copyfile(PRIOR / 'model_profile.json', out / 'model_profile.json')
        requests = []
        for row in chosen:
            reference = templates[identity(row)]
            request = reference['request']
            require(reference['row_sha256'] == object_digest(row), 'Source prompt changed')
            require(reference['request_sha256'] == object_digest(request), 'Source request changed')
            require({k: v for k, v in request.items() if k not in ('model', 'input')} == expected_parameters,
                    'Source Graph/Countdown and prior discovery controls differ')
            require(request['model'] == MODEL and request['input'][1]['content'] == row['problem'], 'Source wording changed')
            for draw in range(start, start + 64):
                sid = f"L{row['level']}_{row['domain']}_{row['row_index']:03d}_{draw}"
                requests.append({'level': row['level'], 'domain': row['domain'], 'row_index': row['row_index'],
                                 'sample_index': draw, 'sample_id': sid, 'group_id': sid, 'choice_index': 0,
                                 'row_sha256': reference['row_sha256'], 'request': request,
                                 'request_sha256': reference['request_sha256'], 'reference_sample_id': reference['sample_id'],
                                 'fresh_response_cohort': True, 'experiment_condition': CONDITION, 'prompt_arm': 'original'})
        requests.sort(key=lambda r: hashlib.sha256(('sol-five-domains-20260912:' + r['sample_id']).encode()).hexdigest())
        require(len(requests) == len(chosen) * 64 and len({r['sample_id'] for r in requests}) == len(requests), 'Duplicate requests')
        groups = [{'group_id': r['group_id'], 'request': r['request'], 'request_sha256': r['request_sha256'],
                   'sample_ids': [r['sample_id']], 'sample_count': 1, 'protocol': 'responses'} for r in requests]
        write_rows(out / 'rows.jsonl', chosen)
        write_rows(out / 'requests.jsonl', requests)
        write_rows(out / 'http_requests.jsonl', groups)
        manifest = {k: v for k, v in prior_manifest.items()
                    if k not in ('artifact_sha256', 'ablation_manifest_sha256', 'notes', 'parent_prompt_ablation_run',
                                 'parent_prompt_ablation_manifest_sha256', 'preparer_sha256')}
        manifest.update(prepared_at_utc=now(), experiment_condition=CONDITION, cohort=name,
                        sample_count=64, sample_index_start=start, combined_draw_budget=start + 64,
                        prompt_count=len(chosen), request_count=len(requests), http_request_count=len(requests),
                        reference_run=str(SOURCE if name == 'graph_countdown' else PRIOR),
                        reference_manifest_sha256=digest((SOURCE if name == 'graph_countdown' else PRIOR) / 'manifest.json'),
                        prior_three_domain_run=binding(PRIOR / 'manifest.json'),
                        source_selection=binding(base / 'selection.json'), protocol_binding=binding(base / 'protocol.json'),
                        notes=['All 64 newly collected draws retained regardless of answer outcome.',
                               'Original native request bytes, model profile, frozen collector and verifier preserved.',
                               'Pantry draw indices 64-127 join the existing complete 0-63 pool only during analysis.'])
        manifest['artifact_sha256'] = {n: digest(out / n) for n in ('datasets.json', 'model_profile.json', 'rows.jsonl', 'requests.jsonl', 'http_requests.jsonl')}
        write(out / 'manifest.json', manifest)
        authenticate(out, manifest)
        collection_runs[name] = {'run_dir': str(out), 'manifest': binding(out / 'manifest.json'),
                                 'new_samples': len(requests), 'sample_index_start': start}
    shutil.copyfile(__file__, base / 'prepare_gpt56_five_domain_discovery.py')
    manifest = {'schema': CONDITION, 'frozen_at_utc': now(), 'collection_runs': collection_runs,
                'prior_three_domain_run': binding(PRIOR / 'manifest.json'), 'source_five_domain_run': binding(SOURCE / 'manifest.json'),
                'artifact_sha256': {n: digest(base / n) for n in ('selection.json', 'protocol.json', 'prepare_gpt56_five_domain_discovery.py')},
                'total_new_requests': 6144, 'credentials_recorded': False}
    write(base / 'manifest.json', manifest)
    authenticate_base(base)
    return manifest


def authenticate_base(base):
    manifest = json.loads((base / 'manifest.json').read_text())
    authenticate(base, manifest)
    for entry in manifest['collection_runs'].values():
        out = Path(entry['run_dir'])
        require(digest(out / 'manifest.json') == entry['manifest']['sha256'], 'Changed cohort manifest')
        authenticate(out, json.loads((out / 'manifest.json').read_text()))
    return manifest


def load_adapter(out):
    path = out / 'code/ops/evaluate_native_prompt_ablation.py'
    spec = importlib.util.spec_from_file_location('_five_domain_frozen_native', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authenticated_records(out):
    adapter = load_adapter(out)
    requests = {r['sample_id']: r for r in rows(out / 'requests.jsonl')}
    groups = {r['group_id']: r for r in rows(out / 'http_requests.jsonl')}
    records, raw_cache, seen = [], {}, set()
    for path in sorted((out / 'sample_receipts').glob('*.json')):
        record = json.loads(path.read_text())
        sid = record['sample_id']
        require(sid in requests and path.name == sid + '.json', 'Unexpected saved sample')
        item = requests[sid]
        adapter.validate_completed(out, item, groups[item['group_id']], record, raw_cache)
        provider = tuple(record['provider_sample_identity'])
        require(provider not in seen, 'Duplicate provider sample')
        seen.add(provider)
        records.append(record)
    return records


def expected_returned_controls():
    record = json.loads(next((PRIOR / 'sample_receipts').glob('*.json')).read_text())
    body = json.loads((PRIOR / record['raw_receipt']).read_text())['response']
    return {k: body.get(k) for k in ('model', 'reasoning', 'temperature', 'top_p', 'max_output_tokens', 'store')}


def validate_returned_controls(out, records):
    expected = expected_returned_controls()
    for record in records:
        body = json.loads((out / record['raw_receipt']).read_text())['response']
        actual = {k: body.get(k) for k in expected}
        require(actual == expected, 'Served model controls differ from prior cohort; inspect retained receipt before more calls')
    return expected


def credential(path):
    if path is None:
        value = os.environ.get('AZURE_OPENAI_API_KEY', '')
    else:
        info = path.stat()
        require(info.st_uid == os.getuid() and not stat.S_IMODE(info.st_mode) & 0o077, 'Credential file must be private and user-owned')
        value = path.read_text().strip()
    require(bool(value) and '\n' not in value and '\r' not in value, 'Supply intended Azure environment or explicit private credential file')
    return value


def collect(base, cohort, stage, credential_file=None, workers=16):
    manifest = authenticate_base(base)
    out = Path(manifest['collection_runs'][cohort]['run_dir'])
    expected_count = manifest['collection_runs'][cohort]['new_samples']
    existing = authenticated_records(out)
    if len(existing) == expected_count:
        validate_returned_controls(out, existing)
        print(json.dumps({'cohort': cohort, 'complete': True, 'new_calls': 0}))
        return
    marker = out / 'preflight.json'
    if stage == 'full':
        require(marker.exists() and existing, 'Run and inspect authenticated preflight before full collection')
        preflight = json.loads(marker.read_text())
        require(preflight['manifest_sha256'] == digest(out / 'manifest.json'), 'Preflight manifest changed')
        validate_returned_controls(out, existing)
    key = credential(credential_file)
    env = os.environ.copy()
    env['AZURE_OPENAI_API_KEY'] = key
    command = [sys.executable, str(out / 'code/ops/evaluate_native_prompt_ablation.py'), 'run',
               '--model', MODEL, '--output', str(out), '--workers', str(workers),
               '--request-timeout', '300', '--max-attempts', '8']
    if stage == 'preflight':
        command += ['--max-new', '1']
    with (out / ('collection_' + stage + '.log')).open('a') as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    key = None
    env.pop('AZURE_OPENAI_API_KEY', None)
    require(result.returncode == 0, 'Native collector failed; inspect retained collection log')
    records = authenticated_records(out)
    controls = validate_returned_controls(out, records)
    if stage == 'preflight' and not marker.exists():
        require(len(records) > 0, 'No authenticated preflight sample')
        write(marker, {'schema': 'sol-five-domain-preflight-v1', 'checked_at_utc': now(),
              'manifest_sha256': digest(out / 'manifest.json'), 'retained_in_production': True,
              'served_controls': controls, 'prior_returned_controls_match': True,
              'samples': [{'sample_id': r['sample_id'], 'provider_sample_identity': r['provider_sample_identity'],
                           'raw_receipt': binding(out / r['raw_receipt']),
                           'sample_receipt': binding(out / 'sample_receipts' / (r['sample_id'] + '.json'))} for r in records]})
    require(stage != 'full' or len(records) == expected_count, 'Incomplete production cohort')
    print(json.dumps({'cohort': cohort, 'stage': stage, 'authenticated_samples': len(records),
                      'expected_samples': expected_count, 'served_controls': controls}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'status', 'preflight', 'full'))
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--cohort', choices=tuple(RUNS), default='graph_countdown')
    parser.add_argument('--credential-file', type=Path)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    require(1 <= args.workers <= 64, 'Workers must be 1-64')
    if args.command == 'prepare':
        manifest = prepare(args.base)
        print(json.dumps({'prepared': manifest['collection_runs'], 'new_requests': manifest['total_new_requests'], 'model_calls': 0}))
    elif args.command == 'status':
        manifest = authenticate_base(args.base)
        for cohort, item in manifest['collection_runs'].items():
            path = Path(item['run_dir']) / 'status.json'
            print(json.dumps({'cohort': cohort, 'status': json.loads(path.read_text()) if path.exists() else 'not_started'}))
    else:
        collect(args.base, args.cohort, args.command, args.credential_file, args.workers)


if __name__ == '__main__':
    main()
