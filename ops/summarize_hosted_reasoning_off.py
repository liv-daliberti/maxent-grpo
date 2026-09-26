#!/usr/bin/env python3
"""Offline paired reasoning-control summaries; never query a model or edit raw evidence.

Only complete 480-prompt deployments are admitted. Missing/partial deployments
receive status records without scores. Serial Python audit and frozen formatting
normalization write derived sidecars only; original medium samples are reused.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/hosted_reasoning_off_32_20260911_v2'
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')
LEVELS = (1, 2, 3)
PROMPTS_PER_CELL, DRAWS = 32, 8
PROMPTS, RESPONSES = 480, 3840
CONDITION = 'hosted-reasoning-off-first32-v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prompt_key(record):
    return record['level'], record['domain'], record['row_index']


def slot_key(record):
    return (*prompt_key(record), record['sample_index'])


def validate_rows(rows):
    expected = {(level, domain, index) for level in LEVELS for domain in DOMAINS
                for index in range(PROMPTS_PER_CELL)}
    require(len(rows) == PROMPTS and {prompt_key(row) for row in rows} == expected,
            'Require exactly the first 32 frozen rows in every domain and level')
    require(all(type(row['level']) is int and type(row['row_index']) is int for row in rows),
            'Prompt identities must be integers')
    return expected


def summarize_cohort(rows, samples):
    """Empirical prompt success and verified canonical-mode counts on a full grid."""
    expected = validate_rows(rows)
    slots = {(*key, index) for key in expected for index in range(DRAWS)}
    require(len(samples) == RESPONSES and {slot_key(s) for s in samples} == slots,
            'Require all eight unique draws for all 480 prompts before any score is admitted')
    grouped = defaultdict(list)
    for sample in samples:
        require(type(sample['sample_index']) is int and type(sample['verified']) is bool,
                'Invalid sample index or verified outcome')
        require(not sample['verified'] or sample.get('canonical_key') is not None,
                'Verified response is missing its canonical mode')
        grouped[prompt_key(sample)].append(sample)
    prompt_stats = []
    for level, domain, index in sorted(expected):
        group = grouped[level, domain, index]
        modes = {sha(s['canonical_key']) for s in group if s['verified']}
        prompt_stats.append({'level': level, 'domain': domain, 'row_index': index,
                             'correct_responses': sum(s['verified'] for s in group),
                             'pass8': int(bool(modes)), 'distinct8': len(modes)})

    def aggregate(prompts):
        counts = {'prompts': len(prompts), 'responses': DRAWS * len(prompts),
                  'prompts_with_correct': sum(p['pass8'] for p in prompts),
                  'distinct_correct_modes': sum(p['distinct8'] for p in prompts),
                  'correct_responses': sum(p['correct_responses'] for p in prompts)}
        return {'counts': counts, 'metrics': {
            'pass8': counts['prompts_with_correct'] / counts['prompts'],
            'distinct8': counts['distinct_correct_modes'] / counts['prompts']}}

    cells = {f'level{level}/{domain}': aggregate([p for p in prompt_stats
             if p['level'] == level and p['domain'] == domain])
             for level in LEVELS for domain in DOMAINS}
    levels = {str(level): aggregate([p for p in prompt_stats if p['level'] == level]) for level in LEVELS}
    return {'cells': cells, 'levels': levels, 'overall': aggregate(prompt_stats), 'prompts': prompt_stats}


def collection_state(directory):
    """Do not turn absent draws or an incomplete model into scored failures."""
    directory = Path(directory)
    paths = list((directory / 'sample_receipts').glob('*.json'))
    violation_path = directory / 'provider_control_violation.json'
    if violation_path.exists():
        violation = json.loads(violation_path.read_text())
        return {'status': 'blocked_control_violation', 'terminal_sample_receipts': len(paths),
                'expected_responses': RESPONSES, 'scores_admitted': False,
                'reason': violation['reason'], 'control_violation': {
                    'path': str(violation_path), 'sha256': file_sha(violation_path)}}
    if len(paths) != RESPONSES or not (directory / 'samples.jsonl').exists():
        return {'status': 'pending_collection', 'terminal_sample_receipts': len(paths),
                'expected_responses': RESPONSES, 'scores_admitted': False}
    samples = read_jsonl(directory / 'samples.jsonl')
    rows = read_jsonl(directory / 'rows.jsonl')
    expected = validate_rows(rows)
    require(len(samples) == RESPONSES and {slot_key(s) for s in samples} == {
        (*key, i) for key in expected for i in range(DRAWS)}, 'Incomplete or duplicate finalized sample slots')
    require({p.stem for p in paths} == {s['sample_id'] for s in samples},
            'Atomic and finalized sample identities differ')
    require(len({s['sample_id'] for s in samples}) == RESPONSES, 'Duplicate sample identifiers')
    require(all(json.loads((directory / 'sample_receipts' / (s['sample_id'] + '.json')).read_text()) == s
                for s in samples), 'Atomic and finalized sample receipts differ')
    return {'status': 'ready_for_offline_audit', 'terminal_sample_receipts': RESPONSES,
            'expected_responses': RESPONSES, 'scores_admitted': False}


def normalized_samples(directory, primary, summary):
    """Reconstruct frozen normalized outcomes using exact strict receipt hashes."""
    directory = Path(directory)
    secondary = summary['normalized_secondary']
    cache_path = directory / 'normalized_samples.jsonl'
    require(file_sha(cache_path) == secondary['cache_sha256'], 'Normalized cache differs from source summary')
    for path_key, hash_key in (('normalization_source_path', 'normalization_source_sha256'),
                               ('frozen_grader_contract_path', 'frozen_grader_contract_sha256')):
        require(file_sha(secondary[path_key]) == secondary[hash_key], 'Frozen grading source changed')
    original = {slot_key(s): s for s in read_jsonl(directory / 'samples.jsonl')}
    reconciliation = secondary.get('python_cache_reconciliation')
    if reconciliation:
        require(reconciliation.get('status') == 'complete'
                and file_sha(reconciliation['path']) == reconciliation['sha256'],
                'Normalized Python reconciliation is not authenticated')
    cache = {}
    for item in read_jsonl(cache_path):
        if item['normalization_source_sha256'] == secondary['normalization_source_sha256']:
            key = item['strict_receipt_sha256']
            require(key not in cache or cache[key] == item['normalization']
                    or (reconciliation and 'python_grade_reconciliation' in item),
                    'Conflicting normalized cache receipts without audited reconciliation')
            cache[key] = item['normalization']
    result = []
    for sample in primary:
        raw = original[slot_key(sample)]
        receipt = {**raw, **{k: sample[k] for k in ('verified', 'canonical_key', 'graded_text')}}
        grade = cache[sha(receipt)]
        require(not sample['verified'] or (grade['verified'] and grade['canonical_key'] == sample['canonical_key']),
                'Normalization changed a strict success or its mode')
        result.append({**sample, 'verified': grade['verified'], 'canonical_key': grade['canonical_key']})
    return result


def paired_metrics(rows, on, off):
    result = {'on': summarize_cohort(rows, on), 'off': summarize_cohort(rows, off)}
    result['off_minus_on'] = {str(level): {
        metric: result['off']['levels'][str(level)]['metrics'][metric]
        - result['on']['levels'][str(level)]['metrics'][metric]
        for metric in ('pass8', 'distinct8')} for level in LEVELS}
    return result


def frozen_off_analysis(directory):
    """Run existing audited grading tools in isolated processes, never inference."""
    sys.path.insert(0, str(ROOT / 'ops'))
    from postprocess_frontier_temperature import freeze_assets
    freeze_assets(directory)
    code = directory / 'analysis_code/ops'
    logs = directory / 'reasoning_comparison_logs'
    logs.mkdir(exist_ok=True)
    steps = [
        ('completion', [sys.executable, '-c',
         'import sys;sys.path.insert(0,sys.argv[1]);from audit_hosted_modebench_completion import audit;audit(sys.argv[2],expected_samples=3840)',
         str(code), str(directory)]),
        ('python_audit', [sys.executable, str(code / 'audit_hosted_modebench_python.py'),
                          '--output-root', str(directory), '--expected-samples', str(RESPONSES)]),
        ('normalized_summary', [sys.executable, str(directory / 'secondary_code/ops/summarize_frontier_modebench.py'),
                                '--input-dir', str(directory), '--primary-samples', 'audited_primary_samples.jsonl',
                                '--with-normalization', '--bootstrap-replicates', '2000', '--seed', '20260911', '--no-plot']),
    ]
    for name, command in steps:
        print(json.dumps({'model_run': directory.name, 'offline_step': name, 'api_calls': 0}), flush=True)
        with (logs / (name + '.log')).open('w') as handle:
            result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
        require(result.returncode == 0, f'Offline {name} failed; see {logs / (name + ".log")}')


def analyze_run(directory):
    """Authenticate complete paired evidence and publish one model admission record."""
    directory = Path(directory).resolve()
    require(collection_state(directory)['status'] == 'ready_for_offline_audit', 'Deployment is incomplete')
    marker = directory / 'reasoning_comparison.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        if (saved['analysis_script_sha256'] == file_sha(__file__)
                and all(file_sha(path) == digest for path, digest in saved['evidence_sha256'].items())):
            return saved
    sys.path.insert(0, str(ROOT / 'ops'))
    from audit_hosted_modebench_completion import load_inventory, validate_native_records
    inventory = load_inventory(directory, expected_samples=RESPONSES)
    manifest = inventory['manifest']
    reference = Path(manifest['reference_run'])
    require(manifest.get('experiment_condition') == CONDITION and manifest['prompt_count'] == PROMPTS,
            'Wrong registered reasoning-off experiment')
    require(file_sha(reference / 'manifest.json') == manifest['reference_manifest_sha256'], 'Reference manifest changed')
    require(file_sha(reference / 'audited_primary_samples.jsonl') == manifest['reference_primary_samples_sha256'],
            'Original audited medium samples changed')
    reference_manifest = json.loads((reference / 'manifest.json').read_text())
    for name, digest in reference_manifest['artifact_sha256'].items():
        require(file_sha(reference / name) == digest, 'Frozen reference artifact changed: ' + name)
    rows = list(inventory['rows'].values())
    validate_rows(rows)
    source_rows = {prompt_key(r): r for r in read_jsonl(reference / 'rows.jsonl')}
    require(all(source_rows[prompt_key(row)] == row for row in rows), 'Original prompt/task bytes changed')
    source_requests = {item['sample_id']: item for item in read_jsonl(reference / 'requests.jsonl')}
    original_primary = {s['sample_id']: s for s in read_jsonl(reference / 'audited_primary_samples.jsonl')}
    medium = read_jsonl(directory / 'paired_medium_samples.jsonl')
    require(len(medium) == RESPONSES and {s['sample_id'] for s in medium} == set(inventory['expected']),
            'Paired medium sample inventory differs')
    require(all(s == original_primary[s['sample_id']] for s in medium), 'Paired medium sample content changed')
    prepare = load_module(directory / 'code/ops/prepare_hosted_reasoning_off.py', '_frozen_off_prepare')
    slug = next(key for key, model in prepare.SOURCES.items() if model == manifest['model'])
    for item in inventory['requests']:
        original = source_requests[item['sample_id']]
        require(item['reference_request_sha256'] == original['request_sha256'], 'Reference request hash mismatch')
        require(item['request'] == prepare.off_payload(original['request'], slug),
                'A request setting besides the provider reasoning control changed')
    raw_samples = read_jsonl(directory / 'samples.jsonl')
    raw_cache = validate_native_records(inventory, raw_samples)
    wrapper = load_module(directory / 'code/ops/evaluate_hosted_reasoning_off.py', '_frozen_off_control')
    control_auditor = load_module(ROOT / 'ops/evaluate_hosted_reasoning_off.py', '_current_off_control_audit')
    all_raw_paths = sorted((directory / 'raw_responses').glob('*.json'))
    generated_slots = []
    for path in all_raw_paths:
        raw = json.loads(path.read_text())
        if raw.get('http_status') != 200:
            continue
        sid = raw.get('sample_id') or raw.get('group_id')
        require(sid in inventory['expected'], 'Unexpected generated response slot')
        item = inventory['expected'][sid]
        require(raw['request_sha256'] == item['request_sha256'], 'Generated response request differs')
        control_auditor.control_evidence(item['request'], raw['response'])
        generated_slots.append(sid)
    require(len(generated_slots) == RESPONSES and set(generated_slots) == set(inventory['expected']),
            'Extra, missing, or selectively admitted generated responses')
    controls = sorted((directory / 'control_receipts').glob('*.json'))
    require({p.stem for p in controls} == set(inventory['expected']), 'Incomplete native off-control receipt inventory')
    for sample in raw_samples:
        item = inventory['expected'][sample['sample_id']]
        raw = raw_cache[sample['raw_receipt']]
        control_auditor.control_evidence(item['request'], raw['response'])
        control = json.loads((directory / 'control_receipts' / (sample['sample_id'] + '.json')).read_text())
        expected_control = {'sample_id': sample['sample_id'], 'request_sha256': item['request_sha256'],
                            'raw_receipt': raw['relative_path'], 'raw_receipt_sha256': sha(raw),
                            'native_control_check': wrapper.control_evidence(item['request'], raw['response'])}
        require(control == expected_control, 'Native off-control evidence changed')
    evidence_paths = [ROOT / 'ops/evaluate_hosted_reasoning_off.py', directory / 'manifest.json', directory / 'samples.jsonl', reference / 'manifest.json',
                      reference / 'audited_primary_samples.jsonl', reference / 'summary.json',
                      reference / 'normalized_samples.jsonl', reference / 'samples.jsonl', reference / 'requests.jsonl',
                      reference / 'rows.jsonl', reference / 'primary_python_regrade_audit.json']
    evidence_paths += [directory / name for name in manifest['artifact_sha256']]
    evidence_paths += [directory / 'code' / name for name in manifest['code_sha256']]
    evidence_paths += list((directory / 'sample_receipts').glob('*.json')) + controls
    evidence_paths += all_raw_paths
    before = {str(path): file_sha(path) for path in evidence_paths}
    frozen_off_analysis(directory)
    require(all(file_sha(path) == digest for path, digest in before.items()), 'Primary evidence changed during offline analysis')
    off_summary = json.loads((directory / 'summary.json').read_text())
    on_summary = json.loads((reference / 'summary.json').read_text())
    off_primary = read_jsonl(directory / 'audited_primary_samples.jsonl')
    for root, summary, expected_count in ((directory, off_summary, RESPONSES), (reference, on_summary, 15360)):
        require(summary['status'] == 'complete' and summary['received_responses'] == expected_count,
                'Grading summary is incomplete')
        require(file_sha(root / 'audited_primary_samples.jsonl') == summary['primary_samples_sha256'],
                'Grading summary primary hash mismatch')
        audit = json.loads((root / 'primary_python_regrade_audit.json').read_text())
        require(audit['status'] == 'complete_for_snapshot' and audit['full_sampling_complete']
                and audit['derived_primary']['sha256'] == summary['primary_samples_sha256'],
                'Serial Python audit does not authenticate the full graded cohort')
    for field in ('normalization_source_sha256', 'frozen_grader_contract_sha256'):
        require(on_summary['normalized_secondary'][field] == off_summary['normalized_secondary'][field],
                'Reasoning conditions use different grading sources')
    analyses = {
        'strict': paired_metrics(rows, medium, off_primary),
        'normalized': paired_metrics(rows, normalized_samples(reference, medium, on_summary),
                                     normalized_samples(directory, off_primary, off_summary)),
    }
    derived = ['audited_primary_samples.jsonl', 'primary_python_regrade_audit.json', 'completion_audit.json',
               'normalized_samples.jsonl', 'summary.json', 'postprocessing_source_manifest.json']
    before.update({str(directory / name): file_sha(directory / name) for name in derived})
    for summary in (on_summary, off_summary):
        reconciliation = summary['normalized_secondary'].get('python_cache_reconciliation')
        if reconciliation:
            before[reconciliation['path']] = file_sha(reconciliation['path'])
        for key in ('normalization_source_path', 'frozen_grader_contract_path'):
            path = summary['normalized_secondary'][key]
            before[path] = file_sha(path)
    result = {'schema': 'hosted-reasoning-paired-model-summary-v1', 'status': 'admitted_complete',
              'model': manifest['model'], 'run_directory': str(directory), 'analysis_script_sha256': file_sha(__file__),
              'evidence_sha256': before, 'api_calls': 0, 'analyses': analyses,
              'settings': {'on': manifest['original_reasoning_configuration'],
                           'off_request_controls': {key: value for key, value in inventory['requests'][0]['request'].items()
                                                    if key not in ('input', 'messages', 'system')},
                           'off_native_evidence': 'Every accepted response authenticated against its saved native control receipt.'},
              'protocol': json.loads((directory / 'reasoning_condition.json').read_text()),
              'returned_sampling': {condition: {field: dict(Counter(json.dumps(sample.get(field), sort_keys=True)
                                    for sample in samples)) for field in ('temperature', 'top_p', 'reasoning', 'service_tier')}
                                    for condition, samples in (('on', medium), ('off', off_primary))}}
    write_json(marker, result)
    return result


def resolved_registry(base):
    sys.path.insert(0, str(ROOT / 'ops'))
    from run_hosted_reasoning_off import registry_for
    return registry_for(Path(base))


def build_report(base=BASE):
    base = Path(base).resolve()
    registry_path = base / 'experiment.json'
    registry = resolved_registry(base)
    require(registry['schema'] == CONDITION, 'Wrong experiment registry')
    runs = registry['runs']
    require(len(runs) == 7 and len({r['model'] for r in runs}) == 7 and len({r['slug'] for r in runs}) == 7,
            'Expected seven distinct registered deployments')
    admitted, pending = [], []
    for run in runs:
        directory = Path(run['run_directory'])
        state = {'slug': run['slug'], 'model': run['model'], 'run_directory': str(directory)}
        try:
            require(file_sha(directory / 'manifest.json') == run['manifest_sha256'], 'Registered manifest changed')
            progress = collection_state(directory)
            if progress['status'] != 'ready_for_offline_audit':
                pending.append({**state, **progress})
                continue
            log = directory / 'reasoning_comparison_worker.log'
            with log.open('w') as handle:
                process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--analyze-run', str(directory)],
                                         cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
            require(process.returncode == 0, f'Offline admission failed; see {log}')
            result = json.loads((directory / 'reasoning_comparison.json').read_text())
            require(result['model'] == run['model'] and result['status'] == 'admitted_complete', 'Wrong admitted model')
            admitted.append(result)
        except (ValueError, KeyError, OSError) as exc:
            pending.append({**state, 'status': 'not_admitted', 'scores_admitted': False, 'reason': str(exc)})
    return {'schema': 'hosted-reasoning-paired-summary-v1', 'status': 'complete' if len(admitted) == 7 else 'incomplete',
            'registry': {'path': str(registry_path), 'sha256': file_sha(registry_path)},
            'analysis_script': {'path': str(Path(__file__).resolve()), 'sha256': file_sha(__file__)},
            'deployment_amendments': ({'path': str(base / 'deployment_amendments.json'),
                'sha256': file_sha(base / 'deployment_amendments.json')}
                if (base / 'deployment_amendments.json').exists() else None),
            'registry_resolver': {'path': str(ROOT / 'ops/run_hosted_reasoning_off.py'),
                                  'sha256': file_sha(ROOT / 'ops/run_hosted_reasoning_off.py')},
            'api_calls': 0, 'models': admitted, 'pending_models': pending,
            'metric_definitions': {'pass8': 'Fraction of prompts with at least one verified answer among all eight observed draws.',
                                   'distinct8': 'Mean distinct verified canonical modes in eight draws, including zero-success prompts.'},
            'aggregation': 'Equal weights on five domains within each level: 160 prompts and 1,280 responses per model/condition/level.',
            'limits': ['Only complete paired deployments are scored; missing responses are not assigned zero and partial models are not pooled.',
                       'First 32 original row indices per domain/level were selected without consulting outputs; this is a 480-prompt subset.',
                       'The on condition reuses historical medium-control outputs; collection time differs, and matching slots are not shared random seeds.',
                       'Both conditions use original prompt wording, including Opus 5 Python; the revised Python cohort in the main display is separate.',
                       'Only the provider reasoning control changes in requests; omitted/default sampling controls are retained. Provider defaults may change over time.',
                       'A supported disabled-thinking control does not establish absence of hidden internal computation.',
                       'These descriptive point estimates include no uncertainty intervals or causal training claims. The main manuscript table is unchanged.']}


def markdown(report):
    lines = ['# Hosted reasoning-control comparison', '',
             f"Status: **{report['status']}**; {len(report['models'])}/7 complete deployments admitted. No model API calls were made by this analysis.", '',
             report['aggregation'], '',
             '| Model | Level | On pass@8 (%) | On distinct@8 | Off pass@8 (%) | Off distinct@8 |',
             '|---|---:|---:|---:|---:|---:|']
    for model in report['models']:
        for level in LEVELS:
            on = model['analyses']['normalized']['on']['levels'][str(level)]['metrics']
            off = model['analyses']['normalized']['off']['levels'][str(level)]['metrics']
            lines.append(f"| {model['model']} | {level} | {100*on['pass8']:.1f} | {on['distinct8']:.2f} | {100*off['pass8']:.1f} | {off['distinct8']:.2f} |")
    if not report['models']:
        lines += ['', 'No scores are available until a deployment passes complete-cohort admission.']
    lines += ['', 'The table uses the frozen formatting normalizer in both conditions. Strict grades, all domain estimates, prompt counts, controls, and source hashes remain in the JSON.', '']
    if report['pending_models']:
        lines += ['## Pending or not admitted', '']
        for item in report['pending_models']:
            detail = item.get('reason', f"{item.get('terminal_sample_receipts', 0)}/{RESPONSES} terminal receipts")
            lines.append(f"- {item['model']}: {item['status']}; {detail}.")
    lines += ['', '## Interpretation', '', *['- ' + limit for limit in report['limits']], '']
    return '\n'.join(lines)


def active_collectors(base):
    """Check registered process identities, excluding exited/zombie or reused PIDs."""
    base = Path(base)
    paths = set(base.glob('*launch*.json'))
    paths.update(Path(run['run_directory']) / 'active_collection.json'
                 for run in resolved_registry(base)['runs'])
    active = []
    for path in sorted(paths):
        if not path.exists():
            continue
        marker = json.loads(path.read_text())
        pid, command = marker.get('pid'), marker.get('command')
        if (not isinstance(pid, int) or not isinstance(command, list)
                or not any(Path(arg).name in ('run_hosted_reasoning_off.py', 'evaluate_hosted_reasoning_off.py')
                           for arg in command if isinstance(arg, str))):
            continue
        proc = Path('/proc') / str(pid)
        try:
            state = (proc / 'stat').read_text().rsplit(')', 1)[1].strip().split()[0]
            actual = (proc / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        if state != 'Z' and actual[1:] == command[1:]:
            active.append({'pid': pid, 'marker': str(path)})
    return active


def wait_for_collectors(base, max_wait_seconds=7200, poll_seconds=30):
    require(max_wait_seconds > 0 and 1 <= poll_seconds <= 60, 'Invalid bounded wait settings')
    started = time.monotonic()
    status_path = Path(base) / 'reasoning_comparison_finalizer_status.json'
    while True:
        active = active_collectors(base)
        elapsed = time.monotonic() - started
        state = 'collectors_stopped' if not active else 'wait_deadline_reached' if elapsed >= max_wait_seconds else 'waiting_for_collectors'
        write_json(status_path, {'status': state, 'elapsed_seconds': elapsed, 'active_collectors': active, 'api_calls': 0})
        if state != 'waiting_for_collectors':
            return state
        print(json.dumps({'status': state, 'active_collectors': len(active), 'elapsed_seconds': round(elapsed)}), flush=True)
        time.sleep(min(poll_seconds, max_wait_seconds - elapsed))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--output', type=Path, help='Summary output stem; defaults to BASE/reasoning_comparison')
    parser.add_argument('--analyze-run', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--wait-for-collectors', action='store_true', help='Wait for registered collectors to exit, then summarize complete and blocked models')
    parser.add_argument('--max-wait-seconds', type=float, default=7200)
    parser.add_argument('--poll-seconds', type=float, default=30)
    args = parser.parse_args(argv)
    if args.analyze_run:
        result = analyze_run(args.analyze_run)
        print(json.dumps({'model': result['model'], 'status': result['status'], 'api_calls': 0}))
        return 0
    if args.wait_for_collectors:
        wait_for_collectors(args.base, args.max_wait_seconds, args.poll_seconds)
    report = build_report(args.base)
    output = args.output or args.base / 'reasoning_comparison'
    write_json(output.with_suffix('.json'), report)
    output.with_suffix('.md').write_text(markdown(report))
    if args.wait_for_collectors:
        write_json(args.base / 'reasoning_comparison_finalizer_status.json', {
            'status': 'summary_written', 'summary_status': report['status'], 'output': str(output),
            'admitted_models': len(report['models']), 'pending_models': len(report['pending_models']), 'api_calls': 0})
    print(json.dumps({'status': report['status'], 'admitted_models': len(report['models']),
                      'pending_models': len(report['pending_models']), 'output': str(output), 'api_calls': 0}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
