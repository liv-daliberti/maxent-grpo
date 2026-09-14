#!/usr/bin/env python3
"""Resume offline grading and optional final analysis of the frozen32x512 study.

Default: inspect once and grade completed, unlocked new cohorts. --analyze-all
also runs the independent full collection audit and35-source final analysis
when all15 cohorts are complete. --watch repeats until its requested work ends.
No collection command is invoked; provider credentials are removed from child
environments. Existing frozen collection and analysis sources remain unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
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
BASE = ROOT / 'artifacts/modebench_discovery_all_levels_sol32x512_20260913'
ANALYZER = ROOT / 'ops/analyze_gpt56_all_levels32_discovery.py'
AUDITOR = ROOT / 'ops/audit_gpt56_all_levels32_collection.py'
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')
EXPECTED_COHORTS = {f'L{level}_{domain}' for level in (1, 2, 3) for domain in DOMAINS}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def check_binding(item, expected_path=None):
    path = ROOT / item['path']
    if expected_path is not None:
        require(path.resolve() == Path(expected_path).resolve(), 'Binding points to a different artifact')
    require(sha(path) == item['sha256'], 'Changed bound artifact: ' + str(path))


def write_json(path, value):
    """Atomic progress updates tolerate a terminated watcher."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def authenticate_plan(base):
    manifest = read_json(base / 'manifest.json')
    require(set(manifest['collection_runs']) == EXPECTED_COHORTS and manifest['new_requests'] == 122880
            and manifest['final_responses'] == 245760 and manifest['prompts'] == 480,
            'Expected the complete frozen15-cohort32x512 plan')
    for name, expected in manifest['artifact_sha256'].items():
        require(sha(base / name) == expected, 'Changed frozen collection input: ' + name)
    frozen = read_json(base / 'preparation_code/manifest.json')
    sources = {str((ROOT / item['source']['path']).resolve()): item for item in frozen['files']}
    for source in (ANALYZER, AUDITOR):
        require(str(source) in sources, 'Required offline tool was not prospectively frozen')
        item = sources[str(source)]
        check_binding(item['source'], source)
        check_binding(item['copy'])
        require(item['source']['sha256'] == item['copy']['sha256'], 'Frozen tool copy differs from its source')
    runtime = read_json(base / 'execution_runtime.json')
    require(Path(sys.executable).resolve() == Path(runtime['python_executable']).resolve(),
            'Use the unchanged original runtime: ' + runtime['python_executable'])
    prior = read_json(base / 'retained_prior_runs.json')
    require(len(prior['runs']) == 20 and prior['responses'] == 122880 and prior['prompts'] == 240,
            'Expected all20 immutable prior sources')
    check_binding(prior['analysis'])
    for name, entry in manifest['collection_runs'].items():
        require(Path(entry['run_dir']).resolve() == base / 'hosted/gpt56sol' / name
                and entry['new_samples'] == 8192, 'Changed new cohort location or budget')
        check_binding(entry['manifest'], Path(entry['run_dir']) / 'manifest.json')
    return manifest, prior


@contextmanager
def available_locks(paths):
    """Probe and hold existing collector locks without waiting or creating them."""
    with ExitStack() as stack:
        try:
            for path in dict.fromkeys(Path(p) for p in paths):
                handle = stack.enter_context(path.open('r'))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, FileNotFoundError):
            stack.close()
            yield False
        else:
            yield True


def complete_status(out, expected=8192):
    path = out / 'status.json'
    if not path.exists():
        return False
    try:
        state = read_json(path)
    except json.JSONDecodeError:
        return False  # A running collector may be midway through a status write.
    if state.get('complete') is not True:
        return False
    require(state.get('completed_samples') == state.get('expected_samples') == expected
            and state.get('failed_groups_this_session') == 0, 'Complete status has an invalid sample budget or failures')
    return True


def grading_complete(out, expected=8192):
    audit_path = out / 'all_domain_grading_audit.json'
    audit = existing_json(audit_path)
    if audit is None:
        return False  # Interrupted audit/grade files are preserved before retry.
    require(audit.get('status') == 'complete' and audit.get('responses') == expected and audit.get('api_calls') == 0,
            'Invalid existing grading audit; preserve and inspect it')
    for key, path in [('samples', out / 'samples.jsonl'), ('grades', out / 'discovery_hosted_grades.jsonl'),
                      ('grader', out / 'code/ops/frontier_modebench_contract.py'),
                      ('normalizer', out / 'code/ops/frontier_modebench_normalization.py')]:
        check_binding(audit[key], path)
    check_binding(audit['grader_driver'], ROOT / 'ops/analyze_gpt56_all_levels_discovery.py')
    return True


def run_offline(command, log_path):
    environment = {key: value for key, value in os.environ.items()
                   if key not in ('AZURE_OPENAI_API_KEY', 'OPENAI_API_KEY', 'AZURE_OPENAI_AD_TOKEN')}
    with Path(log_path).open('a') as log:
        result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    require(result.returncode == 0, 'Offline stage failed; inspect ' + str(log_path))


def grade_completed(manifest, logs, runner=run_offline):
    states = {}
    for name, entry in sorted(manifest['collection_runs'].items()):
        out = Path(entry['run_dir'])
        if not complete_status(out, entry['new_samples']):
            states[name] = 'waiting_for_collection'
            continue
        with available_locks([out / '.all_levels_orchestrator.lock', out / '.runner.lock']) as ready:
            if not ready:
                states[name] = 'waiting_for_collection_locks'
                continue
            require(complete_status(out, entry['new_samples']), 'Collection status changed under its locks')
            if grading_complete(out, entry['new_samples']):
                states[name] = 'reused_authenticated_grades'
                continue
            partial = out / 'discovery_hosted_grades.jsonl'
            if partial.exists():
                partial.rename(partial.with_name('discovery_hosted_grades.unaudited-' + str(time.time_ns()) + '.jsonl'))
            runner([sys.executable, str(ANALYZER), '--grade-run', str(out)], logs / (name + '_grading.log'))
            require(grading_complete(out, entry['new_samples']), 'Grading child did not produce an authenticated completion')
            states[name] = 'graded_complete'
    return states


def collection_locks(base, manifest, prior):
    paths = [base / '.production_scheduler.lock']
    for entry in manifest['collection_runs'].values():
        out = Path(entry['run_dir'])
        paths.extend([out / '.all_levels_orchestrator.lock', out / '.runner.lock'])
    paths.append((ROOT / prior['collection_manifest']['path']).parent / '.production_scheduler.lock')
    for entry in prior['runs']:
        out = Path(entry['directory'])
        paths.append(out / '.runner.lock')
        if (out / '.all_levels_orchestrator.lock').exists():
            paths.append(out / '.all_levels_orchestrator.lock')
    return paths


def existing_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return read_json(path)
    except json.JSONDecodeError:
        # Preserve interrupted offline output, then regenerate from locked inputs.
        path.rename(path.with_name(path.name + '.incomplete-' + str(time.time_ns())))
        return None


def authenticated_existing_audit(base, manifest, prior):
    path = base / 'collection_completion_audit.json'
    audit = existing_json(path)
    if audit is None:
        return False
    require(audit.get('status') == 'complete' and audit.get('total_authenticated_responses') == 245760
            and audit.get('new_authenticated_responses') == audit.get('retained_authenticated_responses') == 122880
            and audit.get('unique_prompts') == 480 and audit.get('api_calls_by_audit') == 0,
            'Invalid existing full collection audit')
    require(all(audit.get(key) is True for key in ('all_execution_locks_released_before_audit',
                'all_retained_native_receipts_reauthenticated', 'global_provider_samples_unique',
                'global_sample_slots_disjoint', 'global_sample_slots_gap_free')), 'Incomplete collection-audit guarantees')
    check_binding(audit['collection_manifest'], base / 'manifest.json')
    check_binding(audit['audit_driver'], AUDITOR)
    require(set(audit['runs']) == set(manifest['collection_runs']) and len(audit['retained_runs']) == 20,
            'Incomplete audited run inventory')
    for name, entry in audit['runs'].items():
        out = Path(manifest['collection_runs'][name]['run_dir'])
        for field, filename in [('manifest', 'manifest.json'), ('samples', 'samples.jsonl'), ('status', 'status.json')]:
            check_binding(entry[field], out / filename)
    expected_old = {str(Path(r['directory']).resolve()) for r in prior['runs']}
    require({str(Path(r['directory']).resolve()) for r in audit['retained_runs']} == expected_old, 'Changed retained audit inventory')
    for entry in audit['retained_runs']:
        out = Path(entry['directory'])
        check_binding(entry['manifest'], out / 'manifest.json')
        check_binding(entry['samples'], out / 'samples.jsonl')
    return True


def authenticated_existing_analysis(base, directories, path=None):
    path = Path(path) if path is not None else base / 'analysis/analysis.json'
    report = existing_json(path)
    if report is None:
        return False
    require(report.get('status') == 'complete' and report.get('responses') == 245760 and report.get('prompts') == 480,
            'Invalid existing expanded analysis; preserve and inspect it')
    require({str(Path(s['directory']).resolve()) for s in report['sources']} == set(directories)
            and len(report['sources']) == 35, 'Existing analysis has an incomplete source inventory')
    check_binding(report['analyzer'], ANALYZER)
    check_binding(report['new_support'], base / 'support_reference.json')
    check_binding(report['support_certificate'], base / 'support_certificate_manifest.json')
    for source in report['sources']:
        out = Path(source['directory'])
        for field, filename in [('manifest', 'manifest.json'), ('samples', 'samples.jsonl'), ('grades', 'discovery_hosted_grades.jsonl')]:
            check_binding(source[field], out / filename)
    return True


def finalize_if_ready(base, manifest, prior, states, logs, workers, runner=run_offline):
    require(set(states) == set(manifest['collection_runs']), 'Incomplete progress inventory')
    if len(states) != 15 or any(value not in ('graded_complete', 'reused_authenticated_grades') for value in states.values()):
        return 'waiting_for_all_fifteen_graded_cohorts'
    locks = collection_locks(base, manifest, prior)
    with available_locks(locks) as ready:
        if not ready:
            return 'waiting_for_all_collection_locks'
        require(all(complete_status(Path(entry['run_dir'])) for entry in manifest['collection_runs'].values()),
                'A complete cohort changed before final audit')
        audit_exists = authenticated_existing_audit(base, manifest, prior)
    if not audit_exists:
        # The independent auditor acquires every collector lock itself.
        runner([sys.executable, str(AUDITOR), '--base', str(base), '--workers', str(workers)], logs / 'collection_audit.log')
    with available_locks(locks) as ready:
        if not ready:
            return 'waiting_for_analysis_collection_locks'
        require(authenticated_existing_audit(base, manifest, prior), 'Full native collection audit is absent')
        directories = [str(Path(entry['directory']).resolve()) for entry in prior['runs']]
        directories += [str(Path(entry['run_dir']).resolve()) for _, entry in sorted(manifest['collection_runs'].items())]
        require(len(set(directories)) == len(directories) == 35, 'Expected35 distinct final source runs')
        if not authenticated_existing_analysis(base, directories):
            pending = base / 'analysis' / ('analysis.attempt-' + str(time.time_ns()) + '.json')
            command = [sys.executable, str(ANALYZER), '--support', str(base / 'support_reference.json'),
                       '--output', str(pending), '--workers', str(workers)]
            for directory in directories:
                command.extend(['--run', directory])
            runner(command, logs / 'final_analysis.log')
            require(authenticated_existing_analysis(base, directories, pending), 'Final analysis child did not authenticate its output')
            pending.rename(base / 'analysis/analysis.json')
    return 'analysis_complete'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--once', action='store_true', help='Inspect once; this is the default')
    mode.add_argument('--watch', action='store_true', help='Repeat until all requested offline work completes')
    parser.add_argument('--analyze-all', action='store_true', help='After all15 cohorts finish, audit all native evidence and build the merged analysis')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--interval', type=float, default=60)
    args = parser.parse_args()
    require(1 <= args.workers <= 8 and 1 <= args.interval <= 60, 'Use1--8 workers and a1--60 second watch interval')
    base = args.base.resolve()
    manifest, prior = authenticate_plan(base)
    logs = base / 'postcollection_logs'
    logs.mkdir(exist_ok=True)
    with (base / '.analysis_completion.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            states = grade_completed(manifest, logs)
            finalization = (finalize_if_ready(base, manifest, prior, states, logs, args.workers)
                            if args.analyze_all else 'not_requested')
            result = {'schema': 'sol32x512-offline-completion-v1', 'at_utc': datetime.now(timezone.utc).isoformat(),
                      'cohorts': states, 'graded_cohorts': sum(s in ('graded_complete', 'reused_authenticated_grades') for s in states.values()),
                      'finalization': finalization, 'collection_manifest': binding(base / 'manifest.json'),
                      'completion_driver': binding(__file__), 'api_calls': 0}
            write_json(base / 'postcollection_status.json', result)
            print(json.dumps(result, sort_keys=True), flush=True)
            done = finalization == 'analysis_complete' if args.analyze_all else result['graded_cohorts'] == 15
            if not args.watch or done:
                return
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
