#!/usr/bin/env python3
"""Shared v3 boundaries for adaptive matching to fixed measured L1 references.

The previous failed confirmation remains immutable. Historical L1 outcomes are
explicit benchmark targets; only new candidate DEV outcomes may fit a recipe.
No registration, dataset, receipt, fit or scheduler action is published here.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
for directory in ('ops', 'ops/exp_scaling', 'src'):
    if str(ROOT / directory) not in sys.path:
        sys.path.insert(0, str(ROOT / directory))

OLD_CAMPAIGN = ROOT / 'var/artifacts/modebench_level3_v2'
OLD_DATASET = ROOT / 'var/data/modebench_level3_matched_v2'
CAMPAIGN = ROOT / 'var/artifacts/modebench_level3_v3'
RESULTS = ROOT / 'var/results/modebench_level3_v3'
DATASET = ROOT / 'var/data/modebench_level3_matched_v3'
REGISTRATION = CAMPAIGN / 'registration.json'
OLD_SEAL = OLD_CAMPAIGN / 'confirmation_python_v6/seal.json'
OLD_SEAL_SHA = '5c9cae0c3d5509db20b8c22418da759710b2a9540eac74cc81ce8e743adca79c'
OLD_REPORT = OLD_CAMPAIGN / 'confirmation_python_v6/confirmation_report.json'
OLD_REPORT_SHA = 'fd77c76274b81c73bfb78626666cb4078d4b3968eecd5cac3b05f9e33c854b12'
OLD_REPLAY = OLD_CAMPAIGN / 'confirmation_python_v6_completed_review/original_grader_replay.json'
OLD_REPLAY_SHA = '62c112aa891d47f4e5db359f62d5fb1047235d64caeea55cd2c7b8cf72b4c6d5'
DOMAINS = ('countdown', 'graph_coloring', 'python_factors', 'mathir', 'pantry')
RETAINED = ('countdown', 'mathir', 'pantry')
REVISED = ('graph_coloring', 'python_factors')
DEV_LABELS = (6428000, 6428001, 6428002, 6428003)
CONF_LABELS = (6429000, 6429001, 6429002, 6429003)
REFERENCE_LABELS = (6329000, 6329001, 6329002, 6329003)
TOLERANCES = {'pass1': 0.04, 'pass8': 0.08}
SPLITS = {'train': 384, 'dev': 128, 'eval': 128}
SELECTION_SEED = 6391701
SEEDS = {
    'graph_coloring': {'development': 8737100, 'capacity': 8937100, 'train': 9137100, 'eval': 9337100},
    'python_factors': {'development': 8837100, 'capacity': 9037100, 'train': 9237100, 'eval': 9437100},
}
POOL_ROOTS = {domain: ROOT / ('var/data/modebench_level3_calibration_' + revision)
              for domain, revision in zip(REVISED, ('graph_v8', 'python_v7'))}
TARGET_SHA = {
    'countdown': 'd333ba84408706e031d71f95a8843b3462fe948acbcb7fe3c22dd2a6f78c8993',
    'graph_coloring': 'c46133cbefebd63c024f2c25edeb2ebd02a3c6aab5c4a696a0e5d067c6aaf9c4',
    'python_factors': '6787a0df405740a5846be55d675783d4a8b48912a71490dc96a7629f71b43cb9',
    'mathir': 'a025c4f943a7407b97944a2d0b57ca929c725b60465cfc0d1db5073af17cef58',
    'pantry': '6c623f16392f671f53eda9b9a41262d81a9a068f344bbfa7fb795765d7aff091',
}
REGISTRATION_SCHEMA = 'modebench_level3_v3_fixed_reference_registration_v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(part)
    return value.hexdigest()


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def merge_pins(*mappings):
    result = {}
    for mapping in mappings:
        for path, expected in mapping.items():
            require(path not in result or result[path] == expected, f'conflicting immutable pin: {path}')
            result[path] = expected
    return result


def verify_pins(files, directory_files=None):
    for path, expected in files.items():
        require(digest(path) == expected, f'immutable file changed: {path}')
    for directory, expected in (directory_files or {}).items():
        actual = sorted(str(path.resolve()) for path in Path(directory).rglob('*') if path.is_file())
        require(actual == expected and all(path in files for path in actual),
                f'immutable directory inventory changed: {directory}')


def atomic_new(path, payload):
    """Publish one new artifact exclusively; never replace an existing record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f'fresh output required: {path}')
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def inherited_metadata():
    return {'seal': {'path': str(OLD_SEAL), 'sha256': OLD_SEAL_SHA},
            'report': {'path': str(OLD_REPORT), 'sha256': OLD_REPORT_SHA},
            'original_grader_replay': {'path': str(OLD_REPLAY), 'sha256': OLD_REPLAY_SHA}}


def benchmark_metadata():
    return {domain: {'role': 'frozen_benchmark_reference',
                     'receipt_path': str(ROOT / f'var/results/modebench_level3_v2/confirmation_python_v6/confirmation_05b_{domain}.json'),
                     'receipt_sha256': expected}
            for domain, expected in TARGET_SHA.items()}


def contract():
    """JSON-compatible fixed policy; generation and sampling namespaces differ."""
    return {'reference_semantics': 'fixed_measured_level1_benchmark', 'adaptive_confirmation_round': 2,
            'retained_domains': list(RETAINED), 'revised_domains': list(REVISED),
            'split_sizes': dict(SPLITS), 'development_draw_labels': list(DEV_LABELS),
            'confirmation_draw_labels': list(CONF_LABELS), 'historical_reference_draw_labels': list(REFERENCE_LABELS),
            'seed_policy': 'sha256_domain_problem_draw_aligned_n8_v2', 'samples_per_draw': 8,
            'selection_seed': SELECTION_SEED, 'weight_denominator': 20, 'weight_combinations': 1771,
            'tolerances': dict(TOLERANCES), 'selected_development_sets_scored': 1,
            'selected_evaluation_sets_scored': 0,
            'selection_algorithm': 'dual_full_pool_dev_eval_forecasts_then_one_fixed_dev_selection_v3',
            'weight_objective': 'minimax_normalized_dev_eval_forecast_then_squared_sum_then_weights',
            'forecast_reference_splits': ['dev', 'eval'], 'selected_reference_split': 'dev',
            'required_development_gates': ['expected_dev', 'expected_eval', 'selected'],
            'calibration_quota_rule': 'componentwise_maximum_of_level2_dev_and_eval_cells',
            'calibration_rows_per_tier': {'graph_coloring': 142, 'python_factors': 166},
            'generation_seed_bases': deepcopy(SEEDS), 'generation_tier_stride': 1000,
            'pool_roots': {domain: str(path) for domain, path in POOL_ROOTS.items()},
            'dataset_root': str(DATASET), 'results_root': str(RESULTS),
            'old_rng_sources': 47, 'old_request_blocks': 23808,
            'new_development_sources': 8, 'new_development_request_blocks': 4928,
            'new_confirmation_sources': 2, 'new_confirmation_request_blocks': 1024,
            'total_sources': 57, 'total_request_blocks': 29760, 'total_child_seeds': 238080,
            'historical_level1_confirmation_used_as_fixed_reference': True,
            'historical_level3_confirmation_used_for_mixture_fitting': False,
            'new_candidate_fitting_uses_development_outcomes_only': True,
            'fresh_heldout_claim_applies_to_revised_level3_only': True,
            'all_five_fresh_same_round': False, 'statistical_equivalence_claimed': False,
            'treatment_training_started': False}


def authenticate_inherited():
    """Authenticate completed old evidence without repeating grading or fitting."""
    for item in inherited_metadata().values():
        require(digest(item['path']) == item['sha256'], 'inherited confirmation anchor changed')
    replay, report, seal = read(OLD_REPLAY), read(OLD_REPORT), read(OLD_SEAL)
    require(replay.get('schema') == 'modebench_level3_python_v6_completed_confirmation_original_grader_replay_v1'
            and replay.get('status') == 'verified_original_grader_replay'
            and replay.get('attempts') == 40960 and replay.get('batches') == 640
            and replay.get('jobs') == 10 and replay.get('rows') == 1280
            and replay.get('every_attempt_regraded_including_failures') is True
            and replay.get('full_canonical_keys_compared') is True
            and replay.get('recorded_verification_outcomes_regraded') is True
            and replay.get('draw_prompt_cell_metrics_and_all_five_gates_recomputed') is True
            and replay.get('report_sha256') == OLD_REPORT_SHA
            and replay.get('scientific_seal_sha256') == OLD_SEAL_SHA,
            'complete original-grader historical replay required')
    require(report.get('status') == 'outside_match_tolerance' and report.get('phase') == 'confirmation'
            and report.get('all_five_domains_complete') is True
            and report.get('confirmation_match_verified') is False
            and set(report.get('domains', {})) == set(DOMAINS)
            and report.get('errors') == {} and report.get('missing_domains') == [],
            'failed all-five historical confirmation must remain intact')
    require(len(seal['files_sha256']) == 2232 and len(seal['directory_files']) == 79
            and len(replay['files_sha256']) == 2978 and len(replay['directory_files']) == 89,
            'inherited complete input/evidence coverage differs')
    require(all(replay['files_sha256'].get(path) == expected for path, expected in seal['files_sha256'].items()),
            'replay omits an original sealed input')
    files = merge_pins(replay['files_sha256'], {item['path']: item['sha256'] for item in inherited_metadata().values()})
    verify_pins(files, replay['directory_files'])
    from evaluate_modebench_level3_independent import model_identity
    for label, expected in seal['models'].items():
        actual = model_identity(Path(expected['path']), label)
        actual['vllm_version'] = importlib.metadata.version('vllm')
        require(actual == expected, f'frozen model/engine changed: {label}')
    return {'report': report, 'seal': seal, 'replay': replay, 'models': seal['models'],
            'files_sha256': files, 'directory_files': deepcopy(replay['directory_files']),
            'metadata': inherited_metadata()}


def reference_histograms(domain):
    require(domain in DOMAINS, 'unknown reference domain')
    from fit_modebench_level3 import cell_histogram
    from materialize_modebench_level3 import reference_rows
    return {split: cell_histogram(domain, reference_rows(domain, split)) for split in SPLITS}


def calibration_histogram(domain):
    require(domain in REVISED, 'only revised domains receive fresh calibration')
    histograms = reference_histograms(domain)
    keys = set(histograms['dev']) | set(histograms['eval'])
    result = Counter({key: max(histograms['dev'][key], histograms['eval'][key]) for key in keys})
    require(sum(result.values()) == contract()['calibration_rows_per_tier'][domain], 'calibration quota total differs')
    return result


def fixed_references():
    """All five original eval receipts remain eval evidence, never synthetic DEV."""
    inherited = authenticate_inherited()
    from audit_modebench_level3_python_v6_independent_match import validate_receipt, verify_dataset
    from evaluate_modebench_level3_independent import load_rows
    from fit_modebench_level3 import cell_histogram, serialize_cells
    targets = {}
    for domain, metadata in benchmark_metadata().items():
        path = Path(metadata['receipt_path'])
        require(digest(path) == metadata['receipt_sha256'], 'frozen measured target changed')
        receipt = read(path)
        values = validate_receipt(receipt, domain=domain, role='baseline', development=False)
        recorded = inherited['report']['domains'][domain]
        require(recorded['receipts']['baseline']['sha256'] == metadata['receipt_sha256']
                and recorded['receipts']['baseline']['path'] == str(path)
                and receipt['identity']['model'] == inherited['models']['05b']
                and receipt['identity']['seeds'] == list(REFERENCE_LABELS), 'fixed target model/source/seed provenance differs')
        source_path = Path(recorded['source_datasets']['baseline']['path'])
        verify_dataset(receipt, source_path)
        rows, source = load_rows({'domain': domain, 'dataset': str(source_path), 'row_offset': 0, 'row_limit': 0})
        require(len(rows) == len(values) == 128 and source == receipt['identity']['source'], 'fixed reference source differs')
        histograms = reference_histograms(domain)
        baseline_cells = cell_histogram(domain, rows)
        baseline_support, reference_support = Counter(), Counter()
        for cell, count in baseline_cells.items():
            baseline_support[cell[0]] += count
        for cell, count in histograms['eval'].items():
            reference_support[cell[0]] += count
        require(baseline_support == reference_support, 'fixed reference marginal support differs from original EVAL reference')
        metrics = {metric: sum(row[metric] for row in values) / len(values) for metric in TOLERANCES}
        require(all(metrics[metric] == recorded['means']['baseline'][metric] for metric in TOLERANCES),
                'frozen reference metrics differ from completed report')
        if domain in RETAINED:
            require(recorded['observed_approximate_match'] is True
                    and recorded['within_tolerance'] == {'pass1': True, 'pass8': True}, 'retained domain did not pass')
        targets[domain] = {'receipt': receipt, 'rows': rows, 'metrics': metrics,
                           'provenance': {**metadata, 'identity_sha256': receipt['identity_sha256'],
                                          'source': source, 'draw_labels': list(REFERENCE_LABELS),
                                          'historical_level1_cells': serialize_cells(baseline_cells),
                                          'level2_eval_cells': serialize_cells(histograms['eval']),
                                          'marginal_support_matches_level2_eval': True,
                                          'joint_cells_match_level2_eval': baseline_cells == histograms['eval'],
                                          'historical_report_sha256': OLD_REPORT_SHA,
                                          'original_grader_replay_sha256': OLD_REPLAY_SHA}}
    verify_pins(inherited['files_sha256'], inherited['directory_files'])
    return targets


def old_rng_inventory():
    """All old DEV and CONF schedules remain excluded, including failed ranges."""
    inherited = authenticate_inherited()
    from audit_modebench_level3_python_v6_independent_match import authenticated_development_sources
    from evaluate_modebench_level3_independent import load_rows, schedule_record, validate_task
    development = authenticated_development_sources()
    blocks = set(development['blocks'])
    require(len(development['manifest']) == 37 and len(blocks) == 18688, 'old development inventory differs')
    manifests = [{'phase': 'historical_development', **item} for item in development['manifest']]
    plan = read(OLD_CAMPAIGN / 'confirmation_python_v6/plan.json')
    sealed_sources = {item['name']: item for item in inherited['seal']['sources']}
    for job in plan['jobs']:
        task = read(job['tasks'])[0]
        validate_task(task, True)
        rows, source = load_rows(task)
        schedule = schedule_record(job['domain'], rows, task['seeds'])
        current = {base for row in schedule['request_seeds'] for base in row}
        expected = sealed_sources[job['name']]
        require(source == expected['identity'] and sha(schedule) == expected['seed_schedule_sha256']
                and len(current) == 512 and task['seeds'] == list(REFERENCE_LABELS)
                and not current & blocks and all(base % 8 == 0 for base in current), 'old confirmation RNG source differs')
        blocks |= current
        manifests.append({'phase': 'historical_confirmation', **expected, 'tasks': job['tasks'],
                          'tasks_sha256': digest(job['tasks']), 'seeds': task['seeds']})
    require(len(manifests) == 47 and len(blocks) == 23808, 'complete historical RNG inventory differs')
    return {'manifests': manifests, 'blocks': blocks, 'files_sha256': inherited['files_sha256'],
            'directory_files': inherited['directory_files']}


def validate_registration(path, expected_sha256):
    """Caller-supplied pin prevents registration/source hash cycles."""
    path = Path(path).resolve()
    require(path == REGISTRATION and isinstance(expected_sha256, str) and len(expected_sha256) == 64
            and digest(path) == expected_sha256, 'explicit canonical v3 registration pin required')
    record = read(path)
    require(record.get('schema') == REGISTRATION_SCHEMA and record.get('contract') == contract()
            and record.get('inherited_confirmation') == inherited_metadata()
            and record.get('benchmark_targets') == benchmark_metadata(), 'prospective v3 registration contract differs')
    require(set(record.get('candidate_revisions', {})) == set(REVISED), 'both new candidate revisions required')
    require(str(path) not in record['files_sha256'], 'registration cannot pin itself')
    require(record['files_sha256'].get(str(Path(__file__).resolve())) == digest(__file__), 'common implementation not registered')
    for domain, revision in zip(REVISED, ('graph_v8', 'python_v7')):
        descriptor = record['candidate_revisions'][domain]
        require(descriptor.get('name') == revision and descriptor.get('domain') == domain
                and descriptor.get('pool_root') == str(POOL_ROOTS[domain])
                and descriptor.get('rows_per_tier') == contract()['calibration_rows_per_tier'][domain],
                'canonical candidate revision/pool/quota differs')
        for kind, relative in (
                ('generator', f'ops/exp_scaling/modebench_level3_{revision}.py'),
                ('materializer', f'ops/exp_scaling/materialize_modebench_level3_{revision}.py'),
                ('tests', f'tests/test_modebench_level3_{revision}.py')):
            source = str(ROOT / relative)
            require(descriptor.get(kind + '_path') == source
                    and descriptor.get(kind + '_sha256') == digest(source)
                    and record['files_sha256'].get(source) == digest(source),
                    'candidate implementation/test bytes not registered')
        require(descriptor.get('development_receipts') == {
                    str(tier): str(RESULTS / f'calibration_3b_{revision}_d{tier}.json') for tier in range(4)},
                'candidate DEV receipt namespace differs')
        for key, seed_kind in (('development_seed_base', 'development'), ('capacity_seed_base', 'capacity'),
                               ('final_train_seed_base', 'train'), ('final_eval_seed_base', 'eval')):
            require(descriptor.get(key) == SEEDS[domain][seed_kind], 'candidate generation seed differs')
        snapshot = descriptor.get('source_snapshot', {})
        require(isinstance(snapshot.get('files_sha256'), dict) and bool(snapshot['files_sha256'])
                and isinstance(snapshot.get('directory_files'), dict)
                and all(record['files_sha256'].get(source) == expected
                        for source, expected in snapshot['files_sha256'].items())
                and all(record['directory_files'].get(source) == expected
                        for source, expected in snapshot['directory_files'].items()),
                'candidate source snapshot omitted from registration')
    inherited = authenticate_inherited()
    require(all(record['files_sha256'].get(source) == expected for source, expected in inherited['files_sha256'].items()),
            'v3 registration omits inherited evidence')
    require(all(record['directory_files'].get(source) == expected for source, expected in inherited['directory_files'].items()),
            'v3 registration omits inherited inventories')
    verify_pins(record['files_sha256'], record['directory_files'])
    require(digest(path) == expected_sha256, 'registration changed during authentication')
    return record


def validate_new_candidate_receipt(path, domain, labels=DEV_LABELS):
    """New candidate DEV outcomes; no historical candidate CONF can enter fit."""
    require(domain in REVISED and tuple(labels) == DEV_LABELS, 'registered new candidate DEV labels required')
    from fit_modebench_level3 import load_receipt
    from evaluate_modebench_level3_independent import validate_seed_receipt
    from audit_modebench_level3_python_v6_independent_match import validate_receipt
    before = digest(path)
    receipt, rows, scores = load_receipt(path, domain, 'level3', '3b')
    require(receipt['identity']['seeds'] == list(DEV_LABELS), 'candidate uses another round of draws')
    validate_seed_receipt(receipt, rows=rows)
    validate_receipt(receipt, domain=domain, role='candidate', development=True)
    require(digest(path) == before, 'candidate receipt changed during verification')
    return receipt, rows, scores
