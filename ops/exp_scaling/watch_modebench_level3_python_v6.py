#!/usr/bin/env python3
"""Fit the four sealed Python v6 development cells once under one domain lock."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / 'var/artifacts/modebench_level3_v2/python_v6'
SEAL = CAMPAIGN / 'implementation_seal.json'
LAUNCHER = CAMPAIGN / 'launch.py'
BASELINE = ROOT / 'var/results/modebench_level3_v2/calibration_05b_python_factors.json'
SCORES = [ROOT / f'var/results/modebench_level3_v2/calibration_3b_python_v6_d{i}.json' for i in range(4)]
RECEIPTS = [BASELINE, *SCORES]
RECIPE = CAMPAIGN / 'recipe.json'
INTENT = CAMPAIGN / 'development_fit_intent.json'
RESULT = CAMPAIGN / 'development_fit_report.json'
FAILURE = CAMPAIGN / 'development_fit_failure.json'
EVENTS = CAMPAIGN / 'development_fit_events.jsonl'
LOCK = CAMPAIGN / '.python_v6_fit.lock'
for directory in ('ops', 'ops/exp_scaling', 'src'):
    sys.path.insert(0, str(ROOT / directory))
import fit_modebench_level3_python_v6_independent as adapter
from evaluate_modebench_level3 import atomic_new


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify(seal_sha):
    if not isinstance(seal_sha, str) or not re.fullmatch('[0-9a-f]{64}', seal_sha) or digest(SEAL) != seal_sha:
        raise ValueError('explicit Python v6 implementation seal hash differs')
    sealed = read(SEAL)
    for path in (Path(__file__).resolve(), LAUNCHER):
        if sealed.get('files_sha256', {}).get(str(path)) != digest(path):
            raise ValueError(f'Python v6 watcher/launcher is not bound to the supplied seal: {path}')
    # Authenticate launcher bytes before importing it, then invoke its complete
    # scientific/execution-source and checkpoint verifier.
    spec = importlib.util.spec_from_file_location('python_v6_sealed_fit_launcher', LAUNCHER)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    return launcher.authenticate_saved_seal(seal_sha)


def event(status, **fields):
    record = {'created_at': now(), 'status': status, **fields}
    encoded = json.dumps(record, sort_keys=True, allow_nan=False)
    with EVENTS.open('a') as handle:
        handle.write(encoded + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    print(encoded, flush=True)


def unclaimed():
    for path in (INTENT, RESULT, RECIPE, FAILURE):
        if path.exists():
            raise FileExistsError(f'Python v6 fit evidence already exists; inspect without retry: {path}')


def status():
    return {'domain': 'python_factors', 'candidate_revision': 'python_v6',
            'missing_receipts': [str(path) for path in RECEIPTS if not path.is_file()],
            'recipe_exists': RECIPE.exists(), 'intent_exists': INTENT.exists(),
            'result_exists': RESULT.exists(), 'failure_exists': FAILURE.exists()}


def _fit_locked(seal_sha):
    unclaimed()
    verify(seal_sha)
    receipt_pins = {str(path): digest(path) for path in RECEIPTS}
    if any(read(path).get('status') != 'complete' for path in RECEIPTS):
        raise ValueError('all five registered Python receipts must be complete')
    intent = {'schema': 'modebench_level3_python_v6_fit_intent_v1', 'created_at': now(),
              'domain': 'python_factors', 'candidate_revision': 'python_v6',
              'seal_path': str(SEAL), 'seal_sha256': seal_sha,
              'watcher_sha256': digest(__file__), 'adapter_sha256': digest(adapter.__file__),
              'receipt_sha256': receipt_pins, 'recipe_path': str(RECIPE), 'result_path': str(RESULT),
              'fit_rule': 'unchanged_full_pool_forecast_then_one_hash_fixed_selected_set',
              'confirmation_outcomes_used': False}
    atomic_new(INTENT, intent)
    event('fit_started', intent_sha256=digest(INTENT))
    try:
        recipe = adapter.fit_recipe(BASELINE, SCORES, 'python_factors')
        verify(seal_sha)
        if receipt_pins != {str(path): digest(path) for path in RECEIPTS}:
            raise ValueError('Python v6 development receipt changed during fitting')
        if type(recipe.get('development_fit_pass')) is not bool:
            raise ValueError('fitter did not return an explicit development gate result')
        atomic_new(RECIPE, recipe)
        result = {'schema': 'modebench_level3_python_v6_fit_result_v1', 'created_at': now(),
                  'domain': 'python_factors', 'candidate_revision': 'python_v6',
                  'status': 'development_fit_pass' if recipe['development_fit_pass'] else 'needs_calibration_revision',
                  'intent_path': str(INTENT), 'intent_sha256': digest(INTENT),
                  'seal_sha256': seal_sha, 'watcher_sha256': digest(__file__),
                  'receipt_sha256': receipt_pins, 'recipe_path': str(RECIPE), 'recipe_sha256': digest(RECIPE),
                  'confirmation_outcomes_used': False,
                  **{key: recipe[key] for key in ('development_fit_pass', 'decision', 'weights', 'development')}}
        atomic_new(RESULT, result)
        event(result['status'], recipe_sha256=result['recipe_sha256'], result_sha256=digest(RESULT))
        return result
    except BaseException as error:
        failure = {'schema': 'modebench_level3_python_v6_fit_failure_v1', 'created_at': now(),
                   'intent_sha256': digest(INTENT), 'seal_sha256': seal_sha,
                   'error_type': type(error).__name__, 'error': str(error),
                   'recipe_exists': RECIPE.exists(), 'result_exists': RESULT.exists(),
                   'action': 'inspect_preserved_evidence_no_automatic_retry'}
        if not FAILURE.exists():
            atomic_new(FAILURE, failure)
        event('fit_failed_for_review', **failure)
        raise


def run(seal_sha, watch=False):
    verify(seal_sha)
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        unclaimed()
        while True:
            current = status()
            if not current['missing_receipts']:
                return _fit_locked(seal_sha)
            event('waiting_for_python_v6_receipts', **current)
            if not watch:
                return {'status': 'waiting_for_python_v6_receipts', **current}
            time.sleep(30)
            verify(seal_sha)
            unclaimed()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--watch', action='store_true')
    actions.add_argument('--fit-once', action='store_true')
    parser.add_argument('--seal-sha256')
    args = parser.parse_args()
    if not (args.watch or args.fit_once):
        print(json.dumps({'status': 'read_only', **status()}, sort_keys=True), flush=True)
        return
    result = run(args.seal_sha256, watch=args.watch)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
