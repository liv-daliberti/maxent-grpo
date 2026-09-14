#!/usr/bin/env python3
"""Read-only display compatibility for the recorded array31243495 throttle2.

The original classifier charges every unfinished task irrespective of throttle.
Accepting the already applied %2 suffix changes no task identity or reservation.
All original source, array, systems and training gates remain active. This
context is process-local and must be used by a serialized admission caller.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e122_storage_with_modebench_array_20260911 as original

ROOT = original.ROOT
ORIGINAL_SHA256 = 'ffe7063c1a9e729ad4b0dc744e2dfff791c4bae0bc8ec1f5b1de37eca98cbbd0'
EVIDENCE = {
    'var/artifacts/modebench_scale_schedule_v2/amendment.json': '613cd33537b4777ffb7547730143c279b94c58a19e1adf058b8fe66b0690a970',
    'var/artifacts/modebench_scale_schedule_v2/action_5_result.json': '4bad004cd378f268bacafdda46d798be2dc90a41d0ff256af2391927449e6867',
}


def verify_amendment():
    original.require(hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest() == ORIGINAL_SHA256,
                     'original array classifier changed')
    records = []
    for name, expected in EVIDENCE.items():
        path = ROOT / name
        original.require(original.original.digest(path) == expected, 'throttle2 evidence changed')
        records.append(original.bounded_json(path))
    plan, result = records
    original.require(plan['array_job_id'] == original.ARRAY_ID and plan['array_throttle_after'] == 2
        and result['command'] == ['scontrol', 'update', 'JobId=31243495', 'ArrayTaskThrottle=2']
        and result['returncode'] == 0 and result['status'] == 'applied_observed'
        and result['timed_out'] is False, 'throttle2 amendment not positively applied')


@contextmanager
def compatible_array_parser():
    verify_amendment()
    previous = original.array_indices
    def amended(text):
        if isinstance(text, str) and text.endswith('%2'):
            return previous(text[:-2])
        return previous(text)
    original.array_indices = amended
    try:
        yield
    finally:
        original.array_indices = previous


def storage_report(include_held_job_ids=()):
    try:
        with compatible_array_parser():
            result = original.storage_report(include_held_job_ids=include_held_job_ids)
        result['throttle_display_adapter'] = str(Path(__file__).resolve())
        result['throttle_amendment_pins'] = EVIDENCE.copy()
        return result
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {'allowed': False, 'status': 'rejected', 'blocked_reason': 'unresolved_storage_safety',
                'errors': [type(error).__name__ + ': ' + str(error)]}


if __name__ == '__main__':
    print(json.dumps(storage_report(), indent=2))
