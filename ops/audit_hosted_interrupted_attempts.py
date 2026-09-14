"""Account for registered interrupted request starts without inventing responses.

The original append-only event log and all raw receipts remain untouched. Only
explicitly frozen, timestamp-bound orphan starts can lack a response receipt.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def timestamp(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError('Attempt timestamps must have a timezone')
    return result


def identity(value):
    return value.get('group_id') or value.get('sample_id'), value['attempt']


def validate_attempt_accounting(events, raw_receipts, registration):
    if registration.get('schema') != 'hosted-interrupted-attempt-registration-v1':
        raise ValueError('Unsupported interruption registration')
    resume_index = registration['resume_event_index']
    if not isinstance(resume_index, int) or not 0 <= resume_index < len(events):
        raise ValueError('Missing registered resume event')
    resume = events[resume_index]
    if resume != registration['resume_event'] or resume.get('event') != 'session_started':
        raise ValueError('Registered resume event changed')
    if sha(events[:resume_index]) != registration['pre_resume_events_sha256']:
        raise ValueError('Pre-resume event history changed')
    resume_time = timestamp(resume['at_utc'])
    orphan_indices = set()
    for item in registration['orphan_starts']:
        index = item['event_index']
        if index in orphan_indices or not isinstance(index, int) or not 0 <= index < resume_index:
            raise ValueError('Invalid or duplicate registered orphan index')
        event = events[index]
        if event != item['event'] or sha(event) != item['event_sha256'] or event.get('event') != 'request_started':
            raise ValueError('Registered orphan start changed')
        if timestamp(event['at_utc']) >= resume_time:
            raise ValueError('Orphan was not started before the registered resume')
        if item.get('outcome') != 'unknown' or item.get('usage') != 'unknown' or item.get('possibly_billed') is not True:
            raise ValueError('Interrupted outcomes and usage must remain unknown')
        orphan_indices.add(index)
    if len(orphan_indices) != registration['orphan_attempt_count'] or not orphan_indices:
        raise ValueError('Unexpected registered orphan count')
    retained = {}
    all_counts = Counter()
    orphan_keys = set()
    for index, event in enumerate(events):
        if event.get('event') != 'request_started':
            continue
        key = identity(event)
        all_counts[key] += 1
        if index in orphan_indices:
            if key in orphan_keys:
                raise ValueError('Multiple registered orphans share a physical identity')
            orphan_keys.add(key)
            continue
        if key in retained:
            raise ValueError('Unregistered duplicate physical request-start identity')
        retained[key] = event
    if set(retained) != set(raw_receipts):
        raise ValueError('Unregistered missing receipt or receipt without a request start')
    for key in orphan_keys:
        if all_counts[key] != 2 or key not in retained:
            raise ValueError('Each registered orphan requires exactly one resumed request')
        if timestamp(retained[key]['at_utc']) < resume_time:
            raise ValueError('Replacement request predates the registered resume')
    for key, raw in raw_receipts.items():
        event = retained[key]
        if identity(raw) != key or raw['request_sha256'] != event['request_sha256']:
            raise ValueError('Raw request identity differs from retained start')
        if raw.get('sample_ids') != event.get('sample_ids'):
            raise ValueError('Raw sample inventory differs from retained start')
        start, received = timestamp(raw['started_at_utc']), timestamp(raw['received_at_utc'])
        if start < timestamp(event['at_utc']) or received < start:
            raise ValueError('Raw timestamps do not bind to retained request start')
        if key in orphan_keys and start < resume_time:
            raise ValueError('Raw receipt belongs to the interrupted request, not its replacement')
    return {
        'registered_interrupted_attempts_with_unknown_outcome': len(orphan_indices),
        'registered_interrupted_attempts_with_unknown_usage': len(orphan_indices),
        'saved_raw_attempt_receipts': len(raw_receipts),
        'logged_physical_request_starts': sum(all_counts.values()),
        'accounted_physical_attempts': len(raw_receipts) + len(orphan_indices),
        'possibly_billed_unknown_attempts': len(orphan_indices),
        'note': 'Unknown interrupted outcomes are not model samples or zero-cost attempts; original events and raw receipts are unchanged.',
    }
