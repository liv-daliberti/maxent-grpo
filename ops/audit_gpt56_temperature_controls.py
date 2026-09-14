#!/usr/bin/env python3
"""Authenticate returned temperature and reasoning effort for each curve draw."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path

from audit_hosted_modebench_completion import atomic, file_sha, load_inventory, read_jsonl, validate_native_records


def audit(directory):
    inventory = load_inventory(directory, expected_samples=960)
    directory, manifest = inventory['directory'], inventory['manifest']
    if manifest.get('experiment_condition') != 'gpt56_none_temperature_curve_v1' or manifest.get('reasoning_effort') != 'none':
        raise ValueError('Expected a separately registered GPT no-reasoning temperature curve')
    temperature = manifest['temperature']
    records = read_jsonl(directory / 'samples.jsonl')
    if len(records) != 960 or len({r['sample_id'] for r in records}) != 960:
        raise ValueError('Incomplete or duplicated response cohort')
    raw = validate_native_records(inventory, records)
    temperatures, efforts, top_ps, models = Counter(), Counter(), Counter(), Counter()
    for record in records:
        request = inventory['expected'][record['sample_id']]['request']
        if request.get('temperature') != temperature or request.get('reasoning', {}).get('effort') != 'none':
            raise ValueError('Frozen request controls differ from registered condition')
        body = raw[record['raw_receipt']]['response']
        if body.get('temperature') != temperature or body.get('reasoning', {}).get('effort') != 'none':
            raise ValueError('Native response does not echo the requested temperature and reasoning effort')
        temperatures[str(body['temperature'])] += 1
        efforts[body['reasoning']['effort']] += 1
        top_ps[str(body.get('top_p'))] += 1
        models[str(body.get('model'))] += 1
    result = {'schema': 'gpt56-temperature-native-control-audit-v1', 'status': 'pass',
              'audited_at_utc': datetime.now(timezone.utc).isoformat(), 'api_calls': 0,
              'auditor_sha256': file_sha(Path(__file__)), 'responses': len(records),
              'manifest_sha256': file_sha(directory / 'manifest.json'),
              'samples_sha256': file_sha(directory / 'samples.jsonl'),
              'requested_temperature': temperature, 'requested_reasoning_effort': 'none',
              'returned_temperature_counts': dict(temperatures), 'returned_reasoning_effort_counts': dict(efforts),
              'returned_top_p_counts': dict(top_ps), 'returned_model_counts': dict(models),
              'interpretation': 'Native response echoes authenticate the exposed controls; none is a different reasoning configuration from the original medium cohort.'}
    atomic(directory / 'native_control_audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    print(json.dumps(audit(parser.parse_args().directory), indent=2))
