#!/usr/bin/env python3
"""Reproduce independent synthetic validation; never read experiment outcomes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = ROOT / 'ops/exp_scaling/analyze_paper_conditional_concentration.py'
TESTS = ROOT / 'tests/test_paper_collision_statistics.py'
RECEIPT = Path(__file__).with_suffix('.json')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = (ANALYSIS, TESTS, Path(__file__).resolve())
    before = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    with tempfile.TemporaryDirectory(prefix='paper-collision-synthetic-') as temporary:
        junit = Path(temporary) / 'pytest.xml'
        command = [sys.executable, '-m', 'pytest', '-q', str(TESTS), f'--junitxml={junit}']
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        suites = list(ET.parse(junit).getroot().iter('testsuite')) if junit.exists() else []
        counts = {name: sum(int(s.attrib.get(name, 0)) for s in suites)
                  for name in ('tests', 'failures', 'errors', 'skipped')}
    after = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    passed = completed.returncode == 0 and before == after
    receipt = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'passed' if passed else 'failed_or_source_changed',
        'scope': 'synthetic statistics and structural API checks only; no experiment outputs read',
        'command': [sys.executable, '-m', 'pytest', '-q', str(TESTS.relative_to(ROOT))],
        'returncode': completed.returncode,
        'counts': counts,
        'sha256_before': before,
        'sha256_after': after,
        'stdout': completed.stdout,
        'stderr': completed.stderr,
        'exact_reference_values_verified_by_tests': {
            'correct_count_conditional_unbiasedness': '5/9 for P=3/5, q=(1/3,2/3), budgets 2 through 6',
            'shared_rng_common_eligibility_spurious_delta': '1/8 despite C_A=C_B=1/2',
            'independent_endpoint_version_delta': '0',
            'cross_prompt_rng_random_eligible_mean_delta': '1/16 despite promptwise equal C between endpoints',
            'cross_prompt_rng_fixed_denominator_score': '0 in the same counterexample',
            'nominal_distinct_streams_from_four_n8_groups': 11,
            'saved_occurrences': 32,
            'repeated_occurrences': 21,
            'duplicate_stream_unordered_pairs': 38,
            'naive_32_expected_collision_fair_binary': '267/496',
            'distinct_11_expected_collision_fair_binary': '1/2',
            'old_first_two_vs_last_two_draw_group_overlap': 7,
            'categorical_collision_derivative': '2*(sum(q^3)-sum(q^2)^2) = 2*Var_{c~q}(q_c) >= 0',
        },
        'limitations': [
            'Distinct nominal stream IDs do not certify statistical independence under arbitrary runtime effects.',
            'The same stream IDs recur across prompts, so iid prompt uncertainty is not certified.',
            'The random common-eligible mean and two-orientation complete-case average remain descriptive.',
            'Five-seed t intervals describe nominal paired seed variability; they are not full sampling uncertainty.',
            'Passing tests does not establish empirical collapse or replay benefit.',
        ],
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'status': receipt['status'], 'counts': counts, 'receipt': str(RECEIPT.relative_to(ROOT))}, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
