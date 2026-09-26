#!/usr/bin/env python3
"""Render an authenticated v3 fixed-reference summary; only --publish writes it."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
for directory in ('ops', 'ops/exp_scaling', 'src'):
    if str(ROOT / directory) not in sys.path:
        sys.path.insert(0, str(ROOT / directory))
import modebench_level3_v3_common as common

SOURCE = Path(__file__).resolve()
TEST = ROOT / 'tests/test_summarize_modebench_level3_v3.py'
REPORT = common.CAMPAIGN / 'confirmation/confirmation_report.json'
OUTPUT = ROOT / 'artifacts/modebench_level3_matched_v3_summary.md'
HERE = common.CAMPAIGN / 'summary'
INTENT = HERE / 'publication_intent.json'
PUBLICATION = HERE / 'publication.json'
NAMES = {'countdown': 'Countdown', 'graph_coloring': 'Graph Coloring',
         'python_factors': 'Python Factors', 'mathir': 'MathIR', 'pantry': 'Pantry'}
METRICS = ('pass1', 'pass8')
require = common.require


def link(label, path):
    return f'[{label}](<{Path(path).resolve()}>)'


def pair(values):
    return ' / '.join(str(values[metric]) for metric in METRICS)


def validate_display_contract(report):
    require(report['all_five_domains_complete'] is True and report['errors'] == {}
            and report['missing_domains'] == [] and set(report['domains']) == set(common.DOMAINS)
            and report['tolerances'] == common.TOLERANCES, 'complete all-five fixed-tolerance report required')
    for domain, record in report['domains'].items():
        baseline, candidate = record['means']['baseline'], record['means']['candidate']
        require(all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
                    for values in (baseline, candidate) for value in values.values()), 'finite pass rates required')
        differences = {metric: candidate[metric] - baseline[metric] for metric in METRICS}
        gates = {metric: abs(differences[metric]) <= common.TOLERANCES[metric] for metric in METRICS}
        require(record['differences'] == differences and record['within_tolerance'] == gates
                and record['observed_approximate_match'] is all(gates.values()), 'displayed numerical decision differs')
        fresh = domain in common.REVISED
        require(record['fresh_candidate_confirmation'] is fresh
                and record['candidate_confirmation_round'] == (2 if fresh else 1), 'candidate evidence origin differs')
    passed = all(record['observed_approximate_match'] for record in report['domains'].values())
    require(report['confirmation_match_verified'] is passed
            and report['status'] == ('matched_fixed_reference' if passed else 'outside_fixed_reference_tolerance'),
            'aggregate status differs from actual domain gates')
    boundary = report['information_boundary']
    require(boundary['reference_semantics'] == 'fixed_measured_level1_benchmark'
            and boundary['adaptive_confirmation_round'] == 2
            and boundary['historical_level1_confirmation_used_as_fixed_reference'] is True
            and boundary['historical_level3_confirmation_used_for_mixture_fitting'] is False
            and boundary['new_candidate_fitting_uses_development_outcomes_only'] is True
            and boundary['all_five_fresh_same_round'] is False
            and boundary['statistical_equivalence_claimed'] is False
            and boundary['treatment_training_started'] is False, 'adaptive fixed-reference information boundary differs')
    require(report['dataset']['dataset_root'] == str(common.DATASET)
            and report['dataset']['split_sizes'] == common.SPLITS
            and report['dataset']['retained_domains'] == list(common.RETAINED)
            and report['dataset']['fresh_candidate_domains'] == list(common.REVISED), 'final dataset/origin metadata differs')
    require(report['attempts'] == 8192 and report['new_confirmation_jobs'] == 2
            and report['all_attempts_regraded_with_original_grader'] is True
            and report['every_attempt_including_failures_and_full_canonical_keys_compared'] is True,
            'complete original-grader confirmation evidence required')


def render(report, report_sha256, implementation_pins):
    validate_display_contract(report)
    passed = report['confirmation_match_verified']
    outcome = ('All five comparisons passed the fixed-reference tolerances.' if passed
               else 'The campaign did not meet all fixed-reference tolerances.')
    lines = [f'**{outcome}**', '',
        'The gates use absolute candidate-minus-target differences: pass@1 ≤ 0.04 and pass@8 ≤ 0.08. '
        'Rates and deltas below retain the recorded numeric precision; gates use the unrounded values.', '',
        '| Domain | Fixed L1 target p1 / p8 | L3 candidate p1 / p8 | Delta p1 / p8 | Gates p1 / p8 | L3 evidence |',
        '|---|---|---|---|---|---|']
    for domain in common.DOMAINS:
        record = report['domains'][domain]
        reference = record['fixed_reference']
        target = link(pair(record['means']['baseline']), reference['receipt_path'])
        candidate = link(pair(record['means']['candidate']), record['candidate_receipt']['path'])
        gates = ' / '.join('PASS' if record['within_tolerance'][metric] else 'FAIL' for metric in METRICS)
        origin = 'Fresh V3, round 2' if domain in common.REVISED else 'Retained V2, round 1'
        lines.append(f'| {NAMES[domain]} | {target} | {candidate} | {pair(record["differences"])} | {gates} | {origin} |')
    lines += ['', 'All five Level 1 targets are the fixed, completed historical V2 confirmation measurements. '
        'They were explicitly adopted as empirical benchmarks before the new candidate pools and model outcomes. '
        'They were not relabeled as development data or measured again in V3.', '',
        'Graph Coloring and Python Factors use fresh V3 candidate confirmation prompts and draws after fitting only new candidate development outcomes. '
        'The three retained domains keep their V2 split bytes and completed candidate results. '
        'This is adaptive second-round confirmation, not five fresh comparisons from one round. '
        'Historical candidate confirmation informed prospective generator design but did not rank the new mixtures. '
        'The fixed reference sampling uncertainty was not reestimated; passing these numerical gates is not a claim of statistical equivalence. '
        'No treatment training has started.', '',
        f'Dataset: {link(str(common.DATASET.relative_to(ROOT)), common.DATASET)}. '
        'Each domain has 384 train, 128 dev and 128 eval rows: 3,200 rows in total.', '',
        '| Domain | Train (384) | Dev (128) | Eval (128) | Origin |', '|---|---|---|---|---|']
    for domain in common.DOMAINS:
        paths = [link(domain + '/' + split, common.DATASET / domain / split) for split in common.SPLITS]
        origin = (link('Byte-identical V2 domain', common.OLD_DATASET / domain) if domain in common.RETAINED
                  else 'V3: new train/eval; fixed selected development rows')
        lines.append('| ' + ' | '.join([NAMES[domain], *paths, origin]) + ' |')
    development = report['completed_development_audit']
    dataset = report['dataset']
    lines += ['',
        f'The {link("completed V3 confirmation report", REPORT)} authenticates all five comparisons and the original grader replay '
        'of all 8,192 attempts from the two new confirmation jobs, including failures and full canonical keys. '
        f'The {link("completed V3 development audit", development["path"])} covers all 39,424 new development attempts. '
        f'The retained evidence is covered by the {link("completed V2 original-grader replay", common.OLD_REPLAY)} '
        'of all 40,960 historical confirmation attempts. This summary validates those saved proofs without repeating grading.', '',
        f'Provenance: {link("scientific registration", report["registration_path"])}; '
        f'{link("fixed-reference amendment", common.CAMPAIGN / "fixed_reference_amendment.json")}; '
        f'{link("dataset identity", dataset["path"])}; '
        f'{link("frozen recipe bundle", dataset["frozen_recipe_bundle_path"])}; '
        f'{link("summary publication record", PUBLICATION)}.', '',
        f'Canonical report SHA-256: `{report_sha256}`. '
        f'Summary helper SHA-256: `{implementation_pins[str(SOURCE)]}`; '
        f'tests SHA-256: `{implementation_pins[str(TEST)]}`.', '']
    return '\n'.join(lines)


def build_summary(expected_report_sha256=None):
    before = common.digest(REPORT)
    require(expected_report_sha256 is None or before == expected_report_sha256, 'explicit canonical report pin differs')
    implementation = {str(path): common.digest(path) for path in (SOURCE, TEST)}
    auditor = importlib.import_module('audit_modebench_level3_v3')
    evidence = auditor.validate_confirmation_report(path=REPORT)
    require(evidence['path'] == str(REPORT) and evidence['sha256'] == before
            and common.digest(REPORT) == before, 'canonical report changed during authentication')
    text = render(evidence['report'], before, implementation)
    files = common.merge_pins(evidence['files_sha256'], implementation, {str(REPORT): before})
    common.verify_pins(files, evidence['directory_files'])
    return text, {'status': evidence['status'], 'report_path': str(REPORT), 'report_sha256': before,
                  'source_path': str(SOURCE), 'source_sha256': implementation[str(SOURCE)],
                  'tests_path': str(TEST), 'tests_sha256': implementation[str(TEST)],
                  'files_sha256': files, 'directory_files': evidence['directory_files']}


def atomic_new_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(text); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def publish(expected_report_sha256):
    require(expected_report_sha256, 'publication requires an explicit completed report SHA-256')
    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE / '.publication.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not any(path.exists() for path in (OUTPUT, INTENT, PUBLICATION)), 'existing summary publication or intent forbids retry')
        text, evidence = build_summary(expected_report_sha256)
        intent = {'schema': 'modebench_level3_v3_summary_publication_intent_v1',
                  'created_at': datetime.now(timezone.utc).isoformat(), 'output': str(OUTPUT),
                  'summary_sha256': hashlib.sha256(text.encode()).hexdigest(), **evidence}
        common.atomic_new(INTENT, intent)
        common.verify_pins(evidence['files_sha256'], evidence['directory_files'])
        atomic_new_text(OUTPUT, text)
        common.verify_pins(evidence['files_sha256'], evidence['directory_files'])
        require(common.digest(OUTPUT) == intent['summary_sha256'], 'published summary bytes differ')
        record = {'schema': 'modebench_level3_v3_summary_publication_v1',
                  'created_at': datetime.now(timezone.utc).isoformat(), 'output': str(OUTPUT),
                  'summary_sha256': common.digest(OUTPUT), 'intent_path': str(INTENT),
                  'intent_sha256': common.digest(INTENT), **evidence}
        common.atomic_new(PUBLICATION, record)
        return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--report-sha256')
    args = parser.parse_args(argv)
    if args.publish:
        result = publish(args.report_sha256)
        print(json.dumps({'status': result['status'], 'summary': result['output'],
                          'sha256': result['summary_sha256'], 'publication_record': str(PUBLICATION)}))
    elif not REPORT.is_file():
        print(json.dumps({'status': 'waiting_for_completed_confirmation_report', 'summary_published': False}))
    else:
        text, _ = build_summary(args.report_sha256)
        print(text, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
