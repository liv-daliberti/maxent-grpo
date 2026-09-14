#!/usr/bin/env python3
"""Publish a human summary only from the completed canonical v6 confirmation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / 'var/artifacts/modebench_level3_v2'
CONFIRMATION = CAMPAIGN / 'confirmation_python_v6'
CONTINUATION = CAMPAIGN / 'continuation_python_v6'
REPORT = CONFIRMATION / 'confirmation_report.json'
DATASET = ROOT / 'var/data/modebench_level3_matched_v2'
OUTPUT = ROOT / 'artifacts/modebench_level3_matched_v2_summary.md'
DRIVER = ROOT / 'ops/exp_scaling/continue_modebench_level3_python_v6.py'
DRIVER_SHA = '4dc9c7d5473b178db1a333423febc3f5dcc121c27b8080eda0059cd1ce169322'
CONTINUATION_SHA = '70f7cc8451fd253ae2c1cfecf6dc8ed72e198ee0b791035d9436484a00c171a0'
DOMAINS = ('countdown', 'graph_coloring', 'mathir', 'python_factors', 'pantry')
LABELS = dict(zip(DOMAINS, ('Countdown', 'Graph Coloring', 'MathIR', 'Python Factors', 'Pantry')))
TOLERANCES = {'pass1': .04, 'pass8': .08}
SPLITS = {'train': 384, 'dev': 128, 'eval': 128}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def validate_report(report):
    require(report.get('schema') == 'modebench-level3-independent-observed-match-audit-v2'
            and report.get('phase') == 'confirmation', 'a canonical confirmation report is required')
    require(report.get('status') in ('observed_approximate_match', 'outside_match_tolerance')
            and report.get('all_five_domains_complete') is True
            and report.get('errors') == {} and report.get('missing_domains') == []
            and set(report.get('domains', {})) == set(DOMAINS), 'complete valid results for all five domains are required')
    criteria = report['criteria']
    expected = {'absolute_pass1_difference_at_most': .04, 'absolute_pass8_difference_at_most': .08,
                'distinct8_is_diagnostic_only': True, 'statistical_equivalence_claimed': False,
                'confirmation_rows_per_domain_per_model': 128, 'confirmation_draws_per_prompt': 4,
                'samples_per_draw': 8, 'all_five_domains_required': True,
                'expected_confirmation_seeds': [6329000, 6329001, 6329002, 6329003]}
    require(criteria == expected and all(type(criteria[k]) is type(v) for k, v in expected.items()),
            'registered confirmation criteria differ')
    validation = report['evidence_validation']
    require(all(validation.get(key) is True for key in (
        'metrics_recomputed_from_attempts', 'source_rows_verified_when_dataset_supplied',
        'independent_seed_schedules_recomputed', 'legacy_overlapping_seed_evidence_rejected',
        'level1_controls_reused_with_corrected_sampling', 'fresh_heldout_claim_applies_to_level3_only',
        'frozen_recipe_settings_required', 'baseline_source_pinned_to_level1_confirmation')),
        'confirmation evidence or fixed-control limitation is missing')
    require(isinstance(report.get('prospective_confirmation_seal'), dict), 'prospective confirmation seal evidence required')
    matches = []
    for domain in DOMAINS:
        row = report['domains'][domain]
        require(all(type(row.get(key)) is int and row[key] == value for key, value in (
            ('baseline_rows', 128), ('candidate_rows', 128), ('draws_per_prompt', 4))),
            domain + ': full confirmation sampling is required')
        require(set(row['within_tolerance']) == set(TOLERANCES), domain + ': both gates are required')
        for metric, tolerance in TOLERANCES.items():
            baseline, candidate = (row['means'][role][metric] for role in ('baseline', 'candidate'))
            delta = row['candidate_minus_baseline'][metric]
            require(all(type(v) in (int, float) and math.isfinite(v) for v in (baseline, candidate, delta))
                    and 0 <= baseline <= 1 and 0 <= candidate <= 1
                    and math.isclose(candidate - baseline, delta, rel_tol=0, abs_tol=1e-12),
                    domain + ': inconsistent observed confirmation metrics')
            gate = abs(delta) <= tolerance + 1e-12
            require(row['within_tolerance'][metric] is gate, domain + ': inconsistent observed gate')
        matched = all(row['within_tolerance'].values())
        require(row['observed_approximate_match'] is matched, domain + ': inconsistent domain match decision')
        matches.append(matched)
    matched = all(matches)
    require(report.get('confirmation_match_verified') is matched
            and report.get('all_five_observed_approximate_match') is matched
            and report['status'] == ('observed_approximate_match' if matched else 'outside_match_tolerance'),
            'overall confirmation decision disagrees with observed gates')
    return matched


def authenticated_inputs():
    # Check absence before importing any scientific modules. No legacy fallback.
    require(REPORT.is_file(), 'canonical confirmation report is absent; no summary can be published')
    report_hash = digest(REPORT)
    report = read(REPORT)
    validate_report(report)
    require(digest(DRIVER) == DRIVER_SHA, 'reviewed continuation implementation changed')
    spec = importlib.util.spec_from_file_location('summary_completed_continuation', DRIVER)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver.verify_seal(CONTINUATION_SHA)
    phases = driver.Driver(CONTINUATION_SHA).journal()  # Read-only authentication.
    require(set(phases) == set(driver.PHASES), 'all registered continuation phases must be complete')
    terminal = read(CONTINUATION / 'result.json')
    require(terminal['seal_sha256'] == CONTINUATION_SHA and terminal['report'] == str(REPORT)
            and terminal['report_sha256'] == report_hash and terminal['status'] == report['status']
            and terminal['confirmation_match_verified'] is report['confirmation_match_verified'],
            'report is not bound to the completed continuation')
    require(phases['confirmation_audit']['files_sha256'].get(str(REPORT)) == report_hash,
            'report is not bound to the completed audit phase')
    command = read(CONTINUATION / 'confirmation_audit_command.json')['command']
    require(command == list(map(str, [driver.PYTHON, driver.AUDITOR, '--pairs-json', CONFIRMATION / 'pairs.json',
                '--dataset-root', DATASET, '--output', REPORT])), 'final report came from another audit command')
    audit = driver.module(driver.AUDITOR, 'summary_frozen_confirmation_auditor')
    identity_path = DATASET / 'identity.json'
    identity = read(identity_path)
    require(report['dataset_identity'] == {'path': str(identity_path), 'sha256': digest(identity_path)}
            and report['pairs_manifest'] == {'path': str(CONFIRMATION / 'pairs.json'),
                                            'sha256': digest(CONFIRMATION / 'pairs.json')},
            'report dataset or pair manifest binding differs')
    require(identity['split_sizes'] == SPLITS and identity['generation_seed_offset'] == 1_000_000,
            'published split sizes or generation protocol differ')
    bundle_path = DATASET / 'frozen_recipes.json'
    bundle = read(bundle_path)
    require(digest(bundle_path) == identity['frozen_recipe_bundle_sha256']
            and bundle['confirmation_outcomes_used'] is False, 'pre-generation recipe bundle differs')
    recipes, receipts = {}, {}
    for domain in DOMAINS:
        row = report['domains'][domain]
        recipe_path = DATASET / 'recipes' / (domain + '.json')
        require(row['frozen_recipe']['path'] == str(recipe_path)
                and row['frozen_recipe']['sha256'] == digest(recipe_path) == identity['recipe_sha256'][domain]
                and row['frozen_recipe']['bundle_sha256'] == digest(bundle_path), 'frozen recipe provenance differs')
        recipes[domain] = read(recipe_path)
        for split, size in SPLITS.items():
            record = identity['domains'][domain][split]
            require(type(record['rows']) is int and record['rows'] == size
                    and record['path'] == str(DATASET / domain / split), 'published split path or count differs')
        for role, label in (('baseline', '05b'), ('candidate', '3b')):
            path = ROOT / 'var/results/modebench_level3_v2/confirmation_python_v6' / f'confirmation_{label}_{domain}.json'
            require(row['receipts'][role]['path'] == str(path)
                    and row['receipts'][role]['sha256'] == digest(path), 'confirmation receipt binding differs')
            receipt = read(path)
            samples = audit.validate_receipt(receipt, domain=domain, role=role, development=False)
            for metric in TOLERANCES:
                require(math.isclose(statistics.mean(item[metric] for item in samples), row['means'][role][metric],
                                     rel_tol=0, abs_tol=1e-12), 'report means differ from actual recorded attempts')
            receipts[str(path)] = digest(path)
    require(digest(REPORT) == report_hash, 'report changed during summary authentication')
    return report, identity, bundle, recipes, {'report': report_hash, 'identity': digest(identity_path),
        'bundle': digest(bundle_path), 'continuation_seal': CONTINUATION_SHA, 'receipts': receipts}


def link(label, path):
    return f'[{label}](<{Path(path)}>)'


def render_summary(report, identity, bundle, recipes, hashes):
    matched = validate_report(report)
    lines = [('# ModeBench Level3 confirmation: all five domains match' if matched else
              '# ModeBench Level3 confirmation: outside the registered tolerances'), '',
        'Frozen Qwen2.5-3B-Instruct on Level3 was compared with frozen Qwen2.5-0.5B-Instruct on Level1 before training. '
        'Each model received 128 evaluation problems per domain, four independent draws and eight attempts per draw. '
        'The gates allow absolute differences of 4 percentage points for per-attempt pass@1 and 8 percentage points for pass@8.', '',
        '| Domain | L1 / 0.5B pass@1 | L3 / 3B pass@1 | Δ pp | Gate | L1 / 0.5B pass@8 | L3 / 3B pass@8 | Δ pp | Gate |',
        '|---|---:|---:|---:|:---:|---:|---:|---:|:---:|']
    for domain in DOMAINS:
        row = report['domains'][domain]
        values = [LABELS[domain]]
        for metric in TOLERANCES:
            values += [f"{100 * row['means'][role][metric]:.4f}%" for role in ('baseline', 'candidate')]
            values += [f"{100 * row['candidate_minus_baseline'][metric]:+.4f}",
                       'PASS' if row['within_tolerance'][metric] else 'FAIL']
        lines.append('| ' + ' | '.join(values) + ' |')
    lines += ['', 'Δ is Level3/3B minus Level1/0.5B. Gates use unrounded values. '
        'The observed tolerance result is shown above; bootstrap intervals in the audit are diagnostic.', '',
        'Fresh held-out problems apply to Level3 only. The Level1 evaluation controls were reused under the '
        'prospective fixed-control amendment because the finite Countdown control catalogue was exhausted. '
        'The original graders and canonicalization were preserved; every split retains the Level2 support histogram '
        'and Pantry also retains the joint family/support histogram.', '',
        '| Domain | Train (384) | Development (128) | Evaluation (128) |', '|---|---|---|---|']
    for domain in DOMAINS:
        lines.append('| ' + ' | '.join([LABELS[domain], *[
            link(split, identity['domains'][domain][split]['path']) for split in SPLITS]]) + ' |')
    lines += ['', 'The 3,200 published rows use recipes frozen before fresh train/evaluation generation. '
        'Development selection uses seed 6391701; the generation seed offset is 1,000,000. '
        'Each recipe records its development receipts, fitted weights and generator source hashes.', '',
        '| Domain | Frozen recipe (SHA-256 prefix) | Tier weights d0–d3 | Train seed | Evaluation seed |',
        '|---|---|---|---:|---:|']
    for domain in DOMAINS:
        recipe = recipes[domain]
        values = [LABELS[domain], link(identity['recipe_sha256'][domain][:12], DATASET / 'recipes' / (domain + '.json')),
                  ', '.join(f'{weight:.2f}' for weight in recipe['weights']),
                  str(identity['domains'][domain]['train']['seed']), str(identity['domains'][domain]['eval']['seed'])]
        lines.append('| ' + ' | '.join(values) + ' |')
    lines += ['', f"Full generation/source provenance: {link('frozen recipe bundle', DATASET / 'frozen_recipes.json')} "
              f"(SHA-256 `{hashes['bundle']}`).",
        f"Finalizer: {link('source', ROOT / 'ops/exp_scaling/finalize_modebench_level3_python_v6_independent.py')} "
        f"(SHA-256 `{bundle['finalizer_source_sha256']}`).", '',
        f"Audited results: {link('confirmation report', REPORT)} (SHA-256 `{hashes['report']}`).",
        f"Dataset identity: {link('identity.json', DATASET / 'identity.json')} (SHA-256 `{hashes['identity']}`).",
        f"Execution/input provenance: {link('confirmation seal', CONFIRMATION / 'seal.json')}, "
        f"{link('completed execution audit', CONTINUATION / 'confirmation_completed_execution.json')}, "
        f"{link('fixed-control amendment', CAMPAIGN / 'confirmation_control_amendment.json')}.", '']
    return '\n'.join(lines)


def publish():
    require(not OUTPUT.exists(), 'summary already exists; existing evidence will not be overwritten')
    report, identity, bundle, recipes, hashes = authenticated_inputs()
    content = render_summary(report, identity, bundle, recipes, hashes)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + OUTPUT.name + '.', dir=OUTPUT.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(content); handle.flush(); os.fsync(handle.fileno())
        require(digest(REPORT) == hashes['report'], 'report changed before summary publication')
        os.link(temporary, OUTPUT)
    finally:
        os.unlink(temporary)
    return {'path': str(OUTPUT), 'sha256': digest(OUTPUT), 'confirmation_match_verified': report['confirmation_match_verified']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='authenticate completed inputs without publishing')
    args = parser.parse_args()
    if args.check:
        report, *_ = authenticated_inputs()
        print(json.dumps({'status': 'complete_confirmation_authenticated', 'confirmation_match_verified': report['confirmation_match_verified']}))
    else:
        print(json.dumps(publish()))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({'status': 'summary_refused', 'detail': str(error)}), file=sys.stderr)
        raise SystemExit(2)
