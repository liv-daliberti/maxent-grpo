#!/usr/bin/env python3
"""Inspect or advance the frozen five-domain Level 3 calibration campaign.

Default operation is read-only. --advance may merge complete development draws,
fit current candidate tiers, then materialize all five matched recipes together.
It never submits jobs, evaluates a model, or declares confirmation successful.
Every mutating invocation leaves an immutable journal of commands and artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import uuid

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / 'ops/exp_scaling', ROOT / 'ops', ROOT / 'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DOMAINS = ('countdown', 'graph_coloring', 'python_factors', 'mathir', 'pantry')
SEEDS = [6318000, 6318001, 6318002, 6318003]
RESULTS = ROOT / 'var/results/modebench_level3_v1'
ARTIFACTS = ROOT / 'var/artifacts/modebench_level3_v1'
GENERATORS = {
    'countdown': 'modebench_level3_countdown_v3.py',
    'graph_coloring': 'modebench_level3_graph_v6.py',
    'python_factors': 'modebench_level3_python_v4.py',
    'mathir': 'modebench_level3_mathir_sign_v1.py',
    'pantry': 'modebench_level3_constraints.py',
}
SCHEMA = 'modebench_level3_calibration_advance_v1'


class NotReady(Exception):
    """A required immutable upstream artifact has not arrived."""


def read(path):
    path = Path(path)
    if not path.exists():
        raise NotReady(str(path))
    return json.loads(path.read_text())


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_generator(domain):
    from materialize_modebench_level3 import generator
    from fit_modebench_level3 import generator_sources
    actual = Path(sys.modules[generator(domain).__module__].__file__).resolve()
    expected = ROOT / 'ops/exp_scaling' / GENERATORS[domain]
    require(actual == expected, f'{domain}: unexpected generator mapping {actual}')
    return actual, generator_sources(domain)


@lru_cache(maxsize=64)
def receipt(path, domain, level, model, seeds, difficulty=None):
    """Use the public attempt/identity validator and authenticate source rows."""
    from merge_modebench_level3_receipts import validate_input
    from fit_modebench_level3 import load_receipt, receipt_rows, row_hash
    payload = read(path)
    validate_input(payload)
    require((payload['domain'], payload['level'], payload['model_label'], payload['split']) ==
            (domain, level, model, 'dev'), f'{path}: unexpected receipt cell')
    require(payload['identity']['seeds'] == list(seeds), f'{path}: wrong registered seeds')
    require(payload['identity']['interface']['name'] == 'level2_qwen_r5', f'{path}: wrong campaign interface')
    if list(seeds) == SEEDS:
        payload, rows, scores = load_receipt(path, domain, level, model)
    else:
        rows, scores = receipt_rows(payload), None
    if difficulty is not None:
        main, _ = check_generator(domain)
        require({row.get('level3_difficulty') for row in rows} == {difficulty}, f'{path}: wrong candidate tier')
        source = payload['identity']['source']
        require(source['kind'] == 'jsonl', f'{path}: candidate source must be frozen JSONL')
        certificate_path = Path(source['path']).with_suffix('.identity.json')
        certificate = read(certificate_path)
        require(certificate.get('source_sha256') == file_sha(main), f'{path}: stale candidate generator hash')
        require(certificate.get('rows_sha256') == row_hash(rows), f'{path}: candidate certificate row mismatch')
        require(certificate.get('domain') == domain and certificate.get('difficulty') == difficulty,
                f'{path}: candidate certificate cell mismatch')
        require(certificate.get('checks') and all(value is True for value in certificate['checks'].values()),
                f'{path}: candidate structural checks did not pass')
    return payload, rows, scores


def paths_for(domain, results, recipes):
    baseline = results / f'calibrated_baseline_05b_{domain}.json'
    if domain == 'pantry':
        scores = [results / f'calibrated_3b_pantry_d{d}.json' for d in range(4)]
    else:
        version = {'graph_coloring': 'v6', 'countdown': 'v5', 'python_factors': 'v7', 'mathir': 'sign'}[domain]
        scores = [results / f'calibrated_{version}_3b_{domain}_d{d}.json' for d in range(4)]
    return baseline, scores, recipes / f'{domain}.json'


def validate_recipe(path, domain, baseline, scores):
    """Public freeze guards plus authentication against this campaign's inputs."""
    from finalize_modebench_level3 import load_recipe, selected_development
    from fit_modebench_level3 import SELECTION_SEED, TOLERANCES, sha
    read(path)
    recipe = load_recipe(path, domain)
    require(recipe['provenance']['seeds'] == SEEDS, f'{path}: wrong recipe seeds')
    require(recipe['selection']['seed'] == SELECTION_SEED, f'{path}: changed selection seed')
    require(recipe['development']['tolerances'] == TOLERANCES, f'{path}: changed matching tolerances')
    provenance = recipe['provenance']
    require(Path(provenance['baseline_receipt_path']).resolve() == baseline.resolve() and
            provenance['baseline_receipt_sha256'] == file_sha(baseline), f'{path}: stale baseline binding')
    baseline_payload, _, _ = receipt(str(baseline), domain, 'level1', '05b', tuple(SEEDS))
    lookups = {}
    for d, score in enumerate(scores):
        bound = provenance['pools'][str(d)]
        require(Path(bound['receipt_path']).resolve() == score.resolve() and
                bound['receipt_sha256'] == file_sha(score), f'{path}: stale tier {d} receipt binding')
        payload, _, lookups[d] = receipt(str(score), domain, 'level3', '3b', tuple(SEEDS), d)
        require(payload['identity']['interface'] == baseline_payload['identity']['interface'] and
                payload['identity']['code_sha256'] == baseline_payload['identity']['code_sha256'],
                f'{path}: baseline/candidate interface or code mismatch')
    selected = selected_development(domain, recipe)
    for metric, tolerance in TOLERANCES.items():
        value = statistics.mean(lookups[row['level3_difficulty']][sha(row)][metric] for row in selected)
        baseline_value = baseline_payload['metrics'][metric]
        require(math.isclose(recipe['development']['selected_metrics'][metric], value, abs_tol=1e-12) and
                math.isclose(recipe['development']['baseline_metrics'][metric], baseline_value, abs_tol=1e-12),
                f'{path}: recorded recipe metrics do not reproduce')
        require(abs(value - baseline_value) <= tolerance, f'{path}: development fit is outside tolerance')
    return recipe


def validate_dataset(output, mapping):
    """Authenticate an existing final publication; never overwrite or re-freeze it."""
    from datasets import load_from_disk
    from finalize_modebench_level3 import SCHEMA as DATA_SCHEMA
    from fit_modebench_level3 import cell_histogram, generator_sources
    from materialize_modebench_level3 import SPLITS, identity_set, reference_rows, row_hash
    identity = read(output / 'identity.json')
    require(identity == read(output / 'admission_fairness_report.json'), 'final dataset reports disagree')
    require(identity.get('schema') == DATA_SCHEMA and identity.get('status') == 'structural_checks_pass' and
            identity.get('decision') == 'pending_confirmation', 'existing output is not the expected pending-confirmation dataset')
    expected_hashes = {domain: file_sha(path) for domain, path in mapping.items()}
    require(identity['recipe_sha256'] == expected_hashes, 'existing dataset binds different recipes')
    sources = {domain: generator_sources(domain) for domain in DOMAINS}
    require(identity['generator_sources_sha256'] == sources, 'existing dataset generator dependencies changed')
    frozen = read(output / 'frozen_recipes.json')
    require(file_sha(output / 'frozen_recipes.json') == identity['frozen_recipe_bundle_sha256'] and
            frozen['recipes_sha256'] == expected_hashes and frozen['generator_sources_sha256'] == sources and
            frozen['finalizer_source_sha256'] == file_sha(ROOT / 'ops/exp_scaling/finalize_modebench_level3.py'),
            'existing dataset freeze bundle changed')
    require(identity['split_sizes'] == {split: pair[0] for split, pair in SPLITS.items()} and
            set(identity['domains']) == set(DOMAINS), 'existing dataset split/domain contract changed')
    for domain in DOMAINS:
        require(file_sha(output / 'recipes' / f'{domain}.json') == expected_hashes[domain], 'archived recipe changed')
        seen = set()
        for split, (count, dataset_split) in SPLITS.items():
            rows = [dict(row) for row in load_from_disk(str(output / domain / split))[dataset_split]]
            record = identity['domains'][domain][split]
            require(len(rows) == count and row_hash(rows) == record['rows_sha256'], f'{domain}/{split}: dataset row/hash drift')
            require(record.get('checks') and all(value is True for value in record['checks'].values()),
                    f'{domain}/{split}: final structural checks failed')
            require(cell_histogram(domain, rows) == cell_histogram(domain, reference_rows(domain, split)),
                    f'{domain}/{split}: actual support/family histogram changed')
            ids = identity_set(domain, rows)
            require(len(ids) == count and not ids & seen, f'{domain}/{split}: final identity overlap')
            seen |= ids
    return identity


class Campaign:
    def __init__(self, args):
        self.args = args
        self.report = {'schema': SCHEMA, 'observed_at': datetime.now(timezone.utc).isoformat(),
                       'mode': 'advance' if args.advance else 'read_only', 'actions': [], 'domains': {},
                       'recipe_scope': str(args.recipes_root),
                       'ignored_diagnostic_recipes': {'path': str(ARTIFACTS / 'recipes_v1'),
                           'reason': 'Earlier selected-subset fit is diagnostic only; current fits require the full-pool cell forecast and selected-row gates.'},
                       'produced_artifacts': [], 'event_files': [], 'confirmation': 'not_run; no difficulty-match claim'}
        self.journal = None
        if args.advance:
            self.journal = args.journal_root / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex + '.json')
            self.report['journal_path'] = str(self.journal)

    def event(self, payload):
        if self.journal is not None:
            from fit_modebench_level3 import atomic_new
            path = self.journal.with_name(self.journal.stem + f".event-{len(self.report['event_files']):04d}.json")
            atomic_new(path, {'journal_path': str(self.journal), **payload})
            self.report['event_files'].append({'path': str(path), 'sha256': file_sha(path)})

    def record_artifact(self, path):
        path = Path(path).resolve()
        files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
        artifacts = [{'path': str(p), 'sha256': file_sha(p), 'bytes': p.stat().st_size} for p in files]
        self.report['produced_artifacts'].extend(artifacts)
        self.event({'event': 'artifacts_produced', 'artifacts': artifacts})

    def run_command(self, label, command, output):
        require(not output.exists(), f'fresh output required: {output}')
        action = {'action': label, 'command': command, 'script_sha256': file_sha(command[1]),
                  'output': str(output), 'status': 'running'}
        self.report['actions'].append(action)
        self.event({'event': 'action_started', 'action': action})
        print(json.dumps({'event': 'advance_action_started', 'action': label, 'output': str(output)}), flush=True)
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        action.update(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr,
                      status='produced' if completed.returncode == 0 else 'failed')
        if output.exists():
            self.record_artifact(output)
        receipt.cache_clear()
        self.event({'event': 'action_finished', 'action': action})
        require(completed.returncode == 0, f'{label} failed; see immutable advance journal')
        require(output.exists(), f'{label} did not produce {output}')

    def merge(self, domain, level, model, parts, output, difficulty=None):
        from merge_modebench_level3_receipts import merge_receipts
        info = {'output': str(output), 'inputs': [str(path) for path in parts]}
        try:
            for path, seeds in zip(parts, (SEEDS[:1], SEEDS[1:])):
                receipt(str(path), domain, level, model, tuple(seeds), difficulty)
            expected = merge_receipts(parts)
            require(expected['identity']['seeds'] == SEEDS, 'merged seeds differ from the four registered draws')
            if output.exists():
                actual, _, _ = receipt(str(output), domain, level, model, tuple(SEEDS), difficulty)
                # generated_at is the sole intentionally variable merge field.
                require({k: v for k, v in actual.items() if k != 'generated_at'} ==
                        {k: v for k, v in expected.items() if k != 'generated_at'},
                        f'{output}: existing merge is stale or unexpected')
                info['status'] = 'valid_existing'
            elif self.args.advance:
                self.run_command('merge', [sys.executable, str(ROOT / 'ops/merge_modebench_level3_receipts.py'),
                                          '--receipts', *map(str, parts), '--output', str(output)], output)
                actual, _, _ = receipt(str(output), domain, level, model, tuple(SEEDS), difficulty)
                require({k: v for k, v in actual.items() if k != 'generated_at'} ==
                        {k: v for k, v in expected.items() if k != 'generated_at'}, 'new merge differs from validated inputs')
                info['status'] = 'produced'
            else:
                info['status'] = 'ready_to_merge'
        except NotReady as error:
            info.update(status='waiting', detail=str(error))
        except Exception as error:
            info.update(status='blocked_unchanged', detail=f'{type(error).__name__}: {error}')
        return info

    def execute(self):
        results, recipes = self.args.results_root, self.args.recipes_root
        mapping = {}
        for domain in DOMAINS:
            baseline, scores, recipe_path = paths_for(domain, results, recipes)
            record = self.report['domains'][domain] = {'recipe': str(recipe_path), 'tiers': []}
            try:
                _, record['generator_sources_sha256'] = check_generator(domain)
                record['baseline'] = self.merge(domain, 'level1', '05b',
                    [results / f'guided_baseline_05b_{domain}.json', results / f'guided_baseline_extra_05b_{domain}.json'], baseline)
                if domain == 'pantry':
                    for d, path in enumerate(scores):
                        record['tiers'].append(self.merge(domain, 'level3', '3b',
                            [results / f'guided_pool_3b_pantry_d{d}.json', results / f'calibration_extra_3b_pantry_d{d}.json'], path, d))
                else:
                    for d, path in enumerate(scores):
                        tier = {'output': str(path)}
                        try:
                            receipt(str(path), domain, 'level3', '3b', tuple(SEEDS), d)
                            tier['status'] = 'valid_existing'
                        except NotReady as error:
                            tier.update(status='waiting', detail=str(error))
                        except Exception as error:
                            tier.update(status='blocked_unchanged', detail=f'{type(error).__name__}: {error}')
                        record['tiers'].append(tier)
                upstream = [record['baseline'], *record['tiers']]
                if recipe_path.exists():
                    # An invalid upstream merge cannot be legitimized by an old recipe.
                    require(all(part['status'] in ('valid_existing', 'produced') for part in upstream),
                            'existing recipe has missing, stale, or unexpected upstream artifacts')
                    recipe = validate_recipe(recipe_path, domain, baseline, scores)
                    record['status'] = 'valid_existing_recipe'
                elif all(part['status'] in ('valid_existing', 'produced') for part in upstream):
                    if self.args.advance:
                        self.run_command('fit_' + domain,
                            [sys.executable, str(ROOT / 'ops/exp_scaling/fit_modebench_level3.py'),
                             '--baseline', str(baseline), '--scores', *map(str, scores), '--domain', domain,
                             '--output', str(recipe_path)], recipe_path)
                        recipe = validate_recipe(recipe_path, domain, baseline, scores)
                        record['status'] = 'produced_passing_recipe'
                    else:
                        record['status'] = 'ready_to_fit'
                        continue
                else:
                    record['status'] = 'blocked_unchanged' if any(p['status'] == 'blocked_unchanged' for p in upstream) else 'waiting_for_receipts'
                    continue
                record['weights'] = recipe['weights']
                record['development_differences'] = recipe['development']['differences']
                mapping[domain] = recipe_path.resolve()
            except NotReady as error:
                record.update(status='waiting_for_receipts', detail=str(error))
            except Exception as error:
                record.update(status='blocked_unchanged', detail=f'{type(error).__name__}: {error}')
        self.finalize(mapping)
        return self.report

    def finalize(self, mapping):
        output = self.args.output_root
        info = self.report['finalization'] = {'output': str(output), 'confirmation': 'pending_external_evaluation'}
        if set(mapping) != set(DOMAINS):
            info.update(status='blocked_existing_output' if output.exists() else 'waiting_for_five_passing_recipes',
                        missing_domains=sorted(set(DOMAINS) - set(mapping)))
            return
        try:
            if output.exists():
                validate_dataset(output, mapping)
                info['status'] = 'valid_existing_pending_confirmation_dataset'
            elif not self.args.advance or self.args.fit_only:
                info['status'] = 'ready_to_finalize'
            else:
                from fit_modebench_level3 import atomic_new
                mapping_path = self.args.recipes_root / 'recipes.json'
                payload = {domain: str(path) for domain, path in mapping.items()}
                if mapping_path.exists():
                    require(read(mapping_path) == payload, 'existing recipe mapping differs; refusing replacement')
                else:
                    atomic_new(mapping_path, payload)
                    self.record_artifact(mapping_path)
                self.run_command('finalize', [sys.executable, str(ROOT / 'ops/exp_scaling/finalize_modebench_level3.py'),
                                             '--recipes-json', str(mapping_path), '--output-root', str(output)], output)
                validate_dataset(output, mapping)
                info['status'] = 'produced_pending_confirmation_dataset'
        except Exception as error:
            info.update(status='blocked_unchanged', detail=f'{type(error).__name__}: {error}')

    def save_journal(self):
        if self.journal is not None:
            from fit_modebench_level3 import atomic_new
            atomic_new(self.journal, self.report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--advance', action='store_true', help='explicitly permit ready merges, fits, and five-domain finalization')
    parser.add_argument('--fit-only', action='store_true', help='with --advance, stop before dataset finalization')
    parser.add_argument('--results-root', type=Path, default=RESULTS)
    parser.add_argument('--recipes-root', type=Path, default=ARTIFACTS / 'recipes_v5')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'var/data/modebench_level3_matched_v1')
    parser.add_argument('--journal-root', type=Path, default=ARTIFACTS / 'advance_runs')
    args = parser.parse_args(argv)
    for name in ('results_root', 'recipes_root', 'output_root', 'journal_root'):
        setattr(args, name, getattr(args, name).resolve())
    campaign = Campaign(args)
    if args.advance:
        args.journal_root.mkdir(parents=True, exist_ok=True)
        with (args.journal_root / '.advance.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.exit(1, 'another advance invocation holds the campaign lock\n')
            try:
                campaign.execute()
            finally:
                campaign.save_journal()
    else:
        campaign.execute()
    print(json.dumps(campaign.report, indent=2, sort_keys=True))
    blocked = any(row.get('status') == 'blocked_unchanged' for row in campaign.report['domains'].values())
    blocked |= campaign.report['finalization']['status'].startswith('blocked')
    return 1 if blocked else 0


if __name__ == '__main__':
    raise SystemExit(main())
