#!/usr/bin/env python3
"""Create frozen Python v4 dev pools and audit original external witnesses.

The 384/128/128 split builds are CPU capacity evidence, not finalized data.
No model outcomes enter generation or any structural check.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import islice
import json
from pathlib import Path
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import materialize_modebench_level3 as materializer
import modebench_level3_python_v4 as candidate
import make_python_factor_mode_data as certified
from oat_drgrpo.python_modebench import parse_python_factor_spec, python_factor_mode_count

ROOT = materializer.ROOT
DOMAIN = 'python_factors'
POOL_ROOT = ROOT / 'var/data/modebench_level3_calibration_v7'
OUTPUT = ROOT / 'var/artifacts/modebench_level3_python_v4/capacity_audit.json'
SPLIT_OFFSETS = {'train': 0, 'dev': 10_000, 'eval': 20_000}


def paths():
    return sorted(path for root in (ROOT / 'var/data').glob('modebench_level3_calibration*')
                  for path in (root / 'pools' / DOMAIN).glob('*.jsonl'))


def exclusions():
    blocked = materializer.historical_ids(DOMAIN)
    historical = len(blocked)
    all_paths = paths()
    for path in all_paths:
        blocked |= materializer.identity_set(DOMAIN, [json.loads(line) for line in path.read_text().splitlines()])
    return blocked, {'historical_identities': historical, 'historical_and_candidate_identities': len(blocked),
                     'candidate_pool_paths': [str(path.relative_to(ROOT)) for path in all_paths]}


def fingerprints(files):
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(files))}


def capacity_table(blocked):
    targets = {split: materializer.modes(materializer.reference_rows(DOMAIN, split)) for split in SPLIT_OFFSETS}
    total = sum(targets.values(), Counter())
    strict = total + Counter({support: 3 * count for support, count in targets['dev'].items()})
    result = {}
    for difficulty in candidate.PRESETS:
        cells = {}
        for support in sorted(total):
            available = candidate.available_capacity(support, difficulty, blocked)
            raw = candidate.available_capacity(support, difficulty, set())
            assert available >= strict[support]
            cells[support] = {'raw': raw, 'remaining': available, 'excluded': raw - available,
                              'demand_640': total[support], 'demand_1024': strict[support],
                              'split_histogram': {split: target[support] for split, target in targets.items()}}
        result[difficulty] = {'preset': candidate.PRESETS[difficulty],
                              'catalog_size': len(candidate.catalog(difficulty)[0]), 'cells': cells}
    return result


class ExternalWitnessAudit:
    """Observe real worker results while preserving the original call unchanged."""
    def __init__(self):
        self.original = certified.validate_python_factor_function_external
        self.successful_calls = Counter()
        self.keys = defaultdict(set)
        self.calls = 0

    def __call__(self, program, spec):
        result = self.original(program, spec)
        self.calls += 1
        if result is None:
            raise RuntimeError('original external witness validation failed; no alternative row is selected')
        cases = tuple(spec['cases'])
        self.successful_calls[cases] += 1
        self.keys[cases].add(result.canonical_key)
        return result

    def check(self, rows):
        modes = Counter()
        for row in rows:
            spec = json.loads(row['answer'])
            cases = tuple(spec['cases'])
            difficulty = row['level3_difficulty']
            assert len(cases) == len(set(cases)) == 4
            assert parse_python_factor_spec(spec) == cases
            assert all(value in candidate.catalog(difficulty)[0] for value in cases)
            assert python_factor_mode_count(cases) == row['answer_mode_count'] == spec['num_modes']
            assert row['problem'] == certified._prompt(cases)
            assert self.successful_calls[cases] == 2 and len(self.keys[cases]) == 2
            assert spec['num_externally_certified_modes'] == 2
            expected_hash = hashlib.sha256('\n'.join(sorted(self.keys[cases])).encode()).hexdigest()
            assert spec['certified_mode_key_sha256'] == expected_hash
            if difficulty < 2:
                assert all(value % 2 == 0 for value in cases)
            modes[row['answer_mode_count']] += 1
        return {'rows': len(rows), 'original_external_witnesses': 2 * len(rows),
                'exact_canonical_product_histogram': dict(sorted(modes.items()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--materialize-development', action='store_true')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    old_files = [HERE / 'modebench_level3_python_v3.py', HERE / 'modebench_level3_python_v2.py',
                 Path(materializer.__file__), Path(certified.__file__),
                 ROOT / 'src/oat_drgrpo/python_modebench.py', ROOT / 'src/oat_drgrpo/python_modebench_process.py']
    old_files += [path for path in paths() if POOL_ROOT not in path.parents]
    old_files += [path.with_suffix('.identity.json') for path in paths()
                  if POOL_ROOT not in path.parents and path.with_suffix('.identity.json').exists()]
    before = fingerprints(old_files)
    source_hash = hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest()
    preblocked, preexclusions = exclusions()
    before_table = capacity_table(preblocked)
    witness = ExternalWitnessAudit()
    development = []
    runs = []
    with patch.object(certified, 'validate_python_factor_function_external', witness):
        if args.materialize_development:
            seed = materializer.SEEDS[DOMAIN] + 600_000
            with patch.object(materializer, 'generator', return_value=candidate.build_pool), \
                 patch.dict(materializer.SEEDS, {DOMAIN: seed}):
                for difficulty in candidate.PRESETS:
                    record = materializer.build_development_pool(DOMAIN, difficulty, POOL_ROOT, multiplier=1)
                    path = POOL_ROOT / 'pools' / DOMAIN / f'difficulty_{difficulty}.jsonl'
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    record['witness_audit'] = witness.check(rows)
                    development.append(record)
                    print(json.dumps({'development_difficulty': difficulty, 'rows': len(rows),
                                      'seed': record['seed'], 'rows_sha256': record['rows_sha256']}), flush=True)
        blocked, after_exclusions = exclusions()
        after_table = capacity_table(blocked)
        global_blocked = set(blocked)
        for difficulty in candidate.PRESETS:
            for split in SPLIT_OFFSETS:
                reference = materializer.reference_rows(DOMAIN, split)
                target = materializer.modes(reference)
                seed = materializer.SEEDS[DOMAIN] + 700_000 + 1000 * difficulty + SPLIT_OFFSETS[split]
                tag = f'python_v4_capacity_{split}'
                rows = candidate.build_pool(DOMAIN, target, global_blocked, seed, tag, difficulty, multiplier=1)
                checks = materializer.verify_rows(DOMAIN, rows, reference, target, global_blocked)
                for support, count in target.items():
                    extended = list(islice(candidate.case_stream(support, global_blocked, seed, difficulty), count + 1))
                    emitted = sorted((row for row in rows if row['answer_mode_count'] == support),
                                     key=lambda row: row['level3_cell_index'])
                    assert extended[:count] == [tuple(json.loads(row['answer'])['cases']) for row in emitted]
                checks['per_cell_quota_prefix_invariance'] = True
                record = {'difficulty': difficulty, 'split': split, 'seed': seed,
                          'rows_sha256': materializer.row_hash(rows), 'checks': checks, **witness.check(rows)}
                runs.append(record)
                global_blocked |= materializer.identity_set(DOMAIN, rows)
                print(json.dumps({'capacity_difficulty': difficulty, 'split': split, 'rows': len(rows),
                                  'original_external_witnesses': record['original_external_witnesses']}), flush=True)
    after = fingerprints(old_files)
    assert before == after
    assert source_hash == hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest()
    assert len(global_blocked) - len(blocked) == sum(run['rows'] for run in runs)
    result = {
        'schema': 'modebench_level3_python_v4_original_external_witness_capacity_v1',
        'generator_profile': candidate.PROFILE, 'generator_source_sha256': source_hash,
        'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'information_boundary': 'CPU structural capacity only; no model outcomes, fitting, finalization or match claim.',
        'development_seed_rule': 'SEEDS[python_factors]+600000+1000*difficulty',
        'capacity_seed_rule': 'SEEDS[python_factors]+700000+1000*difficulty+split_offset',
        'capacity_split_offsets': SPLIT_OFFSETS,
        'frozen_presets': candidate.PRESETS,
        'exclusions_before_pilots': preexclusions, 'exclusions_after_pilots': after_exclusions,
        'all_60_cell_capacities_before_pilots': before_table,
        'all_60_cell_capacities_after_pilots': after_table,
        'development_pools': development, 'capacity_runs': runs,
        'capacity_rows': sum(run['rows'] for run in runs),
        'capacity_original_external_witnesses': sum(run['original_external_witnesses'] for run in runs),
        'all_original_external_witness_calls': witness.calls,
        'all_capacity_rows_globally_disjoint': True,
        'preserved_original_source_and_old_pool_sha256': before,
        'original_sources_old_pools_root_routing_unchanged': before == after,
    }
    materializer.write_json(args.output, result)
    print(json.dumps({'audit': str(args.output), 'source_sha256': source_hash,
                      'capacity_rows': result['capacity_rows'], 'external_witnesses': witness.calls}), flush=True)


if __name__ == '__main__':
    main()
