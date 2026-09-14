#!/usr/bin/env python3
"""Audit fixed graph v6 capacity and optionally create fresh development pools.

All checks are CPU-only and use no model outcomes. The 384/128/128 dry-run
splits demonstrate capacity; they are not finalized or confirmation datasets.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import combinations, product
import json
from math import comb
from pathlib import Path
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import materialize_modebench_level3 as materializer
import modebench_level3_graph_v6 as graph
from oat_drgrpo.math_grader import (
    _verify_graph_coloring_answer, validated_modebench_outcome_key,
)

ROOT = materializer.ROOT
DOMAIN = 'graph_coloring'
DEFAULT_POOLS = ROOT / 'var/data/modebench_level3_calibration_v6'
DEFAULT_OUTPUT = ROOT / 'var/artifacts/modebench_level3_graph_v6/structural_audit.json'
DEVELOPMENT_SEED_OFFSET = 500_000
CAPACITY_SEED_OFFSET = 600_000
SPLIT_OFFSETS = {'train': 0, 'dev': 10_000, 'eval': 20_000}


def pool_paths():
    return sorted(path for root in (ROOT / 'var/data').glob('modebench_level3_calibration*')
                  for path in (root / 'pools' / DOMAIN).glob('*.jsonl'))


def exclusions():
    blocked = materializer.historical_ids(DOMAIN)
    historical_count = len(blocked)
    paths = pool_paths()
    for path in paths:
        blocked |= materializer.identity_set(DOMAIN, [json.loads(line) for line in path.read_text().splitlines()])
    return blocked, {'historical_identities': historical_count,
                     'historical_and_candidate_identities': len(blocked),
                     'candidate_pool_files': [str(path.relative_to(ROOT)) for path in paths]}


def fingerprints(paths):
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths))}


def monochrome_capacity(difficulty, support, blocked):
    """Enumerate the entire labeled shared-color law, without sampling/ranking."""
    n, hidden_count, _ = graph.structure(support, difficulty)
    visible_count = n - hidden_count
    connected = next(k for k in range(hidden_count + 1)
                     if 2 ** k * 3 ** (hidden_count - k) == support)
    expected = comb(n, hidden_count) * 3 * comb(hidden_count, connected) * \
               (2 ** visible_count - 1) ** connected
    available = Counter()
    total = Counter()
    for hidden_tuple in combinations(range(n), hidden_count):
        hidden = set(hidden_tuple)
        cross_edges = [(u + 1, v + 1) for u, v in combinations(range(n), 2)
                       if (u in hidden) != (v in hidden)]
        for mask in range(1 << len(cross_edges)):
            edges = tuple(edge for index, edge in enumerate(cross_edges) if mask & (1 << index))
            touched = {vertex - 1 for edge in edges for vertex in edge} & hidden
            if 2 ** len(touched) * 3 ** (hidden_count - len(touched)) != support:
                continue
            for color in (1, 2, 3):
                partial = ''.join('?' if vertex in hidden else str(color) for vertex in range(n))
                identity = (DOMAIN, n, edges, partial)
                total[color] += 1
                if identity not in blocked:
                    available[color] += 1
    assert sum(total.values()) == expected
    assert len(set(total.values())) == 1  # Exact equal proposal capacity for each shared color.
    return {'support': support, 'total_identities': expected,
            'excluded_identities': expected - sum(available.values()),
            'remaining_identities': sum(available.values()),
            'total_by_visible_color': dict(total), 'remaining_by_visible_color': dict(available)}


def verify_uniform_color_law():
    class ScriptedRNG:
        def __init__(self, n, hidden_count, color):
            self.n, self.hidden_count, self.color = n, hidden_count, color
            self.color_calls, self.edge_calls = [], 0

        def sample(self, population, count):
            assert list(population) == list(range(self.n)) and count == self.hidden_count
            return list(range(count))

        def randint(self, low, high):
            self.color_calls.append((low, high))
            return self.color

        def random(self):
            self.edge_calls += 1
            return (0.25, 0.5)[self.edge_calls % 2]

    for n, hidden_count in ((5, 2), (5, 3), (6, 2), (6, 3)):
        proposals = []
        for color in (1, 2, 3):
            rng = ScriptedRNG(n, hidden_count, color)
            edges, partial = graph._candidate(n, hidden_count, True, rng, monochrome=True)
            assert rng.color_calls == [(1, 3)]
            assert rng.edge_calls == hidden_count * (n - hidden_count)
            assert {value for value in partial if value is not None} == {color}
            proposals.append((edges, graph.graph_completion_count(n, edges, partial)))
        assert proposals[0] == proposals[1] == proposals[2]
    return True


def verify_witnesses(rows):
    histogram = Counter()
    visible_colors = Counter()
    total_witnesses = 0
    for row in rows:
        spec = json.loads(row['answer'])
        support, difficulty = row['answer_mode_count'], row['level3_difficulty']
        n, hidden_count, independent = graph.structure(support, difficulty)
        assert spec['n'] == n and spec['verifier'] == DOMAIN
        hidden = {i + 1 for i, color in enumerate(spec['partial_colors']) if color is None}
        assert len(hidden) == hidden_count
        assert row['problem'] == graph._graph_prompt(n, spec['edges'], spec['partial_colors'])
        if independent:
            assert all(not (u in hidden and v in hidden) for u, v in spec['edges'])
        if graph.proposal_law(support, difficulty) == 'shared_visible_color':
            colors = {color for color in spec['partial_colors'] if color is not None}
            assert len(colors) == 1
            visible_colors[next(iter(colors))] += 1
            assert all((u in hidden) != (v in hidden) for u, v in spec['edges'])
        witnesses = [''.join(map(str, fill)) for fill in product((1, 2, 3), repeat=hidden_count)
                     if _verify_graph_coloring_answer(''.join(map(str, fill)), spec)]
        canonical = {validated_modebench_outcome_key('\\boxed{' + fill + '}', row['answer'])
                     for fill in witnesses}
        assert None not in canonical and len(canonical) == len(witnesses) == support
        assert graph.graph_completion_count(n, spec['edges'], spec['partial_colors']) == support
        assert graph.graph_completion_count(n, spec['edges'], [None] * n) == spec['num_solutions']
        histogram[support] += 1
        total_witnesses += len(witnesses)
    return {'rows_verified': len(rows), 'original_grader_witnesses': total_witnesses,
            'canonical_support_histogram': dict(sorted(histogram.items())),
            'shared_visible_color_row_counts': dict(sorted(visible_colors.items()))}


def cell(rows, support):
    return sorted((row for row in rows if row['answer_mode_count'] == support),
                  key=lambda row: row['level3_cell_index'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--materialize-development', action='store_true')
    parser.add_argument('--pool-root', type=Path, default=DEFAULT_POOLS)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    preserved_paths = [HERE / 'modebench_level3_graph_v5.py', Path(materializer.__file__),
                       ROOT / 'src/oat_drgrpo/math_grader.py', ROOT / 'ops/make_modebench_data.py']
    preserved_paths += [path for path in pool_paths() if args.pool_root not in path.parents]
    preserved_paths += [path.with_suffix('.identity.json') for path in pool_paths()
                        if args.pool_root not in path.parents and path.with_suffix('.identity.json').exists()]
    before = fingerprints(preserved_paths)
    development = []
    if args.materialize_development:
        seed = materializer.SEEDS[DOMAIN] + DEVELOPMENT_SEED_OFFSET
        with patch.object(materializer, 'generator', return_value=graph.build_pool), \
             patch.dict(materializer.SEEDS, {DOMAIN: seed}):
            for difficulty in graph.PRESETS:
                record = materializer.build_development_pool(DOMAIN, difficulty, args.pool_root, multiplier=1)
                rows_path = args.pool_root / 'pools' / DOMAIN / f'difficulty_{difficulty}.jsonl'
                rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
                record['independent_witness_audit'] = verify_witnesses(rows)
                development.append(record)
                print(json.dumps({'development_difficulty': difficulty, 'rows': len(rows),
                                  'seed': record['seed'], 'rows_sha256': record['rows_sha256']}), flush=True)
    blocked, exclusion_record = exclusions()
    capacity = {str(difficulty): {
        str(support): monochrome_capacity(difficulty, support, blocked)
        for support in sorted(graph.SUPPORTS)
        if graph.proposal_law(support, difficulty) == 'shared_visible_color'
    } for difficulty in (0, 1)}
    global_blocked = set(blocked)
    runs = []
    for difficulty in graph.PRESETS:
        for split in materializer.SPLITS:
            reference = materializer.reference_rows(DOMAIN, split)
            target = materializer.modes(reference)
            seed = materializer.SEEDS[DOMAIN] + CAPACITY_SEED_OFFSET + 1000 * difficulty + SPLIT_OFFSETS[split]
            tag = f'graph_v6_capacity_{split}'
            rows = graph.build_pool(DOMAIN, target, global_blocked, seed, tag, difficulty, multiplier=1)
            checks = materializer.verify_rows(DOMAIN, rows, reference, target, global_blocked)
            larger = graph.build_pool(DOMAIN, Counter({support: count + 1 for support, count in target.items()}),
                                      global_blocked, seed, tag, difficulty, multiplier=1)
            checks['per_cell_prefix_quota_invariance'] = all(
                cell(larger, support)[:count] == cell(rows, support) for support, count in target.items())
            assert checks['per_cell_prefix_quota_invariance']
            witness_audit = verify_witnesses(rows)
            global_blocked |= materializer.identity_set(DOMAIN, rows)
            record = {'difficulty': difficulty, 'split': split, 'seed': seed, 'rows': len(rows),
                      'support_histogram': dict(sorted(target.items())), 'rows_sha256': materializer.row_hash(rows),
                      'checks': checks, **witness_audit}
            runs.append(record)
            print(json.dumps({'capacity_difficulty': difficulty, 'split': split, 'rows': len(rows),
                              'original_grader_witnesses': witness_audit['original_grader_witnesses']}), flush=True)
    after = fingerprints(preserved_paths)
    assert before == after
    result = {
        'schema': 'modebench_level3_graph_v6_structural_audit_v1',
        'generator': graph.SCHEMA, 'generator_source_sha256': hashlib.sha256(Path(graph.__file__).read_bytes()).hexdigest(),
        'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'development_seed_rule': 'SEEDS[graph_coloring] + 500000 + 1000 * difficulty',
        'capacity_seed_rule': 'SEEDS[graph_coloring] + 600000 + 1000 * difficulty + split_offset',
        'capacity_split_offsets': SPLIT_OFFSETS,
        'information_boundary': 'Structural capacity only; no model outcomes, fitting, finalization, or match claim.',
        'hypothesis': 'Uniform shared visible color may simplify redundant constraints and increase pass@1 relative to pass@8; coupled presets remain harder mixture components.',
        'exceptions': {'5': 'Exact v5 coupled proposal; independent color choices cannot yield prime support 5.',
                       '9': 'Exact v5 independent random visible colors; monochrome support 9 requires the empty graph.'},
        'uniform_visible_color_generation_law_verified': verify_uniform_color_law(),
        'uniform_visible_color_law': 'One random.Random.randint(1, 3); each hidden-visible edge independently random() < 0.5; exact support and identity rejection only.',
        'exclusions': exclusion_record, 'exact_monochrome_capacity_after_all_candidate_exclusions': capacity,
        'development_pools': development, 'capacity_runs': runs,
        'capacity_rows_verified': sum(run['rows'] for run in runs),
        'capacity_witnesses_verified': sum(run['original_grader_witnesses'] for run in runs),
        'all_capacity_rows_globally_disjoint': len(global_blocked) - len(blocked) == sum(run['rows'] for run in runs),
        'preserved_source_and_old_pool_sha256': before,
        'original_sources_old_pools_and_root_routing_unchanged': before == after,
    }
    materializer.write_json(args.output, result)
    print(json.dumps({'audit': str(args.output), 'generator_source_sha256': result['generator_source_sha256'],
                      'capacity_rows_verified': result['capacity_rows_verified']}), flush=True)


if __name__ == '__main__':
    main()
