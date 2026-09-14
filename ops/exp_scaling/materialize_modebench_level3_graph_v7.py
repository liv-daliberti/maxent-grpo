#!/usr/bin/env python3
"""Register and materialize isolated graph-v7 development candidates, CPU only.

The catalogue audit enumerates every labeled n5 proposal identity, preserving
color symmetry. Separate ephemeral capacity witnesses demonstrate complete
384/128/128 construction for every preset; they are not evaluation datasets.
No model outcomes are opened, no fitter is called, and no GPU job is submitted.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from itertools import combinations, permutations, product
import json
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (HERE, ROOT / 'ops', ROOT / 'src'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
import materialize_modebench_level3 as materializer
import modebench_level3_graph_v7 as graph
from fit_modebench_level3 import file_sha, local_dependency_sources
from evaluate_modebench_level3 import atomic_new
from oat_drgrpo.math_grader import _verify_graph_coloring_answer, validated_modebench_outcome_key

DOMAIN = 'graph_coloring'
ARTIFACTS = ROOT / 'var/artifacts/modebench_level3_v2/graph_v7'
REGISTRATION = ARTIFACTS / 'candidate_protocol.json'
POOL_ROOT = ROOT / 'var/data/modebench_level3_calibration_graph_v7'
CALIBRATION_SEAL = ROOT / 'var/artifacts/modebench_level3_v2/implementation_seal.json'
CALIBRATION_SEAL_SHA = 'e03ffa74a476638401ddadb49b950f3d74377457114bf13b630ed8834be34736'
BASELINE = ROOT / 'var/results/modebench_level3_v2/calibration_05b_graph_coloring.json'
BASELINE_SHA = '9f4b407a47c5ffab7aec124252502d466e06ceff120ae7965af60218fa200f4c'
FAILED_FIT = ROOT / 'var/artifacts/modebench_level3_v2/recipes/graph_coloring.json'
DEVELOPMENT_SEED = materializer.SEEDS[DOMAIN] + 2_000_000
CAPACITY_SEED = materializer.SEEDS[DOMAIN] + 2_100_000


def identity(n, edges, partial):
    return (DOMAIN, n, tuple(sorted(tuple(sorted(edge)) for edge in edges)),
            ''.join('?' if value is None else str(value) for value in partial))


@lru_cache(maxsize=21)
def catalogue(difficulty, support):
    """Exact finite identity support of each n5 proposal, independent of RNG."""
    if difficulty not in (0, 1, 2) or support not in graph.SUPPORTS:
        raise ValueError('exact catalogue covers the three n5 presets only')
    found = set()
    for hidden in permutations(range(5), 3):
        visible = [vertex for vertex in range(5) if vertex not in hidden]
        for colors in product((1, 2, 3), repeat=2):
            partial = [None if vertex in hidden else colors[visible.index(vertex)] for vertex in range(5)]
            choices = []
            if support == 5:
                for left in visible:
                    for right in visible:
                        if partial[left] != partial[right]:
                            choices.append([(hidden[0], hidden[1]), (hidden[1], hidden[2]),
                                            (hidden[0], left), (hidden[2], right)])
            elif difficulty == 1:
                forbidden = {4: 2, 6: 2, 8: 1, 9: 2, 12: 0, 18: 0}[support]
                for anchors in combinations(visible, forbidden):
                    if len({partial[vertex] for vertex in anchors}) != forbidden:
                        continue
                    edges = [(hidden[0], anchor) for anchor in anchors]
                    if support in (4, 6, 8, 12, 18):
                        edges.append((hidden[0], hidden[1]))
                    if support in (4, 8, 12):
                        edges.append((hidden[1], hidden[2]))
                    choices.append(edges)
            else:
                neighbors = [[vertices for vertices in combinations(visible, 3 - available)
                              if len({partial[v] for v in vertices}) == 3 - available]
                             for available in graph.AVAILABILITIES[support]]
                for anchors in product(*neighbors):
                    choices.append([(vertex, anchor) for vertex, group in zip(hidden, anchors) for anchor in group])
            visible_edge = tuple(visible) if colors[0] != colors[1] else None
            for edges in choices:
                variants = [edges] if difficulty != 2 else []
                if difficulty in (1, 2) and visible_edge:
                    variants.append([*edges, visible_edge])
                for variant in variants:
                    found.add(identity(5, [(u + 1, v + 1) for u, v in variant], partial))
    return frozenset(found)


def source_snapshot():
    if file_sha(CALIBRATION_SEAL) != CALIBRATION_SEAL_SHA:
        raise ValueError('immutable calibration seal changed')
    seal = json.loads(CALIBRATION_SEAL.read_text())
    pins = {**seal['files_sha256'], str(CALIBRATION_SEAL): CALIBRATION_SEAL_SHA,
            str(BASELINE): BASELINE_SHA, str(FAILED_FIT): file_sha(FAILED_FIT)}
    trees = dict(seal['directory_files'])
    roots = sorted(set((ROOT / 'var/data').glob('modebench_harder*')) |
                   set((ROOT / 'var/data').glob('modebench_level3*')))
    pools = []
    for root in roots:
        for split in materializer.SPLITS:
            path = root / DOMAIN / split
            if (path / 'dataset_dict.json').is_file():
                trees[str(path.resolve())] = sorted(str(item.resolve()) for item in path.rglob('*') if item.is_file())
        pools.extend(sorted((root / 'pools' / DOMAIN).glob('*.jsonl')))
    for path in pools:
        pins[str(path.resolve())] = file_sha(path)
        certificate = path.with_suffix('.identity.json')
        if certificate.exists():
            pins[str(certificate.resolve())] = file_sha(certificate)
    for files in trees.values():
        for name in files:
            value = file_sha(name)
            if name in pins and pins[name] != value:
                raise ValueError('historical source differs from calibration seal')
            pins[name] = value
    for name, value in local_dependency_sources([Path(graph.__file__), Path(__file__)]).items():
        source = str(ROOT / name)
        if source in pins and pins[source] != value:
            raise ValueError('sealed source differs from local dependency')
        pins[source] = value
    tests = ROOT / 'tests/test_modebench_level3_graph_v7.py'
    pins[str(tests)] = file_sha(tests)
    snapshot = {'files_sha256': pins, 'directory_files': trees,
                'candidate_pool_paths': [str(path.resolve()) for path in pools],
                'historical_identity_sha256': materializer.row_hash(sorted(materializer.historical_ids(DOMAIN), key=repr))}
    verify_snapshot(snapshot)
    return snapshot


def verify_snapshot(snapshot):
    current_pools = sorted(str(path.resolve()) for root in (ROOT / 'var/data').glob('modebench_level3*')
                           if root.resolve() != POOL_ROOT.resolve()
                           for path in (root / 'pools' / DOMAIN).glob('*.jsonl'))
    if current_pools != sorted(snapshot['candidate_pool_paths']):
        raise ValueError('historical candidate pool inventory changed')
    if materializer.row_hash(sorted(materializer.historical_ids(DOMAIN), key=repr)) != snapshot['historical_identity_sha256']:
        raise ValueError('historical semantic identity inventory changed')
    for name, expected in snapshot['files_sha256'].items():
        if file_sha(name) != expected:
            raise ValueError(f'authenticated input changed: {name}')
    for directory, expected in snapshot['directory_files'].items():
        actual = sorted(str(path.resolve()) for path in Path(directory).rglob('*') if path.is_file())
        if actual != expected:
            raise ValueError(f'authenticated input inventory changed: {directory}')


def exclusions(snapshot):
    blocked = materializer.historical_ids(DOMAIN)
    historical = len(blocked)
    for name in snapshot['candidate_pool_paths']:
        rows = [json.loads(line) for line in Path(name).read_text().splitlines()]
        blocked |= materializer.identity_set(DOMAIN, rows)
    return blocked, historical


def verify_witnesses(rows):
    count = 0
    for row in rows:
        spec = json.loads(row['answer'])
        hidden = sum(value is None for value in spec['partial_colors'])
        if hidden != 3 or row['problem'] != graph._graph_prompt(spec['n'], spec['edges'], spec['partial_colors']):
            raise ValueError('original three-hidden graph prompt differs')
        witnesses = [''.join(map(str, fill)) for fill in product((1, 2, 3), repeat=3)
                     if _verify_graph_coloring_answer(''.join(map(str, fill)), spec)]
        canonical = {validated_modebench_outcome_key('\\boxed{' + value + '}', row['answer']) for value in witnesses}
        if (None in canonical or len(canonical) != len(witnesses) or len(witnesses) != row['answer_mode_count']
                or graph.graph_completion_count(spec['n'], spec['edges'], spec['partial_colors']) != len(witnesses)
                or graph.graph_completion_count(spec['n'], spec['edges'], [None] * spec['n']) != spec['num_solutions']):
            raise ValueError('original verifier/canonical support differs')
        count += len(witnesses)
    return count


def exact_capacity(blocked):
    result = {}
    for difficulty in (0, 1, 2):
        result[str(difficulty)] = {}
        for support in sorted(graph.SUPPORTS):
            catalog = catalogue(difficulty, support)
            result[str(difficulty)][str(support)] = {
                'total_identities': len(catalog), 'excluded_identities': len(catalog & blocked),
                'remaining_identities': len(catalog - blocked)}
    return result


def registration(snapshot, blocked, historical):
    for difficulty in graph.PRESETS:
        output = ROOT / 'var/results/modebench_level3_v2' / f'calibration_3b_graph_v7_d{difficulty}.json'
        if output.exists() or Path(str(output) + '.batches').exists():
            raise ValueError('v7 model work already exists before prospective registration')
    return {
        'schema': 'modebench_level3_graph_candidate_protocol_v7', 'candidate_revision': 'graph_v7',
        'created_at': datetime.now(timezone.utc).isoformat(), 'development_only': True,
        'generator_path': str(Path(graph.__file__).resolve()), 'generator_source_sha256': file_sha(graph.__file__),
        'materializer_path': str(Path(__file__).resolve()), 'materializer_source_sha256': file_sha(__file__),
        'pool_root': str(POOL_ROOT), 'development_draw_labels': [6328000, 6328001, 6328002, 6328003],
        'baseline_receipt_path': str(BASELINE), 'baseline_receipt_sha256': BASELINE_SHA,
        'failed_development_fit_path': str(FAILED_FIT), 'failed_development_fit_sha256': file_sha(FAILED_FIT),
        'rationale': 'Corrected v6 development fit failed. Two-hidden n5 cells frequently elicit three/four digits; n5 three-hidden cells parse better. Test sparse visible anchors and simple hidden forests while retaining a six-vertex coupled harder component. These are development-derived hypotheses, not guarantees of success.',
        'presets': graph.PRESETS, 'supports': sorted(graph.SUPPORTS), 'hidden_count_all_presets': 3,
        'independent_support_availabilities': graph.AVAILABILITIES,
        'support_five': 'Three-hidden path; endpoint visible anchors have distinct colors; center choices yield2+2+1=5.',
        'color_symmetry': 'Visible colors iid uniform1..3; vertex roles uniformly permuted; all rejection predicates invariant under any global color permutation. No preferred color or vertex label.',
        'proposal_laws': {'0': 'n5, independent minimal anchors; support5 anchored path; no visible-visible edge',
                         '1': 'n5 hidden forest: root availability times2 per tree edge; support5 anchored path; optional legal visible edge with fixed probability0.5',
                         '2': 'n5 same minimal anchors as0, require distinct visible colors and include their legal visible edge',
                         '3': 'n6, uniformly chosen3 hidden; iid visible colors; uniform edge count4..8 and uniform edge subset; reject unless exact support'},
        'development_seed_base': DEVELOPMENT_SEED, 'development_seed_rule': 'base +1000*difficulty',
        'per_cell_stream_seed_rule': 'SHA256(schema,seed,difficulty,support,accepted_cell_index,proposal)',
        'quota_prefix_stability': 'Independent fixed per-support streams; only exact support and identity exclusion rejection; no quota-dependent ordering, fallback, outcome selection, or hashseed retries.',
        'exclusions': {'historical_identities': historical, 'historical_and_candidate_identities': len(blocked),
                       'semantic_identity': 'original (domain,n,sorted labeled edges,partial color string); canonical prompt follows identity',
                       'include_all_prior_candidates': True, 'cross_new_pool_disjointness': True},
        'exact_n5_capacity_before_new_pools': exact_capacity(blocked),
        'capacity_audit': 'After all four new dev pools, independently check full384/128/128 quotas for every preset with globally disjoint ephemeral structural witnesses. Only audit hashes are retained; no confirmation dataset is generated or published.',
        'source_snapshot': snapshot,
        'information_boundary': {'candidate_model_outcomes_exist': False, 'confirmation_model_outcomes_used': False,
                                 'failed_corrected_development_used_for_structural_hypothesis': True,
                                 'fitting_or_gpu_submission_performed_here': False, 'treatment_training_started': False},
    }


def construct(blocked):
    pools, pool_records = {}, {}
    reference = materializer.reference_rows(DOMAIN, 'dev')
    target = materializer.modes(reference)
    current = set(blocked)
    for difficulty in graph.PRESETS:
        seed = DEVELOPMENT_SEED + 1000 * difficulty
        rows = graph.build_pool(DOMAIN, target, current, seed, 'level3_development_pool', difficulty, multiplier=1)
        checks = materializer.verify_rows(DOMAIN, rows, reference, target, current)
        witness_count = verify_witnesses(rows)
        checks.update({'exactly_three_hidden_vertices': True, 'original_canonical_support_verified': True})
        pools[difficulty] = rows
        pool_records[difficulty] = {'schema': 'modebench_level3_development_pool_v1', 'domain': DOMAIN,
            'difficulty': difficulty, 'seed': seed, 'rows': len(rows), 'rows_sha256': materializer.row_hash(rows),
            'checks': checks, 'support_histogram': dict(sorted(target.items())),
            'source_sha256': file_sha(graph.__file__), 'candidate_revision': 'graph_v7',
            'original_grader_witnesses': witness_count,
            'information_boundary': 'development candidates only; no confirmation model outcomes used'}
        current |= materializer.identity_set(DOMAIN, rows)
    after_pool_capacity = exact_capacity(current)
    runs = []
    for difficulty in graph.PRESETS:
        for split_index, split in enumerate(materializer.SPLITS):
            reference = materializer.reference_rows(DOMAIN, split)
            target = materializer.modes(reference)
            seed = CAPACITY_SEED + 1000 * difficulty + 10_000 * split_index
            rows = graph.build_pool(DOMAIN, target, current, seed, f'v7_capacity_{split}', difficulty, multiplier=1)
            checks = materializer.verify_rows(DOMAIN, rows, reference, target, current)
            larger = graph.build_pool(DOMAIN, Counter({support: count + 1 for support, count in target.items()}),
                                      current, seed, f'v7_capacity_{split}', difficulty, multiplier=1)
            for support, count in target.items():
                first = sorted((row for row in rows if row['answer_mode_count'] == support), key=lambda row: row['level3_cell_index'])
                extended = sorted((row for row in larger if row['answer_mode_count'] == support), key=lambda row: row['level3_cell_index'])
                if first != extended[:count]:
                    raise ValueError('quota-prefix invariance failed')
            witness_count = verify_witnesses(rows)
            current |= materializer.identity_set(DOMAIN, rows)
            runs.append({'difficulty': difficulty, 'split': split, 'seed': seed, 'rows': len(rows),
                         'rows_sha256': materializer.row_hash(rows), 'support_histogram': dict(sorted(target.items())),
                         'checks': {**checks, 'per_cell_quota_prefix_invariance': True}, 'original_grader_witnesses': witness_count})
            print(json.dumps({'capacity_difficulty': difficulty, 'split': split, 'rows': len(rows)}), flush=True)
    return pools, pool_records, {'exact_n5_capacity_after_all_four_dev_pools': after_pool_capacity,
             'capacity_runs': runs, 'capacity_rows_verified': sum(run['rows'] for run in runs),
             'all_capacity_rows_globally_disjoint_from_history_candidates_and_each_other': True,
             'capacity_rows_are_ephemeral_structural_witnesses_not_confirmation_data': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--materialize-development', action='store_true')
    args = parser.parse_args()
    if REGISTRATION.exists() or POOL_ROOT.exists():
        raise FileExistsError('fresh v7 registration and pool root required')
    snapshot = source_snapshot()
    blocked, historical = exclusions(snapshot)
    prospective = registration(snapshot, blocked, historical)
    if args.materialize_development:
        atomic_new(REGISTRATION, prospective)
    pools, records, capacity = construct(blocked)
    verify_snapshot(snapshot)
    if not args.materialize_development:
        print(json.dumps({'status': 'structural_preview_pass', 'development_rows': 512,
                          'capacity_rows': capacity['capacity_rows_verified']}))
        return
    registration_sha = file_sha(REGISTRATION)
    staging = Path(tempfile.mkdtemp(prefix='.' + POOL_ROOT.name + '.', dir=POOL_ROOT.parent))
    try:
        folder = staging / 'pools' / DOMAIN
        folder.mkdir(parents=True)
        for difficulty, rows in pools.items():
            path = folder / f'difficulty_{difficulty}.jsonl'
            with path.open('x') as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + '\n')
            records[difficulty].update({'candidate_protocol_path': str(REGISTRATION),
                                       'candidate_protocol_sha256': registration_sha})
            atomic_new(path.with_suffix('.identity.json'), records[difficulty])
        verify_snapshot(snapshot)
        if file_sha(REGISTRATION) != registration_sha:
            raise ValueError('prospective registration changed during construction')
        staging.rename(POOL_ROOT)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    report = {'schema': 'modebench_level3_graph_v7_structural_audit_v1',
        'registration_path': str(REGISTRATION), 'registration_sha256': registration_sha,
        'generator_source_sha256': file_sha(graph.__file__), 'pool_root': str(POOL_ROOT),
        'development_pools': records, **capacity, 'sealed_files_unchanged': True,
        'confirmation_outcomes_loaded': False, 'fitting_performed': False, 'gpu_submission_performed': False,
        'pool_files_sha256': {str(path): file_sha(path) for path in sorted(POOL_ROOT.rglob('*')) if path.is_file()}}
    atomic_new(ARTIFACTS / 'structural_audit.json', report)
    print(json.dumps({'status': 'registered_and_materialized', 'registration_sha256': registration_sha,
                      'development_rows': 512, 'capacity_rows_verified': capacity['capacity_rows_verified']}))


if __name__ == '__main__':
    main()
