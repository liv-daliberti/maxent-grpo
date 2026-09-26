"""Independent point-estimate counts from audited draws, without analyzer imports."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/modebench_discovery_curves_20260911'
AUDIT = Path(__file__).resolve().parent
GRID = (1, 2, 4, 8, 16, 32, 64)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def estimate(samples):
    assert set(samples) == set(range(64))
    keys = [samples[i] for i in range(64)]
    counts = Counter(key for key in keys if key is not None)
    c = sum(counts.values())
    output = {}
    for k in GRID:
        total_subsets = math.comb(64, k)
        missed_success = math.comb(64 - c, k) if 64 - c >= k else 0
        p = (total_subsets - missed_success) / total_subsets
        d = sum((total_subsets - (math.comb(64 - n, k) if 64 - n >= k else 0)) / total_subsets for n in counts.values())
        for metric, value in [('pass', p), ('distinct', d), ('breadth', d - p)]:
            output[f'rarefaction/{metric}/k{k}'] = value
        distinct_prefix = len({key for key in keys[:k] if key is not None})
        for metric, value in [('pass', float(bool(distinct_prefix))), ('distinct', distinct_prefix), ('breadth', distinct_prefix - bool(distinct_prefix))]:
            output[f'prefix/{metric}/k{k}'] = value
    assert output['rarefaction/pass/k64'] == float(c > 0)
    assert output['rarefaction/distinct/k64'] == len(counts)
    return output, math.comb(c, 2), sum(math.comb(n, 2) for n in counts.values())


def close(a, b, label):
    assert (a is None and b is None) or (a is not None and b is not None and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)), (label, a, b)


def main():
    report_path = BASE / 'analysis_local_complete/analysis.json'
    report = json.loads(report_path.read_text())
    assert report['status'] == 'complete' and report['scope']['included_panels'] == ['local']
    assert len(report['models']) == 25 and report['inventory']['expected_draws'] == report['inventory']['finalized_draws'] == 110592
    checked = []; sources = []; response_count = 0; comparisons = 0
    for model in report['models']:
        binding = model['conditions']['original']['sources']['discovery_grades.jsonl']
        path = Path(binding['path']); assert sha(path) == binding['sha256']; sources.append(binding)
        groups = defaultdict(dict)
        for line in path.read_text().splitlines():
            row = json.loads(line); response_count += 1
            for label, field in [('strict', 'strict'), ('normalized_secondary', 'normalization')]:
                grade = row[field]; assert grade['verified'] == (grade['canonical_key'] is not None)
                key = json.dumps(grade['canonical_key'], sort_keys=True, separators=(',', ':')) if grade['verified'] else None
                group = (label, row['arm'], f"level{row['level']}/{row['domain']}", row['row_index'])
                assert row['sample_index'] not in groups[group]
                groups[group][row['sample_index']] = key
        cells = defaultdict(list)
        for (grading, arm, cell, row_index), samples in groups.items():
            cells[grading, arm, cell].append(estimate(samples))
        for (grading, arm, cell), prompts in sorted(cells.items()):
            assert len(prompts) == 16
            values = {name: statistics.mean(item[0][name] for item in prompts) for name in prompts[0][0]}
            pairs = sum(item[1] for item in prompts); collisions = sum(item[2] for item in prompts)
            values['collision_own/observed'] = collisions / pairs if pairs else None
            reference = model['analyses'][grading]['cells'][cell][arm]
            for name, value in values.items():
                close(value, reference[name]['estimate'], (model['model_id'], grading, arm, cell, name)); comparisons += 1
            checked.append({'model_id': model['model_id'], 'grading': grading, 'arm': arm, 'cell': cell,
                            'prompt_count': 16, 'correct_pairs': pairs, 'colliding_correct_pairs': collisions, 'point_estimates': values})
    assert response_count == 110592
    lookup = {(x['model_id'], x['grading'], x['arm'], x['cell']): x for x in checked}
    for group in report['local_seed_groups'].values():
        for grading, data in group['analyses'].items():
            for cell, values in data['cells'].items():
                for arm in ('original', 'neutral'):
                    seeds = [lookup[model, grading, arm, cell]['point_estimates'] for model in group['checkpoint_ids']]
                    expected = values['seed_mean_fixed_seed_prompt_ci'][arm]
                    for name in seeds[0]:
                        individual = [x[name] for x in seeds]
                        value = None if None in individual else statistics.mean(individual)
                        close(value, expected[name]['estimate'], (group['training_method'], grading, arm, cell, name)); comparisons += 1
    receipt = {'schema': 'discovery-independent-point-estimate-review-v1', 'status': 'pass',
               'at_utc': datetime.now(timezone.utc).isoformat(), 'report': {'path': str(report_path), 'sha256': sha(report_path)},
               'reviewer_source': {'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__))},
               'audited_responses': response_count, 'cell_arm_grading_records': len(checked), 'point_estimates_compared': comparisons,
               'method': 'Direct canonical-key sets, integer subset counts, ordered prefixes, and correct-pair counts; no analyzer imports. Equal-seed summaries checked separately. Bootstrap intervals are authenticated by the publication checker, not recomputed here.',
               'grade_sources': sources, 'observations': checked}
    output = AUDIT / 'observed_endpoint_review.json'; output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('grade_sources', 'observations')}, indent=2))

if __name__ == '__main__':
    main()
