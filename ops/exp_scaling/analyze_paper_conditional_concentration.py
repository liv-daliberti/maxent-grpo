#!/usr/bin/env python3
"""Retrospective conditional concentration from source-admitted saved keys.

Collection verifies raw source prefixes once and retains compact sufficient
samples. Analysis never rereads changing experiment directories. The protocol
is bound before concentration results are calculated.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ops'))
AUDIT = ROOT / 'paper/audits/conditional_concentration_20260911'
DEFAULT_CACHE = AUDIT / 'verified_samples.jsonl.gz'
DEFAULT_RESULT = ROOT / 'paper/results/conditional_concentration_20260911.json'
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')
SCALES = ('qwen05b', 'falcon1b', 'qwen3b')
T95_DF4 = 2.7764451051977987
METRICS = ('collision', 'mean8', 'pass8', 'distinct8', 'extra8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def collision(counts):
    """Unbiased collision U-statistic for fixed-policy iid correct labels."""
    values = list(counts.values())
    if any(type(n) is not int or n < 0 for n in values):
        raise ValueError('key counts must be nonnegative integers')
    n = sum(values)
    return sum(v * (v - 1) for v in values) / (n * (n - 1)) if n >= 2 else None


def prompt_metrics(draws):
    if not draws or any(len(d) != 8 for d in draws):
        raise ValueError('metrics require nonempty intact K8 draw groups')
    if any(k is not None and (not isinstance(k, str) or not k) for d in draws for k in d):
        raise ValueError('verified keys must be nonempty strings or null')
    counts = Counter(k for d in draws for k in d if k is not None)
    correct = sum(counts.values())
    pass8 = statistics.mean(any(k is not None for k in d) for d in draws)
    distinct8 = statistics.mean(len({k for k in d if k is not None}) for d in draws)
    return {'collision': collision(counts), 'correct': correct, 'total': 8 * len(draws),
            'mean8': correct / (8 * len(draws)), 'pass8': pass8,
            'distinct8': distinct8, 'extra8': distinct8 - pass8,
            'correct_pairs': correct * (correct - 1) // 2,
            'colliding_pairs': sum(n * (n - 1) // 2 for n in counts.values())}


def means(records):
    return {k: statistics.mean(r[k] for r in records) if records else None for k in METRICS}


def pooled(records):
    denominator = sum(r['correct_pairs'] for r in records)
    return sum(r['colliding_pairs'] for r in records) / denominator if denominator else None


def pair_seed(a, b, *, draws_a=(0, 1, 2, 3), draws_b=(0, 1, 2, 3), prompt_ids=None):
    """Equal-prompt B-minus-A contrast on explicitly shared eligibility.

    Inputs map exact prompt IDs to four intact verified-key draw groups.
    Common-seed sample selection is descriptive unless independence assumptions
    for count/label conditioning hold. No missing prompts are silently joined.
    """
    if set(a) != set(b):
        raise ValueError('paired conditions have different exact prompt populations')
    if not draws_a or not draws_b or any(i not in range(4) for i in (*draws_a, *draws_b)):
        raise ValueError('invalid selected draw groups')
    if len(set(draws_a)) != len(draws_a) or len(set(draws_b)) != len(draws_b):
        raise ValueError('draw indices must be unique within each condition')
    ids = set(a) if prompt_ids is None else set(prompt_ids)
    if not ids <= set(a):
        raise ValueError('requested sensitivity population is not in both conditions')
    ma = {x: prompt_metrics([a[x][i] for i in draws_a]) for x in sorted(ids)}
    mb = {x: prompt_metrics([b[x][i] for i in draws_b]) for x in sorted(ids)}
    eligible = [x for x in sorted(ids) if ma[x]['collision'] is not None and mb[x]['collision'] is not None]
    ar, br = [ma[x] for x in eligible], [mb[x] for x in eligible]
    am, bm = means(ar), means(br)
    full_a = {k: statistics.mean(r[k] for r in ma.values()) if ma else None for k in METRICS if k != 'collision'}
    full_b = {k: statistics.mean(r[k] for r in mb.values()) if mb else None for k in METRICS if k != 'collision'}
    return {'n_total': len(ids), 'n_eligible': len(eligible), 'eligible_ids': eligible,
            'coverage': len(eligible) / len(ids) if ids else None,
            'a': am, 'b': bm,
            'delta': {k: bm[k] - am[k] if eligible else None for k in METRICS},
            'full_a': full_a, 'full_b': full_b,
            'pooled_collision_a': pooled(ar), 'pooled_collision_b': pooled(br),
            'pooled_all_eligible_a': pooled([r for r in ma.values() if r['collision'] is not None]),
            'pooled_all_eligible_b': pooled([r for r in mb.values() if r['collision'] is not None]),
            'eligibility': {'both': len(eligible),
                'a_only': sum(ma[x]['collision'] is not None and mb[x]['collision'] is None for x in ids),
                'b_only': sum(ma[x]['collision'] is None and mb[x]['collision'] is not None for x in ids),
                'neither': sum(ma[x]['collision'] is None and mb[x]['collision'] is None for x in ids)},
            'draws_a': list(draws_a), 'draws_b': list(draws_b)}


def summarize_seeds(values, registered_n=5):
    if registered_n != 5:
        raise ValueError('the frozen analysis uses five registered seeds per complete block')
    if any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in values.values()):
        raise ValueError('seed estimates must be finite; undefined seeds must be explicit upstream')
    vals = list(values.values())
    avg = statistics.mean(vals) if vals else None
    interval = None
    if len(vals) == 5:
        half = T95_DF4 * statistics.stdev(vals) / math.sqrt(5)
        interval = [avg - half, avg + half]
    return {'n': len(vals), 'registered_n': 5, 'mean': avg, 'ci95': interval,
            'range': [min(vals), max(vals)] if vals else None,
            'values': {str(k): v for k, v in sorted(values.items())},
            'uncertainty': 'nominal paired Student-t over five measured seed estimates' if interval is not None else 'descriptive; incomplete five-seed block'}


def compact_cell(cell):
    result = {k: v for k, v in cell.items() if k != 'checkpoints'}
    result['checkpoints'] = {}
    for step, cp in cell['checkpoints'].items():
        if cp is None:
            result['checkpoints'][str(step)] = None
            continue
        try:
            prompts = {}
            for draw in cp['draws']:
                rows = []
                for p in draw['prompts']:
                    pm = prompt_metrics([p['verified_keys']])
                    rows.append(pm)
                    item = prompts.setdefault(p['prompt_id'], {'prompt_index': p['prompt_index'],
                        'draws': [], 'request_seeds_by_draw': [], 'option_ids_by_draw': []})
                    item['draws'].append(p['verified_keys'])
                    item['request_seeds_by_draw'].append(p['request_seeds_by_option'])
                    item['option_ids_by_draw'].append(p['option_ids'])
                for metric, original in [('pass8', 'any_correct_at_k'), ('distinct8', 'distinct_correct_modes_at_k'), ('mean8', 'mean_at_k')]:
                    measured = statistics.mean(row[metric] for row in rows)
                    if abs(measured - draw['observed_metrics'][original]) > 1e-12:
                        raise ValueError(f'{metric} reconstructed from verified keys differs from frozen draw')
            result['checkpoints'][str(step)] = {'prompts': prompts,
                'sampling_certificate': cp['sampling_certificate'],
                'origins': [{'draw_index': d['draw_index'], 'origins': d['origins'],
                             'raw_payload_sha256': d['raw_payload_sha256']} for d in cp['draws']],
                'primary_metrics_reconstructed': True}
        except (ValueError, KeyError, TypeError) as exc:
            result['checkpoints'][str(step)] = None
            result['sample_issues'].append({'kind': 'metric_reconstruction_failure', 'step': int(step), 'reason': str(exc)})
    result['before_after_available'] = all(result['checkpoints'].get(s) is not None for s in ('0', '3072'))
    return result


def collect(output, workers=4):
    from exp_scaling.load_paper_collision_samples import load_primary_manifest, iter_primary_sample_cells
    from exp_scaling.load_paper_grpo_collision_sources import load_grpo_manifest, iter_grpo_sample_cells
    if output.exists():
        raise ValueError('refusing to overwrite a frozen collection')
    binding = json.loads((AUDIT / 'analysis_plan_binding.json').read_text())
    if sha(AUDIT / 'analysis_plan.md') != binding['analysis_plan_sha256']:
        raise ValueError('frozen analysis protocol changed')
    primary = load_primary_manifest()
    grpo = load_grpo_manifest()
    metadata = {'record_kind': 'manifest', 'schema': 'paper-conditional-collision-samples-v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': binding,
        'primary': {k: v for k, v in primary.items() if k != 'snapshot'},
        'grpo': {k: v for k, v in grpo.items() if k not in ('cells', 'snapshot')},
        'code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT/'ops/exp_scaling/load_paper_collision_samples.py', ROOT/'ops/exp_scaling/load_paper_grpo_collision_sources.py']}}
    temp = output.with_suffix(output.suffix + '.partial')
    counts, available, issue_count = Counter(), Counter(), 0
    with gzip.open(temp, 'wt', encoding='utf-8') as f:
        f.write(dump(metadata) + '\n')
        streams = (iter_primary_sample_cells(primary, workers=workers), iter_grpo_sample_cells(grpo, workers=workers))
        for stream in streams:
            for raw in stream:
                cell = compact_cell(raw)
                f.write(dump({'record_kind': 'cell', **cell}) + '\n')
                counts[cell['method']] += 1
                for step, cp in cell['checkpoints'].items():
                    if cp is not None:
                        available[f"{cell['level']}/{cell['method']}/{step}"] += 1
                issue_count += len(cell['sample_issues'])
                if sum(counts.values()) % 25 == 0:
                    print(f'Collected {sum(counts.values())} cells; source/sample issues={issue_count}', flush=True)
    temp.replace(output)
    summary = {'status': 'collected', 'path': str(output.relative_to(ROOT)), 'sha256': sha(output),
        'cells_by_method': dict(counts), 'available_checkpoints': dict(available), 'sample_issue_count': issue_count}
    (AUDIT / 'collection_receipt.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


def stream_representatives(draws, parent_seeds):
    """First saved output per nominal V0 child stream, within one prompt.

    Returns every representative and duplicate-key disagreement diagnostics.
    Selection uses only indices/seeds and never labels or correctness.
    """
    if len(draws) != len(parent_seeds) or any(type(s) is not int for s in parent_seeds):
        raise ValueError('every draw must have a recorded integer parent seed')
    if any(len(d) != 8 for d in draws):
        raise ValueError('certified V0 mapping here is for original n=8 groups')
    representatives, occurrences = {}, Counter()
    disagreements = 0
    for d, keys in enumerate(draws):
        for i, key in enumerate(keys):
            sid = parent_seeds[d] + i
            occurrences[sid] += 1
            if sid not in representatives:
                representatives[sid] = {'stream_id': sid, 'draw_index': d, 'option_index': i, 'key': key}
            elif representatives[sid]['key'] != key:
                disagreements += 1
    return {'representatives': [representatives[s] for s in sorted(representatives)],
            'n_saved': sum(len(d) for d in draws), 'n_streams': len(representatives),
            'repeated_occurrences': sum(n - 1 for n in occurrences.values()),
            'duplicate_key_disagreements': disagreements}


def split_stream_ids(a, b, orientation=0):
    """Outcome-independent disjoint subsets of common nominal streams."""
    if orientation not in (0, 1):
        raise ValueError('orientation must be 0 or 1')
    common = sorted(set(a) & set(b))
    midpoint = len(common) // 2
    lower, upper = common[:midpoint], common[midpoint:]
    return (lower, upper) if orientation == 0 else (upper, lower)


def prepare_checkpoint(cp):
    """Bind compact prompts to certified neutral V0 sampling-stream mapping."""
    if cp is None:
        return None
    cert = cp['sampling_certificate']
    seeds = cert['draw_seeds']
    if (len(seeds) != 4 or not all(type(s) is int for s in seeds)
            or not cert['same_prompt_identities'] or not cert['same_recorded_decoder_fields']):
        raise ValueError('checkpoint lacks same-prompt recorded sampling certificate')
    prepared = {}
    for pid, prompt in cp['prompts'].items():
        # This retrospective population uses neutral requests with one n=8 group
        # per prompt; canonical action masking uses that same sampling branch.
        if any(ids not in (None, [], [None] * 8) for ids in prompt['option_ids_by_draw']):
            raise ValueError('option-stratified requests lack a certified stream mapping')
        if len(prompt['request_seeds_by_draw']) != len(seeds) or any(
                seeds_by_option not in (None, [], {}, [parent])
                for parent, seeds_by_option in zip(seeds, prompt['request_seeds_by_draw'])):
            raise ValueError('per-option seeded requests need a distinct certified mapping')
        selected = stream_representatives(prompt['draws'], seeds)
        prepared[pid] = {'draws': prompt['draws'], 'streams': {r['stream_id']: r['key'] for r in selected['representatives']},
            'representatives': selected['representatives'], 'stream_audit': {k: v for k, v in selected.items() if k != 'representatives'},
            'primary_metrics': prompt_metrics(prompt['draws'])}
    return prepared


def selected_metrics(prompt, stream_ids=None):
    streams = prompt['streams']
    ids = sorted(streams) if stream_ids is None else stream_ids
    if len(ids) != len(set(ids)):
        raise ValueError('selected stream IDs must be unique')
    if not set(ids) <= set(streams):
        raise ValueError('selected stream absent from condition')
    keys = [streams[i] for i in ids]
    counts = Counter(k for k in keys if k is not None)
    n = sum(counts.values())
    return {'collision': collision(counts), 'selected_correct': n, 'selected_total': len(keys),
        'selected_correct_fraction': n / len(keys) if keys else None,
        'correct_pairs': n * (n - 1) // 2,
        'colliding_pairs': sum(v * (v - 1) // 2 for v in counts.values()),
        **{k: prompt['primary_metrics'][k] for k in METRICS if k != 'collision'}}


def paired_streams(a, b, *, orientation=None, prompt_ids=None):
    if set(a) != set(b):
        raise ValueError('paired conditions have different exact prompt populations')
    ids = sorted(set(a) if prompt_ids is None else set(prompt_ids))
    if not set(ids) <= set(a):
        raise ValueError('sensitivity population absent from paired conditions')
    ma, mb, selections = {}, {}, {}
    for x in ids:
        if orientation is None:
            ia, ib = sorted(a[x]['streams']), sorted(b[x]['streams'])
        else:
            ia, ib = split_stream_ids(a[x]['streams'], b[x]['streams'], orientation)
        ma[x], mb[x] = selected_metrics(a[x], ia), selected_metrics(b[x], ib)
        selections[x] = {'a_streams': ia, 'b_streams': ib, 'overlap': len(set(ia) & set(ib))}
    eligible = [x for x in ids if ma[x]['collision'] is not None and mb[x]['collision'] is not None]
    ar, br = [ma[x] for x in eligible], [mb[x] for x in eligible]
    am, bm = means(ar), means(br)
    return {'n_total': len(ids), 'n_eligible': len(eligible), 'eligible_ids': eligible,
        'coverage': len(eligible) / len(ids) if ids else None,
        'a': am, 'b': bm, 'delta': {k: bm[k]-am[k] if eligible else None for k in METRICS},
        'selected_correct_a': statistics.mean(r['selected_correct'] for r in ar) if ar else None,
        'selected_correct_b': statistics.mean(r['selected_correct'] for r in br) if br else None,
        'selected_total_a': sorted({r['selected_total'] for r in ma.values()}),
        'selected_total_b': sorted({r['selected_total'] for r in mb.values()}),
        'full_a': {k: statistics.mean(a[x]['primary_metrics'][k] for x in ids) if ids else None for k in METRICS if k != 'collision'},
        'full_b': {k: statistics.mean(b[x]['primary_metrics'][k] for x in ids) if ids else None for k in METRICS if k != 'collision'},
        'pooled_collision_a': pooled(ar), 'pooled_collision_b': pooled(br),
        'pooled_all_eligible_a': pooled([r for r in ma.values() if r['collision'] is not None]),
        'pooled_all_eligible_b': pooled([r for r in mb.values() if r['collision'] is not None]),
        'eligibility': {'both': len(eligible),
            'a_only': sum(ma[x]['collision'] is not None and mb[x]['collision'] is None for x in ids),
            'b_only': sum(ma[x]['collision'] is None and mb[x]['collision'] is not None for x in ids),
            'neither': sum(ma[x]['collision'] is None and mb[x]['collision'] is None for x in ids)},
        'stream_selection': {'orientation': orientation, 'disjoint_in_every_prompt': all(s['overlap'] == 0 for s in selections.values()),
            'shared_stream_counts': sorted({s['overlap'] for s in selections.values()})}}


def draw_sensitivity(a, b):
    raw_a = {x: p['draws'] for x, p in a.items()}
    raw_b = {x: p['draws'] for x, p in b.items()}
    if set(a) != set(b):
        raise ValueError('intact draw comparison has different prompt identities')
    individual = [pair_seed(raw_a, raw_b, draws_a=(d,), draws_b=(d,)) for d in range(4)]
    eligible, per_prompt = [], []
    for x in sorted(a):
        paired = []
        for d in range(4):
            ma, mb = prompt_metrics([raw_a[x][d]]), prompt_metrics([raw_b[x][d]])
            if ma['collision'] is not None and mb['collision'] is not None:
                paired.append((ma, mb))
        if paired:
            eligible.append(x)
            per_prompt.append({'prompt_id': x, 'eligible_draws': len(paired),
                'a': statistics.mean(v[0]['collision'] for v in paired),
                'b': statistics.mean(v[1]['collision'] for v in paired)})
    return {'individual_draws': individual, 'eligible_ids': eligible, 'n_total': len(a), 'n_eligible': len(eligible),
        'eligible_draw_counts': dict(Counter(r['eligible_draws'] for r in per_prompt)),
        'delta': statistics.mean(r['b'] - r['a'] for r in per_prompt) if per_prompt else None,
        'distinct_streams_same_prompt_set': paired_streams(a, b, prompt_ids=eligible),
        'naive32_same_prompt_set': pair_seed(raw_a, raw_b, prompt_ids=eligible)}


def four_condition_change(a0, a1, b0, b1):
    if not (set(a0) == set(a1) == set(b0) == set(b1)):
        raise ValueError('change difference has different prompt populations')
    records = []
    for x in sorted(a0):
        vals = [selected_metrics(v[x])['collision'] for v in (a0, a1, b0, b1)]
        if all(v is not None for v in vals):
            records.append({'prompt_id': x, 'a0': vals[0], 'a1': vals[1], 'b0': vals[2], 'b1': vals[3],
                            'delta': (vals[3]-vals[2]) - (vals[1]-vals[0])})
    return {'n_total': len(a0), 'n_eligible': len(records), 'eligible_ids': [r['prompt_id'] for r in records],
            'delta': statistics.mean(r['delta'] for r in records) if records else None}


def make_block(kind, key, method, seeds, cells, prepared):
    level, scale, domain = key
    block = {'kind': kind, 'level': level, 'scale': scale, 'domain': domain, 'method': method,
             'admitted_seeds': sorted(seeds), 'registered_n': 5, 'per_seed': {}, 'issues': []}
    endpoint_pairs = {}
    for seed in sorted(seeds):
        akey = (level, scale, domain, method, seed)
        bmethod = 'replay_' + method if kind == 'replay_effect' else method
        bkey = (level, scale, domain, bmethod, seed)
        astep, bstep = ('3072', '3072') if kind == 'replay_effect' else ('0', '3072')
        a, b = prepared.get((akey, astep)), prepared.get((bkey, bstep))
        if a is None or b is None:
            block['issues'].append({'seed': seed, 'kind': 'endpoint_unavailable', 'a_available': a is not None, 'b_available': b is not None})
            continue
        try:
            endpoint_pairs[seed] = (a, b)
            result = {'distinct_streams': paired_streams(a,b),
                      'orientation0': paired_streams(a,b,orientation=0),
                      'orientation1': paired_streams(a,b,orientation=1),
                      'intact_k8': draw_sensitivity(a,b),
                      'naive_reused_streams_32': pair_seed({x:p['draws'] for x,p in a.items()}, {x:p['draws'] for x,p in b.items()})}
            if kind == 'replay_effect':
                a0, b0 = prepared.get((akey,'0')), prepared.get((bkey,'0'))
                if a0 is not None and b0 is not None:
                    result['change_difference'] = four_condition_change(a0,a,b0,b)
            block['per_seed'][str(seed)] = result
        except ValueError as exc:
            endpoint_pairs.pop(seed,None)
            block['issues'].append({'seed': seed, 'kind': 'comparison_integrity_failure', 'reason': str(exc)})
    summaries = {}
    for name in ('distinct_streams','orientation0','orientation1','naive_reused_streams_32'):
        records = {int(s):v[name] for s,v in block['per_seed'].items()}
        summary = summarize_seeds({s:r['delta']['collision'] for s,r in records.items() if r['delta']['collision'] is not None})
        summary['a_mean'] = statistics.mean(r['a']['collision'] for r in records.values() if r['a']['collision'] is not None) if summary['n'] else None
        summary['b_mean'] = statistics.mean(r['b']['collision'] for r in records.values() if r['b']['collision'] is not None) if summary['n'] else None
        summary['eligible_counts'] = {str(s): r['n_eligible'] for s,r in records.items()}
        summary['coverage_range'] = [min(r['coverage'] for r in records.values()),max(r['coverage'] for r in records.values())] if records else None
        summary['original_metrics_on_eligible'] = {k:summarize_seeds({s:r['delta'][k] for s,r in records.items() if r['delta'][k] is not None}) for k in METRICS if k!='collision'}
        summaries[name] = summary
    cross = {int(s):statistics.mean(v[n]['delta']['collision'] for n in ('orientation0','orientation1'))
             for s,v in block['per_seed'].items() if all(v[n]['delta']['collision'] is not None for n in ('orientation0','orientation1'))}
    summaries['orientation_mean_descriptive'] = summarize_seeds(cross)
    summaries['intact_k8'] = summarize_seeds({int(s):v['intact_k8']['delta'] for s,v in block['per_seed'].items() if v['intact_k8']['delta'] is not None})
    summaries['first_k8'] = summarize_seeds({int(s):v['intact_k8']['individual_draws'][0]['delta']['collision'] for s,v in block['per_seed'].items() if v['intact_k8']['individual_draws'][0]['delta']['collision'] is not None})
    summaries['change_difference'] = summarize_seeds({int(s):v['change_difference']['delta'] for s,v in block['per_seed'].items() if v.get('change_difference',{}).get('delta') is not None})
    block['summaries'] = summaries
    fixed = {}
    for name, orientation in [('distinct_streams',None),('orientation0',0),('orientation1',1)]:
        if set(endpoint_pairs) != set(seeds):
            fixed[name] = {'available': False, 'reason': 'not every originally admitted seed has a paired sample endpoint'}
            continue
        common = set.intersection(*(set(block['per_seed'][str(s)][name]['eligible_ids']) for s in seeds)) if seeds else set()
        records = {s: paired_streams(*endpoint_pairs[s],orientation=orientation,prompt_ids=common) for s in seeds}
        fixed[name] = {'available': bool(common), 'eligible_ids': sorted(common), 'n_eligible': len(common),
                      'summary': summarize_seeds({s:r['delta']['collision'] for s,r in records.items() if r['delta']['collision'] is not None}),
                      'per_seed': {str(s):r for s,r in records.items()}}
    block['fixed_across_seed_population'] = fixed
    return block


def analyze(cache, output, receipt_path=None, cohort_binding=None):
    binding = json.loads((AUDIT/'analysis_plan_binding.json').read_text())
    amendment = json.loads((AUDIT/'analysis_plan_amendment_1_binding.json').read_text())
    if sha(AUDIT/'analysis_plan.md') != binding['analysis_plan_sha256'] or sha(AUDIT/'analysis_plan_amendment_1.md') != amendment['sha256']:
        raise ValueError('analysis protocol or stream amendment changed')
    with gzip.open(cache,'rt',encoding='utf-8') as f:
        metadata = json.loads(next(f)); rows = [json.loads(line) for line in f]
    receipt = json.loads((receipt_path or AUDIT/'collection_receipt.json').read_text())
    extension = json.loads(cohort_binding.read_text()) if cohort_binding else None
    if extension:
        if receipt.get('cohort_extension') != extension or sha(cohort_binding) != receipt['binding_sha256']:
            raise ValueError('extended cache is not bound to the declared cohort amendment')
        if sha(ROOT / extension['amendment_path']) != extension['amendment_sha256']:
            raise ValueError('cohort amendment changed')
        for entry in ('original_snapshot','completed_snapshot','original_cache','original_collection_receipt','preserved_initial_results'):
            record = extension[entry]
            if sha(ROOT / record['path']) != record['sha256']:
                raise ValueError(f'cohort-extension source changed: {entry}')
        certificate = receipt['source_stream_certificate']
        if sha(ROOT / certificate['path']) != certificate['sha256']:
            raise ValueError('completed-cohort stream certificate changed')
    elif receipt.get('cohort_extension'):
        raise ValueError('extended sample cache requires its cohort binding')
    if sha(cache) != receipt['sha256']:
        raise ValueError('compact frozen sample cache changed')
    cells = {tuple(r[k] for k in ('level','scale','domain','method','seed')):r for r in rows}
    if len(cells)!=len(rows):raise ValueError('duplicate collected cell')
    prepared, issues, stream_counts, repeats, disagreements = {}, [], Counter(), 0, 0
    for key, cell in cells.items():
        for step, cp in cell['checkpoints'].items():
            try:
                converted = prepare_checkpoint(cp)
                prepared[key,step] = converted
                if converted is not None:
                    for p in converted.values():
                        audit=p['stream_audit'];stream_counts[audit['n_streams']]+=1
                        repeats+=audit['repeated_occurrences'];disagreements+=audit['duplicate_key_disagreements']
            except ValueError as exc:
                prepared[key,step] = None
                issues.append({'cell':list(key),'step':step,'reason':str(exc)})
    blocks=[]
    for level in ('level1','level2'):
        for scale in SCALES:
            for domain in DOMAINS:
                for method in ('drgrpo','grpo','maxrl','replay_drgrpo','replay_maxrl'):
                    seeds=sorted(k[4] for k in cells if k[:4]==(level,scale,domain,method))
                    if seeds:blocks.append(make_block('before_after',(level,scale,domain),method,seeds,cells,prepared))
    for panel in metadata['primary']['cohorts']:
        for objective,seeds in panel['paired_cohorts'].items():
            blocks.append(make_block('replay_effect',(panel['level'],panel['scale'],panel['domain']),objective,seeds,cells,prepared))
    result={'schema':'paper-conditional-concentration-v1','status':'analyzed','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'protocol':binding,'stream_amendment':amendment,'cohort_extension':extension,'cache':receipt,'analysis_code_sha256':sha(Path(__file__)),
        'stream_source_audit': {'certificates': {str(p.relative_to(ROOT)):sha(p) for p in (AUDIT/'stream_source_audit.json', AUDIT/'stream_source_audit_grpo_extension.json')}, 'historical_source_limit':'Archived actor source is available for 150 of 400 primary cells and 0 of 75 GRPO cells. Historical vLLM dependency identity is not cryptographically established for every run.', 'nominal_child_mapping':'parent seed + output index in verified V0 n=8 evaluation',
            'representative':'earliest draw/output position per prompt and nominal child stream',
            'stream_counts_per_prompt_checkpoint':dict(stream_counts),'repeated_saved_occurrences':repeats,
            'duplicate_verified_key_disagreements':disagreements,'unsupported_checkpoint_mappings':issues,
            'scope':'Distinct nominal RNG streams under the audited V0 mapping. Historical runtime source is incomplete; this mapping and fixed-law independent sampling are assumptions, not certified statistical independence. The same stream IDs also repeat across prompts.'},
        'source_metadata':metadata,'cell_availability':[{'cell':list(k),'checkpoints':{s:prepared.get((k,s)) is not None for s in c['checkpoints']},'sample_issues':c['sample_issues']} for k,c in cells.items()],
        'blocks':blocks,
        'interpretation':['C increases mean greater conditional concentration on the stated observable population; not literal extinction.',
          'Primary distinct-stream paired estimates share RNG across conditions and are descriptive after joint eligibility.',
          'Two disjoint nominal-stream orientations retain separate eligibility; their complete-case mean is descriptive.',
          'Stream IDs repeat across prompts. No iid-prompt or unbiased random-eligible-mean claim is made; promptwise collision unbiasedness is conditional on fixed-law iid correct labels.',
          'Original K8 metrics retain all saved responses; reused streams are not 32 independent observations.',
          'Intervals quantify measured seed variability, are nominal/unadjusted, and do not certify unseen support.']}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'analyzed','output':str(output),'blocks':len(blocks),'stream_counts':dict(stream_counts),'unsupported_mappings':len(issues)},indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['collect', 'analyze'])
    parser.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--output', type=Path, default=DEFAULT_RESULT)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--receipt', type=Path, help='explicit receipt for a separately frozen sample cache')
    parser.add_argument('--cohort-binding', type=Path, help='binding for a declared current-census extension')
    args = parser.parse_args()
    if args.action == 'collect':
        collect(args.cache, args.workers)
    else:
        analyze(args.cache, args.output, args.receipt, args.cohort_binding)


if __name__ == '__main__':
    main()
