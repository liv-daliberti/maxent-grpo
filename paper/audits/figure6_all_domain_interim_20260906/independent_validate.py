#!/usr/bin/env python3
"""Independent frozen-source validation of Figure6 all-domain selection."""
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'paper/audits/figure6_all_domain_interim_20260906'
selected_path=ROOT/'paper/results/modebench_level_comparison_snapshot.json'
selected_bytes=selected_path.read_bytes();selected=json.loads(selected_bytes)
summary_path=OUT/'selection_summary.json'; summary_bytes=summary_path.read_bytes();summary=json.loads(summary_bytes)
LEVELS=('level1','level2'); METHODS=('drgrpo','replay_drgrpo','maxrl','replay_maxrl')
DOMAINS=('graph_coloring','countdown','python_factors','mathir','pantry_plan')
METRICS={'pass8':'any_correct_at_k','distinct8':'distinct_correct_modes_at_k'}
index={};snapshots={}
for level in LEVELS:
    source=selected['input_snapshots'][level];path=ROOT/source['path'];raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==source['sha256'], ('snapshot hash',level)
    snapshot=json.loads(raw);snapshots[level]=snapshot
    for cell in snapshot['cells']:
        key=(level,cell['domain'],cell.get('method',cell.get('arm')),cell['seed'])
        assert key not in index, ('duplicate source cell',key)
        index[key]=cell
assert len(index)==200
availability={}
for cell in selected['availability']:
    key=(cell['level'],cell['domain'],cell['method'],cell['seed'])
    assert key not in availability,('duplicate availability cell',key)
    source=index[key]
    for field in ('complete_steps','invalid_or_conflicted_steps','run_dir','source_files'):
        assert cell[field]==source[field],('availability mismatch',key,field)
    availability[key]=cell
assert set(availability)==set(index)

expected_steps={}
for domain in DOMAINS:
    for seed in range(43,48):
        shared=set.intersection(*(set(index[(level,domain,method,seed)]['complete_steps'])
                                  for level in LEVELS for method in METHODS))
        if shared:
            expected_steps[(domain,seed)]=max(shared)
    assert any(key[0]==domain for key in expected_steps),('missing domain',domain)
expected_draw_keys={(level,domain,method,seed,step,draw)
                    for (domain,seed),step in expected_steps.items()
                    for level in LEVELS for method in METHODS for draw in range(4)}


def verify_evaluations(records, expected):
    observed={}
    for record in records:
        key=tuple(record[k] for k in ('level','domain','method','seed','step','draw_index'))
        assert key not in observed,('duplicate selected draw',key)
        assert key in expected,('unexpected selected draw',key)
        level,domain,method,seed,step,draw=key
        cell=index[(level,domain,method,seed)];checkpoint=cell['complete_checkpoints'][str(step)]
        assert checkpoint['draw_count']==4
        originals=[r for r in checkpoint['draws'] if r['draw_index']==draw]
        assert len(originals)==1
        original=originals[0]
        assert record['metrics']==original['metrics'],('metric drift',key)
        assert record['evaluation_metadata']==original['metadata'],('metadata drift',key)
        assert record['origins']==original['origins'],('origin drift',key)
        assert record['evaluation_kind']=='fixed_seed_sampled_k_neutral'
        assert record['sample_count']==8
        assert original['metadata']['sample_count']==8
        assert original['metadata']['temperature']==1.0
        assert original['metadata']['prompt_count']==128
        files={r['path']:r for r in cell['source_files']}
        assert record['origins']
        for origin in record['origins']:
            assert origin['path'] in files,('origin not in hashed source',key)
            assert 1<=origin['line']<=files[origin['path']]['line_count']
            assert len(files[origin['path']]['sha256_read_prefix'])==64
        for field in METRICS.values():assert math.isfinite(record['metrics'][field])
        observed[key]=record
    assert set(observed)==expected,('missing draws',len(expected-set(observed)))
    return observed

observed=verify_evaluations(selected['evaluations'],expected_draw_keys)
terminal_expected={(level,domain,method,seed,3072,draw)
                   for (level,domain,method,seed),cell in index.items()
                   if level=='level2' and 3072 in cell['complete_steps'] for draw in range(4)}
terminal=verify_evaluations(selected['level2_terminal_evaluations'],terminal_expected)
domain_means={}
for domain in DOMAINS:
    chosen={seed:step for (d,seed),step in expected_steps.items() if d==domain}
    coverage=summary['coverage_by_domain'][domain]
    assert coverage['n']==len(chosen)
    assert coverage['seeds']==sorted(chosen)
    assert coverage['steps_by_seed']=={str(seed):chosen[seed] for seed in sorted(chosen)}
    assert coverage['initial_checkpoint_only']==all(step==0 for step in chosen.values())
    assert coverage['initial_checkpoint_seeds']==[seed for seed,step in sorted(chosen.items()) if step==0]
    domain_means[domain]={}
    for level in LEVELS:
        domain_means[domain][level]={}
        for method in METHODS:
            means={}
            for metric,field in METRICS.items():
                seed_means=[statistics.fmean(observed[(level,domain,method,seed,step,draw)]['metrics'][field]
                                            for draw in range(4)) for seed,step in chosen.items()]
                means[metric]=statistics.fmean(seed_means)
                assert math.isclose(means[metric],summary['domain_means'][domain][level][method][metric],rel_tol=0,abs_tol=1e-12),('domain mean drift',domain,level,method,metric)
            domain_means[domain][level][method]=means
for level in LEVELS:
    for method in METHODS:
        for metric in METRICS:
            mean=statistics.fmean(domain_means[domain][level][method][metric] for domain in DOMAINS)
            assert math.isclose(mean,summary['means'][level][method][metric],rel_tol=0,abs_tol=1e-12),('equal domain mean drift',level,method,metric)
assert summary['eligible_domain_seed_cells']==len(expected_steps)
assert selected_path.read_bytes()==selected_bytes, 'selected snapshot changed during validation'
assert summary_path.read_bytes()==summary_bytes, 'summary changed during validation'
result={'status':'passed','verified_utc':datetime.now(timezone.utc).isoformat(),
        'selected_snapshot':str(selected_path.relative_to(ROOT)),'selected_snapshot_sha256':hashlib.sha256(selected_bytes).hexdigest(),
        'selection_summary_sha256':hashlib.sha256(summary_bytes).hexdigest(),
        'source_snapshots':selected['input_snapshots'],'availability_cells_verified':len(index),
        'domain_seed_pairs_verified':len(expected_steps),'selected_draws_verified':len(observed),
        'separate_level2_terminal_draws_verified':len(terminal),
        'coverage_by_domain':summary['coverage_by_domain'],
        'checks':['all200availability records exactly match frozen inventories',
                  'allqualifying domain/seed pairs included at latest all8-series exact shared step',
                  'every selected metric payload, metadata and origin exactly matches admitted source draw',
                  'allsource prefixes have hashes and origin line numbers within frozen prefix',
                  'K8,4draws,temperature1,128prompts verified for every contribution',
                  'separate terminal draw inventory complete and exact',
                  'withinseed draw means, withindomain seed means, equal five-domain means independently recomputed',
                  'initial-only Pantryseed45 correctly recorded, no other initial contributions',
                  'selection/source integrity validation used frozen records only; live logs not reread']}
(OUT/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('status','availability_cells_verified','domain_seed_pairs_verified','selected_draws_verified','separate_level2_terminal_draws_verified','selected_snapshot_sha256')}))
