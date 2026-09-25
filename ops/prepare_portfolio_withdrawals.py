#!/usr/bin/env python3
"""Freeze the five-domain withdrawal protocol before any outcome is read.

The evaluation prompts, their certified supports and every withdrawn option are
fixed here. The analyzer may then only read saved outcomes; it can neither add
an option nor drop a prompt. Reading is scoped: the source list comes from the
frozen key container rather than from a directory walk, and each source file is
opened once for its first line, which already carries all 128 prompt
specifications for that cell.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'ops') not in sys.path:
    sys.path.insert(0, str(ROOT / 'ops'))

from followup_metrics import atomic_new, file_sha, sha  # noqa: E402
import portfolio_withdrawals as pw  # noqa: E402

CONTAINER = ROOT / 'var/artifacts/mode_diversity_curves/verified_keys_by_step.jsonl.gz'
CURVES = ROOT / 'paper/results/mode_diversity_curves.json'
BASE = ROOT / 'artifacts/modebench_portfolio_withdrawals_20260917'
CODE = ('ops/portfolio_withdrawals.py', 'ops/prepare_portfolio_withdrawals.py',
        'ops/make_pantry_plan_mode_data.py', 'ops/make_modebench_data.py',
        'src/oat_drgrpo/math_grader.py', 'src/oat_drgrpo/mathir.py',
        'src/oat_drgrpo/pantry_plan.py')
#: Design record. The withdrawal families, the input-only option order and the
#: exclusion-only certificate were fixed before any arm was compared. The
#: analyzer's estimator choices were not: a prototype had already shown the raw
#: survival contrast when the fixed-budget estimator and the binding-option
#: restriction were adopted. Both are therefore reported alongside the
#: unrestricted raw contrast so no variant is selected after the fact.
DISCLOSURE = {
    'fixed_before_any_contrast': ['withdrawal family per domain', 'option enumeration order',
                                  'exclusion-only feasibility certificate',
                                  'terminal-endpoint cohort', 'survival definition'],
    'chosen_with_the_raw_contrast_visible': ['fixed verified-draw budget estimator',
                                             'restriction to binding options'],
    'mitigation': 'Raw, all-feasible and binding variants at budgets 1, 2 and 4 are all reported.',
}


def sources():
    """Every cell the frozen key container registers, with its source file."""
    records = []
    with gzip.open(CONTAINER, 'rt') as handle:
        for line in handle:
            record = json.loads(line)
            if record.get('record_kind') != 'source':
                continue
            records.append({k: record[k] for k in ('level', 'scale', 'domain', 'method', 'seed', 'path')})
    return records


def first_line_specs(path):
    """Prompt references from one evaluation file, ordered by prompt index."""
    with open(path) as handle:
        row = json.loads(handle.readline())
    prompts = sorted(row['prompts'], key=lambda p: p['prompt_index'])
    if [p['prompt_index'] for p in prompts] != list(range(len(prompts))):
        raise ValueError('prompt indices are not contiguous: ' + path)
    return [p['reference'] for p in prompts]


def _digest_one(path):
    references = first_line_specs(path)
    return path, sha(references), len(references)


def _row(job):
    level, domain, index, reference = job
    row = pw.prompt_row(domain, json.loads(reference))
    row.update({'level': level, 'domain': domain, 'prompt_index': index})
    return row


def prepare(workers, read_workers, rewrite):
    if rewrite:
        for name in ('inputs.json', 'table.json.gz'):
            (BASE / name).unlink(missing_ok=True)
    records = sources()
    groups = {}
    for record in records:
        groups.setdefault((record['level'], record['domain']), []).append(record['path'])
    started = time.time()
    with ProcessPoolExecutor(max_workers=read_workers) as pool:
        digests = dict((path, (digest, count)) for path, digest, count
                       in pool.map(_digest_one, [p for paths in groups.values() for p in paths], chunksize=4))
    print(json.dumps({'stage': 'specs_read', 'files': len(digests),
                      'seconds': round(time.time() - started, 1)}), flush=True)
    specs, identity = {}, {}
    for key, paths in sorted(groups.items()):
        unique = {digests[path][0] for path in paths}
        if len(unique) != 1:
            raise ValueError('evaluation prompts differ across cells of ' + '/'.join(key))
        counts = {digests[path][1] for path in paths}
        if counts != {128}:
            raise ValueError('unexpected prompt count for ' + '/'.join(key))
        specs['|'.join(key)] = first_line_specs(sorted(paths)[0])
        identity['|'.join(key)] = {'prompts': 128, 'cells': len(paths),
                                   'references_sha256': unique.pop(),
                                   'spec_source': sorted(paths)[0]}
    jobs = [(level, domain, index, reference)
            for key, references in sorted(specs.items())
            for level, domain in [key.split('|')]
            for index, reference in enumerate(references)]
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_row, jobs, chunksize=4))
    mismatched = [r for r in rows if not r['certified_match']]
    if mismatched:
        raise ValueError('enumerated support disagrees with the certified mode count: '
                         + json.dumps(mismatched[0])[:200])
    table = {'schema': 'modebench-portfolio-withdrawal-table-v1', 'rows': rows}
    BASE.mkdir(parents=True, exist_ok=True)
    payload = gzip.compress(json.dumps(table, sort_keys=True, separators=(',', ':')).encode())
    (BASE / 'table.json.gz').write_bytes(payload)
    inputs = {
        'schema': 'modebench-portfolio-withdrawal-inputs-v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'repo_root': str(ROOT), 'base': str(BASE),
        'container': str(CONTAINER), 'container_sha256': file_sha(CONTAINER),
        'curve_archive': str(CURVES), 'curve_archive_sha256': file_sha(CURVES),
        'cells': len(records), 'prompt_identity': identity,
        'domains': list(pw.DOMAINS), 'withdrawal': pw.WITHDRAWAL,
        'budgets': [1, 2, 4], 'bootstrap': {'replicates': 20000, 'seed': 20260917},
        'design_disclosure': DISCLOSURE,
        'certificate': 'exhaustive_original_support_enumeration_then_exclusion_check',
        'certified_counts_reproduced': len(rows),
        'options': sum(len(r['options']) for r in rows),
        'feasible_options': sum(sum(o['feasible'] for o in r['options']) for r in rows),
        'binding_options': sum(sum(o['binding'] for o in r['options']) for r in rows),
        'pairs': sum(len(r['pairs']) for r in rows),
        'feasible_pairs': sum(sum(p['feasible'] for p in r['pairs']) for r in rows),
        'binding_pairs': sum(sum(p['feasible'] and p['binding'] for p in r['pairs']) for r in rows),
        'pair_by_intersection': list(pw.PAIR_BY_INTERSECTION),
        'table_sha256': hashlib.sha256(payload).hexdigest(),
        'code_sha256': {name: file_sha(ROOT / name) for name in CODE},
        'outcomes_read': False,
    }
    atomic_new(BASE / 'inputs.json', inputs)
    print(json.dumps({'stage': 'frozen', 'prompt_rows': len(rows), 'options': inputs['options'],
                      'feasible': inputs['feasible_options'], 'binding': inputs['binding_options'],
                      'pairs': inputs['pairs'], 'binding_pairs': inputs['binding_pairs'],
                      'seconds': round(time.time() - started, 1)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--read-workers', type=int, default=4)
    parser.add_argument('--rewrite', action='store_true',
                        help='replace an existing freeze (the record keeps its own disclosure)')
    prepare(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
