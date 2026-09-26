#!/usr/bin/env python3
"""Evaluate the trained terminal checkpoints on every level's test set.

The before/after concentration figure is currently Level 1 only, and only two
domains clear the support bar there, which makes it look like a thin slice of a
broader result rather than the whole of what was measured. Every level's test
set exists and every terminal policy is archived, so the wider version needs no
training -- only inference.

Two things this does not do, deliberately. It does not read prompts from the
run's own saved draw log, as the terminal resample does, because the point here
is a *different* test set than the run was evaluated on. And it does not borrow
the base grid's native-chat interface: each checkpoint answers through its own
training prompt surface, because those are not interchangeable. The same
Qwen2.5-0.5B scores 0.73 on Python Factors through one and 0.00 through the
other, so a before/after difference taken across surfaces would measure the
interface rather than the training.

Levels 4 and 5 are transfer measurements: the policies were trained at Level 1
(or 2 and 3 for the E119 and E122 cohorts), so a change measured at a harder
level says whether training concentrates outputs on problems it never saw.

Level 4 is the one level here that is not admitted. Its held-out confirmation
passed four of five domains and MathIR failed at 1.07 of tolerance, and the
decision on 2026-09-15 was to close it there rather than rebuild
(``artifacts/modebench_scale_level4_admission_closure_20260915.json``). Its five
datasets are frozen and usable, and the paper already reports Level 4 in the
local base-grid tables on exactly that footing, so the cells stay in the cohort.
What must not happen is a Level-4 number being read later as an admitted-level
result, so every cell carries its level's admission status and MathIR's
``difficulty_matched: false`` travels with it.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_terminal_cohort_manifest as base  # noqa: E402

SCHEMA = 'pmd-independent-resample-levels-cohort-v1'
RUN = ROOT / 'var/artifacts/pmd_independent_resample_20260915'
OUT = RUN / 'levels_cohort_manifest.json'

# Where each level's evaluation split lives. Level 1 is per-domain and comes
# from the run's own recorded eval_data; the rest are single roots.
LEVEL_ROOTS = {
    'level2': ROOT / 'var/data/modebench_harder_v2_matched_r5',
    'level3': ROOT / 'var/data/modebench_level3_matched_v3',
    'level5': ROOT / 'var/data/modebench_scale_release_v2/level5/dataset',
}
# Level 4 is not gathered under one directory: its release records a per-domain
# source root, some from the original campaign and some from later domain
# revisions. Resolving through that manifest is what makes it the bound Level 4
# rather than whichever revision happens to sort last on disk. The manifest was
# sealed before any held-out receipt existed, so it is the source selection and
# says nothing about whether that selection then passed.
LEVEL4_MANIFEST = ROOT / 'var/data/modebench_scale_release_v2/level4/source_manifest.json'
# Whether a level passed its gates is a separate published record from where its
# rows live, and only these files may answer it. Levels 1-3 are carried by the
# base-grid admission; Level 5 by its own release admission; Level 4 by the
# partial release status that records four of five confirmed.
BASE_GRID_ADMISSION = ROOT / 'artifacts/modebench_base_level_grid_20260911/data_admission.json'
LEVEL5_ADMISSION = ROOT / 'var/data/modebench_scale_release_v2/level5/admission.json'
LEVEL4_STATUS = ROOT / 'var/data/modebench_scale_release_v2/level4/release_status.json'
# Dataset directories name the domain without the task suffix the cohort uses.
DOMAIN_DIR = {'graph_coloring': 'graph_coloring', 'countdown': 'countdown',
              'mathir': 'mathir', 'python_factors': 'python_factors',
              'pantry_plan': 'pantry'}


def admission(level: str) -> dict:
    """Resolve one level's admission status from its own published record.

    Returned per level, and attached to every cell, so a number measured here
    cannot later be read as an admitted-level result without the caller having
    seen that it is not one. Nothing is inferred from disk layout: a level whose
    record is missing is an error, not an unadmitted level.
    """
    if level in ('level1', 'level2', 'level3'):
        record = json.loads(BASE_GRID_ADMISSION.read_text())['levels'][level[-1]]
        return {'admitted': True, 'status': record['status'],
                'evidence': str(BASE_GRID_ADMISSION),
                'evidence_sha256': base.file_sha(BASE_GRID_ADMISSION),
                'domains_not_difficulty_matched': []}
    if level == 'level5':
        record = json.loads(LEVEL5_ADMISSION.read_text())
        base.require(record['difficulty_matched'] is True,
                     'level5 admission exists but does not report difficulty_matched')
        failed = sorted(d for d, v in record['domains'].items()
                        if v.get('difficulty_matched') is False)
        return {'admitted': True, 'status': 'difficulty_matched',
                'evidence': str(LEVEL5_ADMISSION),
                'evidence_sha256': base.file_sha(LEVEL5_ADMISSION),
                'domains_not_difficulty_matched': failed}
    base.require(level == 'level4', f'no admission record registered for {level}')
    base.require(not (LEVEL4_MANIFEST.parent / 'admission.json').is_file(),
                 'level4 now carries an admission record; this builder must be updated')
    record = json.loads(LEVEL4_STATUS.read_text())
    base.require(record['difficulty_matched'] is False,
                 'level4 release status no longer reports a partial release')
    return {'admitted': False, 'status': record['status'],
            'evidence': str(LEVEL4_STATUS),
            'evidence_sha256': base.file_sha(LEVEL4_STATUS),
            'decision': record['closure_decision'],
            'domains_not_difficulty_matched': sorted(record['domains_not_difficulty_matched'])}


def level4_split(domain: str) -> Path:
    manifest = json.loads(LEVEL4_MANIFEST.read_text())
    source = manifest['sources'].get(DOMAIN_DIR[domain])
    base.require(source is not None, f'level4: no bound source for {domain}')
    root = Path(source['source_root'])
    for candidate in (root / 'level4' / 'dataset' / DOMAIN_DIR[domain] / 'eval',
                      root / 'dataset' / DOMAIN_DIR[domain] / 'eval'):
        if candidate.is_dir():
            return candidate
    raise SystemExit(f'level4/{domain}: bound source root has no eval split: {root}')


def eval_split(level: str, domain: str) -> Path:
    if level == 'level4':
        path = level4_split(domain)
        base.require((path / 'dataset_dict.json').is_file(),
                     f'level4/{domain}: eval split is not a saved dataset: {path}')
        return path
    root = LEVEL_ROOTS[level]
    path = root / DOMAIN_DIR[domain] / 'eval'
    base.require(path.is_dir(), f'{level}/{domain}: no eval split at {path}')
    base.require((path / 'dataset_dict.json').is_file(),
                 f'{level}/{domain}: eval split is not a saved dataset: {path}')
    return path


def build(levels: tuple[str, ...]) -> dict:
    terminal = json.loads((RUN / 'cohort_manifest.json').read_text())['cells']
    admissions = {level: admission(level) for level in levels}
    cells, missing = [], []
    for level in levels:
        status = admissions[level]
        for cell in terminal:
            domain = cell['domain']
            try:
                split = eval_split(level, domain)
            except SystemExit as error:
                key = f'{level}/{domain}'
                if key not in {m['cell'] for m in missing}:
                    missing.append({'cell': key, 'reason': str(error)})
                continue
            config = dict(cell['eval_config'])
            # The surface stays the run's own; only the problems change.
            config['eval_data'] = str(split)
            cells.append({**{k: cell[k] for k in
                             ('scale', 'domain', 'method', 'seed', 'run_dir', 'gpu',
                              'canonical_action_task', 'terminal_step', 'draws_path',
                              'weights')},
                          'level': level,
                          'evaluated_level': level,
                          'trained_level': cell['level'],
                          'transfer': level != cell['level'],
                          'level_admitted': status['admitted'],
                          'level_admission_status': status['status'],
                          # True only where the level's own record says this
                          # domain cleared its held-out difficulty gate.
                          'domain_difficulty_matched':
                              DOMAIN_DIR[domain] not in status['domains_not_difficulty_matched'],
                          'registered_job_id': None,
                          'prompt_source': 'dataset',
                          'eval_config': config,
                          'registered_terminal_reference': None})
    return {
        'schema': SCHEMA,
        'purpose': ('measure terminal policies on every level test set through their '
                    'own training prompt surface; inference only, no training'),
        'levels': list(levels),
        'level_admission': admissions,
        'unadmitted_levels': sorted(l for l, a in admissions.items() if not a['admitted']),
        'cells_not_difficulty_matched': sum(
            1 for c in cells if not c['domain_difficulty_matched']),
        'builder': {'path': str(Path(__file__).resolve().relative_to(ROOT)),
                    'sha256': base.file_sha(Path(__file__).resolve())},
        'cell_count': len(cells),
        'cells_by_level': dict(sorted(Counter(c['level'] for c in cells).items())),
        'transfer_cells': sum(1 for c in cells if c['transfer']),
        'unavailable': missing,
        'cells': cells,
    }


SPLIT_REASON = 'one manifest per level: array indices must stay below MaxArraySize'


def write(path: Path, payload: dict) -> None:
    """Replace a manifest by rename.

    Array tasks read these by index as they launch, so a truncate-and-write
    would let a worker starting mid-write read a half-written manifest.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(path.name + '.new')
    scratch.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    scratch.replace(path)


def per_level(payload: dict, level: str) -> dict:
    """One level's slice, in the combined manifest's own cell order.

    The arrays were submitted against these files, so the slice must keep the
    order the combined build produces: an index that moves re-points a queued
    task at a different cell.
    """
    cells = [c for c in payload['cells'] if c['level'] == level]
    return {**{k: v for k, v in payload.items()
               if k not in ('cells', 'levels', 'cell_count', 'cells_by_level',
                            'transfer_cells', 'level_admission', 'unadmitted_levels',
                            'cells_not_difficulty_matched')},
            'levels': [level],
            'split_reason': SPLIT_REASON,
            'level_admission': {level: payload['level_admission'][level]},
            'unadmitted_levels': [level] if not payload['level_admission'][level]['admitted'] else [],
            'cells_not_difficulty_matched': sum(
                1 for c in cells if not c['domain_difficulty_matched']),
            'cell_count': len(cells),
            'cells_by_level': {level: len(cells)},
            'transfer_cells': sum(1 for c in cells if c['transfer']),
            'cells': cells}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--levels', nargs='+',
                        default=('level2', 'level3', 'level4', 'level5'))
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--split-by-level', action='store_true',
                        help='also write <output stem>_<level>.json, the files the arrays read')
    args = parser.parse_args()
    payload = build(tuple(args.levels))
    write(args.output, payload)
    written = [str(args.output)]
    if args.split_by_level:
        for level in args.levels:
            path = args.output.with_name(f'{args.output.stem}_{level}{args.output.suffix}')
            write(path, per_level(payload, level))
            written.append(str(path))
    print(json.dumps({'cells': payload['cell_count'],
                      'by_level': payload['cells_by_level'],
                      'transfer_cells': payload['transfer_cells'],
                      'unadmitted_levels': payload['unadmitted_levels'],
                      'cells_not_difficulty_matched': payload['cells_not_difficulty_matched'],
                      'unavailable': payload['unavailable'],
                      'written': written}, indent=2))


if __name__ == '__main__':
    main()
