#!/usr/bin/env python3
"""Structural validation of every ModeBench level's frozen splits, read-only.

The per-level records already answer whether a level was *admitted*: the base
grid's `data_admission.json` for Levels 1-3, `release_v2/level5/admission.json`
for Level 5, and the closure record for Level 4. What none of them re-checks is
that the rows on disk today are still the rows those decisions were made about.
That is what this does, in one pass over all five levels:

  - every train/dev/eval split present, loadable, and the registered size
  - the five columns every ModeBench row must carry
  - no duplicate problem identity inside a split
  - no identity shared between train, dev and eval of the same level+domain
  - no identity shared between two levels of the same domain, which is what
    makes a harder level a fresh population rather than a re-scored copy
  - every recorded row or file digest still matches, where the level publishes
    one: Level 2 and Level 3 row hashes, Level 3 retained file pins, and the
    Level 5 admission's file pins

Identity is the verifier-facing problem, not the rendered prompt: two rows with
different wording and the same graph are the same problem. `_row_identity` here
is the same definition `materialize_e117_evaluation_reserves` builds the
reserves with, restated so this check does not import the module whose output it
is checking.

Two knowingly accepted exceptions are declared in ACCEPTED below rather than
silently tolerated: Level 1 Pantry dev is natively 64 rows, and the Level 5
admission pins two `src/oat_drgrpo` modules that have since drifted and are
rebound by the snapshot source view.

Exit status is 1 if any finding survives, so this can gate a release step.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in ('ops', 'ops/exp_scaling', 'src'):
    if str(ROOT / directory) not in sys.path:
        sys.path.insert(0, str(ROOT / directory))
from datasets import load_from_disk  # noqa: E402

SCHEMA = 'modebench_level_structural_validation_v1'
LEVELS = ('level1', 'level2', 'level3', 'level4', 'level5')
DOMAINS = ('countdown', 'graph_coloring', 'mathir', 'pantry', 'python_factors')
SPLITS = ('train', 'dev', 'eval')
EXPECTED_ROWS = {'train': 384, 'dev': 128, 'eval': 128}
REQUIRED_COLUMNS = ('problem', 'answer', 'modebench_task',
                    'answer_mode_count', 'answer_mode_split')

# Level 1 is per-domain: the native roots hold train and the terminal test set,
# and development comes from the e117 reserve.
RESERVE = ROOT / 'var/data/e117_evaluation_reserve_v1'
NATIVE = {'countdown': 'exact_countdown_easy3_probe',
          'graph_coloring': 'graph_coloring_modebench_v2',
          'mathir': 'mathir_action_menu_v1',
          'pantry': 'pantry_plan_modebench_v2',
          'python_factors': 'python_factor_modebench_v1'}
LEVEL_ROOTS = {'level2': ROOT / 'var/data/modebench_harder_v2_matched_r5',
               'level3': ROOT / 'var/data/modebench_level3_matched_v3',
               'level5': ROOT / 'var/data/modebench_scale_release_v2/level5/dataset'}
LEVEL4_MANIFEST = ROOT / 'var/data/modebench_scale_release_v2/level4/source_manifest.json'
LEVEL5_ADMISSION = ROOT / 'var/data/modebench_scale_release_v2/level5/admission.json'

# Departures from the rules above that are recorded decisions, not drift. Each
# is keyed by what would otherwise be reported, so a second, unrelated departure
# still surfaces.
ACCEPTED = {
    'level1/pantry/dev:rows': (
        64, 'Level 1 Pantry dev is natively 64 rows; the Level 2 fairness '
            'contract expands its support histogram by factor 2 to reach 128.'),
    'level5/admission:drifted_pins': (
        {str(ROOT / 'src/oat_drgrpo/args.py'), str(ROOT / 'src/oat_drgrpo/templates.py')},
        'Known 2026-09-15 source drift. Both files are rebound to their '
        'protocol-pinned content by the snapshot source view, which is where '
        'every production step runs; no dataset pin is affected.'),
}


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rows_sha(rows: list[dict]) -> str:
    payload = '\n'.join(json.dumps(row, sort_keys=True, separators=(',', ':')) for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def _graph_color_string(colors) -> str:
    return ''.join('?' if c is None else str(int(c)) for c in colors)


def _row_identity(domain: str, row: dict) -> tuple:
    if domain == 'pantry':
        return ('pantry', str(row['instance_fingerprint']))
    spec = json.loads(str(row['answer']))
    if domain == 'countdown':
        return ('countdown', tuple(sorted(int(v) for v in spec['numbers'])), int(spec['target']))
    if domain == 'graph_coloring':
        return ('graph_coloring', int(spec['n']),
                tuple(sorted(tuple(int(v) for v in edge) for edge in spec['edges'])),
                _graph_color_string(spec['partial_colors']))
    if domain == 'python_factors':
        return ('python_factors', tuple(sorted(int(v) for v in spec['cases'])))
    if domain == 'mathir':
        return ('mathir', str(spec['family']),
                tuple(sorted((str(k), int(v)) for k, v in spec['bindings'].items())))
    raise ValueError('unknown domain: ' + domain)


def split_path(level: str, domain: str, split: str) -> Path:
    if level == 'level1':
        native = ROOT / 'var/data' / NATIVE[domain]
        if domain == 'pantry':
            return native / split
        return native / split if split in ('train', 'eval') else RESERVE / 'development' / domain / 'eval'
    if level == 'level4':
        source = json.loads(LEVEL4_MANIFEST.read_text())['sources'][domain]
        root = Path(source['source_root'])
        for candidate in (root / 'level4' / 'dataset' / domain / split,
                          root / 'dataset' / domain / split):
            if candidate.is_dir():
                return candidate
        return root / 'level4' / 'dataset' / domain / split
    return LEVEL_ROOTS[level] / domain / split


def load_rows(path: Path) -> tuple[str, list[dict]]:
    dataset = load_from_disk(str(path))
    names = list(dataset.keys())
    name = 'multi_answer' if 'multi_answer' in names else ('train' if 'train' in names else names[0])
    return name, [dict(row) for row in dataset[name]]


def check_recorded_digests(levels: tuple[str, ...], findings: list[str]) -> dict:
    """Verify each level's own published digests against what is on disk now.

    Only Levels 2, 3 and 5 publish digests to check; a run that leaves one of
    them out skips that level's digests and nothing else.
    """
    out = {}
    for level, root in (('level2', LEVEL_ROOTS['level2']), ('level3', LEVEL_ROOTS['level3'])):
        if level not in levels:
            continue
        identity = json.loads((root / 'identity.json').read_text())
        checked = failed = 0
        for domain, splits in identity['domains'].items():
            for split, record in splits.items():
                if not record.get('rows_sha256'):
                    continue
                checked += 1
                _, rows = load_rows(root / domain / split)
                if rows_sha(rows) != record['rows_sha256']:
                    failed += 1
                    findings.append(f'{level}/{domain}/{split}: recorded rows_sha256 no longer matches')
        out[level] = {'rows_sha256_checked': checked, 'rows_sha256_failed': failed}
        pins = identity.get('retained_domain_files_sha256', {})
        if pins:
            checked = failed = 0
            for domain, files in pins.items():
                for relative, expected in files.items():
                    checked += 1
                    path = root / domain / relative
                    if not path.is_file() or file_sha(path) != expected:
                        failed += 1
                        findings.append(f'{level}/{domain}: retained file pin broken: {relative}')
            out[level].update({'retained_files_checked': checked, 'retained_files_failed': failed})

    if 'level5' not in levels:
        return out
    admission = json.loads(LEVEL5_ADMISSION.read_text())
    accepted, _ = ACCEPTED['level5/admission:drifted_pins']
    checked = failed = 0
    for path, expected in admission['files_sha256'].items():
        checked += 1
        target = Path(path)
        if target.is_file() and file_sha(target) == expected:
            continue
        if path in accepted:
            continue
        failed += 1
        findings.append(f'level5: admission pin broken: {path}')
    out['level5'] = {'admission_pins_checked': checked, 'admission_pins_failed': failed,
                     'admission_pins_accepted_drift': len(accepted)}
    return out


def validate(levels: tuple[str, ...]) -> dict:
    findings: list[str] = []
    identities: dict[tuple[str, str, str], set] = {}
    report: dict[str, dict] = {}

    for level in levels:
        report[level] = {}
        for domain in DOMAINS:
            entry = {}
            for split in SPLITS:
                path, key = split_path(level, domain, split), f'{level}/{domain}/{split}'
                if not path.is_dir():
                    findings.append(f'{key}: missing split directory {path}')
                    entry[split] = {'path': str(path), 'status': 'missing'}
                    continue
                try:
                    name, rows = load_rows(path)
                except Exception as error:  # noqa: BLE001
                    findings.append(f'{key}: unreadable ({error})')
                    entry[split] = {'path': str(path), 'status': 'unreadable'}
                    continue
                missing = [c for c in REQUIRED_COLUMNS if rows and c not in rows[0]]
                if missing:
                    findings.append(f'{key}: missing columns {missing}')
                accepted = ACCEPTED.get(f'{key}:rows')
                if len(rows) != EXPECTED_ROWS[split] and (accepted is None or len(rows) != accepted[0]):
                    findings.append(f'{key}: {len(rows)} rows, expected {EXPECTED_ROWS[split]}')
                identifiers = [_row_identity(domain, row) for row in rows]
                duplicates = sum(1 for count in Counter(identifiers).values() if count > 1)
                if duplicates:
                    findings.append(f'{key}: {duplicates} duplicated identities inside the split')
                degenerate = sum(1 for row in rows if int(row['answer_mode_count']) < 1)
                if degenerate:
                    findings.append(f'{key}: {degenerate} rows claim fewer than one answer mode')
                identities[(level, domain, split)] = set(identifiers)
                entry[split] = {'path': str(path), 'dataset_key': name, 'rows': len(rows),
                                'unique_identities': len(set(identifiers)),
                                'accepted_exception': accepted[1] if accepted else None,
                                'mode_histogram': dict(sorted(Counter(
                                    int(row['answer_mode_count']) for row in rows).items())),
                                'status': 'ok'}
            report[level][domain] = entry

    for level in levels:
        for domain in DOMAINS:
            for first in range(len(SPLITS)):
                for second in range(first + 1, len(SPLITS)):
                    left = identities.get((level, domain, SPLITS[first]))
                    right = identities.get((level, domain, SPLITS[second]))
                    if left is None or right is None:
                        continue
                    if left & right:
                        findings.append(
                            f'{level}/{domain}: {len(left & right)} identities shared between '
                            f'{SPLITS[first]} and {SPLITS[second]}')

    for domain in DOMAINS:
        pooled = {level: set().union(*(identities.get((level, domain, s), set()) for s in SPLITS))
                  for level in levels}
        for first in range(len(levels)):
            for second in range(first + 1, len(levels)):
                shared = pooled[levels[first]] & pooled[levels[second]]
                if shared:
                    findings.append(
                        f'{domain}: {len(shared)} identities shared between '
                        f'{levels[first]} and {levels[second]}')

    digests = check_recorded_digests(levels, findings)
    return {'schema': SCHEMA, 'levels': list(levels), 'splits': report,
            'recorded_digests': digests,
            'accepted_exceptions': {k: v[1] for k, v in ACCEPTED.items()},
            'findings': findings, 'finding_count': len(findings),
            'status': 'pass' if not findings else 'fail'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--levels', nargs='+', default=LEVELS, choices=LEVELS)
    parser.add_argument('--output', type=Path, help='also write the full report here')
    parser.add_argument('--quiet', action='store_true', help='print only the findings summary')
    args = parser.parse_args()
    report = validate(tuple(args.levels))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    if args.quiet:
        print(json.dumps({'status': report['status'], 'findings': report['findings']}, indent=1))
    else:
        print(json.dumps(report, indent=1, sort_keys=True))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
