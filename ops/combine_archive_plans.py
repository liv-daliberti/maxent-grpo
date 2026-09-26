#!/usr/bin/env python3
"""Union a published archive plan with a newly prepared one; never upload or delete.

The catalog and public READMEs are rendered from a plan but committed to a single
shared repository, so running a second plan on its own would replace the existing
catalog with one describing only the new models. Worse, ``render_readmes`` asserts
cross-study invariants over the whole selection -- the E120 comparator mapping, for
one -- which a partial plan cannot satisfy, so a standalone pass stops in its
catalog phase and, because ``fill_slots`` is gated on there being no failures,
stops admitting new uploads with most of its models still on local disk.

The union keeps one catalog describing everything in the repository. The base
plan's model records are preserved verbatim and in order; the new plan's records
are appended. Both plans must already agree on repository, visibility and
retention, and the new records must collide with the base on neither their remote
folder nor their physical export.

Record-only coverage is carried forward unchanged. A study whose weights are
missing stays missing: adding downloadable models to the repository grows the
deployable census and leaves the scientific-record census alone.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def combine(base_path: Path, added_path: Path, output: Path, why: str) -> dict:
    base_path, added_path = Path(base_path).resolve(), Path(added_path).resolve()
    output = Path(output).resolve()
    base, added = read(base_path), read(added_path)
    for plan, label in ((base, 'base'), (added, 'added')):
        require(plan.get('schema') == 'completed-model-hf-archive-plan-v1', f'{label} plan contract differs')
        require(len(plan['models']) == plan['expected_model_count'], f'{label} plan count differs')
    require(base['repo_id'] == added['repo_id'], 'Plans publish to different repositories')
    require(base['private'] == added['private'], 'Plans disagree on repository visibility')
    require(base['retention'] == added['retention'], 'Plans disagree on retention')
    # The engine keys per-model state off output_dir, so the union has to inherit
    # the base state directory: that is where the already-finished models carry the
    # verification and retirement receipts that keep them from being uploaded again.
    require(Path(base['output_dir']).is_dir(), 'Base state directory is missing')

    prefixes = {m['repo_prefix'] for m in base['models']}
    exports = {m['terminal_export'] for m in base['models']}
    ids = {m['archive_id'] for m in base['models']}
    require(len(prefixes) == len(exports) == len(ids) == len(base['models']), 'Base plan identities collide')
    for model in added['models']:
        require(model['repo_prefix'] not in prefixes, 'Remote folder collision: ' + model['repo_prefix'])
        require(model['terminal_export'] not in exports, 'Physical export collision: ' + model['terminal_export'])
        require(model['archive_id'] not in ids, 'Archive id collision: ' + model['archive_id'])
        prefixes.add(model['repo_prefix']); exports.add(model['terminal_export']); ids.add(model['archive_id'])

    pins = dict(base['ledger_pins'])
    for path, value in (added.get('ledger_pins') or {}).items():
        require(path not in pins or pins[path] == value, 'Conflicting ledger pin: ' + path)
        pins[path] = value
    for path, value in pins.items():
        require(digest(Path(path)) == value, 'Ledger changed before combination: ' + path)

    models = list(base['models']) + list(added['models'])
    result = {key: base[key] for key in base if key not in ('models', 'combined_from')}
    result.update({
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'ledger_pins': pins, 'models': models, 'expected_model_count': len(models),
        'expected_total_export_bytes': sum(m['terminal_bytes'] for m in models),
        'combined_from': {'base_plan': str(Path(base_path).relative_to(ROOT)),
                          'base_plan_sha256': digest(base_path),
                          'added_plan': str(Path(added_path).relative_to(ROOT)),
                          'added_plan_sha256': digest(added_path),
                          'added_models': len(added['models']),
                          'previous_combination': base.get('combined_from'),
                          'why': why},
    })

    coverage = result.get('paper_coverage')
    if coverage is not None:
        was = {'deployable_model_exports': coverage['deployable_model_exports'],
               'logical_model_records': coverage['logical_model_records']}
        record_only = coverage['record_only_by_source']
        require(coverage['record_only_count'] == sum(record_only.values()), 'Record-only census differs')
        require(was['logical_model_records'] == len(base['models']) + coverage['record_only_count'],
                'Base coverage does not reconcile with its own model count')
        coverage['deployable_model_exports'] = len(models)
        coverage['logical_model_records'] = len(models) + coverage['record_only_count']
        coverage['coverage_amendment'] = {
            'at_utc': datetime.now(timezone.utc).date().isoformat(), 'was': was, 'why': why}

    require(result['models'][:len(base['models'])] == base['models'], 'Base model records were altered')
    require(result['output_dir'] == base['output_dir'], 'Base state directory was altered')
    require(output.parent == Path(base['output_dir']).resolve() == base_path.parent,
            'Combined plan must share the base state directory')
    require(not output.exists() and not output.with_suffix('.sha256').exists(),
            'Combined plan destination already exists')
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    output.with_suffix('.sha256').write_text(digest(output) + '\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-plan', type=Path, required=True)
    parser.add_argument('--added-plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--why', required=True, help='why these plans are published as one catalog')
    args = parser.parse_args()
    result = combine(args.base_plan, args.added_plan, args.output, args.why)
    print(json.dumps({'models': result['expected_model_count'],
                      'terabytes': round(result['expected_total_export_bytes'] / 1e12, 3),
                      'coverage': result.get('paper_coverage', {}).get('logical_model_records'),
                      'plan': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
