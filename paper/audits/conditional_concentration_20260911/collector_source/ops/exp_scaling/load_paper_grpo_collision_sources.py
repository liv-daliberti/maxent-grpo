#!/usr/bin/env python3
"""Read the frozen 75-run GRPO sample cohort without computing contrasts.

The parent precheck registers 150 Dr.GRPO/GRPO runs. This adapter retains that
registration verbatim in provenance and loads only its 75 plain-GRPO members,
which are outside the four-arm primary training snapshot. Exact source hashes
are mandatory; identical full draw records may be deduplicated, while
conflicting records make that checkpoint unavailable without changing its
original cohort membership. No new generation or grading is performed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
COHORT = Path('paper/results/baseline_collapse_precheck.json')
KINDS = 'fixed_seed_sampled_k_neutral'
STEPS = (0, 3072)
SEEDS = {'qwen05b': (43, 44, 45, 46, 47),
         'falcon1b': (55, 56, 57, 58, 59),
         'qwen3b': (70, 71, 72, 73, 74)}
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')
ENDPOINT_HINT = re.compile(rb'"step"\s*:\s*(?:0|3072)\s*[,}]')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def compact(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def _normalizer(root: Path):
    path = root / 'ops/exp_scaling/load_paper_collision_samples.py'
    spec = importlib.util.spec_from_file_location('paper_collision_samples_shared', path)
    require(spec is not None and spec.loader is not None, 'shared sample loader unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.normalize_draw, module.assemble_checkpoint


def baseline_registration(root: Path = ROOT) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify the frozen registration without selecting on sample outcomes."""
    path = root / COHORT
    payload = path.read_bytes()
    cohort = json.loads(payload)
    require(cohort.get('schema') == 'paper-baseline-collapse-precheck-v1',
            'unsupported baseline source cohort')
    require(cohort.get('expected_draws') == 4 and cohort.get('evaluation_kind') == KINDS,
            'baseline evaluation contract differs')
    expected = {(method, scale, domain, seed)
                for method in ('drgrpo', 'grpo') for scale in SEEDS
                for domain in DOMAINS for seed in SEEDS[scale]}
    registered = set()
    for method, arm in cohort['arms'].items():
        for scale, scale_record in arm['scales'].items():
            require(scale_record['target_step'] == 3072 and scale_record['training_pass'] == 8,
                    'baseline target checkpoint drifted')
            for domain, record in scale_record['domains'].items():
                registered.update((method, scale, domain, int(seed)) for seed in record['per_seed'])
    require(registered == expected, 'frozen baseline registration differs from 150 prescribed runs')
    sources = {}
    for source in cohort['telemetry_sources']:
        key = (source['arm'], source['scale'], source['domain'], int(source['seed']))
        require(key in expected and key not in sources, f'duplicate/unregistered baseline source: {key}')
        sources[key] = source
    require(set(sources) == expected, 'baseline source membership incomplete')
    for ledger in cohort['ledgers'].values():
        path = root / ledger['path']
        require(hashlib.sha256(path.read_bytes()).hexdigest() == ledger['sha256'],
                f'frozen baseline ledger changed: {ledger["path"]}')
    registration = []
    for method, scale, domain, seed in sorted(expected):
        source = sources[method, scale, domain, seed]
        registration.append({
            'level': 'level1', 'scale': scale, 'domain': domain,
            'method': method, 'seed': seed, 'terminal_admitted': True,
            'checkpoints': list(STEPS), 'source': source,
        })
    cohort['_source_receipt'] = {
        'path': str(COHORT), 'sha256': hashlib.sha256(payload).hexdigest(),
        'byte_length': len(payload),
    }
    return cohort, registration


def _frozen_lines(path: Path, size: int):
    """Read only the hash-bound prefix; later file appends are not evidence."""
    consumed = 0
    with path.open('rb') as handle:
        while consumed < size:
            raw = handle.readline(size - consumed)
            require(bool(raw), f'unexpected EOF in frozen baseline source: {path}')
            consumed += len(raw)
            yield raw


def _load_one(root: Path, descriptor: dict[str, Any], normalize_draw, assemble_checkpoint) -> tuple[dict[str, Any], dict[str, Any]]:
    source = descriptor['source']
    path = root / source['path']
    require(path.stat().st_size >= source['byte_length'], f'frozen source truncated: {path}')
    rows: dict[int, dict[int, list[tuple[dict[str, Any], dict[str, Any], str]]]] = {
        step: {} for step in STEPS}
    issues = []
    digest = hashlib.sha256()
    byte_count = 0
    matching_rows = 0
    for line_number, raw in enumerate(_frozen_lines(path, source['byte_length']), 1):
        digest.update(raw)
        byte_count += len(raw)
        if not ENDPOINT_HINT.search(raw):
            continue
        try:
            row = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            issues.append({'step': None, 'reason': 'unparseable_endpoint_candidate',
                           'path': source['path'], 'line_number': line_number})
            continue
        step = row.get('step')
        if type(step) is not int or step not in STEPS or row.get('evaluation_kind') != KINDS:
            continue
        matching_rows += 1
        draw = row.get('draw_index')
        if type(draw) is not int or draw not in range(4):
            issues.append({'step': step, 'reason': 'unexpected_draw_index', 'draw_index': draw,
                           'path': source['path'], 'line_number': line_number})
            continue
        origin = {'path': source['path'], 'line': line_number,
                  'source_sha256': source['sha256']}
        signature = hashlib.sha256(compact(row)).hexdigest()
        rows[step].setdefault(draw, []).append((row, origin, signature))
    require(byte_count == source['byte_length'] and digest.hexdigest() == source['sha256'],
            f'frozen baseline source hash changed: {path}')
    checkpoints = {}
    duplicates = 0
    for step in STEPS:
        step_issues = [issue for issue in issues if issue['step'] in (None, step)]
        if sorted(rows[step]) != list(range(4)):
            issue = {'step': step, 'reason': 'incomplete_registered_draw_set',
                     'draw_indices': sorted(rows[step])}
            issues.append(issue); step_issues.append(issue)
        draws = []
        for draw, candidates in sorted(rows[step].items()):
            if len({signature for _, _, signature in candidates}) != 1:
                issue = {'step': step, 'reason': 'conflicting_duplicate_draw',
                         'draw_index': draw, 'origins': [origin for _, origin, _ in candidates]}
                issues.append(issue); step_issues.append(issue)
                continue
            duplicates += len(candidates) - 1
            row, origin, _ = candidates[0]
            try:
                normalized = normalize_draw(row, origin=origin)
            except (ValueError, RuntimeError, AssertionError, KeyError, TypeError) as exc:
                issue = {'step': step, 'reason': 'invalid_sample_record',
                         'draw_index': draw, 'detail': str(exc), 'origins': [origin]}
                issues.append(issue); step_issues.append(issue)
                continue
            normalized['origins'] = [origin for _, origin, _ in candidates]
            draws.append(normalized)
        if not step_issues and len(draws) == 4:
            populations = [frozenset(p['prompt_id'] for p in draw['prompts']) for draw in draws]
            if any(population != populations[0] for population in populations[1:]):
                issue = {'step': step, 'reason': 'prompt_population_changes_across_draws'}
                issues.append(issue); step_issues.append(issue)
        checkpoints[str(step)] = None
        if not step_issues:
            try:
                checkpoints[str(step)] = assemble_checkpoint(draws, step=step)
            except (ValueError, RuntimeError, KeyError, TypeError) as exc:
                issues.append({'step': step, 'reason': 'checkpoint_sampling_contract', 'detail': str(exc)})
    before_after = all(checkpoints[str(step)] is not None for step in STEPS)
    if before_after:
        populations = [frozenset(p['prompt_id'] for p in checkpoints[str(step)]['draws'][0]['prompts'])
                       for step in STEPS]
        if populations[0] != populations[1]:
            issues.append({'step': None, 'reason': 'before_after_prompt_population_mismatch'})
            before_after = False
    cell = {key: descriptor[key] for key in ('level', 'scale', 'domain', 'method', 'seed', 'terminal_admitted')}
    cell.update({'in_terminal_paired_cohort': False, 'checkpoints': checkpoints,
                 'sample_issues': issues, 'before_after_available': before_after})
    receipt = {**source, 'verified_byte_length': byte_count, 'verified_sha256': digest.hexdigest(),
               'appended_bytes_ignored': path.stat().st_size - byte_count,
               'matching_endpoint_rows': matching_rows, 'identical_duplicate_rows': duplicates,
               'available_checkpoints': [step for step in STEPS if checkpoints[str(step)] is not None],
               'sample_issues': issues}
    return cell, receipt


def load_grpo_manifest(root: Path = ROOT) -> dict[str, Any]:
    """Return frozen cohort/source metadata only; do not open response logs."""
    root = Path(root)
    cohort, registration = baseline_registration(root)
    return {
        'schema': 'paper-grpo-collision-sources-v1',
        'source_root': str(root.resolve()),
        'source_cohort': cohort['_source_receipt'],
        'registration': registration,
        'cells': [record for record in registration if record['method'] == 'grpo'],
        'snapshot': cohort,
        'cohort_counts': {'baseline_registered_runs': 150, 'grpo_registered_runs': 75},
    }


def iter_grpo_sample_cells(manifest: dict[str, Any] | None = None, *, workers: int = 4):
    """Stream normalized GRPO cells, keeping every frozen registration."""
    manifest = load_grpo_manifest() if manifest is None else manifest
    root = Path(manifest['source_root'])
    cohort = manifest['snapshot']
    normalize_draw, assemble_checkpoint = _normalizer(root)
    require(type(workers) is int and workers > 0, 'workers must be a positive integer')
    descriptors = manifest['cells']
    require(len(descriptors) == 75 and all(d['method'] == 'grpo' for d in descriptors),
            'GRPO sample registration shrank or contains another method')

    def read(descriptor):
        cell, receipt = _load_one(root, descriptor, normalize_draw, assemble_checkpoint)
        old = cohort['arms']['grpo']['scales'][cell['scale']]['domains'][cell['domain']]['per_seed'][str(cell['seed'])]
        cell['frozen_reference_metrics'] = {'0': old['pass0'], '3072': old['pass8']}
        cell['source_checks'] = [receipt]
        return cell

    if workers == 1:
        for descriptor in descriptors:
            yield read(descriptor)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(read, descriptors)


def load_grpo_samples(root: Path = ROOT, *, workers: int = 4) -> dict[str, Any]:
    """Materializing convenience API; streaming callers should use iter_grpo_sample_cells."""
    manifest = load_grpo_manifest(root)
    cells = list(iter_grpo_sample_cells(manifest, workers=workers))
    receipts = [receipt for cell in cells for receipt in cell['source_checks']]
    return {
        **{key: value for key, value in manifest.items() if key not in ('snapshot', 'cells')},
        'cells': cells, 'source_files': receipts,
        'audit': {'baseline_registered_runs': len(manifest['registration']), 'grpo_registered_runs': len(cells),
                  'checkpoint_count': sum(checkpoint is not None for cell in cells
                                          for checkpoint in cell['checkpoints'].values()),
                  'before_after_available': sum(cell['before_after_available'] for cell in cells),
                  'identical_duplicate_rows': sum(r['identical_duplicate_rows'] for r in receipts),
                  'runs_with_sample_issues': sum(bool(cell['sample_issues']) for cell in cells)},
    }


def source_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Compact provenance excludes sample values and all outcome comparisons."""
    return {key: value for key, value in payload.items() if key != 'cells'} | {
        'cells': [{key: value for key, value in cell.items() if key not in ('checkpoints', 'frozen_reference_metrics')} | {
            'checkpoints': {step: None if checkpoint is None else {
                'step': checkpoint['step'],
                'sampling_certificate': checkpoint['sampling_certificate'],
                'draws': [{key: draw[key] for key in ('draw_index', 'metadata', 'origins')}
                          for draw in checkpoint['draws']],
            } for step, checkpoint in cell['checkpoints'].items()},
        } for cell in payload['cells']],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/grpo_collision_sources.json'))
    args = parser.parse_args()
    result = load_grpo_samples()
    args.output.write_text(json.dumps(source_manifest(result), indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), **result['audit']}, sort_keys=True))


if __name__ == '__main__':
    main()
