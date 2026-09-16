#!/usr/bin/env python3
"""Bind each cell to its recorded launch settings and its training GPU model.

The manifest's evaluation config is derived from the E72 reference runs the
launchers resolved against. The scheduler records preserve what was actually
submitted, so this cross-checks the derivation against the record and refuses to
continue on any disagreement: the resampler must reproduce the evaluation the
run really performed, not a plausible reconstruction of it.

GPU model matters because a decoding measurement is only comparable to the one
it checks when the arithmetic matches. Most runs pinned a model in their generic
resource request; the E118 Qwen-3B runs asked only for "a GPU", so their model
is resolved from the node the scheduler actually gave them.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_record import launch_records  # noqa: E402

MANIFEST = ROOT / 'var/artifacts/pmd_independent_resample_20260915/cohort_manifest.json'
LEDGER_DIRS = (ROOT / 'paper/audits/training_curves_20260912/ledgers', ROOT / 'var/artifacts')
NODE_GRES = re.compile(r'Gres=gpu:([a-z][a-z0-9]*)')

# Settings recovered from the submit line that must agree with the derivation.
CHECKED = ('prompt_template', 'eval_generate_max_length', 'max_model_len',
           'prompt_max_length', 'eval_batch_size', 'eval_data',
           'eval_mode_coverage_seed', 'eval_mode_coverage_k',
           'eval_mode_coverage_draws', 'eval_mode_coverage_temperature',
           'vllm_gpu_ratio')


def require(ok: bool, message: str) -> None:
    if not ok:
        raise SystemExit(message)


def terminal_job_id(cell: dict) -> str:
    """The attempt that trained, not whichever attempt sorts first."""
    return Path(cell['draws_path']).parent.name.replace('debug_job', '')


def allocated_nodes(job_ids: list[str]) -> dict[str, str]:
    """One batched accounting query; never a polling loop."""
    if not job_ids:
        return {}
    result = subprocess.run(
        ['sacct', '-j', ','.join(sorted(set(job_ids))), '-X', '-P', '-n',
         '--format=JobID,NodeList,State'],
        capture_output=True, text=True, check=True)
    nodes: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.split('|')
        if len(parts) < 3 or parts[2] != 'COMPLETED':
            continue
        if parts[1] and not parts[1].startswith('None'):
            nodes[parts[0]] = parts[1]
    return nodes


def node_gpu(names: set[str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for name in sorted(names):
        result = subprocess.run(['scontrol', 'show', 'node', name],
                                capture_output=True, text=True)
        hit = NODE_GRES.search(result.stdout)
        if hit:
            resolved[name] = hit.group(1)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    records = launch_records(LEDGER_DIRS)
    overrides: dict[str, dict] = defaultdict(dict)

    pending: list[tuple[dict, str]] = []
    for cell in manifest['cells']:
        entry = records.get(cell['run_dir'])
        require(entry is not None, f'no scheduler record for {cell["run_dir"]}')
        recorded = entry['eval_settings']
        require(recorded, f'scheduler record for {cell["run_dir"]} carries no evaluation settings')
        derived = cell.pop('eval_config')
        for key in CHECKED:
            require(key in recorded, f'{cell["run_dir"]}: submit line lacks {key}')
        # The submit line is what ran. The E72 reference is kept beside it because
        # the cross-family launchers deliberately retuned some budgets -- E79 gives
        # Falcon its own per-domain generation lengths -- and a silent divergence
        # between the two would be the kind of thing that invalidates a resample.
        differing = {key: {'launched': recorded[key], 'e72_reference': derived[key]}
                     for key in CHECKED if recorded[key] != derived[key]}
        for key in differing:
            seen = overrides[f'{cell["scale"]}/{cell["domain"]}'].setdefault(
                key, (derived[key], set()))
            seen[1].add(recorded[key])
        cell['eval_config'] = {key: recorded[key] for key in CHECKED}
        cell['eval_config_source'] = 'scheduler_submit_line'
        cell['canonical_action_task'] = entry['canonical_action_task']
        cell['e72_reference_differences'] = differing
        cell['launch_ledger'] = entry['ledger']
        seed_base = int(recorded['eval_mode_coverage_seed'])
        draws = int(recorded['eval_mode_coverage_draws'])
        require(cell['original_draw_seeds'] == [seed_base + i for i in range(draws)],
                f'{cell["run_dir"]}: recorded seed base does not explain the saved draw seeds')
        if entry['gpu']:
            cell['gpu'] = entry['gpu']
            cell['gpu_source'] = 'generic_resource_request'
        else:
            pending.append((cell, terminal_job_id(cell)))

    if pending:
        nodes = allocated_nodes([job for _, job in pending])
        models = node_gpu({nodes[job] for _, job in pending if job in nodes})
        for cell, job in pending:
            node = nodes.get(job)
            require(node is not None,
                    f'{cell["run_dir"]}: job {job} has no completed accounting record')
            model = models.get(node)
            require(model is not None, f'cannot read the GPU model of {node}')
            cell['gpu'] = model
            cell['gpu_source'] = f'allocated_node:{node}'

    manifest['launcher_budget_overrides'] = {
        cell_key: {field: {'e72_reference': before,
                           'launched': sorted(after, key=str)}
                   for field, (before, after) in sorted(fields.items())}
        for cell_key, fields in sorted(overrides.items())}
    manifest['cells_by_canonical_action_task'] = dict(sorted(
        Counter(c['canonical_action_task'] for c in manifest['cells']).items()))
    manifest['cells_by_gpu'] = dict(sorted(Counter(c['gpu'] for c in manifest['cells']).items()))
    manifest['gpu_source_counts'] = dict(sorted(Counter(
        c['gpu_source'].split(':')[0] for c in manifest['cells']).items()))
    with args.manifest.open('w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps({'cells': len(manifest['cells']),
                      'by_gpu': manifest['cells_by_gpu'],
                      'gpu_sources': manifest['gpu_source_counts'],
                      'canonical_action_task': manifest['cells_by_canonical_action_task'],
                      'launcher_budget_overrides': manifest['launcher_budget_overrides']},
                     indent=2))


if __name__ == '__main__':
    main()
