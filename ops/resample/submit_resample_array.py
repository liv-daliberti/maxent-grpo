#!/usr/bin/env python3
"""Submit the cohort as one job array per GPU class, skipping finished cells.

Each cell must run on the GPU model it trained on, and the three classes live in
different partitions, so the cohort is submitted as three arrays rather than one.
Cells whose receipt already exists are dropped from the index list, which makes
resubmission after a partial run cheap and idempotent.

Prints the submission plan by default; ``--submit`` actually calls sbatch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'var/artifacts/pmd_independent_resample_20260915'
SCRIPT = 'ops/resample/slurm/resample_cell.slurm'

# node302 holds every A100 and sits in the group's own partition; the A5000 and
# A6000 pools are the general cs partition.
PLACEMENT = {
    'a100': {'partition': 'mltheory', 'account': 'mltheory', 'nodelist': 'node302'},
    'a5000': {'partition': 'cs'},
    # The cs partition sees only nodes 205-207 of the A6000 pool; 'all' also
    # reaches node103 and node208, which are the same GPU model. Staying on cs
    # put these cells a day out in the queue for no scientific gain.
    'a6000': {'partition': 'all'},
}


def receipt_stamp(cell: dict) -> str:
    """Identify a receipt uniquely: the level it was evaluated on, and the
    context it was read under.

    Without the level, a Level-3 measurement of a cell writes to the same name
    as its Level-1 measurement: the second is skipped as already done, or worse
    overwrites the first. The same policy measured on two test sets is two
    results, not one.

    The same argument applies to the context. A cell measured on a widened
    surface is a different measurement from the same cell on the trained one,
    and the two have to be able to coexist -- otherwise the narrow proof holds
    the name and the widened surface can never be proven at all. Only widened
    cells carry the suffix, so every existing receipt keeps its name.
    """
    parts = [str(cell['level']), str(cell['scale']), str(cell['domain']),
             str(cell['method']), f"s{cell['seed']}"]
    if cell.get('context_widening'):
        parts.append(f"w{cell['eval_config']['max_model_len']}")
    return '__'.join(parts)


def ranges(indices: list[int]) -> str:
    """Compress sorted indices into an sbatch array specification."""
    spans, start, previous = [], None, None
    for index in indices:
        if start is None:
            start = previous = index
        elif index == previous + 1:
            previous = index
        else:
            spans.append((start, previous))
            start = previous = index
    if start is not None:
        spans.append((start, previous))
    return ','.join(str(a) if a == b else f'{a}-{b}' for a, b in spans)


def surface(cell: dict) -> str:
    """The distinct sampling surfaces a reproduction must cover.

    Reproducing one cell proves the offline harness rebuilds prompts, seeds and
    grading correctly for everything that shares its prompt template, generation
    budget, context length, KV-cache fraction, action support and GPU
    arithmetic. Those are the things that can differ between cells, so they are
    what the gate is keyed on. The cache fraction is in the key because it sets
    how many requests batch together, and batching sets the reduction order the
    kernels use.
    """
    config = cell['eval_config']
    return '|'.join(str(part) for part in (
        cell['gpu'], config['prompt_template'], config['eval_generate_max_length'],
        config['max_model_len'], config['vllm_gpu_ratio'],
        cell['canonical_action_task']))


def receipt_surface(receipt: dict) -> str:
    """The same key, read off a receipt.

    Receipts carry their own settings, so a surface confirmed while measuring
    one cohort also clears it for another. Indexing back into a manifest by
    position would silently mis-key as soon as a second cohort exists.
    """
    return surface({'gpu': receipt['gpu'],
                    'eval_config': receipt['eval_config'],
                    'canonical_action_task': receipt['canonical_action_task']})


def reproduction_gate(manifest: dict, cells: list[dict] | None = None) -> dict:
    """Refuse the independent run until each surface's checkpoint is confirmed.

    The gate deliberately does not demand byte-exact replay. The training
    evaluation ran inside an engine collocated with the optimizer, and a
    standalone engine differs slightly in the forward pass; measured on a
    Qwen-3B MathIR cell, that flips about five percent of greedy token choices
    and eight percent of sampled ones. No amount of configuration matching
    removes it, so requiring byte-equality would only mean never running.

    What must hold is that the restored weights are the policy the run
    evaluated. The greedy traces establish that: agreement climbs monotonically
    across the saved steps and peaks at the terminal one. A surface is cleared
    when a greedy receipt shows that peak.

    The engine difference is then handled by design rather than by assumption.
    Every cell is measured twice on this engine -- once under the original seed
    schedule, once under the independent one -- so the seed-policy effect is a
    within-engine contrast, and the engine effect is visible separately as
    replay against the registered values.
    """
    import gzip
    confirmed: set[str] = set()
    failures: list[str] = []
    for path in sorted((RUN / 'receipts' / 'greedy').glob('*.json.gz')):
        with gzip.open(path, 'rt') as handle:
            receipt = json.loads(handle.read())
        key = receipt_surface(receipt)
        identity = receipt.get('checkpoint_identity') or {}
        if identity.get('peaks_at_terminal_step'):
            confirmed.add(key)
        else:
            failures.append(f'{path.name}: best_step={identity.get("best_step")} '
                            f'terminal={identity.get("terminal_step")} '
                            f'agreement={identity.get("terminal_agreement")}')
    # Gate only what is being submitted: a surface with no cell in this batch
    # has nothing to hold up.
    required = {surface(cell) for cell in (manifest['cells'] if cells is None else cells)}
    return {'required_surfaces': len(required), 'proven_surfaces': len(confirmed),
            'unproven_surfaces': sorted(required - confirmed),
            'failed_reproductions': failures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=RUN / 'cohort_manifest.json')
    parser.add_argument('--mode', choices=('independent', 'reproduce', 'greedy'),
                        default='independent')
    parser.add_argument('--max-concurrent', type=int, default=6,
                        help='array throttle per GPU class; node302 has eight A100s '
                             'and is shared with other work')
    parser.add_argument('--gpu', action='append', choices=sorted(PLACEMENT),
                        help='restrict to one or more GPU classes')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--allow-unreproduced', action='store_true',
                        help='skip the reproduction gate; only for debugging')
    parser.add_argument('--job-tag', default=None,
                        help='override the job-name stem, to tell cohorts apart in squeue')
    parser.add_argument('--partition', default=None,
                        help='override the placement partition. The site reroutes '
                             'this script to cs regardless of --partition unless an '
                             'account is given, so pair it with --account.')
    parser.add_argument('--account', default=None,
                        help='slurm account; needed for the mltheory partition')
    parser.add_argument('--nodelist', default=None,
                        help='pin to specific nodes, e.g. an idle one the general '
                             'queue is not reaching')
    parser.add_argument('--mem', default=None,
                        help='override the sbatch memory request. The default 64G in '
                             'the slurm script is ~6x the measured 10G a cell uses, '
                             'which blocks scheduling on a memory-tight node while '
                             'GPUs sit idle. Size it from seff, not from caution.')
    parser.add_argument('--exclude-domain', action='append', default=[],
                        help='hold a domain back; repeatable. Use when one '
                             'domain awaits a measurement decision and the rest '
                             'should not wait for it.')
    parser.add_argument('--one-per-surface', action='store_true',
                        help='submit a single representative cell per sampling surface, '
                             'which is all a reproduction check needs')
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    if args.mode == 'independent' and not args.allow_unreproduced:
        scoped = [c for c in manifest['cells'] if not args.gpu or c['gpu'] in args.gpu]
        gate = reproduction_gate(manifest, scoped)
        if gate['unproven_surfaces']:
            print(json.dumps(gate, indent=2))
            sys.exit('checkpoint identity is unconfirmed on the surfaces listed above; '
                     'run --mode greedy --one-per-surface first')
    done = {p.name.removesuffix('.json.gz')
            for p in (RUN / 'receipts' / args.mode).glob('*.json.gz')}
    pending: dict[str, list[int]] = {}
    claimed: set[str] = set()
    if args.one_per_surface:
        claimed = {surface(manifest['cells'][c['index']]) for c in []}
        proven = reproduction_gate(manifest) if args.mode == 'greedy' else None
        if proven is not None:
            claimed = set(s for s in {surface(cell) for cell in manifest['cells']}
                          if s not in set(proven['unproven_surfaces']))
    for index, cell in enumerate(manifest['cells']):
        stamp = receipt_stamp(cell)
        if stamp in done:
            continue
        if args.gpu and cell['gpu'] not in args.gpu:
            continue
        if cell['domain'] in args.exclude_domain:
            continue
        # A greedy trace is compared against the run's own saved draws, so a
        # dataset-sourced cell can never prove a surface: the worker rejects the
        # mode and the job burns a GPU to fail. Surfaces are proved from the
        # saved-draws cohort, which is where those cells live.
        if args.mode == 'greedy' and cell.get('prompt_source', 'saved_draws') != 'saved_draws':
            continue
        if args.one_per_surface:
            key = surface(cell)
            if key in claimed:
                continue
            claimed.add(key)
        pending.setdefault(cell['gpu'], []).append(index)

    plan = []
    for gpu, indices in sorted(pending.items()):
        placement = dict(PLACEMENT[gpu])
        if args.partition:
            placement['partition'] = args.partition
            placement.pop('nodelist', None)
        if args.account:
            placement['account'] = args.account
        if args.nodelist:
            placement['nodelist'] = args.nodelist
        command = ['sbatch', f'--partition={placement["partition"]}']
        if 'account' in placement:
            command.append(f'--account={placement["account"]}')
        if 'nodelist' in placement:
            command.append(f'--nodelist={placement["nodelist"]}')
        if args.mem:
            command.append(f'--mem={args.mem}')
        command += [f'--gres=gpu:{gpu}:1',
                    f'--array={ranges(sorted(indices))}%{args.max_concurrent}',
                    f'--job-name=pmd-{args.job_tag or args.mode[:4]}-{gpu}',
                    f'--export=ALL,MODE={args.mode},MANIFEST={args.manifest.resolve()}',
                    SCRIPT]
        plan.append({'gpu': gpu, 'cells': len(indices), 'command': ' '.join(command)})
        if args.submit:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            plan[-1]['result'] = (result.stdout or result.stderr).strip()
            if result.returncode != 0:
                print(json.dumps(plan, indent=2))
                sys.exit(f'sbatch failed for {gpu}')
    print(json.dumps({'mode': args.mode, 'already_done': len(done),
                      'pending_cells': sum(len(v) for v in pending.values()),
                      'submitted': args.submit, 'arrays': plan}, indent=2))


if __name__ == '__main__':
    main()
