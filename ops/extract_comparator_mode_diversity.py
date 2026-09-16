#!/usr/bin/env python3
"""Terminal PCMD for the Panel-B comparator arms that the curve archive omits.

The frozen curve archive covers Dr.GRPO, GRPO, MaxRL and the two replay arms.
Panel B of the retention matrix also carries UCPO, sparse RLEP-Dr and the fixed
Semantic-MaxEnt comparator, which were never extracted, so this walks those
runs' saved draw files once and reports terminal pairwise correct-mode diversity.

Reading is deliberately modest, as in ``extract_mode_diversity_curves.py``: the
file list is enumerated from the registered payloads rather than discovered by
traversal, each file is streamed once line by line, and only the sampled
evaluation's canonical keys are kept.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'ops') not in sys.path:
    sys.path.insert(0, str(ROOT / 'ops'))
from mode_diversity import DEFAULT_MIN_DEFINED_PROMPTS, mode_diversity  # noqa: E402

SAMPLED = 'fixed_seed_sampled_k_neutral'
OUT = ROOT / 'paper/results/mode_diversity_comparators_05b.json'
UCPO = ROOT / 'paper/results/ucpo_interim_05b.json'
SEMANTIC = ROOT / 'paper/results/semantic_current_summary_20260912.json'
RLEP = ROOT / 'var/artifacts/e98r1_sparse_rlep_dr_05b_jobs.json'
DOMAIN = {'graph': 'graph_coloring', 'graph_coloring': 'graph_coloring',
          'python': 'python_factors', 'python_factors': 'python_factors',
          'pantry': 'pantry_plan', 'pantry_plan': 'pantry_plan',
          'countdown': 'countdown', 'mathir': 'mathir'}


def identity(path: str) -> dict | None:
    """Recover (method, domain, seed) from the run directory name.

    The sparse RLEP runs use a different layout, ``e98_rlep_pools/<domain>/s<seed>``,
    rather than encoding the cell in a single directory name.
    """
    if '/var/data/e98_rlep_pools/' in path:
        # Prompt-pool construction, evaluated once at step 0 over a 384-prompt
        # pool. Not the trained RLEP arm and not on the 128-prompt eval split.
        return None
    run = re.search(r'/?var/data/([^/]+)/', path)
    if not run:
        return None
    stem = run.group(1)
    seed = re.search(r'_s(\d+)(?:_|$)', stem)
    if not seed:
        return None
    # Order matters: several of these stems contain the token 'replay_only'
    # because that is the experiment's name, not the arm. The rehearsal run is
    # the replay arm; the compute-matched run beside it is its control.
    if 'semantic_maxent' in stem:
        method = 'semantic'
    elif 'ucpo' in stem:
        method = 'ucpo'
    elif 'rlep' in stem:
        method = 'rlep'
    elif 'verified_first_replay_rehearsal' in stem:
        method = 'replay'
    elif re.search(r'_control(_|$)', stem) or 'compute_matched' in stem:
        method = 'control'
    else:
        return None
    experiment = re.search(r'_(e\d+[a-z]?\d*)_', stem)
    for token, domain in DOMAIN.items():
        if f'_{token}_' in stem:
            return {'method': method, 'domain': domain, 'seed': int(seed.group(1)),
                    'experiment': experiment.group(1) if experiment else 'unknown',
                    'run': stem}
    return None


def inventory() -> list[dict]:
    files: dict[str, dict] = {}

    def add(path: str) -> None:
        if not path.endswith('.jsonl') or 'eval_mode_coverage_draws' not in path:
            return
        absolute = path if path.startswith('/') else str(ROOT / path)
        cell = identity(absolute)
        job = re.search(r'/(debug_job\d+)/', absolute)
        if cell:
            cell = {**cell, 'job': job.group(1) if job else 'unknown'}
        # Two naming eras for the same 0.5B model: the E97/E98 runs spell it
        # ``qwen25_0p5b`` and the later E115/E116 ones ``qwen25_05b``.
        if cell and cell['method'] != 'rlep' and not any(
                token in absolute for token in ('qwen25_0p5b', 'qwen25_05b')):
            return
        if cell and Path(absolute).is_file():
            files[absolute] = {**cell, 'path': absolute}

    for payload, key in ((UCPO, 'input_sha256'), (SEMANTIC, 'source_sha256')):
        if payload.is_file():
            for path in json.loads(payload.read_text()).get(key, {}):
                add(path)
    # The trained sparse RLEP arm lives in its own run directories rather than
    # in a registered payload, so enumerate those 15 runs by name.
    data = ROOT / 'var/data'
    for run in sorted(data.glob('xdr_qwen25_0p5b_instruct_rlep_*')):
        for draws in sorted(run.glob('*/eval_mode_coverage_draws.jsonl')):
            add(str(draws))
    # Countdown and MathIR came later than the registries above: E115 carries
    # the 0.5B UCPO arm in those two domains and E116 the sparse RLEP-Dr arm,
    # which is why their PCMD was missing while their pass@8 was not.
    for pattern in ('xdr_qwen25_05b_instruct_e115_qwen05b_*_ucpo_s*',
                    'xdr_qwen25_05b_instruct_e116_qwen05b_*_sparse_rlep_s*'):
        for run in sorted(data.glob(pattern)):
            for draws in sorted(run.glob('*/eval_mode_coverage_draws.jsonl')):
                add(str(draws))
    return sorted(files.values(), key=lambda row: (row['method'], row['domain'], row['seed']))


def terminal_prompt_counts(path: str) -> tuple[int, dict[int, Counter]]:
    """Pool a prompt's draws at each step; return the last step's counters."""
    steps: dict[int, dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
    with open(path) as handle:
        for line in handle:
            record = json.loads(line)
            if record.get('evaluation_kind') != SAMPLED:
                continue
            step = record.get('step')
            for index, prompt in enumerate(record.get('prompts') or []):
                position = prompt.get('prompt_index', index)
                for key in prompt.get('answer_keys') or []:
                    if key is not None:
                        steps[step][position][key] += 1
    if not steps:
        return -1, {}
    last = max(steps)
    return last, steps[last]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--min-defined', type=int, default=DEFAULT_MIN_DEFINED_PROMPTS)
    parser.add_argument('--methods', nargs='*', help='restrict to these arms')
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--runs', nargs='*',
                        help='restrict to runs whose name contains one of these')
    parser.add_argument('--merge', action='store_true',
                        help='keep the cells already in --output and update in place; '
                             'reading every draw file costs several GB of shared I/O, '
                             'so a run that only adds cells should not re-read the rest')
    args = parser.parse_args()

    rows = inventory()
    if args.methods:
        rows = [row for row in rows if row['method'] in args.methods]
    if args.runs:
        rows = [row for row in rows
                if any(token in row['run'] for token in args.runs)]
    print(json.dumps({'event': 'inventory', 'files': len(rows)}), flush=True)
    cells: dict[tuple, dict] = {}
    if args.merge and args.output.is_file():
        # Cells written before the job field exists carry no ``job``; the run
        # name already separates the duplicates, so it is optional in the key.
        for cell in json.loads(args.output.read_text())['cells']:
            cells[(cell['method'], cell['domain'], cell['seed'], cell['experiment'],
                   cell['run'], cell.get('job'))] = cell
    for position, row in enumerate(rows, start=1):
        step, prompts = terminal_prompt_counts(row['path'])
        per_prompt = {str(index): mode_diversity(counts)
                      for index, counts in sorted(prompts.items())}
        values = list(per_prompt.values())
        defined = [value for value in values if value is not None]
        key = (row['method'], row['domain'], row['seed'], row['experiment'],
               row['run'], row['job'])
        assert args.merge or key not in cells, f'duplicate cell {key}'
        cells[key] = {
            'method': row['method'], 'domain': row['domain'], 'seed': row['seed'],
            'experiment': row['experiment'], 'run': row['run'], 'job': row['job'],
            'terminal_step': step, 'prompts': len(values),
            'defined_prompts': len(defined),
            'pmd': statistics.fmean(defined) if defined else None,
            'reportable': len(defined) >= args.min_defined,
            # Kept so two arms can be compared on the prompts where both define
            # PCMD, rather than each averaging over its own defined subset.
            'prompt_pmd': {k: v for k, v in per_prompt.items() if v is not None},
            'source': row['path'],
        }
        print(json.dumps({'event': 'cell', 'n': position, 'of': len(rows),
                          'method': row['method'], 'domain': row['domain'],
                          'seed': row['seed'], 'step': step,
                          'defined': len(defined)}), flush=True)
    payload = {
        'schema': 'paper-mode-diversity-comparators-v1',
        'definition': {
            'metric': 'pairwise correct-mode diversity (PCMD)',
            'estimator': 'PCMD = 1 - sum_m n_m (n_m - 1) / (K (K - 1)) on a prompt',
            'aggregation': "pooled over a prompt's draws at the terminal step, "
                           'unweighted mean over defined prompts',
            'min_defined_prompts': args.min_defined,
        },
        'cells': sorted(cells.values(),
                        key=lambda c: (c['method'], c['domain'], c['seed'], c['experiment'])),
    }
    args.output.write_text(json.dumps(payload, indent=1) + '\n')
    print(json.dumps({'event': 'built', 'output': str(args.output),
                      'cells': len(cells)}), flush=True)


if __name__ == '__main__':
    main()
