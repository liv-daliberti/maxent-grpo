#!/usr/bin/env python3
"""Pin the untrained baselines behind the paper's before/after concentration claim.

The before-state in that comparison is not "the base model" in the abstract: it
is the base model answering through the training run's own prompt surface and
generation budget. That distinction is load-bearing. Under the native-chat
interface the base grid solves 40-82% of Python Factors prompts, and under the
training surface the same checkpoints solve none at all, so a before-value read
from one cannot be differenced against a terminal state measured in the other.

Two problems are fixed by re-collecting these. The saved step-0 draws carry the
same eleven-stream seeding as every other training evaluation. And Countdown and
MathIR sit just under the support bar -- roughly 24 of the 30 prompts a cell
needs -- which a larger sampling budget lifts, because PCMD is unbiased at every
K>=2 and more draws simply resolve more prompts that admit the measure at all.

The base model is a property of scale, and the surface a property of the domain,
so this is fifteen cells rather than one per training run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_terminal_cohort_manifest as base  # noqa: E402
from draw_tail import terminal_draws  # noqa: E402

SCHEMA = 'pmd-independent-resample-step0-cohort-v1'
OUT = ROOT / 'var/artifacts/pmd_independent_resample_20260915/step0_cohort_manifest.json'
LEDGER_DIRS = (ROOT / 'paper/audits/training_curves_20260912/ledgers', ROOT / 'var/artifacts')
PRETRAIN = re.compile(r'OAT_ZERO_PRETRAIN=([^,\s]+)')
STEP0 = 0


def pretrain_of(run_dir: str) -> str | None:
    for directory in LEDGER_DIRS:
        for path in sorted(Path(directory).glob('*jobs.json')):
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            for run in (payload.get('runs') or []):
                if run.get('run_dir') != run_dir:
                    continue
                hit = PRETRAIN.search(str(run.get('held_scheduler_record', '')))
                if hit:
                    return hit.group(1)
    return None


def build(draws: int, k: int) -> dict:
    manifest = json.loads((OUT.parent / 'cohort_manifest.json').read_text())
    surfaces: dict[tuple[str, str], dict] = {}
    for cell in manifest['cells']:
        surfaces.setdefault((cell['scale'], cell['domain']), cell)
    cells = []
    for (scale, domain), template in sorted(surfaces.items()):
        pretrain = pretrain_of(template['run_dir'])
        base.require(pretrain is not None, f'{scale}/{domain}: no recorded base model')
        export = Path(pretrain)
        base.require(export.is_dir(), f'base model directory missing: {export}')
        draws_path = Path(template['draws_path'])
        # Step-0 records sit at the head of the draw log, outside the bounded
        # tail this reads. The evaluation set does not change during a run, so
        # the prompts are taken from the terminal record and then checked
        # against the terminal manifest's digest below.
        saved = terminal_draws(draws_path, int(template['terminal_step']))
        base.require(saved, f'{scale}/{domain}: no saved draws to take prompts from')
        rows = sorted(({'prompt_index': int(p['prompt_index']), 'problem': p['prompt'],
                        'reference': p['reference']} for p in saved[0]['prompts']),
                      key=lambda r: r['prompt_index'])
        digest = base.sha_text(json.dumps(rows, sort_keys=True, separators=(',', ':')))
        base.require(digest == template['prompt_set_sha256'],
                     f'{scale}/{domain}: step-0 prompt set differs from the terminal one')
        config = dict(template['eval_config'])
        # A wider budget than the run used. PCMD is unbiased at every K >= 2, so
        # this estimates the same quantity and resolves more prompts that admit
        # it; pass@8 and distinct@8 are budget-dependent and are not comparable
        # against the registered values at this budget.
        config['eval_mode_coverage_draws'] = draws
        config['eval_mode_coverage_k'] = k
        cells.append({
            'scale': scale, 'level': base.LEVEL, 'domain': domain, 'method': 'base',
            'seed': 0, 'run_dir': template['run_dir'], 'registered_job_id': None,
            'terminal_step': STEP0, 'draws_path': str(draws_path),
            'gpu': template['gpu'], 'gpu_source': 'inherited from the training surface',
            'canonical_action_task': template['canonical_action_task'],
            'prompt_count': len(rows), 'prompt_set_sha256': digest,
            'registered_step0_seed_base': int(template['eval_config']['eval_mode_coverage_seed']),
            'eval_config': config,
            'weights': {'location': 'local', 'export_dir': str(export),
                        'files': [{'relative_path': p.name} for p in
                                  sorted(list(export.glob('*.safetensors'))
                                         + list(export.glob('pytorch_model*.bin')))]},
            'registered_terminal_reference': None,
        })
    return {
        'schema': SCHEMA, 'level': base.LEVEL, 'terminal_step': STEP0,
        'purpose': ('re-measure the untrained baselines through each training run\'s own '
                    'prompt surface, with disjoint sampling streams and a wider budget'),
        'budget': {'draws': draws, 'k': k,
                   'note': 'pass@8 and distinct@8 at this budget are not comparable to the '
                           'registered eight-sample values; PCMD is.'},
        'builder': {'path': str(Path(__file__).resolve().relative_to(ROOT)),
                    'sha256': base.file_sha(Path(__file__).resolve())},
        'cell_count': len(cells), 'cells': cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draws', type=int, default=8)
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    payload = build(args.draws, args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'cells': payload['cell_count'], 'budget': payload['budget'],
                      'samples_per_prompt': args.draws * args.k,
                      'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
