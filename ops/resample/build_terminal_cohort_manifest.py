#!/usr/bin/env python3
"""Pin the Level-1 terminal cells that the independent resampling will re-measure.

The registered terminal draws seed draw ``d`` at ``seed_base + d`` and vLLM 0.8.4
V0 expands an ``n=8`` request with seed ``s`` into children ``s..s+7``, so four
consecutive draws span eleven nominal streams rather than thirty-two. This
manifest binds everything the resampler needs to re-measure those same cells
under the already-registered aligned-block seed policy: the evaluation config
each run inherited, where its terminal weights now live, and the original draw
seeds so the reproduction check can prove the offline harness is faithful before
any new number is believed.

Nothing here samples or grades. It reads the frozen cohort record, a bounded
tail of each run's draw log, and the E72 source manifest the launchers resolved
their evaluation config from.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from draw_tail import terminal_draws  # noqa: E402

COHORT = ROOT / 'paper/audits/conditional_concentration_20260912/verified_samples_completed_cohort.jsonl.gz'
E72 = ROOT / 'var/artifacts/e72_frontier_source_runs.json'
LEDGERS = ROOT / 'paper/audits/training_curves_20260912/ledgers'
OUT = ROOT / 'var/artifacts/pmd_independent_resample_20260915/cohort_manifest.json'

SCHEMA = 'pmd-independent-resample-cohort-v1'
TERMINAL_STEP = 3072
LEVEL = 'level1'
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')
METHODS = ('drgrpo', 'replay_drgrpo', 'maxrl', 'replay_maxrl')
SCALES = ('qwen05b', 'falcon1b', 'qwen3b')

# Falcon runs use the pure surface swap of the Qwen contract; Qwen scales keep
# the template the E72 reference run inherited.
TWINS = {
    'qwen_boxed': 'falcon_boxed',
    'qwen_countdown_digits': 'falcon_countdown_digits',
    'qwen_graph_digits': 'falcon_graph_digits',
    'qwen_pantry_support_mask': 'falcon_pantry_support_mask',
    'qwen_math': 'falcon_math',
    'qwen_math_route': 'falcon_math_route',
}
CONFIG_KEYS = ('prompt_template', 'eval_generate_max_length', 'max_model_len',
               'prompt_max_length', 'eval_batch_size', 'eval_data',
               'eval_mode_coverage_seed', 'eval_mode_coverage_k',
               'eval_mode_coverage_draws', 'eval_mode_coverage_temperature',
               'vllm_gpu_ratio')


def require(ok: bool, message: str) -> None:
    if not ok:
        raise SystemExit(message)


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def domain_eval_config() -> dict[str, dict]:
    """One evaluation config per domain, as the E72 reference runs recorded it."""
    runs = json.loads(E72.read_text())['runs']
    configs: dict[str, dict] = {}
    for domain in DOMAINS:
        selected = [r for r in runs if r['arm'] == 'xgrpo' and r['domain'] == domain]
        require(selected, f'no E72 reference run for {domain}')
        distinct = {json.dumps({k: r['inherited_eval_config'][k] for k in CONFIG_KEYS},
                               sort_keys=True) for r in selected}
        require(len(distinct) == 1,
                f'{domain}: E72 reference runs disagree on the evaluation config')
        configs[domain] = json.loads(distinct.pop())
    return configs


def scale_template(scale: str, qwen_template: str) -> str:
    if scale != 'falcon1b':
        return qwen_template
    require(qwen_template in TWINS, f'no Falcon surface twin for {qwen_template}')
    return TWINS[qwen_template]


def ledger_gpu() -> dict[str, str]:
    """Map run_dir to the GPU model it trained on, so the resample can pin it."""
    placement: dict[str, str] = {}
    for path in sorted(LEDGERS.glob('*.json')):
        payload = json.loads(path.read_text())
        for run in payload.get('runs', []):
            gpu = run.get('gpu')
            if gpu and run.get('run_dir'):
                placement[str(run['run_dir'])] = str(gpu)
    return placement


def terminal_draws_path(run_dir: Path, cell: dict) -> Path:
    """Pick the draw log by provenance, never by glob order or by metric.

    A requeued run leaves more than one ``debug_job*`` attempt behind and only
    one of them trained to the terminal step. ``TRAINING_COMPLETE.json`` names
    that attempt, and the frozen cohort record independently names the file its
    registered analysis read; both must agree before the cell is admitted.
    """
    complete = json.loads((run_dir / 'TRAINING_COMPLETE.json').read_text())
    path = Path(complete['terminal_attempt']) / 'eval_mode_coverage_draws.jsonl'
    require(path.is_file(), f'terminal attempt has no draw log: {path}')
    origins = {Path(origin['path'])
               for group in cell['checkpoints'][str(TERMINAL_STEP)]['origins']
               for origin in group['origins']}
    require(origins == {path},
            f'{run_dir}: cohort read {sorted(map(str, origins))}, '
            f'but the completion receipt names {path}')
    return path


def weights_record(run_dir: Path) -> dict:
    """Where the terminal weights are now: retired to the Hub, or still local."""
    receipt = run_dir / 'MODEL_ARCHIVE.json'
    complete = json.loads((run_dir / 'TRAINING_COMPLETE.json').read_text())
    export = Path(complete['terminal_export'])
    require(export.is_dir(), f'terminal export directory is gone: {export}')
    for required in ('config.json', 'tokenizer_config.json'):
        require((export / required).is_file(), f'{export} lacks {required}')
    if receipt.is_file():
        archive = json.loads(receipt.read_text())
        require(archive['status'] == 'retired', f'unexpected archive status in {receipt}')
        require(Path(archive['original_terminal_export']) == export,
                f'archive receipt points at another export: {receipt}')
        return {
            'location': 'hub',
            'export_dir': str(export),
            'repo_id': archive['repo_id'],
            'repo_prefix': archive['repo_prefix'],
            'commit_sha': archive['commit_sha'],
            'files': [{'relative_path': f['relative_path'], 'sha256': f['sha256'],
                       'bytes': f['bytes']} for f in archive['removed_files']],
        }
    present = sorted(p.name for p in export.glob('*.safetensors'))
    present += sorted(p.name for p in export.glob('pytorch_model*.bin'))
    require(present, f'no local weights and no archive receipt for {export}')
    return {'location': 'local', 'export_dir': str(export), 'files':
            [{'relative_path': name} for name in present]}


def build() -> dict:
    configs = domain_eval_config()
    gpus = ledger_gpu()
    cells = []
    prompt_sets: dict[tuple[str, str], str] = {}
    with gzip.open(COHORT, 'rt') as handle:
        records = [json.loads(line) for line in handle]
    cohort = [r for r in records if r.get('record_kind') == 'cell'
              and r.get('in_terminal_paired_cohort') and r.get('level') == LEVEL]
    require(cohort, 'no Level-1 terminal paired cells in the cohort archive')
    for cell in sorted(cohort, key=lambda c: (c['scale'], c['domain'], c['method'], c['seed'])):
        scale, domain, method = cell['scale'], cell['domain'], cell['method']
        require(scale in SCALES and domain in DOMAINS and method in METHODS,
                f'unexpected cohort cell {scale}/{domain}/{method}')
        run_dir = Path(cell['run_dir'])
        draws_path = terminal_draws_path(run_dir, cell)
        draws = terminal_draws(draws_path, TERMINAL_STEP)
        config = dict(configs[domain])
        require(len(draws) == config['eval_mode_coverage_draws'],
                f'{run_dir}: expected {config["eval_mode_coverage_draws"]} terminal draws, '
                f'read {len(draws)}')
        seed_base = int(config['eval_mode_coverage_seed'])
        observed = [int(d['seed']) for d in draws]
        require(observed == [seed_base + i for i in range(len(draws))],
                f'{run_dir}: recorded draw seeds {observed} are not the inherited schedule')
        k = int(config['eval_mode_coverage_k'])
        for draw in draws:
            require(int(draw['sample_count']) == k, f'{run_dir}: sample count drift')
            require(float(draw['temperature']) == float(config['eval_mode_coverage_temperature']),
                    f'{run_dir}: temperature drift')
            require(float(draw.get('top_p', 1.0)) == 1.0, f'{run_dir}: unexpected nucleus truncation')
        prompts = [{'prompt_index': int(p['prompt_index']), 'problem': p['prompt'],
                    'reference': p['reference']} for p in draws[0]['prompts']]
        prompts.sort(key=lambda p: p['prompt_index'])
        require([p['prompt_index'] for p in prompts] == list(range(len(prompts))),
                f'{run_dir}: terminal prompt indices are not a dense range')
        digest = sha_text(json.dumps(prompts, sort_keys=True, separators=(',', ':')))
        key = (scale, domain)
        if key in prompt_sets:
            require(prompt_sets[key] == digest,
                    f'{scale}/{domain}: cells disagree on the terminal prompt set')
        else:
            prompt_sets[key] = digest
        cells.append({
            'scale': scale, 'level': LEVEL, 'domain': domain, 'method': method,
            'seed': int(cell['seed']), 'run_dir': str(run_dir),
            'registered_job_id': cell.get('registered_job_id'),
            'terminal_step': TERMINAL_STEP,
            'draws_path': str(draws_path),
            'gpu': gpus.get(str(run_dir)),
            'prompt_count': len(prompts),
            'prompt_set_sha256': digest,
            'original_draw_seeds': observed,
            'eval_config': {**config, 'prompt_template': scale_template(scale, config['prompt_template'])},
            'weights': weights_record(run_dir),
            'registered_terminal_reference': cell.get('terminal_reference'),
        })
    counts: dict[str, int] = defaultdict(int)
    for cell in cells:
        counts[f'{cell["scale"]}/{cell["method"]}'] += 1
    return {
        'schema': SCHEMA,
        'level': LEVEL,
        'terminal_step': TERMINAL_STEP,
        'purpose': ('re-measure the Level-1 terminal cells under the registered aligned-block '
                    'seed policy, so PCMD rests on thirty-two disjoint streams per prompt '
                    'rather than eleven overlapping ones'),
        'sources': {
            'cohort': {'path': str(COHORT.relative_to(ROOT)), 'sha256': file_sha(COHORT)},
            'e72_source_runs': {'path': str(E72.relative_to(ROOT)), 'sha256': file_sha(E72)},
        },
        'builder': {'path': str(Path(__file__).resolve().relative_to(ROOT)),
                    'sha256': file_sha(Path(__file__).resolve())},
        'cell_count': len(cells),
        'cells_by_scale_method': dict(sorted(counts.items())),
        'prompt_set_sha256_by_scale_domain': {f'{s}/{d}': v for (s, d), v in sorted(prompt_sets.items())},
        'cells': cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    payload = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps({'cells': payload['cell_count'],
                      'by_scale_method': payload['cells_by_scale_method'],
                      'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
