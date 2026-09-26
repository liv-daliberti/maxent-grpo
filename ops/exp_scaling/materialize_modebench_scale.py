#!/usr/bin/env python3
"""Prepare fresh Level 4/7B and Level 5/14B calibration and train/test splits.

Model difficulty is a measured property: generated data remain unconfirmed
until a frozen development recipe passes fresh held-out evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
for directory in ('ops', 'ops/exp_scaling', 'src'):
    sys.path.insert(0, str(ROOT / directory))
from fit_modebench_level3 import (atomic_new, sha, file_sha, cell_histogram,
                                 serialize_cells, allocate_cells, select_rows)
from materialize_modebench_harder_v2 import (
    DOMAINS, SPLITS, LEVEL1, existing_ids, identity_set, load_rows,
)

DEFAULT_ROOT = ROOT / 'var/data/modebench_scale_v1'
REFERENCE = ROOT / 'var/data/modebench_level3_matched_v3'
LEVELS = {'level4': '7b', 'level5': '14b'}
MODEL_REVISIONS = {'7b': ('7B', 'a09a35458c702b33eeacc393d103063234e8bc28'),
                   '14b': ('14B', 'cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8')}
SCHEMA = 'modebench_scale_protocol_v1'
SELECTION_SEED = 6491701
TOLERANCES = {'pass1': .04, 'pass8': .08}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def tuples(value):
    return tuple(tuples(x) for x in value) if isinstance(value, list) else value


def labels(level, phase):
    require(level in LEVELS and phase in ('dev', 'eval'), 'unknown level or phase')
    base = (6548000 if level == 'level4' else 6558000) + (1000 if phase == 'eval' else 0)
    return list(range(base, base + 4))


def generation_seed(level, domain, split, tier):
    return int(sha([SCHEMA, level, domain, split, tier])[:12], 16)


def deserialize_cells(values):
    return Counter({tuple(item['cell']): item['rows'] for item in values})


def union_histogram(histograms):
    # Include training-only cells, but development/eval are the fitted margins.
    keys = set().union(*histograms.values())
    return Counter({key: max(histograms['dev'].get(key, 0),
                             histograms['eval'].get(key, 0),
                             (histograms['train'].get(key, 0) + 2) // 3)
                    for key in keys})


def source_pins():
    import modebench_scale_candidates as candidates
    paths = set(candidates.source_paths()) | {Path(__file__),
        ROOT / 'ops/exp_scaling/fit_modebench_level3.py',
        ROOT / 'ops/exp_scaling/modebench_level3_allocation.py',
        ROOT / 'ops/exp_scaling/materialize_modebench_harder_v2.py',
        ROOT / 'ops/exp_scaling/fit_modebench_scale.py',
        ROOT / 'ops/evaluate_modebench_scale.py',
        ROOT / 'ops/exp_scaling/materialize_e117_evaluation_reserves.py',
        ROOT / 'ops/exp_scaling/materialize_modebench_level3.py',
        ROOT / 'var/data/pantry_plan_v1/ingredients.json'}
    paths.update((ROOT / 'src/oat_drgrpo').glob('*.py'))
    import evaluate_modebench_scale as evaluator
    paths.update(ROOT / path for path in evaluator.code_identity())
    return {str(Path(p).resolve()): file_sha(p) for p in sorted(paths)}


def authenticate(protocol_path):
    protocol = read(protocol_path)
    require(protocol.get('schema') == SCHEMA, 'unknown scale protocol')
    for path, digest in protocol['files_sha256'].items():
        require(file_sha(path) == digest, 'registered source/input changed: ' + path)
    return protocol


def history(domain, exclude_root):
    """Snapshot semantic IDs and exact prompts from all historical level data."""
    from datasets import load_from_disk
    from materialize_e117_evaluation_reserves import _load_source_rows
    blocked, prompts, pins, seen = set(existing_ids(domain)), set(), {}, set()
    def add(rows):
        blocked.update(identity_set(domain, rows))
        prompts.update(sha(row['problem']) for row in rows)
    if domain != 'pantry':
        add(_load_source_rows(domain))
    for path in LEVEL1[domain].values():
        seen.add(Path(path).resolve())
    if domain == 'pantry':
        for root in (ROOT / 'var/data').glob('pantry_plan_modebench*'):
            for marker in root.glob('**/dataset_dict.json'):
                seen.add(marker.parent.resolve())
    roots = [p for p in (ROOT / 'var/data').glob('modebench*')
             if p.is_dir() and p.resolve() != Path(exclude_root).resolve()]
    for root in roots:
        for marker in root.glob('**/' + domain + '/**/dataset_dict.json'):
            seen.add(marker.parent.resolve())
        for path in root.glob('**/pools/' + domain + '/*.jsonl'):
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            add(rows)
            pins[str(path.resolve())] = file_sha(path)
    for path in sorted(seen):
        dataset = load_from_disk(str(path))
        for subset in dataset.values():
            add([dict(row) for row in subset])
        pins.update({str(p.resolve()): file_sha(p) for p in path.rglob('*') if p.is_file()})
    return blocked, prompts, pins


def prepare(campaign):
    campaign = Path(campaign).resolve()
    require(not (campaign / 'protocol.json').exists(), 'protocol already exists')
    require(not campaign.exists(), 'use a fresh campaign directory')
    import modebench_scale_candidates as candidates
    from modebench_level3_v3_common import TARGET_SHA
    import evaluate_modebench_level3_independent as evaluator
    from evaluate_modebench_level3 import summarize
    targets, histograms, histories, pins = {}, {}, {}, source_pins()
    models = {}
    for label, (size, revision) in MODEL_REVISIONS.items():
        path = ROOT / 'var/cache/huggingface/transformers' / ('models--Qwen--Qwen2.5-' + size + '-Instruct') / 'snapshots' / revision
        index = read(path / 'model.safetensors.index.json')
        require(all((path / name).is_file() for name in set(index['weight_map'].values())), 'complete model checkpoint required: ' + label)
        models[label] = evaluator.model_identity(path, label)
    for domain in DOMAINS:
        receipt_path = ROOT / ('var/results/modebench_level3_v2/confirmation_python_v6/'
                              'confirmation_05b_' + domain + '.json')
        require(file_sha(receipt_path) == TARGET_SHA[domain], 'fixed Level 1 reference changed')
        receipt = read(receipt_path)
        evaluator.validate_seed_receipt(receipt)
        require(receipt['metrics'] == summarize(receipt['prompt_results']), 'reference metrics disagree')
        targets[domain] = {'model': 'Qwen/Qwen2.5-0.5B-Instruct', 'level': 'level1',
                           'receipt': str(receipt_path), 'receipt_sha256': file_sha(receipt_path),
                           'metrics': {key: receipt['metrics'][key] for key in TOLERANCES}}
        pins[str(receipt_path)] = file_sha(receipt_path)
        histograms[domain] = {}
        for split, (count, subset) in SPLITS.items():
            rows = load_rows(REFERENCE / domain / split, subset)
            require(len(rows) == count, 'reference split size changed')
            histograms[domain][split] = serialize_cells(cell_histogram(domain, rows))
        blocked, prompts, historical_pins = history(domain, campaign)
        histories[domain] = {'identities': sorted(blocked, key=sha),
                             'prompt_sha256': sorted(prompts)}
        pins.update(historical_pins)
        print(json.dumps({'event': 'history', 'domain': domain, 'identities': len(blocked)}), flush=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + campaign.name + '.', dir=campaign.parent))
    try:
        for domain, payload in histories.items():
            atomic_new(staging / 'history' / (domain + '.json'), payload)
            pins[str(campaign / 'history' / (domain + '.json'))] = file_sha(staging / 'history' / (domain + '.json'))
        protocol = {'schema': SCHEMA, 'created_utc': datetime.now(timezone.utc).isoformat(),
                    'levels': LEVELS, 'models': models, 'targets': targets, 'histograms': histograms,
                    'split_sizes': {key: value[0] for key, value in SPLITS.items()},
                    'candidate_profiles': candidates.PROFILES, 'tolerances': TOLERANCES,
                    'selection_seed': SELECTION_SEED, 'files_sha256': pins,
                    'draw_labels': {level: {phase: labels(level, phase) for phase in ('dev', 'eval')}
                                    for level in LEVELS},
                    'sampling': {'samples_per_draw': 8, 'draws_per_problem': 4,
                                 'temperature': 1.0, 'top_p': 1.0, 'max_tokens': 192,
                                 'dtype': 'float16', 'chat_template': 'model_native',
                                 'interface': 'level2_qwen_r5 with independent per-prompt draw blocks'},
                    'fit': 'full-pool cell forecasts over dev/eval histograms; grid20; one fixed dev selection',
                    'boundary': {'target': 'historical fixed empirical Level1 controls, not fresh two-sided confirmation',
                                 'test': 'fresh held-out eval after development recipe freeze',
                                 'training': 'dataset splits only; no treatment training',
                                 'match_claim': 'requires both held-out gates in every domain'}}
        atomic_new(staging / 'protocol.json', protocol)
        staging.rename(campaign)
        return {'protocol': str(campaign / 'protocol.json'), 'sha256': file_sha(campaign / 'protocol.json')}
    except BaseException:
        shutil.rmtree(staging)
        raise


@contextmanager
def domain_lock(campaign, domain):
    path = Path(campaign) / ('.' + domain + '.lock')
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def current_exclusions(campaign, domain):
    value = read(Path(campaign) / 'history' / (domain + '.json'))
    blocked = {tuples(item) for item in value['identities']}
    prompts = set(value['prompt_sha256'])
    for path in Path(campaign).glob('level*/pools/' + domain + '/*.jsonl'):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        blocked.update(identity_set(domain, rows))
        prompts.update(sha(row['problem']) for row in rows)
    for path in Path(campaign).glob('level*/dataset/' + domain + '/*.jsonl'):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        blocked.update(identity_set(domain, rows))
        prompts.update(sha(row['problem']) for row in rows)
    return blocked, prompts


def generate(domain, target, blocked, level, split, tier):
    import modebench_scale_candidates as candidates
    marginal = Counter()
    for key, count in target.items():
        marginal[key[0]] += count
    extra = {'joint_target': target} if domain == 'pantry' else {}
    return candidates.build_pool(domain, marginal, blocked,
        generation_seed(level, domain, split, tier), level + '_' + split,
        tier, multiplier=1, **extra)


def verify(domain, rows, target, blocked, prompts):
    ids = identity_set(domain, rows)
    require(len(rows) == sum(target.values()) == len(ids), 'row count or identity duplicate')
    require(not ids & blocked, 'semantic identity overlaps earlier data')
    texts = {sha(row['problem']) for row in rows}
    require(len(texts) == len(rows) and not texts & prompts, 'exact prompt overlap')
    require(cell_histogram(domain, rows) == target, 'exact support/family histogram mismatch')
    import modebench_scale_candidates as candidates
    certificate = candidates.verify_rows(domain, rows)
    return {'rows': len(rows), 'rows_sha256': sha(rows), 'semantic_disjoint': True,
            'prompt_disjoint': True, 'cells': serialize_cells(target), 'verification': certificate}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')


def materialize_pools(campaign, level, domain):
    campaign = Path(campaign).resolve()
    protocol = authenticate(campaign / 'protocol.json')
    require(level in LEVELS and domain in DOMAINS, 'invalid level/domain')
    histograms = {s: deserialize_cells(h) for s, h in protocol['histograms'][domain].items()}
    target = union_histogram(histograms)
    with domain_lock(campaign, domain):
        directory = campaign / level / 'pools' / domain
        require(not directory.exists(), 'pool directory exists; never overwrite a partial generation')
        blocked, prompts = current_exclusions(campaign, domain)
        staging = Path(tempfile.mkdtemp(prefix='.' + domain + '.', dir=campaign))
        try:
            records = {}
            for tier in range(4):
                rows = generate(domain, target, blocked, level, 'development', tier)
                records[str(tier)] = verify(domain, rows, target, blocked, prompts)
                records[str(tier)]['generation_seed'] = generation_seed(level, domain, 'development', tier)
                write_jsonl(staging / f'difficulty_{tier}.jsonl', rows)
                blocked.update(identity_set(domain, rows))
                prompts.update(sha(row['problem']) for row in rows)
                print(json.dumps({'event': 'pool', 'level': level, 'domain': domain,
                                  'tier': tier, 'rows': len(rows)}), flush=True)
            atomic_new(staging / 'identity.json', {'schema': 'modebench_scale_development_pools_v1',
                'level': level, 'domain': domain, 'protocol_sha256': file_sha(campaign / 'protocol.json'),
                'status': 'verified_candidates_pending_model_calibration', 'tiers': records})
            directory.parent.mkdir(parents=True, exist_ok=True)
            staging.rename(directory)
            return records
        except BaseException:
            shutil.rmtree(staging)
            raise


def freeze_dataset(campaign, level, domain):
    """Materialize train/dev/test from the sole passing development recipe."""
    from datasets import Dataset, DatasetDict
    from fit_modebench_scale import fit_domain
    campaign = Path(campaign).resolve()
    protocol = authenticate(campaign / 'protocol.json')
    recipe_path = campaign / level / 'recipes' / (domain + '.json')
    recipe = read(recipe_path)
    reconstructed = fit_domain(campaign, level, domain, publish=False)
    require(recipe == reconstructed and recipe['development_fit_pass'], 'passing reproducible recipe required')
    with domain_lock(campaign, domain):
        destination = campaign / level / 'dataset' / domain
        require(not destination.exists(), 'frozen dataset already exists')
        blocked, prompts = current_exclusions(campaign, domain)
        pools = {tier: [json.loads(line) for line in
                 (campaign / level / 'pools' / domain / f'difficulty_{tier}.jsonl').read_text().splitlines()]
                 for tier in range(4)}
        histograms = {s: deserialize_cells(h) for s, h in protocol['histograms'][domain].items()}
        selection = select_rows(domain, pools, histograms['dev'], recipe['weights'], SELECTION_SEED)
        built = {'dev': [item['row'] for item in selection]}
        records = {'dev': {'rows': 128, 'rows_sha256': sha(built['dev']),
                           'origin': 'sole fixed selected development rows; model scored for calibration'}}
        for split in ('train', 'eval'):
            allocation = allocate_cells(histograms[split], recipe['weights'], SELECTION_SEED)
            rows = []
            for tier in range(4):
                target = Counter({cell: counts[tier] for cell, counts in allocation.items() if counts[tier]})
                if not target:
                    continue
                generated = generate(domain, target, blocked, level, split, tier)
                verify(domain, generated, target, blocked, prompts)
                rows.extend(generated)
                blocked.update(identity_set(domain, generated))
                prompts.update(sha(row['problem']) for row in generated)
            rows.sort(key=lambda row: sha([level, domain, split, sha(row)]))
            require(cell_histogram(domain, rows) == histograms[split], 'split histogram drift')
            built[split] = rows
            records[split] = {'rows': len(rows), 'rows_sha256': sha(rows), 'fresh': True,
                              'cells': serialize_cells(histograms[split])}
        staging = Path(tempfile.mkdtemp(prefix='.' + domain + '-dataset.', dir=campaign))
        try:
            for split, rows in built.items():
                write_jsonl(staging / (split + '.jsonl'), rows)
                DatasetDict({SPLITS[split][1]: Dataset.from_list(rows)}).save_to_disk(str(staging / split))
                require(load_rows(staging / split, SPLITS[split][1]) == rows, 'Arrow serialization changed frozen row identity')
            atomic_new(staging / 'identity.json', {'schema': 'modebench_scale_frozen_domain_v1',
                'status': 'frozen_pending_heldout_confirmation', 'level': level, 'domain': domain,
                'protocol_sha256': file_sha(campaign / 'protocol.json'),
                'recipe_sha256': file_sha(recipe_path), 'splits': records,
                'test_split': 'eval', 'difficulty_matched': False})
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging.rename(destination)
            return str(destination)
        except BaseException:
            shutil.rmtree(staging)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'pools', 'freeze'))
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--level', choices=LEVELS)
    parser.add_argument('--domain', choices=DOMAINS)
    args = parser.parse_args()
    if args.action == 'prepare':
        result = prepare(args.root)
    else:
        require(args.level and args.domain, '--level and --domain required')
        result = (materialize_pools if args.action == 'pools' else freeze_dataset)(
            args.root, args.level, args.domain)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
