#!/usr/bin/env python3
"""Export Levels 4 and 5 as additional configs of the published ModeBench dataset.

The sealed Levels 1-3 release is not touched: this writes a separate package of
ten new configs that are added alongside it, so nothing already published is
rewritten or re-derived.

These two levels are frozen and confirmed per domain, and they already carry the
paper's base-model scale figure. They have *not* cleared the held-out
confirmation gate that Levels 1-3 cleared before publication, and they are not
difficulty-matched. Publishing them silently beside Levels 1-3 would let a user
infer a warrant they do not have, so every config records its own
``status`` and ``difficulty_matched`` verbatim from the dataset's identity file,
and the README says what the difference means.

Each split is verified against the row digest its identity froze, then
round-tripped through Parquet and re-read to prove rows, feature types and
column order survive.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'var/artifacts/modebench_levels45_release_20260915/package'
REPO_ID = 'od2961/ModeBench'
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')
L4_MANIFEST = ROOT / 'var/data/modebench_scale_release_v2/level4/source_manifest.json'
L5_ROOT = ROOT / 'var/data/modebench_scale_release_v2/level5/dataset'
ROLES = ('train', 'dev', 'eval')


def require(ok: bool, message: str) -> None:
    if not ok:
        raise SystemExit(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def native(domain: str) -> str:
    return 'pantry' if domain == 'pantry_plan' else domain


def row_digest(rows: list[dict], style: str) -> str:
    if style == 'json_array':
        text = json.dumps(rows, sort_keys=True, separators=(',', ':'), allow_nan=False)
    else:
        text = '\n'.join(json.dumps(r, sort_keys=True, separators=(',', ':'), allow_nan=False)
                         for r in rows)
    return hashlib.sha256(text.encode()).hexdigest()


def domain_root(level: int, domain: str) -> Path:
    if level == 5:
        return L5_ROOT / native(domain)
    sources = json.loads(L4_MANIFEST.read_text())['sources']
    entry = sources.get(native(domain))
    require(entry is not None, f'level4: no admitted source for {domain}')
    root = Path(entry['source_root'])
    for candidate in (root / 'level4' / 'dataset' / native(domain),
                      root / 'dataset' / native(domain)):
        if candidate.is_dir():
            return candidate
    raise SystemExit(f'level4/{domain}: no dataset under admitted source {root}')


def roundtrip(dataset, path: Path) -> dict:
    from datasets import Dataset
    rows, features, columns = list(dataset), dataset.features.to_dict(), dataset.column_names
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f'parquet output exists {path}')
    dataset.to_parquet(str(path), compression='zstd')
    with tempfile.TemporaryDirectory(prefix='modebench45_') as cache:
        recovered = Dataset.from_parquet(str(path), keep_in_memory=True, cache_dir=cache)
    require(list(recovered) == rows, 'Parquet changed rows or order')
    require(recovered.features.to_dict() == features, 'Parquet changed feature types')
    require(recovered.column_names == columns, 'Parquet changed column order')
    return {'rows': len(rows), 'features': features, 'source_column_order': columns,
            'rows_sha256': row_digest(rows, 'json_array'),
            'parquet_sha256': sha(path), 'parquet_bytes': path.stat().st_size}


def export(output: Path) -> dict:
    from datasets import load_from_disk, disable_progress_bar
    disable_progress_bar()
    require(not output.exists(), f'fresh staging directory required {output}')
    output.mkdir(parents=True)
    configs: dict[str, list] = defaultdict(list)
    splits, caveats = [], {}
    for level in (4, 5):
        for domain in DOMAINS:
            root = domain_root(level, domain)
            identity = json.loads((root / 'identity.json').read_text())
            require(identity.get('domain') == native(domain)
                    and identity.get('level') == f'level{level}', f'identity mismatch {root}')
            config = f'level{level}_{domain}'
            caveats[config] = {
                'status': identity.get('status'),
                'difficulty_matched': identity.get('difficulty_matched'),
                'recipe_sha256': identity.get('recipe_sha256'),
                'protocol_sha256': identity.get('protocol_sha256'),
                'exclusions_sha256': identity.get('exclusions_sha256'),
                'source_repository_path': str(root.relative_to(ROOT)),
            }
            for role in ROLES:
                path = root / role
                require(path.is_dir(), f'missing split {path}')
                data = load_from_disk(str(path))
                require(set(data.keys()) <= ({'train'} if role == 'train' else {'multi_answer'}),
                        f'unexpected serialized split {path}')
                for subset, dataset in data.items():
                    rows = list(dataset)
                    require(rows and all(isinstance(r.get('problem'), str)
                                         and isinstance(r.get('answer'), str) for r in rows),
                            f'invalid row surface {path}')
                    specs = [json.loads(r['answer']) for r in rows]
                    require(all(isinstance(s, dict) and isinstance(s.get('verifier'), str)
                                for s in specs), f'missing executable specification {path}')
                    frozen = (identity.get('splits') or {}).get(role)
                    require(frozen is not None, f'identity records no {role} split for {config}')
                    require(frozen['rows'] == len(rows),
                            f'frozen row count differs {config}/{role}')
                    digests = {row_digest(rows, s) for s in
                               ('json_array', 'json_lines_no_final_newline')}
                    require(frozen['rows_sha256'] in digests,
                            f'frozen row digest differs {config}/{role}')
                    relative = f'data/{config}/{role}.parquet'
                    result = roundtrip(dataset, output / relative)
                    splits.append({'config_name': config, 'level': level, 'domain': domain,
                                   'split': role, 'original_datasetdict_split': subset,
                                   'source_repository_path': str(path.relative_to(ROOT)),
                                   'data_file': relative, **result,
                                   'frozen_identity_record': frozen,
                                   'support_histogram': dict(sorted(Counter(
                                       str(r['answer_mode_count']) for r in rows).items())),
                                   'verifier_values': sorted({s['verifier'] for s in specs}),
                                   **caveats[config]})
                    configs[config].append({'split': role, 'path': relative})
    manifest = {
        'schema': 'modebench-levels45-extension-manifest-v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'repo_id': REPO_ID,
        'extends': 'the sealed Levels 1-3 release; nothing already published is rewritten',
        'config_count': len(configs), 'split_count': len(splits),
        'row_count': sum(s['rows'] for s in splits),
        'configs': {name: sorted(items, key=lambda i: i['split']) for name, items in configs.items()},
        'splits': splits,
        'admission_caveat': {
            'applies_to': sorted(configs),
            'status': 'frozen_pending_heldout_confirmation',
            'difficulty_matched': False,
            'meaning': ('these levels are frozen and confirmed per domain, and they back the '
                        'published base-model scale figure, but they have not cleared the '
                        'held-out confirmation that Levels 1-3 cleared before publication and '
                        'they are not difficulty-matched against a fixed reference'),
        },
    }
    (output / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    manifest = export(args.output)
    print(json.dumps({k: manifest[k] for k in
                      ('config_count', 'split_count', 'row_count')} |
                     {'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
