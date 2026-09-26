"""Read-only, current-source disjointness for frozen scale domain datasets.

Each external source is compared separately. Equal bytes in another source do
not become an exemption; only canonical aliases of the dataset are skipped,
and only selected DEV rows may also occur in the canonical own candidate pools.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in ('ops', 'ops/exp_scaling', 'src'):
    if str(ROOT / directory) not in sys.path:
        sys.path.insert(0, str(ROOT / directory))
from datasets import DatasetDict, load_from_disk
from fit_modebench_level3 import cell_histogram, file_sha, sha
from materialize_modebench_harder_v2 import DOMAINS, LEVEL1, SPLITS, identity_set
from materialize_e117_evaluation_reserves import SOURCE_ROOTS as NATIVE_SOURCE_ROOTS

SCHEMA = 'modebench_scale_current_source_disjointness_v1'
DATASET_SCHEMAS = {'modebench_scale_frozen_domain_v1',
                   'modebench_scale_domain_revision_dataset_v1'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def inside(path, directory):
    return Path(path).resolve().is_relative_to(Path(directory).resolve())


def file_pins(path):
    path = Path(path).resolve()
    files = [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file())
    require(files, 'missing or empty source: ' + str(path))
    return {str(p.resolve()): file_sha(p) for p in files}


def check_pins(pins):
    for path, expected in pins.items():
        require(Path(path).is_file() and file_sha(path) == expected,
                'source changed during disjointness audit: ' + path)


def jsonl_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def row_sets(domain, rows, source):
    require(all(isinstance(row, dict) and isinstance(row.get('problem'), str)
                and 'answer' in row for row in rows), 'invalid source rows: ' + str(source))
    return identity_set(domain, rows), {sha(row['problem']) for row in rows}


def discover_sources(data_root, level, domain, search_root):
    """Canonical source -> kind, retaining distinct copies with equal contents."""
    sources = {}
    def add(path, kind):
        path = Path(path)
        if kind == 'dataset':
            require((path / 'dataset_dict.json').is_file(), 'missing historical dataset: ' + str(path))
        else:
            require(path.is_file(), 'missing historical JSONL: ' + str(path))
        canonical = path.resolve()
        require(canonical not in sources or sources[canonical] == kind, 'ambiguous source kind')
        sources[canonical] = kind
    for path in LEVEL1[domain].values():
        add(path, 'dataset')
    native = NATIVE_SOURCE_ROOTS.get(domain)
    if native is not None:
        require(Path(native).is_dir(), 'missing native historical root: ' + str(native))
        for marker in Path(native).glob('**/dataset_dict.json'):
            add(marker.parent, 'dataset')
    roots = [p for p in Path(search_root).glob('modebench*') if p.is_dir()]
    if domain == 'pantry':
        roots.extend(p for p in Path(search_root).glob('pantry_plan_modebench*') if p.is_dir())
        for campaign in roots:
            if campaign.name.startswith('pantry_plan_modebench'):
                for marker in campaign.glob('**/dataset_dict.json'):
                    add(marker.parent, 'dataset')
    for campaign in roots:
        for marker in campaign.glob('**/' + domain + '/**/dataset_dict.json'):
            add(marker.parent, 'dataset')
        for path in campaign.glob('**/' + domain + '/*.jsonl'):
            add(path, 'jsonl')
    # An explicitly supplied source may live outside the global search tree.
    own_pools = Path(data_root) / level / 'pools' / domain
    for path in own_pools.glob('*.jsonl'):
        add(path, 'jsonl')
    return sources


def verify_dataset(data_root, level, domain, *, search_root=ROOT / 'var/data'):
    """Verify all frozen rows against every source currently discoverable.

    The deterministic report describes a fresh observation. New external
    sources can legitimately change the report and must be audited again at
    confirmation/release; an old report is not a substitute for this call.
    No model outcomes are loaded and no files are written.
    """
    data_root, search_root = Path(data_root).resolve(), Path(search_root).resolve()
    require(level in ('level4', 'level5') and domain in DOMAINS, 'unknown scale dataset')
    require(search_root.is_dir(), 'source search root does not exist')
    base = (data_root / level / 'dataset' / domain).resolve()
    own_pools = (data_root / level / 'pools' / domain).resolve()
    protocol_path = data_root / 'protocol.json'
    recipe_path = data_root / level / 'recipes' / (domain + '.json')
    identity_path = base / 'identity.json'
    pins = {str(p.resolve()): file_sha(p) for p in (protocol_path, recipe_path, identity_path)}
    protocol, identity = read(protocol_path), read(identity_path)
    require(identity.get('schema') in DATASET_SCHEMAS
            and identity.get('status') == 'frozen_pending_heldout_confirmation'
            and identity.get('level') == level and identity.get('domain') == domain
            and identity.get('protocol_sha256') == pins[str(protocol_path.resolve())]
            and identity.get('recipe_sha256') == pins[str(recipe_path.resolve())]
            and identity.get('test_split') == 'eval', 'frozen dataset identity differs from its source')
    expected_histograms = protocol['histograms'][domain]
    split_ids, split_prompts, split_rows = {}, {}, {}
    for split, (count, subset) in SPLITS.items():
        source = base / (split + '.jsonl')
        pins.update(file_pins(source)); pins.update(file_pins(base / split))
        rows = jsonl_rows(source)
        data = load_from_disk(str(base / split))
        require(isinstance(data, DatasetDict) and set(data) == {subset},
                'wrong frozen Arrow subset: ' + str(base / split))
        restored = [dict(row) for row in data[subset]]
        record = identity['splits'][split]
        require(len(rows) == count == record['rows'] and sha(rows) == record['rows_sha256']
                and restored == rows, 'frozen row count/hash/Arrow mismatch: ' + str(source))
        target = Counter({tuple(cell['cell']): cell['rows'] for cell in expected_histograms[split]})
        require(cell_histogram(domain, rows) == target, 'frozen support/family histogram mismatch: ' + str(source))
        ids, prompts = row_sets(domain, rows, source)
        require(len(ids) == len(prompts) == len(rows), 'duplicate identities/prompts within frozen ' + split)
        for earlier in split_ids:
            require(not ids & split_ids[earlier] and not prompts & split_prompts[earlier],
                    'frozen splits overlap: ' + earlier + '/' + split)
        split_ids[split], split_prompts[split], split_rows[split] = ids, prompts, len(rows)
    baseline = protocol.get('files_sha256', {})
    baseline_kind = 'protocol_file_pins'
    if 'exclusions' in identity or 'exclusions_sha256' in identity:
        snapshot_path = Path(identity['exclusions']).resolve()
        require(file_sha(snapshot_path) == identity['exclusions_sha256'], 'frozen exclusion snapshot changed')
        snapshot = read(snapshot_path)
        require(snapshot['protocol_sha256'] == pins[str(protocol_path.resolve())], 'wrong frozen exclusion snapshot')
        pins[str(snapshot_path)] = identity['exclusions_sha256']
        baseline = snapshot['files_sha256']; baseline_kind = 'freeze_exclusion_snapshot'
    baseline_paths = {str(Path(path).resolve()) for path in baseline}
    sources = discover_sources(data_root, level, domain, search_root)
    results, skipped, overlaps = [], [], []
    for source, kind in sorted(sources.items(), key=lambda item: str(item[0])):
        if inside(source, base):
            skipped.append(str(source))
            continue
        before = file_pins(source)
        pins.update(before)
        if kind == 'jsonl':
            rows = jsonl_rows(source)
        else:
            data = load_from_disk(str(source))
            require(isinstance(data, DatasetDict), 'historical source must be a DatasetDict: ' + str(source))
            rows = [dict(row) for subset in data.values() for row in subset]
        ids, prompts = row_sets(domain, rows, source)
        own_pool = inside(source, own_pools)
        compared = ('train', 'eval') if own_pool else tuple(SPLITS)
        found = {split: {'semantic_identities': len(ids & split_ids[split]),
                         'exact_prompts': len(prompts & split_prompts[split])} for split in compared}
        found = {split: counts for split, counts in found.items() if any(counts.values())}
        if found:
            overlaps.append({'source': str(source), 'kind': kind, 'overlaps': found})
        check_pins(before)
        results.append({'path': str(source), 'kind': kind, 'rows': len(rows),
                        'canonical_own_pool': own_pool, 'compared_splits': list(compared),
                        'own_dev_pool_identity_overlaps_allowed': len(ids & split_ids['dev']) if own_pool else 0,
                        'own_dev_pool_prompt_overlaps_allowed': len(prompts & split_prompts['dev']) if own_pool else 0,
                        'new_source_files_since_baseline': len(set(before) - baseline_paths)})
    require(discover_sources(data_root, level, domain, search_root) == sources,
            'source inventory changed during disjointness audit; repeat the audit')
    check_pins(pins)
    require(not overlaps, 'cross-source overlap: ' + json.dumps(overlaps, sort_keys=True))
    return {'schema': SCHEMA, 'status': 'pass', 'level': level, 'domain': domain,
            'data_root': str(data_root), 'canonical_dataset': str(base), 'search_root': str(search_root),
            'dataset_identity_sha256': pins[str(identity_path.resolve())],
            'protocol_sha256': pins[str(protocol_path.resolve())], 'split_rows': split_rows,
            'sources_audited': len(results), 'source_rows_audited_including_duplicate_serializations': sum(r['rows'] for r in results),
            'discovery_baseline': baseline_kind,
            'sources_with_new_files_since_baseline': sum(r['new_source_files_since_baseline'] > 0 for r in results),
            'canonical_own_dataset_sources_skipped': skipped, 'sources': results,
            'overlap_counts': {'semantic_identities': 0, 'exact_prompts': 0},
            'files_sha256': dict(sorted(pins.items()))}
