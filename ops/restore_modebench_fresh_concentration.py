#!/usr/bin/env python3
"""Restore only this campaign's pinned checkpoints into a separate verified cache."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'artifacts/modebench_fresh_concentration_20260912'
CACHE = ROOT / 'var/cache/modebench_fresh_concentration_20260912'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_new(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.writing-')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        try:
            os.link(name, path)
        except FileExistsError:
            if path.read_bytes() != raw:
                raise ValueError(f'existing receipt differs: {path}')
    finally:
        Path(name).unlink(missing_ok=True)


def task_id(record):
    return record['cell_id'].replace('/', '__')


def target_path(record, cache=CACHE):
    return Path(cache) / 'models' / task_id(record)


def verify_binding(binding):
    p = Path(binding['path'])
    if not p.is_file() or digest(p) != binding['sha256']:
        raise ValueError(f'source binding differs: {p}')
    return p


def valid_relative(name):
    p = Path(name)
    if not name or p.is_absolute() or '..' in p.parts:
        raise ValueError('invalid model-relative path')
    return p


def restore_record(record, *, cache=CACHE, downloader=None):
    """Use immutable archive identities, leaving the original run untouched."""
    cache = Path(cache); target = target_path(record, cache)
    target.mkdir(parents=True, exist_ok=True)
    receipt_path = cache / 'receipts' / (task_id(record) + '.json')
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with (target / '.restoration.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        verify_binding(record['completion_receipt'])
        restoration = record.get('restore')
        if restoration:
            receipt_file = verify_binding(restoration['receipt'])
            receipt = json.loads(receipt_file.read_text())
            manifest_file = verify_binding(record['archive_manifest'])
            manifest = json.loads(manifest_file.read_text())
            if (receipt['commit_sha'] != restoration['revision']
                    or receipt['repo_id'] != restoration['repo_id']
                    or receipt['manifest_sha256'] != record['archive_manifest']['sha256']):
                raise ValueError('archive receipt and inventory disagree')
            sources = {x['relative_path']: x for x in manifest['files']}
        else:
            sources = {}
        entries = record['files']
        needed = sum(e['bytes'] for e in entries if not (target / e['name']).exists())
        if shutil.disk_usage(target).free < needed + 10 * 1024**3:
            raise ValueError('insufficient free disk space for this checkpoint')
        verified = []
        for entry in entries:
            name = str(valid_relative(entry['name']))
            expected = entry.get('sha256')
            if name in sources:
                archive = sources[name]
                if archive['size'] != entry['bytes'] or archive['sha256'] != expected:
                    raise ValueError('archive file differs from inventory')
            destination = target / name
            local = Path(record['model_path']) / name
            # A previous cache-link attempt may have linked the Hub's relative
            # symlink instead of its blob. Repair only dangling links in our
            # dedicated campaign destination; never remove a regular file.
            if destination.is_symlink() and not destination.exists():
                destination.unlink()
            if destination.exists():
                actual = digest(destination)
                if destination.stat().st_size != entry['bytes'] or (expected and actual != expected):
                    raise ValueError(f'restored target differs: {destination}')
            else:
                if local.is_file():
                    source = local
                elif restoration and name in sources:
                    if downloader is None:
                        from huggingface_hub import hf_hub_download
                        download = hf_hub_download
                    else:
                        download = downloader
                    source = None
                    for attempt in range(3):
                        try:
                            source = Path(download(repo_id=restoration['repo_id'],
                                filename=sources[name]['path_in_repo'], revision=restoration['revision'],
                                token=False, cache_dir=str(cache / 'hub'))).resolve()
                            break
                        except Exception:
                            if attempt == 2:
                                raise
                            time.sleep(2 ** (attempt + 1))
                    assert source is not None
                else:
                    raise ValueError(f'no authenticated restoration source: {name}')
                actual = digest(source)
                if source.stat().st_size != entry['bytes'] or (expected and actual != expected):
                    raise ValueError(f'restoration source differs: {source}')
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Hard-link only immutable archive cache blobs; copy surviving
                # historical files to isolate our cache from old-run maintenance.
                is_download = str(source).startswith(str(cache / 'hub') + os.sep)
                if is_download:
                    os.link(source, destination)
                else:
                    fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix='.restoring-')
                    try:
                        with source.open('rb') as inp, os.fdopen(fd, 'wb') as out:
                            shutil.copyfileobj(inp, out, 8 * 1024**2)
                            out.flush(); os.fsync(out.fileno())
                        if digest(temporary) != actual:
                            raise ValueError('restored copy differs')
                        os.link(temporary, destination)
                    finally:
                        Path(temporary).unlink(missing_ok=True)
            verified.append({'name': name, 'bytes': destination.stat().st_size, 'sha256': actual})
        result = {'schema': 'modebench-fresh-concentration-restoration-v1',
                  'cell_id': record['cell_id'], 'model_path': str(target), 'files': verified,
                  'completion_receipt': record['completion_receipt'],
                  'archive_manifest': record.get('archive_manifest'),
                  'archive_revision': restoration['revision'] if restoration else None,
                  'original_model_path': record['model_path'],
                  'original_run_modified': False}
        atomic_new(receipt_path, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, default=DEFAULT / 'checkpoint_inventory.json')
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--cell-id', action='append')
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('--workers must be 1..8')
    inventory = json.loads(args.inventory.read_text())
    records = inventory['records']
    if args.cell_id:
        records = [r for r in records if r['cell_id'] in args.cell_id]
        if {r['cell_id'] for r in records} != set(args.cell_id):
            parser.error('unknown requested cell')
    # Start with small-model Dr/replay pairs and both task interfaces.
    order = {'qwen05b': 0, 'falcon1b': 1, 'qwen3b': 2}
    methods = {'drgrpo': 0, 'replay_drgrpo': 1, 'maxrl': 2, 'replay_maxrl': 3}
    records.sort(key=lambda r: (order[r['model_scale']], r['training_seed'], methods[r['method']], r['domain']))
    if args.limit is not None:
        records = records[:args.limit]
    log = DEFAULT / 'restoration_events.jsonl'; log.parent.mkdir(exist_ok=True)
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(restore_record, r, cache=args.cache): r for r in records}
        for future in as_completed(futures):
            record = futures[future]
            event = {'at_utc': datetime.now(timezone.utc).isoformat(), 'cell_id': record['cell_id']}
            try:
                result = future.result()
                event.update(status='verified', model_path=result['model_path'], bytes=sum(e['bytes'] for e in result['files']))
            except Exception as exc:
                event.update(status='error', error=str(exc)); errors.append(event)
            line = json.dumps(event, sort_keys=True)
            with log.open('a') as f:
                f.write(line + '\n'); f.flush()
            print(line, flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
