#!/usr/bin/env python3
"""Repoint resample cells whose weights were retired to the Hub after planning.

The cohort manifests record where each checkpoint's weights live at the moment
the cohort is planned. A model archived *after* that keeps its metadata in the
run directory but loses its local weight files, so a cell planned as ``local``
fails at load time with ``local weight file vanished`` even though the bytes are
safe on the Hub.

The worker already knows how to stream a ``hub`` cell and check every file
against the digest the archive recorded, so the repair is a manifest edit, not a
download: copy the Hub coordinates out of the run's ``MODEL_ARCHIVE.json`` and
the matching archive manifest, and leave everything else about the cell alone.

A ``local`` cell records only a relative path: it never needed a digest, because
the file was on disk. The digest the Hub fetch checks against therefore comes
from the archive manifest, which is only trustworthy if the upload was actually
checked -- so a run is repaired only when its receipt says ``retired`` and its
verification record says ``verified`` with no errors. A prepared-but-unverified
archive is reported and left alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def archive_for(export_dir: Path) -> tuple[dict, dict] | None:
    """The run's archive receipt and the file manifest it points at."""
    receipt_path = export_dir.parents[2] / 'MODEL_ARCHIVE.json'
    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text())
    manifest_path = Path(receipt['manifest_path'])
    if not manifest_path.is_file():
        return None
    return receipt, json.loads(manifest_path.read_text())


def upload_verified(receipt: dict) -> bool:
    """True when the Hub copy was checked file by file after upload."""
    if receipt.get('status') != 'retired':
        return False
    path = Path(receipt.get('verification_path', ''))
    if not path.is_file():
        return False
    record = json.loads(path.read_text())
    remote = record.get('remote_metadata') or []
    entries = remote if isinstance(remote, list) else [remote]
    return bool(entries) and all(
        e.get('status') == 'verified' and not e.get('errors') for e in entries)


def repair_cell(cell: dict) -> tuple[str, str]:
    """Return (status, detail) and rewrite the cell in place when it is fixable."""
    weights = cell['weights']
    if weights['location'] != 'local':
        return 'skipped', 'already hub-backed'
    export = Path(weights['export_dir'])
    if all((export / f['relative_path']).is_file() for f in weights['files']):
        return 'skipped', 'local weights still present'
    found = archive_for(export)
    if found is None:
        return 'unrecoverable', 'no archive receipt or manifest for this run'
    receipt, manifest = found
    if not upload_verified(receipt):
        return 'unverified', f"{export.parents[2].name}: archive not verified; left local"
    by_path = {f['relative_path']: f for f in manifest['files']}
    for item in weights['files']:
        archived = by_path.get(item['relative_path'])
        if archived is None:
            return 'unrecoverable', f"archive lacks {item['relative_path']}"
        planned = item.get('sha256')
        if planned is not None and planned != archived['sha256']:
            return 'conflict', (f"{item['relative_path']}: planned {planned[:12]} "
                                f"!= archived {archived['sha256'][:12]}")
        item['sha256'] = archived['sha256']
        item.setdefault('bytes', archived['size'])
    weights.update({
        'location': 'hub',
        'repo_id': receipt['repo_id'],
        'repo_prefix': receipt['repo_prefix'],
        'commit_sha': receipt['commit_sha'],
        'retired_at_utc': receipt.get('retired_at_utc'),
    })
    return 'repaired', receipt['repo_prefix']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifests', type=Path, nargs='+')
    parser.add_argument('--apply', action='store_true',
                        help='write the repaired manifests; default reports only')
    args = parser.parse_args()
    totals: dict[str, int] = {}
    for path in args.manifests:
        payload = json.loads(path.read_text())
        counts: dict[str, int] = {}
        for cell in payload['cells']:
            status, detail = repair_cell(cell)
            counts[status] = counts.get(status, 0) + 1
            totals[status] = totals.get(status, 0) + 1
            if status in ('conflict', 'unrecoverable', 'unverified'):
                print(f'  !! {path.name}: {detail}')
        if args.apply and counts.get('repaired'):
            path.write_text(json.dumps(payload, indent=1, sort_keys=True) + '\n')
        print(json.dumps({'manifest': path.name, **counts}, sort_keys=True))
    print(json.dumps({'event': 'repair', 'applied': args.apply, **totals}, sort_keys=True))


if __name__ == '__main__':
    main()
