#!/usr/bin/env python3
"""Re-pin a figure-source manifest to the receipts as they stand now.

The manifest binds each receipt by SHA so the payload builder refuses to read a
receipt that changed underneath it. That guard is what makes re-grading visible
rather than silent, so after a deliberate re-grade the manifest has to be
re-pinned explicitly - and only the receipts whose content actually moved.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
from followup_metrics import file_sha  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    changed = []
    for entry in manifest['receipts']:
        path = ROOT / entry['path']
        if not path.is_file():
            raise FileNotFoundError(f'manifest names a missing receipt: {entry["path"]}')
        current = file_sha(path)
        if current != entry['sha256']:
            changed.append(entry['path'])
            entry['sha256'] = current
    manifest['repinned_from'] = str(args.manifest.resolve().relative_to(ROOT))
    manifest['repinned_receipts'] = changed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'event': 'repinned', 'receipts': len(manifest['receipts']),
                      'changed': len(changed), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
