#!/usr/bin/env python3
"""Resume the paused model archive from any host, using a durable credential.

The 2026-09-11 archive stopped at 533 of 715 uploads. Two things kept it
stopped, neither of which was the upload itself: its runtime and credential
lived in a node-local ``/tmp`` that disappeared when work moved hosts, and its
preflight compared the repository root's NFS device number against a constant
that is assigned per host and per mount rather than per machine.

This drives the unchanged archive engine directly. Everything that makes the
archive safe is the engine's, not this script's: each model is uploaded, then
independently re-downloaded and hashed, and only then are its local weight files
removed, with a restore receipt written beside the run. Commit pacing is the
same module the earlier supervisor used, so the repository's 256-commits-per-
hour limit is respected and a rate-limit rejection produces a durable cooldown
rather than a retry storm.

The credential is read from a file outside the repository, defaulting to
``~/.maxent_hf_token``; it is never logged, and the file must be mode 0600.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))

# The Hub client's Xet backend keeps a chunk store of many small files. Left at
# its default it lands in the user's home, which on this cluster is a small NFS
# share -- it filled, and the uploader failed mid-transfer with "MerkleDB Shard
# error: File I/O error". Small-file churn does not belong on shared storage
# anyway, so both caches default to node-local scratch.
_SCRATCH = Path(os.environ.get('MAXENT_HF_SCRATCH') or f'/tmp/maxent_hf_{os.getuid()}')
os.environ.setdefault('HF_HOME', str(_SCRATCH / 'hf'))
os.environ.setdefault('HF_XET_CACHE', str(_SCRATCH / 'xet'))
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
for _d in (_SCRATCH / 'hf', _SCRATCH / 'xet'):
    _d.mkdir(parents=True, exist_ok=True)

DEFAULT_PLAN = ROOT / 'var/artifacts/hf_model_archive_20260911/all/plan_completed_delta715_amended_20260915.json'
DEFAULT_TOKEN = Path(os.environ.get('MAXENT_HF_TOKEN_FILE') or (Path.home() / '.maxent_hf_token'))
PACING_DIR = ROOT / 'var/artifacts/hf_model_archive_20260911/paced_supervisor/commit_pacing'
ADAPTER = ROOT / 'var/artifacts/hf_archive_resume_20260915_lineage/ledger_admission.py'


def require(ok: bool, message: str) -> None:
    if not ok:
        raise SystemExit(message)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_credential(token_file: Path) -> str:
    require(token_file.is_file(), f'credential file missing: {token_file}')
    mode = token_file.stat().st_mode
    require(stat.S_IMODE(mode) == 0o600,
            f'credential must be mode 0600, found {oct(stat.S_IMODE(mode))}')
    require(not token_file.resolve().is_relative_to(ROOT),
            'credential must not live inside the repository')
    secret = token_file.read_text().strip()
    require(secret.startswith('hf_') and 20 < len(secret) < 4096, 'credential does not look like a Hub token')
    return secret


def preflight(plan_path: Path, token_file: Path) -> dict:
    """Verify everything the run depends on, without uploading anything."""
    secret = check_credential(token_file)
    admission = load_module(ADAPTER, 'ledger_admission').LedgerAdmission().validate_live()
    require(admission['status'] == 'exact_reviewed_disjoint_successor',
            f'ledger admission refused: {admission["status"]}')
    engine = load_module(ROOT / 'ops/archive_completed_models.py', 'archive_engine')
    plan = json.loads(plan_path.read_text())
    import hashlib
    require(hashlib.sha256(plan_path.read_bytes()).hexdigest()
            == plan_path.with_suffix('.sha256').read_text().strip(), 'plan digest mismatch')
    engine.check_pins(plan)
    from huggingface_hub import HfApi
    info = HfApi(token=secret).model_info(plan['repo_id'])
    require(info.private == plan['private'], 'repository visibility differs from the plan')
    output = Path(plan['output_dir'])
    done = sum(1 for m in plan['models'] if (output / 'models' / m['archive_id'] / 'retired.json').exists())
    return {'status': 'ready', 'repo_id': plan['repo_id'], 'public': not plan['private'],
            'models_planned': len(plan['models']), 'already_retired': done,
            'remaining': len(plan['models']) - done,
            'ledger_admission': admission['status'],
            'selected_ledger_rows_unchanged': admission['selected_ledger_rows_unchanged'],
            'credential': str(token_file), 'pacing_state': str(PACING_DIR)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('preflight', 'run'))
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--token-file', type=Path, default=DEFAULT_TOKEN)
    parser.add_argument('--limit', type=int, default=0, help='stop after this many models (0 = all)')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--keep-local', action='store_true',
                        help='upload and verify but do not remove local weights')
    args = parser.parse_args()

    report = preflight(args.plan, args.token_file)
    if args.phase == 'preflight':
        print(json.dumps(report, indent=2))
        return

    secret = check_credential(args.token_file)
    pacing = load_module(ROOT / 'ops/model_archive_commit_pacing.py', 'commit_pacing')
    pacer = pacing.CommitPacer(PACING_DIR)
    pacer.snapshot()
    pacing.install_pacing(pacer, secret)
    engine = load_module(ROOT / 'ops/archive_completed_models.py', 'archive_engine')
    print(json.dumps({**report, 'phase': 'run', 'limit': args.limit,
                      'workers': args.workers, 'keep_local': args.keep_local}, indent=2), flush=True)
    engine.run(str(args.plan), str(args.token_file), args.limit, args.keep_local, args.workers)


if __name__ == '__main__':
    main()
