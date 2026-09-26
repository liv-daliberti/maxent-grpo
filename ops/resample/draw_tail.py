"""Read the terminal mode-coverage draw records without loading a whole run log.

Each ``eval_mode_coverage_draws.jsonl`` is tens of megabytes and only its last
few records describe the terminal step, so the cohort manifest reads a bounded
tail rather than the file. Shared storage makes the difference matter: 298 cells
at ~27 MB each is ~8 GB of reads that nothing needs.
"""
from __future__ import annotations

import json
from pathlib import Path

TAIL_BYTES = 8 << 20


def tail_records(path: Path, tail_bytes: int = TAIL_BYTES) -> list[dict]:
    """Return the complete JSON records contained in the file's last bytes."""
    size = path.stat().st_size
    with path.open('rb') as handle:
        start = max(0, size - tail_bytes)
        handle.seek(start)
        blob = handle.read()
    if start:
        # The first line is almost certainly truncated; drop it.
        blob = blob.split(b'\n', 1)[1] if b'\n' in blob else b''
    records = []
    for line in blob.split(b'\n'):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def terminal_draws(path: Path, step: int, kind: str = 'fixed_seed_sampled_k_neutral',
                   tail_bytes: int = TAIL_BYTES) -> list[dict]:
    """Return the sampled-K draws recorded at ``step``, ordered by draw index."""
    hits = [r for r in tail_records(path, tail_bytes)
            if r.get('evaluation_kind') == kind and int(r.get('step', -1)) == step]
    # The greedy trace records a null draw index; sampled draws are 0..n-1.
    hits.sort(key=lambda r: -1 if r.get('draw_index') is None else int(r['draw_index']))
    return hits
