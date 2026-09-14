"""Terminal endpoint and preregistered E120 analysis guards for paper artifacts."""
from __future__ import annotations
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any
import build_paper_core_terminal_endpoints as core
from audit_e120_frequency_smoke import REQUIRED


def terminal_endpoint(run_dir: str | Path, *, audit: list[dict], require_receipt: bool) -> dict[str, float] | None:
    root = Path(run_dir)
    receipt = root / 'TRAINING_COMPLETE.json'
    if require_receipt:
        if not receipt.is_file():
            audit.append({'run_dir': str(root), 'status': 'incomplete', 'reason': 'no training-completion receipt'})
            return None
        payload = json.loads(receipt.read_text())
        if payload.get('schema') != 'oat_zero_training_complete_v1' or payload.get('terminal_step') not in (3072, 3073):
            raise RuntimeError(f'invalid terminal receipt: {receipt}')
        for key in ('terminal_attempt', 'terminal_export'):
            path = Path(payload[key]).resolve()
            path.relative_to(root.resolve())
            if not path.is_dir():
                raise RuntimeError(f'missing {key}: {path}')
    result = core.sampled_endpoint(root, step=3072, audit=audit)
    if require_receipt and result is None:
        raise RuntimeError(f'completed run lacks an admissible terminal four-draw endpoint: {root}')
    if require_receipt:
        audit[-1]['completion_receipt'] = {'path': str(receipt), 'sha256': hashlib.sha256(receipt.read_bytes()).hexdigest()}
    if result is not None:
        result['breadth8'] = result['distinct8'] - result['pass8']
    return result


def bootstrap_summary(per_seed: dict[str, float], *, complete: bool) -> dict[str, Any]:
    values = list(per_seed.values())
    result: dict[str, Any] = {'per_seed': per_seed, 'mean': statistics.fmean(values)}
    if complete:
        if len(values) != 5:
            raise ValueError('intervals require the complete five-seed block')
        draws = sorted(statistics.fmean(draw) for draw in itertools.product(values, repeat=5))
        def quantile(q):
            pos = q * (len(draws) - 1)
            lo = math.floor(pos); hi = math.ceil(pos)
            return draws[lo] + (pos-lo) * (draws[hi]-draws[lo])
        result['paired_bootstrap_percentile_95'] = [quantile(.025), quantile(.975)]
        result['bootstrap_ordered_resamples'] = len(draws)
    return result


def audit_frequency_telemetry(run_dir: str | Path) -> dict[str, Any]:
    """Audit persisted applied replay rows; never write back into a run."""
    paths = sorted(Path(run_dir).glob('debug_job*/train_metrics.jsonl'))
    sources = []; count = 0; empty = 0; violations = []
    required = {**REQUIRED, 'canonical_replay_reward_estimator_scale': 15.0/16.0,
                'canonical_replay_capacity': 16.0, 'canonical_replay_actuator_groups': 1.0,
                'canonical_replay_global_scheduler_active': 1.0,
                'canonical_replay_verified_likelihood_active': 1.0,
                'canonical_replay_compute_only': 0.0}
    def fail(path, line, why):
        if len(violations) < 20:
            violations.append({'path': str(path), 'line': line, 'reason': why})
    for path in paths:
        digest = hashlib.sha256(); applied = 0
        with path.open('rb') as f:
            for line_number, raw in enumerate(f, 1):
                digest.update(raw)
                try: row = json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    fail(path, line_number, 'invalid JSON in completed run metrics'); continue
                record = {k.removeprefix('train/'): v for k,v in row.items() if k.startswith('train/canonical_replay_')}
                if not record:
                    continue
                modes = record.get('canonical_replay_actuator_modes', 0)
                if modes == 0:
                    empty += 1; continue
                count += 1; applied += 1
                try:
                    for key, expected in required.items():
                        value = record.get(key)
                        assert isinstance(value, (int,float)) and math.isfinite(value) and math.isclose(value, expected, rel_tol=0, abs_tol=1e-7), f'{key}={value!r}'
                    assert int(modes) == modes and 1 <= modes <= 16, 'invalid materialized bank size'
                    def vector(prefix):
                        return {key[len(prefix):]: value for key,value in record.items() if key.startswith(prefix)}
                    weights = vector('canonical_replay_target_weight_row_')
                    counts = vector('canonical_replay_fresh_count_row_')
                    outcomes = vector('canonical_replay_outcome_fingerprint_row_')
                    assert set(weights) == set(counts) == set(outcomes) == {f'{i:02d}' for i in range(int(modes))}, 'unaligned key/count/weight vector'
                    assert all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in weights.values()), 'nonpositive/nonfinite target weight'
                    assert all(isinstance(v,(int,float)) and math.isfinite(v) and v>=1 for v in counts.values()), 'nonpositive/nonfinite fresh count'
                    total = sum(counts.values())
                    assert math.isclose(sum(weights.values()), modes, rel_tol=0, abs_tol=1e-5), 'target budget differs from bank size'
                    assert math.isclose(record['canonical_replay_target_weight_sum'], modes, rel_tol=0, abs_tol=1e-5), 'reported target budget differs from bank size'
                    for key,value in weights.items():
                        assert math.isclose(value, modes*counts[key]/total, rel_tol=2e-6, abs_tol=2e-6), 'target not proportional to aligned fresh count'
                    for key in ('canonical_replay_fresh_advantage_fingerprint','canonical_replay_prompt_fingerprint_group_00','canonical_replay_membership_fingerprint_group_00'):
                        assert isinstance(record.get(key),(int,float)) and math.isfinite(record[key]), f'missing/nonfinite {key}'
                    assert all(isinstance(v,(int,float)) and math.isfinite(v) for v in outcomes.values()), 'invalid outcome fingerprint'
                except (AssertionError, KeyError, TypeError, ValueError) as error:
                    fail(path, line_number, str(error))
        sources.append({'path': str(path), 'sha256': digest.hexdigest(), 'applied_replay_rows': applied})
    if count == 0:
        violations.append({'reason': 'no applied frequency-replay telemetry'})
    return {'status': 'pass' if not violations else 'provisional', 'applied_replay_rows': count,
            'empty_replay_rows': empty, 'sources': sources, 'violations': violations,
            'required_constants': required,
            'scope': 'All persisted applied replay rows, including recovered attempts. Alignment and numerical invariants are checked; source-level fresh-only provenance also relies on the prerelease implementation gate.'}
