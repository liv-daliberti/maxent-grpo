#!/usr/bin/env python3
"""Amend future E119 allocations' runtime watchdog; no scheduler actions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "var/artifacts/source_snapshots/e76_tuned_scale_50d36295558a8958/ops"
AUDIT = ROOT / "var/artifacts/e119_health_recovery_20260905"
RECORD = AUDIT / "watchdog-progress-amendment.json"
GUARD = r'''  # BEGIN E119_WATCHDOG_PROGRESS_20260905
  # Preserve longer explicitly configured allowances; these are minimums.
  for e119_watchdog_knob in OAT_ZERO_WATCHDOG_STALE_SECONDS OAT_ZERO_WATCHDOG_MAX_RESTARTS; do
    e119_watchdog_floor=7200
    [[ "$e119_watchdog_knob" == OAT_ZERO_WATCHDOG_MAX_RESTARTS ]] && e119_watchdog_floor=12
    e119_watchdog_value="${!e119_watchdog_knob:-0}"
    if [[ ! "$e119_watchdog_value" =~ ^[0-9]+$ ]]; then
      echo "Invalid E119 watchdog value: $e119_watchdog_knob" >&2
      exit 2
    fi
    e119_watchdog_value="$((10#$e119_watchdog_value))"
    (( e119_watchdog_value < e119_watchdog_floor )) && e119_watchdog_value="$e119_watchdog_floor"
    export "$e119_watchdog_knob=$e119_watchdog_value"
  done
  unset e119_watchdog_knob e119_watchdog_floor e119_watchdog_value
  export OAT_ZERO_WATCHDOG_ARTIFACT_PROGRESS=1
  # END E119_WATCHDOG_PROGRESS_20260905
'''
PROGRESS = r'''  # BEGIN E119_WATCHDOG_ARTIFACT_PROGRESS_20260905
  # Only this allocation's files can extend its inactivity allowance.
  # Prior checkpoint/evaluation timestamps are ignored by the startup baseline.
  if [[ "${RUN_STAMP:-}" == e119_level2_* && "${OAT_ZERO_WATCHDOG_ARTIFACT_PROGRESS:-0}" == 1 && -n "${SLURM_JOB_ID:-}" ]]; then
    newest_artifact_mtime="$(
      {
        for watchdog_attempt_path in "$SAVE_PATH"/*_"${OAT_ZERO_FIXED_EXP_SUFFIX:-job${SLURM_JOB_ID}}"; do
          [[ -d "$watchdog_attempt_path" ]] || continue
          find "$watchdog_attempt_path" -maxdepth 3 -type f \
            \( -name train_metrics.jsonl -o -name eval_mode_coverage_draws.jsonl \
            -o -path '*/eval_results/*.json' -o -path '*/checkpoints/*/*.pt' \
            -o -name TRAINING_COMPLETE.json \) -printf '%T@\n' 2>/dev/null || true
        done
      } | sort -nr | head -n 1 | cut -d. -f1
    )"
    if [[ -n "$newest_artifact_mtime" ]] && (( newest_artifact_mtime > last_artifact_mtime )); then
      last_artifact_mtime="$newest_artifact_mtime"
      last_progress_at="$now"
    fi
  fi
  # END E119_WATCHDOG_ARTIFACT_PROGRESS_20260905
'''


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def replace_once(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise RuntimeError(f"Runtime anchor drift: {before!r}")
    return text.replace(before, after, 1)


def prepare() -> dict[Path, tuple[bytes, bytes]]:
    runner, train = OPS / "run_experiment.sh", OPS / "train.sh"
    originals = {p: p.read_bytes() for p in (runner, train)}
    if any(b"E119_WATCHDOG_PROGRESS_20260905" in v for v in originals.values()):
        raise RuntimeError("Watchdog amendment already present without matching receipt")
    text = replace_once(originals[runner].decode(),
                        '  export OAT_ZERO_WATCHDOG_LOG_PROGRESS=1\n',
                        '  export OAT_ZERO_WATCHDOG_LOG_PROGRESS=1\n' + GUARD)
    text = replace_once(text, '  export OAT_ZERO_WATCHDOG_STALE_SECONDS=7200\n', '')
    after_train = replace_once(originals[train].decode(),
                              'last_log_mtime="$started_at"\n',
                              'last_log_mtime="$started_at"\nlast_artifact_mtime="$started_at"\n')
    start = after_train.index('  newest_metrics_mtime="$(\n')
    end = after_train.index('  # BEGIN E119_RUNTIME_DURABILITY_20260904\n', start)
    legacy = after_train[start:end]
    scoped = ('  # BEGIN E119_LEGACY_METRICS_SCOPE_20260905\n'
              '  if [[ "${RUN_STAMP:-}" != e119_level2_* || "${OAT_ZERO_WATCHDOG_ARTIFACT_PROGRESS:-0}" != 1 ]]; then\n'
              + textwrap.indent(legacy, '  ')
              + '  fi\n  # END E119_LEGACY_METRICS_SCOPE_20260905\n')
    after_train = after_train[:start] + scoped + after_train[end:]
    after_train = replace_once(after_train,
                              '  if (( now - started_at < watchdog_startup_grace_seconds )); then\n',
                              PROGRESS + '  if (( now - started_at < watchdog_startup_grace_seconds )); then\n')
    return {runner: (originals[runner], text.encode()), train: (originals[train], after_train.encode())}


def validate(changes: dict[Path, tuple[bytes, bytes]]) -> list[str]:
    checks = []
    for _, after in changes.values():
        subprocess.run(['bash', '-n'], input=after, check=True)
    runner = changes[OPS / 'run_experiment.sh'][1].decode()
    block = runner.split('# BEGIN E119_RUNTIME_DURABILITY_20260904', 1)[1].split('# END E119_RUNTIME_DURABILITY_20260904', 1)[0]
    for stamp, window, budget, expected in (
        ('e119_level2_mathir_replay_maxrl_s45', '2700', '6', '7200|12|1'),
        ('e119_level2_pantry_maxrl_s43', '2700', '6', '7200|12|1'),
        ('e119_level2_pantry_maxrl_s43', '10800', '20', '10800|20|1'),
        ('e119_level2_pantry_maxrl_s43', '010800', '020', '10800|20|1'),
        ('e118q3_python_replay_maxrl_s72', '2700', '6', '2700|6|0'),
    ):
        env = {'PATH': os.environ['PATH'], 'ROOT_DIR': str(ROOT), 'RUN_STAMP': stamp,
               'SLURM_JOB_ID': '123', 'SLURM_JOB_NAME': 'audit',
               'OAT_ZERO_WATCHDOG_STALE_SECONDS': window, 'OAT_ZERO_WATCHDOG_MAX_RESTARTS': budget}
        out = subprocess.check_output(['bash', '-eu', '-c', block + '\nprintf "%s|%s|%s\\n" "$OAT_ZERO_WATCHDOG_STALE_SECONDS" "$OAT_ZERO_WATCHDOG_MAX_RESTARTS" "${OAT_ZERO_WATCHDOG_ARTIFACT_PROGRESS:-0}"'], env=env, text=True)
        if out.splitlines()[-1] != expected:
            raise RuntimeError(f"E119 watchdog guard mismatch for {stamp}: {out!r}")
    checks.append('E119 minimums, longer custom allowances, and unrelated campaign isolation passed')
    with tempfile.TemporaryDirectory(prefix='e119-watchdog-') as temporary:
        root = Path(temporary)
        current, old = root / 'debug_job123', root / 'debug_job122'
        current.mkdir(); old.mkdir()
        old_metric = old / 'train_metrics.jsonl'; old_metric.write_text('{}\n'); os.utime(old_metric, (9000, 9000))
        sidecar = current / 'eval_mode_coverage_draws.jsonl'; sidecar.write_text('{}\n'); os.utime(sidecar, (800, 800))
        env = {'PATH': os.environ['PATH'], 'RUN_STAMP': 'e119_level2_pantry_maxrl_s43',
               'OAT_ZERO_WATCHDOG_ARTIFACT_PROGRESS': '1', 'SLURM_JOB_ID': '123', 'SAVE_PATH': str(root)}
        def observe(expected: str) -> None:
            out = subprocess.check_output(['bash', '-euo', 'pipefail', '-c',
                'last_artifact_mtime=1000\nlast_progress_at=1000\nnow=1010\n' + PROGRESS + '\nprintf "%s" "$last_progress_at"'], env=env, text=True)
            if out != expected:
                raise RuntimeError(f"Artifact progress mismatch: {out!r} != {expected!r}")
        observe('1000')
        os.utime(sidecar, (1001, 1001)); observe('1010')
        os.utime(sidecar, (800, 800))
        checkpoint = current / 'checkpoints/step_00096/optim_states.pt'; checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b'partial'); os.utime(checkpoint, (1002, 1002)); observe('1010')
        env['RUN_STAMP'] = 'e118q3_pantry_maxrl_s73'; observe('1000')
        # Exercise the actual composed loop section, including the legacy
        # scanner, to prove other-attempt metrics cannot reset E119's timer.
        after_train = changes[OPS / 'train.sh'][1].decode()
        actual_loop = after_train.split('  # BEGIN E119_LEGACY_METRICS_SCOPE_20260905\n', 1)[1].split('  if (( now - started_at < watchdog_startup_grace_seconds )); then\n', 1)[0]
        os.utime(checkpoint, (800, 800))
        for stamp, expected in [('e119_level2_pantry_maxrl_s43', '1000'), ('e118q3_pantry_maxrl_s73', '1010')]:
            env['RUN_STAMP'] = stamp
            out = subprocess.check_output(['bash', '-euo', 'pipefail', '-c',
                'last_artifact_mtime=1000\nlast_metrics_mtime=0\nlast_log_mtime=1000\nwatchdog_log_path=""\nlast_progress_at=1000\nnow=1010\n' + actual_loop + '\nprintf "%s" "$last_progress_at"'], env=env, text=True)
            if out != expected:
                raise RuntimeError(f'Composed loop attempt-isolation mismatch: {stamp}: {out!r}')
    checks.append('Growing current evaluation/checkpoint files reset timer; old attempts, stale files, and unrelated campaigns do not')
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if RECORD.exists():
        prior = json.loads(RECORD.read_text())
        if not all(sha(Path(x['path']).read_bytes()) == x['after_sha256'] for x in prior['files']):
            raise RuntimeError('Applied watchdog source has drifted')
        print(json.dumps({'already_applied': str(RECORD)})); return
    changes = prepare(); checks = validate(changes)
    record = {'schema': 'e119-watchdog-progress-amendment-v1',
              'created_at': datetime.now(timezone.utc).isoformat(), 'applied': args.apply,
              'scientific_settings_changed': False, 'future_allocations_only': True,
              'scheduler_actions': [], 'checks': checks,
              'files': [{'path': str(p), 'before_sha256': sha(a), 'after_sha256': sha(b)} for p, (a, b) in changes.items()]}
    AUDIT.mkdir(parents=True, exist_ok=True)
    if args.apply:
        for path, (before, after) in changes.items():
            if path.read_bytes() != before:
                raise RuntimeError(f'Concurrent source change: {path}')
            backup = AUDIT / f'{path.name}.before-progress-{sha(before)[:12]}'
            backup.write_bytes(before)
            for item in record['files']:
                if item['path'] == str(path): item['backup'] = str(backup)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                handle.write(after); temporary = Path(handle.name)
            temporary.chmod(path.stat().st_mode & 0o777); temporary.replace(path)
        RECORD.write_text(json.dumps(record, indent=2) + '\n')
    else:
        (AUDIT / 'watchdog-progress-plan.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'applied': args.apply, 'checks': checks, 'record': str(RECORD)}))


if __name__ == '__main__':
    main()
