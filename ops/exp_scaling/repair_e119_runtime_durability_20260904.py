#!/usr/bin/env python3
"""Apply an audited E119-only checkpoint/watchdog runtime amendment.

No scheduler action or scientific learner change is performed. Existing jobs
pick up the amended shell scripts when their next allocation starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "var/artifacts/source_snapshots/e76_tuned_scale_50d36295558a8958/ops"
AUDIT = ROOT / "var/artifacts/e119_runtime_recovery_20260904"
INITIAL_RECORD = AUDIT / "durability-watchdog-amendment.json"
PRIOR_RECORD = AUDIT / "durability-watchdog-amendment-r2.json"
RECORD = AUDIT / "durability-watchdog-amendment-r3.json"
BEGIN = "# BEGIN E119_RUNTIME_DURABILITY_20260904"
END = "# END E119_RUNTIME_DURABILITY_20260904"
RUNTIME_BLOCK = r'''# BEGIN E119_RUNTIME_DURABILITY_20260904
# Operational recovery only: retain scientific evaluation and treatment knobs.
if [[ "${RUN_STAMP:-}" == e119_level2_* ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NAME:-}" ]]; then
    export OAT_ZERO_WATCHDOG_LOG_PATH="$ROOT_DIR/var/artifacts/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
  fi
  export OAT_ZERO_WATCHDOG_LOG_PROGRESS=1
  export OAT_ZERO_WATCHDOG_FATAL_PATTERN="${OAT_ZERO_WATCHDOG_FATAL_PATTERN:-Fatal Python error:|ChildProcessError: worker died|Worker VllmWorkerProcess .* died}|RuntimeError: Ninja is required to load C\+\+ extensions|AttributeError:.*_infer_resume_step"
fi
if [[ "${RUN_STAMP:-}" =~ ^e119_level2_pantry_(drgrpo|replay_drgrpo|maxrl|replay_maxrl)_s(43|44|45|46|47)$ ]]; then
  # Reapply the registered 2026-09-03 Level-2 policy-exclusivity repair.
  # Scheduler replacements reconstructed from the original ledger lost it.
  export OAT_ZERO_CANONICAL_ACTION_TASK=none
  export OAT_ZERO_CANONICAL_GRAPH_ACTIONS=0
  export OAT_ZERO_CANONICAL_GRAPH_ACTION_COUNT=3
  export OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING=0
  export OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING=0
  # Save before the first evaluation at 96, keeping one atomic rolling copy.
  export OAT_ZERO_RESUME_STEPS=96
  export OAT_ZERO_RESUME_FROM=96
  export OAT_ZERO_SAVE_STEPS=96
  export OAT_ZERO_SAVE_FROM=96
  export OAT_ZERO_MAX_RESUME_NUM=1
  export OAT_ZERO_WATCHDOG_STALE_SECONDS=7200
  echo "[experiment] e119_pantry_durability=96_steps rolling_keep=1 eval_cadence_unchanged=1"
fi
# END E119_RUNTIME_DURABILITY_20260904
'''
PROGRESS_BLOCK = r'''  # BEGIN E119_RUNTIME_DURABILITY_20260904
  # During evaluation the metrics writer is silent. Real writes to this
  # allocation's stdout are progress; old attempts never reset this timer.
  if [[ "${OAT_ZERO_WATCHDOG_LOG_PROGRESS:-0}" == "1" ]] \
    && [[ -n "$watchdog_log_path" && -f "$watchdog_log_path" ]]; then
    newest_log_mtime="$(stat -c %Y "$watchdog_log_path" 2>/dev/null || printf '0')"
    if (( newest_log_mtime > last_log_mtime )); then
      last_log_mtime="$newest_log_mtime"
      last_progress_at="$now"
    fi
  fi
  # END E119_RUNTIME_DURABILITY_20260904
'''


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected one patch anchor: {old!r}")
    return text.replace(old, new, 1)


def prepare() -> dict[Path, tuple[bytes, bytes]]:
    experiment = OPS / "run_experiment.sh"
    train = OPS / "train.sh"
    originals = {p: p.read_bytes() for p in (experiment, train)}
    if any(BEGIN.encode() in raw for raw in originals.values()):
        expected = {}
        for prior_path in (INITIAL_RECORD, PRIOR_RECORD):
            prior = json.loads(prior_path.read_text())
            expected.update({Path(item["path"]): item["after_sha256"]
                             for item in prior["files"]})
        if any(sha(raw) != expected.get(p) for p, raw in originals.items()):
            raise RuntimeError("r2 runtime hashes drifted before r3 amendment")
        text = originals[experiment].decode()
        if text.count(BEGIN) != 1 or text.count(END) != 1:
            raise RuntimeError("expected one existing E119 runtime block")
        start, finish = text.index(BEGIN), text.index(END) + len(END) + 1
        changed = text[:start] + RUNTIME_BLOCK + text[finish:]
        return {experiment: (originals[experiment], changed.encode())}
    changed_experiment = replace_once(
        originals[experiment].decode(),
        'MODEL="${OAT_ZERO_MODEL:-qwen2.5-0.5b-instruct}"\n',
        RUNTIME_BLOCK + '\nMODEL="${OAT_ZERO_MODEL:-qwen2.5-0.5b-instruct}"\n',
    )
    changed_train = replace_once(
        originals[train].decode(),
        'last_metrics_mtime=0\n',
        'last_metrics_mtime=0\nlast_log_mtime="$started_at"\n',
    )
    changed_train = replace_once(
        changed_train,
        '  if (( now - started_at < watchdog_startup_grace_seconds )); then\n',
        PROGRESS_BLOCK
        + '  if (( now - started_at < watchdog_startup_grace_seconds )); then\n',
    )
    return {
        experiment: (originals[experiment], changed_experiment.encode()),
        train: (originals[train], changed_train.encode()),
    }


def validate(changes: dict[Path, tuple[bytes, bytes]]) -> list[dict[str, object]]:
    for _, after in changes.values():
        subprocess.run(["bash", "-n"], input=after, check=True)
    checks = []
    # Exercise the real guarded block with registered and unrelated run stamps.
    cases = [
        (f"e119_level2_pantry_{arm}_s{seed}", "96|96|1|7200|96|1")
        for arm in ("drgrpo", "replay_drgrpo", "maxrl", "replay_maxrl")
        for seed in range(43, 48)
    ] + [
        ("e119_level2_countdown_drgrpo_s43", "192|192|1|2700|96|1"),
        ("e118q3_pantry_maxrl_s73", "192|192|1|2700|96|0"),
        ("e119_level2_pantry_maxrl_s42", "192|192|1|2700|96|1"),
        ("e119_level2_pantry_maxrl_s48", "192|192|1|2700|96|1"),
    ]
    for stamp, expected in cases:
        script = RUNTIME_BLOCK + r'''
printf '%s|%s|%s|%s|%s|%s\n' "$OAT_ZERO_RESUME_STEPS" "$OAT_ZERO_RESUME_FROM" "$OAT_ZERO_MAX_RESUME_NUM" "$OAT_ZERO_WATCHDOG_STALE_SECONDS" "$OAT_ZERO_EVAL_STEPS" "${OAT_ZERO_WATCHDOG_LOG_PROGRESS:-0}"
'''
        env = {
            "PATH": os.environ["PATH"], "RUN_STAMP": stamp,
            "ROOT_DIR": str(ROOT), "SLURM_JOB_ID": "123", "SLURM_JOB_NAME": "e119-test",
            "OAT_ZERO_RESUME_STEPS": "192", "OAT_ZERO_RESUME_FROM": "192",
            "OAT_ZERO_MAX_RESUME_NUM": "1", "OAT_ZERO_WATCHDOG_STALE_SECONDS": "2700",
            "OAT_ZERO_EVAL_STEPS": "96",
        }
        output = subprocess.check_output(["bash", "-eu", "-c", script], env=env, text=True)
        if output.splitlines()[-1] != expected:
            raise RuntimeError(f"guard validation failed for {stamp}: {output!r}")
        if expected.startswith("96|96|"):
            canonical_script = RUNTIME_BLOCK + '\nprintf "%s|%s|%s|%s|%s\\n" "$OAT_ZERO_CANONICAL_ACTION_TASK" "$OAT_ZERO_CANONICAL_GRAPH_ACTIONS" "$OAT_ZERO_CANONICAL_GRAPH_ACTION_COUNT" "$OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING" "$OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING"\n'
            canonical_env = {**env, "OAT_ZERO_CANONICAL_ACTION_TASK": "pantry_support_mask", "OAT_ZERO_CANONICAL_GRAPH_ACTIONS": "0", "OAT_ZERO_CANONICAL_GRAPH_ACTION_COUNT": "6", "OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING": "1", "OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING": "1"}
            canonical_output = subprocess.check_output(["bash", "-eu", "-c", canonical_script], env=canonical_env, text=True).splitlines()[-1]
            if canonical_output != "none|0|3|0|0":
                raise RuntimeError(f"canonical policy repair failed: {canonical_output}")
        checks.append({"run_stamp": stamp, "observed": output.splitlines()[-1]})
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if RECORD.exists():
        prior = json.loads(RECORD.read_text())
        verified_files = prior["files"] + prior.get("unchanged_runtime_files", [])
        if all(sha(Path(r["path"]).read_bytes()) == r["after_sha256"] for r in verified_files):
            print(json.dumps({"already_applied": str(RECORD)}))
            return 0
        raise RuntimeError("record exists but runtime hashes drifted")
    changes = prepare()
    checks = validate(changes)
    record = {
        "schema": "e119-runtime-durability-watchdog-v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scheduler_mutated": False, "scientific_treatment_changed": False,
        "evaluation_configuration_changed": False, "source_learner_changed": False,
        "pantry_checkpoint_interval_before": 32, "pantry_checkpoint_interval_after": 96,
        "pantry_rolling_keep": 1, "pantry_stale_seconds": 7200,
        "scope": "All 20 Pantry cells use checkpoint96 before evaluation96 on future allocations; preserve r2 policy/watchdog repairs",
        "future_allocations_only": True,
        "active_job_ids_requeued": [],
        "inflight_allocation_exception": {
            "job_id": 31041178,
            "run_stamp": "e119_level2_pantry_replay_drgrpo_s43",
            "allocation_start": "2026-09-04T23:45:11-04:00",
            "restart_count": 8,
            "retained_resume_steps": 32,
            "retained_resume_from": 32,
            "reason": "Current per-allocation shell copy and running learner arguments remain unchanged; let step32 checkpoint finish",
        },
        "helper_source": str(Path(__file__).resolve()),
        "helper_sha256": sha(Path(__file__).read_bytes()),
        "unchanged_runtime_files": [{"path": str(OPS / "train.sh"),
                                     "after_sha256": sha((OPS / "train.sh").read_bytes())}],
        "prior_amendment": str(PRIOR_RECORD) if PRIOR_RECORD.exists() else None,
        "pantry_policy_repair_reapplied": True,
        "pantry_policy_repair_source": str(ROOT / "ops/exp_scaling/repair_e119_failed_and_pantry_cells.py"),
        "checks": checks,
        "files": [{"path": str(p), "before_sha256": sha(before), "after_sha256": sha(after)}
                  for p, (before, after) in changes.items()],
        "applied": args.apply,
    }
    if args.apply:
        AUDIT.mkdir(parents=True, exist_ok=True)
        for p, (before, after) in changes.items():
            if p.read_bytes() != before:
                raise RuntimeError(f"concurrent runtime change: {p}")
            backup = AUDIT / f"{p.name}.before-durability-{sha(before)[:12]}"
            if backup.exists() and backup.read_bytes() != before:
                raise RuntimeError(f"backup collision: {backup}")
            backup.write_bytes(before)
            for item in record["files"]:
                if item["path"] == str(p): item["backup"] = str(backup)
            with tempfile.NamedTemporaryFile(dir=p.parent, delete=False) as handle:
                handle.write(after)
                temporary = Path(handle.name)
            temporary.chmod(p.stat().st_mode & 0o777)
            temporary.replace(p)
        RECORD.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
