#!/usr/bin/env python3
"""Repair E120 checkpoint resume and fatal-log detection; no scheduler actions."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import tempfile
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from e118_resume_runtime_patch import METHOD_SOURCE, patched_source

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "var/artifacts/source_snapshots/e76_tuned_scale_243893d9ae89f578"
AUDIT = ROOT / "var/artifacts/e120_runtime_recovery_20260905"
RECORD = AUDIT / "runtime-amendment.json"
BEGIN = "# BEGIN E120_RUNTIME_RECOVERY_20260905"
BLOCK = r'''# BEGIN E120_RUNTIME_RECOVERY_20260905
# Runtime recovery only. E120's actual sbatch stdout is the root slurm log.
if [[ "${RUN_STAMP:-}" == e120r1_* && "${E120_SMOKE:-0}" == "0" && -n "${SLURM_JOB_ID:-}" ]]; then
  export OAT_ZERO_WATCHDOG_LOG_PATH="$ROOT_DIR/slurm-${SLURM_JOB_ID}.out"
  export OAT_ZERO_WATCHDOG_FATAL_PATTERN='Fatal Python error:|ChildProcessError: worker died|Worker VllmWorkerProcess .* died|AttributeError:.*_infer_resume_step'
  echo "[experiment] e120_runtime_recovery=20260905 watchdog_log=$OAT_ZERO_WATCHDOG_LOG_PATH"
fi
# END E120_RUNTIME_RECOVERY_20260905
'''


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare() -> dict[Path, tuple[bytes, bytes]]:
    source = SNAPSHOT / "src/oat_drgrpo/learner/run.py"
    runner = SNAPSHOT / "ops/run_experiment.sh"
    before_source, before_runner = source.read_bytes(), runner.read_bytes()
    after_source = patched_source(before_source.decode()).encode()
    if after_source.decode().replace(textwrap.indent(METHOD_SOURCE, "    "), "", 1).encode() != before_source:
        raise RuntimeError("resume patch changes more than the missing helper")
    anchor = 'MODEL="${OAT_ZERO_MODEL:-qwen2.5-0.5b-instruct}"\n'
    text = before_runner.decode()
    if BEGIN in text or text.count(anchor) != 1:
        raise RuntimeError("unexpected E120 shell patch layout")
    return {source: (before_source, after_source), runner: (before_runner, text.replace(anchor, BLOCK + "\n" + anchor, 1).encode())}


def validate(changes: dict[Path, tuple[bytes, bytes]]) -> list[str]:
    checks = []
    for path, (_, after) in changes.items():
        if path.suffix == ".sh":
            subprocess.run(["bash", "-n"], input=after, check=True)
        else:
            tree = ast.parse(after)
            mixin = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ZeroMathRunMixin")
            method = next(n for n in mixin.body if isinstance(n, ast.FunctionDef) and n.name == "_infer_resume_step")
            isolated = ast.Module(body=[method], type_ignores=[])
            namespace: dict[str, Any] = {"Path": Path, "Any": Any}
            exec(compile(ast.fix_missing_locations(isolated), str(path), "exec"), namespace)
            infer = namespace["_infer_resume_step"]
            for step in (2496, 2688):
                learner = SimpleNamespace(args=SimpleNamespace(resume_tag=f"step_{step:05d}"))
                assert infer(learner, {"steps": step}) == step
                assert infer(learner, {}) == step
                for invalid in (-1, True, None, "bad", step - 1, step + 0.5):
                    try:
                        infer(learner, {"steps": invalid})
                    except RuntimeError:
                        pass
                    else:
                        raise RuntimeError(f"accepted invalid checkpoint steps {invalid!r}")
            checks.append("actual snapshot resume method accepts 2496/2688 and rejects corrupt or mismatched counters")
    for stamp, smoke, job_id, expected in (
        ("e120r1_falcon1b_graph_fresh_frequency_s56", "0", "31048527", f"{ROOT}/slurm-31048527.out"),
        ("e120r1_falcon1b_pantry_fresh_frequency_s57", "0", "31048530", f"{ROOT}/slurm-31048530.out"),
        ("e119_level2_pantry_drgrpo_s43", "0", "123", "unchanged"),
        ("e120r1_smoke_qwen05b_graph_fresh_frequency_s43", "1", "123", "unchanged"),
    ):
        env = {"PATH": os.environ["PATH"], "ROOT_DIR": str(ROOT), "RUN_STAMP": stamp,
               "E120_SMOKE": smoke, "SLURM_JOB_ID": job_id, "OAT_ZERO_WATCHDOG_LOG_PATH": "unchanged"}
        output = subprocess.check_output(["bash", "-eu", "-c", BLOCK + '\nprintf "%s\\n" "$OAT_ZERO_WATCHDOG_LOG_PATH"\n'], env=env, text=True)
        assert output.splitlines()[-1] == expected
    checks.append("bash syntax and E120 science-only watchdog guards passed")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if RECORD.exists():
        prior = json.loads(RECORD.read_text())
        if all(sha(Path(row["path"]).read_bytes()) == row["after_sha256"] for row in prior["files"]):
            print(json.dumps({"already_applied": str(RECORD)}))
            return
        raise RuntimeError("record exists but patched runtime hashes drifted")
    changes = prepare()
    checks = validate(changes)
    record = {"schema": "e120-runtime-resume-recovery-v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "applied": args.apply, "scheduler_mutated": False, "scientific_configuration_changed": False,
              "source_change": "add missing checkpoint step inference helper already used by E118",
              "shell_change": "detect fatal errors in actual E120 stdout on future allocations",
              "qwen3_holds_changed": False, "checks": checks,
              "helper_path": str(Path(__file__).resolve()), "helper_sha256": sha(Path(__file__).read_bytes()),
              "files": [{"path": str(p), "before_sha256": sha(a), "after_sha256": sha(b)} for p, (a, b) in changes.items()]}
    if args.apply:
        AUDIT.mkdir(parents=True, exist_ok=True)
        for path, (before, after) in changes.items():
            if path.read_bytes() != before:
                raise RuntimeError(f"concurrent change: {path}")
            backup = AUDIT / f"{path.name}.before-{sha(before)[:12]}"
            backup.write_bytes(before)
            for row in record["files"]:
                if row["path"] == str(path):
                    row["backup"] = str(backup)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                handle.write(after)
                temporary = Path(handle.name)
            temporary.chmod(path.stat().st_mode & 0o777)
            temporary.replace(path)
        RECORD.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
