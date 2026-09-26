#!/usr/bin/env python3
"""Submit E85: re-run every PantryPlan cell whose semantic term never fired.

PantryPlan is a canonical-action task. The semantic term derived its outcome key
from the free-form text extractor, which returns None for an eight-token action
sequence, so every PantryPlan row in E81, E82, and E83 was scored unparseable
and contributed exact zero. The verifier and the replay bank were unaffected,
which is why the failure was silent: reward-positive fraction .544, parseable
fraction .000, in 15 of 15 cells.

E85 re-runs those 15 cells against a runtime that binds the task's own
canonicalization surfaces. The original cells are retained as audit-only
evidence of the defect and are excluded from every result.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e81_semantic_maxent_verified_replay_05b as e81  # noqa: E402
import launch_e82_falcon_semantic_maxent_verified_replay as e82  # noqa: E402
import launch_e83_semantic_maxent_without_replay_05b as e83  # noqa: E402
import launch_e79_falcon1b_aligned_verified_replay as e79  # noqa: E402


DOMAIN = "pantry_plan"
LEDGER = "var/artifacts/e85_pantry_semantic_repair_jobs.json"
PROTOCOL = "paper/preregistration/e85_pantry_semantic_repair_20260809.md"

# One file wider than the parent cohorts: the semantic key derivation itself.
# The new branch sits inside `if semantic_shannon_tracker is not None:`, so it
# is unreachable for every arm that does not enable semantic MaxEnt -- which is
# every comparator this cohort is read against.
PATCHED_FILES = (
    "src/oat_drgrpo/args.py",
    "ops/run_experiment.sh",
    "src/oat_drgrpo/learner/grpo.py",
)

# parent tag -> (module, ledger, base-snapshot key, snapshot prefix)
PARENTS = {
    "e81": (e81, "var/artifacts/e81_semantic_maxent_verified_replay_05b_jobs.json",
            "e78_snapshot_root", "e85_pantry_repair_qwen"),
    "e82": (e82, "var/artifacts/e82_falcon_semantic_maxent_verified_replay_jobs.json",
            "e79_snapshot_root", "e85_pantry_repair_falcon"),
    "e83": (e83, "var/artifacts/e83_semantic_maxent_without_replay_05b_jobs.json",
            "e78_snapshot_root", "e85_pantry_repair_qwen_noreplay"),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_stamp(parent: str, seed: int) -> str:
    return f"e85_pantry_repair_{parent}_pantry_s{seed}"


def save_path(root: Path, parent: str, seed: int, variant: str, model_tag: str) -> Path:
    return root / "var/data" / f"xdr_{model_tag}_{variant}_{run_stamp(parent, seed)}"


def plan_cells(root: Path) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for parent, (module, ledger_name, base_key, prefix) in PARENTS.items():
        ledger = json.loads((root / ledger_name).read_text(encoding="utf-8"))
        base_snapshot = Path(ledger[base_key])
        snapshot, patched = e81.ensure_paired_snapshot(
            root, base_snapshot, prefix=prefix, patched_files=PATCHED_FILES
        )
        if patched != sorted(PATCHED_FILES):
            raise SystemExit(f"{parent}: snapshot diverges outside the patch set")

        if parent == "e82":
            falcon_root = e79.model_root(root)
            templates = {str(r["domain"]): r for r in e79.references(root)
                         if str(r["domain"]) == DOMAIN}
            sources = [r for r in e79.references(root) if str(r["domain"]) == DOMAIN]
        else:
            sources = [r for r in e81.references(root) if str(r["domain"]) == DOMAIN]

        for run in sources:
            seed = int(run["seed"])
            original = next(
                r for r in ledger["runs"]
                if r["domain"] == DOMAIN and int(r["seed"]) == seed
            )
            if parent == "e82":
                env, _ = e82.build_env(root, run, snapshot, falcon_root)
                variant, model_tag = e82.VARIANT, e82.MODEL_TAG
                command_of = lambda r, e: e82.sbatch_command(root, r, e)  # noqa: E731
            elif parent == "e83":
                env, _ = e83.build_env(root, run, snapshot)
                variant, model_tag = e83.VARIANT, e83.MODEL_TAG
                command_of = lambda r, e: e83.sbatch_command(root, r, e)  # noqa: E731
            else:
                env, _ = e81.build_env(root, run, snapshot)
                variant, model_tag = e81.VARIANT, e81.MODEL_TAG
                command_of = lambda r, e: e81.sbatch_command(root, r, e)  # noqa: E731

            target = save_path(root, parent, seed, variant, model_tag)
            if target.exists():
                raise SystemExit(f"refusing to overwrite existing E85 run: {target}")
            env.update({"SAVE_PATH": str(target),
                        "RUN_STAMP": run_stamp(parent, seed)})
            command = command_of(run, env)
            command = [
                part.replace(f"-sem-s{seed}", f"-r{parent[-2:]}-s{seed}")
                     .replace(f"-sonly-s{seed}", f"-r{parent[-2:]}-s{seed}")
                if part.startswith("--job-name=") else part
                for part in command
            ]
            cells.append({
                "parent": parent,
                "domain": DOMAIN,
                "seed": seed,
                "run_stamp": run_stamp(parent, seed),
                "run_dir": str(target),
                "snapshot_root": str(snapshot),
                "supersedes": {
                    "run_stamp": str(original["run_stamp"]),
                    "run_dir": str(original["run_dir"]),
                    "job_id": int(original["job_id"]),
                },
                "command": command,
            })
    return cells


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")

    protocol = root / PROTOCOL
    if not protocol.is_file():
        raise SystemExit(f"required frozen input is absent: {protocol}")
    ledger_path = root / LEDGER
    if args.submit and ledger_path.exists():
        raise SystemExit(f"refusing duplicate E85 submission: {ledger_path}")

    cells = plan_cells(root)
    if len(cells) != 15:
        raise SystemExit(f"E85 expected 15 cells, found {len(cells)}")

    if args.dry_run or not args.submit:
        for cell in cells:
            print(" ".join(shlex.quote(p) for p in cell["command"]))
        print(f"[e85] dry_run=True cells={len(cells)}")
        return 0

    submitted: list[str] = []
    records: list[dict[str, Any]] = []
    try:
        for cell in cells:
            result = subprocess.run(cell["command"], capture_output=True,
                                    text=True, check=False)
            if result.returncode != 0:
                raise RuntimeError(
                    f"submission failed for {cell['run_stamp']}: {result.stderr.strip()}"
                )
            job_id = result.stdout.strip().split(";", 1)[0]
            if not job_id.isdigit():
                raise RuntimeError(f"invalid job id: {result.stdout!r}")
            submitted.append(job_id)
            records.append({k: cell[k] for k in
                            ("parent", "domain", "seed", "run_stamp", "run_dir",
                             "snapshot_root", "supersedes")}
                           | {"job_id": int(job_id), "arm": "semantic"})
        payload = {
            "schema": "e85_pantry_semantic_repair_jobs_v1",
            "protocol": str(protocol),
            "protocol_sha256": e81.digest(protocol),
            "launcher_sha256": e81.digest(Path(__file__)),
            "snapshot_patched_files": list(PATCHED_FILES),
            "domains": [DOMAIN],
            "arms": ["semantic"],
            "seeds": list(e81.SEEDS),
            "train_rows": e81.TRAIN_ROWS,
            "passes": e81.PASSES,
            "target_steps": e81.TARGET_STEPS,
            "checkpoint_interval_steps": e81.CHECKPOINT_INTERVAL,
            "semantic_coefficient": e81.SEMANTIC_COEF,
            "replay_weight": e81.REPLAY_WEIGHT,
            "defect": "semantic outcome key used the free-form text extractor "
                      "on a canonical-action task; parseable fraction .000",
            "supersedes_cohorts": ["e81", "e82", "e83"],
            "runs": records,
            "released": False,
        }
        e81.atomic_json(ledger_path, payload)
        for job_id in submitted:
            release = subprocess.run(["scontrol", "release", job_id],
                                     capture_output=True, text=True, check=False)
            if release.returncode != 0:
                raise RuntimeError(f"release failed for {job_id}: {release.stderr.strip()}")
        payload["released"] = True
        e81.atomic_json(ledger_path, payload)
    except Exception:
        e81.cancel(submitted)
        if ledger_path.exists():
            ledger_path.unlink()
        raise

    print(f"[e85] cells={len(records)} released={len(submitted)} ledger={ledger_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
