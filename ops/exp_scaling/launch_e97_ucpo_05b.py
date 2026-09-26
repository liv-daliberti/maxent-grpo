#!/usr/bin/env python3
"""Submit E97 UCPO cells behind a real 32-query learner smoke.

The 15 scientific cells inherit their data, schedule, placement, and completed
E78 control comparator cell-by-cell. Jobs are submitted held, audited through
Slurm, recorded atomically, and released only after the ledger exists. Every
scientific job has an afterok dependency on the non-scientific smoke.
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
import launch_e78_verified_replay_only_05b as e78  # noqa: E402
import launch_e81_semantic_maxent_verified_replay_05b as e81  # noqa: E402


DOMAINS = ("graph_coloring", "python_factors", "pantry_plan")
SEEDS = e81.SEEDS
PASSES = e81.PASSES
TRAIN_ROWS = e81.TRAIN_ROWS
CHECKPOINT_INTERVAL = e81.CHECKPOINT_INTERVAL
TARGET_STEPS = e81.TARGET_STEPS
ARM = "ucpo"
VARIANT = "ucpo"
TAU = 0.2
LEDGER = "var/artifacts/e97_ucpo_05b_jobs.json"
PROTOCOL = "paper/preregistration/e97_ucpo_05b_20260812.md"
PAIR_LEDGER = e81.PAIR_LEDGER
SOURCE_MANIFEST = e81.SOURCE_MANIFEST
SNAPSHOT_PREFIX = "e97_ucpo"
PATCHED_FILES = (
    "src/oat_drgrpo/args.py",
    "src/oat_drgrpo/learner/grpo.py",
    "src/oat_drgrpo/ucpo.py",
    "src/oat_drgrpo/rlep.py",  # imported unconditionally by grpo.py
    "ops/run_experiment.sh",
    "ops/train.sh",
)
SNAPSHOT_REQUIREMENTS = (
    ("src/oat_drgrpo/args.py", "ucpo_tau:"),
    ("src/oat_drgrpo/learner/grpo.py", "redistribute_ucpo_advantages"),
    ("src/oat_drgrpo/ucpo.py", "mass_error_max"),
    ("ops/run_experiment.sh", "ucpo)"),
    ("ops/train.sh", "--ucpo-tau"),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def selected_references(root: Path) -> list[dict[str, Any]]:
    runs = [r for r in e81.references(root) if str(r["domain"]) in DOMAINS]
    expected = {(domain, seed) for domain in DOMAINS for seed in SEEDS}
    found = {(str(r["domain"]), int(r["seed"])) for r in runs}
    if found != expected or len(runs) != len(expected):
        raise SystemExit("E97 source manifest does not cover exactly 15 cells")
    return runs


def selected_pairs(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    pairs = e81.pair_index(e81.pair_ledger(root))
    return {key: value for key, value in pairs.items() if key[0] in DOMAINS}


def verify_snapshot(snapshot: Path) -> None:
    missing = [
        f"{name}: {needle!r}"
        for name, needle in SNAPSHOT_REQUIREMENTS
        if needle not in (snapshot / name).read_text(encoding="utf-8")
    ]
    if missing:
        raise SystemExit("E97 snapshot is not UCPO-capable:\n  " + "\n  ".join(missing))


def run_stamp(domain: str, seed: int) -> str:
    return f"e97_ucpo_{e81.DOMAIN_TAGS[domain]}_s{seed}"


def save_path(root: Path, domain: str, seed: int) -> Path:
    return root / "var/data" / f"xdr_{e81.MODEL_TAG}_{VARIANT}_{run_stamp(domain, seed)}"


def objective() -> dict[str, str]:
    baseline = dict(e78.fixed_objective("control"))
    result = dict(baseline)
    result.update(
        {
            "OAT_ZERO_VARIANT": VARIANT,
            "OAT_ZERO_UCPO_TAU": repr(TAU),
            "OAT_ZERO_RLEP_EXPERIENCE_ROOT": "",
            "OAT_ZERO_RLEP_REPLAY_COUNT": "0",
        }
    )
    moved = {k for k in set(result) | set(baseline) if result.get(k) != baseline.get(k)}
    # The RLEP keys are explicit inert guards; only variant and tau are live.
    expected = {
        "OAT_ZERO_VARIANT",
        "OAT_ZERO_UCPO_TAU",
        "OAT_ZERO_RLEP_EXPERIENCE_ROOT",
        "OAT_ZERO_RLEP_REPLAY_COUNT",
    }
    if moved != expected:
        raise RuntimeError(f"E97 objective drift relative to E78 control: {sorted(moved)}")
    return result


def build_env(root: Path, run: dict[str, Any], snapshot: Path) -> tuple[dict[str, str], Path]:
    env, _ = e81.build_env(root, run, snapshot)
    domain, seed = str(run["domain"]), int(run["seed"])
    target = save_path(root, domain, seed)
    env.update({"SAVE_PATH": str(target), "RUN_STAMP": run_stamp(domain, seed)})
    env.update(objective())
    return env, target


def smoke_env(root: Path, run: dict[str, Any], snapshot: Path) -> tuple[dict[str, str], Path]:
    env, _ = build_env(root, run, snapshot)
    target = root / "var/data/e97_ucpo_smoke_graph_s43"
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": "e97_ucpo_smoke_graph_s43",
            "OAT_ZERO_MAX_TRAIN": "32",
            "OAT_ZERO_MAX_QUERIES": "32",
            "OAT_ZERO_NUM_PROMPT_EPOCH": "1",
            "OAT_ZERO_MAX_PROMPT_EPOCHS": "1",
            "OAT_ZERO_EVAL_PROMPT_INTERVAL": "32",
            "OAT_ZERO_SAVE_STEPS": "32",
            "OAT_ZERO_SAVE_FROM": "32",
            "OAT_ZERO_AUTO_RESUME": "0",
            "OAT_ZERO_WATCHDOG_REQUEUE": "0",
        }
    )
    return env, target


def sbatch_command(
    root: Path,
    run: dict[str, Any],
    env: dict[str, str],
    *,
    smoke: bool = False,
    dependency: str = "",
) -> list[str]:
    node = str(run["source_node"])
    place = e81.placement(node)
    domain, seed = str(run["domain"]), int(run["seed"])
    name = "e97-ucpo-smoke" if smoke else f"e97-{e81.DOMAIN_TAGS[domain][:6]}-ucpo-s{seed}"
    command = [
        "sbatch",
        "--parsable",
        "--hold",
        f"--job-name={name}",
        f"--export=ALL,{','.join(f'{k}={v}' for k, v in env.items())}",
        f"--partition={place['partition']}",
        f"--account={place['account']}",
        f"--nodelist={node}",
        f"--gres={place['gres']}",
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=1-12:00:00",
        "--nice=100",
    ]
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    command.append(str(root / "ops/slurm/train_node302.slurm"))
    return command


def submit_held(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "sbatch failed")
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"invalid sbatch response: {result.stdout!r}")
    return job_id


def audit_held(job_id: str, *, name: str, expected: tuple[str, ...]) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"cannot inspect held E97 job {job_id}: {result.stderr.strip()}")
    record = result.stdout
    required = ("JobState=PENDING", "Reason=JobHeldUser", f"JobName={name}") + expected
    missing = [item for item in required if item not in record]
    if missing:
        raise RuntimeError(f"held E97 job {job_id} lacks {missing}")
    return record


def cancel(job_ids: list[str]) -> None:
    if job_ids:
        subprocess.run(["scancel", *job_ids], check=False)


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--snapshot-root", type=Path)
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")
    protocol, ledger = root / PROTOCOL, root / LEDGER
    for required in (protocol, root / SOURCE_MANIFEST, root / PAIR_LEDGER):
        if not required.is_file():
            raise SystemExit(f"required E97 input is absent: {required}")
    if args.submit and ledger.exists():
        raise SystemExit(f"refusing duplicate E97 submission: {ledger}")

    pairs = selected_pairs(root)
    runs = selected_references(root)
    base_snapshot = (
        args.snapshot_root.resolve()
        if args.snapshot_root
        else Path(e81.pair_ledger(root)["snapshot_root"])
    )
    snapshot, patched = e81.ensure_paired_snapshot(
        root,
        base_snapshot,
        prefix=SNAPSHOT_PREFIX,
        patched_files=PATCHED_FILES,
    )
    verify_snapshot(snapshot)

    cells: list[dict[str, Any]] = []
    for run in runs:
        domain, seed = str(run["domain"]), int(run["seed"])
        pair = pairs[(domain, seed)]
        if str(pair["control"]["source_node"]) != str(run["source_node"]):
            raise SystemExit(f"{domain}/s{seed}: E78 placement drift")
        env, target = build_env(root, run, snapshot)
        if target.exists():
            raise SystemExit(f"refusing to overwrite E97 output: {target}")
        cells.append({"run": run, "env": env, "target": target, "pair": pair})
    smoke_run = next(r for r in runs if r["domain"] == "graph_coloring" and r["seed"] == 43)
    smoke_vars, smoke_target = smoke_env(root, smoke_run, snapshot)
    if smoke_target.exists():
        raise SystemExit(f"refusing to overwrite E97 smoke: {smoke_target}")

    if args.dry_run or not args.submit:
        print(shlex.join(sbatch_command(root, smoke_run, smoke_vars, smoke=True)))
        for cell in cells:
            print(shlex.join(sbatch_command(root, cell["run"], cell["env"], dependency="SMOKE_JOB_ID")))
        print(f"[e97] smoke=1 scientific={len(cells)} snapshot={snapshot} patched={patched}")
        return 0

    submitted: list[str] = []
    records: list[dict[str, Any]] = []
    try:
        smoke_id = submit_held(sbatch_command(root, smoke_run, smoke_vars, smoke=True))
        submitted.append(smoke_id)
        smoke_record = audit_held(
            smoke_id,
            name="e97-ucpo-smoke",
            expected=("OAT_ZERO_VARIANT=ucpo", "OAT_ZERO_UCPO_TAU=0.2", "OAT_ZERO_MAX_QUERIES=32"),
        )
        for cell in cells:
            run = cell["run"]
            domain, seed = str(run["domain"]), int(run["seed"])
            name = f"e97-{e81.DOMAIN_TAGS[domain][:6]}-ucpo-s{seed}"
            job_id = submit_held(sbatch_command(root, run, cell["env"], dependency=smoke_id))
            submitted.append(job_id)
            held = audit_held(
                job_id,
                name=name,
                expected=(
                    f"Dependency=afterok:{smoke_id}",
                    f"ReqNodeList={run['source_node']}",
                    f"OAT_ZERO_SEED={seed}",
                    "OAT_ZERO_VARIANT=ucpo",
                    "OAT_ZERO_UCPO_TAU=0.2",
                    "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
                    "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=1",
                    "OAT_ZERO_NUM_PROMPT_EPOCH=8",
                    "OAT_ZERO_MAX_PROMPT_EPOCHS=8",
                ),
            )
            control = cell["pair"]["control"]
            records.append(
                {
                    "domain": domain,
                    "arm": ARM,
                    "seed": seed,
                    "source_node": str(run["source_node"]),
                    "run_stamp": run_stamp(domain, seed),
                    "run_dir": str(cell["target"]),
                    "job_id": int(job_id),
                    "smoke_dependency_job_id": int(smoke_id),
                    "paired_e78_control": {
                        "job_id": int(control["job_id"]),
                        "run_stamp": str(control["run_stamp"]),
                        "run_dir": str(control["run_dir"]),
                    },
                    "held_scheduler_record": held,
                }
            )

        payload = {
            "schema": "e97_ucpo_05b_jobs_v1",
            "cohort": "e97",
            "released": False,
            "model": "Qwen2.5-0.5B-Instruct",
            "protocol": str(protocol),
            "protocol_sha256": e81.digest(protocol),
            "launcher_sha256": e81.digest(Path(__file__)),
            "source_manifest": str(root / SOURCE_MANIFEST),
            "source_manifest_sha256": e81.digest(root / SOURCE_MANIFEST),
            "pair_ledger": str(root / PAIR_LEDGER),
            "pair_ledger_sha256": e81.digest(root / PAIR_LEDGER),
            "snapshot_root": str(snapshot),
            "snapshot_patched_files": patched,
            "domains": list(DOMAINS),
            "seeds": list(SEEDS),
            "arms": [ARM],
            "inherited_arms": ["control"],
            "passes": PASSES,
            "train_rows": TRAIN_ROWS,
            "target_steps": TARGET_STEPS,
            "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
            "registered_passes": [i / 2 for i in range(2 * PASSES + 1)],
            "variant": VARIANT,
            "ucpo_tau": TAU,
            "objective": "DrGRPO_with_uniform_correct_advantage_redistribution",
            "scientific_difference": "against E78 control: UCPO tau=.2 advantage redistribution only",
            "smoke": {
                "scientific": False,
                "job_id": int(smoke_id),
                "run_dir": str(smoke_target),
                "max_queries": 32,
                "held_scheduler_record": smoke_record,
            },
            "runs": records,
        }
        e81.atomic_json(ledger, payload)
        for job_id in submitted:
            subprocess.run(["scontrol", "release", job_id], check=True)
        payload["released"] = True
        e81.atomic_json(ledger, payload)
    except Exception:
        cancel(submitted)
        raise
    print(f"[e97] released smoke {smoke_id} and {len(records)} dependent scientific cells")
    print(f"[e97] ledger {ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
