#!/usr/bin/env python3
"""Submit E90: verified replay with the dose normalized by bank occupancy.

Uniform replay splits the dose across the banked modes, so each mode receives
`alpha / n` and a prompt that has discovered more modes protects each one less.
Measured over E78's 25 fixed-dose replay cells, occupancy runs 1..16, so
per-mode pressure spans 16x at an identical nominal coefficient -- the opposite
of the per-mode recurrent dose the retention argument relies on.

E90 replaces the dose with `c * n`, holding per-mode pressure at `c = .0325`
(= .10 / E[n], E[n] = 3.08 pooled over E78). At the mean occupancy both arms
deliver the same dose, so this redistributes a matched total mass rather than
adding one. No new ceiling: `n` is bounded by the registered bank capacity, so
`alpha <= .52` follows from a parameter that already exists.

Cells, seeds, schedule, decoding, placement and code are inherited from the E78
replay arm, which is the comparator. E78 is never re-run.
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


DOMAINS = e81.DOMAINS
ARM = "bank_normalized_replay"
SEEDS = e81.SEEDS
PASSES = e81.PASSES
TRAIN_ROWS = e81.TRAIN_ROWS
CHECKPOINT_INTERVAL = e81.CHECKPOINT_INTERVAL
TARGET_STEPS = e81.TARGET_STEPS
REPLAY_WEIGHT = e81.REPLAY_WEIGHT
VARIANT = "bank_normalized_replay"
MODEL_TAG = e81.MODEL_TAG

# .10 / E[n] with E[n] = 3.08, the pooled mean bank occupancy over E78's 25
# fixed-dose replay cells. Derived from data before submission, as registered.
PER_MODE_COEFFICIENT = 0.0325
MEASURED_MEAN_OCCUPANCY = 3.08

LEDGER = "var/artifacts/e90_bank_normalized_replay_05b_jobs.json"
PROTOCOL = "paper/preregistration/e90_bank_normalized_replay_05b_20260810.md"
SOURCE_MANIFEST = e81.SOURCE_MANIFEST
PAIR_LEDGER = e81.PAIR_LEDGER
SNAPSHOT_PREFIX = "e90_bank_normalized_replay"

# The comparator's default patch set is (args.py, run_experiment.sh) -- the two
# files E81 needed. This arm also changes the learner that applies the dose and
# the launcher script that passes the flag. Inheriting the default silently
# produced 25 cells that ran at the fixed alpha, i.e. exact duplicates of the
# comparator, because a snapshot that lacks the flag plumbing simply never
# passes the flag. Every file the arm touches must be listed here.
PATCHED_FILES = (
    "src/oat_drgrpo/args.py",
    "src/oat_drgrpo/learner/grpo.py",
    "ops/run_experiment.sh",
    "ops/train.sh",
)

# Read out of the snapshot, not the working tree: this is what will actually
# run. The equivalent guard inside train.sh cannot protect this cohort, because
# train.sh is itself snapshotted -- a stale copy has no guard to execute.
SNAPSHOT_REQUIREMENTS = (
    ("src/oat_drgrpo/args.py", "online_canonical_replay_bank_normalized:"),
    ("src/oat_drgrpo/learner/grpo.py", "replay_banked_modes"),
    ("src/oat_drgrpo/learner/grpo.py", "canonical_replay_per_mode_pressure"),
    ("ops/run_experiment.sh", "bank_normalized_replay)"),
    ("ops/train.sh", "online-canonical-replay-bank-normalized"),
)


def verify_snapshot(snapshot_root: Path) -> None:
    """Fail before submission if the frozen runtime cannot apply the dose rule."""

    missing = [
        f"{name}: {needle!r}"
        for name, needle in SNAPSHOT_REQUIREMENTS
        if needle not in (snapshot_root / name).read_text(encoding="utf-8")
    ]
    if missing:
        raise SystemExit(
            "frozen snapshot cannot apply the bank-normalized dose, so these "
            "cells would run as fixed-dose duplicates of the comparator:\n  "
            + "\n  ".join(missing)
        )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_stamp(domain: str, seed: int) -> str:
    return f"e90_bank_normalized_{e81.DOMAIN_TAGS[domain]}_{ARM}_s{seed}"


def save_path(root: Path, domain: str, seed: int) -> Path:
    return root / "var/data" / f"xdr_{MODEL_TAG}_{VARIANT}_{run_stamp(domain, seed)}"


def fixed_objective() -> dict[str, str]:
    """The E78 replay arm's objective with the dose rule as the only addition.

    Taken from the comparator's own launcher rather than retyped or subtracted
    from a third cohort, so "inherits everything from E78's replay arm" is a
    property of the code and not a claim in a docstring. The assertion pins
    exactly which keys move; a future edit to E78 cannot silently change what
    E90 is read against.
    """

    baseline = dict(e78.fixed_objective("replay"))
    objective = dict(baseline)
    objective["OAT_ZERO_VARIANT"] = VARIANT
    objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY_BANK_NORMALIZED"] = "1"
    objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY_PER_MODE_COEFFICIENT"] = repr(
        PER_MODE_COEFFICIENT
    )

    moved = {
        key
        for key in set(objective) | set(baseline)
        if objective.get(key) != baseline.get(key)
    }
    expected = {
        "OAT_ZERO_VARIANT",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_BANK_NORMALIZED",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_PER_MODE_COEFFICIENT",
    }
    if moved != expected:
        raise RuntimeError(
            f"E90 differs from E78's replay arm beyond the dose rule: "
            f"{sorted(moved ^ expected)}"
        )
    if abs(REPLAY_WEIGHT / MEASURED_MEAN_OCCUPANCY - PER_MODE_COEFFICIENT) > 5e-4:
        raise RuntimeError(
            "per-mode coefficient is no longer .10 / E[n]; the registered "
            "mass-matching premise would not hold"
        )
    return objective


def build_env(
    root: Path, run: dict[str, Any], snapshot_root: Path
) -> tuple[dict[str, str], Path]:
    domain = str(run["domain"])
    seed = int(run["seed"])
    env, _ = e81.build_env(root, run, snapshot_root)
    target = save_path(root, domain, seed)
    env.update({"SAVE_PATH": str(target), "RUN_STAMP": run_stamp(domain, seed)})
    env.update(fixed_objective())
    return env, target


def sbatch_command(root: Path, run: dict[str, Any], env: dict[str, str]) -> list[str]:
    node = str(run["source_node"])
    place = e81.placement(node)
    name = f"e90-{e81.DOMAIN_TAGS[str(run['domain'])][:6]}-bnr-s{run['seed']}"
    export_pairs = ",".join(f"{key}={value}" for key, value in env.items())
    return [
        "sbatch",
        "--parsable",
        "--hold",
        f"--job-name={name}",
        f"--export=ALL,{export_pairs}",
        f"--partition={place['partition']}",
        f"--account={place['account']}",
        f"--nodelist={node}",
        f"--gres={place['gres']}",
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=1-12:00:00",
        "--nice=100",
        str(root / "ops/slurm/train_node302.slurm"),
    ]


def held_job_audit(job_id: str, run: dict[str, Any]) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"cannot inspect held E90 job {job_id}: {result.stderr.strip()}"
        )
    record = result.stdout
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"JobName=e90-{e81.DOMAIN_TAGS[str(run['domain'])][:6]}-bnr-s{run['seed']}",
        f"ReqNodeList={run['source_node']}",
        f"OAT_ZERO_VARIANT={VARIANT}",
        f"OAT_ZERO_SEED={run['seed']}",
        "OAT_ZERO_NUM_PROMPT_EPOCH=8",
        "OAT_ZERO_MAX_PROMPT_EPOCHS=8",
        "OAT_ZERO_EVAL_PROMPT_INTERVAL=192",
        "OAT_ZERO_SAVE_STEPS=192",
        # the defining switch, and the constant it applies
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_BANK_NORMALIZED=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_PER_MODE_COEFFICIENT=0.0325",
        # inherited from the comparator, verified present rather than assumed
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=verified_likelihood_per_rollout",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY=16",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=0",
        # the semantic term must be absent: E90 is a replay-dose arm
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.0",
        "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=0",
        "OAT_ZERO_MAXENT_ALPHA=0.0",
        "OAT_ZERO_POLICY_ENTROPY_COEF=0.0",
        "OAT_ZERO_SEED_ENTROPY_ALPHA=0.0",
    )
    missing = [text for text in required if text not in record]
    if missing:
        raise RuntimeError(f"held E90 job {job_id} lacks {missing}")
    return record


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--snapshot-root", type=Path)
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")

    protocol = root / PROTOCOL
    source_manifest = root / SOURCE_MANIFEST
    for required in (protocol, source_manifest):
        if not required.is_file():
            raise SystemExit(f"required frozen E90 input is absent: {required}")
    ledger = root / LEDGER
    if args.submit and ledger.exists():
        raise SystemExit(f"refusing duplicate E90 submission: {ledger}")

    source_runs = e81.references(root)
    e78_snapshot = Path(e81.pair_ledger(root)["snapshot_root"])
    if args.snapshot_root:
        e78_snapshot = args.snapshot_root.resolve()
    snapshot_root, patched = e81.ensure_paired_snapshot(
        root, e78_snapshot, prefix=SNAPSHOT_PREFIX, patched_files=PATCHED_FILES
    )
    verify_snapshot(snapshot_root)

    planned: list[dict[str, Any]] = []
    for run in source_runs:
        domain = str(run["domain"])
        seed = int(run["seed"])
        if domain not in DOMAINS or seed not in SEEDS:
            continue
        env, target = build_env(root, run, snapshot_root)
        planned.append(
            {
                "arm": ARM,
                "domain": domain,
                "seed": seed,
                "source_node": str(run["source_node"]),
                "run_dir": str(target),
                "run_stamp": run_stamp(domain, seed),
                "env": env,
            }
        )

    if len(planned) != len(DOMAINS) * len(SEEDS):
        raise SystemExit(
            f"expected {len(DOMAINS) * len(SEEDS)} E90 cells, planned {len(planned)}"
        )

    if args.dry_run or not args.submit:
        for cell in planned:
            command = sbatch_command(root, cell, cell["env"])
            print(f"{cell['domain']:<16} s{cell['seed']}  {shlex.join(command[:8])}")
        print(f"\n{len(planned)} cells planned; per-mode coefficient "
              f"{PER_MODE_COEFFICIENT} (= .10 / {MEASURED_MEAN_OCCUPANCY})")
        print(f"snapshot: {snapshot_root}  patched files: {len(patched)}")
        return 0

    submitted: list[dict[str, Any]] = []
    job_ids: list[str] = []
    try:
        for cell in planned:
            command = sbatch_command(root, cell, cell["env"])
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            job_id = result.stdout.strip()
            job_ids.append(job_id)
            record = held_job_audit(job_id, cell)
            submitted.append(
                {
                    **{k: v for k, v in cell.items() if k != "env"},
                    "job_id": job_id,
                    "held_scheduler_record": record,
                }
            )
    except Exception:
        if job_ids:
            subprocess.run(["scancel", *job_ids], check=False)
        raise

    payload = {
        "schema": "e90_bank_normalized_replay_05b_jobs_v1",
        "cohort": "e90",
        "released": True,
        "model": "Qwen2.5-0.5B-Instruct",
        "protocol": PROTOCOL,
        "protocol_sha256": e78.digest(protocol),
        "launcher_sha256": e78.digest(Path(__file__).resolve()),
        "source_manifest": SOURCE_MANIFEST,
        "source_manifest_sha256": e78.digest(source_manifest),
        # `arms` names this cohort's own arms and `inherited_arms` those it is
        # read against; the shared snapshot loader requires both, so a cohort
        # that omits them is invisible to the campaign table.
        "arms": [ARM],
        "inherited_arms": ["control", "replay"],
        "pair_ledger": str(root / PAIR_LEDGER),
        "comparator_ledger": PAIR_LEDGER,
        "objective": "uniform_verified_likelihood_at_bank_normalized_dose",
        "scientific_difference": (
            "against E78 replay: the replay dose rule only, "
            "alpha = per_mode_coefficient * banked_modes"
        ),
        "snapshot_root": str(snapshot_root),
        "snapshot_patched_files": len(patched),
        "variant": VARIANT,
        "per_mode_coefficient": PER_MODE_COEFFICIENT,
        "measured_mean_occupancy": MEASURED_MEAN_OCCUPANCY,
        "replay_weight": REPLAY_WEIGHT,
        "replay_weight_of_comparator": REPLAY_WEIGHT,
        "domains": list(DOMAINS),
        "seeds": list(SEEDS),
        "passes": PASSES,
        "registered_passes": [i / 2 for i in range(2 * PASSES + 1)],
        "train_rows": TRAIN_ROWS,
        "target_steps": TARGET_STEPS,
        "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
        "runs": submitted,
    }
    e78.atomic_json(ledger, payload)
    print(f"submitted {len(submitted)} held E90 cells; ledger {ledger}")
    print("release with: scontrol release <job ids>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
