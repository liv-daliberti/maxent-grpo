#!/usr/bin/env python3
"""Submit E87: Qwen2.5-3B verified replay plus fixed semantic MaxEnt, seed 70.

One paired seed, five domains, five cells. Seed 70 is chosen because it is the
only seed where E80-R1 has *both* arms terminal in all five domains, so every
E87 cell is immediately pairable rather than waiting on its own comparator.

Cell definitions come from the E80-R1 launcher and the objective from the E81
launcher; this file only combines them and raises the scheduling priority.
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
import launch_e80r1_qwen3b_aligned_verified_replay as e80r1  # noqa: E402
import launch_e81_semantic_maxent_verified_replay_05b as e81  # noqa: E402


DOMAINS = e80r1.DOMAINS
ARM = "semantic"
SEED = 70
PASSES = e80r1.PASSES
TRAIN_ROWS = e80r1.TRAIN_ROWS
CHECKPOINT_INTERVAL = e80r1.CHECKPOINT_INTERVAL
TARGET_STEPS = e80r1.TARGET_STEPS
REPLAY_WEIGHT = e80r1.REPLAY_WEIGHT
SEMANTIC_COEF = e81.SEMANTIC_COEF
VARIANT = e81.VARIANT
MODEL_TAG = e80r1.MODEL_TAG

LEDGER = "var/artifacts/e87_qwen3b_semantic_maxent_seed70_jobs.json"
PROTOCOL = "paper/preregistration/e87_qwen3b_semantic_maxent_seed70_20260809.md"
PAIR_LEDGER = "var/artifacts/e80r1_qwen3b_aligned_verified_replay_jobs.json"
SNAPSHOT_PREFIX = "e87_qwen3b_semantic_maxent"

# Includes the canonical-action key repair, so PantryPlan's semantic term is
# live here from the first update rather than needing a later repair cohort.
PATCHED_FILES = (
    "src/oat_drgrpo/args.py",
    "ops/run_experiment.sh",
    "src/oat_drgrpo/learner/grpo.py",
)

# Ahead of E80-R1's remaining cells, which sit at nice=100 on the same node.
NICE = 50


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_stamp(domain: str) -> str:
    return f"e87_qwen3b_semantic_{e80r1.DOMAIN_TAGS[domain]}_{ARM}_s{SEED}"


def save_path(root: Path, domain: str) -> Path:
    return root / "var/data" / f"xdr_{MODEL_TAG}_{VARIANT}_{run_stamp(domain)}"


def fixed_objective() -> dict[str, str]:
    """E81's objective, re-checked against E80-R1's replay dose."""

    objective = e81.fixed_objective()
    if float(objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA"]) != REPLAY_WEIGHT:
        raise RuntimeError("E87 replay weight drifted from the E80-R1 dose")
    if float(objective["OAT_ZERO_SEMANTIC_SHANNON_COEF"]) != SEMANTIC_COEF:
        raise RuntimeError("E87 semantic coefficient drifted from E81")
    return objective


def pair_state(root: Path) -> dict[str, dict[str, Any]]:
    """Require both E80-R1 arms terminal at this seed, in every domain."""

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import status_e78 as status

    ledger = json.loads((root / PAIR_LEDGER).read_text(encoding="utf-8"))
    target = int(ledger["target_steps"])
    pairs: dict[str, dict[str, Any]] = {}
    for run in ledger["runs"]:
        if int(run["seed"]) != SEED:
            continue
        directory = Path(run["run_dir"])
        complete = status.is_complete(directory, status.run_step(directory), target)
        pairs.setdefault(str(run["domain"]), {})[str(run["arm"])] = {
            "run_stamp": str(run["run_stamp"]),
            "run_dir": str(run["run_dir"]),
            "job_id": int(run["job_id"]),
            "terminal": bool(complete),
        }
    for domain in DOMAINS:
        arms = pairs.get(domain, {})
        if set(arms) != {"control", "replay"}:
            raise SystemExit(f"E80-R1 seed {SEED} lacks both arms for {domain}")
        missing = [a for a, v in arms.items() if not v["terminal"]]
        if missing:
            raise SystemExit(
                f"{domain}: E80-R1 seed {SEED} {missing} not terminal; "
                "E87 exists to be immediately pairable, so it fails closed here"
            )
    return pairs


def build_env(
    root: Path, run: dict[str, Any], snapshot: Path, qwen_root: Path
) -> tuple[dict[str, str], Path]:
    domain = str(run["domain"])
    env, _ = e80r1.build_env(root, run, "replay", snapshot, qwen_root)
    target = save_path(root, domain)
    env.update({"SAVE_PATH": str(target), "RUN_STAMP": run_stamp(domain)})
    env.update(fixed_objective())
    return env, target


def sbatch_command(root: Path, run: dict[str, Any], env: dict[str, str]) -> list[str]:
    domain = str(run["domain"])
    node, gpu = e80r1.placement(domain, SEED)
    name = f"e87-{e80r1.DOMAIN_TAGS[domain][:6]}-sem-s{SEED}"
    export_pairs = ",".join(f"{key}={value}" for key, value in env.items())
    return [
        "sbatch",
        "--parsable",
        "--hold",
        f"--job-name={name}",
        f"--export=ALL,{export_pairs}",
        "--partition=mltheory",
        "--account=mltheory",
        f"--nodelist={node}",
        f"--gres=gpu:{gpu}:1",
        "--cpus-per-task=16",
        "--mem=128G",
        "--time=3-00:00:00",
        f"--nice={NICE}",
        str(root / "ops/slurm/train_node302.slurm"),
    ]


def held_job_audit(job_id: str, run: dict[str, Any], qwen_root: Path) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot inspect held E87 job {job_id}: {result.stderr.strip()}")
    domain = str(run["domain"])
    node, gpu = e80r1.placement(domain, SEED)
    record = result.stdout
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"JobName=e87-{e80r1.DOMAIN_TAGS[domain][:6]}-sem-s{SEED}",
        f"ReqNodeList={node}",
        f"TresPerNode=gres/gpu:{gpu}:1",
        f"OAT_ZERO_PRETRAIN={qwen_root}",
        f"OAT_ZERO_VARIANT={VARIANT}",
        f"OAT_ZERO_SEED={SEED}",
        "OAT_ZERO_NUM_PROMPT_EPOCH=8",
        "OAT_ZERO_EVAL_PROMPT_INTERVAL=192",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.1",
        "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=1",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=verified_likelihood_per_rollout",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=0",
        "OAT_ZERO_MAXENT_ALPHA=0.0",
        "OAT_ZERO_POLICY_ENTROPY_COEF=0.0",
    )
    missing = [text for text in required if text not in record]
    if missing:
        raise RuntimeError(f"held E87 job {job_id} lacks {missing}")
    return record


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
        raise SystemExit(f"refusing duplicate E87 submission: {ledger_path}")

    pairs = pair_state(root)
    qwen_root = e80r1.model_root(root)
    sources = [r for r in e80r1.references(root) if int(r["seed"]) == SEED]
    if len(sources) != 5:
        raise SystemExit(f"E87 expected 5 seed-{SEED} templates, found {len(sources)}")

    parent = json.loads((root / PAIR_LEDGER).read_text(encoding="utf-8"))
    snapshot, patched = e81.ensure_paired_snapshot(
        root, Path(parent["snapshot_root"]),
        prefix=SNAPSHOT_PREFIX, patched_files=PATCHED_FILES,
    )

    planned = []
    for run in sources:
        env, target = build_env(root, run, snapshot, qwen_root)
        if target.exists():
            raise SystemExit(f"refusing to overwrite existing E87 run: {target}")
        planned.append({
            "domain": str(run["domain"]), "arm": ARM, "seed": SEED,
            "run_stamp": run_stamp(str(run["domain"])), "run_dir": str(target),
            "paired_e80r1_runs": pairs[str(run["domain"])],
            "command": sbatch_command(root, run, env), "template": run,
        })

    if args.dry_run or not args.submit:
        for cell in planned:
            print(" ".join(shlex.quote(p) for p in cell["command"]))
        print(f"[e87] dry_run=True cells={len(planned)} snapshot={snapshot} "
              f"patched={patched} nice={NICE}")
        return 0

    submitted, records = [], []
    try:
        for cell in planned:
            result = subprocess.run(cell["command"], capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"submission failed for {cell['run_stamp']}: {result.stderr.strip()}")
            job_id = result.stdout.strip().split(";", 1)[0]
            if not job_id.isdigit():
                raise RuntimeError(f"invalid job id: {result.stdout!r}")
            submitted.append(job_id)
            held = held_job_audit(job_id, cell["template"], qwen_root)
            records.append({k: cell[k] for k in
                            ("domain", "arm", "seed", "run_stamp", "run_dir",
                             "paired_e80r1_runs")}
                           | {"job_id": int(job_id), "held_scheduler_record": held})
        payload = {
            "schema": "e87_qwen3b_semantic_maxent_seed70_jobs_v1",
            "protocol": str(protocol),
            "protocol_sha256": e81.digest(protocol),
            "launcher_sha256": e81.digest(Path(__file__)),
            "pair_ledger": str(root / PAIR_LEDGER),
            "e80r1_snapshot_root": str(parent["snapshot_root"]),
            "snapshot_root": str(snapshot),
            "snapshot_patched_files": patched,
            "model": "Qwen2.5-3B-Instruct",
            "domains": list(DOMAINS), "arms": [ARM], "seeds": [SEED],
            "inherited_arms": ["control", "replay"],
            "train_rows": TRAIN_ROWS, "passes": PASSES,
            "target_steps": TARGET_STEPS,
            "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
            "registered_passes": [i / 2 for i in range(17)],
            "replay_weight": REPLAY_WEIGHT,
            "semantic_coefficient": SEMANTIC_COEF,
            "scheduling_nice": NICE,
            "objective": "uniform_verified_likelihood_plus_fixed_open_set_semantic_maxent",
            "scientific_difference": "against E80-R1 replay: the applied semantic advantage only",
            "runs": records, "released": False,
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

    print(f"[e87] cells={len(records)} released={len(submitted)} nice={NICE} "
          f"snapshot={snapshot} ledger={ledger_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
