#!/usr/bin/env python3
"""Submit E86: the Falcon replication of E83, closing the Falcon 2x2.

E86 is to E79/E82 what E83 is to E78/E81: fixed semantic MaxEnt with the replay
derivative switched off. Cell definitions come from the E79 launcher and the
objective from the E83 launcher, so the only thing this file decides is that
those two are combined. The E79 ``control`` and ``replay`` runs and the E82
``semantic`` runs are inherited unchanged and are never re-run.

Unlike E82 and E83, E86 runs from the repaired runtime: its patch set is E85's,
so PantryPlan's semantic term is live from the first update and the cohort needs
no later repair. That branch is gated on ``canonical_actions``, which only
PantryPlan sets, so the other four domains run E82's runtime byte for byte.
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
import launch_e79_falcon1b_aligned_verified_replay as e79  # noqa: E402
import launch_e81_semantic_maxent_verified_replay_05b as e81  # noqa: E402
import launch_e82_falcon_semantic_maxent_verified_replay as e82  # noqa: E402
import launch_e83_semantic_maxent_without_replay_05b as e83  # noqa: E402
import launch_e85_pantry_semantic_repair as e85  # noqa: E402


DOMAINS = e79.DOMAINS
ARM = "semantic_only"
SEEDS = e79.SEEDS
PASSES = e79.PASSES
TRAIN_ROWS = e79.TRAIN_ROWS
CHECKPOINT_INTERVAL = e79.CHECKPOINT_INTERVAL
TARGET_STEPS = e79.TARGET_STEPS
REPLAY_WEIGHT = e79.REPLAY_WEIGHT
SEMANTIC_COEF = e81.SEMANTIC_COEF
VARIANT = e83.VARIANT
MODEL_TAG = e79.MODEL_TAG
MODEL_REVISION = e79.MODEL_REVISION

LEDGER = "var/artifacts/e86_falcon_semantic_maxent_without_replay_jobs.json"
PROTOCOL = (
    "paper/preregistration/"
    "e86_semantic_maxent_without_replay_falcon1b_20260809.md"
)
SOURCE_MANIFEST = e79.SOURCE_MANIFEST
PAIR_LEDGER = e82.PAIR_LEDGER
E82_LEDGER = "var/artifacts/e82_falcon_semantic_maxent_verified_replay_jobs.json"
E85_LEDGER = "var/artifacts/e85_pantry_semantic_repair_jobs.json"
SNAPSHOT_PREFIX = "e86_falcon_semantic_maxent_no_replay"

# E85's patch set, not E82's: the semantic key derivation is repaired here from
# the first update rather than by a later repair cohort. The added branch is
# unreachable unless the task resolves to a canonical action task.
PATCHED_FILES = e85.PATCHED_FILES

# The one domain whose runtime differs from E82's, and whose E82 comparator is
# therefore superseded by the E85 repair.
REPAIRED_DOMAIN = e85.DOMAIN


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_stamp(domain: str, seed: int) -> str:
    return f"e86_falcon_semantic_only_{e79.DOMAIN_TAGS[domain]}_{ARM}_s{seed}"


def save_path(root: Path, domain: str, seed: int) -> Path:
    return root / "var/data" / f"xdr_{MODEL_TAG}_{VARIANT}_{run_stamp(domain, seed)}"


def fixed_objective() -> dict[str, str]:
    """E83's objective verbatim, re-checked against E79's replay dose.

    E83's objective is E81's with two keys moved: the variant string and the
    replay compute-only switch. Taking it whole is what makes E86 the Falcon
    replication of E83 rather than a second, independently specified arm.
    """

    objective = dict(e83.fixed_objective())
    if float(objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA"]) != REPLAY_WEIGHT:
        raise RuntimeError("E86 replay dose drifted from the shared E79 dose")
    if float(objective["OAT_ZERO_SEMANTIC_SHANNON_COEF"]) != SEMANTIC_COEF:
        raise RuntimeError("E86 semantic coefficient drifted from E81")
    if objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY"] != "1":
        raise RuntimeError("E86 must traverse replay for compute and never apply it")
    if objective["OAT_ZERO_VARIANT"] != VARIANT:
        raise RuntimeError("E86 variant drifted from E83")
    return objective


def build_env(
    root: Path,
    run: dict[str, Any],
    snapshot_root: Path,
    falcon_root: Path,
) -> tuple[dict[str, str], Path]:
    """E82's Falcon cell, with E83's objective substituted for E82's."""

    domain = str(run["domain"])
    seed = int(run["seed"])
    env, _ = e82.build_env(root, run, snapshot_root, falcon_root)
    target = save_path(root, domain, seed)
    env.update({"SAVE_PATH": str(target), "RUN_STAMP": run_stamp(domain, seed)})
    env.update(fixed_objective())
    return env, target


def sbatch_command(root: Path, run: dict[str, Any], env: dict[str, str]) -> list[str]:
    domain = str(run["domain"])
    seed = int(run["seed"])
    node, gpu = e79.placement(domain, seed)
    name = f"e86-{e79.DOMAIN_TAGS[domain][:6]}-sonly-s{seed}"
    export_pairs = ",".join(f"{key}={value}" for key, value in env.items())
    return [
        "sbatch",
        "--parsable",
        "--hold",
        f"--job-name={name}",
        f"--export=ALL,{export_pairs}",
        "--partition=cs",
        "--account=allcs",
        f"--nodelist={node}",
        f"--gres=gpu:{gpu}:1",
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=3-00:00:00",
        "--nice=100",
        str(root / "ops/slurm/train_node302.slurm"),
    ]


def held_job_audit(job_id: str, run: dict[str, Any], falcon_root: Path) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"cannot inspect held E86 job {job_id}: {result.stderr.strip()}"
        )
    domain = str(run["domain"])
    seed = int(run["seed"])
    node, gpu = e79.placement(domain, seed)
    surface = e79.task_surface(
        domain, str(run["inherited_eval_config"]["prompt_template"])
    )
    record = result.stdout
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"JobName=e86-{e79.DOMAIN_TAGS[domain][:6]}-sonly-s{seed}",
        f"ReqNodeList={node}",
        f"gres/gpu:{gpu}=1",
        f"OAT_ZERO_PRETRAIN={falcon_root}",
        f"OAT_ZERO_VARIANT={VARIANT}",
        f"OAT_ZERO_SEED={seed}",
        "OAT_ZERO_NUM_PROMPT_EPOCH=8",
        "OAT_ZERO_MAX_PROMPT_EPOCHS=8",
        "OAT_ZERO_EVAL_PROMPT_INTERVAL=192",
        "OAT_ZERO_SAVE_STEPS=192",
        "OAT_ZERO_LEARNING_RATE=2e-07",
        "OAT_ZERO_LR_SCHEDULER=constant",
        "OAT_ZERO_LR_WARMUP_RATIO=0.0",
        "OAT_ZERO_ADAM_BETA_1=0.9",
        "OAT_ZERO_ADAM_BETA_2=0.999",
        "OAT_ZERO_L2=0.0",
        "OAT_ZERO_BETA=0.0",
        f"OAT_ZERO_PROMPT_TEMPLATE={surface['OAT_ZERO_PROMPT_TEMPLATE']}",
        f"OAT_ZERO_GENERATE_MAX_LENGTH={surface['OAT_ZERO_GENERATE_MAX_LENGTH']}",
        f"OAT_ZERO_MAX_MODEL_LEN={surface['OAT_ZERO_MAX_MODEL_LEN']}",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.1",
        "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=1",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE=1",
        "OAT_ZERO_SEMANTIC_SHANNON_QUALITY_GATED_ADVANTAGE=0",
        "OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=5.0",
        "OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=1.0",
        "OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=verified_likelihood_per_rollout",
        # the declared dose is present and, by the next line, never applied
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=0",
        "OAT_ZERO_MAXENT_ALPHA=0.0",
        "OAT_ZERO_POLICY_ENTROPY_COEF=0.0",
        "OAT_ZERO_SEED_ENTROPY_ALPHA=0.0",
    )
    missing = [text for text in required if text not in record]
    if missing:
        raise RuntimeError(f"held E86 job {job_id} lacks {missing}")
    return record


def factorial_index(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Map (domain, seed) to the three inherited arms of the Falcon 2x2.

    PantryPlan's ``semantic`` comparator is the E85 repair, not E82's original
    cell: E86's PantryPlan runs a live semantic term and may only be differenced
    against a comparator that does too.
    """

    index = e82.pair_index(e82.pair_ledger(root))

    e82_path = root / E82_LEDGER
    if e82_path.is_file():
        payload = json.loads(e82_path.read_text(encoding="utf-8"))
        for run in payload["runs"]:
            key = (str(run["domain"]), int(run["seed"]))
            if key in index:
                index[key]["semantic"] = run

    e85_path = root / E85_LEDGER
    if e85_path.is_file():
        payload = json.loads(e85_path.read_text(encoding="utf-8"))
        for run in payload["runs"]:
            if str(run.get("parent")) != "e82":
                continue
            key = (str(run["domain"]), int(run["seed"]))
            if key in index:
                index[key]["semantic"] = run
                index[key]["semantic_supersedes"] = run["supersedes"]

    for seed in SEEDS:
        arms = index[(REPAIRED_DOMAIN, seed)]
        if "semantic_supersedes" not in arms:
            raise SystemExit(
                f"{REPAIRED_DOMAIN}/s{seed}: E86 requires the E85 repair as its "
                "semantic comparator; E82's original cell is superseded"
            )
    return index


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
            raise SystemExit(f"required frozen E86 input is absent: {required}")
    ledger = root / LEDGER
    if args.submit and ledger.exists():
        raise SystemExit(f"refusing duplicate E86 submission: {ledger}")

    pairs = factorial_index(root)
    falcon_root = e79.model_root(root)
    source_runs = e79.references(root)
    e79_snapshot = (
        args.snapshot_root.resolve()
        if args.snapshot_root
        else Path(e82.pair_ledger(root)["snapshot_root"])
    )
    snapshot_root, patched = e81.ensure_paired_snapshot(
        root, e79_snapshot, prefix=SNAPSHOT_PREFIX, patched_files=PATCHED_FILES
    )
    if patched != sorted(PATCHED_FILES):
        raise SystemExit("E86 snapshot diverges from E79's outside the patch set")

    planned: list[dict[str, Any]] = []
    for run in source_runs:
        domain = str(run["domain"])
        seed = int(run["seed"])
        node, gpu = e79.placement(domain, seed)
        pair = pairs[(domain, seed)]
        pair_node = str(pair["replay"]["node"])
        pair_gpu = str(pair["replay"]["gpu"])
        if (pair_node, pair_gpu) != (node, gpu):
            raise SystemExit(
                f"{domain}/s{seed}: E79 pair ran on {pair_node}/{pair_gpu}, "
                f"E86 would use {node}/{gpu}"
            )
        env, target = build_env(root, run, snapshot_root, falcon_root)
        if target.exists():
            raise SystemExit(f"refusing to overwrite existing E86 run: {target}")
        planned.append(
            {
                "domain": domain,
                "arm": ARM,
                "seed": seed,
                "node": node,
                "gpu": gpu,
                "run_stamp": run_stamp(domain, seed),
                "run_dir": str(target),
                "paired_runs": {
                    arm: {
                        "run_stamp": str(record["run_stamp"]),
                        "run_dir": str(record["run_dir"]),
                        "job_id": int(record["job_id"]),
                    }
                    for arm, record in pair.items()
                    if arm != "semantic_supersedes"
                },
                "semantic_comparator_supersedes": pair.get("semantic_supersedes"),
                "command": sbatch_command(root, run, env),
                "template": run,
            }
        )

    if len(planned) != 25:
        raise SystemExit(f"E86 expected 25 cells, found {len(planned)}")
    if args.dry_run or not args.submit:
        for cell in planned:
            print(" ".join(shlex.quote(part) for part in cell["command"]))
        print(
            f"[e86] dry_run=True cells={len(planned)} snapshot={snapshot_root} "
            f"patched={patched} model={falcon_root}"
        )
        return 0

    submitted: list[str] = []
    records: list[dict[str, Any]] = []
    try:
        for cell in planned:
            result = subprocess.run(
                cell["command"], capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"submission failed for {cell['run_stamp']}: {result.stderr.strip()}"
                )
            job_id = result.stdout.strip().split(";", 1)[0]
            if not job_id.isdigit():
                raise RuntimeError(
                    f"invalid job id for {cell['run_stamp']}: {result.stdout!r}"
                )
            submitted.append(job_id)
            held_record = held_job_audit(job_id, cell["template"], falcon_root)
            records.append(
                {
                    key: cell[key]
                    for key in (
                        "domain",
                        "arm",
                        "seed",
                        "node",
                        "gpu",
                        "run_stamp",
                        "run_dir",
                        "paired_runs",
                        "semantic_comparator_supersedes",
                    )
                }
                | {"job_id": int(job_id), "held_scheduler_record": held_record}
            )

        payload = {
            "schema": "e86_falcon_semantic_maxent_without_replay_jobs_v1",
            "protocol": str(protocol),
            "protocol_sha256": e81.digest(protocol),
            "launcher_sha256": e81.digest(Path(__file__)),
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": e81.digest(source_manifest),
            "pair_ledger": str(root / PAIR_LEDGER),
            "pair_ledger_sha256": e81.digest(root / PAIR_LEDGER),
            "e79_snapshot_root": str(e79_snapshot),
            "snapshot_root": str(snapshot_root),
            "snapshot_patched_files": patched,
            "model": "tiiuae/Falcon3-1B-Instruct",
            "model_revision": MODEL_REVISION,
            "model_root": str(falcon_root),
            "domains": list(DOMAINS),
            "arms": [ARM],
            "inherited_arms": ["control", "replay", "semantic"],
            "seeds": list(SEEDS),
            "train_rows": TRAIN_ROWS,
            "passes": PASSES,
            "target_steps": TARGET_STEPS,
            "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
            "registered_passes": [index / 2 for index in range(17)],
            "replay_weight": REPLAY_WEIGHT,
            "replay_derivative_applied": False,
            "semantic_coefficient": SEMANTIC_COEF,
            "semantic_surprisal_clip": e81.SEMANTIC_SURPRISAL_CLIP,
            "semantic_pseudocount": e81.SEMANTIC_PSEUDOCOUNT,
            "objective": "fixed_open_set_semantic_maxent_only",
            "born_repaired": {
                "domain": REPAIRED_DOMAIN,
                "patched_file": "src/oat_drgrpo/learner/grpo.py",
                "note": (
                    "canonical-action semantic key is bound from the first "
                    "update, so this cohort needs no repair sibling; the "
                    "PantryPlan comparator is the E85 repair of E82"
                ),
            },
            "scientific_difference": (
                "against E79 control: the applied semantic advantage only; "
                "against E82: the applied replay derivative only"
            ),
            "completes_factorial": "e79_control x e79_replay x e82_semantic",
            "replicates": "e83_semantic_maxent_without_replay_05b_jobs_v1",
            "runs": records,
            "released": False,
        }
        e81.atomic_json(ledger, payload)
        for job_id in submitted:
            release = subprocess.run(
                ["scontrol", "release", job_id],
                capture_output=True,
                text=True,
                check=False,
            )
            if release.returncode != 0:
                raise RuntimeError(
                    f"release failed for {job_id}: {release.stderr.strip()}"
                )
        payload["released"] = True
        e81.atomic_json(ledger, payload)
    except Exception:
        e81.cancel(submitted)
        if ledger.exists():
            ledger.unlink()
        raise

    print(
        f"[e86] cells={len(records)} released={len(submitted)} "
        f"snapshot={snapshot_root} ledger={ledger}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
