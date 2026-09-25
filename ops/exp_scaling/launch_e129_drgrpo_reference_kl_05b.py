#!/usr/bin/env python3
"""Submit E129: compute-matched Dr.GRPO plus a reference KL, at three coefficients.

App. Q.6 proves two things about a reference-KL penalty in the categorical mean
flow, and this cohort is what turns them into a measurement rather than an
argument. Proposition "The retained conditional is the reference's" says the
conditional allocation over correct modes converges to the reference's own,
*independently of beta*; Corollary "Recovery from depth s" says the restoring
force on a depleted mode vanishes with the mode. Since oat resolves an empty
``ref_pretrain`` to ``pretrain``, the reference here is exactly the frozen base
model whose pass-0 concentration Section 5.1 already measures, so the
prediction is sharp: \\pmd{} under these arms should track the frozen model's
rather than rising the way verified replay does.

Three coefficients are run because beta-independence of the conditional limit
is the distinctive prediction, and a single coefficient invites the reply that
a different one would have worked.

Every cell is the E78 *control* configuration with one knob moved. The replay
machinery stays wired with an exact-zero derivative
(``ONLINE_CANONICAL_REPLAY=1`` with ``COMPUTE_ONLY=1``), exactly as in E78's
compute-matched Dr.GRPO arm, so that E129 minus E78-control isolates the
reference KL and nothing else, and E129 against E78-replay is the head-to-head
the appendix argues about. Cells inherit ``source_node`` from the same frozen
manifest E78 read, so each arm sits on the GPU model its matched control ran
on; pass-0 values cluster by GPU pool and a cross-pool comparison would
confound that with the intervention.

``--dry-run`` prints the exact sbatch lines and submits nothing. ``--submit``
submits every cell held, audits each held record for the coefficient it is
supposed to carry, writes the ledger, and only then releases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e72_b3a_replay_ablation as base  # noqa: E402
import launch_e76_tuned_scale as snapshot_util  # noqa: E402


DOMAINS = (
    "graph_coloring",
    "countdown",
    "python_factors",
    "mathir",
    "pantry_plan",
)
SEEDS = (43, 44, 45, 46, 47)
# Arm tag -> reference-KL coefficient. The tags are what run directories and
# job names carry, so they must stay filesystem-safe and unambiguous.
KL_ARMS = {
    "kl0p001": 0.001,
    "kl0p01": 0.01,
    "kl0p04": 0.04,
}
ARMS = tuple(KL_ARMS)
PASSES = 8
TRAIN_ROWS = 384
CHECKPOINT_INTERVAL = 192
TARGET_STEPS = TRAIN_ROWS * PASSES
LEDGER = "var/artifacts/e129_drgrpo_reference_kl_05b_jobs.json"
PROTOCOL = "paper/preregistration/e129_drgrpo_reference_kl_05b_20260918.md"
SOURCE_MANIFEST = "var/artifacts/e72_frontier_source_runs.json"
# The matched zero-KL control this cohort is differenced against; it is not
# re-run here.
CONTROL_LEDGER = "var/artifacts/e78_verified_replay_only_05b_jobs.json"
MODEL_TAG = "qwen25_0p5b_instruct"
# The same variant E78's compute-matched Dr.GRPO control runs under.
VARIANT = "grpo_compute_matched"

DOMAIN_TAGS = {
    "graph_coloring": "graph",
    "countdown": "countdown",
    "python_factors": "python",
    "mathir": "mathir",
    "pantry_plan": "pantry",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as sink:
            json.dump(payload, sink, indent=2, sort_keys=True)
            sink.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def references(root: Path) -> list[dict[str, Any]]:
    """The 25 frozen templates, checked to be the zero-KL ones E78 also read."""
    payload = json.loads((root / SOURCE_MANIFEST).read_text(encoding="utf-8"))
    runs = [
        run
        for run in payload["runs"]
        if run["arm"] == "xgrpo"
        and str(run["domain"]) in DOMAINS
        and int(run["seed"]) in SEEDS
    ]
    pairs = {(str(run["domain"]), int(run["seed"])) for run in runs}
    expected = {(domain, seed) for domain in DOMAINS for seed in SEEDS}
    if pairs != expected or len(runs) != len(expected):
        raise SystemExit(
            "E129 source manifest does not contain exactly the 25 domain/seed templates"
        )
    for run in runs:
        evaluation = run["inherited_eval_config"]
        training = run["inherited_train_config"]
        if int(evaluation["max_train"]) != TRAIN_ROWS:
            raise SystemExit(f"{run['domain']}/s{run['seed']}: expected 384 train rows")
        if int(evaluation["num_samples"]) != 16:
            raise SystemExit(f"{run['domain']}/s{run['seed']}: expected group size 16")
        if float(training["learning_rate"]) != 2e-7:
            raise SystemExit(f"{run['domain']}/s{run['seed']}: learning-rate drift")
        # The template must itself be reference-free; the coefficient this
        # cohort studies is applied here, not inherited.
        if int(training["num_ppo_epochs"]) != 1 or float(training["beta"]) != 0:
            raise SystemExit(f"{run['domain']}/s{run['seed']}: optimizer drift")
        for key in ("pretrain", "prompt_data", "eval_data"):
            if not Path(evaluation[key]).exists():
                raise SystemExit(
                    f"{run['domain']}/s{run['seed']}: missing {key}={evaluation[key]}"
                )
    return sorted(runs, key=lambda run: (str(run["domain"]), int(run["seed"])))


def control_placement(root: Path) -> dict[tuple[str, int], str]:
    """Where E78's matched Dr.GRPO control ran, keyed by domain and seed.

    Read to fail closed if this cohort would land a cell on a different GPU
    model from the control it is differenced against.
    """
    payload = json.loads((root / CONTROL_LEDGER).read_text(encoding="utf-8"))
    return {
        (str(run["domain"]), int(run["seed"])): str(run["source_node"])
        for run in payload["runs"]
        if run["arm"] == "control"
    }


def run_stamp(domain: str, arm: str, seed: int) -> str:
    return f"e129_reference_kl_{DOMAIN_TAGS[domain]}_{arm}_s{seed}"


def save_path(root: Path, domain: str, arm: str, seed: int) -> Path:
    return (
        root
        / "var/data"
        / f"xdr_{MODEL_TAG}_{VARIANT}_{run_stamp(domain, arm, seed)}"
    )


def fixed_objective(arm: str) -> dict[str, str]:
    """E78's control objective, with the reference-KL coefficient turned on."""
    return {
        "OAT_ZERO_VARIANT": VARIANT,
        "OAT_ZERO_SEMANTIC_SHANNON_COEF": "0.0",
        "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE": "0",
        "OAT_ZERO_SEMANTIC_SHANNON_QUALITY_GATED_ADVANTAGE": "0",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE": "0",
        "OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA": "0.0",
        # Kept wired with an exact-zero derivative, as in E78's control, so the
        # only live difference from that arm is the coefficient below.
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY": "1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE": (
            "verified_likelihood_per_rollout"
        ),
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA": "0.1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA": "0.1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY": "16",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP": "1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_BOOTSTRAP_STEPS": "0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY": "1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS": "0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT": "0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SINGLETON_ONLY": "0",
        "OAT_ZERO_MAXENT_ALPHA": "0.0",
        "OAT_ZERO_MAXENT_CONTROL_TARGET_RATIO": "0.0",
        "OAT_ZERO_MAXENT_DUAL_TARGET_RATIO": "0.0",
        "OAT_ZERO_MAXENT_INVERSE_ADAPTATION": "0",
        "OAT_ZERO_POLICY_ENTROPY_COEF": "0.0",
        "OAT_ZERO_SEED_ENTROPY_ALPHA": "0.0",
        # The one live knob. oat loads the reference policy when this is
        # positive and resolves ref_pretrain to pretrain, i.e. the frozen base.
        "OAT_ZERO_BETA": repr(KL_ARMS[arm]),
    }


def build_env(
    root: Path,
    run: dict[str, Any],
    arm: str,
    snapshot_root: Path,
) -> tuple[dict[str, str], Path]:
    domain = str(run["domain"])
    seed = int(run["seed"])
    target = save_path(root, domain, arm, seed)
    env = base.build_export_vars(root, run, target, "b1b")
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": run_stamp(domain, arm, seed),
            "OAT_ZERO_SOURCE_ROOT": str(snapshot_root / "src"),
            "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot_root / "ops"),
            "OAT_ZERO_NUM_PROMPT_EPOCH": str(PASSES),
            "OAT_ZERO_MAX_PROMPT_EPOCHS": str(PASSES),
            "OAT_ZERO_EVAL_PROMPT_INTERVAL": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_SAVE_STEPS": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_SAVE_FROM": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_RESUME_STEPS": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_MAX_SAVE_NUM": "1",
            "OAT_ZERO_MAX_RESUME_NUM": "1",
            "OAT_ZERO_AUTO_RESUME": "1",
            "OAT_ZERO_WATCHDOG_REQUEUE": "1",
            "OAT_ZERO_USE_WB": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    env.update(fixed_objective(arm))
    return env, target


def placement(node: str) -> dict[str, str]:
    if node == "node302":
        return {"partition": "mltheory", "account": "mltheory", "gres": "gpu:a100:1"}
    if node == "node105":
        return {"partition": "mltheory", "account": "mltheory", "gres": "gpu:a5000:1"}
    raise SystemExit(f"E129 has no frozen placement for source node {node!r}")


def job_name(domain: str, arm: str, seed: int) -> str:
    return f"e129-{DOMAIN_TAGS[domain][:6]}-{arm}-s{seed}"


def sbatch_command(
    root: Path,
    run: dict[str, Any],
    arm: str,
    env: dict[str, str],
) -> list[str]:
    node = str(run["source_node"])
    place = placement(node)
    export_pairs = ",".join(f"{key}={value}" for key, value in env.items())
    return [
        "sbatch",
        "--parsable",
        "--hold",
        f"--job-name={job_name(str(run['domain']), arm, int(run['seed']))}",
        f"--export=ALL,{export_pairs}",
        f"--partition={place['partition']}",
        f"--account={place['account']}",
        f"--nodelist={node}",
        f"--gres={place['gres']}",
        "--cpus-per-task=8",
        # The reference policy is resident alongside the trained one; E78 ran
        # the same cells in 64G without it, and 0.5B in bf16 adds about 1 GB.
        "--mem=80G",
        "--time=1-12:00:00",
        "--nice=100",
        str(root / "ops/slurm/train_node302.slurm"),
    ]


def held_job_audit(job_id: str, run: dict[str, Any], arm: str) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot inspect held job {job_id}: {result.stderr.strip()}")
    record = result.stdout
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"JobName={job_name(str(run['domain']), arm, int(run['seed']))}",
        f"ReqNodeList={run['source_node']}",
        f"OAT_ZERO_VARIANT={VARIANT}",
        f"OAT_ZERO_SEED={run['seed']}",
        "OAT_ZERO_NUM_PROMPT_EPOCH=8",
        "OAT_ZERO_MAX_PROMPT_EPOCHS=8",
        "OAT_ZERO_EVAL_PROMPT_INTERVAL=192",
        "OAT_ZERO_SAVE_STEPS=192",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.0",
        "OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=1",
        "OAT_ZERO_MAXENT_ALPHA=0.0",
        "OAT_ZERO_POLICY_ENTROPY_COEF=0.0",
        # The cell must carry its own arm's coefficient and no other.
        f"OAT_ZERO_BETA={KL_ARMS[arm]!r}",
    )
    missing = [text for text in required if text not in record]
    if missing:
        raise RuntimeError(f"held job {job_id} lacks {missing}")
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

    protocol = root / PROTOCOL
    source_manifest = root / SOURCE_MANIFEST
    for required in (protocol, source_manifest, root / CONTROL_LEDGER):
        if not required.is_file():
            raise SystemExit(f"required frozen input is absent: {required}")
    ledger = root / LEDGER
    if args.submit and ledger.exists():
        raise SystemExit(f"refusing duplicate E129 submission: {ledger}")

    source_runs = references(root)
    controls = control_placement(root)
    snapshot_root = snapshot_util.ensure_snapshot(root, args.snapshot_root)

    planned: list[dict[str, Any]] = []
    for run in source_runs:
        domain, seed = str(run["domain"]), int(run["seed"])
        node = str(run["source_node"])
        control_node = controls.get((domain, seed))
        if control_node != node:
            raise SystemExit(
                f"{domain}/s{seed}: control ran on {control_node!r} but this cell "
                f"would run on {node!r}; a cross-pool difference is not admissible"
            )
        for arm in ARMS:
            env, target = build_env(root, run, arm, snapshot_root)
            if target.exists():
                raise SystemExit(f"refusing to overwrite existing E129 run: {target}")
            planned.append(
                {
                    "domain": domain,
                    "arm": arm,
                    "beta": KL_ARMS[arm],
                    "seed": seed,
                    "source_node": node,
                    "run_stamp": run_stamp(domain, arm, seed),
                    "run_dir": str(target),
                    "command": sbatch_command(root, run, arm, env),
                    "template": run,
                }
            )

    expected_cells = len(DOMAINS) * len(SEEDS) * len(ARMS)
    if len(planned) != expected_cells:
        raise SystemExit(f"E129 expected {expected_cells} cells, found {len(planned)}")
    stamps = [cell["run_stamp"] for cell in planned]
    if len(set(stamps)) != len(stamps):
        raise SystemExit("E129 run stamps are not unique")

    if args.dry_run or not args.submit:
        for cell in planned:
            print(" ".join(shlex.quote(part) for part in cell["command"]))
        by_node: dict[str, int] = {}
        for cell in planned:
            by_node[cell["source_node"]] = by_node.get(cell["source_node"], 0) + 1
        print(
            f"[e129] dry_run=True cells={len(planned)} arms={dict(KL_ARMS)} "
            f"placement={by_node} snapshot={snapshot_root}"
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
            held_record = held_job_audit(job_id, cell["template"], str(cell["arm"]))
            records.append(
                {
                    key: cell[key]
                    for key in (
                        "domain",
                        "arm",
                        "beta",
                        "seed",
                        "source_node",
                        "run_stamp",
                        "run_dir",
                    )
                }
                | {"job_id": int(job_id), "held_scheduler_record": held_record}
            )

        payload = {
            "schema": "e129_drgrpo_reference_kl_05b_jobs_v1",
            "protocol": str(protocol),
            "protocol_sha256": digest(protocol),
            "launcher_sha256": digest(Path(__file__)),
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": digest(source_manifest),
            "control_ledger": str(root / CONTROL_LEDGER),
            "control_ledger_sha256": digest(root / CONTROL_LEDGER),
            "snapshot_root": str(snapshot_root),
            "model": "Qwen2.5-0.5B-Instruct",
            "domains": list(DOMAINS),
            "arms": list(ARMS),
            "arm_beta": dict(KL_ARMS),
            "seeds": list(SEEDS),
            "train_rows": TRAIN_ROWS,
            "passes": PASSES,
            "target_steps": TARGET_STEPS,
            "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
            "registered_passes": [index / 2 for index in range(17)],
            "reference_policy": "pretrain (oat resolves empty ref_pretrain)",
            "kl_estimator": "k3, loss side, gated on args.beta",
            "objective": "compute_matched_drgrpo_plus_reference_kl",
            "scientific_difference": (
                "reference-KL coefficient: exact zero (E78 control) versus "
                "0.001, 0.01 and 0.04"
            ),
            "runs": records,
            "released": False,
        }
        atomic_json(ledger, payload)
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
        atomic_json(ledger, payload)
    except Exception:
        cancel(submitted)
        if ledger.exists():
            ledger.unlink()
        raise

    print(
        f"[e129] cells={len(records)} released={len(submitted)} "
        f"snapshot={snapshot_root} ledger={ledger}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
