#!/usr/bin/env python3
"""Submit E81: the E78 verified-replay arm plus fixed semantic MaxEnt.

E81 adds one arm to a completed design. The E78 ``control`` and ``replay``
runs are inherited unchanged and are never re-run; this launcher only
materializes the third arm and pins it, cell by cell, to the same node, data,
seed, schedule, and training code as its E78 pair.
"""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e72_b3a_replay_ablation as base  # noqa: E402


DOMAINS = (
    "graph_coloring",
    "countdown",
    "python_factors",
    "mathir",
    "pantry_plan",
)
ARM = "semantic"
SEEDS = (43, 44, 45, 46, 47)
PASSES = 8
TRAIN_ROWS = 384
CHECKPOINT_INTERVAL = 192
TARGET_STEPS = TRAIN_ROWS * PASSES
REPLAY_WEIGHT = 0.10
SEMANTIC_COEF = 0.10
SEMANTIC_SURPRISAL_CLIP = 5.0
SEMANTIC_PSEUDOCOUNT = 1.0
VARIANT = "verified_replay_semantic_maxent"

LEDGER = "var/artifacts/e81_semantic_maxent_verified_replay_05b_jobs.json"
PROTOCOL = (
    "paper/preregistration/e81_semantic_maxent_on_verified_replay_05b_20260806.md"
)
SOURCE_MANIFEST = "var/artifacts/e72_frontier_source_runs.json"
PAIR_LEDGER = "var/artifacts/e78_verified_replay_only_05b_jobs.json"
MODEL_TAG = "qwen25_0p5b_instruct"

# The only two files E81 is permitted to change relative to the E78 runtime.
# Both are additive: the widened validation predicate is reachable only when
# the semantic coefficient is positive, which no E78 arm sets, and a new
# variant `case` branch is unreachable for any other variant string.
PATCHED_FILES = (
    "src/oat_drgrpo/args.py",
    "ops/run_experiment.sh",
)

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


def tree_paths(root: Path) -> set[str]:
    return {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }


def divergence(left: Path, right: Path) -> list[str]:
    """Return every relative path whose content or presence differs."""

    left_paths = tree_paths(left)
    right_paths = tree_paths(right)
    changed = sorted(left_paths ^ right_paths)
    for relative in sorted(left_paths & right_paths):
        if not filecmp.cmp(left / relative, right / relative, shallow=False):
            changed.append(relative)
    return sorted(set(changed))


def pair_ledger(root: Path) -> dict[str, Any]:
    path = root / PAIR_LEDGER
    if not path.is_file():
        raise SystemExit(f"E78 pair ledger is absent: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("released"):
        raise SystemExit("E78 pair ledger is not released; refusing to pair")
    return payload


def pair_index(payload: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    """Map (domain, seed) to the E78 pair, asserting both arms are present."""

    index: dict[tuple[str, int], dict[str, Any]] = {}
    for run in payload["runs"]:
        key = (str(run["domain"]), int(run["seed"]))
        index.setdefault(key, {})[str(run["arm"])] = run
    expected = {(domain, seed) for domain in DOMAINS for seed in SEEDS}
    if set(index) != expected:
        raise SystemExit("E78 ledger does not cover exactly the E81 cohort")
    for key, arms in index.items():
        if set(arms) != {"control", "replay"}:
            raise SystemExit(f"E78 pair {key} is missing an arm: {sorted(arms)}")
        nodes = {str(run["source_node"]) for run in arms.values()}
        if len(nodes) != 1:
            raise SystemExit(f"E78 pair {key} straddles nodes: {sorted(nodes)}")
    return index


def references(root: Path) -> list[dict[str, Any]]:
    """Return the same 25 domain/seed templates E78 trained from."""

    path = root / SOURCE_MANIFEST
    payload = json.loads(path.read_text(encoding="utf-8"))
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
            "E81 source manifest does not contain exactly the 25 domain/seed templates"
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
        if int(training["num_ppo_epochs"]) != 1 or float(training["beta"]) != 0:
            raise SystemExit(f"{run['domain']}/s{run['seed']}: optimizer drift")
        for key in ("pretrain", "prompt_data", "eval_data"):
            if not Path(evaluation[key]).exists():
                raise SystemExit(
                    f"{run['domain']}/s{run['seed']}: missing {key}={evaluation[key]}"
                )
    return sorted(runs, key=lambda run: (str(run["domain"]), int(run["seed"])))


def ensure_paired_snapshot(
    root: Path,
    base_snapshot: Path,
    *,
    prefix: str = "e81_semantic_maxent",
    patched_files: tuple[str, ...] | None = None,
) -> tuple[Path, list[str]]:
    """Materialize a comparator's runtime with exactly the patch set replaced.

    ``prefix`` names the derived snapshot. E82 reuses this function against the
    E79 runtime so both semantic arms are built, and audited, the same way.
    ``patched_files`` lets a repair cohort declare a wider patch set without
    changing the identity any already-submitted cohort recorded.
    """

    patched = tuple(patched_files) if patched_files else PATCHED_FILES

    if not (base_snapshot / "src/oat_drgrpo/__init__.py").is_file():
        raise SystemExit(f"invalid comparator runtime snapshot: {base_snapshot}")
    for relative in patched:
        if not (root / relative).is_file():
            raise SystemExit(f"patched source is absent: {relative}")
        # A patch set may legitimately introduce a file the comparator runtime
        # never had; the divergence audit below still accounts for it, because
        # a path present in only one tree counts as a difference.
        (base_snapshot / relative).parent.mkdir(parents=True, exist_ok=True)

    identity = hashlib.sha256()
    identity.update(base_snapshot.name.encode("utf-8"))
    for relative in patched:
        identity.update(relative.encode("utf-8"))
        identity.update(digest(root / relative).encode("utf-8"))
    tag = identity.hexdigest()[:16]
    snapshot = root / "var/artifacts/source_snapshots" / f"{prefix}_{tag}"

    if not snapshot.is_dir():
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{snapshot.name}.", dir=snapshot.parent)
        )
        try:
            for entry in ("src", "ops"):
                shutil.copytree(
                    base_snapshot / entry,
                    temporary / entry,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
            for relative in patched:
                shutil.copyfile(root / relative, temporary / relative)
                shutil.copystat(root / relative, temporary / relative)
            (temporary / "SNAPSHOT_IDENTITY.json").write_text(
                json.dumps(
                    {
                        "schema": "paired_semantic_runtime_snapshot_v1",
                        "derived_from": str(base_snapshot),
                        "patched_files": list(patched),
                        "sha256": identity.hexdigest(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            os.replace(temporary, snapshot)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    changed = [
        path
        for path in divergence(base_snapshot, snapshot)
        if path != "SNAPSHOT_IDENTITY.json"
    ]
    if changed != sorted(patched):
        raise SystemExit(
            "derived snapshot diverges from the comparator runtime outside "
            "the declared "
            f"patch set: {changed}"
        )
    return snapshot, changed


def run_stamp(domain: str, seed: int) -> str:
    return f"e81_semantic_maxent_{DOMAIN_TAGS[domain]}_{ARM}_s{seed}"


def save_path(root: Path, domain: str, seed: int) -> Path:
    return root / "var/data" / f"xdr_{MODEL_TAG}_{VARIANT}_{run_stamp(domain, seed)}"


def fixed_objective() -> dict[str, str]:
    """The E78 replay objective with the semantic term as the only addition."""

    return {
        "OAT_ZERO_VARIANT": VARIANT,
        # --- the treatment ---------------------------------------------------
        "OAT_ZERO_SEMANTIC_SHANNON_COEF": repr(SEMANTIC_COEF),
        "OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP": repr(SEMANTIC_SURPRISAL_CLIP),
        "OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT": repr(SEMANTIC_PSEUDOCOUNT),
        "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE": "1",
        "OAT_ZERO_SEMANTIC_SHANNON_QUALITY_GATED_ADVANTAGE": "0",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE": "1",
        # --- everything below is byte-identical to E78's replay arm ----------
        "OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA": "0.0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY": "1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE": (
            "verified_likelihood_per_rollout"
        ),
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA": repr(REPLAY_WEIGHT),
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA": repr(REPLAY_WEIGHT),
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY": "16",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP": "1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_BOOTSTRAP_STEPS": "0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY": "0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS": "0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT": "0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SINGLETON_ONLY": "0",
        "OAT_ZERO_MAXENT_ALPHA": "0.0",
        "OAT_ZERO_MAXENT_CONTROL_TARGET_RATIO": "0.0",
        "OAT_ZERO_MAXENT_DUAL_TARGET_RATIO": "0.0",
        "OAT_ZERO_MAXENT_INVERSE_ADAPTATION": "0",
        "OAT_ZERO_POLICY_ENTROPY_COEF": "0.0",
        "OAT_ZERO_SEED_ENTROPY_ALPHA": "0.0",
        "OAT_ZERO_BETA": "0.0",
    }


def build_env(
    root: Path,
    run: dict[str, Any],
    snapshot_root: Path,
) -> tuple[dict[str, str], Path]:
    domain = str(run["domain"])
    seed = int(run["seed"])
    target = save_path(root, domain, seed)
    env = base.build_export_vars(root, run, target, "b1b")
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": run_stamp(domain, seed),
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
    env.update(fixed_objective())
    return env, target


def placement(node: str) -> dict[str, str]:
    if node == "node302":
        return {
            "partition": "mltheory",
            "account": "mltheory",
            "gres": "gpu:a100:1",
        }
    if node == "node105":
        return {
            "partition": "mltheory",
            "account": "mltheory",
            "gres": "gpu:a5000:1",
        }
    raise SystemExit(f"E81 has no frozen placement for source node {node!r}")


def sbatch_command(
    root: Path,
    run: dict[str, Any],
    env: dict[str, str],
) -> list[str]:
    node = str(run["source_node"])
    place = placement(node)
    name = f"e81-{DOMAIN_TAGS[str(run['domain'])][:6]}-sem-s{run['seed']}"
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
        raise RuntimeError(f"cannot inspect held job {job_id}: {result.stderr.strip()}")
    record = result.stdout
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"JobName=e81-{DOMAIN_TAGS[str(run['domain'])][:6]}-sem-s{run['seed']}",
        f"ReqNodeList={run['source_node']}",
        f"OAT_ZERO_VARIANT={VARIANT}",
        f"OAT_ZERO_SEED={run['seed']}",
        "OAT_ZERO_NUM_PROMPT_EPOCH=8",
        "OAT_ZERO_MAX_PROMPT_EPOCHS=8",
        "OAT_ZERO_EVAL_PROMPT_INTERVAL=192",
        "OAT_ZERO_SAVE_STEPS=192",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.1",
        "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=1",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE=1",
        "OAT_ZERO_SEMANTIC_SHANNON_QUALITY_GATED_ADVANTAGE=0",
        "OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=5.0",
        "OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=1.0",
        "OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=verified_likelihood_per_rollout",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=0",
        "OAT_ZERO_MAXENT_ALPHA=0.0",
        "OAT_ZERO_POLICY_ENTROPY_COEF=0.0",
        "OAT_ZERO_SEED_ENTROPY_ALPHA=0.0",
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
    for required in (protocol, source_manifest):
        if not required.is_file():
            raise SystemExit(f"required frozen input is absent: {required}")
    ledger = root / LEDGER
    if args.submit and ledger.exists():
        raise SystemExit(f"refusing duplicate E81 submission: {ledger}")

    pairs = pair_index(pair_ledger(root))
    source_runs = references(root)
    e78_snapshot = (
        args.snapshot_root.resolve()
        if args.snapshot_root
        else Path(pair_ledger(root)["snapshot_root"])
    )
    snapshot_root, patched = ensure_paired_snapshot(root, e78_snapshot)

    planned: list[dict[str, Any]] = []
    for run in source_runs:
        domain = str(run["domain"])
        seed = int(run["seed"])
        pair = pairs[(domain, seed)]
        pair_node = str(pair["replay"]["source_node"])
        if pair_node != str(run["source_node"]):
            raise SystemExit(
                f"{domain}/s{seed}: E78 pair ran on {pair_node}, "
                f"manifest pins {run['source_node']}"
            )
        env, target = build_env(root, run, snapshot_root)
        if target.exists():
            raise SystemExit(f"refusing to overwrite existing E81 run: {target}")
        planned.append(
            {
                "domain": domain,
                "arm": ARM,
                "seed": seed,
                "source_node": str(run["source_node"]),
                "run_stamp": run_stamp(domain, seed),
                "run_dir": str(target),
                "paired_e78_runs": {
                    arm: {
                        "run_stamp": str(record["run_stamp"]),
                        "run_dir": str(record["run_dir"]),
                        "job_id": int(record["job_id"]),
                    }
                    for arm, record in pair.items()
                },
                "command": sbatch_command(root, run, env),
                "template": run,
            }
        )

    if len(planned) != 25:
        raise SystemExit(f"E81 expected 25 cells, found {len(planned)}")
    if args.dry_run or not args.submit:
        for cell in planned:
            print(" ".join(shlex.quote(part) for part in cell["command"]))
        print(
            f"[e81] dry_run=True cells={len(planned)} snapshot={snapshot_root} "
            f"patched={patched}"
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
            held_record = held_job_audit(job_id, cell["template"])
            records.append(
                {
                    key: cell[key]
                    for key in (
                        "domain",
                        "arm",
                        "seed",
                        "source_node",
                        "run_stamp",
                        "run_dir",
                        "paired_e78_runs",
                    )
                }
                | {"job_id": int(job_id), "held_scheduler_record": held_record}
            )

        payload = {
            "schema": "e81_semantic_maxent_verified_replay_05b_jobs_v1",
            "protocol": str(protocol),
            "protocol_sha256": digest(protocol),
            "launcher_sha256": digest(Path(__file__)),
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": digest(source_manifest),
            "pair_ledger": str(root / PAIR_LEDGER),
            "pair_ledger_sha256": digest(root / PAIR_LEDGER),
            "e78_snapshot_root": str(e78_snapshot),
            "snapshot_root": str(snapshot_root),
            "snapshot_patched_files": patched,
            "model": "Qwen2.5-0.5B-Instruct",
            "domains": list(DOMAINS),
            "arms": [ARM],
            "inherited_arms": ["control", "replay"],
            "seeds": list(SEEDS),
            "train_rows": TRAIN_ROWS,
            "passes": PASSES,
            "target_steps": TARGET_STEPS,
            "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
            "registered_passes": [index / 2 for index in range(17)],
            "replay_weight": REPLAY_WEIGHT,
            "semantic_coefficient": SEMANTIC_COEF,
            "semantic_surprisal_clip": SEMANTIC_SURPRISAL_CLIP,
            "semantic_pseudocount": SEMANTIC_PSEUDOCOUNT,
            "objective": (
                "uniform_verified_likelihood_plus_fixed_open_set_semantic_maxent"
            ),
            "scientific_difference": (
                "against E78 replay: the applied semantic advantage only"
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
        f"[e81] cells={len(records)} released={len(submitted)} "
        f"snapshot={snapshot_root} ledger={ledger}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
