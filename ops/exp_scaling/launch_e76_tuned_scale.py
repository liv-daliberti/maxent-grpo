#!/usr/bin/env python3
"""Build and submit the three gated stages of E76 tuned-scale."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e72_b3a_replay_ablation as base  # noqa: E402

MODELS = ("falcon1b", "qwen3b")
DOMAINS = ("graph_coloring", "pantry_plan")
LEARNING_RATES = (5e-8, 1e-7, 2e-7)
KL_BETAS = (0.0, 0.01)
REPLAY_DOSES = (0.05, 0.10, 0.20)
STAGE_SEEDS = {"a": (53,), "b": (54,), "c": (55, 56, 57)}
TUNING_ROWS = 320
FULL_ROWS = 384
MAX_TUNING_PASSES = 6

ARM_VARIANTS = {
    "grpo": "grpo_compute_matched",
    "xmode": "verified_first_global_replay_canonical",
    "rehearsal": "verified_first_replay_rehearsal_only",
}
MODEL_TAGS = {
    "falcon1b": "falcon3_1b_instruct",
    "qwen3b": "qwen25_3b_instruct",
}
MODEL_CACHE_DIRS = {
    "falcon1b": "var/cache/huggingface/transformers/models--tiiuae--Falcon3-1B-Instruct/snapshots",
    "qwen3b": "var/cache/huggingface/transformers/models--Qwen--Qwen2.5-3B-Instruct/snapshots",
}
VALIDATION_DRAW_SEEDS = {"graph_coloring": 750101, "pantry_plan": 750102}
LEDGERS = {
    "a": "var/artifacts/e76_tuned_scale_stage_a_jobs.json",
    "b": "var/artifacts/e76_tuned_scale_stage_b_jobs.json",
    "c": "var/artifacts/e76_tuned_scale_stage_c_jobs.json",
}
SELECTIONS = {
    "a": "var/artifacts/e76_tuned_scale_stage_a_selection.json",
    "b": "var/artifacts/e76_tuned_scale_stage_b_selection.json",
}


def repo_root() -> Path:
    override = os.environ.get("E76_REPO_ROOT")
    return Path(override).resolve() if override else Path(__file__).resolve().parents[2]


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


def model_root(root: Path, model: str) -> Path:
    snapshots = sorted((root / MODEL_CACHE_DIRS[model]).glob("*"))
    if len(snapshots) != 1:
        raise SystemExit(
            f"{model}: expected exactly one cached snapshot under "
            f"{root / MODEL_CACHE_DIRS[model]}, found {len(snapshots)}"
        )
    return snapshots[0]


def references(root: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(
        (root / "var/artifacts/e72_frontier_source_runs.json").read_text()
    )
    result = {
        str(run["domain"]): run
        for run in payload["runs"]
        if run["arm"] == "xgrpo" and int(run["seed"]) == 43
    }
    missing = set(DOMAINS) - set(result)
    if missing:
        raise SystemExit(f"missing E72 reference domains: {sorted(missing)}")
    return result


def lr_tag(value: float) -> str:
    return {5e-8: "5em8", 1e-7: "1em7", 2e-7: "2em7"}[value]


def beta_tag(value: float) -> str:
    return "0" if value == 0 else "1em2"


def dose_tag(value: float | None) -> str:
    return "na" if value is None else str(int(round(value * 100))).zfill(2)


def run_stamp(
    stage: str,
    model: str,
    domain: str,
    arm: str,
    seed: int,
    lr: float,
    beta: float,
    dose: float | None,
) -> str:
    short_domain = {"graph_coloring": "graph", "pantry_plan": "pantry"}[domain]
    pieces = [
        f"e76{stage}", model, short_domain, arm, f"lr{lr_tag(lr)}",
        f"kl{beta_tag(beta)}",
    ]
    if arm != "grpo":
        pieces.append(f"rep{dose_tag(dose)}")
    pieces.append(f"s{seed}")
    return "_".join(pieces)


def save_path(root: Path, stamp: str, model: str, arm: str) -> Path:
    return root / "var/data" / f"xdr_{MODEL_TAGS[model]}_{ARM_VARIANTS[arm]}_{stamp}"


def selection(root: Path, stage: str) -> dict[str, Any]:
    path = root / SELECTIONS[stage]
    if not path.is_file():
        raise SystemExit(f"required selection is absent: {path}")
    payload = json.loads(path.read_text())
    if not payload.get("complete"):
        raise SystemExit(f"selection is not complete: {path}")
    return payload


def stage_cells(root: Path, stage: str) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    if stage == "a":
        for model in MODELS:
            for domain in DOMAINS:
                for arm in ("grpo", "xmode"):
                    for lr in LEARNING_RATES:
                        for beta in KL_BETAS:
                            cells.append(
                                dict(model=model, domain=domain, arm=arm, lr=lr,
                                     beta=beta, dose=None, seed=53,
                                     target_passes=MAX_TUNING_PASSES)
                            )
        return cells

    selected_a = selection(root, "a")
    if stage == "b":
        for model in MODELS:
            model_choice = selected_a["models"][model]
            for domain in DOMAINS:
                common = dict(
                    model=model,
                    domain=domain,
                    lr=float(model_choice["learning_rate"]),
                    beta=float(model_choice["beta"]),
                    seed=54,
                    target_passes=int(model_choice["domains"][domain]["stopping_pass"]),
                )
                cells.append(dict(**common, arm="grpo", dose=None))
                for arm in ("xmode", "rehearsal"):
                    for dose in REPLAY_DOSES:
                        cells.append(dict(**common, arm=arm, dose=dose))
        return cells

    if stage != "c":
        raise SystemExit(f"unknown stage: {stage}")
    selected_b = selection(root, "b")
    for model in MODELS:
        model_choice = selected_a["models"][model]
        for domain in DOMAINS:
            for seed in STAGE_SEEDS["c"]:
                for arm in ("grpo", "xmode", "rehearsal"):
                    dose = None
                    if arm != "grpo":
                        dose = float(selected_b["models"][model]["domains"][domain][arm]["dose"])
                    cells.append(
                        dict(
                            model=model,
                            domain=domain,
                            arm=arm,
                            lr=float(model_choice["learning_rate"]),
                            beta=float(model_choice["beta"]),
                            dose=dose,
                            seed=seed,
                            target_passes=int(model_choice["domains"][domain]["stopping_pass"]),
                        )
                    )
    return cells


def build_env(
    root: Path,
    source: dict[str, Any],
    cell: dict[str, Any],
    stage: str,
    snapshot_root: Path,
) -> tuple[dict[str, str], Path, str]:
    cloned = base.reseed(source, int(cell["seed"]))
    stamp = run_stamp(stage=stage, **{k: cell[k] for k in ("model", "domain", "arm", "seed", "lr", "beta", "dose")})
    target = save_path(root, stamp, cell["model"], cell["arm"])
    env = base.build_export_vars(root, cloned, target, "xgrpo")
    tuning = stage in ("a", "b")
    pool_rows = TUNING_ROWS if tuning else FULL_ROWS
    prompt_data = (
        root / "var/data/e76_tuned_scale" / cell["domain"] / "train"
        if tuning else Path(source["inherited_eval_config"]["prompt_data"])
    )
    eval_data = (
        root / "var/data/e76_tuned_scale" / cell["domain"] / "validation"
        if tuning else Path(source["inherited_eval_config"]["eval_data"])
    )
    env.update(
        {
            "OAT_ZERO_VARIANT": ARM_VARIANTS[cell["arm"]],
            "OAT_ZERO_PRETRAIN": str(model_root(root, cell["model"])),
            "SAVE_PATH": str(target),
            "RUN_STAMP": stamp,
            "OAT_ZERO_PROMPT_DATA": str(prompt_data),
            "OAT_ZERO_EVAL_DATA": str(eval_data),
            "OAT_ZERO_TEST_SPLIT": "multi_answer",
            "OAT_ZERO_MAX_TRAIN": str(pool_rows),
            "OAT_ZERO_SEED": str(cell["seed"]),
            "OAT_ZERO_LEARNING_RATE": repr(float(cell["lr"])),
            "OAT_ZERO_BETA": repr(float(cell["beta"])),
            "OAT_ZERO_NUM_PROMPT_EPOCH": str(cell["target_passes"]),
            "OAT_ZERO_MAX_PROMPT_EPOCHS": str(cell["target_passes"]),
            "OAT_ZERO_EVAL_PROMPT_INTERVAL": str(80 if tuning else 96),
            "OAT_ZERO_EVAL_MODE_COVERAGE_SEED": str(
                VALIDATION_DRAW_SEEDS[cell["domain"]]
                if tuning else source["inherited_eval_config"]["eval_mode_coverage_seed"]
            ),
            "OAT_ZERO_SAVE_STEPS": str(pool_rows),
            "OAT_ZERO_SAVE_FROM": str(pool_rows),
            "OAT_ZERO_RESUME_STEPS": str(pool_rows),
            "OAT_ZERO_SOURCE_ROOT": str(snapshot_root / "src"),
            "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot_root / "ops"),
            "OAT_ZERO_N_GPU": "1",
            "OAT_ZERO_NUM_GPUS_PER_ACTOR": "1",
            "OAT_ZERO_ADAM_OFFLOAD": "1",
        }
    )
    if cell["model"] == "falcon1b":
        env.update(
            {
                "OAT_ZERO_PROMPT_TEMPLATE": (
                    "falcon_pantry_support_mask"
                    if cell["domain"] == "pantry_plan" else "falcon_boxed"
                ),
                "OAT_ZERO_ACTIVATION_OFFLOADING": "0",
                "OAT_ZERO_EVAL_BATCH_SIZE": "64",
            }
        )
        if cell["domain"] == "pantry_plan":
            env["OAT_ZERO_CANONICAL_ACTION_TASK"] = "pantry_support_mask"
    else:
        env.update(
            {
                "OAT_ZERO_ACTIVATION_OFFLOADING": "1",
                "OAT_ZERO_EVAL_BATCH_SIZE": "32",
            }
        )

    if cell["arm"] == "grpo":
        env.update(
            {
                "OAT_ZERO_SEMANTIC_SHANNON_COEF": "0.0",
                "OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA": "0.0",
                "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE": "0",
                "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE": "0",
                "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE": "verified_likelihood_per_rollout",
                "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY": "1",
            }
        )
    elif cell["dose"] is not None:
        env["OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA"] = repr(float(cell["dose"]))
        env["OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA"] = repr(float(cell["dose"]))
    return env, target, stamp


def tree_hash(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for base_path in paths:
        for path in sorted(p for p in base_path.rglob("*") if p.is_file()):
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            digest.update(str(path.relative_to(base_path.parent)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def ensure_snapshot(root: Path, requested: Path | None) -> Path:
    if requested:
        snapshot = requested.resolve()
        if not (snapshot / "src/oat_drgrpo/__init__.py").is_file():
            raise SystemExit(f"invalid E76 source snapshot: {snapshot}")
        return snapshot
    identity = tree_hash((root / "src/oat_drgrpo", root / "ops"))
    snapshot = root / "var/artifacts/source_snapshots" / f"e76_tuned_scale_{identity[:16]}"
    if snapshot.is_dir():
        return snapshot
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{snapshot.name}.", dir=snapshot.parent))
    try:
        shutil.copytree(
            root / "src/oat_drgrpo", temporary / "src/oat_drgrpo",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copytree(
            root / "ops", temporary / "ops",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        (temporary / "SNAPSHOT_IDENTITY.json").write_text(
            json.dumps(
                {"schema": "e76_runtime_snapshot_v1", "sha256": identity},
                indent=2, sort_keys=True,
            ) + "\n"
        )
        os.replace(temporary, snapshot)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return snapshot


def all_active(ids: Iterable[str]) -> list[str]:
    candidates = sorted({str(job_id) for job_id in ids if str(job_id).isdigit()}, key=int)
    if not candidates:
        return []
    result = subprocess.run(
        ["squeue", "-h", "-j", ",".join(candidates), "-o", "%A"],
        capture_output=True, text=True, check=False, timeout=30,
    )
    if result.returncode != 0:
        raise SystemExit(f"could not resolve active dependencies: {result.stderr.strip()}")
    return sorted({line.strip() for line in result.stdout.splitlines() if line.strip()}, key=int)


def initial_dependencies(root: Path) -> dict[str, list[str]]:
    falcon_ids: list[str] = []
    for path in root.glob("var/artifacts/*e73_falcon3_1b*comparative_jobs.tsv"):
        with path.open(newline="") as source:
            for row in csv.DictReader(source, delimiter="\t"):
                falcon_ids.append(row.get("job_id", ""))
    qwen_ids: list[str] = []
    qwen_ledger = root / "var/artifacts/e74_qwen3b_scale_jobs.json"
    if qwen_ledger.is_file():
        qwen_ids = [str(row.get("job_id", "")) for row in json.loads(qwen_ledger.read_text()).get("runs", [])]
    return {"falcon1b": all_active(falcon_ids), "qwen3b": all_active(qwen_ids)}


def placement(cell: dict[str, Any]) -> dict[str, str]:
    if cell["model"] == "qwen3b":
        return dict(partition="mltheory", account="mltheory", nodelist="node302",
                    gres="gpu:a100:1", cpus="16", memory="128G")
    if cell["domain"] == "pantry_plan":
        return dict(partition="all", account="allcs", nodelist="node205,node206,node208",
                    gres="gpu:a6000:1", cpus="8", memory="64G")
    return dict(partition="all", account="allcs", nodelist="node105,node202,node203,node204",
                gres="gpu:a5000:1", cpus="8", memory="64G")


def submit_cell(
    root: Path,
    stage: str,
    env: dict[str, str],
    cell: dict[str, Any],
    dependency_ids: list[str],
    dry_run: bool,
) -> str | None:
    place = placement(cell)
    name = f"e76{stage}-{cell['model'][:3]}-{cell['domain'][:3]}-{cell['arm'][:3]}-s{cell['seed']}"
    export_pairs = ",".join(f"{key}={value}" for key, value in env.items())
    command = [
        "sbatch", "--parsable", f"--job-name={name}", f"--export=ALL,{export_pairs}",
        f"--partition={place['partition']}", f"--account={place['account']}",
        f"--nodelist={place['nodelist']}", f"--gres={place['gres']}",
        f"--cpus-per-task={place['cpus']}", f"--mem={place['memory']}",
        "--time=2-00:00:00", "--nice=0",
    ]
    if dependency_ids:
        command.append("--dependency=afterany:" + ":".join(dependency_ids))
    if place["partition"] == "all":
        command.append("--hold")
    command.append(str(root / "ops/slurm/train_node302.slurm"))
    if dry_run:
        print(" ".join(shlex.quote(part) for part in command))
        return None
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"submission failed for {name}: {result.stderr.strip()}")
    job_id = result.stdout.strip().split(";")[0]
    if place["partition"] == "all":
        normalize = subprocess.run(
            [
                "scontrol", "update", f"JobId={job_id}", "Partition=all",
                f"NodeList={place['nodelist']}",
            ],
            capture_output=True, text=True, check=False,
        )
        if normalize.returncode != 0:
            subprocess.run(["scancel", job_id], check=False)
            raise SystemExit(
                f"partition normalization failed for {name}: {normalize.stderr.strip()}"
            )
        release = subprocess.run(
            ["scontrol", "release", job_id],
            capture_output=True, text=True, check=False,
        )
        if release.returncode != 0:
            subprocess.run(["scancel", job_id], check=False)
            raise SystemExit(f"release failed for {name}: {release.stderr.strip()}")
    return job_id


def submit_controller(
    root: Path,
    stage: str,
    job_ids: list[str],
    snapshot_root: Path,
    dry_run: bool,
) -> str | None:
    if stage == "c":
        return None
    command = [
        "sbatch", "--parsable", f"--job-name=e76ctl-{stage}",
        f"--dependency=afterany:{':'.join(job_ids)}",
        "--partition=cs", "--account=allcs", "--cpus-per-task=1", "--mem=4G",
        "--time=00:30:00",
        "--export=" + ",".join(
            [
                "ALL", f"E76_REPO_ROOT={root}", f"E76_COMPLETED_STAGE={stage}",
                f"E76_SNAPSHOT_ROOT={snapshot_root}",
                f"E76_OPS_SNAPSHOT_ROOT={snapshot_root / 'ops'}",
            ]
        ),
        str(snapshot_root / "ops/slurm/e76_advance.slurm"),
    ]
    if dry_run:
        print(" ".join(shlex.quote(part) for part in command))
        return None
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"controller submission failed: {result.stderr.strip()}")
    return result.stdout.strip().split(";")[0]


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("a", "b", "c"), default="a")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--queue-controller", action="store_true")
    parser.add_argument("--no-initial-dependencies", action="store_true")
    parser.add_argument("--snapshot-root", type=Path)
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")

    split_manifest = root / "var/artifacts/e76_tuned_scale_splits.json"
    if not split_manifest.is_file():
        raise SystemExit(f"sealed split manifest is absent: {split_manifest}")
    protocol = root / "paper/preregistration/e76_tuned_scale_validation_screen_20260803.md"
    if not protocol.is_file():
        raise SystemExit(f"frozen protocol is absent: {protocol}")
    snapshot_root = ensure_snapshot(root, args.snapshot_root)
    sources = references(root)
    dependencies = {model: [] for model in MODELS}
    if args.stage == "a" and not args.no_initial_dependencies:
        dependencies = initial_dependencies(root)

    ledger_path = root / LEDGERS[args.stage]
    if args.submit and ledger_path.exists():
        existing = json.loads(ledger_path.read_text())
        if existing.get("runs"):
            raise SystemExit(f"refusing duplicate submission; ledger already has runs: {ledger_path}")

    records: list[dict[str, Any]] = []
    for cell in stage_cells(root, args.stage):
        env, target, stamp = build_env(root, sources[cell["domain"]], cell, args.stage, snapshot_root)
        if target.exists():
            raise SystemExit(f"refusing to overwrite existing E76 run directory: {target}")
        job_id = submit_cell(
            root, args.stage, env, cell, dependencies[cell["model"]],
            dry_run=(args.dry_run or not args.submit),
        )
        records.append(
            {
                **cell,
                "stage": args.stage,
                "job_id": job_id,
                "run_stamp": stamp,
                "run_dir": str(target),
                "target_steps": int(cell["target_passes"]) * (TUNING_ROWS if args.stage in ("a", "b") else FULL_ROWS),
                "placement": placement(cell),
                "upstream_dependencies": dependencies[cell["model"]],
            }
        )

    controller_job_id = None
    job_ids = [str(record["job_id"]) for record in records if record["job_id"]]
    if args.queue_controller and args.stage != "c":
        if not job_ids and args.submit:
            raise SystemExit("cannot queue a controller without submitted jobs")
        controller_job_id = submit_controller(
            root, args.stage, job_ids, snapshot_root,
            dry_run=(args.dry_run or not args.submit),
        )
    payload = {
        "schema": f"e76_tuned_scale_stage_{args.stage}_jobs_v1",
        "stage": args.stage,
        "protocol": str(protocol),
        "split_manifest": str(split_manifest),
        "snapshot_root": str(snapshot_root),
        "initial_dependencies": dependencies,
        "controller_job_id": controller_job_id,
        "runs": records,
    }
    if args.submit:
        atomic_json(ledger_path, payload)
    print(
        f"[e76{args.stage}] cells={len(records)} submitted={len(job_ids)} "
        f"controller={controller_job_id or '-'} snapshot={snapshot_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
