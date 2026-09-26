#!/usr/bin/env python3
"""Configure or launch the frozen Ant v15/v17 0.5B viability gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "var/seed_paper_eval/paper310/bin/python"
MODEL = ROOT / "var/models/ant_maze_interactive_warmstart_v13"
WARMSTART_RECEIPT = (
    ROOT / "var/artifacts/ant_maze_interactive_warmstart_sft_v13.json"
)
DATA = ROOT / "var/data/ant_maze_modebench_v15_controller_v17_r1"
ROUTE_IDENTITY = (
    ROOT
    / "var/artifacts/"
    "ant_maze_v15_controller_v17_r1_admission_identity.json"
)
ADMISSION = (
    ROOT
    / "var/artifacts/"
    "ant_maze_modebench_v15_controller_v17_r1_admission_audit.json"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "ant_maze_interactive_viability_v17_20260730.md"
)
EVALUATOR = ROOT / "ops/evaluate_ant_maze_interactive_viability_v17.py"
EVALUATOR_V13 = (
    ROOT / "ops/evaluate_ant_maze_interactive_viability_v13.py"
)
EVALUATOR_BASE = ROOT / "ops/evaluate_point_maze_interactive_viability.py"
BATCH = ROOT / "ops/slurm/evaluate_ant_maze_interactive_viability_v17.slurm"
IDENTITY = (
    ROOT / "var/artifacts/ant_maze_interactive_viability_v17_identity.json"
)
RECEIPT = (
    ROOT / "var/artifacts/ant_maze_interactive_05b_viability_v17.json"
)
SUBMISSION = (
    ROOT / "var/artifacts/ant_maze_interactive_viability_v17_submission.json"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def run(command: Sequence[str], *, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        list(command),
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate() -> None:
    for path in (
        PYTHON,
        MODEL / "config.json",
        WARMSTART_RECEIPT,
        PROTOCOL,
        EVALUATOR,
        EVALUATOR_V13,
        EVALUATOR_BASE,
        BATCH,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{ROOT / 'ops'}:{ROOT / 'src'}"
    library = str(ROOT / "var/seed_paper_eval/paper310/lib")
    environment["LD_LIBRARY_PATH"] = library + (
        ":" + environment["LD_LIBRARY_PATH"]
        if environment.get("LD_LIBRARY_PATH")
        else ""
    )
    run(
        [
            str(PYTHON),
            "-m",
            "py_compile",
            str(EVALUATOR),
            str(EVALUATOR_V13),
            str(EVALUATOR_BASE),
            str(Path(__file__).resolve()),
            str(
                ROOT
                / "src/oat_drgrpo/"
                "ant_maze_interactive_worker_v17_r1.py"
            ),
            str(
                ROOT
                / "src/oat_drgrpo/"
                "ant_maze_interactive_process_v17_r1.py"
            ),
        ],
        env=environment,
    )
    run(["bash", "-n", str(BATCH)])
    run(
        [
            str(PYTHON),
            "-m",
            "pytest",
            "-q",
            str(ROOT / "tests/test_ant_maze_interactive_v13.py"),
            str(ROOT / "tests/test_ant_maze_warmstart_v13.py"),
        ],
        env=environment,
    )


def prerequisites() -> tuple[dict[str, Any], Path, str]:
    for path in (
        DATA / "identity.json",
        DATA / "dev/dataset_dict.json",
        ROUTE_IDENTITY,
        ADMISSION,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    route = json.loads(ROUTE_IDENTITY.read_text(encoding="utf-8"))
    audit = json.loads(ADMISSION.read_text(encoding="utf-8"))
    data = json.loads((DATA / "identity.json").read_text(encoding="utf-8"))
    warmstart = json.loads(WARMSTART_RECEIPT.read_text(encoding="utf-8"))
    if (
        route.get("schema_version")
        != "ant-maze-v17-route-generation-identity-v1"
        or route.get("language_model_sampled") is not False
        or route.get("post_outcome_map_or_route_substitution") is not False
    ):
        raise RuntimeError("Ant v17 route-generation identity drift")
    if (
        audit.get("status") != "pass"
        or audit.get("decision")
        != "admitted_to_ant_v15_v17_frozen_model_viability_gate"
        or audit.get("maps") != 12
        or audit.get("real_simulator_replays") != 24
        or audit.get("perturbation_replays") != 2400
    ):
        raise RuntimeError("Ant v15/v17 admission did not authorize viability")
    if data.get("schema_version") != (
        "ant-maze-modebench-data-v15-controller-v17-r1"
    ):
        raise RuntimeError("Ant v15/v17 data schema drift")
    expected_tree = (
        "0f18039887e77b2e37eb15161dd8bb44ed6103f188755db9029bf0a61fe22090"
    )
    if (
        warmstart.get("status") != "pass"
        or warmstart.get("hashes", {}).get("output_model_tree_sha256")
        != expected_tree
        or tree_hash(MODEL) != expected_tree
        or warmstart.get("information_boundary", {}).get("dev_dataset_loaded")
        is not False
        or warmstart.get("information_boundary", {}).get(
            "eval_dataset_loaded"
        )
        is not False
    ):
        raise RuntimeError("Ant v13 transferred warm-start identity drift")
    source_root = Path(str(route["source_root"]))
    source_hash = str(route["source_hash"])
    if (
        not source_root.is_dir()
        or tree_hash(source_root) != source_hash
        or not (
            source_root
            / "oat_drgrpo/ant_maze_interactive_worker_v17_r1.py"
        ).is_file()
    ):
        raise RuntimeError("Ant v17 exact source snapshot mismatch")
    return route, source_root, source_hash


def snapshot_execution() -> tuple[Path, str]:
    temporary = Path(
        tempfile.mkdtemp(
            prefix=".ant-v17-viability.",
            dir=ROOT / "var/artifacts/source_snapshots",
        )
    )
    for source in (
        PROTOCOL,
        EVALUATOR,
        EVALUATOR_V13,
        EVALUATOR_BASE,
        BATCH,
    ):
        shutil.copy2(source, temporary / source.name)
    digest = tree_hash(temporary)
    target = (
        ROOT
        / f"var/artifacts/source_snapshots/ant_v17_viability_ops_{digest}"
    )
    if target.exists():
        shutil.rmtree(temporary)
    else:
        os.replace(temporary, target)
    if tree_hash(target) != digest:
        raise RuntimeError("Ant v17 viability execution snapshot mismatch")
    return target, digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "run"))
    args = parser.parse_args()
    validate()
    if args.phase == "config":
        print("[ant-v17-viability] configuration passed; no model sampled")
        return
    route, source_root, source_hash = prerequisites()
    for path in (IDENTITY, RECEIPT, SUBMISSION):
        if path.exists():
            raise FileExistsError(
                f"fresh Ant v17 viability artifact required: {path}"
            )
    execution_root, execution_hash = snapshot_execution()
    output = run(
        [
            "sbatch",
            "--parsable",
            "--hold",
            "--partition=all",
            "--account=mltheory",
            "--export=ALL,"
            f"ROOT_DIR={ROOT},OAT_ZERO_SOURCE_ROOT={source_root},"
            f"OAT_ZERO_EXECUTION_ROOT={execution_root},"
            f"OAT_ZERO_SOURCE_HASH={source_hash},"
            f"OAT_ZERO_EXECUTION_HASH={execution_hash},"
            f"OAT_ZERO_ROUTE_IDENTITY={ROUTE_IDENTITY}",
            str(execution_root / BATCH.name),
        ]
    )
    job_id = int(output.split(";", 1)[0])
    try:
        run(["scontrol", "update", f"JobId={job_id}", "Requeue=0"])
        record = run(["scontrol", "show", "job", "-o", str(job_id)])
        for required in (
            "JobState=PENDING",
            "Reason=JobHeldUser",
            "gres/gpu:a5000:1",
            "NumCPUs=8",
            "MinMemoryNode=64G",
            "TimeLimit=04:00:00",
            "Requeue=0",
            f"OAT_ZERO_SOURCE_ROOT={source_root}",
            f"OAT_ZERO_ROUTE_IDENTITY={ROUTE_IDENTITY}",
        ):
            if required not in record:
                raise RuntimeError(
                    f"held Ant v17 viability job lacks {required}"
                )
        atomic(
            IDENTITY,
            {
                "schema_version": "ant-maze-interactive-viability-identity-v17",
                "job_id": job_id,
                "protocol_sha256": sha(PROTOCOL),
                "launcher_sha256": sha(Path(__file__).resolve()),
                "evaluator_sha256": sha(EVALUATOR),
                "source_root": str(source_root),
                "source_hash": source_hash,
                "execution_root": str(execution_root),
                "execution_hash": execution_hash,
                "route_identity_sha256": sha(ROUTE_IDENTITY),
                "admission_audit_sha256": sha(ADMISSION),
                "data_identity_sha256": sha(DATA / "identity.json"),
                "warmstart_receipt_sha256": sha(WARMSTART_RECEIPT),
                "model_tree_sha256": tree_hash(MODEL),
                "model": "Qwen2.5-0.5B-Instruct-v13-train-only-warmstart",
                "split": "dev/multi_answer",
                "evaluation_split_loaded": False,
                "sample_count": 64,
                "prefix_count": 16,
                "sampling_seed": 108317,
                "minimum_prefix_success_prompts": 2,
                "minimum_multimode_prompts": 1,
                "route_job_id": route["job_id"],
                "certified_route_programs_in_context": False,
            },
        )
        atomic(
            SUBMISSION,
            {
                "schema_version": (
                    "ant-maze-interactive-viability-submission-v17"
                ),
                "job_id": job_id,
                "identity_sha256": sha(IDENTITY),
                "released": True,
            },
        )
        run(["scontrol", "release", str(job_id)])
    except BaseException:
        subprocess.run(["scancel", str(job_id)], cwd=ROOT, check=False)
        raise
    print(f"[ant-v17-viability] released job {job_id}")


if __name__ == "__main__":
    main()
