#!/usr/bin/env python3
"""Submit E74: the headline design at 3B, on the idle A100 node.

The manuscript's headline surface is one model size. E73 answered "is this a
property of that model *family*" by moving to Falcon3-1B; this cohort asks the
other question, "is it a property of that model *size*", by holding the family
and the entire design fixed and moving only the parameter count ---
Qwen2.5-0.5B-Instruct to Qwen2.5-3B-Instruct.

Because the family is held, this is a cleaner comparison than the cross-family
one. The chat surface, prompt template, tokenizer lineage, verifier, canonical
keys, data, optimizer, budget, and evaluation draw seeds are all inherited from
the published runs; the only deliberate differences are the checkpoint itself
and the memory settings the larger model needs.

Placement is deliberate too. Every run is pinned to node302, the cluster's only
A100 host, which the a5000-pinned cohorts cannot use and which would otherwise
sit idle. This cohort therefore competes with nothing else in flight.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e72_b3a_replay_ablation as base  # noqa: E402

MODEL_TAG = "qwen25_3b_instruct"
MODEL_DIR = (
    "var/cache/huggingface/transformers/models--Qwen--Qwen2.5-3B-Instruct/snapshots"
)

# The two arms of the published comparison, nothing else.
ARMS: dict[str, str] = {
    "grpo": "grpo_compute_matched",
    "xgrpo": "verified_first_global_replay_canonical",
}

# Only node302 has A100s, and no other cohort in flight can use them.
PLACEMENT_NODE = "node302"
# One A100 per run, not two. The published design draws one prompt per
# optimizer step (`rollout_batch_size=1`), and the learner requires that to be
# divisible by the GPUs per actor --- so any two-GPU layout would have to double
# the prompts per step and would no longer be the same experiment. A 3B run
# fits one A100 with the optimizer and activations offloaded, which is how this
# repository has run 3B before, and it fills node302 with eight concurrent runs
# rather than four.
GPUS_PER_RUN = 1

DOMAIN_TAGS = {
    "graph_coloring": "gce74",
    "countdown": "cde74",
    "python_factors": "pye74",
    "mathir": "mie74",
    "pantry_plan": "ppe74",
}
# The published budget. ``--epochs`` shortens it, and the stamp carries the
# number so a run directory never claims a depth it did not train to: a
# four-epoch run lands in ``*_4pass_*``, not in ``*_12pass_*``.
DEFAULT_EPOCHS = 12


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def model_root(root: Path) -> Path:
    snapshots = sorted((root / MODEL_DIR).glob("*"))
    if not snapshots:
        raise SystemExit(f"no Qwen2.5-3B-Instruct snapshot under {root / MODEL_DIR}")
    if len(snapshots) > 1:
        raise SystemExit(
            f"{len(snapshots)} snapshots present; pin one explicitly: {snapshots}"
        )
    return snapshots[0]


def run_stamp(domain: str, arm: str, seed: int, epochs: int = DEFAULT_EPOCHS) -> str:
    return f"{DOMAIN_TAGS[domain]}_qwen3b_{epochs}pass_{arm}_s{seed}"


def save_path(
    root: Path, domain: str, arm: str, seed: int, epochs: int = DEFAULT_EPOCHS
) -> Path:
    return (
        root
        / "var"
        / "data"
        / f"xdr_{MODEL_TAG}_{ARMS[arm]}_{run_stamp(domain, arm, seed, epochs)}"
    )


def build_env(
    root: Path, reference: dict[str, Any], arm: str, seed: int,
    epochs: int = DEFAULT_EPOCHS,
) -> dict[str, str]:
    """The published run's own configuration, with the model and device changed."""

    domain = str(reference["domain"])
    target = save_path(root, domain, arm, seed, epochs)
    # Start from the treatment's inherited block so data, template, verifier,
    # budgets, and evaluation draw seeds are the published ones verbatim.
    env = base.build_export_vars(root, reference, target, "xgrpo")

    env.update(
        {
            "OAT_ZERO_VARIANT": ARMS[arm],
            "OAT_ZERO_PRETRAIN": str(model_root(root)),
            "SAVE_PATH": str(target),
            "RUN_STAMP": run_stamp(domain, arm, seed, epochs),
            # Everything else is the published configuration verbatim; only the
            # number of passes over the fixed pool is shortened.
            "OAT_ZERO_MAX_PROMPT_EPOCHS": str(epochs),
            "OAT_ZERO_NUM_PROMPT_EPOCH": str(epochs),
            "OAT_ZERO_SEED": str(seed),
            # Offload optimizer state and activations: 3B does not otherwise
            # leave room beside a collocated vLLM engine on one card.
            "OAT_ZERO_N_GPU": str(GPUS_PER_RUN),
            "OAT_ZERO_NUM_GPUS_PER_ACTOR": str(GPUS_PER_RUN),
            "OAT_ZERO_ADAM_OFFLOAD": "1",
            "OAT_ZERO_ACTIVATION_OFFLOADING": "1",
            "OAT_ZERO_ZERO_STAGE": "2",
            "OAT_ZERO_VLLM_GPU_RATIO": "0.25",
            "OAT_ZERO_EVAL_BATCH_SIZE": "32",
        }
    )
    if arm == "grpo":
        # The published compute-matched control: replay bookkeeping runs with
        # its score derivative identically zero, and no discovery channel acts.
        env.update(
            {
                "OAT_ZERO_SEMANTIC_SHANNON_COEF": "0.0",
                "OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA": "0.0",
                "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE": "0",
                "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE": "0",
                "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE": (
                    "verified_likelihood_per_rollout"
                ),
                "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY": "1",
            }
        )
    return env


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "var" / "artifacts" / "e72_frontier_source_runs.json",
    )
    parser.add_argument("--domains", default="graph_coloring")
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help="passes over the fixed pool; also written into the run stamp",
    )
    parser.add_argument("--seeds", default="43")
    parser.add_argument("--arms", default="grpo,xgrpo")
    parser.add_argument("--partition", default="mltheory")
    parser.add_argument("--account", default="mltheory")
    parser.add_argument("--cpus", type=int, default=16)
    parser.add_argument("--memory", default="128G")
    parser.add_argument("--time-limit", default="2-00:00:00")
    parser.add_argument("--hold", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    domains = [part for part in args.domains.split(",") if part]
    seeds = [int(part) for part in args.seeds.split(",") if part]
    arms = [part for part in args.arms.split(",") if part]
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}")

    references = {
        str(run["domain"]): run
        for run in manifest["runs"]
        if run["arm"] == "xgrpo" and int(run["seed"]) == 43
    }
    submitted: list[dict[str, Any]] = []
    for domain in domains:
        reference = references.get(domain)
        if reference is None:
            raise SystemExit(f"no published reference run for domain {domain!r}")
        for seed in seeds:
            for arm in arms:
                target = save_path(root, domain, arm, seed, args.epochs)
                if args.skip_existing and (
                    (target / "TRAINING_COMPLETE.json").exists()
                    or list(target.glob("debug_job*"))
                ):
                    continue
                env = build_env(root, reference, arm, seed, args.epochs)
                export_pairs = ",".join(f"{k}={v}" for k, v in env.items())
                name = f"e74-{arm}-{domain[:6]}-s{seed}"
                sbatch = [
                    "sbatch",
                    "--parsable",
                    f"--job-name={name}",
                    f"--export=ALL,{export_pairs}",
                    f"--nodelist={PLACEMENT_NODE}",
                    f"--gres=gpu:a100:{GPUS_PER_RUN}",
                    f"--cpus-per-task={args.cpus}",
                    f"--mem={args.memory}",
                    f"--time={args.time_limit}",
                    f"--partition={args.partition}",
                    f"--account={args.account}",
                ]
                if args.hold:
                    sbatch.append("--hold")
                sbatch.append(str(root / "ops" / "run_experiment.sh"))
                if args.dry_run:
                    print(f"  would submit {name} -> {target.name}")
                    submitted.append({"name": name, "run_dir": str(target)})
                    continue
                result = subprocess.run(sbatch, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"  submit failed [{name}]: {result.stderr[:300]}")
                    continue
                submitted.append(
                    {
                        "name": name,
                        "domain": domain,
                        "arm": arm,
                        "seed": seed,
                        "job_id": result.stdout.strip(),
                        "run_dir": str(target),
                    }
                )

    ledger = root / "var" / "artifacts" / "e74_qwen3b_scale_jobs.json"
    existing = json.loads(ledger.read_text())["runs"] if ledger.is_file() else []
    payload = {
        "schema": "e74_qwen3b_scale_jobs_v1",
        "model": str(model_root(root)),
        "gpus_per_run": GPUS_PER_RUN,
        "placement_node": PLACEMENT_NODE,
        "runs": existing + submitted,
    }
    if not args.dry_run:
        handle, temporary = tempfile.mkstemp(prefix=f".{ledger.name}.", dir=ledger.parent)
        with os.fdopen(handle, "w", encoding="utf-8") as sink:
            json.dump(payload, sink, indent=2, sort_keys=True)
            sink.write("\n")
        os.replace(temporary, ledger)
    print(f"[e74] submitted={len(submitted)} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
