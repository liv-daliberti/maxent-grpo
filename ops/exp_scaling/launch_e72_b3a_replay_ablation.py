#!/usr/bin/env python3
"""Submit E72 B3a: the xGRPO objective with the replay gradient removed.

B3a answers two questions with one cohort. As a *baseline* it is a count-based
rare-outcome method with no acting memory---open-set rare-mode and
first-discovery advantages, no replay pressure---which is the third family the
review asks for. As an *ablation* it is the compute-matched remove-one control
for verified replay, the component `tab:component-ablations` currently supports
by analytic argument alone.

It is deliberately not dose-screened. Every coefficient is inherited from the
xGRPO run it will be compared against, so the only difference is the replay
score derivative, which `online_canonical_replay_compute_only` sets to exactly
zero while the same banks are still retained, scheduled, teacher-forced, and
traversed backward. Tuning a coefficient here would answer a different question.

Placement is inherited too: each run is pinned to the GPU model that trained its
paired xGRPO seed, because GPU model changes sampling numerics (see the decoding
appendix). Without that, a third arm would reintroduce the confound the frontier
sweep removed.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

REFERENCE_ARM = "xgrpo"
MODEL_TAG = "qwen25_0p5b_instruct"

# Both arms are remove-one ablations of the same frozen treatment, and together
# they decompose it: B3a keeps discovery credit and loses the ability to act on
# a banked discovery; B1a keeps verified replay and loses discovery credit. B1a
# is simultaneously the review's first-named baseline, Dr.GRPO plus ordinary
# verified-response replay.
ARM_SPECS: dict[str, dict[str, Any]] = {
    "b3a": {
        "variant": "verified_first_replay_gradient_ablation",
        "removes": "the replay score gradient",
        # Coefficients are inherited from the reference run unchanged; the
        # variant itself zeroes the replay derivative.
        "overrides": {},
    },
    "b1a": {
        "variant": "verified_first_replay_only_ablation",
        "removes": "open-set discovery credit",
        # The variant, not these overrides, is what disables discovery: the
        # treatment's own block hardcodes separate-advantage, success-
        # conditioned, and open-set adaptation to on, and those switches are
        # invalid at a zero coefficient. Zeroing the coefficient by environment
        # alone produced a run that failed argument validation at startup.
        "overrides": {
            "OAT_ZERO_SEMANTIC_SHANNON_COEF": "0.0",
            "OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA": "0.0",
            "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE": "0",
            "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE": "0",
            "OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_INVERSE_ADAPTATION": "0",
        },
    },
}
VARIANT = ARM_SPECS["b3a"]["variant"]

# The frozen E58/E71 treatment. The cohort must already match these; any
# disagreement means the manifest describes a different experiment than the one
# this ablation claims to remove one component from.
EXPECTED_OBJECTIVE: dict[str, Any] = {
    "semantic_shannon_coef": 0.10,
    "semantic_shannon_surprisal_clip": 5.0,
    "semantic_shannon_pseudocount": 1.0,
    "semantic_shannon_separate_advantage": True,
    "semantic_shannon_success_conditioned_signed_advantage": True,
    "semantic_shannon_open_set_inverse_adaptation": True,
    "semantic_shannon_open_set_warmup_steps": 64,
    "semantic_shannon_open_set_ema_decay": 0.9,
    "online_canonical_bank_alpha": 0.0,
    "online_canonical_novelty_beta": 0.50,
    "online_canonical_bank_pseudocount": 1.0,
    "online_canonical_bank_surprisal_clip": 5.0,
    "online_canonical_key_mode": "modebench_outcome",
    "online_canonical_replay": True,
    "online_canonical_replay_alpha": 0.10,
    "online_canonical_replay_objective": "split_mass_balance_per_rollout",
    "online_canonical_replay_capacity": 16,
    "online_canonical_replay_global_groups_per_step": 1,
    "online_canonical_replay_warmup_steps": 64,
    "online_canonical_replay_ema_decay": 0.9,
    "online_canonical_replay_mass_alpha": 0.10,
    "online_canonical_replay_mass_warmup_steps": 64,
    "online_canonical_replay_mass_ema_decay": 0.9,
    # The treatment trains with a live replay gradient; this ablation is the
    # arm that turns it off, so the reference must have it on.
    "online_canonical_replay_compute_only": False,
    "policy_entropy_coef": 0.0,
    "maxent_alpha": 0.0,
}

EXPECTED_TRAIN: dict[str, Any] = {
    "learning_rate": 2e-07,
    "num_prompt_epoch": 12,
    "num_ppo_epochs": 1,
    "max_norm": 1.0,
    "beta": 0.0,
    "temperature": 1.0,
    "top_p": 1.0,
    "train_batch_size": 16,
    "sync_params_every": 1,
    "save_steps": 384,
    "save_from": 384,
    "resume_steps": 384,
    "export_steps": 0,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def check_inheritance(run: dict[str, Any]) -> list[str]:
    """Fail closed when the reference run is not the experiment we think it is."""
    problems: list[str] = []
    objective = run.get("inherited_objective_config", {})
    training = run.get("inherited_train_config", {})
    label = f"{run['domain']}/s{run['seed']}"
    for key, expected in EXPECTED_OBJECTIVE.items():
        actual = objective.get(key)
        if isinstance(expected, float):
            ok = actual is not None and abs(float(actual) - expected) < 1e-12
        else:
            ok = actual == expected
        if not ok:
            problems.append(f"{label}: objective {key} is {actual!r}, expected {expected!r}")
    for key, expected in EXPECTED_TRAIN.items():
        actual = training.get(key)
        if isinstance(expected, float):
            ok = actual is not None and abs(float(actual) - expected) < 1e-12
        else:
            ok = actual == expected
        if not ok:
            problems.append(f"{label}: training {key} is {actual!r}, expected {expected!r}")
    if int(training.get("seed", -1)) != int(run["seed"]):
        problems.append(f"{label}: recorded training seed does not match the run seed")
    if not run.get("source_node"):
        problems.append(f"{label}: no source node recorded")
    return problems


def run_stamp(run: dict[str, Any], arm: str = "b3a") -> str:
    prefix = run["run_stamp"].split(f"_{run['curve_arm_label']}_s")[0]
    return f"{prefix}_{arm}_s{run['seed']}"


def save_path(root: Path, run: dict[str, Any], arm: str = "b3a") -> Path:
    variant = ARM_SPECS[arm]["variant"]
    return root / "var" / "data" / f"xdr_{MODEL_TAG}_{variant}_{run_stamp(run, arm)}"


def build_export_vars(
    root: Path, run: dict[str, Any], target: Path, arm: str = "b3a"
) -> dict[str, str]:
    spec = ARM_SPECS[arm]
    evaluation = run["inherited_eval_config"]
    training = run["inherited_train_config"]
    objective = run["inherited_objective_config"]

    def flag(value: Any) -> str:
        return "1" if value else "0"

    env: dict[str, str] = {
        "OAT_ZERO_REPO_ROOT": str(root),
        "OAT_ZERO_VARIANT": spec["variant"],
        "OAT_ZERO_MODEL": "custom",
        # The base model, not the reference run's terminal checkpoint: B3a
        # trains from the same initialization as every reported arm.
        "OAT_ZERO_PRETRAIN": str(evaluation["pretrain"]),
        "SAVE_PATH": str(target),
        "RUN_STAMP": run_stamp(run, arm),
        # --- data and interface, inherited ----------------------------------
        "OAT_ZERO_PROMPT_DATA": str(evaluation["prompt_data"]),
        "OAT_ZERO_EVAL_DATA": str(evaluation["eval_data"]),
        "OAT_ZERO_PROMPT_TEMPLATE": str(evaluation["prompt_template"]),
        "OAT_ZERO_TEST_SPLIT": str(evaluation["test_split"]),
        "OAT_ZERO_INPUT_KEY": str(evaluation["eval_input_key"]),
        "OAT_ZERO_OUTPUT_KEY": str(evaluation["eval_output_key"]),
        "OAT_ZERO_EVAL_INPUT_KEY": str(evaluation["eval_input_key"]),
        "OAT_ZERO_EVAL_OUTPUT_KEY": str(evaluation["eval_output_key"]),
        "OAT_ZERO_PROMPT_MAX_LENGTH": str(evaluation["prompt_max_length"]),
        "OAT_ZERO_GENERATE_MAX_LENGTH": str(evaluation["generate_max_length"]),
        "OAT_ZERO_EVAL_GENERATE_MAX_LENGTH": str(
            evaluation["eval_generate_max_length"]
        ),
        "OAT_ZERO_MAX_MODEL_LEN": str(evaluation["max_model_len"]),
        "OAT_ZERO_VERIFIER_VERSION": str(evaluation["verifier_version"]),
        "OAT_ZERO_MAX_TRAIN": str(evaluation["max_train"]),
        "OAT_ZERO_CANONICAL_ACTION_TASK": str(evaluation["canonical_action_task"]),
        "OAT_ZERO_CANONICAL_GRAPH_ACTIONS": flag(evaluation["canonical_graph_actions"]),
        "OAT_ZERO_CANONICAL_GRAPH_ACTION_COUNT": str(
            evaluation["canonical_graph_action_count"]
        ),
        "OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING": flag(
            evaluation["canonical_graph_learner_sampling"]
        ),
        "OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING": flag(
            evaluation["canonical_graph_fixed_shape_sampling"]
        ),
        # --- optimizer and schedule, inherited ------------------------------
        "OAT_ZERO_SEED": str(training["seed"]),
        "OAT_ZERO_LEARNING_RATE": repr(float(training["learning_rate"])),
        "OAT_ZERO_NUM_PROMPT_EPOCH": str(training["num_prompt_epoch"]),
        "OAT_ZERO_MAX_PROMPT_EPOCHS": str(training["num_prompt_epoch"]),
        "OAT_ZERO_NUM_PPO_EPOCHS": str(training["num_ppo_epochs"]),
        "OAT_ZERO_MAX_QUERIES": str(training["max_queries"]),
        "OAT_ZERO_MAX_NORM": repr(float(training["max_norm"])),
        "OAT_ZERO_BETA": repr(float(training["beta"])),
        "OAT_ZERO_TEMPERATURE": repr(float(training["temperature"])),
        "OAT_ZERO_TOP_P": repr(float(training["top_p"])),
        "OAT_ZERO_NUM_SAMPLES": str(evaluation["num_samples"]),
        "OAT_ZERO_TRAIN_BATCH_SIZE": str(training["train_batch_size"]),
        "OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE": str(
            evaluation["train_batch_size_per_device"]
        ),
        "OAT_ZERO_ROLLOUT_BATCH_SIZE": str(evaluation["rollout_batch_size"]),
        "OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE": str(
            evaluation["rollout_batch_size"]
        ),
        "OAT_ZERO_PI_BUFFER_MAXLEN_PER_DEVICE": str(
            training["pi_buffer_maxlen_per_device"]
        ),
        "OAT_ZERO_SYNC_PARAMS_EVERY": str(training["sync_params_every"]),
        "OAT_ZERO_ZERO_STAGE": str(evaluation["zero_stage"]),
        "OAT_ZERO_VLLM_GPU_RATIO": repr(float(evaluation["vllm_gpu_ratio"])),
        "OAT_ZERO_COLLOCATE": flag(evaluation["collocate"]),
        # --- evaluation cadence, inherited ----------------------------------
        "OAT_ZERO_EVAL_PROMPT_INTERVAL": str(training["eval_steps"]),
        "OAT_ZERO_EVAL_BATCH_SIZE": str(evaluation["eval_batch_size"]),
        "OAT_ZERO_EVAL_TEMPERATURE": repr(float(evaluation["eval_temperature"])),
        "OAT_ZERO_EVAL_MODE_COVERAGE_K": str(evaluation["eval_mode_coverage_k"]),
        "OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE": repr(
            float(evaluation["eval_mode_coverage_temperature"])
        ),
        "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS": str(
            evaluation["eval_mode_coverage_draws"]
        ),
        "OAT_ZERO_EVAL_MODE_COVERAGE_SEED": str(evaluation["eval_mode_coverage_seed"]),
        # --- objective, inherited from the reference arm ---------------------
        "OAT_ZERO_SEMANTIC_SHANNON_COEF": repr(
            float(objective["semantic_shannon_coef"])
        ),
        "OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP": repr(
            float(objective["semantic_shannon_surprisal_clip"])
        ),
        "OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT": repr(
            float(objective["semantic_shannon_pseudocount"])
        ),
        "OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_WARMUP_STEPS": str(
            objective["semantic_shannon_open_set_warmup_steps"]
        ),
        "OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_EMA_DECAY": repr(
            float(objective["semantic_shannon_open_set_ema_decay"])
        ),
        "OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA": repr(
            float(objective["online_canonical_bank_alpha"])
        ),
        "OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA": repr(
            float(objective["online_canonical_novelty_beta"])
        ),
        "OAT_ZERO_ONLINE_CANONICAL_BANK_PSEUDOCOUNT": repr(
            float(objective["online_canonical_bank_pseudocount"])
        ),
        "OAT_ZERO_ONLINE_CANONICAL_BANK_SURPRISAL_CLIP": repr(
            float(objective["online_canonical_bank_surprisal_clip"])
        ),
        "OAT_ZERO_ONLINE_CANONICAL_KEY_MODE": str(
            objective["online_canonical_key_mode"]
        ),
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA": repr(
            float(objective["online_canonical_replay_alpha"])
        ),
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY": str(
            objective["online_canonical_replay_capacity"]
        ),
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_WARMUP_STEPS": str(
            objective["online_canonical_replay_warmup_steps"]
        ),
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_EMA_DECAY": repr(
            float(objective["online_canonical_replay_ema_decay"])
        ),
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA": repr(
            float(objective["online_canonical_replay_mass_alpha"])
        ),
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_WARMUP_STEPS": str(
            objective["online_canonical_replay_mass_warmup_steps"]
        ),
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_EMA_DECAY": repr(
            float(objective["online_canonical_replay_mass_ema_decay"])
        ),
        "OAT_ZERO_VERIFIED_DISCOVERY_TRACKING": "1",
        # --- storage and recovery -------------------------------------------
        "OAT_ZERO_SAVE_STEPS": str(training["save_steps"]),
        "OAT_ZERO_SAVE_FROM": str(training["save_from"]),
        "OAT_ZERO_RESUME_STEPS": str(training["resume_steps"]),
        "OAT_ZERO_EXPORT_STEPS": str(training["export_steps"]),
        "OAT_ZERO_MAX_SAVE_NUM": "1",
        "OAT_ZERO_MAX_RESUME_NUM": "1",
        "OAT_ZERO_PRUNE_RESUME_ON_SUCCESS": "1",
        "OAT_ZERO_AUTO_RESUME": "1",
        "OAT_ZERO_WATCHDOG_REQUEUE": "1",
        "OAT_ZERO_USE_WB": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    if str(evaluation["canonical_action_task"]) != "none" or evaluation[
        "canonical_graph_actions"
    ]:
        env["VLLM_USE_V1"] = "0"
    # Applied last so an arm's removals cannot be silently overwritten by an
    # inherited value.
    env.update(spec["overrides"])
    return env


def submit(
    run: dict[str, Any],
    env: dict[str, str],
    *,
    arm: str,
    root: Path,
    partition: str,
    account: str,
    gres: str,
    cpus: int,
    memory: str,
    time_limit: str,
    hold: bool,
    dry_run: bool,
) -> str | None:
    export_pairs = ",".join(f"{key}={value}" for key, value in env.items())
    name = f"e72{arm}-{run['domain'][:6]}-s{run['seed']}"
    sbatch_args = [
        "sbatch",
        "--parsable",
        f"--job-name={name}",
        f"--export=ALL,{export_pairs}",
        f"--nodelist={run['source_node']}",
        f"--gres={gres}",
        f"--cpus-per-task={cpus}",
        f"--mem={memory}",
        f"--time={time_limit}",
        f"--partition={partition}",
        f"--account={account}",
    ]
    if hold:
        sbatch_args.append("--hold")
    sbatch_args.append(str(root / "ops" / "slurm" / "train_node302.slurm"))
    if dry_run:
        print(" ".join(shlex.quote(part) for part in sbatch_args))
        return None
    result = subprocess.run(
        sbatch_args, cwd=str(root), capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise SystemExit(f"sbatch failed for {name}: {result.stderr.strip()}")
    return result.stdout.strip()


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "var" / "artifacts" / "e72_frontier_source_runs.json",
    )
    parser.add_argument(
        "--arm",
        choices=sorted(ARM_SPECS),
        default="b3a",
        help="which remove-one arm to submit",
    )
    parser.add_argument("--domains", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--partition", default="mltheory")
    parser.add_argument("--account", default="mltheory")
    parser.add_argument("--gres", default="gpu:1")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--memory", default="64G")
    parser.add_argument("--time-limit", default="1-00:00:00")
    parser.add_argument("--hold", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    domains = {part for part in args.domains.split(",") if part}
    seeds = {int(part) for part in args.seeds.split(",") if part}

    references = [
        run
        for run in manifest["runs"]
        if run["arm"] == REFERENCE_ARM
        and (not domains or run["domain"] in domains)
        and (not seeds or int(run["seed"]) in seeds)
    ]
    problems: list[str] = []
    for run in references:
        problems.extend(check_inheritance(run))
    if problems:
        for problem in problems[:20]:
            print(f"[e72-b3a] {problem}")
        raise SystemExit(
            f"refusing to launch: {len(problems)} inheritance checks failed"
        )

    submitted: list[dict[str, Any]] = []
    for run in sorted(references, key=lambda item: (item["domain"], item["seed"])):
        target = save_path(root, run, args.arm)
        if args.skip_existing and (
            (target / "TRAINING_COMPLETE.json").is_file()
            or any(target.glob("debug_job*"))
        ):
            # An in-flight attempt counts as existing. Two jobs sharing one run
            # directory would train the same cell twice and leave the analysis
            # to choose between attempts.
            continue
        if args.limit and len(submitted) >= args.limit:
            break
        target.mkdir(parents=True, exist_ok=True)
        env = build_export_vars(root, run, target, args.arm)
        job_id = submit(
            run,
            env,
            arm=args.arm,
            root=root,
            partition=args.partition,
            account=args.account,
            gres=args.gres,
            cpus=args.cpus,
            memory=args.memory,
            time_limit=args.time_limit,
            hold=args.hold,
            dry_run=args.dry_run,
        )
        submitted.append(
            {
                "domain": run["domain"],
                "seed": run["seed"],
                "run_stamp": run_stamp(run, args.arm),
                "save_path": str(target),
                "node": run["source_node"],
                "reference_run": run["run_dir"],
                "slurm_job_id": job_id,
            }
        )

    print(
        f"[e72-{args.arm}] variant={ARM_SPECS[args.arm]['variant']} "
        f"removes={ARM_SPECS[args.arm]['removes']} submitted={len(submitted)} "
        f"checked={len(references)} dry_run={args.dry_run}"
    )
    if args.dry_run or not submitted:
        return 0

    ledger = root / "var" / "artifacts" / f"e72_{args.arm}_ablation_jobs.json"
    existing = (
        json.loads(ledger.read_text()).get("runs", []) if ledger.is_file() else []
    )
    payload = {
        "schema": "e72_b3a_replay_ablation_jobs_v1",
        "arm": args.arm,
        "variant": ARM_SPECS[args.arm]["variant"],
        "removes": ARM_SPECS[args.arm]["removes"],
        "overrides": ARM_SPECS[args.arm]["overrides"],
        "reference_arm": REFERENCE_ARM,
        "manifest": str(args.manifest),
        "expected_objective": {
            key: value for key, value in sorted(EXPECTED_OBJECTIVE.items())
        },
        "runs": existing + submitted,
    }
    handle, temporary = tempfile.mkstemp(prefix=f".{ledger.name}.", dir=ledger.parent)
    with os.fdopen(handle, "w", encoding="utf-8") as sink:
        json.dump(payload, sink, indent=2, sort_keys=True)
        sink.write("\n")
    os.replace(temporary, ledger)
    print(f"[e72-b3a] ledger {ledger} holds {len(payload['runs'])} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
