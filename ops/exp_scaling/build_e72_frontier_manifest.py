#!/usr/bin/env python3
"""Freeze the E72 decoding-frontier source inventory.

The frontier re-measures already-trained terminal checkpoints under new
decoding settings. Every cell must therefore inherit its domain's evaluation
configuration from the run that produced the checkpoint, not from a table typed
out a second time here: a silent divergence in prompt template, response
length, split, or draw seed would produce numbers that look like the published
ones but are not comparable to them.

This script reads, for each completed E70/E71 run:

* its ``TRAINING_COMPLETE.json`` terminal marker and exported checkpoint;
* the resolved argument block its learner printed at startup; and
* its published terminal metrics from the frozen scaling-curve artifact.

It then requires that the two arms within a domain agree on every inherited
evaluation field, and writes ``var/artifacts/e72_frontier_source_runs.json``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

MODEL_TAG = "qwen25_0p5b_instruct"

# (domain, run-stamp prefix, frozen scaling-curve artifact)
DOMAINS: tuple[tuple[str, str, str], ...] = (
    (
        "graph_coloring",
        "gce71_scale384_05b_12pass",
        "gce71_scale384_05b_12pass_scaling_curve.json",
    ),
    (
        "countdown",
        "cde70_clean_stage_a_05b_12pass",
        "cde70_clean_stage_a_05b_12pass_scaling_curve.json",
    ),
    (
        "python_factors",
        "pye70_clean_stage_a_05b_12pass",
        "pye70_clean_stage_a_05b_12pass_scaling_curve.json",
    ),
    (
        "mathir",
        "mie70_clean_stage_a_05b_12pass",
        "mie70_clean_stage_a_05b_12pass_scaling_curve.json",
    ),
    (
        "pantry_plan",
        "ppe71_scale384_05b_12pass",
        "ppe71_scale384_05b_12pass_scaling_curve.json",
    ),
)

# (analysis arm name, runtime variant tag, manifest label used in run stamps)
ARMS: tuple[tuple[str, str, str], ...] = (
    ("drgrpo", "grpo_compute_matched", "grpo"),
    (
        "xgrpo",
        "verified_first_global_replay_canonical",
        "verified_first_global_replay_canonical",
    ),
)

SEEDS: tuple[int, ...] = (43, 44, 45, 46, 47)

# Evaluation-relevant arguments inherited verbatim from the source run. A cell
# that changes any of these is measuring a different quantity.
INHERITED_EVAL_KEYS: tuple[str, ...] = (
    "prompt_template",
    "test_split",
    "prompt_data",
    "eval_data",
    "eval_input_key",
    "eval_output_key",
    "prompt_max_length",
    "generate_max_length",
    "eval_generate_max_length",
    "max_model_len",
    "eval_batch_size",
    "eval_temperature",
    "eval_mode_coverage_k",
    "eval_mode_coverage_temperature",
    "eval_mode_coverage_draws",
    "eval_mode_coverage_seed",
    "canonical_action_task",
    "canonical_graph_actions",
    "canonical_graph_action_count",
    "canonical_graph_learner_sampling",
    "canonical_graph_fixed_shape_sampling",
    "verifier_version",
    "num_samples",
    "rollout_batch_size",
    "train_batch_size_per_device",
    "max_train",
    "zero_stage",
    "vllm_gpu_ratio",
    "collocate",
    "pretrain",
)

# Fields that must match across the two arms of a domain. The arms are matched
# by protocol, so a disagreement means a parse error or a broken cohort rather
# than a legitimate difference.
ARM_INVARIANT_KEYS = tuple(key for key in INHERITED_EVAL_KEYS if key != "pretrain")

# Optimizer and schedule arguments. A follow-on arm that trains against this
# cohort must reproduce them exactly or it is not comparable to it.
INHERITED_TRAIN_KEYS: tuple[str, ...] = (
    "learning_rate",
    "num_prompt_epoch",
    "num_ppo_epochs",
    "max_queries",
    "max_norm",
    "beta",
    "temperature",
    "top_p",
    "train_batch_size",
    "pi_buffer_maxlen_per_device",
    "sync_params_every",
    "eval_steps",
    "save_steps",
    "save_from",
    "resume_steps",
    "export_steps",
    "seed",
)

# The treatment objective. B3a and later remove-one arms inherit these from the
# xGRPO run rather than restating a dose table that could drift from it.
INHERITED_OBJECTIVE_KEYS: tuple[str, ...] = (
    "semantic_shannon_coef",
    "semantic_shannon_surprisal_clip",
    "semantic_shannon_pseudocount",
    "semantic_shannon_separate_advantage",
    "semantic_shannon_success_conditioned_signed_advantage",
    "semantic_shannon_open_set_inverse_adaptation",
    "semantic_shannon_open_set_warmup_steps",
    "semantic_shannon_open_set_ema_decay",
    "online_canonical_bank_alpha",
    "online_canonical_novelty_beta",
    "online_canonical_bank_pseudocount",
    "online_canonical_bank_surprisal_clip",
    "online_canonical_key_mode",
    "online_canonical_replay",
    "online_canonical_replay_alpha",
    "online_canonical_replay_objective",
    "online_canonical_replay_capacity",
    "online_canonical_replay_global_groups_per_step",
    "online_canonical_replay_warmup_steps",
    "online_canonical_replay_ema_decay",
    "online_canonical_replay_mass_alpha",
    "online_canonical_replay_mass_warmup_steps",
    "online_canonical_replay_mass_ema_decay",
    "online_canonical_replay_compute_only",
    "policy_entropy_coef",
    "maxent_alpha",
)

ANSI = re.compile(r"\x1b\[[0-9;]*m")
ARG_LINE = re.compile(r"^\s*'(?P<key>[A-Za-z_][A-Za-z0-9_]*)':\s*(?P<value>.*?),?\s*$")
ATTEMPT_JOB = re.compile(r"debug_job(?P<job>\d+)$")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_learner_args(log_path: Path, *, max_lines: int = 6000) -> dict[str, Any]:
    """Recover the resolved argument block a learner printed at startup.

    Only the first block is read: a requeued attempt re-prints the same
    configuration, and the logs are large enough that scanning all of one is
    wasteful.
    """
    parsed: dict[str, Any] = {}
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for index, raw in enumerate(handle):
            if index >= max_lines:
                break
            line = ANSI.sub("", raw).rstrip("\n")
            marker = line.find("│")
            if marker < 0:
                continue
            match = ARG_LINE.match(line[marker + 1 :])
            if match is None:
                continue
            key = match.group("key")
            if key in parsed:
                # Later blocks belong to other components (actors, oracles);
                # the learner's own value is the first one printed.
                continue
            try:
                parsed[key] = ast.literal_eval(match.group("value"))
            except (ValueError, SyntaxError):
                # Values such as ``<RLAlgo.PPO: 100>`` are not literals. None of
                # the inherited evaluation fields are of that kind.
                continue
    return parsed


def source_nodes(job_ids: list[str]) -> dict[str, str]:
    """Map each source job to the node it ran on.

    GPU type changes floating-point reduction order, which changes sampled
    tokens: the frozen pass-0 rows of a single domain already differ between
    the a5000 and a100 halves of the cohort even though the weights and draw
    seeds are identical. A frontier cell must therefore be measured on the node
    that produced its checkpoint, or hardware would enter the very comparison
    the sweep exists to make.
    """
    if not job_ids:
        return {}
    result = subprocess.run(
        ["sacct", "-j", ",".join(job_ids), "-X", "--format=JobID,NodeList%20", "-n"],
        capture_output=True,
        text=True,
        check=False,
    )
    placement: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] not in {"None", "None assigned"}:
            placement[parts[0]] = parts[1]
    return placement


def terminal_curve_rows(curve_path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = json.loads(curve_path.read_text())
    multi = [row for row in rows if row.get("split") == "multi_answer"]
    if not multi:
        raise SystemExit(f"{curve_path} has no multi_answer rows")
    terminal_step = max(int(row["step"]) for row in multi)
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in multi:
        if int(row["step"]) != terminal_step:
            continue
        out[(str(row["arm"]), int(row["seed"]))] = row
    return out


def export_identity(export_dir: Path, *, hash_weights: bool) -> dict[str, Any]:
    weights = sorted(export_dir.glob("*.safetensors")) or sorted(
        export_dir.glob("*.bin")
    )
    if not weights:
        raise SystemExit(f"terminal export has no weight file: {export_dir}")
    identity: dict[str, Any] = {
        "path": str(export_dir),
        "weight_files": [path.name for path in weights],
        "weight_bytes": sum(path.stat().st_size for path in weights),
        "config_sha256": hashlib.sha256(
            (export_dir / "config.json").read_bytes()
        ).hexdigest(),
    }
    if hash_weights:
        digest = hashlib.sha256()
        for path in weights:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 22), b""):
                    digest.update(chunk)
        identity["weights_sha256"] = digest.hexdigest()
    return identity


def base_reference_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add the pre-RL policy as a measurable arm, once per domain and GPU model.

    The base model is what both arms started from, so sweeping it answers a
    question the trained arms cannot: whether RL pushed sampled support *below*
    initialization, or merely failed to raise it. Its published reference is the
    cohort's own frozen pass-0 row, which every run recorded before its first
    optimizer step.

    One entry per (domain, GPU model) rather than per domain: pass-0 rows differ
    between GPU models, so a single base curve would silently average two
    different measurements.
    """
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for run in runs:
        key = (run["domain"], str(run["source_node"]))
        grouped.setdefault(key, run)

    base: list[dict[str, Any]] = []
    for (domain, node), representative in sorted(grouped.items()):
        pretrain = str(representative["inherited_eval_config"]["pretrain"])
        if not Path(pretrain, "config.json").is_file():
            continue
        base.append(
            {
                "domain": domain,
                "arm": f"base_{node}",
                "curve_arm_label": None,
                "runtime_variant": "pretrained_initialization",
                "seed": 0,
                "run_stamp": f"base_{node}",
                "run_dir": None,
                "terminal_attempt": None,
                "terminal_step": 0,
                "slurm_job_id": None,
                "learner_log": representative["learner_log"],
                "source_node": node,
                "export": {
                    "path": pretrain,
                    "weight_files": [],
                    "weight_bytes": 0,
                    "config_sha256": hashlib.sha256(
                        Path(pretrain, "config.json").read_bytes()
                    ).hexdigest(),
                },
                "inherited_eval_config": representative["inherited_eval_config"],
                "inherited_train_config": representative["inherited_train_config"],
                "inherited_objective_config": representative[
                    "inherited_objective_config"
                ],
                # The pass-0 reference is supplied by the aggregator from the
                # frozen curve; a base cell has no terminal row of its own.
                "published_terminal": None,
                "curve_artifact": representative["curve_artifact"],
            }
        )
    return base


def build(*, hash_weights: bool, include_base: bool = True) -> dict[str, Any]:
    root = repo_root()
    data_root = root / "var" / "data"
    log_root = root / "var" / "artifacts" / "logs"
    artifact_root = root / "var" / "artifacts"

    runs: list[dict[str, Any]] = []
    problems: list[str] = []

    for domain, prefix, curve_name in DOMAINS:
        curve_path = artifact_root / curve_name
        if not curve_path.is_file():
            problems.append(f"{domain}: missing scaling curve {curve_path}")
            continue
        published = terminal_curve_rows(curve_path)
        domain_inherited: dict[str, dict[str, Any]] = {}

        for arm, variant, label in ARMS:
            for seed in SEEDS:
                stamp = f"{prefix}_{label}_s{seed}"
                run_dir = data_root / f"xdr_{MODEL_TAG}_{variant}_{stamp}"
                cell = f"{domain}/{arm}/s{seed}"
                if not run_dir.is_dir():
                    problems.append(f"{cell}: missing run dir {run_dir}")
                    continue
                marker_path = run_dir / "TRAINING_COMPLETE.json"
                if not marker_path.is_file():
                    problems.append(f"{cell}: run is not terminal ({marker_path})")
                    continue
                marker = json.loads(marker_path.read_text())
                export_dir = Path(str(marker["terminal_export"]))
                if not (export_dir / "config.json").is_file():
                    problems.append(f"{cell}: missing terminal export {export_dir}")
                    continue

                attempt = Path(str(marker["terminal_attempt"]))
                job_match = ATTEMPT_JOB.search(attempt.name)
                if job_match is None:
                    problems.append(f"{cell}: cannot infer job id from {attempt.name}")
                    continue
                job_id = job_match.group("job")
                log_path = log_root / f"xdr_train-{job_id}.out"
                if not log_path.is_file():
                    problems.append(f"{cell}: missing learner log {log_path}")
                    continue

                learner_args = parse_learner_args(log_path)
                inherited = {}
                training = {}
                objective = {}
                missing = []
                for key in INHERITED_EVAL_KEYS:
                    if key not in learner_args:
                        missing.append(key)
                    else:
                        inherited[key] = learner_args[key]
                for key in INHERITED_TRAIN_KEYS:
                    if key not in learner_args:
                        missing.append(key)
                    else:
                        training[key] = learner_args[key]
                for key in INHERITED_OBJECTIVE_KEYS:
                    if key not in learner_args:
                        missing.append(key)
                    else:
                        objective[key] = learner_args[key]
                if missing:
                    problems.append(f"{cell}: log lacks {', '.join(missing)}")
                    continue

                published_row = published.get((label, seed))
                if published_row is None:
                    problems.append(f"{cell}: no terminal row in {curve_name}")
                    continue

                comparable = {
                    key: inherited[key] for key in ARM_INVARIANT_KEYS
                }
                domain_inherited.setdefault(json.dumps(comparable, sort_keys=True), {})[
                    cell
                ] = True

                runs.append(
                    {
                        "domain": domain,
                        "arm": arm,
                        "curve_arm_label": label,
                        "runtime_variant": variant,
                        "seed": seed,
                        "run_stamp": stamp,
                        "run_dir": str(run_dir),
                        "terminal_attempt": str(attempt),
                        "terminal_step": int(marker["terminal_step"]),
                        "slurm_job_id": job_id,
                        "learner_log": str(log_path),
                        "export": export_identity(
                            export_dir, hash_weights=hash_weights
                        ),
                        "inherited_eval_config": inherited,
                        "inherited_train_config": training,
                        "inherited_objective_config": objective,
                        "published_terminal": {
                            "step": int(published_row["step"]),
                            "greedy": published_row.get("greedy"),
                            "mean8": published_row.get("mean8"),
                            "pass8": published_row.get("pass8"),
                            "distinct8": published_row.get("distinct8"),
                            "mean8_draw_se": published_row.get("mean8_draw_se"),
                            "pass8_draw_se": published_row.get("pass8_draw_se"),
                            "distinct8_draw_se": published_row.get("distinct8_draw_se"),
                        },
                        "curve_artifact": curve_name,
                    }
                )

        if len(domain_inherited) > 1:
            problems.append(
                f"{domain}: arms disagree on inherited evaluation config across "
                f"{len(domain_inherited)} distinct settings"
            )

    placement = source_nodes([run["slurm_job_id"] for run in runs])
    unplaced = []
    for run in runs:
        node = placement.get(run["slurm_job_id"])
        run["source_node"] = node
        if not node:
            unplaced.append(f"{run['domain']}/{run['arm']}/s{run['seed']}")
    if unplaced:
        problems.append(
            "cannot resolve the source node for: " + ", ".join(sorted(unplaced))
        )

    base_runs = base_reference_runs(runs) if include_base and not unplaced else []
    runs.extend(base_runs)

    trained_runs = [run for run in runs if not str(run["arm"]).startswith("base_")]
    payload = {
        "schema": "e72_frontier_source_runs_v3",
        "model_tag": MODEL_TAG,
        "domains": [domain for domain, _, _ in DOMAINS],
        "arms": [arm for arm, _, _ in ARMS],
        "seeds": list(SEEDS),
        "expected_runs": len(DOMAINS) * len(ARMS) * len(SEEDS),
        "resolved_runs": len(trained_runs),
        "base_reference_cells": len(runs) - len(trained_runs),
        "weights_hashed": bool(hash_weights),
        "problems": problems,
        "runs": runs,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root() / "var" / "artifacts" / "e72_frontier_source_runs.json",
    )
    parser.add_argument(
        "--hash-weights",
        action="store_true",
        help="also SHA-256 every exported weight file (slow; ~1.3 GB per run)",
    )
    parser.add_argument(
        "--no-base",
        action="store_true",
        help="omit the pre-RL base-model reference entries",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write the manifest even when some cells could not be resolved",
    )
    args = parser.parse_args()

    payload = build(hash_weights=args.hash_weights, include_base=not args.no_base)
    complete = payload["resolved_runs"] == payload["expected_runs"] and not payload[
        "problems"
    ]
    if not complete and not args.allow_incomplete:
        for problem in payload["problems"]:
            print(f"[e72-manifest] problem: {problem}")
        print(
            "[e72-manifest] resolved "
            f"{payload['resolved_runs']}/{payload['expected_runs']} runs; refusing to "
            "write an incomplete manifest (use --allow-incomplete to override)"
        )
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    with os.fdopen(handle, "w", encoding="utf-8") as sink:
        json.dump(payload, sink, indent=2, sort_keys=True)
        sink.write("\n")
    os.replace(temporary, args.output)
    print(
        f"[e72-manifest] wrote {args.output} "
        f"({payload['resolved_runs']}/{payload['expected_runs']} runs, "
        f"{len(payload['problems'])} problems)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
