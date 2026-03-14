#!/usr/bin/env python3
"""Inspect a W&B triplet group and emit health + tuning recommendations."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import wandb


ENTITY_DEFAULT = "ogd3-princeton-university"
PROJECT_DEFAULT = "huggingface"

COMMON_FIELDS = (
    "model_name_or_path",
    "reference_model_name_or_path",
    "dataset_name",
    "eval_dataset_name",
    "learning_rate",
    "num_generations",
    "max_steps",
    "max_prompt_length",
    "max_completion_length",
    "beta",
    "warmup_ratio",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "eval_before_train",
    "eval_steps",
    "seed",
)

VARIANT_FIELDS = (
    "objective",
    "maxent_alpha",
    "maxent_tau",
    "maxent_policy_entropy",
    "maxent_policy_entropy_mode",
    "maxent_length_normalize_ref",
    "maxent_reference_ema_enabled",
)

SUMMARY_FIELDS = (
    "train/global_step",
    "train/kl",
    "train/reward",
    "train/reward_std",
    "train/frac_reward_zero_std",
    "train/loss/total",
    "train/loss/policy",
    "train/maxent/alpha",
    "train/tau",
    "train/beta",
    "train/maxent/listwise_weight_mean",
    "train/maxent/listwise_weight_std",
    "eval/reward",
    "eval/reward_std",
)

REQUIRED_VARIANTS = ("grpo", "maxent", "listwise")


@dataclass
class RunView:
    run_id: str
    name: str
    state: str
    url: str
    variant: str
    config: Dict[str, Any]
    variant_config: Dict[str, Any]
    summary: Dict[str, Any]


def _numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    return None


def _infer_variant(run: Any) -> str:
    name = str(getattr(run, "name", "") or "")
    summary = getattr(run, "summary", {})
    if summary.get("train/maxent/objective_variant_listwise") == 1:
        return "listwise"
    if summary.get("train/maxent/objective_variant_entropy") == 1:
        return "maxent"
    if name.endswith("-listwise"):
        return "listwise"
    if name.endswith("-maxent"):
        return "maxent"
    if name.endswith("-grpo"):
        return "grpo"
    return "unknown"


def _collect_run_view(run: Any) -> RunView:
    config = {field: run.config.get(field) for field in COMMON_FIELDS if field in run.config}
    variant_config = {field: run.config.get(field) for field in VARIANT_FIELDS if field in run.config}
    summary = {field: run.summary.get(field) for field in SUMMARY_FIELDS}
    return RunView(
        run_id=run.id,
        name=run.name,
        state=run.state,
        url=run.url,
        variant=_infer_variant(run),
        config=config,
        variant_config=variant_config,
        summary=summary,
    )


def _fairness_report(runs: Iterable[RunView]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    run_list = list(runs)
    for field in COMMON_FIELDS:
        values = {run.variant: run.config.get(field) for run in run_list}
        non_null = {k: v for k, v in values.items() if v is not None}
        grouped[field] = {
            "values": values,
            "fair": len(set(json.dumps(v, sort_keys=True) for v in non_null.values())) <= 1,
        }
    return grouped


def _health_flags(run: RunView, min_steps: int) -> List[str]:
    flags: List[str] = []
    step = _numeric(run.summary.get("train/global_step")) or 0.0
    if step < min_steps:
        flags.append("too_early")
        return flags

    raw_kl = run.summary.get("train/kl")
    if isinstance(raw_kl, str) and raw_kl.lower() == "nan":
        flags.append("nan_kl")
    kl = _numeric(raw_kl)
    if kl is not None and kl < 0.005:
        flags.append("low_kl")
    if kl is not None and kl > 1.0:
        flags.append("high_kl")

    zero_std = _numeric(run.summary.get("train/frac_reward_zero_std"))
    if zero_std is not None and zero_std > 0.5:
        flags.append("sparse_reward_signal")

    if run.variant == "listwise":
        weight_std = _numeric(run.summary.get("train/maxent/listwise_weight_std"))
        if weight_std is not None and weight_std < 0.08:
            flags.append("flat_listwise_weights")

    if run.summary.get("eval/reward") is None:
        flags.append("no_eval_yet")
    return flags


def _recommendations(runs: Iterable[RunView], min_steps: int) -> Dict[str, List[str]]:
    run_list = list(runs)
    flag_map = {run.variant: _health_flags(run, min_steps) for run in run_list}
    shared_cfg = _fairness_report(run_list)

    shared: List[str] = []
    maxent: List[str] = []
    listwise: List[str] = []

    mature_runs = [run for run in run_list if "too_early" not in flag_map[run.variant]]
    warmup_values = shared_cfg.get("warmup_ratio", {}).get("values", {})
    if (
        mature_runs
        and all("low_kl" in flag_map[run.variant] for run in mature_runs)
        and any((_numeric(value) or 0.0) > 0.05 for value in warmup_values.values())
    ):
        shared.append("--warmup_ratio 0.05")
    generation_values = shared_cfg.get("num_generations", {}).get("values", {})
    if (
        mature_runs
        and sum("sparse_reward_signal" in flag_map[run.variant] for run in mature_runs) >= 2
        and any((_numeric(value) or 0.0) < 8.0 for value in generation_values.values())
    ):
        shared.append("--num_generations 8")
    eval_before_train_values = shared_cfg.get("eval_before_train", {}).get("values", {})
    if (
        mature_runs
        and all("no_eval_yet" in flag_map[run.variant] for run in mature_runs)
        and any(value in (None, False, 0) for value in eval_before_train_values.values())
    ):
        shared.append("--eval_before_train true")
    listwise_run = next((run for run in run_list if run.variant == "listwise"), None)
    listwise_tau = _numeric((listwise_run.variant_config or {}).get("maxent_tau")) if listwise_run else None
    if "flat_listwise_weights" in flag_map.get("listwise", []) and (listwise_tau is None or listwise_tau > 0.2):
        listwise.append("--maxent_tau 0.2")
    maxent_run = next((run for run in run_list if run.variant == "maxent"), None)
    maxent_alpha = _numeric((maxent_run.variant_config or {}).get("maxent_alpha")) if maxent_run else None
    if maxent_run and any(flag in flag_map.get("maxent", []) for flag in ("high_kl", "nan_kl")):
        if maxent_alpha is None or maxent_alpha > 0.01:
            maxent.extend(
                [
                    "--maxent_alpha 0.01",
                    "--maxent_alpha_lower_on_high_kl true",
                    "--maxent_alpha_kl_threshold 0.04",
                    "--maxent_alpha_kl_min_multiplier 0.1",
                ]
            )

    # Keep the shared overrides explicit for the two maxent variants as well.
    maxent.extend(shared)
    listwise = shared + listwise

    return {
        "shared": shared,
        "grpo_args": shared,
        "maxent_args": maxent,
        "listwise_args": listwise,
    }


def _print_run(run: RunView, min_steps: int) -> None:
    print(f"{run.variant.upper():8} {run.name} [{run.state}]")
    print(f"  id={run.run_id} url={run.url}")
    print(f"  config={json.dumps(run.config, sort_keys=True)}")
    print(f"  variant_config={json.dumps(run.variant_config, sort_keys=True)}")
    print(f"  summary={json.dumps(run.summary, sort_keys=True)}")
    print(f"  flags={','.join(_health_flags(run, min_steps)) or 'none'}")


def _run_once(entity: str, project: str, group: str, min_steps: int) -> int:
    api = wandb.Api()
    runs = [
        _collect_run_view(api.run("/".join(run.path)))
        for run in api.runs(
            f"{entity}/{project}",
            filters={"group": group},
        )
    ]
    if not runs:
        print(f"No runs found for group {group}", file=sys.stderr)
        return 1

    runs.sort(key=lambda run: (run.variant, run.name))
    for run in runs:
        _print_run(run, min_steps)

    print("\nFAIRNESS")
    fairness = _fairness_report(runs)
    for field, payload in fairness.items():
        verdict = "OK" if payload["fair"] else "MISMATCH"
        print(f"  {field}: {verdict} {json.dumps(payload['values'], sort_keys=True)}")

    print("\nPRESENCE")
    observed = sorted({run.variant for run in runs})
    missing = [variant for variant in REQUIRED_VARIANTS if variant not in observed]
    print(f"  observed: {', '.join(observed) if observed else '(none)'}")
    print(f"  missing: {', '.join(missing) if missing else '(none)'}")

    recs = _recommendations(runs, min_steps)
    print("\nRECOMMENDED OVERRIDES")
    for key, values in recs.items():
        joined = " ".join(values) if values else "(none)"
        print(f"  {key}: {joined}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=ENTITY_DEFAULT)
    parser.add_argument("--project", default=PROJECT_DEFAULT)
    parser.add_argument("--group", required=True)
    parser.add_argument("--min-steps", type=int, default=5)
    parser.add_argument("--watch-interval", type=int, default=0)
    args = parser.parse_args()

    if args.watch_interval <= 0:
        return _run_once(args.entity, args.project, args.group, args.min_steps)

    while True:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"=== {stamp} ===")
        rc = _run_once(args.entity, args.project, args.group, args.min_steps)
        sys.stdout.flush()
        sys.stderr.flush()
        if rc == 0:
            time.sleep(args.watch_interval)
            continue
        time.sleep(min(args.watch_interval, 60))


if __name__ == "__main__":
    raise SystemExit(main())
