#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from oat_drgrpo.passk_eval import (  # noqa: E402
    CheckpointSpec,
    _flatten_checkpoint_metric_row,
    _list_local_checkpoint_steps,
    _list_remote_checkpoint_steps,
    _parse_int_list,
    _parse_seed_list,
    _parse_template_list,
    _resolve_checkpoints,
    _safe_json_dump,
    _safe_repo_name,
    _step_to_int,
    _template_output_root,
    _utc_now,
    main as passk_eval_main,
)


DEFAULT_MODEL_FAMILIES: tuple[str, ...] = ("drgrpo", "drx_grpo")
DEFAULT_MODEL_SEEDS: tuple[int, ...] = (42, 43, 44)
DEFAULT_TEMPLATES: tuple[str, ...] = ("no", "qwen_math", "r1")
DEFAULT_EVAL_SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5)
DEFAULT_MODEL_RUN_ORDER = "seed_then_template_then_family"

# The math_12k train split is consumed in ~94 prompt batches at train_batch_size=128.
# With save/eval every 16 steps, this resolves epoch boundaries to step_00096,
# step_00192, and then the last checkpoint at or before the epoch-3 boundary
# (typically step_00272 when step_00280 is not present).
DEFAULT_PROMPT_BATCHES_PER_EPOCH = 94
DEFAULT_INFER_EPOCH_CHECKPOINTS = 3


@dataclass(frozen=True)
class ModelSpec:
    family: str
    family_display: str
    model_seed: int
    repo_id: str
    local_source_root: str = ""


def _default_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            family="drgrpo",
            family_display="dr.GRPO",
            model_seed=42,
            repo_id="od2961/qwen2.5-Math-1.5b-drgrpo-readmeflash-a6000-5epoch-seed42",
            local_source_root=str(
                ROOT_DIR / "var/data/oat_zero_exact_1p5b_20260412_031159_seed42"
            ),
        ),
        ModelSpec(
            family="drgrpo",
            family_display="dr.GRPO",
            model_seed=43,
            repo_id="od2961/qwen2.5-Math-1.5b-drgrpo-readmeflash-a6000-5epoch-seed43",
            local_source_root=str(
                ROOT_DIR / "var/data/oat_zero_exact_1p5b_20260412_160206_seed43"
            ),
        ),
        ModelSpec(
            family="drgrpo",
            family_display="dr.GRPO",
            model_seed=44,
            repo_id="od2961/qwen2.5-Math-1.5b-drgrpo-readmeflash-a6000-5epoch-seed44",
            local_source_root=str(
                ROOT_DIR / "var/data/oat_zero_exact_1p5b_20260412_210207_seed44"
            ),
        ),
        ModelSpec(
            family="drx_grpo",
            family_display="Dr.X-GRPO",
            model_seed=42,
            repo_id="od2961/qwen2.5-Math-1.5b-drx-grpo-readmeflash-node302-seed42",
            local_source_root=str(
                ROOT_DIR / "var/data/oat_zero_drx_grpo_1p5b_20260415_162700_seed42"
            ),
        ),
        ModelSpec(
            family="drx_grpo",
            family_display="Dr.X-GRPO",
            model_seed=43,
            repo_id="od2961/qwen2.5-Math-1.5b-drx-grpo-readmeflash-node302-seed43",
            local_source_root=str(
                ROOT_DIR / "var/data/oat_zero_drx_grpo_1p5b_20260416_100347_seed43"
            ),
        ),
        ModelSpec(
            family="drx_grpo",
            family_display="Dr.X-GRPO",
            model_seed=44,
            repo_id="od2961/qwen2.5-Math-1.5b-drx-grpo-readmeflash-node302-seed44",
            local_source_root=str(
                ROOT_DIR / "var/data/oat_zero_drx_grpo_1p5b_20260416_163518_seed44"
            ),
        ),
    ]


def _reorder_model_specs(
    specs: Sequence[ModelSpec],
    *,
    order: str,
) -> list[ModelSpec]:
    indexed_specs = list(enumerate(specs))
    if order == "family_then_seed":
        return [spec for _, spec in indexed_specs]
    if order == "seed_then_family":
        return [
            spec
            for _, spec in sorted(
                indexed_specs,
                key=lambda item: (int(item[1].model_seed), int(item[0])),
            )
        ]
    if order == "seed_then_template_then_family":
        return [
            spec
            for _, spec in sorted(
                indexed_specs,
                key=lambda item: (int(item[1].model_seed), int(item[0])),
            )
        ]
    raise ValueError(f"Unsupported model run order: {order}")


def _discover_local_saved_model_roots(source_root: Path) -> list[Path]:
    roots: list[Path] = []
    primary = source_root / "saved_models"
    if primary.is_dir():
        roots.append(primary.resolve())

    child_roots: list[tuple[float, Path]] = []
    for child in source_root.iterdir():
        candidate = child / "saved_models"
        if child.is_dir() and candidate.is_dir():
            child_roots.append((candidate.stat().st_mtime, candidate.resolve()))
    for _, path in sorted(child_roots):
        roots.append(path)
    return roots


def _stage_local_checkpoint_root(
    *,
    spec: ModelSpec,
    staging_root: Path,
    cache_root: Path,
) -> tuple[Path | None, dict[str, Any]]:
    if not spec.local_source_root:
        return None, {
            "has_local_root": False,
            "staged_root": "",
            "sources": [],
            "steps": {},
        }

    source_root = Path(spec.local_source_root).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Local source root not found: {source_root}")

    merged_root = staging_root / spec.family / f"seed{spec.model_seed}"
    saved_models_root = merged_root / "saved_models"
    saved_models_root.mkdir(parents=True, exist_ok=True)

    source_dirs = _discover_local_saved_model_roots(source_root)
    chosen_steps: dict[str, Path] = {}
    for source_dir in source_dirs:
        for child in sorted(source_dir.iterdir()):
            if (
                child.is_dir()
                and child.name.startswith("step_")
                and (child / "config.json").is_file()
            ):
                chosen_steps[child.name] = child.resolve()

    # Reuse already-cached HF checkpoints when local training roots do not expose
    # every requested snapshot directly. This keeps sweep planning purely local
    # while still preferring the original local training artifacts when present.
    cached_repo_root = cache_root / _safe_repo_name(spec.repo_id)
    if cached_repo_root.is_dir():
        for cached_step_root in sorted(cached_repo_root.iterdir()):
            if not (
                cached_step_root.is_dir() and cached_step_root.name.startswith("step_")
            ):
                continue
            cached_step_dir = cached_step_root / "checkpoints" / cached_step_root.name
            if not (cached_step_dir / "config.json").is_file():
                continue
            chosen_steps.setdefault(cached_step_root.name, cached_step_dir.resolve())

    for step_name, source_path in chosen_steps.items():
        destination = saved_models_root / step_name
        if destination.is_symlink() or destination.exists():
            try:
                if destination.resolve() == source_path:
                    continue
            except Exception:
                pass
            if destination.is_symlink() or destination.is_file():
                destination.unlink()
            else:
                raise RuntimeError(
                    f"Refusing to replace non-symlink directory at {destination}"
                )
        destination.symlink_to(source_path, target_is_directory=True)

    manifest = {
        "generated_at_utc": _utc_now(),
        "family": spec.family,
        "family_display": spec.family_display,
        "model_seed": spec.model_seed,
        "local_source_root": str(source_root),
        "has_local_root": True,
        "staged_root": str(merged_root),
        "sources": [str(path) for path in source_dirs],
        "steps": {
            step_name: str(path) for step_name, path in sorted(chosen_steps.items())
        },
    }
    _safe_json_dump(merged_root / "local_root_manifest.json", manifest)
    return merged_root, manifest


def _logical_checkpoint_targets(
    infer_epoch_checkpoints: int,
    prompt_batches_per_epoch: int,
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = [
        {"alias": "pretrain", "target_step": 0, "selection_mode": "exact"},
    ]
    for epoch_index in range(1, infer_epoch_checkpoints + 1):
        selection_mode = (
            "last_at_or_before"
            if epoch_index == infer_epoch_checkpoints
            else "first_at_or_after"
        )
        targets.append(
            {
                "alias": f"epoch{epoch_index}",
                "target_step": int(prompt_batches_per_epoch) * epoch_index,
                "selection_mode": selection_mode,
            }
        )
    return targets


def _select_step_for_target(
    *,
    available_steps: Sequence[int],
    target_step: int,
    selection_mode: str,
) -> int:
    unique_steps = sorted(set(int(step) for step in available_steps))
    if not unique_steps:
        raise ValueError("No available steps to select from.")
    if selection_mode == "exact":
        if 0 not in unique_steps:
            raise ValueError("Step 0 is required for pretrain selection.")
        return 0
    positive_steps = [step for step in unique_steps if step > 0]
    if not positive_steps:
        raise ValueError("No positive checkpoint steps available.")
    if selection_mode == "first_at_or_after":
        return next(
            (step for step in positive_steps if step >= target_step), positive_steps[-1]
        )
    if selection_mode == "last_at_or_before":
        return next(
            (step for step in reversed(positive_steps) if step <= target_step),
            positive_steps[0],
        )
    raise ValueError(f"Unknown selection mode: {selection_mode}")


def _resolve_model_checkpoints(
    *,
    spec: ModelSpec,
    cache_root: Path,
    local_checkpoint_root: Path | None,
    infer_epoch_checkpoints: int,
    prompt_batches_per_epoch: int,
    force_download: bool,
    local_files_only: bool,
) -> tuple[list[CheckpointSpec], list[dict[str, Any]]]:
    target_rows = _logical_checkpoint_targets(
        infer_epoch_checkpoints=infer_epoch_checkpoints,
        prompt_batches_per_epoch=prompt_batches_per_epoch,
    )
    local_step_set: set[str] = set()
    if local_checkpoint_root is not None and local_checkpoint_root.is_dir():
        local_step_set = set(_list_local_checkpoint_steps(local_checkpoint_root))
    if local_step_set:
        available_steps = [
            _step_to_int(step_name)
            for step_name in sorted(local_step_set, key=_step_to_int)
        ]
    else:
        if not spec.repo_id:
            raise ValueError(
                "Checkpoint resolution requires either local checkpoints or a repo id."
            )
        remote_steps = _list_remote_checkpoint_steps(
            spec.repo_id,
            token=os.environ.get("HF_TOKEN", "") or None,
        )
        available_steps = [_step_to_int(step_name) for step_name in remote_steps]

    explicit_specs: list[str] = []
    targets_by_alias: dict[str, dict[str, Any]] = {}
    for item in target_rows:
        alias = str(item["alias"])
        target_step = int(item["target_step"])
        selection_mode = str(item["selection_mode"])
        chosen_step = _select_step_for_target(
            available_steps=available_steps,
            target_step=target_step,
            selection_mode=selection_mode,
        )
        step_name = f"step_{chosen_step:05d}"
        explicit_specs.append(f"{alias}={step_name}")
        targets_by_alias[alias] = {
            "target_step": target_step,
            "selection_mode": selection_mode,
        }

    resolved = _resolve_checkpoints(
        checkpoint_specs=explicit_specs,
        repo_id=spec.repo_id,
        cache_root=cache_root,
        local_checkpoint_root=local_checkpoint_root,
        infer_epoch_checkpoints=None,
        prompt_batches_per_epoch=None,
        token=os.environ.get("HF_TOKEN", "") or None,
        force_download=force_download,
        local_files_only=local_files_only,
    )
    resolution_rows: list[dict[str, Any]] = []
    for checkpoint in resolved:
        resolved_step = int(checkpoint.step_number or 0)
        target_meta = targets_by_alias.get(
            checkpoint.alias,
            {"target_step": resolved_step, "selection_mode": "exact"},
        )
        target_step = int(target_meta["target_step"])
        selection_mode = str(target_meta["selection_mode"])
        resolution_rows.append(
            {
                "alias": checkpoint.alias,
                "target_step": target_step,
                "resolved_step": resolved_step,
                "resolved_step_name": checkpoint.step_name,
                "source": checkpoint.source,
                "local_path": checkpoint.local_path,
                "selection_mode": selection_mode,
                "fell_back_before_target": bool(
                    checkpoint.alias != "pretrain"
                    and selection_mode == "first_at_or_after"
                    and resolved_step < target_step
                ),
            }
        )
    return resolved, resolution_rows


def _collect_metric_rows(
    *,
    spec: ModelSpec,
    model_output_root: Path,
    templates: Sequence[str],
    pass_ks: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    trace_index_rows: list[dict[str, Any]] = []
    template_status_rows: list[dict[str, Any]] = []
    multi_template = len(templates) > 1
    for template in templates:
        template_root = _template_output_root(
            model_output_root,
            template,
            multi_template=multi_template,
        )
        status_path = template_root / "run_status.json"
        summary_path = template_root / "combined_checkpoint_summary.json"
        status_payload = (
            json.loads(status_path.read_text()) if status_path.is_file() else {}
        )
        summaries = (
            json.loads(summary_path.read_text()) if summary_path.is_file() else []
        )
        template_status_rows.append(
            {
                "family": spec.family,
                "family_display": spec.family_display,
                "model_seed": spec.model_seed,
                "template": template,
                "output_root": str(template_root),
                "run_status_path": str(status_path),
                "all_complete": bool(status_payload.get("all_complete")),
            }
        )
        for summary in summaries:
            row = {
                "family": spec.family,
                "family_display": spec.family_display,
                "model_seed": spec.model_seed,
                "template": template,
                "checkpoint_output_root": str(
                    template_root / "checkpoints" / str(summary["checkpoint_alias"])
                ),
            }
            row.update(_flatten_checkpoint_metric_row(summary, pass_ks=pass_ks))
            metric_rows.append(row)
            for task_name, path_items in (summary.get("source_paths") or {}).items():
                for path_item in path_items:
                    trace_index_rows.append(
                        {
                            "family": spec.family,
                            "family_display": spec.family_display,
                            "model_seed": spec.model_seed,
                            "template": template,
                            "checkpoint_alias": summary["checkpoint_alias"],
                            "checkpoint_step": summary.get("checkpoint_step"),
                            "task": task_name,
                            "summary_path": path_item.get("summary_path", ""),
                            "attempts_path": path_item.get("attempts_path", ""),
                            "weight": path_item.get("weight"),
                        }
                    )
    return metric_rows, trace_index_rows, template_status_rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_sweep_status(
    *,
    output_root: Path,
    run_specs: Sequence[dict[str, Any]],
    metric_rows: Sequence[dict[str, Any]],
    trace_index_rows: Sequence[dict[str, Any]],
    template_status_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "generated_at_utc": _utc_now(),
        "output_root": str(output_root),
        "completed_model_runs": int(
            sum(1 for item in run_specs if bool(item.get("all_complete")))
        ),
        "total_model_runs": int(len(run_specs)),
        "all_complete": bool(run_specs)
        and all(bool(item.get("all_complete")) for item in run_specs),
        "model_runs": list(run_specs),
        "metric_summary_path": str(output_root / "combined_metric_summary.json"),
        "trace_file_index_path": str(output_root / "trace_file_index.json"),
    }
    _safe_json_dump(output_root / "combined_metric_summary.json", list(metric_rows))
    _safe_json_dump(output_root / "trace_file_index.json", list(trace_index_rows))
    _safe_json_dump(output_root / "template_statuses.json", list(template_status_rows))
    _write_csv(output_root / "combined_metric_summary.csv", metric_rows)
    _write_csv(output_root / "trace_file_index.csv", trace_index_rows)
    _write_csv(output_root / "template_statuses.csv", template_status_rows)
    _safe_json_dump(output_root / "run_status.json", payload)
    return payload


def _build_passk_argv(
    *,
    spec: ModelSpec,
    resolved_checkpoints: Sequence[CheckpointSpec],
    local_checkpoint_root: Path | None,
    model_output_root: Path,
    args: argparse.Namespace,
    templates_override: Sequence[str] | None = None,
    output_root_override: Path | None = None,
) -> list[str]:
    selected_templates = (
        list(templates_override)
        if templates_override is not None
        else _parse_template_list(args.templates)
    )
    selected_output_root = (
        output_root_override if output_root_override is not None else model_output_root
    )
    argv = [
        f"--repo-id={spec.repo_id}",
        f"--output-root={selected_output_root}",
        f"--dataset-root={Path(args.dataset_root).resolve()}",
        f"--checkpoint-cache-root={Path(args.checkpoint_cache_root).resolve()}",
        f"--templates={','.join(str(item) for item in selected_templates)}",
        f"--pass-ks={args.pass_ks}",
        f"--mean-k={int(args.mean_k)}",
        f"--sample-count={int(args.sample_count)}",
        f"--sample-temperature={float(args.sample_temperature)}",
        f"--sample-top-p={float(args.sample_top_p)}",
        f"--max-tokens={int(args.max_tokens)}",
        f"--max-model-len={int(args.max_model_len)}",
        f"--greedy-batch-size={int(args.greedy_batch_size)}",
        f"--sampled-batch-size={int(args.sampled_batch_size)}",
        f"--gpu-memory-utilization={float(args.gpu_memory_utilization)}",
        f"--swap-space={float(args.swap_space)}",
        f"--inference-seeds={args.inference_seeds}",
        f"--max-workers={int(args.max_workers)}",
        f"--semantic-similarity-threshold={float(args.semantic_similarity_threshold)}",
    ]
    for checkpoint in resolved_checkpoints:
        if not checkpoint.step_name:
            raise ValueError(
                f"Resolved checkpoint for alias '{checkpoint.alias}' is missing step_name."
            )
        argv.append(f"--checkpoint={checkpoint.alias}={checkpoint.step_name}")
    if local_checkpoint_root is not None:
        argv.append(f"--local-checkpoint-root={local_checkpoint_root}")
    if args.include_token_ids:
        argv.append("--include-token-ids")
    if args.force_download:
        argv.append("--force-download")
    if args.local_files_only:
        argv.append("--local-files-only")
    argv.append("--skip-existing" if args.skip_existing else "--no-skip-existing")
    argv.append("--use-wandb" if args.use_wandb else "--no-use-wandb")
    if args.use_wandb:
        argv.extend(
            [
                f"--wandb-project={args.wandb_project}",
                f"--wandb-run-name={spec.family}-seed{spec.model_seed}-passk",
                f"--wandb-artifact-name={spec.family}-seed{spec.model_seed}-passk",
            ]
        )
        if str(args.wandb_entity).strip():
            argv.append(f"--wandb-entity={args.wandb_entity}")
    return argv


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a resumable node302 8-GPU pass@k evaluation sweep across "
            "dr.GRPO and Dr.X-GRPO model seeds."
        )
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-families", default=",".join(DEFAULT_MODEL_FAMILIES))
    parser.add_argument(
        "--model-seeds",
        default=",".join(str(seed) for seed in DEFAULT_MODEL_SEEDS),
    )
    parser.add_argument("--templates", default=",".join(DEFAULT_TEMPLATES))
    parser.add_argument(
        "--model-run-order",
        choices=(
            "family_then_seed",
            "seed_then_family",
            "seed_then_template_then_family",
        ),
        default=DEFAULT_MODEL_RUN_ORDER,
    )
    parser.add_argument(
        "--dataset-root", default=str(ROOT_DIR / "datasets/evaluation_suite")
    )
    parser.add_argument(
        "--checkpoint-cache-root",
        default=str(ROOT_DIR / "var/cache/passk_eval_checkpoints"),
    )
    parser.add_argument(
        "--infer-epoch-checkpoints",
        type=int,
        default=DEFAULT_INFER_EPOCH_CHECKPOINTS,
    )
    parser.add_argument(
        "--prompt-batches-per-epoch",
        type=int,
        default=DEFAULT_PROMPT_BATCHES_PER_EPOCH,
    )
    parser.add_argument("--pass-ks", default="1,8")
    parser.add_argument("--mean-k", type=int, default=8)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--inference-seeds", default="1,2,3,4,5")
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--sample-top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--greedy-batch-size", type=int, default=8)
    parser.add_argument("--sampled-batch-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--swap-space", type=float, default=32.0)
    parser.add_argument("--semantic-similarity-threshold", type=float, default=0.75)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--include-token-ids", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--wandb-project", default="oat-zero")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _refresh_sweep_outputs(
    *,
    output_root: Path,
    run_specs: Sequence[dict[str, Any]],
    prepared_specs: Sequence[dict[str, Any]],
    templates: Sequence[str],
    pass_ks: Sequence[int],
) -> None:
    all_metric_rows: list[dict[str, Any]] = []
    all_trace_index_rows: list[dict[str, Any]] = []
    all_template_status_rows: list[dict[str, Any]] = []
    run_spec_map = {
        (str(item["family"]), int(item["model_seed"])): item for item in run_specs
    }
    for prepared in prepared_specs:
        spec = prepared["spec"]
        model_output_root = prepared["model_output_root"]
        metric_rows, trace_index_rows, template_status_rows = _collect_metric_rows(
            spec=spec,
            model_output_root=model_output_root,
            templates=templates,
            pass_ks=pass_ks,
        )
        all_metric_rows.extend(metric_rows)
        all_trace_index_rows.extend(trace_index_rows)
        all_template_status_rows.extend(template_status_rows)
        run_spec = run_spec_map[(str(spec.family), int(spec.model_seed))]
        matching_statuses = [
            item
            for item in template_status_rows
            if item["family"] == spec.family
            and int(item["model_seed"]) == spec.model_seed
        ]
        run_spec["all_complete"] = bool(matching_statuses) and all(
            bool(item.get("all_complete")) for item in matching_statuses
        )
    _write_sweep_status(
        output_root=output_root,
        run_specs=run_specs,
        metric_rows=all_metric_rows,
        trace_index_rows=all_trace_index_rows,
        template_status_rows=all_template_status_rows,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.checkpoint_cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    staging_root = output_root / "_prepared_local_roots"
    staging_root.mkdir(parents=True, exist_ok=True)

    templates = _parse_template_list(args.templates)
    pass_ks = _parse_int_list(args.pass_ks)
    model_seeds = set(_parse_seed_list(args.model_seeds))
    allowed_families = {
        part.strip() for part in str(args.model_families).split(",") if part.strip()
    }

    selected_specs = [
        spec
        for spec in _default_model_specs()
        if spec.family in allowed_families and spec.model_seed in model_seeds
    ]
    selected_specs = _reorder_model_specs(
        selected_specs,
        order=str(args.model_run_order),
    )
    if not selected_specs:
        raise ValueError(
            "No model specs selected after applying --model-families/--model-seeds."
        )

    run_specs: list[dict[str, Any]] = []
    prepared_specs: list[dict[str, Any]] = []

    sweep_manifest = {
        "generated_at_utc": _utc_now(),
        "output_root": str(output_root),
        "templates": templates,
        "model_run_order": str(args.model_run_order),
        "pass_ks": pass_ks,
        "mean_k": int(args.mean_k),
        "sample_count": int(args.sample_count),
        "inference_seeds": _parse_seed_list(args.inference_seeds),
        "prompt_batches_per_epoch": int(args.prompt_batches_per_epoch),
        "infer_epoch_checkpoints": int(args.infer_epoch_checkpoints),
        "model_runs": [],
    }

    for spec in selected_specs:
        model_output_root = output_root / spec.family / f"seed{spec.model_seed}"
        model_output_root.mkdir(parents=True, exist_ok=True)
        local_checkpoint_root, local_root_manifest = _stage_local_checkpoint_root(
            spec=spec,
            staging_root=staging_root,
            cache_root=cache_root,
        )
        resolved_checkpoints, resolution_rows = _resolve_model_checkpoints(
            spec=spec,
            cache_root=cache_root,
            local_checkpoint_root=local_checkpoint_root,
            infer_epoch_checkpoints=int(args.infer_epoch_checkpoints),
            prompt_batches_per_epoch=int(args.prompt_batches_per_epoch),
            force_download=bool(args.force_download),
            local_files_only=bool(args.local_files_only),
        )

        run_spec = {
            "family": spec.family,
            "family_display": spec.family_display,
            "model_seed": spec.model_seed,
            "repo_id": spec.repo_id,
            "output_root": str(model_output_root),
            "local_checkpoint_root": str(local_checkpoint_root)
            if local_checkpoint_root
            else "",
            "resolved_checkpoints": resolution_rows,
            "local_root_manifest": local_root_manifest,
            "all_complete": False,
        }
        sweep_manifest["model_runs"].append(run_spec)
        run_specs.append(run_spec)
        prepared_specs.append(
            {
                "spec": spec,
                "model_output_root": model_output_root,
                "local_checkpoint_root": local_checkpoint_root,
                "resolved_checkpoints": resolved_checkpoints,
                "resolution_rows": resolution_rows,
            }
        )

        _safe_json_dump(output_root / "run_manifest.json", sweep_manifest)
        _refresh_sweep_outputs(
            output_root=output_root,
            run_specs=run_specs,
            prepared_specs=prepared_specs,
            templates=templates,
            pass_ks=pass_ks,
        )

    if args.dry_run:
        print(json.dumps(sweep_manifest, indent=2, sort_keys=True))
        return 0

    if str(args.model_run_order) == "seed_then_template_then_family":
        run_units: list[tuple[dict[str, Any], str]] = []
        for model_seed in sorted(
            {int(item["spec"].model_seed) for item in prepared_specs}
        ):
            seed_prepared = [
                item
                for item in prepared_specs
                if int(item["spec"].model_seed) == model_seed
            ]
            for template in templates:
                for item in seed_prepared:
                    run_units.append((item, template))
        for prepared, template in run_units:
            spec = prepared["spec"]
            model_output_root = prepared["model_output_root"]
            local_checkpoint_root = prepared["local_checkpoint_root"]
            resolved_checkpoints = prepared["resolved_checkpoints"]
            resolution_rows = prepared["resolution_rows"]
            template_output_root = model_output_root / "templates" / str(template)
            run_argv = _build_passk_argv(
                spec=spec,
                resolved_checkpoints=resolved_checkpoints,
                local_checkpoint_root=local_checkpoint_root,
                model_output_root=model_output_root,
                args=args,
                templates_override=[template],
                output_root_override=template_output_root,
            )
            print(
                json.dumps(
                    {
                        "generated_at_utc": _utc_now(),
                        "family": spec.family,
                        "family_display": spec.family_display,
                        "model_seed": spec.model_seed,
                        "template": template,
                        "repo_id": spec.repo_id,
                        "output_root": str(template_output_root),
                        "local_checkpoint_root": str(local_checkpoint_root)
                        if local_checkpoint_root
                        else "",
                        "resolved_checkpoints": resolution_rows,
                        "passk_argv": run_argv,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
            result = int(passk_eval_main(run_argv))
            if result != 0:
                raise RuntimeError(
                    f"passk_eval failed for {spec.family} seed {spec.model_seed} template {template} "
                    f"with exit code {result}"
                )
            _refresh_sweep_outputs(
                output_root=output_root,
                run_specs=run_specs,
                prepared_specs=prepared_specs,
                templates=templates,
                pass_ks=pass_ks,
            )
    else:
        for prepared in prepared_specs:
            spec = prepared["spec"]
            model_output_root = prepared["model_output_root"]
            local_checkpoint_root = prepared["local_checkpoint_root"]
            resolved_checkpoints = prepared["resolved_checkpoints"]
            resolution_rows = prepared["resolution_rows"]
            run_argv = _build_passk_argv(
                spec=spec,
                resolved_checkpoints=resolved_checkpoints,
                local_checkpoint_root=local_checkpoint_root,
                model_output_root=model_output_root,
                args=args,
            )
            print(
                json.dumps(
                    {
                        "generated_at_utc": _utc_now(),
                        "family": spec.family,
                        "family_display": spec.family_display,
                        "model_seed": spec.model_seed,
                        "repo_id": spec.repo_id,
                        "output_root": str(model_output_root),
                        "local_checkpoint_root": str(local_checkpoint_root)
                        if local_checkpoint_root
                        else "",
                        "resolved_checkpoints": resolution_rows,
                        "passk_argv": run_argv,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
            result = int(passk_eval_main(run_argv))
            if result != 0:
                raise RuntimeError(
                    f"passk_eval failed for {spec.family} seed {spec.model_seed} with exit code {result}"
                )
            _refresh_sweep_outputs(
                output_root=output_root,
                run_specs=run_specs,
                prepared_specs=prepared_specs,
                templates=templates,
                pass_ks=pass_ks,
            )

    _refresh_sweep_outputs(
        output_root=output_root,
        run_specs=run_specs,
        prepared_specs=prepared_specs,
        templates=templates,
        pass_ks=pass_ks,
    )
    final_status = json.loads((output_root / "run_status.json").read_text())
    print(json.dumps(final_status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
