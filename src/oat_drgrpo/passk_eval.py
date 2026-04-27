from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


TASK_ORDER: tuple[str, ...] = (
    "aime",
    "amc",
    "math",
    "minerva",
    "olympiad_bench",
)
DEFAULT_TEMPLATE_SWEEP: tuple[str, ...] = ("no", "r1", "qwen_math")
DEFAULT_PASS_KS: tuple[int, ...] = (1, 8, 16, 32, 64, 128)

# These defaults are intentionally conservative for pass@128 on 8x A6000s.
DEFAULT_CHUNK_SIZES: dict[str, int] = {
    "aime": 6,
    "amc": 8,
    "math": 8,
    "minerva": 6,
    "olympiad_bench": 6,
}
TASK_COST_WEIGHTS: dict[str, float] = {
    "aime": 1.0,
    "amc": 1.0,
    "math": 1.15,
    "minerva": 1.2,
    "olympiad_bench": 1.25,
}


@dataclass(frozen=True)
class ShardSpec:
    label: str
    task: str
    start: int
    end: int
    weight: int


@dataclass(frozen=True)
class CheckpointSpec:
    alias: str
    source: str
    local_path: str
    step_name: str | None = None
    step_number: int | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_repo_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def _normalize_step_name(value: str) -> str:
    value = str(value).strip()
    if value.startswith("step_"):
        return value
    return f"step_{int(value):05d}"


def _step_to_int(step_name: str) -> int:
    return int(_normalize_step_name(step_name).split("_", 1)[1])


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _safe_json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    )


def _parse_chunk_size_overrides(raw_values: Sequence[str]) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError(f"Invalid --chunk-size value '{raw}'. Expected task=size.")
        task, size_text = raw.split("=", 1)
        task = task.strip()
        if task not in TASK_ORDER:
            raise ValueError(f"Unknown task '{task}' in --chunk-size override.")
        size = int(size_text.strip())
        if size <= 0:
            raise ValueError(f"Chunk size for task '{task}' must be positive.")
        overrides[task] = size
    return overrides


def _parse_checkpoint_spec(raw: str) -> tuple[str, str]:
    text = raw.strip()
    if not text:
        raise ValueError("Checkpoint spec cannot be empty.")
    if "=" in text:
        alias, value = text.split("=", 1)
        alias = alias.strip()
        value = value.strip()
        if not alias or not value:
            raise ValueError(f"Invalid checkpoint spec '{raw}'.")
        return alias, value
    value = text
    normalized = (
        _normalize_step_name(value)
        if value.startswith("step_") or value.isdigit()
        else value
    )
    alias = Path(normalized).name
    return alias, value


def _flatten_checkpoint_metric_row(
    summary: dict[str, Any], pass_ks: Sequence[int]
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "checkpoint_alias": summary["checkpoint_alias"],
        "checkpoint_step": summary.get("checkpoint_step"),
        "completed": bool(summary.get("completed")),
        "completed_shards": int(summary.get("completed_shards", 0)),
        "total_shards": int(summary.get("total_shards", 0)),
        "inference_seed_count": int(summary.get("inference_seed_count", 0)),
    }
    for scope in ("macro", "micro"):
        metrics = summary.get(f"{scope}_metrics", {})
        for metric_name, metric_value in sorted(metrics.items()):
            row[f"{scope}_{metric_name}"] = float(metric_value)
    return row


def _label_for_shard(task: str, start: int, end: int, total: int) -> str:
    if start == 0 and end == total:
        return task
    return f"{task}_{start:04d}_{end:04d}"


def _build_shards(
    dataset_root: Path,
    chunk_sizes: dict[str, int],
    tasks: Sequence[str] | None = None,
) -> list[ShardSpec]:
    from datasets import load_from_disk

    selected_tasks = set(tasks or TASK_ORDER)
    datasets_by_task = load_from_disk(str(dataset_root))
    shards: list[ShardSpec] = []
    for task in TASK_ORDER:
        if task not in selected_tasks:
            continue
        dataset = datasets_by_task[task]
        total = len(dataset)
        chunk_size = min(int(chunk_sizes[task]), total)
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            shards.append(
                ShardSpec(
                    label=_label_for_shard(task, start, end, total),
                    task=task,
                    start=start,
                    end=end,
                    weight=end - start,
                )
            )
    return shards


def _estimate_shard_cost(shard: ShardSpec) -> float:
    return float(shard.weight) * float(TASK_COST_WEIGHTS.get(shard.task, 1.0))


def _assign_shards_to_workers(
    shards: Sequence[ShardSpec], worker_count: int
) -> list[list[ShardSpec]]:
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    buckets: list[list[ShardSpec]] = [[] for _ in range(worker_count)]
    bucket_costs = [0.0 for _ in range(worker_count)]
    for shard in sorted(shards, key=_estimate_shard_cost, reverse=True):
        index = min(range(worker_count), key=lambda idx: bucket_costs[idx])
        buckets[index].append(shard)
        bucket_costs[index] += _estimate_shard_cost(shard)
    return buckets


def _detect_visible_gpus() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        values = [part.strip() for part in visible.split(",") if part.strip()]
        if values:
            return values
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _parse_int_list(raw: str) -> list[int]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("Expected a non-empty comma-separated integer list.")
    values = sorted({int(part) for part in parts})
    if any(value <= 0 for value in values):
        raise ValueError("Pass@k values must be positive.")
    return values


def _parse_seed_list(raw: str) -> list[int]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("Expected a non-empty comma-separated seed list.")
    values = sorted({int(part) for part in parts})
    if any(value < 0 for value in values):
        raise ValueError("Inference seeds must be non-negative integers.")
    return values


def _normalize_template_name(template: str) -> str:
    normalized = str(template).strip()
    alias_map = {
        "r1": "r1",
        "qwen_math": "qwen_math",
        "qwen-math": "qwen_math",
        "no": "no",
        "raw": "no",
    }
    if normalized not in alias_map:
        raise ValueError(f"Unsupported template '{template}'.")
    return alias_map[normalized]


def _parse_template_list(raw: str) -> list[str]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("Expected a non-empty comma-separated template list.")
    templates: list[str] = []
    seen: set[str] = set()
    for part in parts:
        normalized = _normalize_template_name(part)
        if normalized in seen:
            continue
        seen.add(normalized)
        templates.append(normalized)
    return templates


def _average_metric_dicts(metric_dicts: Sequence[dict[str, float]]) -> dict[str, float]:
    if not metric_dicts:
        return {}
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for metrics in metric_dicts:
        for metric_name, metric_value in metrics.items():
            if metric_value is None:
                continue
            sums[metric_name] = sums.get(metric_name, 0.0) + float(metric_value)
            counts[metric_name] = counts.get(metric_name, 0) + 1
    return {
        metric_name: float(sums[metric_name] / counts[metric_name])
        for metric_name in sorted(sums)
    }


def _compute_prefix_metrics(
    rewards: Sequence[float], pass_ks: Sequence[int], mean_k: int
) -> dict[str, float]:
    if not rewards:
        raise ValueError("Cannot compute pass@k metrics for an empty reward list.")
    if max(pass_ks) > len(rewards):
        raise ValueError(
            f"Need at least {max(pass_ks)} sampled rewards, got {len(rewards)}."
        )
    metrics: dict[str, float] = {}
    for pass_k in pass_ks:
        metrics[f"pass_at_{pass_k}"] = float(max(rewards[:pass_k]) > 0.0)
    if mean_k > len(rewards):
        raise ValueError(f"Need at least {mean_k} sampled rewards for mean@{mean_k}.")
    metrics[f"mean_at_{mean_k}"] = float(sum(rewards[:mean_k]) / float(mean_k))
    metrics["sampled_pass_at_1"] = float(max(rewards[:1]) > 0.0)
    return metrics


def _select_epoch_boundary_steps(
    available_steps: Sequence[int],
    num_epochs: int,
    prompt_batches_per_epoch: int | None = None,
) -> list[tuple[str, str]]:
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")
    unique_steps = sorted(set(int(step) for step in available_steps))
    if not unique_steps:
        raise ValueError("No available steps to select from.")
    if 0 not in unique_steps:
        raise ValueError("Epoch selection requires step 0 to be available.")
    final_step = unique_steps[-1]
    if prompt_batches_per_epoch is None:
        prompt_batches_per_epoch = max(1, (final_step - 1) // num_epochs)
    targets = [
        prompt_batches_per_epoch * epoch + 1 for epoch in range(1, num_epochs + 1)
    ]
    selections: list[tuple[str, str]] = [("pretrain", "step_00000")]
    candidate_steps = [step for step in unique_steps if step > 0]
    for epoch_index, target in enumerate(targets, start=1):
        chosen = next(
            (step for step in candidate_steps if step >= target), candidate_steps[-1]
        )
        selections.append((f"epoch{epoch_index}", _normalize_step_name(str(chosen))))
    return selections


def _list_remote_checkpoint_steps(repo_id: str, token: str | None = None) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    checkpoints = sorted(
        {
            piece.split("/")[1]
            for piece in files
            if piece.startswith("checkpoints/") and piece.count("/") >= 2
        },
        key=_step_to_int,
    )
    return checkpoints


def _resolve_local_step_dir(root: Path, step_name: str) -> Path | None:
    candidates = (
        root / step_name,
        root / "checkpoints" / step_name,
        root / "saved_models" / step_name,
    )
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "config.json").is_file():
            return candidate.resolve()
    return None


def _list_local_checkpoint_steps(local_checkpoint_root: Path) -> list[str]:
    step_dirs: set[str] = set()
    for child in local_checkpoint_root.iterdir():
        if (
            child.is_dir()
            and child.name.startswith("step_")
            and (child / "config.json").is_file()
        ):
            step_dirs.add(child.name)
    for subdir_name in ("checkpoints", "saved_models"):
        subdir = local_checkpoint_root / subdir_name
        if not subdir.is_dir():
            continue
        for child in subdir.iterdir():
            if (
                child.is_dir()
                and child.name.startswith("step_")
                and (child / "config.json").is_file()
            ):
                step_dirs.add(child.name)
    return sorted(step_dirs, key=_step_to_int)


def _download_remote_checkpoint(
    *,
    repo_id: str,
    step_name: str,
    cache_root: Path,
    token: str | None = None,
    force_download: bool = False,
    local_files_only: bool = False,
) -> Path:
    from huggingface_hub import snapshot_download

    target_root = cache_root / _safe_repo_name(repo_id) / step_name
    checkpoint_dir = target_root / "checkpoints" / step_name
    if (
        not force_download
        and (checkpoint_dir / "config.json").is_file()
        and (checkpoint_dir / "model.safetensors").is_file()
    ):
        return checkpoint_dir
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(target_root),
        local_dir_use_symlinks=False,
        allow_patterns=[
            f"checkpoints/{step_name}/*",
            "metadata/checkpoints_index.json",
            "metadata/run_manifest.json",
            "README.md",
        ],
        token=token,
        force_download=force_download,
        local_files_only=local_files_only,
        resume_download=True,
        max_workers=4,
    )
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Downloaded checkpoint directory not found: {checkpoint_dir}"
        )
    return checkpoint_dir


def _resolve_checkpoints(
    *,
    checkpoint_specs: Sequence[str],
    repo_id: str,
    cache_root: Path,
    local_checkpoint_root: Path | None,
    infer_epoch_checkpoints: int | None,
    prompt_batches_per_epoch: int | None,
    token: str | None,
    force_download: bool,
    local_files_only: bool,
) -> list[CheckpointSpec]:
    resolved_specs = list(checkpoint_specs)
    remote_steps: list[str] | None = None
    local_step_set: set[str] = set()
    if local_checkpoint_root is not None:
        local_checkpoint_root = local_checkpoint_root.resolve()
        if not local_checkpoint_root.is_dir():
            raise FileNotFoundError(
                f"Local checkpoint root not found: {local_checkpoint_root}"
            )
        local_step_set = set(_list_local_checkpoint_steps(local_checkpoint_root))
    if infer_epoch_checkpoints is not None:
        selections: list[tuple[str, str]] | None = None
        if local_step_set:
            try:
                selections = _select_epoch_boundary_steps(
                    [
                        _step_to_int(step_name)
                        for step_name in sorted(local_step_set, key=_step_to_int)
                    ],
                    num_epochs=infer_epoch_checkpoints,
                    prompt_batches_per_epoch=prompt_batches_per_epoch,
                )
            except ValueError:
                selections = None
        if selections is None:
            if not repo_id:
                raise ValueError(
                    "--infer-epoch-checkpoints requires --repo-id unless --local-checkpoint-root is "
                    "provided with enough checkpoints to infer epoch boundaries."
                )
            remote_steps = _list_remote_checkpoint_steps(repo_id, token=token)
            selections = _select_epoch_boundary_steps(
                [_step_to_int(step_name) for step_name in remote_steps],
                num_epochs=infer_epoch_checkpoints,
                prompt_batches_per_epoch=prompt_batches_per_epoch,
            )
        resolved_specs.extend(f"{alias}={step_name}" for alias, step_name in selections)
    if not resolved_specs:
        raise ValueError(
            "No checkpoints selected. Use --checkpoint or --infer-epoch-checkpoints."
        )

    if (
        repo_id
        and remote_steps is None
        and (not local_files_only or not local_step_set)
    ):
        remote_steps = _list_remote_checkpoint_steps(repo_id, token=token)
    remote_step_set = set(remote_steps or [])

    checkpoints: list[CheckpointSpec] = []
    seen_aliases: set[str] = set()
    for raw_spec in resolved_specs:
        alias, value = _parse_checkpoint_spec(raw_spec)
        if alias in seen_aliases:
            raise ValueError(f"Duplicate checkpoint alias '{alias}'.")
        seen_aliases.add(alias)
        candidate_path = Path(value).expanduser()
        if candidate_path.exists():
            local_path = candidate_path.resolve()
            step_name = local_path.name if local_path.name.startswith("step_") else None
            step_number = _step_to_int(step_name) if step_name else None
            checkpoints.append(
                CheckpointSpec(
                    alias=alias,
                    source=str(candidate_path),
                    local_path=str(local_path),
                    step_name=step_name,
                    step_number=step_number,
                )
            )
            continue
        step_name = _normalize_step_name(value)
        local_step_path = (
            _resolve_local_step_dir(local_checkpoint_root, step_name)
            if local_checkpoint_root is not None
            else None
        )
        if local_step_path is not None:
            checkpoints.append(
                CheckpointSpec(
                    alias=alias,
                    source=str(local_step_path),
                    local_path=str(local_step_path),
                    step_name=step_name,
                    step_number=_step_to_int(step_name),
                )
            )
            continue
        if not repo_id:
            raise ValueError(
                f"Checkpoint '{value}' is not available locally and no --repo-id was provided."
            )
        if step_name not in remote_step_set:
            raise ValueError(
                f"Remote step '{step_name}' not found in {repo_id}. "
                f"Available steps: {sorted(remote_step_set, key=_step_to_int)}"
            )
        local_path = _download_remote_checkpoint(
            repo_id=repo_id,
            step_name=step_name,
            cache_root=cache_root,
            token=token,
            force_download=force_download,
            local_files_only=local_files_only,
        )
        checkpoints.append(
            CheckpointSpec(
                alias=alias,
                source=f"{repo_id}:{step_name}",
                local_path=str(local_path),
                step_name=step_name,
                step_number=_step_to_int(step_name),
            )
        )
    return checkpoints


def _select_task_template(template: str) -> Callable[[str], str]:
    if template == "r1":
        return lambda question: (
            "A conversation between User and Assistant. The User asks a question, and the "
            "Assistant solves it. The Assistant first thinks about the reasoning process "
            "in the mind and then provides the User with the answer. The reasoning process "
            "is enclosed within <think> </think> and answer is enclosed within <answer> "
            "</answer> tags, respectively, i.e., <think> reasoning process here </think> "
            "<answer> answer here </answer>.\nUser: "
            + question
            + "\nAssistant: <think>"
        )
    if template in {"qwen_math", "qwen-math"}:
        return lambda question: (
            "<|im_start|>system\nPlease reason step by step, and put your final answer within "
            "\\boxed{}.<|im_end|>\n<|im_start|>user\n"
            + question
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
    if template in {"no", "raw"}:
        return lambda question: question
    raise ValueError(f"Unsupported template '{template}'.")


def _get_reward_fn(
    template: str,
) -> Callable[[str, str, bool], tuple[dict[str, Any], float]]:
    from oat_drgrpo.math_grader import answer_tag_reward_fn, boxed_reward_fn

    if template == "r1":
        return answer_tag_reward_fn
    if template in {"qwen_math", "qwen-math", "no", "raw"}:
        return boxed_reward_fn
    raise ValueError(f"Unsupported template '{template}'.")


def _configure_grader_logging() -> None:
    raw = os.environ.get("MAXENT_SEED_PAPER_SUPPRESS_GRADER_LOGS", "1")
    if str(raw).strip().lower() in {"0", "false", "no", "off"}:
        return
    for logger_name in ("math_verify", "math_verify.grader"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False


def _annotate_trace_semantic_metrics(
    *,
    attempts: list[dict[str, Any]],
    template: str,
    similarity_threshold: float,
) -> dict[str, Any]:
    """Attach semantic-cluster metadata to sampled trace attempts.

    The per-attempt annotations are designed for downstream HF upload / analysis.
    We record both:
    - all-row semantic cluster statistics over semantic-valid sampled traces
    - correct-only cluster statistics for Dr.X-style semantic analysis
    """

    if not attempts:
        return {
            "semantic_cluster_count_all": 0,
            "semantic_cluster_entropy_all_norm": 0.0,
            "semantic_cluster_count_correct": 0,
            "semantic_cluster_entropy_correct": 0.0,
            "correct_row_count": 0,
            "semantic_correct_row_frac": 0.0,
            "semantic_valid_row_count": 0,
            "semantic_valid_row_frac": 0.0,
            "semantic_answer_key_extracted_row_frac": 0.0,
            "semantic_signature_extracted_row_frac": 0.0,
        }

    import torch

    from oat_drgrpo.listwise import (
        build_answer_family_semantic_cluster_bundle,
        build_connected_component_semantic_cluster_bundle,
    )
    from oat_drgrpo.math_grader import (
        extract_normalized_final_answer_for_clustering,
        extract_reasoning_signature_for_clustering,
    )

    final_answer_keys = [
        extract_normalized_final_answer_for_clustering(
            str(attempt.get("text", "")),
            template=template,
        )
        for attempt in attempts
    ]
    reasoning_signature_keys = [
        extract_reasoning_signature_for_clustering(
            str(attempt.get("text", "")),
            template=template,
        )
        for attempt in attempts
    ]
    valid_row_mask = torch.ones((1, len(attempts)), dtype=torch.bool)
    cluster_bundle = build_connected_component_semantic_cluster_bundle(
        final_answer_keys_grouped=[final_answer_keys],
        reasoning_signature_keys_grouped=[reasoning_signature_keys],
        valid_row_mask_grouped=valid_row_mask,
        signature_jaccard_merge_threshold=float(similarity_threshold),
    )
    answer_family_bundle = build_answer_family_semantic_cluster_bundle(
        final_answer_keys_grouped=[final_answer_keys],
        valid_row_mask_grouped=valid_row_mask,
    )

    cluster_ids = cluster_bundle.cluster_ids_grouped[0].detach().cpu().tolist()
    semantic_valid_mask = (
        cluster_bundle.semantic_valid_row_mask_grouped[0].detach().cpu().tolist()
    )
    semantic_entropy_all_norm = float(
        cluster_bundle.semantic_entropy_grouped[0].detach().cpu().item()
    )
    cluster_count_all = int(
        cluster_bundle.num_clusters_per_group[0].detach().cpu().item()
    )

    all_cluster_prob_map: dict[int, float] = {}
    all_cluster_count_map: dict[int, int] = {}
    valid_cluster_ids = [
        int(cluster_id)
        for cluster_id, is_valid in zip(cluster_ids, semantic_valid_mask)
        if bool(is_valid) and int(cluster_id) >= 0
    ]
    if valid_cluster_ids:
        counts = Counter(valid_cluster_ids)
        total = float(sum(counts.values()))
        all_cluster_count_map = {
            int(cluster_id): int(count) for cluster_id, count in counts.items()
        }
        all_cluster_prob_map = {
            int(cluster_id): float(count) / total
            for cluster_id, count in counts.items()
        }

    def _attempt_semantic_quality(attempt: Mapping[str, Any]) -> float:
        for key in (
            "solve_quality",
            "semantic_quality",
            "quality",
            "graded_reward",
            "accuracy",
            "reward",
        ):
            value = attempt.get(key)
            if value is not None:
                return float(max(0.0, min(1.0, float(value))))
        return 1.0 if bool(attempt.get("correct", False)) else 0.0

    answer_family_cluster_ids = (
        answer_family_bundle.cluster_ids_grouped[0].detach().cpu().tolist()
    )
    correct_row_mask = [
        _attempt_semantic_quality(attempt) >= 0.999 for attempt in attempts
    ]
    correct_row_count = int(sum(bool(flag) for flag in correct_row_mask))
    correct_cluster_ids = [
        int(cluster_id)
        for cluster_id, is_valid, is_correct in zip(
            answer_family_cluster_ids,
            semantic_valid_mask,
            correct_row_mask,
        )
        if bool(is_valid) and bool(is_correct) and int(cluster_id) >= 0
    ]
    correct_cluster_prob_map: dict[int, float] = {}
    if correct_cluster_ids:
        correct_counts = Counter(correct_cluster_ids)
        correct_total = float(sum(correct_counts.values()))
        correct_cluster_prob_map = {
            int(cluster_id): float(count) / correct_total
            for cluster_id, count in correct_counts.items()
        }
        semantic_entropy_correct = -sum(
            prob * math.log(prob)
            for prob in correct_cluster_prob_map.values()
            if prob > 0.0
        )
        if len(correct_cluster_prob_map) > 1:
            semantic_entropy_correct /= math.log(float(len(correct_cluster_prob_map)))
        else:
            semantic_entropy_correct = 0.0
    else:
        semantic_entropy_correct = 0.0
    cluster_count_correct = int(len(correct_cluster_prob_map))

    for index, attempt in enumerate(attempts):
        reward = float(attempt.get("reward", 0.0))
        correct = bool(reward > 0.0)
        cluster_id = int(cluster_ids[index]) if int(cluster_ids[index]) >= 0 else None
        cluster_prob_all = (
            float(all_cluster_prob_map[cluster_id])
            if cluster_id is not None and cluster_id in all_cluster_prob_map
            else None
        )
        cluster_prob_correct = (
            float(
                correct_cluster_prob_map.get(int(answer_family_cluster_ids[index]), 0.0)
            )
            if bool(correct_row_mask[index])
            and int(answer_family_cluster_ids[index]) >= 0
            and float(
                correct_cluster_prob_map.get(int(answer_family_cluster_ids[index]), 0.0)
            )
            > 0.0
            else None
        )
        attempt["accuracy"] = reward
        attempt["correct"] = correct
        attempt["semantic_final_answer_key"] = final_answer_keys[index]
        attempt["semantic_reasoning_signature_key"] = reasoning_signature_keys[index]
        attempt["semantic_valid"] = bool(semantic_valid_mask[index])
        attempt["semantic_cluster_id"] = cluster_id
        attempt["semantic_cluster_count_all"] = cluster_count_all
        attempt["semantic_cluster_entropy_all_norm"] = semantic_entropy_all_norm
        attempt["semantic_cluster_prob_all"] = cluster_prob_all
        attempt["semantic_cluster_count_all_members"] = (
            int(all_cluster_count_map[cluster_id])
            if cluster_id is not None and cluster_id in all_cluster_count_map
            else None
        )
        attempt["semantic_cluster_self_information_all"] = (
            float(-math.log(cluster_prob_all)) if cluster_prob_all else None
        )
        attempt["semantic_cluster_count_correct"] = cluster_count_correct
        attempt["semantic_cluster_entropy_correct"] = semantic_entropy_correct
        attempt["semantic_cluster_prob_correct"] = cluster_prob_correct
        attempt["semantic_cluster_self_information_correct"] = (
            float(-math.log(cluster_prob_correct)) if cluster_prob_correct else None
        )

    return {
        "semantic_cluster_count_all": cluster_count_all,
        "semantic_cluster_entropy_all_norm": semantic_entropy_all_norm,
        "semantic_cluster_count_correct": cluster_count_correct,
        "semantic_cluster_entropy_correct": semantic_entropy_correct,
        "correct_row_count": correct_row_count,
        "semantic_correct_row_frac": float(correct_row_count) / float(len(attempts)),
        "semantic_valid_row_count": int(
            sum(bool(flag) for flag in semantic_valid_mask)
        ),
        "semantic_valid_row_frac": float(
            sum(bool(flag) for flag in semantic_valid_mask)
        )
        / float(len(attempts)),
        "semantic_answer_key_extracted_row_frac": float(
            sum(value is not None for value in final_answer_keys)
        )
        / float(len(attempts)),
        "semantic_signature_extracted_row_frac": float(
            sum(value is not None for value in reasoning_signature_keys)
        )
        / float(len(attempts)),
    }


def _sampling_params(
    *,
    n: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    template: str,
) -> Any:
    import vllm

    params = vllm.SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
    )
    if template == "r1":
        params.stop = ["</answer>"]
        params.include_stop_str_in_output = True
    return params


def _coerce_formatted_flag(info: Any) -> bool | None:
    if isinstance(info, dict) and "formatted" in info:
        return bool(info["formatted"])
    return None


def _existing_summary_is_complete(
    path: Path,
    *,
    expected_inference_seeds: Sequence[int] | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return False
    if not bool(payload.get("complete")):
        return False
    if expected_inference_seeds is None:
        return True
    actual_seeds = payload.get("inference_seeds")
    normalized_expected = [int(seed) for seed in expected_inference_seeds]
    if actual_seeds is None:
        return False
    normalized_actual = [int(seed) for seed in actual_seeds]
    if normalized_actual != normalized_expected:
        return False
    by_inference_seed = payload.get("by_inference_seed")
    if not isinstance(by_inference_seed, dict):
        return False
    return all(str(seed) in by_inference_seed for seed in normalized_expected)


def _worker_runtime_env(
    *,
    output_root: Path,
    checkpoint_alias: str,
    worker_tag: str,
) -> dict[str, str]:
    repo_root = _repo_root()
    runtime_root = output_root / "_runtime_cache" / checkpoint_alias / worker_tag
    home_dir = repo_root / "var" / "seed_paper_eval" / "home"
    hf_home = repo_root / "var" / "seed_paper_eval" / "hf_cache"
    xdg_cache = repo_root / "var" / "seed_paper_eval" / "xdg_cache"
    xdg_config = repo_root / "var" / "seed_paper_eval" / "xdg_config"
    vllm_cache_root = runtime_root / "vllm_cache"
    torchinductor_cache_dir = runtime_root / "torchinductor_cache"
    for path in (
        runtime_root,
        home_dir,
        hf_home / "hub",
        hf_home / "datasets",
        hf_home / "transformers",
        hf_home / "xet",
        xdg_cache,
        xdg_config,
        vllm_cache_root,
        torchinductor_cache_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home_dir),
            "PYTHONNOUSERSITE": "0",
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "HF_HOME": str(hf_home),
            "HF_HUB_CACHE": str(hf_home / "hub"),
            "HUGGINGFACE_HUB_CACHE": str(hf_home / "hub"),
            "HF_DATASETS_CACHE": str(hf_home / "datasets"),
            "TRANSFORMERS_CACHE": str(hf_home / "transformers"),
            "HF_XET_CACHE": str(hf_home / "xet"),
            "XDG_CACHE_HOME": str(xdg_cache),
            "XDG_CONFIG_HOME": str(xdg_config),
            "VLLM_CACHE_ROOT": str(vllm_cache_root),
            "TORCHINDUCTOR_CACHE_DIR": str(torchinductor_cache_dir),
            "TOKENIZERS_PARALLELISM": "false",
            "MAXENT_SEED_PAPER_SUPPRESS_GRADER_LOGS": "1",
            "PYTHONPATH": str(repo_root / "src")
            + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        }
    )
    return env


def _evaluate_loaded_shard(
    *,
    checkpoint: CheckpointSpec,
    shard: ShardSpec,
    datasets_by_task: Any,
    llm: Any,
    output_root: Path,
    template: str,
    apply_template: Callable[[str], str],
    reward_fn: Callable[[str, str, bool], tuple[dict[str, Any], float]],
    pass_ks: Sequence[int],
    mean_k: int,
    max_tokens: int,
    max_model_len: int,
    greedy_batch_size: int,
    sampled_batch_size: int,
    sample_count: int,
    sample_temperature: float,
    sample_top_p: float,
    inference_seeds: Sequence[int],
    include_token_ids: bool,
    semantic_similarity_threshold: float,
) -> Path:
    checkpoint_root = output_root / "checkpoints" / checkpoint.alias
    shard_root = checkpoint_root / "shards" / shard.label
    shard_root.mkdir(parents=True, exist_ok=True)
    attempts_path = shard_root / "attempts.json"
    summary_path = shard_root / "summary.json"
    metadata_path = shard_root / "metadata.json"

    dataset = datasets_by_task[shard.task]
    rows = list(range(shard.start, shard.end))
    examples = [dataset[index] for index in rows]
    problems = [example["problem"] for example in examples]
    answers = [example["answer"] for example in examples]
    model_inputs = [apply_template(problem) for problem in problems]
    normalized_inference_seeds = [int(seed) for seed in inference_seeds]
    if not normalized_inference_seeds:
        raise ValueError("Need at least one inference seed.")

    def generation_seed(
        inference_seed: int, *, phase_offset: int, batch_start: int
    ) -> int:
        # Keep the user-facing inference seed stable while still deriving unique
        # request seeds across shards and batched generate calls.
        return (
            int(inference_seed) * 1_000_000
            + int(shard.start)
            + int(phase_offset)
            + int(batch_start)
        )

    greedy_records: dict[int, dict[str, Any]] = {}
    greedy_lengths: list[int] = []
    greedy_rewards: list[float] = []
    formatted_flags: list[float] = []

    for batch_start in range(0, len(model_inputs), max(int(greedy_batch_size), 1)):
        batch_end = min(batch_start + max(int(greedy_batch_size), 1), len(model_inputs))
        batch_inputs = model_inputs[batch_start:batch_end]
        params = _sampling_params(
            n=1,
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
            seed=generation_seed(
                normalized_inference_seeds[0], phase_offset=0, batch_start=batch_start
            ),
            template=template,
        )
        outputs = llm.generate(batch_inputs, params)
        for offset, output in enumerate(outputs):
            row_index = batch_start + offset
            text = output.outputs[0].text
            info, reward = reward_fn(text, answers[row_index], fast=False)
            token_ids = output.outputs[0].token_ids
            greedy_lengths.append(len(token_ids))
            greedy_rewards.append(float(reward))
            formatted = _coerce_formatted_flag(info)
            if formatted is not None:
                formatted_flags.append(float(formatted))
            greedy_records[row_index] = {
                "text": text,
                "token_length": len(token_ids),
                "reward": float(reward),
                "accuracy": float(reward),
                "correct": bool(float(reward) > 0.0),
                "formatted": formatted,
            }
            if include_token_ids:
                greedy_records[row_index]["token_ids"] = list(token_ids)

    pass_metric_names = [f"pass_at_{pass_k}" for pass_k in pass_ks if int(pass_k) != 1]
    sampled_records_by_seed: dict[int, dict[int, dict[str, Any]]] = {}
    seed_summaries: dict[str, dict[str, Any]] = {}
    for inference_seed in normalized_inference_seeds:
        sampled_records: dict[int, dict[str, Any]] = {}
        sampled_lengths: list[int] = []
        pass_sums = {name: 0.0 for name in pass_metric_names}
        sampled_pass_at_1_sum = 0.0
        mean_at_sum = 0.0
        semantic_summaries: list[dict[str, float]] = []
        for batch_start in range(0, len(model_inputs), max(int(sampled_batch_size), 1)):
            batch_end = min(
                batch_start + max(int(sampled_batch_size), 1), len(model_inputs)
            )
            batch_inputs = model_inputs[batch_start:batch_end]
            params = _sampling_params(
                n=sample_count,
                temperature=sample_temperature,
                top_p=sample_top_p,
                max_tokens=max_tokens,
                seed=generation_seed(
                    inference_seed, phase_offset=10_000, batch_start=batch_start
                ),
                template=template,
            )
            outputs = llm.generate(batch_inputs, params)
            for offset, output in enumerate(outputs):
                row_index = batch_start + offset
                attempt_seed_records: list[dict[str, Any]] = []
                rewards: list[float] = []
                token_lengths: list[int] = []
                formatted_attempts: list[float] = []
                for sample_index, sample_output in enumerate(output.outputs, start=1):
                    info, reward = reward_fn(
                        sample_output.text, answers[row_index], fast=False
                    )
                    token_lengths.append(len(sample_output.token_ids))
                    rewards.append(float(reward))
                    formatted = _coerce_formatted_flag(info)
                    if formatted is not None:
                        formatted_attempts.append(float(formatted))
                    attempt_record = {
                        "sample_index": sample_index,
                        "text": sample_output.text,
                        "token_length": len(sample_output.token_ids),
                        "reward": float(reward),
                        "accuracy": float(reward),
                        "correct": bool(float(reward) > 0.0),
                        "formatted": formatted,
                    }
                    if include_token_ids:
                        attempt_record["token_ids"] = list(sample_output.token_ids)
                    attempt_seed_records.append(attempt_record)
                semantic_summary = _annotate_trace_semantic_metrics(
                    attempts=attempt_seed_records,
                    template=template,
                    similarity_threshold=float(semantic_similarity_threshold),
                )
                semantic_summaries.append(
                    {
                        key: float(value)
                        for key, value in semantic_summary.items()
                        if isinstance(value, (int, float))
                    }
                )
                prompt_metrics = _compute_prefix_metrics(
                    rewards, pass_ks=pass_ks, mean_k=mean_k
                )
                for metric_name in pass_metric_names:
                    pass_sums[metric_name] += float(prompt_metrics[metric_name])
                sampled_pass_at_1_sum += float(prompt_metrics["sampled_pass_at_1"])
                mean_at_sum += float(prompt_metrics[f"mean_at_{mean_k}"])
                sampled_lengths.extend(token_lengths)
                sampled_records[row_index] = {
                    "attempts": attempt_seed_records,
                    "metrics": prompt_metrics,
                    "formatted_rate": (
                        float(sum(formatted_attempts) / len(formatted_attempts))
                        if formatted_attempts
                        else None
                    ),
                    "semantic_summary": semantic_summary,
                }
        sampled_records_by_seed[inference_seed] = sampled_records
        seed_metrics = {
            "pass_at_1": float(sum(greedy_rewards) / len(greedy_rewards)),
            "sampled_pass_at_1": float(sampled_pass_at_1_sum / len(rows)),
            f"mean_at_{mean_k}": float(mean_at_sum / len(rows)),
            **{name: float(value / len(rows)) for name, value in pass_sums.items()},
            "formatted_rate": (
                float(sum(formatted_flags) / len(formatted_flags))
                if formatted_flags
                else None
            ),
            "greedy_avg_len": float(sum(greedy_lengths) / len(greedy_lengths)),
            "greedy_max_len": int(max(greedy_lengths)),
            "sampled_avg_len": float(sum(sampled_lengths) / len(sampled_lengths)),
            "sampled_max_len": int(max(sampled_lengths)),
        }
        seed_metrics.update(_average_metric_dicts(semantic_summaries))
        seed_summaries[str(inference_seed)] = {
            "inference_seed": int(inference_seed),
            "attempt_count": int(len(rows) * sample_count),
            "metrics": seed_metrics,
        }

    attempt_records: list[dict[str, Any]] = []
    for row_offset, dataset_index in enumerate(rows):
        inference_runs: list[dict[str, Any]] = []
        for inference_seed in normalized_inference_seeds:
            sampled_record = sampled_records_by_seed[inference_seed][row_offset]
            inference_runs.append(
                {
                    "inference_seed": int(inference_seed),
                    "sampled": sampled_record,
                }
            )
        attempt_records.append(
            {
                "task_name": shard.task,
                "shard_label": shard.label,
                "dataset_index": dataset_index,
                "dataset_row": _jsonable(examples[row_offset]),
                "model_input": model_inputs[row_offset],
                "greedy": greedy_records[row_offset],
                "inference_runs": inference_runs,
            }
        )

    formatted_rate = (
        float(sum(formatted_flags) / len(formatted_flags)) if formatted_flags else None
    )
    aggregate_metrics = _average_metric_dicts(
        [seed_summary["metrics"] for seed_summary in seed_summaries.values()]
    )
    aggregate_metrics["formatted_rate"] = formatted_rate
    summary = {
        "generated_at_utc": _utc_now(),
        "complete": True,
        "checkpoint_alias": checkpoint.alias,
        "checkpoint_source": checkpoint.source,
        "checkpoint_step": checkpoint.step_name,
        "checkpoint_path": checkpoint.local_path,
        "task": shard.task,
        "label": shard.label,
        "start": shard.start,
        "end": shard.end,
        "weight": shard.weight,
        "metrics": aggregate_metrics,
        "inference_seeds": normalized_inference_seeds,
        "inference_seed_count": len(normalized_inference_seeds),
        "metrics_aggregation": "mean_over_inference_seeds",
        "sample_count_per_seed": int(sample_count),
        "attempt_count": int(
            len(rows) * sample_count * len(normalized_inference_seeds)
        ),
        "by_inference_seed": seed_summaries,
        "attempts_path": str(attempts_path),
    }
    metadata = {
        "generated_at_utc": _utc_now(),
        "checkpoint": asdict(checkpoint),
        "shard": asdict(shard),
        "template": template,
        "max_tokens": max_tokens,
        "max_model_len": max_model_len,
        "greedy_batch_size": greedy_batch_size,
        "sampled_batch_size": sampled_batch_size,
        "sample_count": sample_count,
        "sample_temperature": sample_temperature,
        "sample_top_p": sample_top_p,
        "inference_seeds": normalized_inference_seeds,
        "pass_ks": list(pass_ks),
        "mean_k": int(mean_k),
        "include_token_ids": bool(include_token_ids),
        "semantic_similarity_threshold": float(semantic_similarity_threshold),
    }
    _safe_json_dump(metadata_path, metadata)
    _safe_json_dump(attempts_path, attempt_records)
    _safe_json_dump(summary_path, summary)
    return summary_path


def _summarize_task_metric_buckets(
    *,
    task_metric_buckets: dict[str, dict[str, list[tuple[int, float]]]],
    shards: Sequence[ShardSpec],
) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, float]]:
    def weighted_mean(entries: Sequence[tuple[int, float]]) -> float:
        total_weight = sum(weight for weight, _ in entries)
        if total_weight <= 0:
            raise ValueError("Weighted mean requires positive total weight.")
        return sum(weight * value for weight, value in entries) / float(total_weight)

    metric_names = sorted(
        {
            metric_name
            for metric_buckets in task_metric_buckets.values()
            for metric_name in metric_buckets
        }
    )
    by_task_averages: dict[str, dict[str, float]] = {}
    for task, metric_buckets in task_metric_buckets.items():
        by_task_averages[task] = {
            metric_name: weighted_mean(metric_buckets[metric_name])
            for metric_name in metric_names
            if metric_name in metric_buckets
        }

    macro_metrics: dict[str, float] = {}
    micro_metrics: dict[str, float] = {}
    if by_task_averages:
        for metric_name in metric_names:
            task_values = [
                metrics[metric_name]
                for metrics in by_task_averages.values()
                if metric_name in metrics
            ]
            if task_values:
                macro_metrics[metric_name] = float(sum(task_values) / len(task_values))

        total_weight = sum(
            shard.weight for shard in shards if shard.task in by_task_averages
        )
        if total_weight > 0:
            for metric_name in metric_names:
                weighted_values = [
                    shard.weight * by_task_averages[shard.task][metric_name]
                    for shard in shards
                    if shard.task in by_task_averages
                    and metric_name in by_task_averages[shard.task]
                ]
                if weighted_values:
                    micro_metrics[metric_name] = float(
                        sum(weighted_values) / total_weight
                    )

    return by_task_averages, macro_metrics, micro_metrics


def _aggregate_checkpoint_results(
    *,
    checkpoint: CheckpointSpec,
    output_root: Path,
    shards: Sequence[ShardSpec],
    pass_ks: Sequence[int],
    mean_k: int,
    inference_seeds: Sequence[int],
) -> dict[str, Any]:
    checkpoint_root = output_root / "checkpoints" / checkpoint.alias
    by_task: dict[str, dict[str, list[tuple[int, float]]]] = {}
    by_inference_seed_task: dict[str, dict[str, dict[str, list[tuple[int, float]]]]] = {
        str(int(seed)): {} for seed in inference_seeds
    }
    source_paths: dict[str, list[dict[str, Any]]] = {}
    completed_shards = 0
    for shard in shards:
        summary_path = checkpoint_root / "shards" / shard.label / "summary.json"
        if not _existing_summary_is_complete(
            summary_path,
            expected_inference_seeds=inference_seeds,
        ):
            continue
        completed_shards += 1
        summary = json.loads(summary_path.read_text())
        attempts_path = checkpoint_root / "shards" / shard.label / "attempts.json"
        task_metrics = summary["metrics"]
        bucket = by_task.setdefault(shard.task, {})
        for metric_name, metric_value in task_metrics.items():
            bucket.setdefault(metric_name, []).append(
                (shard.weight, float(metric_value))
            )
        for inference_seed in inference_seeds:
            seed_summary = summary["by_inference_seed"][str(int(inference_seed))]
            seed_bucket = by_inference_seed_task[str(int(inference_seed))].setdefault(
                shard.task, {}
            )
            for metric_name, metric_value in seed_summary["metrics"].items():
                seed_bucket.setdefault(metric_name, []).append(
                    (shard.weight, float(metric_value))
                )
        source_paths.setdefault(shard.task, []).append(
            {
                "summary_path": str(summary_path),
                "attempts_path": str(attempts_path),
                "weight": shard.weight,
            }
        )

    completed = completed_shards == len(shards)
    by_task_averages, macro_metrics, micro_metrics = _summarize_task_metric_buckets(
        task_metric_buckets=by_task,
        shards=shards,
    )
    by_inference_seed: dict[str, dict[str, Any]] = {}
    for inference_seed, seed_task_buckets in by_inference_seed_task.items():
        seed_by_task, seed_macro_metrics, seed_micro_metrics = (
            _summarize_task_metric_buckets(
                task_metric_buckets=seed_task_buckets,
                shards=shards,
            )
        )
        by_inference_seed[inference_seed] = {
            "inference_seed": int(inference_seed),
            "by_task": seed_by_task,
            "macro_metrics": seed_macro_metrics,
            "micro_metrics": seed_micro_metrics,
        }

    aggregate = {
        "generated_at_utc": _utc_now(),
        "checkpoint_alias": checkpoint.alias,
        "checkpoint_source": checkpoint.source,
        "checkpoint_step": checkpoint.step_name,
        "checkpoint_path": checkpoint.local_path,
        "completed": completed,
        "completed_shards": completed_shards,
        "total_shards": len(shards),
        "inference_seeds": [int(seed) for seed in inference_seeds],
        "inference_seed_count": len(inference_seeds),
        "metrics_aggregation": "mean_over_inference_seeds",
        "by_task": by_task_averages,
        "macro_metrics": macro_metrics,
        "micro_metrics": micro_metrics,
        "by_inference_seed": by_inference_seed,
        "source_paths": source_paths,
    }
    aggregate_path = checkpoint_root / "aggregated_summary.json"
    _safe_json_dump(aggregate_path, aggregate)
    return aggregate


def _write_top_level_status(
    *,
    output_root: Path,
    checkpoints: Sequence[CheckpointSpec],
    aggregates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    summary_by_alias = {item["checkpoint_alias"]: item for item in aggregates}
    payload = {
        "generated_at_utc": _utc_now(),
        "output_root": str(output_root),
        "all_complete": all(item.get("completed") for item in aggregates),
        "checkpoints": [
            {
                "alias": checkpoint.alias,
                "source": checkpoint.source,
                "step_name": checkpoint.step_name,
                "local_path": checkpoint.local_path,
                "summary": summary_by_alias.get(checkpoint.alias),
            }
            for checkpoint in checkpoints
        ],
    }
    _safe_json_dump(output_root / "run_status.json", payload)
    return payload


def _init_wandb(
    *,
    project: str,
    run_name: str,
    entity: str,
    output_root: Path,
    config: dict[str, Any],
) -> Any:
    import wandb

    wandb_dir = _repo_root() / "var" / "wandb" / "runs"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=project,
        entity=entity or None,
        name=run_name,
        job_type="dataset",
        dir=str(wandb_dir),
        config=config,
    )
    wandb.define_metric("checkpoint_step")
    wandb.define_metric("*", step_metric="checkpoint_step")
    return run


def _log_wandb_outputs(
    *,
    output_root: Path,
    aggregates: Sequence[dict[str, Any]],
    project: str,
    run_name: str,
    entity: str,
    artifact_name: str,
    pass_ks: Sequence[int],
    inference_seeds: Sequence[int],
    template: str,
) -> None:
    import wandb

    config = {
        "output_root": str(output_root),
        "generated_at_utc": _utc_now(),
        "checkpoint_count": len(aggregates),
        "pass_ks": list(pass_ks),
        "inference_seeds": [int(seed) for seed in inference_seeds],
        "template": str(template),
    }
    run = _init_wandb(
        project=project,
        run_name=run_name,
        entity=entity,
        output_root=output_root,
        config=config,
    )
    try:
        table_columns = [
            "checkpoint_alias",
            "checkpoint_step",
            "completed",
            "completed_shards",
            "total_shards",
            "inference_seed_count",
            "macro_pass_at_1",
            "micro_pass_at_1",
            "macro_pass_at_8",
            "micro_pass_at_8",
            "macro_pass_at_16",
            "micro_pass_at_16",
            "macro_pass_at_32",
            "micro_pass_at_32",
            "macro_pass_at_64",
            "micro_pass_at_64",
            "macro_pass_at_128",
            "micro_pass_at_128",
            "macro_mean_at_8",
            "micro_mean_at_8",
        ]
        table = wandb.Table(columns=table_columns)
        for summary in aggregates:
            step_value = summary.get("checkpoint_step")
            checkpoint_step_int = _step_to_int(step_value) if step_value else 0
            payload = {
                "checkpoint_step": checkpoint_step_int,
                "checkpoint_alias": summary["checkpoint_alias"],
                "progress/completed_shards": int(summary["completed_shards"]),
                "progress/total_shards": int(summary["total_shards"]),
                "progress/completed": int(bool(summary["completed"])),
                "inference_seed_count": int(summary.get("inference_seed_count", 0)),
            }
            for scope in ("macro", "micro"):
                metrics = summary.get(f"{scope}_metrics", {})
                for metric_name, metric_value in metrics.items():
                    payload[f"{scope}/{metric_name}"] = float(metric_value)
            run.log(payload)
            row = _flatten_checkpoint_metric_row(summary, pass_ks=pass_ks)
            table.add_data(
                row.get("checkpoint_alias"),
                row.get("checkpoint_step"),
                row.get("completed"),
                row.get("completed_shards"),
                row.get("total_shards"),
                row.get("inference_seed_count"),
                row.get("macro_pass_at_1"),
                row.get("micro_pass_at_1"),
                row.get("macro_pass_at_8"),
                row.get("micro_pass_at_8"),
                row.get("macro_pass_at_16"),
                row.get("micro_pass_at_16"),
                row.get("macro_pass_at_32"),
                row.get("micro_pass_at_32"),
                row.get("macro_pass_at_64"),
                row.get("micro_pass_at_64"),
                row.get("macro_pass_at_128"),
                row.get("micro_pass_at_128"),
                row.get("macro_mean_at_8"),
                row.get("micro_mean_at_8"),
            )
        run.log({"checkpoint_summary": table})
        artifact = wandb.Artifact(
            name=artifact_name,
            type="dataset",
            metadata={
                "generated_at_utc": _utc_now(),
                "checkpoint_count": len(aggregates),
                "output_root": str(output_root),
            },
        )
        artifact.add_dir(str(output_root))
        run.log_artifact(artifact, aliases=["latest"])
    finally:
        run.finish()


def _checkpoint_manifest_path(output_root: Path, checkpoint_alias: str) -> Path:
    return output_root / "checkpoints" / checkpoint_alias / "checkpoint_manifest.json"


def _write_checkpoint_manifest(
    *,
    output_root: Path,
    checkpoint: CheckpointSpec,
    shards: Sequence[ShardSpec],
    template: str,
    pass_ks: Sequence[int],
    mean_k: int,
    inference_seeds: Sequence[int],
    args: argparse.Namespace,
) -> Path:
    path = _checkpoint_manifest_path(output_root, checkpoint.alias)
    payload = {
        "generated_at_utc": _utc_now(),
        "checkpoint": asdict(checkpoint),
        "shards": [asdict(shard) for shard in shards],
        "template": template,
        "pass_ks": list(pass_ks),
        "mean_k": int(mean_k),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "output_root": str(output_root),
        "max_tokens": int(args.max_tokens),
        "max_model_len": int(args.max_model_len),
        "greedy_batch_size": int(args.greedy_batch_size),
        "sampled_batch_size": int(args.sampled_batch_size),
        "sample_count": int(args.sample_count),
        "sample_temperature": float(args.sample_temperature),
        "sample_top_p": float(args.sample_top_p),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "swap_space": float(args.swap_space),
        "inference_seeds": [int(seed) for seed in inference_seeds],
        "sample_seed": int(args.sample_seed),
        "include_token_ids": bool(args.include_token_ids),
        "semantic_similarity_threshold": float(args.semantic_similarity_threshold),
    }
    _safe_json_dump(path, payload)
    return path


def _load_checkpoint_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _run_worker_mode(args: argparse.Namespace) -> int:
    manifest = _load_checkpoint_manifest(Path(args.worker_manifest))
    checkpoint = CheckpointSpec(**manifest["checkpoint"])
    shard_map = {item["label"]: ShardSpec(**item) for item in manifest["shards"]}
    inference_seeds = [
        int(seed)
        for seed in (
            manifest.get("inference_seeds") or [manifest.get("sample_seed", 0)]
        )
    ]
    shard_labels = list(args.worker_shard_label)
    if not shard_labels:
        raise ValueError("Worker mode requires at least one --worker-shard-label.")
    selected_shards = [shard_map[label] for label in shard_labels]

    deadline_ts = float(args.deadline_ts) if args.deadline_ts else None
    grace_seconds = float(args.deadline_grace_seconds)
    worker_tag = f"gpu{os.environ.get('CUDA_VISIBLE_DEVICES', 'unknown')}"
    env = _worker_runtime_env(
        output_root=Path(manifest["output_root"]),
        checkpoint_alias=checkpoint.alias,
        worker_tag=worker_tag,
    )
    os.environ.update(env)
    import vllm
    from datasets import load_from_disk

    _configure_grader_logging()
    apply_template = _select_task_template(str(manifest["template"]))
    reward_fn = _get_reward_fn(str(manifest["template"]))
    datasets_by_task = load_from_disk(str(manifest["dataset_root"]))
    llm = vllm.LLM(
        model=checkpoint.local_path,
        dtype="bfloat16",
        swap_space=float(manifest["swap_space"]),
        max_model_len=int(manifest["max_model_len"]),
        enable_prefix_caching=True,
        gpu_memory_utilization=float(manifest["gpu_memory_utilization"]),
    )

    for shard in selected_shards:
        summary_path = (
            Path(manifest["output_root"])
            / "checkpoints"
            / checkpoint.alias
            / "shards"
            / shard.label
            / "summary.json"
        )
        if bool(args.skip_existing) and _existing_summary_is_complete(
            summary_path,
            expected_inference_seeds=inference_seeds,
        ):
            print(
                f"[passk-eval] worker={worker_tag} skip checkpoint={checkpoint.alias} shard={shard.label}",
                flush=True,
            )
            continue
        if deadline_ts is not None and time.time() >= (deadline_ts - grace_seconds):
            print(
                f"[passk-eval] worker={worker_tag} stop checkpoint={checkpoint.alias} "
                f"reason=deadline shard={shard.label}",
                flush=True,
            )
            break
        print(
            f"[passk-eval] worker={worker_tag} start checkpoint={checkpoint.alias} shard={shard.label}",
            flush=True,
        )
        _evaluate_loaded_shard(
            checkpoint=checkpoint,
            shard=shard,
            datasets_by_task=datasets_by_task,
            llm=llm,
            output_root=Path(manifest["output_root"]),
            template=str(manifest["template"]),
            apply_template=apply_template,
            reward_fn=reward_fn,
            pass_ks=list(manifest["pass_ks"]),
            mean_k=int(manifest["mean_k"]),
            max_tokens=int(manifest["max_tokens"]),
            max_model_len=int(manifest["max_model_len"]),
            greedy_batch_size=int(manifest["greedy_batch_size"]),
            sampled_batch_size=int(manifest["sampled_batch_size"]),
            sample_count=int(manifest["sample_count"]),
            sample_temperature=float(manifest["sample_temperature"]),
            sample_top_p=float(manifest["sample_top_p"]),
            inference_seeds=inference_seeds,
            include_token_ids=bool(manifest["include_token_ids"]),
            semantic_similarity_threshold=float(
                manifest.get("semantic_similarity_threshold", 0.75)
            ),
        )
        print(
            f"[passk-eval] worker={worker_tag} done checkpoint={checkpoint.alias} shard={shard.label}",
            flush=True,
        )
    return 0


def _spawn_checkpoint_workers(
    *,
    checkpoint: CheckpointSpec,
    shards: Sequence[ShardSpec],
    output_root: Path,
    gpu_ids: Sequence[str],
    manifest_path: Path,
    deadline_ts: float | None,
    deadline_grace_seconds: float,
    skip_existing: bool,
) -> None:
    worker_buckets = _assign_shards_to_workers(shards, max(len(gpu_ids), 1))
    processes: list[tuple[str, subprocess.Popen[str], Path, Any]] = []
    for gpu_id, bucket in zip(gpu_ids, worker_buckets):
        if not bucket:
            continue
        log_path = (
            output_root / "checkpoints" / checkpoint.alias / f"worker_gpu{gpu_id}.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "oat_drgrpo.passk_eval",
            "--worker-mode",
            f"--worker-manifest={manifest_path}",
            f"--deadline-grace-seconds={deadline_grace_seconds}",
            "--skip-existing" if skip_existing else "--no-skip-existing",
        ]
        if deadline_ts is not None:
            cmd.append(f"--deadline-ts={deadline_ts}")
        for shard in bucket:
            cmd.append(f"--worker-shard-label={shard.label}")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONPATH"] = str(_repo_root() / "src") + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        log_file = log_path.open("a")
        process = subprocess.Popen(
            cmd,
            cwd=str(_repo_root()),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((gpu_id, process, log_path, log_file))
    failures: list[str] = []
    for gpu_id, process, log_path, log_file in processes:
        return_code = process.wait()
        log_file.close()
        if return_code != 0:
            failures.append(
                f"checkpoint={checkpoint.alias} gpu={gpu_id} exit_code={return_code} log={log_path}"
            )
    if failures:
        raise RuntimeError("Some workers failed:\n" + "\n".join(failures))


def _remaining_shards(
    output_root: Path,
    checkpoint_alias: str,
    shards: Sequence[ShardSpec],
    inference_seeds: Sequence[int],
) -> list[ShardSpec]:
    remaining: list[ShardSpec] = []
    for shard in shards:
        summary_path = (
            output_root
            / "checkpoints"
            / checkpoint_alias
            / "shards"
            / shard.label
            / "summary.json"
        )
        if not _existing_summary_is_complete(
            summary_path,
            expected_inference_seeds=inference_seeds,
        ):
            remaining.append(shard)
    return remaining


def _write_run_manifest(
    *,
    output_root: Path,
    repo_id: str,
    checkpoints: Sequence[CheckpointSpec],
    shards: Sequence[ShardSpec],
    template: str,
    pass_ks: Sequence[int],
    mean_k: int,
    inference_seeds: Sequence[int],
    chunk_sizes: dict[str, int],
    args: argparse.Namespace,
) -> None:
    payload = {
        "generated_at_utc": _utc_now(),
        "repo_id": repo_id,
        "output_root": str(output_root),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "template": str(template),
        "pass_ks": list(pass_ks),
        "mean_k": int(mean_k),
        "inference_seeds": [int(seed) for seed in inference_seeds],
        "semantic_similarity_threshold": float(args.semantic_similarity_threshold),
        "chunk_sizes": chunk_sizes,
        "checkpoints": [asdict(checkpoint) for checkpoint in checkpoints],
        "shards": [asdict(shard) for shard in shards],
        "time_budget_minutes": float(args.time_budget_minutes)
        if args.time_budget_minutes
        else None,
    }
    _safe_json_dump(output_root / "run_manifest.json", payload)


def _template_output_root(
    base_output_root: Path, template: str, *, multi_template: bool
) -> Path:
    if not multi_template:
        return base_output_root
    return base_output_root / "templates" / str(template)


def _template_scoped_name(
    base_name: str, template: str, *, multi_template: bool
) -> str:
    if not base_name:
        return ""
    if not multi_template:
        return base_name
    return f"{base_name}-{template}"


def _write_template_sweep_manifest(
    *,
    output_root: Path,
    repo_id: str,
    templates: Sequence[str],
    checkpoints: Sequence[CheckpointSpec],
    shards: Sequence[ShardSpec],
    pass_ks: Sequence[int],
    mean_k: int,
    inference_seeds: Sequence[int],
    chunk_sizes: dict[str, int],
    args: argparse.Namespace,
) -> None:
    payload = {
        "generated_at_utc": _utc_now(),
        "repo_id": repo_id,
        "output_root": str(output_root),
        "templates": [str(template) for template in templates],
        "template_output_roots": {
            str(template): str(
                _template_output_root(output_root, str(template), multi_template=True)
            )
            for template in templates
        },
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "pass_ks": list(pass_ks),
        "mean_k": int(mean_k),
        "inference_seeds": [int(seed) for seed in inference_seeds],
        "semantic_similarity_threshold": float(args.semantic_similarity_threshold),
        "chunk_sizes": chunk_sizes,
        "checkpoints": [asdict(checkpoint) for checkpoint in checkpoints],
        "shards": [asdict(shard) for shard in shards],
        "time_budget_minutes": float(args.time_budget_minutes)
        if args.time_budget_minutes
        else None,
    }
    _safe_json_dump(output_root / "run_manifest.json", payload)


def _write_template_sweep_status(
    *,
    output_root: Path,
    templates: Sequence[str],
    template_statuses: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "generated_at_utc": _utc_now(),
        "output_root": str(output_root),
        "all_complete": all(
            bool(template_statuses.get(str(template), {}).get("all_complete"))
            for template in templates
        ),
        "templates": [
            {
                "template": str(template),
                "output_root": str(
                    _template_output_root(
                        output_root, str(template), multi_template=True
                    )
                ),
                "run_status_path": str(
                    _template_output_root(
                        output_root, str(template), multi_template=True
                    )
                    / "run_status.json"
                ),
                "summary": template_statuses.get(str(template)),
            }
            for template in templates
        ],
    }
    _safe_json_dump(output_root / "run_status.json", payload)
    _safe_json_dump(
        output_root / "combined_template_summary.json", payload["templates"]
    )
    return payload


def _run_single_template_eval(
    *,
    template: str,
    base_output_root: Path,
    template_output_root: Path,
    checkpoints: Sequence[CheckpointSpec],
    shards: Sequence[ShardSpec],
    pass_ks: Sequence[int],
    mean_k: int,
    inference_seeds: Sequence[int],
    chunk_sizes: dict[str, int],
    multi_template: bool,
    gpu_ids: Sequence[str],
    deadline_ts: float | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    template_output_root.mkdir(parents=True, exist_ok=True)
    _write_run_manifest(
        output_root=template_output_root,
        repo_id=args.repo_id,
        checkpoints=checkpoints,
        shards=shards,
        template=template,
        pass_ks=pass_ks,
        mean_k=mean_k,
        inference_seeds=inference_seeds,
        chunk_sizes=chunk_sizes,
        args=args,
    )

    aggregates: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        checkpoint_shards = list(shards)
        manifest_path = _write_checkpoint_manifest(
            output_root=template_output_root,
            checkpoint=checkpoint,
            shards=checkpoint_shards,
            template=template,
            pass_ks=pass_ks,
            mean_k=mean_k,
            inference_seeds=inference_seeds,
            args=args,
        )
        remaining = _remaining_shards(
            template_output_root,
            checkpoint.alias,
            checkpoint_shards,
            inference_seeds,
        )
        if remaining:
            print(
                f"[passk-eval] template={template} checkpoint={checkpoint.alias} "
                f"start remaining_shards={len(remaining)}",
                flush=True,
            )
            _spawn_checkpoint_workers(
                checkpoint=checkpoint,
                shards=remaining,
                output_root=template_output_root,
                gpu_ids=gpu_ids,
                manifest_path=manifest_path,
                deadline_ts=deadline_ts,
                deadline_grace_seconds=float(args.deadline_grace_seconds),
                skip_existing=bool(args.skip_existing),
            )
        aggregate = _aggregate_checkpoint_results(
            checkpoint=checkpoint,
            output_root=template_output_root,
            shards=checkpoint_shards,
            pass_ks=pass_ks,
            mean_k=mean_k,
            inference_seeds=inference_seeds,
        )
        aggregates.append(aggregate)
        if deadline_ts is not None and time.time() >= (
            deadline_ts - float(args.deadline_grace_seconds)
        ):
            print(
                f"[passk-eval] stop reason=deadline template={template} checkpoint={checkpoint.alias}",
                flush=True,
            )
            break

    status = _write_top_level_status(
        output_root=template_output_root,
        checkpoints=checkpoints,
        aggregates=aggregates,
    )
    _safe_json_dump(
        template_output_root / "combined_checkpoint_summary.json", aggregates
    )

    if args.use_wandb:
        _log_wandb_outputs(
            output_root=template_output_root,
            aggregates=aggregates,
            project=args.wandb_project,
            run_name=(
                _template_scoped_name(
                    args.wandb_run_name, template, multi_template=multi_template
                )
                or (
                    f"{base_output_root.name}-{template}"
                    if multi_template
                    else template_output_root.name
                )
            ),
            entity=args.wandb_entity,
            artifact_name=(
                _template_scoped_name(
                    args.wandb_artifact_name, template, multi_template=multi_template
                )
                or (
                    f"{base_output_root.name}-{template}-dataset"
                    if multi_template
                    else f"{template_output_root.name}-dataset"
                )
            ),
            pass_ks=pass_ks,
            inference_seeds=inference_seeds,
            template=template,
        )

    return status


def _main_impl(args: argparse.Namespace) -> int:
    if args.worker_mode:
        return _run_worker_mode(args)

    if not args.output_root:
        raise ValueError("--output-root is required outside --worker-mode.")
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    pass_ks = _parse_int_list(args.pass_ks)
    if max(pass_ks) > int(args.sample_count):
        raise ValueError("--sample-count must be at least max(pass@k).")
    mean_k = int(args.mean_k)
    if mean_k > int(args.sample_count):
        raise ValueError("--sample-count must be at least --mean-k.")
    inference_seeds = (
        _parse_seed_list(args.inference_seeds)
        if str(args.inference_seeds).strip()
        else [int(args.sample_seed)]
    )
    templates = (
        _parse_template_list(args.templates)
        if str(args.templates).strip()
        else [_normalize_template_name(args.template)]
    )
    multi_template = len(templates) > 1

    cache_root = Path(args.checkpoint_cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    chunk_sizes = DEFAULT_CHUNK_SIZES | _parse_chunk_size_overrides(args.chunk_size)
    tasks = [part.strip() for part in args.tasks.split(",") if part.strip()]
    tasks = tasks or list(TASK_ORDER)
    unknown_tasks = sorted(set(tasks).difference(TASK_ORDER))
    if unknown_tasks:
        raise ValueError(f"Unknown tasks requested: {unknown_tasks}")

    checkpoints = _resolve_checkpoints(
        checkpoint_specs=args.checkpoint,
        repo_id=args.repo_id,
        cache_root=cache_root,
        local_checkpoint_root=(
            Path(args.local_checkpoint_root).resolve()
            if str(args.local_checkpoint_root).strip()
            else None
        ),
        infer_epoch_checkpoints=args.infer_epoch_checkpoints,
        prompt_batches_per_epoch=args.prompt_batches_per_epoch,
        token=args.hf_token or None,
        force_download=bool(args.force_download),
        local_files_only=bool(args.local_files_only),
    )
    dataset_root = Path(args.dataset_root).resolve()
    shards = _build_shards(
        dataset_root=dataset_root, chunk_sizes=chunk_sizes, tasks=tasks
    )
    if multi_template:
        _write_template_sweep_manifest(
            output_root=output_root,
            repo_id=args.repo_id,
            templates=templates,
            checkpoints=checkpoints,
            shards=shards,
            pass_ks=pass_ks,
            mean_k=mean_k,
            inference_seeds=inference_seeds,
            chunk_sizes=chunk_sizes,
            args=args,
        )

    if args.dry_run:
        if multi_template:
            plan = {
                "generated_at_utc": _utc_now(),
                "output_root": str(output_root),
                "templates": [
                    {
                        "template": template,
                        "output_root": str(
                            _template_output_root(
                                output_root, template, multi_template=True
                            )
                        ),
                    }
                    for template in templates
                ],
                "checkpoints": [asdict(checkpoint) for checkpoint in checkpoints],
                "shards": [asdict(shard) for shard in shards],
                "chunk_sizes": chunk_sizes,
                "pass_ks": pass_ks,
                "mean_k": mean_k,
                "inference_seeds": inference_seeds,
                "semantic_similarity_threshold": float(
                    args.semantic_similarity_threshold
                ),
            }
        else:
            plan = {
                "generated_at_utc": _utc_now(),
                "output_root": str(output_root),
                "template": templates[0],
                "checkpoints": [asdict(checkpoint) for checkpoint in checkpoints],
                "shards": [asdict(shard) for shard in shards],
                "chunk_sizes": chunk_sizes,
                "pass_ks": pass_ks,
                "mean_k": mean_k,
                "inference_seeds": inference_seeds,
                "semantic_similarity_threshold": float(
                    args.semantic_similarity_threshold
                ),
            }
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    gpu_ids = _detect_visible_gpus()[: max(int(args.max_workers), 1)]
    if not gpu_ids:
        raise RuntimeError(
            "No visible GPUs found. Set CUDA_VISIBLE_DEVICES or run on a GPU node."
        )

    deadline_ts = None
    if args.time_budget_minutes and float(args.time_budget_minutes) > 0:
        deadline_ts = time.time() + float(args.time_budget_minutes) * 60.0

    template_statuses: dict[str, dict[str, Any]] = {}
    for template in templates:
        if deadline_ts is not None and time.time() >= (
            deadline_ts - float(args.deadline_grace_seconds)
        ):
            print(
                f"[passk-eval] stop reason=deadline template={template}",
                flush=True,
            )
            break
        template_output_root = _template_output_root(
            output_root,
            template,
            multi_template=multi_template,
        )
        template_statuses[template] = _run_single_template_eval(
            template=template,
            base_output_root=output_root,
            template_output_root=template_output_root,
            checkpoints=checkpoints,
            shards=shards,
            pass_ks=pass_ks,
            mean_k=mean_k,
            inference_seeds=inference_seeds,
            chunk_sizes=chunk_sizes,
            multi_template=multi_template,
            gpu_ids=gpu_ids,
            deadline_ts=deadline_ts,
            args=args,
        )

    if multi_template:
        status = _write_template_sweep_status(
            output_root=output_root,
            templates=templates,
            template_statuses=template_statuses,
        )
    else:
        status = template_statuses[templates[0]]

    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download dr.GRPO checkpoints, evaluate pass@k in resumable shard-sized chunks, "
            "save every attempt, and upload the resulting dataset to W&B."
        )
    )
    parser.add_argument("--repo-id", default="")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Repeatable alias=step or alias=/local/path.",
    )
    parser.add_argument("--infer-epoch-checkpoints", type=int, default=None)
    parser.add_argument("--prompt-batches-per-epoch", type=int, default=None)
    parser.add_argument(
        "--checkpoint-cache-root", default="var/cache/passk_eval_checkpoints"
    )
    parser.add_argument("--local-checkpoint-root", default="")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dataset-root", default="datasets/evaluation_suite")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--tasks", default=",".join(TASK_ORDER))
    parser.add_argument("--chunk-size", action="append", default=[])
    parser.add_argument("--template", default="r1")
    parser.add_argument("--templates", default="")
    parser.add_argument("--pass-ks", default="1,8,16,32,64,128")
    parser.add_argument("--mean-k", type=int, default=8)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--sample-top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--greedy-batch-size", type=int, default=8)
    parser.add_argument("--sampled-batch-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--swap-space", type=float, default=32.0)
    parser.add_argument("--inference-seeds", default="")
    parser.add_argument("--sample-seed", type=int, default=20260415)
    parser.add_argument("--include-token-ids", action="store_true")
    parser.add_argument("--semantic-similarity-threshold", type=float, default=0.75)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--time-budget-minutes", type=float, default=0.0)
    parser.add_argument("--deadline-grace-seconds", type=float, default=300.0)
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-wandb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--wandb-project", default="oat-zero")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-artifact-name", default="")
    parser.add_argument("--dry-run", action="store_true")

    # Hidden worker-mode args.
    parser.add_argument("--worker-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-manifest", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-shard-label", action="append", default=[], help=argparse.SUPPRESS
    )
    parser.add_argument("--deadline-ts", default="", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return _main_impl(args)


if __name__ == "__main__":
    raise SystemExit(main())
