import json
from pathlib import Path


from oat_drgrpo.passk_eval import (
    ShardSpec,
    _average_metric_dicts,
    _assign_shards_to_workers,
    _compute_prefix_metrics,
    _existing_summary_is_complete,
    _list_local_checkpoint_steps,
    _parse_seed_list,
    _parse_template_list,
    _resolve_checkpoints,
    _select_epoch_boundary_steps,
)


def test_select_epoch_boundary_steps_for_completed_seed43_shape():
    available_steps = [0] + list(range(16, 465, 16)) + [466]
    selected = _select_epoch_boundary_steps(available_steps, num_epochs=5)
    assert selected == [
        ("pretrain", "step_00000"),
        ("epoch1", "step_00096"),
        ("epoch2", "step_00192"),
        ("epoch3", "step_00288"),
        ("epoch4", "step_00384"),
        ("epoch5", "step_00466"),
    ]


def test_compute_prefix_metrics_uses_prefixes_and_mean_at_8():
    rewards = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    metrics = _compute_prefix_metrics(rewards, pass_ks=[1, 8], mean_k=8)
    assert metrics["pass_at_1"] == 0.0
    assert metrics["pass_at_8"] == 1.0
    assert metrics["sampled_pass_at_1"] == 0.0
    assert metrics["mean_at_8"] == 0.125


def test_assign_shards_to_workers_balances_heaviest_shards_first():
    shards = [
        ShardSpec(label="math_1", task="math", start=0, end=8, weight=8),
        ShardSpec(label="math_2", task="math", start=8, end=16, weight=8),
        ShardSpec(label="olympiad_1", task="olympiad_bench", start=0, end=6, weight=6),
        ShardSpec(label="aime_1", task="aime", start=0, end=6, weight=6),
    ]
    buckets = _assign_shards_to_workers(shards, worker_count=2)
    assert len(buckets) == 2
    flattened = {shard.label for bucket in buckets for shard in bucket}
    assert flattened == {"math_1", "math_2", "olympiad_1", "aime_1"}


def test_parse_seed_list_allows_zero_and_deduplicates():
    assert _parse_seed_list("4,0,2,2,1,3") == [0, 1, 2, 3, 4]


def test_parse_template_list_normalizes_aliases_and_preserves_order():
    assert _parse_template_list("qwen-boxed,qwen-math,no,raw,r1,qwen_math") == [
        "qwen_boxed",
        "qwen_math",
        "no",
        "r1",
    ]


def test_average_metric_dicts_means_over_inference_seeds():
    averaged = _average_metric_dicts(
        [
            {"pass_at_8": 0.0, "mean_at_8": 0.125, "formatted_rate": None},
            {"pass_at_8": 1.0, "mean_at_8": 0.25, "formatted_rate": 1.0},
        ]
    )
    assert averaged["pass_at_8"] == 0.5
    assert averaged["mean_at_8"] == 0.1875
    assert averaged["formatted_rate"] == 1.0


def test_existing_summary_requires_matching_inference_seeds(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "complete": True,
                "inference_seeds": [0, 1, 2, 3, 4],
                "by_inference_seed": {
                    str(seed): {"metrics": {"pass_at_8": 0.0}} for seed in range(5)
                },
            }
        )
    )
    assert _existing_summary_is_complete(
        summary_path,
        expected_inference_seeds=[0, 1, 2, 3, 4],
    )
    assert not _existing_summary_is_complete(
        summary_path,
        expected_inference_seeds=[0, 1, 2, 3],
    )


def test_list_local_checkpoint_steps_checks_saved_models(tmp_path: Path):
    root = tmp_path / "seed43"
    (root / "saved_models" / "step_00000").mkdir(parents=True)
    (root / "saved_models" / "step_00000" / "config.json").write_text("{}")
    (root / "saved_models" / "step_00096").mkdir(parents=True)
    (root / "saved_models" / "step_00096" / "config.json").write_text("{}")
    assert _list_local_checkpoint_steps(root) == ["step_00000", "step_00096"]


def test_resolve_checkpoints_prefers_local_checkpoint_root(tmp_path: Path):
    local_root = tmp_path / "seed43"
    step0 = local_root / "saved_models" / "step_00000"
    step96 = local_root / "saved_models" / "step_00096"
    step0.mkdir(parents=True)
    step96.mkdir(parents=True)
    (step0 / "config.json").write_text("{}")
    (step96 / "config.json").write_text("{}")

    checkpoints = _resolve_checkpoints(
        checkpoint_specs=[],
        repo_id="",
        cache_root=tmp_path / "cache",
        local_checkpoint_root=local_root,
        infer_epoch_checkpoints=1,
        prompt_batches_per_epoch=95,
        token=None,
        force_download=False,
        local_files_only=False,
    )
    assert [
        (checkpoint.alias, checkpoint.local_path) for checkpoint in checkpoints
    ] == [
        ("pretrain", str(step0.resolve())),
        ("epoch1", str(step96.resolve())),
    ]


def test_resolve_checkpoints_falls_back_to_remote_when_local_root_is_incomplete(
    tmp_path: Path,
    monkeypatch,
):
    local_root = tmp_path / "seed43"
    final_step = local_root / "saved_models" / "step_00466"
    final_step.mkdir(parents=True)
    (final_step / "config.json").write_text("{}")

    remote_cache_root = (
        tmp_path / "cache" / "repo" / "step_00000" / "checkpoints" / "step_00000"
    )
    remote_cache_root.mkdir(parents=True)
    (remote_cache_root / "config.json").write_text("{}")

    monkeypatch.setattr(
        "oat_drgrpo.passk_eval._list_remote_checkpoint_steps",
        lambda repo_id, token=None: ["step_00000", "step_00466"],
    )
    monkeypatch.setattr(
        "oat_drgrpo.passk_eval._download_remote_checkpoint",
        lambda **kwargs: remote_cache_root,
    )

    checkpoints = _resolve_checkpoints(
        checkpoint_specs=[],
        repo_id="od2961/test",
        cache_root=tmp_path / "cache",
        local_checkpoint_root=local_root,
        infer_epoch_checkpoints=1,
        prompt_batches_per_epoch=465,
        token=None,
        force_download=False,
        local_files_only=False,
    )
    assert [checkpoint.alias for checkpoint in checkpoints] == ["pretrain", "epoch1"]
    assert checkpoints[0].local_path == str(remote_cache_root.resolve())
    assert checkpoints[1].local_path == str(final_step.resolve())
