"""Contract for the four-A100 canonical E23/E24 source amendment."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE_HASH = "044f6df047788dc8b67bbe224281a403c6d5eab04de89881f5a17cfe5c147cf9"
SOURCE = (
    ROOT
    / "var/artifacts/source_snapshots"
    / f"e23_e24_canonical_4gpu_fix_{SOURCE_HASH}"
    / "src/oat_drgrpo"
)
OPS = (
    ROOT
    / "var/artifacts/source_snapshots"
    / "e23_e24_canonical_4gpu_ops_72abacc11178931c3d3923c691fa303d06e45aec0dfef686e87ec607ef4bd808"
    / "ops"
)
COUNTDOWN_FIX_LAUNCHER = (
    ROOT / "ops/exp_scaling/launch_e23_canonical_maxent_7b_countdown_v5_fix.sh"
)
GRAPH_FIX_LAUNCHER = (
    ROOT / "ops/exp_scaling/launch_e24_canonical_maxent_7b_graph_v3_fix.sh"
)


def test_four_gpu_args_freeze_replicated_prompt_and_exact_group_partition():
    text = (SOURCE / "args.py").read_text(encoding="utf-8")

    assert "rollout_batch_size) != gpu_count" in text
    assert "rollout_batch_size_per_device) != 1" in text
    assert "num_gpus_per_actor) != gpu_count" in text
    assert "train_batch_size) != int(args.num_samples)" in text
    assert "train_batch_size_per_device) * gpu_count" in text


def test_four_gpu_sampler_replicates_prompt_and_shards_candidates_for_backward():
    run = (SOURCE / "learner/run.py").read_text(encoding="utf-8")
    grpo = (SOURCE / "learner/grpo.py").read_text(encoding="utf-8")

    assert "num_replicas=1" in run
    assert "rank=0" in run
    assert "unique_local_count = len(feedback_data) // replication_factor" in run
    assert "replicated_canonical" in grpo
    assert "permutation_seed" in grpo
    assert "shard_start = rank * local_candidate_count" in grpo
    assert "canonical rank shard must equal train_batch_size_per_device" in grpo
    assert "0, len(batch_inds), args.train_batch_size_per_device" in grpo


def test_four_gpu_rank_shards_produce_no_empty_microbatches():
    group_size = 16
    world_size = 4
    per_device_batch = 4
    permutation = list(range(group_size))

    shards = [
        permutation[rank * per_device_batch : (rank + 1) * per_device_batch]
        for rank in range(world_size)
    ]
    microbatches = [
        shard[start : start + per_device_batch]
        for shard in shards
        for start in range(0, len(shard), per_device_batch)
    ]

    assert sorted(item for batch in microbatches for item in batch) == permutation
    assert len(microbatches) == world_size
    assert all(len(batch) == per_device_batch for batch in microbatches)


def test_four_gpu_submitter_materializes_topology_in_slurm_record():
    text = (OPS / "submit_countdown_comparative.sh").read_text(encoding="utf-8")

    for variable in (
        "OAT_ZERO_TRAIN_BATCH_SIZE",
        "OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE",
        "OAT_ZERO_ROLLOUT_BATCH_SIZE",
        "OAT_ZERO_N_GPU",
        "OAT_ZERO_NUM_GPUS_PER_ACTOR",
        "OAT_ZERO_ZERO_STAGE",
        "OAT_ZERO_ADAM_OFFLOAD",
        "OAT_ZERO_VLLM_SLEEP",
    ):
        assert f'export_vars+=",{variable}=' in text


def test_active_submitter_materializes_sync_cadence_in_slurm_record():
    text = (ROOT / "ops/submit_countdown_comparative.sh").read_text(
        encoding="utf-8"
    )

    assert 'export_vars+=",OAT_ZERO_SYNC_PARAMS_EVERY=' in text
    for launcher in (COUNTDOWN_FIX_LAUNCHER, GRAPH_FIX_LAUNCHER):
        launch = launcher.read_text(encoding="utf-8")
        assert 'SUBMITTER="$ROOT_DIR/ops/submit_countdown_comparative.sh"' in launch
        assert 'bash "$SUBMITTER"' in launch


def test_fix_launchers_pin_new_source_and_default_to_held():
    for launcher, stamp in (
        (COUNTDOWN_FIX_LAUNCHER, "cde23_canonical_maxent_7b_v6_4xa100_evalsync_fix"),
        (GRAPH_FIX_LAUNCHER, "gce24_canonical_maxent_7b_v4_4xa100_evalsync_fix"),
    ):
        text = launcher.read_text(encoding="utf-8")
        assert f"SOURCE_HASH={SOURCE_HASH}" in text
        assert "e23_e24_canonical_4gpu_fix_${SOURCE_HASH}" in text
        assert f"STAMP={stamp}" in text
        assert 'OAT_ZERO_7B_FIX_RELEASE_MODE:-held' in text


def test_four_gpu_launchers_express_prompt_cadence_in_learner_steps():
    countdown = COUNTDOWN_FIX_LAUNCHER.read_text(encoding="utf-8")
    graph = GRAPH_FIX_LAUNCHER.read_text(encoding="utf-8")

    for name in ("SYNC_PARAMS_EVERY", "SAVE_STEPS", "SAVE_FROM"):
        assert f"OAT_ZERO_{name}=24" in countdown
        assert f"OAT_ZERO_{name}=12" in graph
