"""Contracts for narrow graph-coloring 3B idle-GPU recoveries."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "ops/exp_scaling/restart_graph_3b_idle_gpu_canaries.sh"


def test_recovery_submits_only_two_named_cells_held_then_releases():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "OAT_ZERO_TRAIN_SEEDS=45" in text
    assert "OAT_ZERO_ONLY_ARMS=xdr_tau_control" in text
    assert "OAT_ZERO_TRAIN_SEEDS=43" in text
    assert "OAT_ZERO_ONLY_ARMS=xdr_sac_dual" in text
    assert "OAT_ZERO_SBATCH_HOLD=1" in text
    assert 'scontrol release "$feedback_id" "$dual_id"' in text


def test_recovery_uses_separate_idle_pools_and_safe_runtime():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "OAT_ZERO_TRAIN_NODELIST=node302" in text
    assert "OAT_ZERO_TRAIN_GRES=gpu:a100:2" in text
    assert "OAT_ZERO_TRAIN_NODELIST=node204" in text
    assert "OAT_ZERO_TRAIN_GRES=gpu:a5000:2" in text
    assert "OAT_ZERO_TRAIN_PARTITION=lowprio" in text
    assert "OAT_ZERO_TRAIN_ACCOUNT=mltheory" in text
    assert "OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=4" in text
    assert "OAT_ZERO_TRAIN_MEMORY=96G" in text
    assert "OAT_ZERO_OPS_SNAPSHOT_ROOT=${ops_root}" in text
    assert "OAT_ZERO_VLLM_SLEEP=1" in text
    assert '"PYTORCH_CUDA_ALLOC_CONF="' in text
    assert "expandable_segments:True" not in text
    assert "dual-config|dual-retry" in text
