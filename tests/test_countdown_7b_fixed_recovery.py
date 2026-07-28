"""Contract for the narrow Countdown-7B fixed-xDr recovery."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "ops/exp_scaling/restart_countdown_7b_fixed_xdr.sh"


def test_recovery_is_exactly_fixed_xdr_seeds_43_and_44():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "OAT_ZERO_TRAIN_SEEDS=43,44" in text
    assert "OAT_ZERO_ONLY_ARMS=xdr_tau0p05" in text
    assert "OAT_ZERO_SBATCH_HOLD=1" in text
    assert "OAT_ZERO_TRAIN_GRES=gpu:a6000:2" in text
    assert "OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=16" in text
    assert "OAT_ZERO_VLLM_GPU_RATIO=0.25" in text
    assert "OAT_ZERO_VLLM_SLEEP=1" in text
    assert '"PYTORCH_CUDA_ALLOC_CONF="' in text
    assert "expandable_segments:True" not in text
    assert "OAT_ZERO_OPS_SNAPSHOT_ROOT=${runtime_ops_root}" in text
    assert "slurm/train_node302.slurm" in text
    assert "released seeds 43/44" in text


def test_countdown_launcher_allows_recovery_placement_override():
    text = (
        ROOT / "ops/exp_scaling/launch_e4_tau_control_7b_countdown.sh"
    ).read_text(encoding="utf-8")

    assert 'OAT_ZERO_TRAIN_GRES="${OAT_ZERO_TRAIN_GRES:-gpu:a6000:2}"' in text
    assert 'OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_TRAIN_PARTITION:-cs}"' in text
    assert 'OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_TRAIN_ACCOUNT:-allcs}"' in text
