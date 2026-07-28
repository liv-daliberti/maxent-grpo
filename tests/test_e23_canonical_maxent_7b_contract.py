"""Contract for the E23 canonical Countdown-7B cohort."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "ops/exp_scaling/launch_e23_canonical_maxent_7b_countdown.sh"
PROTOCOL = ROOT / "paper/preregistration/e23_canonical_maxent_7b_countdown.md"


def test_e23_is_canonical_and_not_the_failed_e11_freeform_grid():
    launch = LAUNCHER.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")

    assert "cde23_canonical_maxent_7b_v4_4xa100" in launch
    assert "cde11_standard_maxent_7b_v1" not in launch
    assert "OAT_ZERO_CANONICAL_ACTION_TASK=countdown" in launch
    assert "OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING=1" in launch
    assert "OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING=1" in launch
    assert "does not revive E11" in protocol


def test_e23_matches_the_e17_treatment_and_five_pass_budget():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "OAT_ZERO_TRAIN_SEEDS=43,44,45" in text
    assert "OAT_ZERO_ONLY_ARMS=maxent,maxent_control,maxent_dual" in text
    assert "OAT_ZERO_NUM_SAMPLES=16" in text
    assert "OAT_ZERO_NUM_PROMPT_EPOCH=5" in text
    assert "OAT_ZERO_MAX_TRAIN=30704" in text
    assert "OAT_ZERO_MAXENT_FIXED_ALPHA=0.10" in text
    assert "OAT_ZERO_MAXENT_CONTROL_BASE_ALPHA=0.075" in text
    assert "OAT_ZERO_MAXENT_DUAL_BASE_ALPHA=0.075" in text
    assert text.count("3.9741470618167156") == 2


def test_e23_stages_audits_and_releases_exactly_nine_jobs():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "OAT_ZERO_SBATCH_HOLD=1" in text
    assert "E23 cohort incomplete (${#job_ids[@]}/9)" in text
    assert "Reason=JobHeldUser" in text
    assert "ReqNodeList=node302" in text
    assert "TresPerNode=gres/gpu:a100:4" in text
    assert "mem=192G" in text
    assert 'scontrol release "${job_ids[@]}"' in text


def test_e23_uses_validated_7b_sleep_offload_layout():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "OAT_ZERO_N_GPU=4" in text
    assert "OAT_ZERO_NUM_GPUS_PER_ACTOR=4" in text
    assert "OAT_ZERO_ROLLOUT_BATCH_SIZE=4" in text
    assert "OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=4" in text
    assert "OAT_ZERO_VLLM_GPU_RATIO=0.25" in text
    assert "OAT_ZERO_VLLM_SLEEP=1" in text
    assert "OAT_ZERO_ADAM_OFFLOAD=1" in text
    assert "OAT_ZERO_ACTIVATION_OFFLOADING=1" in text
    assert "unset PYTORCH_CUDA_ALLOC_CONF" in text
