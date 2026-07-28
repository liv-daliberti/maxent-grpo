"""Contract for the E24 canonical graph-coloring-7B cohort."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "ops/exp_scaling/launch_e24_canonical_maxent_7b_graph.sh"
PROTOCOL = ROOT / "paper/preregistration/e24_canonical_maxent_7b_graph.md"


def test_e24_is_canonical_graph_coloring_not_e11_freeform():
    launch = LAUNCHER.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")

    assert "gce24_canonical_maxent_7b_v2_4xa100" in launch
    assert "gce11_standard_maxent_7b_v1" not in launch
    assert "OAT_ZERO_CANONICAL_ACTION_TASK=graph_coloring" in launch
    assert "OAT_ZERO_CANONICAL_GRAPH_ACTIONS=1" in launch
    assert "does not revive E11" in protocol


def test_e24_matches_e17_treatments_and_graph_budget():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "OAT_ZERO_TRAIN_SEEDS=43,44,45" in text
    assert "OAT_ZERO_ONLY_ARMS=maxent,maxent_control,maxent_dual" in text
    assert "OAT_ZERO_NUM_SAMPLES=16" in text
    assert "OAT_ZERO_NUM_PROMPT_EPOCH=5" in text
    assert "OAT_ZERO_MAX_TRAIN=15344" in text
    assert "OAT_ZERO_MAXENT_FIXED_ALPHA=0.10" in text
    assert text.count("2.7974740052946436") == 2
    assert "OAT_ZERO_SAVE_STEPS=48" in text


def test_e24_stages_audits_and_releases_nine_jobs_on_validated_layout():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "OAT_ZERO_SBATCH_HOLD=1" in text
    assert "E24 cohort incomplete (${#job_ids[@]}/9)" in text
    assert "Reason=JobHeldUser" in text
    assert "ReqNodeList=node302" in text
    assert "TresPerNode=gres/gpu:a100:4" in text
    assert "mem=192G" in text
    assert "OAT_ZERO_N_GPU=4" in text
    assert "OAT_ZERO_NUM_GPUS_PER_ACTOR=4" in text
    assert "OAT_ZERO_ROLLOUT_BATCH_SIZE=4" in text
    assert "OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=4" in text
    assert "OAT_ZERO_VLLM_SLEEP=1" in text
    assert 'scontrol release "${job_ids[@]}"' in text
