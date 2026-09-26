from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_e32_countdown_preemption_replacement.sh"
PROTOCOL = ROOT / "paper/preregistration/e32_countdown_preemption_replacement_20260722.md"


def test_replacement_is_fresh_nonpreemptible_and_checkpoint_aligned():
    text = LAUNCHER.read_text(encoding="utf-8")
    for literal in (
        "cde32_freeform_05b_ema_10ep_v4_preemptsafe",
        "OAT_ZERO_ONLY_ARMS=grpo,maxent_dual",
        "OAT_ZERO_TRAIN_SEEDS=43,44,45",
        "OAT_ZERO_NUM_PROMPT_EPOCH=10",
        "OAT_ZERO_MAX_PROMPT_EPOCHS=10",
        "OAT_ZERO_EVAL_PROMPT_INTERVAL=96",
        "OAT_ZERO_SAVE_STEPS=96",
        "OAT_ZERO_RESUME_STEPS=96",
        "OAT_ZERO_TRAIN_PARTITION=cs",
        "OAT_ZERO_TRAIN_ACCOUNT=allcs",
        "OAT_ZERO_TRAIN_NODELIST=node203,node204",
        "OAT_ZERO_SBATCH_HOLD=1",
    ):
        assert literal in text


def test_protocol_excludes_preempted_identity():
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "**Status: FROZEN" in text
    assert "must not be combined" in text
    assert "non-preemptible `cs`" in text
    assert "hard-coded five-epoch cap" in text
    assert "zero elapsed runtime" in text


def test_audit_accepts_only_the_two_slurm_spellings_of_the_frozen_nodes():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "ReqNodeList=node[203-204]" in text
    assert "ReqNodeList=node203,node204" in text
