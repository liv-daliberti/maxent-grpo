"""Contract for E29's exact-group matched free-form four-GPU 7B cohort."""

import importlib.util
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_e29_modebench_freeform_7b_4gpu.sh"
PROTOCOL = ROOT / "paper/preregistration/e29_modebench_freeform_7b_4gpu.md"
RUNNER = ROOT / "ops/run_experiment.sh"
SUBMITTER = ROOT / "ops/submit_countdown_comparative.sh"
_CHECKER_PATH = ROOT / "ops/exp_scaling/check_e14_preflight.py"
_CHECKER_SPEC = importlib.util.spec_from_file_location("e29_source_hash", _CHECKER_PATH)
assert _CHECKER_SPEC is not None and _CHECKER_SPEC.loader is not None
_CHECKER = importlib.util.module_from_spec(_CHECKER_SPEC)
_CHECKER_SPEC.loader.exec_module(_CHECKER)
source_tree_hash = _CHECKER.source_tree_hash


def test_e29_has_both_matched_arms_tasks_and_three_seeds():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "gce29_freeform_7b_4gpu_v5_buffer_restore" in text
    assert "cde29_freeform_7b_4gpu_v5_buffer_restore" in text
    assert "v4_discard_sync_export_fix" not in text
    assert "OAT_ZERO_ONLY_ARMS=grpo,maxent_dual" in text
    assert "OAT_ZERO_TRAIN_SEEDS=43,44,45" in text
    assert "submit_task graph_coloring" in text
    assert "submit_task countdown" in text
    assert "${#job_ids[@]} != 12" in text


def test_e29_v5_pins_the_repaired_source_tree():
    text = LAUNCHER.read_text(encoding="utf-8")
    expected = re.search(r"^SOURCE_HASH=([0-9a-f]{64})$", text, re.MULTILINE)
    assert expected is not None
    assert expected.group(1) == source_tree_hash(
        ROOT / "src", logical_repo_root=ROOT
    )


def test_e29_preserves_group_16_on_four_gpus():
    text = LAUNCHER.read_text(encoding="utf-8")
    for value in (
        "OAT_ZERO_NUM_SAMPLES=16",
        "OAT_ZERO_TRAIN_BATCH_SIZE=16",
        "OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=4",
        "OAT_ZERO_ROLLOUT_BATCH_SIZE=4",
        "OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE=1",
        "OAT_ZERO_N_GPU=4",
        "OAT_ZERO_NUM_GPUS_PER_ACTOR=1",
        "OAT_ZERO_REPLICATED_FREEFORM_SAMPLING=1",
        "OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC=1",
        "OAT_ZERO_VLLM_SLEEP_LEVEL=2",
    ):
        assert f"export {value}" in text
    protocol = " ".join(PROTOCOL.read_text(encoding="utf-8").split())
    assert "does not change the one-prompt, group-16 optimizer recipe" in protocol


def test_e29_uses_the_ten_pass_campaign_boundary():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "export OAT_ZERO_NUM_PROMPT_EPOCH=10" in text
    assert 'OAT_ZERO_NUM_PROMPT_EPOCH:-10' in RUNNER.read_text(encoding="utf-8")
    assert 'OAT_ZERO_NUM_PROMPT_EPOCH:-10' in SUBMITTER.read_text(encoding="utf-8")
    protocol = " ".join(PROTOCOL.read_text(encoding="utf-8").split())
    assert "extended from five to ten complete prompt-pool passes" in protocol


def test_e29_matches_e25_dual_and_e28_control():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "OAT_ZERO_MAXENT_OBJECTIVE=conditional_token_mean" in text
    assert "OAT_ZERO_MAXENT_DUAL_MIN_ALPHA=0.000075" in text
    assert "OAT_ZERO_MAXENT_DUAL_MAX_ALPHA=0.00060" in text
    assert "OAT_ZERO_MAXENT_DUAL_ALPHA_LR=0.010" in text
    assert "GRAPH_TARGET=1.622718550885717" in text
    assert "COUNTDOWN_TARGET=1.347109432487438" in text
    assert "OAT_ZERO_CANONICAL_ACTION_TASK=none" in text


def test_e29_submits_held_single_node_four_a100_jobs():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "OAT_ZERO_SBATCH_HOLD=1" in text
    assert "OAT_ZERO_E29_TRAIN_NODELIST:-node302" in text
    assert "OAT_ZERO_E29_TRAIN_GRES:-gpu:a100:4" in text
    assert "OAT_ZERO_E29_TRAIN_MEMORY:-256G" in text
    assert "jobs remain held" in text


def test_e29_v3_discards_actor_weights_during_cpu_optimizer_work():
    protocol = " ".join(PROTOCOL.read_text(encoding="utf-8").split())
    assert "using vLLM level-2 sleep" in protocol
    assert "not an experimental invariant" in protocol


def test_e29_has_audited_48gb_cache_ratio_recovery():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "config|full|recover48" in text
    assert "OAT_ZERO_VLLM_GPU_RATIO=0.40" in text
    assert "E29 48GB recovery incomplete" in text
    assert "OAT_ZERO_TRAIN_GRES=gpu:a6000:4" in text
