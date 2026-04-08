"""Tests for the repository Slurm training launcher."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_archived_ops_script(name: str) -> Path:
    repo_root = _repo_root()
    candidates = [
        repo_root / "archive" / "trl" / "ops" / name,
        repo_root / "ops" / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Archived ops script not found: {name}")


def _resolve_archived_ops_slurm(name: str) -> Path:
    repo_root = _repo_root()
    candidates = [
        repo_root / "archive" / "trl" / "ops" / "slurm" / name,
        repo_root / "ops" / "slurm" / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Archived slurm launcher not found: {name}")


def _resolve_train_slurm() -> Path:
    return _resolve_archived_ops_slurm("train_dual_4plus4.slurm")


def _resolve_tiny_smoke_slurm() -> Path:
    return _resolve_archived_ops_slurm("train_tiny_gpu_smoke.slurm")


def test_slurm_launcher_uses_shared_grpo_entrypoint():
    script = _resolve_train_slurm().read_text()
    assert 'launch_train "grpo-train" "src/maxent_grpo/grpo.py"' in script
    assert 'launch_train "${SECONDARY_STACK_TAG}-train" "src/maxent_grpo/grpo.py"' in script
    assert "/grpo/config_" in script
    assert "/maxent-grpo/config_" in script
    assert "maxent_entropy_" in script
    assert "maxent_listwise_" in script
    assert "seed_grpo_" in script
    assert "RUN_ONLY must be both|grpo|maxent|listwise|seed" in script
    assert 'if [[ "$RUN_ONLY" == "grpo" || "$RUN_ONLY" == "maxent" || "$RUN_ONLY" == "listwise" || "$RUN_ONLY" == "seed" ]]; then' in script
    assert "/n/fs/similarity/kalshi" not in script
    assert 'REPO_RUNTIME_SITE_PACKAGES="${REPO_RUNTIME_SITE_PACKAGES:-$VAR_DIR/e2e-venv/lib/python3.11/site-packages:$VAR_DIR/seed_paper_eval/paper_venv/lib/python3.11/site-packages}"' in script
    assert 'EXTRA_SITE_PACKAGES="${EXTRA_SITE_PACKAGES:-$REPO_RUNTIME_SITE_PACKAGES}"' in script
    assert 'WANDB_DATA_DIR="${WANDB_DATA_DIR:-$WANDB_ROOT/data}"' in script
    assert 'WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$WANDB_ROOT/artifacts}"' in script
    assert 'WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"' in script
    assert 'WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"' in script
    assert 'WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"' in script
    assert 'python -c "import trl, vllm"' not in script
    assert "#SBATCH --exclusive" not in script
    assert 'python "$ROOT_DIR/tools/seed_paper_eval.py"' in script
    assert 'export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"' in script
    assert 'STEP0_PAPER_EVAL_EXPECTED_PROFILE="table2_qwen2_5_math_1_5b"' in script
    assert 'STEP0_PAPER_EVAL_WANDB_PROJECT="${MAXENT_STEP0_PAPER_EVAL_WANDB_PROJECT:-${WANDB_PROJECT:-huggingface}}"' in script
    assert 'cmd+=(--wandb)' in script
    assert 'echo "[step0-paper-eval] completed"' in script
    assert 'JOB_PORT_OFFSET=$(( (JOB_PORT_SEED % 1000) * 10 ))' in script
    assert 'GRPO_VLLM_PORT_DEFAULT=$((18000 + JOB_PORT_OFFSET))' in script
    assert 'MAXENT_VLLM_PORT_DEFAULT=$((18001 + JOB_PORT_OFFSET))' in script
    assert 'GRPO_MASTER_PORT_DEFAULT=$((38000 + JOB_PORT_OFFSET))' in script
    assert 'MAXENT_MASTER_PORT_DEFAULT=$((38001 + JOB_PORT_OFFSET))' in script


def test_slurm_launcher_help_text(tmp_path):
    env = os.environ.copy()
    env["VAR_DIR"] = str(tmp_path / "var")
    env.setdefault("HF_TOKEN", "test")
    script_path = _resolve_train_slurm()
    cmd = ["bash", str(script_path), "--help"]
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[4],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Usage:" in result.stdout
    assert ("--run-only" in result.stdout) or ("--task" in result.stdout)


def test_tiny_smoke_launcher_keeps_site_packages_repo_local():
    script = _resolve_tiny_smoke_slurm().read_text()
    assert "/n/fs/similarity/kalshi" not in script
    assert 'ENV_ACTIVATE="${ENV_ACTIVATE:-$ROOT_DIR/var/openr1/bin/activate}"' in script
    assert 'VENV_SITE_PACKAGES="${VENV_SITE_PACKAGES:-$ROOT_DIR/var/openr1/lib/python3.11/site-packages}"' in script
    assert 'EXTERNAL_SITE_PACKAGES="${EXTERNAL_SITE_PACKAGES:-}"' in script


def test_active_oat_runner_requires_eval_unless_explicitly_overridden() -> None:
    script = (_repo_root() / "ops" / "run_oat_zero_exact_1p5b_upstream.sh").read_text()
    assert 'EVAL_STEPS="${OAT_ZERO_EVAL_STEPS:-16}"' in script
    assert 'SAVE_STEPS="${OAT_ZERO_SAVE_STEPS:-$EVAL_STEPS}"' in script
    assert 'SAVE_FROM="${OAT_ZERO_SAVE_FROM:-0}"' in script
    assert 'SAVE_INITIAL_MODEL="${OAT_ZERO_SAVE_INITIAL_MODEL:-1}"' in script
    assert 'MAX_SAVE_NUM="${OAT_ZERO_MAX_SAVE_NUM:-999999}"' in script
    assert 'MAX_SAVE_MEM="${OAT_ZERO_MAX_SAVE_MEM:-99999999}"' in script
    assert 'ALLOW_NO_SAVE="${OAT_ZERO_ALLOW_NO_SAVE:-0}"' in script
    assert 'ALLOW_NO_EVAL="${OAT_ZERO_ALLOW_NO_EVAL:-0}"' in script
    assert 'MAX_EVAL_STEPS_ALLOWED="${OAT_ZERO_MAX_EVAL_STEPS_ALLOWED:-1000}"' in script
    assert 'Checkpoint saving must stay enabled: OAT_ZERO_SAVE_STEPS must be > 0.' in script
    assert 'Eval must stay enabled: OAT_ZERO_EVAL_STEPS must be > 0.' in script
    assert 'Eval is effectively disabled: OAT_ZERO_EVAL_STEPS=${EVAL_STEPS} exceeds OAT_ZERO_MAX_EVAL_STEPS_ALLOWED=${MAX_EVAL_STEPS_ALLOWED}.' in script
    assert 'echo "[oat-zero-exact] save_steps=${SAVE_STEPS}"' in script
    assert 'echo "[oat-zero-exact] save_initial_model=${SAVE_INITIAL_MODEL}"' in script
    assert 'echo "[oat-zero-exact] max_save_num=${MAX_SAVE_NUM}"' in script
    assert 'echo "[oat-zero-exact] max_save_mem=${MAX_SAVE_MEM}"' in script
    assert 'echo "[oat-zero-exact] allow_no_eval=${ALLOW_NO_EVAL}"' in script
    assert 'echo "[oat-zero-exact] max_eval_steps_allowed=${MAX_EVAL_STEPS_ALLOWED}"' in script


def test_active_oat_wrappers_make_eval_steps_explicit() -> None:
    baseline = (
        _repo_root()
        / "ops"
        / "slurm"
        / "train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_node302.slurm"
    ).read_text()
    explorer = (
        _repo_root()
        / "ops"
        / "slurm"
        / "train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_explorer_node302.slurm"
    ).read_text()
    explorer_runner = (_repo_root() / "ops" / "run_oat_zero_explorer_1p5b_upstream.sh").read_text()
    assert 'export OAT_ZERO_EVAL_STEPS="${OAT_ZERO_EVAL_STEPS:-16}"' in baseline
    assert 'export OAT_ZERO_SAVE_STEPS="${OAT_ZERO_SAVE_STEPS:-${OAT_ZERO_EVAL_STEPS}}"' in baseline
    assert 'export OAT_ZERO_SAVE_INITIAL_MODEL="${OAT_ZERO_SAVE_INITIAL_MODEL:-1}"' in baseline
    assert 'export OAT_ZERO_MAX_SAVE_NUM="${OAT_ZERO_MAX_SAVE_NUM:-999999}"' in baseline
    assert 'export OAT_ZERO_MAX_SAVE_MEM="${OAT_ZERO_MAX_SAVE_MEM:-99999999}"' in baseline
    assert 'echo "[slurm] oat_zero_eval_steps=${OAT_ZERO_EVAL_STEPS}"' in baseline
    assert 'echo "[slurm] oat_zero_save_steps=${OAT_ZERO_SAVE_STEPS}"' in baseline
    assert 'echo "[slurm] oat_zero_save_initial_model=${OAT_ZERO_SAVE_INITIAL_MODEL}"' in baseline
    assert 'export OAT_ZERO_EVAL_STEPS="${OAT_ZERO_EVAL_STEPS:-16}"' in explorer
    assert 'export OAT_ZERO_SAVE_STEPS="${OAT_ZERO_SAVE_STEPS:-${OAT_ZERO_EVAL_STEPS}}"' in explorer
    assert 'export OAT_ZERO_SAVE_INITIAL_MODEL="${OAT_ZERO_SAVE_INITIAL_MODEL:-1}"' in explorer
    assert 'export OAT_ZERO_MAX_SAVE_NUM="${OAT_ZERO_MAX_SAVE_NUM:-999999}"' in explorer
    assert 'export OAT_ZERO_MAX_SAVE_MEM="${OAT_ZERO_MAX_SAVE_MEM:-99999999}"' in explorer
    assert 'echo "[slurm] oat_zero_eval_steps=${OAT_ZERO_EVAL_STEPS}"' in explorer
    assert 'echo "[slurm] oat_zero_save_steps=${OAT_ZERO_SAVE_STEPS}"' in explorer
    assert 'echo "[slurm] oat_zero_save_initial_model=${OAT_ZERO_SAVE_INITIAL_MODEL}"' in explorer
    assert 'export OAT_ZERO_EVAL_STEPS="${OAT_ZERO_EVAL_STEPS:-16}"' in explorer_runner


def test_triplet_wrapper_submits_all_three_single_stack_runs() -> None:
    script = _resolve_archived_ops_script("run_experiment_triplet_single_node.sh").read_text()
    assert 'submit_triplet_job "listwise"' in script
    assert 'submit_triplet_job "maxent"' in script
    assert 'submit_triplet_job "grpo"' in script
    assert 'SBATCH_GRES="$SINGLE_STACK_GRES"' in script
    assert 'TRAIN_NUM_PROCESSES="$SINGLE_STACK_NUM_PROCESSES"' in script
    assert 'SBATCH_DEPENDENCY="$dependency"' in script
    assert 'GRPO_DEPENDENCY="afterany:${LISTWISE_JOB_ID}:${MAXENT_JOB_ID}"' in script
    assert 'LISTWISE_VLLM_PORT="${LISTWISE_VLLM_PORT:-8001}"' in script
    assert 'ENTROPY_VLLM_PORT="${ENTROPY_VLLM_PORT:-8002}"' in script
    assert 'GRPO_VLLM_PORT="${GRPO_VLLM_PORT:-8000}"' in script
    assert 'MAXENT_VLLM_PORT="$stack_vllm_port"' in script
    assert 'MAXENT_MASTER_PORT="$stack_master_port"' in script
    assert 'GRPO_MASTER_PORT="$GRPO_MASTER_PORT"' in script
    assert 'SINGLE_STACK_GRES="${SINGLE_STACK_GRES:-gpu:a100:4}"' in script
    assert 'SINGLE_STACK_NUM_PROCESSES="${SINGLE_STACK_NUM_PROCESSES:-3}"' in script
    assert 'SINGLE_STACK_TRAIN_GPUS="${SINGLE_STACK_TRAIN_GPUS:-1,2,3}"' in script
    assert 'RECIPE_PROFILE="${RECIPE_PROFILE:-experiment}"' in script


def test_quartet_wrapper_submits_all_four_single_stack_runs() -> None:
    script = _resolve_archived_ops_script("run_experiment_quartet_single_node.sh").read_text()
    assert 'submit_quartet_job "grpo"' in script
    assert 'submit_quartet_job "listwise"' in script
    assert 'submit_quartet_job "maxent"' in script
    assert 'submit_quartet_job "seed"' in script
    assert 'WAVE2_DEPENDENCY="afterok:${GRPO_JOB_ID}:${LISTWISE_JOB_ID}"' in script
    assert 'grpo_output_and_id="$(submit_quartet_job "grpo"' in script
    assert 'listwise_output_and_id="$(submit_quartet_job "listwise"' in script
    assert 'maxent_output_and_id="$(submit_quartet_job "maxent"' in script
    assert 'seed_output_and_id="$(submit_quartet_job "seed"' in script
    assert 'SEED_VLLM_PORT="${SEED_VLLM_PORT:-8003}"' in script
    assert 'SEED_GROUP_PORT="${SEED_GROUP_PORT:-29538}"' in script
    assert 'SEED_MASTER_PORT="${SEED_MASTER_PORT:-6003}"' in script
    assert 'MAXENT_ARGS="$variant_args"' in script
    assert 'GRPO_ARGS="$variant_args"' in script


def test_interim_pair_wrapper_routes_maxent_and_seed_to_general_compute() -> None:
    script = _resolve_archived_ops_script("run_experiment_interim_pair_single_node.sh").read_text()
    assert 'submit_interim_job "maxent"' in script
    assert 'submit_interim_job "seed"' in script
    assert 'CONFIG_SUFFIX="${CONFIG_SUFFIX:-math_fair}"' in script
    assert 'RESOURCE_PROFILE="${RESOURCE_PROFILE:-interim_a6000}"' in script
    assert 'SINGLE_STACK_PARTITION="${SINGLE_STACK_PARTITION:-lowprio}"' in script
    assert 'SINGLE_STACK_ACCOUNT="${SINGLE_STACK_ACCOUNT:-allcs}"' in script
    assert 'SINGLE_STACK_GRES="${SINGLE_STACK_GRES:-gpu:a6000:4}"' in script
    assert 'SINGLE_STACK_CPUS_PER_TASK="${SINGLE_STACK_CPUS_PER_TASK:-32}"' in script
    assert 'SINGLE_STACK_MEM="${SINGLE_STACK_MEM:-128G}"' in script
    assert 'MAXENT_JOB_NAME="${MAXENT_JOB_NAME:-cs_interim_maxent}"' in script
    assert 'SEED_JOB_NAME="${SEED_JOB_NAME:-cs_interim_seed_drgrpo}"' in script
    assert 'RESOURCE_PROFILE="$RESOURCE_PROFILE"' in script
    assert 'RUN_ONLY="$stack"' in script
    assert 'MAXENT_ARGS="$variant_args"' in script


def test_listwise_sweep_wrapper_targets_general_compute_short_grid() -> None:
    script = _resolve_archived_ops_script("run_listwise_tau_beta_sweep.sh").read_text()
    assert 'CONFIG_SUFFIX="${CONFIG_SUFFIX:-math_fair}"' in script
    assert 'RESOURCE_PROFILE="${RESOURCE_PROFILE:-interim_a6000}"' in script
    assert 'SBATCH_PARTITION="${SBATCH_PARTITION:-all}"' in script
    assert 'TAU_VALUES="${TAU_VALUES:-0.35,0.50,0.70}"' in script
    assert 'BETA_VALUES="${BETA_VALUES:-0.04,0.08,0.12}"' in script
    assert 'SWEEP_SEEDS="${SWEEP_SEEDS:-${SWEEP_SEED:-42}}"' in script
    assert 'SWEEP_MAX_STEPS="${SWEEP_MAX_STEPS:-50}"' in script
    assert 'SWEEP_EVAL_STEPS="${SWEEP_EVAL_STEPS:-25}"' in script
    assert 'SWEEP_TASKS="${SWEEP_TASKS:-aime,amc,math,minerva,olympiad_bench}"' in script
    assert 'TRAIN_LIVE_EVAL_TEMPLATE="${TRAIN_LIVE_EVAL_TEMPLATE:-no}"' in script
    assert 'RUN_ONLY="listwise"' in script
    assert '--maxent_tau $tau' in script
    assert '--beta $beta' in script
    assert '--save_strategy no' in script
    assert '--final_model_save_enabled false' in script
    assert '--seed_paper_eval_template $TRAIN_LIVE_EVAL_TEMPLATE' in script
    assert '--output_dir $output_dir' in script
    assert '--seed $seed' in script
    assert '--seed_paper_eval_tasks $SWEEP_TASKS' in script
    assert 'MAXENT_STEP0_PAPER_EVAL_TEMPLATE="$TRAIN_LIVE_EVAL_TEMPLATE"' in script
    assert 'run_name="${RUN_GROUP}-seed${seed_slug}-tau${tau_slug}-beta${beta_slug}"' in script
    assert "printf 'tau\\tbeta\\tseed\\tjob_id\\tjob_name\\trun_name\\toutput_dir\\n' > \"$MANIFEST_PATH\"" in script
    assert 'python $SCRIPT_DIR/../tools/listwise_sweep_report.py --manifest $MANIFEST_PATH' in script


def test_full_eval_pair_wrapper_propagates_listwise_knobs_and_eval_resources() -> None:
    script = _resolve_archived_ops_script("run_grpo_listwise_full_eval_pair.sh").read_text()
    assert 'CONFIG_SUFFIX="${CONFIG_SUFFIX:-math_fair}"' in script
    assert 'LISTWISE_TAU="${LISTWISE_TAU:-0.5}"' in script
    assert 'LISTWISE_BETA="${LISTWISE_BETA:-0.08}"' in script
    assert 'LISTWISE_Q_TEMPERATURE="${LISTWISE_Q_TEMPERATURE:-2.0}"' in script
    assert 'EVAL_SBATCH_GRES="${EVAL_SBATCH_GRES:-gpu:a6000:1}"' in script
    assert 'EVAL_SBATCH_CPUS_PER_TASK="${EVAL_SBATCH_CPUS_PER_TASK:-16}"' in script
    assert 'EVAL_SBATCH_MEM="${EVAL_SBATCH_MEM:-96G}"' in script
    assert 'EVAL_SBATCH_TIME="${EVAL_SBATCH_TIME:-24:00:00}"' in script
    assert 'TRAIN_LIVE_EVAL_TEMPLATE="${TRAIN_LIVE_EVAL_TEMPLATE:-no}"' in script
    assert '--seed_paper_eval_template ${TRAIN_LIVE_EVAL_TEMPLATE}' in script
    assert 'MAXENT_STEP0_PAPER_EVAL_TEMPLATE="$TRAIN_LIVE_EVAL_TEMPLATE"' in script
    assert '--maxent_tau ${LISTWISE_TAU}' in script
    assert '--beta ${LISTWISE_BETA}' in script
    assert '--maxent_q_temperature ${LISTWISE_Q_TEMPERATURE}' in script
    assert '--gres "$EVAL_SBATCH_GRES"' in script
    assert '--cpus-per-task "$EVAL_SBATCH_CPUS_PER_TASK"' in script
    assert '--mem "$EVAL_SBATCH_MEM"' in script
    assert '--time "$EVAL_SBATCH_TIME"' in script


def test_full_eval_richsidecar_wrapper_targets_mltheory_long_run() -> None:
    script = _resolve_archived_ops_script(
        "run_grpo_listwise_full_eval_richsidecar_pair.sh"
    ).read_text()
    assert 'RESOURCE_PROFILE="${RESOURCE_PROFILE:-quartet_a100}"' in script
    assert 'SBATCH_PARTITION="${SBATCH_PARTITION:-mltheory}"' in script
    assert 'SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-mltheory}"' in script
    assert 'SBATCH_GRES="${SBATCH_GRES:-gpu:a100:4}"' in script
    assert 'TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:--1}"' in script
    assert 'TRAIN_NUM_EPOCHS="${TRAIN_NUM_EPOCHS:-20}"' in script
    assert 'TRAIN_EVAL_STEPS="${TRAIN_EVAL_STEPS:-25}"' in script
    assert 'TRAIN_SAVE_STEPS="${TRAIN_SAVE_STEPS:-25}"' in script
    assert 'TRAIN_LIVE_EVAL_ENABLED="${TRAIN_LIVE_EVAL_ENABLED:-false}"' in script
    assert 'TRAIN_LIVE_PASS_AT_8_ENABLED="${TRAIN_LIVE_PASS_AT_8_ENABLED:-false}"' in script
    assert 'TRAIN_LIVE_EVAL_FAIL_ON_ERROR="${TRAIN_LIVE_EVAL_FAIL_ON_ERROR:-false}"' in script
    assert 'TRAIN_RICH_SIDECAR_ENABLED="${TRAIN_RICH_SIDECAR_ENABLED:-false}"' in script
    assert 'TRAIN_LIVE_EVAL_TEMPLATE="${TRAIN_LIVE_EVAL_TEMPLATE:-no}"' in script
    assert 'SUBMIT_FINAL_EVAL_JOBS="${SUBMIT_FINAL_EVAL_JOBS:-false}"' in script
    assert 'SUBMIT_FINAL_EVAL_PLOT="${SUBMIT_FINAL_EVAL_PLOT:-false}"' in script
    assert 'SUBMIT_RICH_SIDECAR_PLOT="${SUBMIT_RICH_SIDECAR_PLOT:-false}"' in script
    assert 'LISTWISE_TAU="${LISTWISE_TAU:-0.35}"' in script
    assert 'LISTWISE_BETA="${LISTWISE_BETA:-0.12}"' in script
    assert 'LISTWISE_Q_TEMPERATURE="${LISTWISE_Q_TEMPERATURE:-2.0}"' in script
    assert '--rich_log_completions ${TRAIN_RICH_SIDECAR_ENABLED}' in script
    assert '--seed_paper_eval_enabled ${TRAIN_LIVE_EVAL_ENABLED}' in script
    assert '--seed_paper_eval_pass_at_8_enabled ${TRAIN_LIVE_PASS_AT_8_ENABLED}' in script
    assert '--seed_paper_eval_fail_on_error ${TRAIN_LIVE_EVAL_FAIL_ON_ERROR}' in script
    assert '--seed_paper_eval_template ${TRAIN_LIVE_EVAL_TEMPLATE}' in script
    assert 'STEP0_PAPER_EVAL_TEMPLATE="${STEP0_PAPER_EVAL_TEMPLATE:-}"' in script
    assert 'MAXENT_STEP0_PAPER_EVAL_TEMPLATE="$step0_template_env"' in script
    assert 'if is_truthy "$SUBMIT_FINAL_EVAL_JOBS"; then' in script
    assert 'if is_truthy "$SUBMIT_FINAL_EVAL_PLOT"; then' in script
    assert 'if is_truthy "$SUBMIT_RICH_SIDECAR_PLOT"; then' in script
    assert '--maxent_tau ${LISTWISE_TAU}' in script
    assert '--beta ${LISTWISE_BETA}' in script
    assert '--maxent_q_temperature ${LISTWISE_Q_TEMPERATURE}' in script
    assert 'python tools/plot_listwise_vs_grpo_distribution.py' in script
    assert 'python tools/plot_listwise_vs_grpo_accuracy_dynamics.py' in script
    assert 'plot_rich_sidecar' in script


def test_single_stack_wrapper_uses_shareable_defaults() -> None:
    script = _resolve_archived_ops_script("run_dual_4plus4_single_node.sh").read_text()
    assert 'RESOURCE_PROFILE="interim_a6000"' in script
    assert 'RESOURCE_PROFILE="quartet_a100"' in script
    assert 'SBATCH_CPUS_DEFAULT="64"' in script
    assert 'SBATCH_MEM_DEFAULT="256G"' in script
    assert 'SBATCH_CPUS_DEFAULT="48"' in script
    assert 'SBATCH_MEM_DEFAULT="240G"' in script
    assert 'STEP0_PAPER_EVAL_ENFORCE_EXPECTED_DEFAULT="0"' in script
    assert 'STEP0_PAPER_EVAL_ENFORCE_EXPECTED_DEFAULT="1"' in script


def test_oat_upstream_launcher_supports_opt_in_listwise_explorer() -> None:
    repo_root = _repo_root()
    script = (repo_root / "ops" / "run_oat_zero_exact_1p5b_upstream.sh").read_text()
    assert 'CANONICAL_UPSTREAM_ROOT="${ROOT_DIR}/understand-r1-zero"' in script
    assert 'CANONICAL_PYTHON_BIN="${ROOT_DIR}/var/seed_paper_eval/paper310/bin/python"' in script
    assert 'EXPECTED_VLLM_VERSION="${OAT_ZERO_EXPECTED_VLLM_VERSION:-0.8.4}"' in script
    assert 'EXPECTED_OAT_VERSION="${OAT_ZERO_EXPECTED_OAT_VERSION:-0.1.3.post1}"' in script
    assert 'EXPECTED_DEEPSPEED_VERSION="${OAT_ZERO_EXPECTED_DEEPSPEED_VERSION:-0.16.8}"' in script
    assert "understand-r1-zero-pristine" not in script
    assert "requirements.no_flash.txt" not in script
    assert 'OBJECTIVE="${OAT_ZERO_OBJECTIVE:-grpo}"' in script
    assert 'VLLM_GPU_RATIO="${OAT_ZERO_VLLM_GPU_RATIO:-0.35}"' in script
    assert 'MAX_NORM="${OAT_ZERO_MAX_NORM:-1.0}"' in script
    assert 'pytorch_cuda_alloc_conf=${PYTORCH_CUDA_ALLOC_CONF:-<unset>}' in script
    assert 'MAXENT_TAU="${OAT_ZERO_MAXENT_TAU:-0.5}"' in script
    assert 'MAXENT_TAU_CONTROLLER_ENABLED="${OAT_ZERO_MAXENT_TAU_CONTROLLER_ENABLED:-1}"' in script
    assert 'MAXENT_Q_TEMPERATURE="${OAT_ZERO_MAXENT_Q_TEMPERATURE:-2.0}"' in script
    assert 'MAXENT_CLIP_PRESERVE_REWARD_MASS="${OAT_ZERO_MAXENT_CLIP_PRESERVE_REWARD_MASS:-0}"' in script
    assert 'MAXENT_CLIP_MODE="${OAT_ZERO_MAXENT_CLIP_MODE:-sequence}"' in script
    assert 'MAXENT_TOKEN_SURROGATE_PRIMARY="${OAT_ZERO_MAXENT_TOKEN_SURROGATE_PRIMARY:-0}"' in script
    assert 'MAXENT_DRGRPO_TOKEN_PRIMARY="${OAT_ZERO_MAXENT_DRGRPO_TOKEN_PRIMARY:-0}"' in script
    assert 'MAXENT_SEQUENCE_AUX_COEF="${OAT_ZERO_MAXENT_SEQUENCE_AUX_COEF:-1.0}"' in script
    assert 'MAXENT_BRANCH_GRAD_DIAGNOSTICS="${OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS:-0}"' in script
    assert 'MAXENT_BRANCH_GRAD_DIAGNOSTICS_INTERVAL="${OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS_INTERVAL:-1}"' in script
    assert 'MAXENT_BRANCH_GRAD_DIAGNOSTICS_MAX_STEPS="${OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS_MAX_STEPS:-0}"' in script
    assert 'MAXENT_TARGET_WEIGHT_ENTROPY_PEAK="${OAT_ZERO_MAXENT_TARGET_WEIGHT_ENTROPY_PEAK:-}"' in script
    assert 'MAXENT_TARGET_WEIGHT_ENTROPY_PEAK_STEP="${OAT_ZERO_MAXENT_TARGET_WEIGHT_ENTROPY_PEAK_STEP:-0}"' in script
    assert 'MAXENT_BACKWARD_CHUNK_SIZE="${OAT_ZERO_MAXENT_BACKWARD_CHUNK_SIZE:-4}"' in script
    assert 'MAXENT_BACKWARD_TOKEN_BUDGET="${OAT_ZERO_MAXENT_BACKWARD_TOKEN_BUDGET:-8192}"' in script
    assert 'MAXENT_REFERENCE_LOGPROBS_SOURCE="${OAT_ZERO_MAXENT_REFERENCE_LOGPROBS_SOURCE:-model}"' in script
    assert 'SAVE_STEPS="${OAT_ZERO_SAVE_STEPS:-$EVAL_STEPS}"' in script
    assert 'SAVE_FROM="${OAT_ZERO_SAVE_FROM:-0}"' in script
    assert 'SAVE_CKPT="${OAT_ZERO_SAVE_CKPT:-0}"' in script
    assert 'SAVE_INITIAL_MODEL="${OAT_ZERO_SAVE_INITIAL_MODEL:-1}"' in script
    assert 'MAX_SAVE_NUM="${OAT_ZERO_MAX_SAVE_NUM:-999999}"' in script
    assert 'MAX_SAVE_MEM="${OAT_ZERO_MAX_SAVE_MEM:-99999999}"' in script
    assert 'if [[ "$OBJECTIVE" == "maxent_listwise" ]]; then' in script
    assert 'OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE to be divisible by OAT_ZERO_NUM_SAMPLES' in script
    assert '--objective "$OBJECTIVE"' in script
    assert '--vllm_gpu_ratio "$VLLM_GPU_RATIO"' in script
    assert '--max_norm "$MAX_NORM"' in script
    assert '--maxent-tau "$MAXENT_TAU"' in script
    assert '"${maxent_tau_controller_flag[@]}"' in script
    assert '--maxent-q-temperature "$MAXENT_Q_TEMPERATURE"' in script
    assert '"${maxent_clip_preserve_reward_mass_flag[@]}"' in script
    assert '--maxent-clip-mode "$MAXENT_CLIP_MODE"' in script
    assert '"${maxent_token_surrogate_primary_flag[@]}"' in script
    assert '"${maxent_drgrpo_token_primary_flag[@]}"' in script
    assert '--maxent-sequence-aux-coef "$MAXENT_SEQUENCE_AUX_COEF"' in script
    assert '"${maxent_branch_grad_diagnostics_flag[@]}"' in script
    assert '--maxent-branch-grad-diagnostics-interval "$MAXENT_BRANCH_GRAD_DIAGNOSTICS_INTERVAL"' in script
    assert '--maxent-branch-grad-diagnostics-max-steps "$MAXENT_BRANCH_GRAD_DIAGNOSTICS_MAX_STEPS"' in script
    assert '--maxent-target-weight-entropy-peak "$MAXENT_TARGET_WEIGHT_ENTROPY_PEAK"' in script
    assert '--maxent-target-weight-entropy-peak-step "$MAXENT_TARGET_WEIGHT_ENTROPY_PEAK_STEP"' in script
    assert '--maxent-backward-chunk-size "$MAXENT_BACKWARD_CHUNK_SIZE"' in script
    assert '--maxent-backward-token-budget "$MAXENT_BACKWARD_TOKEN_BUDGET"' in script
    assert '--maxent-reference-logprobs-source "$MAXENT_REFERENCE_LOGPROBS_SOURCE"' in script
    assert '--save_steps "$SAVE_STEPS"' in script
    assert '--save_from "$SAVE_FROM"' in script
    assert '--max_save_num "$MAX_SAVE_NUM"' in script
    assert '--max_save_mem "$MAX_SAVE_MEM"' in script
    assert 'if [[ "$SAVE_CKPT" == "1" ]]; then' in script
    assert 'cmd+=(--save-ckpt)' in script
    assert 'save_initial_model_snapshot' in script


def test_oat_upstream_explorer_wrappers_reuse_readme_flash_runtime() -> None:
    repo_root = _repo_root()
    run_script = (repo_root / "ops" / "run_oat_zero_explorer_1p5b_upstream.sh").read_text()
    slurm_script = (
        repo_root
        / "ops"
        / "slurm"
        / "train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_explorer_node302.slurm"
    ).read_text()
    assert 'OAT_ZERO_OBJECTIVE="${OAT_ZERO_OBJECTIVE:-maxent_listwise}"' in run_script
    assert 'OAT_ZERO_SAVE_STEPS="${OAT_ZERO_SAVE_STEPS:-${OAT_ZERO_EVAL_STEPS}}"' in run_script
    assert 'OAT_ZERO_SAVE_FROM="${OAT_ZERO_SAVE_FROM:-0}"' in run_script
    assert 'OAT_ZERO_SAVE_INITIAL_MODEL="${OAT_ZERO_SAVE_INITIAL_MODEL:-1}"' in run_script
    assert 'OAT_ZERO_MAX_SAVE_NUM="${OAT_ZERO_MAX_SAVE_NUM:-999999}"' in run_script
    assert 'OAT_ZERO_MAX_SAVE_MEM="${OAT_ZERO_MAX_SAVE_MEM:-99999999}"' in run_script
    assert 'OAT_ZERO_BETA="${OAT_ZERO_BETA:-0}"' in run_script
    assert 'OAT_ZERO_VLLM_GPU_RATIO="${OAT_ZERO_VLLM_GPU_RATIO:-0.35}"' in run_script
    assert 'OAT_ZERO_MAX_NORM="${OAT_ZERO_MAX_NORM:-1.0}"' in run_script
    assert 'OAT_ZERO_MAXENT_TAU="${OAT_ZERO_MAXENT_TAU:-0.1}"' in run_script
    assert 'OAT_ZERO_MAXENT_TAU_LEARNABLE="${OAT_ZERO_MAXENT_TAU_LEARNABLE:-0}"' in run_script
    assert 'OAT_ZERO_MAXENT_TAU_CONTROLLER_ENABLED="${OAT_ZERO_MAXENT_TAU_CONTROLLER_ENABLED:-0}"' in run_script
    assert 'OAT_ZERO_MAXENT_TAU_LR="${OAT_ZERO_MAXENT_TAU_LR:-0.02}"' in run_script
    assert 'OAT_ZERO_MAXENT_TAU_MIN="${OAT_ZERO_MAXENT_TAU_MIN:-0.01}"' in run_script
    assert 'OAT_ZERO_MAXENT_TAU_MAX="${OAT_ZERO_MAXENT_TAU_MAX:-0}"' in run_script
    assert 'OAT_ZERO_MAXENT_TAU_WARMUP_STEPS="${OAT_ZERO_MAXENT_TAU_WARMUP_STEPS:-0}"' in run_script
    assert 'OAT_ZERO_MAXENT_Q_TEMPERATURE="${OAT_ZERO_MAXENT_Q_TEMPERATURE:-2.0}"' in run_script
    assert 'OAT_ZERO_MAXENT_CLIP_PRESERVE_REWARD_MASS="${OAT_ZERO_MAXENT_CLIP_PRESERVE_REWARD_MASS:-0}"' in run_script
    assert 'OAT_ZERO_MAXENT_CLIP_MODE="${OAT_ZERO_MAXENT_CLIP_MODE:-sequence}"' in run_script
    assert 'OAT_ZERO_MAXENT_TOKEN_SURROGATE_PRIMARY="${OAT_ZERO_MAXENT_TOKEN_SURROGATE_PRIMARY:-0}"' in run_script
    assert 'OAT_ZERO_MAXENT_DRGRPO_TOKEN_PRIMARY="${OAT_ZERO_MAXENT_DRGRPO_TOKEN_PRIMARY:-1}"' in run_script
    assert 'OAT_ZERO_MAXENT_SEQUENCE_AUX_COEF="${OAT_ZERO_MAXENT_SEQUENCE_AUX_COEF:-0.01}"' in run_script
    assert 'OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS="${OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS:-0}"' in run_script
    assert 'OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS_INTERVAL="${OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS_INTERVAL:-1}"' in run_script
    assert 'OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS_MAX_STEPS="${OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS_MAX_STEPS:-16}"' in run_script
    assert 'OAT_ZERO_MAXENT_LENGTH_NORMALIZE_REF="${OAT_ZERO_MAXENT_LENGTH_NORMALIZE_REF:-1}"' in run_script
    assert 'OAT_ZERO_MAXENT_LENGTH_NORMALIZE_POLICY="${OAT_ZERO_MAXENT_LENGTH_NORMALIZE_POLICY:-1}"' in run_script
    assert 'OAT_ZERO_MAXENT_TARGET_WEIGHT_ENTROPY_START="${OAT_ZERO_MAXENT_TARGET_WEIGHT_ENTROPY_START:-1.75}"' in run_script
    assert 'OAT_ZERO_MAXENT_TARGET_WEIGHT_ENTROPY_PEAK="${OAT_ZERO_MAXENT_TARGET_WEIGHT_ENTROPY_PEAK:-1.95}"' in run_script
    assert 'OAT_ZERO_MAXENT_TARGET_WEIGHT_ENTROPY_PEAK_STEP="${OAT_ZERO_MAXENT_TARGET_WEIGHT_ENTROPY_PEAK_STEP:-16}"' in run_script
    assert 'OAT_ZERO_MAXENT_TARGET_WEIGHT_ENTROPY_FINAL="${OAT_ZERO_MAXENT_TARGET_WEIGHT_ENTROPY_FINAL:-1.70}"' in run_script
    assert 'OAT_ZERO_MAXENT_TARGET_WEIGHT_ENTROPY_HORIZON="${OAT_ZERO_MAXENT_TARGET_WEIGHT_ENTROPY_HORIZON:-96}"' in run_script
    assert 'OAT_ZERO_MAXENT_BACKWARD_CHUNK_SIZE="${OAT_ZERO_MAXENT_BACKWARD_CHUNK_SIZE:-4}"' in run_script
    assert 'OAT_ZERO_MAXENT_BACKWARD_TOKEN_BUDGET="${OAT_ZERO_MAXENT_BACKWARD_TOKEN_BUDGET:-4096}"' in run_script
    assert 'export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"' in run_script
    assert 'OAT_ZERO_TRAIN_BATCH_SIZE="${OAT_ZERO_TRAIN_BATCH_SIZE:-128}"' in run_script
    assert 'OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE="${OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE:-8}"' in run_script
    assert 'OAT_ZERO_MAXENT_LISTWISE_SKIP_ZERO_VARIANCE_GROUPS="${OAT_ZERO_MAXENT_LISTWISE_SKIP_ZERO_VARIANCE_GROUPS:-1}"' in run_script
    assert 'exec "$ROOT_DIR/ops/run_oat_zero_exact_1p5b_upstream.sh" "$@"' in run_script
    assert 'export OAT_ZERO_ENABLE_FLASH_ATTN=1' in slurm_script
    assert 'HOST_TAG="${SLURMD_NODENAME:-$(hostname -s)}"' in slurm_script
    assert 'OAT_ZERO_LOCAL_ROOT="${OAT_ZERO_LOCAL_ROOT:-/tmp/${USER}/maxent-grpo-oat-zero-${HOST_TAG}-readmeflash-explorer}"' in slurm_script
    assert 'exec "${ROOT_DIR}/ops/run_oat_zero_explorer_1p5b_upstream.sh"' in slurm_script


def test_oat_upstream_7b_r1_wrapper_uses_existing_oat_recipe() -> None:
    repo_root = _repo_root()
    base_script = (repo_root / "ops" / "run_oat_zero_exact_1p5b_upstream.sh").read_text()
    run_script = (repo_root / "ops" / "run_oat_zero_math_7b_r1_upstream.sh").read_text()
    slurm_script = (
        repo_root
        / "ops"
        / "slurm"
        / "train_understand_r1_zero_qwen2p5_math_7b_r1_readme_flash_node105.slurm"
    ).read_text()
    assert 'OAT_ZERO_PRETRAIN="${OAT_ZERO_PRETRAIN:-Qwen/Qwen2.5-Math-7B}"' in run_script
    assert 'OAT_ZERO_PROMPT_TEMPLATE="${OAT_ZERO_PROMPT_TEMPLATE:-r1}"' in run_script
    assert 'OAT_ZERO_VERIFIER_VERSION="${OAT_ZERO_VERIFIER_VERSION:-math_verify}"' in run_script
    assert 'OAT_ZERO_PROMPT_DATA="${OAT_ZERO_PROMPT_DATA:-$UPSTREAM_ROOT/datasets/train/math_lvl3to5_8k}"' in run_script
    assert 'OAT_ZERO_VLLM_GPU_RATIO="${OAT_ZERO_VLLM_GPU_RATIO:-0.35}"' in run_script
    assert 'USE_WB="${OAT_ZERO_USE_WB:-1}"' in base_script
    assert 'COLLOCATE="${OAT_ZERO_COLLOCATE:-1}"' in base_script
    assert 'ZERO_STAGE="${OAT_ZERO_ZERO_STAGE:-2}"' in base_script
    assert 'ADAM_OFFLOAD="${OAT_ZERO_ADAM_OFFLOAD:-0}"' in base_script
    assert 'ACTIVATION_OFFLOADING="${OAT_ZERO_ACTIVATION_OFFLOADING:-0}"' in base_script
    assert 'DISABLE_TRACE_CACHE="${OAT_ZERO_DISABLE_TRACE_CACHE:-0}"' in base_script
    assert 'GRAD_ACCUM_DTYPE="${OAT_ZERO_GRAD_ACCUM_DTYPE:-}"' in base_script
    assert 'VLLM_SLEEP="${OAT_ZERO_VLLM_SLEEP:-1}"' in base_script
    assert 'torch_cuda_version="$("$PYTHON_BIN" - <<\'PY\' | tail -n 1 | tr -d \'\\r\'' in base_script
    assert 'echo "[oat-zero-exact] torch_cuda_version=${torch_cuda_version:-cpu}"' in base_script
    assert '"/usr/local/cuda-${torch_cuda_version}/bin/nvcc"' in base_script
    assert 'echo "[oat-zero-exact] use_wb=${USE_WB}"' in base_script
    assert 'echo "[oat-zero-exact] collocate=${COLLOCATE}"' in base_script
    assert 'echo "[oat-zero-exact] zero_stage=${ZERO_STAGE}"' in base_script
    assert 'echo "[oat-zero-exact] adam_offload=${ADAM_OFFLOAD}"' in base_script
    assert 'echo "[oat-zero-exact] activation_offloading=${ACTIVATION_OFFLOADING}"' in base_script
    assert 'echo "[oat-zero-exact] disable_trace_cache=${DISABLE_TRACE_CACHE}"' in base_script
    assert 'echo "[oat-zero-exact] grad_accum_dtype=${GRAD_ACCUM_DTYPE:-<unset>}"' in base_script
    assert 'echo "[oat-zero-exact] vllm_sleep=${VLLM_SLEEP}"' in base_script
    assert 'if [[ "$USE_WB" == "1" ]]; then' in base_script
    assert 'if [[ "$COLLOCATE" == "1" ]]; then' in base_script
    assert '--zero-stage "$ZERO_STAGE"' in base_script
    assert 'if [[ "$ADAM_OFFLOAD" == "1" ]]; then' in base_script
    assert 'if [[ "$DISABLE_TRACE_CACHE" == "1" ]]; then' in base_script
    assert 'if [[ "$VLLM_SLEEP" == "1" ]]; then' in base_script
    assert '--use-wb' in base_script
    assert 'exec "$ROOT_DIR/ops/run_oat_zero_exact_1p5b_upstream.sh" "$@"' in run_script
    assert '#SBATCH --gres=gpu:a5000:8' in slurm_script
    assert '#SBATCH --nodelist=node105' in slurm_script
    assert 'export OAT_ZERO_COLLOCATE="${OAT_ZERO_COLLOCATE:-0}"' in slurm_script
    assert 'export OAT_ZERO_VLLM_GPU_RATIO="${OAT_ZERO_VLLM_GPU_RATIO:-0.75}"' in slurm_script
    assert 'export OAT_ZERO_VLLM_SLEEP="${OAT_ZERO_VLLM_SLEEP:-1}"' in slurm_script
    assert 'export OAT_ZERO_CUDA_HOME="${OAT_ZERO_CUDA_HOME:-/usr/local/cuda-12.4}"' in slurm_script
    assert 'export OAT_ZERO_ZERO_STAGE="${OAT_ZERO_ZERO_STAGE:-3}"' in slurm_script
    assert 'export OAT_ZERO_ADAM_OFFLOAD="${OAT_ZERO_ADAM_OFFLOAD:-1}"' in slurm_script
    assert 'export OAT_ZERO_MAX_NORM="${OAT_ZERO_MAX_NORM:-0}"' in slurm_script
    assert 'export OAT_ZERO_ACTIVATION_OFFLOADING="${OAT_ZERO_ACTIVATION_OFFLOADING:-1}"' in slurm_script
    assert 'export OAT_ZERO_DISABLE_TRACE_CACHE="${OAT_ZERO_DISABLE_TRACE_CACHE:-1}"' in slurm_script
    assert 'export OAT_ZERO_GRAD_ACCUM_DTYPE="${OAT_ZERO_GRAD_ACCUM_DTYPE:-bf16}"' in slurm_script
    assert 'export OAT_ZERO_TRAIN_BATCH_SIZE="${OAT_ZERO_TRAIN_BATCH_SIZE:-32}"' in slurm_script
    assert 'export OAT_ZERO_ROLLOUT_BATCH_SIZE="${OAT_ZERO_ROLLOUT_BATCH_SIZE:-32}"' in slurm_script
    assert 'export OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE="${OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE:-4}"' in slurm_script
    assert 'export OAT_ZERO_PI_BUFFER_MAXLEN_PER_DEVICE="${OAT_ZERO_PI_BUFFER_MAXLEN_PER_DEVICE:-32}"' in slurm_script
    assert 'export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"' in slurm_script
    assert 'export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"' in slurm_script
    assert 'export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"' in slurm_script
    assert 'echo "[slurm] oat_zero_collocate=${OAT_ZERO_COLLOCATE}"' in slurm_script
    assert 'echo "[slurm] oat_zero_vllm_gpu_ratio=${OAT_ZERO_VLLM_GPU_RATIO}"' in slurm_script
    assert 'echo "[slurm] oat_zero_vllm_sleep=${OAT_ZERO_VLLM_SLEEP}"' in slurm_script
    assert 'echo "[slurm] oat_zero_cuda_home=${OAT_ZERO_CUDA_HOME}"' in slurm_script
    assert 'echo "[slurm] oat_zero_zero_stage=${OAT_ZERO_ZERO_STAGE}"' in slurm_script
    assert 'echo "[slurm] oat_zero_adam_offload=${OAT_ZERO_ADAM_OFFLOAD}"' in slurm_script
    assert 'echo "[slurm] oat_zero_max_norm=${OAT_ZERO_MAX_NORM}"' in slurm_script
    assert 'echo "[slurm] oat_zero_activation_offloading=${OAT_ZERO_ACTIVATION_OFFLOADING}"' in slurm_script
    assert 'echo "[slurm] oat_zero_disable_trace_cache=${OAT_ZERO_DISABLE_TRACE_CACHE}"' in slurm_script
    assert 'echo "[slurm] oat_zero_grad_accum_dtype=${OAT_ZERO_GRAD_ACCUM_DTYPE}"' in slurm_script
    assert 'echo "[slurm] oat_zero_train_batch_size=${OAT_ZERO_TRAIN_BATCH_SIZE}"' in slurm_script
    assert 'echo "[slurm] oat_zero_rollout_batch_size=${OAT_ZERO_ROLLOUT_BATCH_SIZE}"' in slurm_script
    assert 'echo "[slurm] oat_zero_rollout_batch_size_per_device=${OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE}"' in slurm_script
    assert 'echo "[slurm] oat_zero_pi_buffer_maxlen_per_device=${OAT_ZERO_PI_BUFFER_MAXLEN_PER_DEVICE}"' in slurm_script
    assert 'echo "[slurm] pythonunbuffered=${PYTHONUNBUFFERED}"' in slurm_script
    assert 'echo "[slurm] pythonioencoding=${PYTHONIOENCODING}"' in slurm_script
    assert 'echo "[slurm] pytorch_cuda_alloc_conf=${PYTORCH_CUDA_ALLOC_CONF}"' in slurm_script
    assert 'export OAT_ZERO_USE_WB="${OAT_ZERO_USE_WB:-1}"' in slurm_script
    assert 'echo "[slurm] oat_zero_use_wb=${OAT_ZERO_USE_WB}"' in slurm_script
    assert 'OAT_ZERO_WB_RUN_NAME="${OAT_ZERO_WB_RUN_NAME:-qwen2.5-Math-7b-drgrpo-r1template-readmeflash-node105}"' in slurm_script
    assert 'export PYTHONPATH="${ROOT_DIR}/tools/runtime_patches${PYTHONPATH:+:${PYTHONPATH}}"' in slurm_script
    assert 'echo "[slurm] pythonpath=${PYTHONPATH}"' in slurm_script
    assert 'exec bash "${ROOT_DIR}/ops/run_oat_zero_math_7b_r1_upstream.sh"' in slurm_script


def test_experiment_presets_disable_hub_pushes() -> None:
    repo_root = _repo_root()
    preset_names = (
        "grpo_custom_math.yaml",
        "grpo_custom_math_fair.yaml",
        "grpo_custom_math_stable.yaml",
        "grpo_custom_code_mbpp.yaml",
        "maxent_entropy_math.yaml",
        "maxent_entropy_math_fair.yaml",
        "maxent_entropy_math_stable.yaml",
        "maxent_entropy_code_mbpp.yaml",
        "maxent_listwise_math.yaml",
        "maxent_listwise_math_fair.yaml",
        "maxent_listwise_math_stable.yaml",
        "maxent_listwise_code_mbpp.yaml",
        "seed_grpo_math.yaml",
        "seed_grpo_math_fair.yaml",
        "seed_grpo_math_stable.yaml",
    )
    for name in preset_names:
        preset = repo_root / "configs" / "recipes" / "hydra" / name
        contents = preset.read_text()
        assert "push_to_hub: false" in contents


def test_math_triplet_presets_use_dr_grpo_defaults() -> None:
    repo_root = _repo_root()
    base_math = (
        repo_root / "configs" / "recipes" / "Qwen2.5-1.5B-Instruct" / "grpo" / "config_math.yaml"
    ).read_text()
    grpo = (repo_root / "configs" / "recipes" / "hydra" / "grpo_custom_math.yaml").read_text()
    entropy = (repo_root / "configs" / "recipes" / "hydra" / "maxent_entropy_math.yaml").read_text()
    listwise = (repo_root / "configs" / "recipes" / "hydra" / "maxent_listwise_math.yaml").read_text()
    seed = (repo_root / "configs" / "recipes" / "hydra" / "seed_grpo_math.yaml").read_text()

    assert "model_name_or_path: Qwen/Qwen2.5-Math-1.5B" in base_math
    assert "dataset_name: axon-rl/MATH-lvl3to5-8k" in base_math
    assert "eval_dataset_name: aime24,amc,math_500,minerva,olympiad_bench" in base_math
    assert "reward_funcs:" in base_math
    assert "- seed_paper_boxed_accuracy_math" in base_math
    assert "eval_reward_funcs:" in base_math
    assert 'prompt_template: "no"' in base_math
    assert "system_prompt: null" in base_math
    assert "chat_template: null" in base_math
    assert "maxent_logprob_chunk_size: 2" in base_math
    assert "save_steps: 1" in base_math
    for contents in (grpo, entropy, seed):
        assert "num_generations: 8" in contents
        assert "max_steps: -1" in contents
        assert "learning_rate: 1e-6" in contents
        assert "beta: 0.0" in contents
        assert "grpo_loss_type: dr_grpo" in contents
        assert "num_iterations: 1" in contents
        assert "max_grad_norm: 1.0" in contents
        assert "warmup_ratio: 0.0" in contents
        assert "eval_on_start: true" in contents
        assert "eval_steps: 10" in contents
        assert "seed_paper_eval_enabled: true" in contents
        assert 'seed_paper_eval_template: "no"' in contents
        assert 'save_strategy: "no"' in contents
    assert "num_generations: 8" in listwise
    assert "max_steps: -1" in listwise
    assert "learning_rate: 1e-6" in listwise
    assert "beta: 0.0" in listwise
    assert "grpo_loss_type: dr_grpo" in listwise
    assert "num_iterations: 1" in listwise
    assert "max_grad_norm: 1.0" in listwise
    assert "warmup_ratio: 0.0" in listwise
    assert "eval_on_start: true" in listwise
    assert "eval_steps: 10" in listwise
    assert "seed_paper_eval_enabled: true" in listwise
    assert 'seed_paper_eval_template: "no"' in listwise
    assert 'save_strategy: "no"' in listwise
    assert "maxent_alpha: 0.005" in entropy
    assert "maxent_alpha_lower_on_high_kl: true" in entropy
    assert "maxent_tau: 0.5" in listwise
    assert "maxent_q_temperature: 2.0" in listwise
    assert "maxent_use_clip_objective: true" in listwise
    assert "maxent_clip_objective_coef: 1.0" in listwise
    assert "maxent_reference_logprobs_source: model" in listwise
    assert "maxent_trl_reference_scoring: true" in listwise
    assert "seed_grpo_enabled: true" in seed
    assert "seed_grpo_alpha: 0.0417" in seed


def test_math_stable_quartet_presets_use_shared_stabilizers() -> None:
    repo_root = _repo_root()
    base_math = (
        repo_root
        / "configs"
        / "recipes"
        / "Qwen2.5-1.5B-Instruct"
        / "grpo"
        / "config_math_stable.yaml"
    ).read_text()
    grpo = (
        repo_root / "configs" / "recipes" / "hydra" / "grpo_custom_math_stable.yaml"
    ).read_text()
    entropy = (
        repo_root
        / "configs"
        / "recipes"
        / "hydra"
        / "maxent_entropy_math_stable.yaml"
    ).read_text()
    listwise = (
        repo_root
        / "configs"
        / "recipes"
        / "hydra"
        / "maxent_listwise_math_stable.yaml"
    ).read_text()
    seed = (
        repo_root / "configs" / "recipes" / "hydra" / "seed_grpo_math_stable.yaml"
    ).read_text()

    assert "model_name_or_path: Qwen/Qwen2.5-Math-1.5B" in base_math
    assert "dataset_name: axon-rl/MATH-lvl3to5-8k" in base_math
    assert "eval_dataset_name: aime24,amc,math_500,minerva,olympiad_bench" in base_math
    assert "gradient_accumulation_steps: 40" in base_math
    assert "max_completion_length: 3000" in base_math
    assert "reward_funcs:" in base_math
    assert "- seed_paper_boxed_accuracy_math" in base_math
    assert "- missing_boxed_answer_penalty_math" in base_math
    assert "eval_reward_funcs:" in base_math
    assert "<answer>" in base_math
    assert "chat_template:" in base_math
    assert "dr_grpo_denominator_mode: active_tokens" in base_math
    assert "vllm_stop_sequences:" in base_math
    assert '  - "</answer>"' in base_math
    assert "maxent_length_normalize_ref: true" in base_math
    assert "maxent_length_normalize_policy: true" in base_math
    assert "missing_boxed_answer_penalty: -0.05" in base_math
    assert "beta: 0.0" in base_math
    assert "kl_target: 0.0" in base_math
    assert "eval_on_start: true" in base_math
    assert "greedy_eval_enabled: true" in base_math
    assert "eval_greedy_only_enabled: true" in base_math
    assert "truncate_completions_at_first_boxed_answer: true" in base_math
    assert "save_steps: 1" in base_math
    assert "grpo_beta_controller_enabled: false" in base_math
    assert "maxent_beta_controller_enabled: false" in base_math
    for contents in (grpo, entropy, listwise, seed):
        assert "num_generations: 8" in contents
        assert "max_steps: -1" in contents
        assert "learning_rate: 1e-6" in contents
        assert "beta: 0.0" in contents
        assert "grpo_loss_type: dr_grpo" in contents
        assert "num_iterations: 1" in contents
        assert "grpo_beta_controller_enabled: false" in contents
        assert "maxent_beta_controller_enabled: false" in contents
        assert "max_grad_norm: 1.0" in contents
        assert "warmup_ratio: 0.0" in contents
        assert "eval_steps: 10" in contents
        assert "seed_paper_eval_enabled: true" in contents
        assert 'seed_paper_eval_template: "no"' in contents
        assert 'save_strategy: "no"' in contents
    assert "maxent_alpha: 0.005" in entropy
    assert "maxent_alpha_lower_on_high_kl: true" in entropy
    assert "maxent_alpha_disable_outside_trust_zone: true" in entropy
    assert "maxent_tau: 0.5" in listwise
    assert "maxent_q_temperature: 2.0" in listwise
    assert "maxent_use_clip_objective: true" in listwise
    assert "maxent_clip_objective_coef: 1.0" in listwise
    assert "maxent_reference_logprobs_source: model" in listwise
    assert "maxent_trl_reference_scoring: true" in listwise
    assert "seed_grpo_enabled: true" in seed
    assert "seed_grpo_alpha: 0.0417" in seed


def test_math_fair_quartet_presets_use_paper_no_template_shared_backbone() -> None:
    repo_root = _repo_root()
    base_math = (
        repo_root
        / "configs"
        / "recipes"
        / "Qwen2.5-1.5B-Instruct"
        / "grpo"
        / "config_math_fair.yaml"
    ).read_text()
    grpo = (
        repo_root / "configs" / "recipes" / "hydra" / "grpo_custom_math_fair.yaml"
    ).read_text()
    entropy = (
        repo_root
        / "configs"
        / "recipes"
        / "hydra"
        / "maxent_entropy_math_fair.yaml"
    ).read_text()
    listwise = (
        repo_root
        / "configs"
        / "recipes"
        / "hydra"
        / "maxent_listwise_math_fair.yaml"
    ).read_text()
    seed = (
        repo_root / "configs" / "recipes" / "hydra" / "seed_grpo_math_fair.yaml"
    ).read_text()
    quartet = _resolve_archived_ops_script("run_experiment_quartet_single_node.sh").read_text()
    dual = _resolve_archived_ops_script("run_dual_4plus4_single_node.sh").read_text()
    slurm = _resolve_archived_ops_slurm("train_dual_4plus4.slurm").read_text()

    assert 'CONFIG_SUFFIX="${CONFIG_SUFFIX:-math_fair}"' in quartet
    assert "math|math_fair|math_stable" in quartet
    assert "math|math_fair|math_stable" in dual
    assert "math|math_fair|math_stable" in slurm
    assert 'STEP0_PAPER_EVAL_TEMPLATE="${MAXENT_STEP0_PAPER_EVAL_TEMPLATE:-}"' in slurm
    assert 'cmd+=(--template "$STEP0_PAPER_EVAL_TEMPLATE")' in slurm
    assert 'cmd+=(--tasks "$STEP0_PAPER_EVAL_TASKS")' in slurm
    assert 'cmd+=(--max-test "$STEP0_PAPER_EVAL_MAX_TEST")' in slurm
    assert 'STEP0_PAPER_EVAL_PASS_AT_8_ENABLED="${MAXENT_STEP0_PAPER_EVAL_PASS_AT_8_ENABLED:-auto}"' in slurm
    assert 'cmd+=(--pass-at-8)' in slurm
    assert 'cmd+=(--pass-at-8-samples "$STEP0_PAPER_EVAL_PASS_AT_8_SAMPLES")' in slurm
    assert 'STEP0_PAPER_EVAL_TEMPLATE="no"' in slurm

    assert "model_name_or_path: Qwen/Qwen2.5-Math-1.5B" in base_math
    assert "dataset_name: axon-rl/MATH-lvl3to5-8k" in base_math
    assert 'prompt_template: "no"' in base_math
    assert "system_prompt: null" in base_math
    assert "chat_template: null" in base_math
    assert 'seed_paper_eval_template: "no"' in base_math
    assert "seed_paper_eval_pass_at_8_enabled: true" in base_math
    assert "seed_paper_eval_pass_at_8_samples: 8" in base_math
    assert "dr_grpo_denominator_mode: fixed_max" in base_math
    assert "max_completion_length: 3000" in base_math
    assert "vllm_stop_sequences" not in base_math
    assert "truncate_completions_at_first_boxed_answer" not in base_math
    assert "missing_boxed_answer_penalty_math" not in base_math

    for contents in (grpo, entropy, seed):
        assert "gradient_accumulation_steps: 40" in contents
        assert "learning_rate: 1e-6" in contents
        assert "grpo_loss_type: dr_grpo" in contents
        assert "dr_grpo_denominator_mode: fixed_max" in contents
        assert "eval_on_start: true" in contents
        assert "eval_steps: 25" in contents
        assert 'seed_paper_eval_template: "no"' in contents
        assert 'save_strategy: "steps"' in contents
        assert "save_steps: 25" in contents
        assert "push_to_hub: false" in contents

    assert "beta: 0.08" in listwise
    assert "maxent_tau: 0.5" in listwise
    assert "maxent_q_temperature: 2.0" in listwise
    assert "maxent_use_clip_objective: true" in listwise
    assert "maxent_clip_objective_coef: 1.0" in listwise
    assert "maxent_reference_logprobs_source: model" in listwise
    assert 'seed_paper_eval_template: "no"' in listwise
    assert "seed_grpo_enabled: true" in seed
    assert "seed_grpo_alpha: 0.0417" in seed


def test_active_training_ops_surface_is_oat_only() -> None:
    repo_root = _repo_root()
    ops_entries = {path.name for path in (repo_root / "ops").iterdir()}
    slurm_entries = {path.name for path in (repo_root / "ops" / "slurm").iterdir()}

    assert "run_oat_zero_exact_1p5b_upstream.sh" in ops_entries
    assert "run_oat_zero_explorer_1p5b_upstream.sh" in ops_entries
    assert "run_experiment_quartet_single_node.sh" not in ops_entries
    assert "run_drgrpo_1p5b_pure_single_node.sh" not in ops_entries

    assert (
        "train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_node302.slurm"
        in slurm_entries
    )
    assert (
        "train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_explorer_node302.slurm"
        in slurm_entries
    )
    assert "train_dual_4plus4.slurm" not in slurm_entries
    assert "train_drgrpo_pure_8gpu.slurm" not in slurm_entries
