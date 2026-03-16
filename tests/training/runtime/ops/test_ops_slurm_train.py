"""Tests for the repository Slurm training launcher."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _resolve_train_slurm() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    candidates = [
        repo_root / "ops" / "slurm" / "train_dual_4plus4.slurm",
        repo_root / "ops" / "slurm" / "train.slurm",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Slurm training launcher not found in expected locations")


def _resolve_tiny_smoke_slurm() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    candidate = repo_root / "ops" / "slurm" / "train_tiny_gpu_smoke.slurm"
    if candidate.exists():
        return candidate
    raise FileNotFoundError("Tiny smoke Slurm launcher not found in expected location")


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
    assert 'EXTRA_SITE_PACKAGES="${EXTRA_SITE_PACKAGES:-}"' in script
    assert 'python -c "import trl, vllm"' not in script
    assert "#SBATCH --exclusive" not in script


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


def test_triplet_wrapper_submits_all_three_single_stack_runs() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    script = (repo_root / "ops" / "run_experiment_triplet_single_node.sh").read_text()
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
    repo_root = Path(__file__).resolve().parents[4]
    script = (repo_root / "ops" / "run_experiment_quartet_single_node.sh").read_text()
    assert 'submit_quartet_job "listwise"' in script
    assert 'submit_quartet_job "maxent"' in script
    assert 'submit_quartet_job "seed"' in script
    assert 'submit_quartet_job "grpo"' in script
    assert 'WAVE2_DEPENDENCY="afterok:${LISTWISE_JOB_ID}:${MAXENT_JOB_ID}"' in script
    assert 'SEED_VLLM_PORT="${SEED_VLLM_PORT:-8003}"' in script
    assert 'SEED_GROUP_PORT="${SEED_GROUP_PORT:-29538}"' in script
    assert 'SEED_MASTER_PORT="${SEED_MASTER_PORT:-6003}"' in script
    assert 'MAXENT_ARGS="$variant_args"' in script
    assert 'GRPO_ARGS="$variant_args"' in script


def test_single_stack_wrapper_uses_shareable_defaults() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    script = (repo_root / "ops" / "run_dual_4plus4_single_node.sh").read_text()
    assert 'SBATCH_CPUS_DEFAULT="64"' in script
    assert 'SBATCH_MEM_DEFAULT="256G"' in script
    assert 'SBATCH_CPUS_DEFAULT="48"' in script
    assert 'SBATCH_MEM_DEFAULT="240G"' in script


def test_experiment_presets_disable_hub_pushes() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    preset_names = (
        "grpo_custom_math.yaml",
        "grpo_custom_math_stable.yaml",
        "grpo_custom_code_mbpp.yaml",
        "maxent_entropy_math.yaml",
        "maxent_entropy_math_stable.yaml",
        "maxent_entropy_code_mbpp.yaml",
        "maxent_listwise_math.yaml",
        "maxent_listwise_math_stable.yaml",
        "maxent_listwise_code_mbpp.yaml",
        "seed_grpo_math.yaml",
        "seed_grpo_math_stable.yaml",
    )
    for name in preset_names:
        preset = repo_root / "configs" / "recipes" / "hydra" / name
        contents = preset.read_text()
        assert "push_to_hub: false" in contents


def test_math_triplet_presets_use_dr_grpo_defaults() -> None:
    repo_root = Path(__file__).resolve().parents[4]
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
    assert "- boxed_accuracy_math" in base_math
    assert "maxent_logprob_chunk_size: 2" in base_math
    for contents in (grpo, entropy, listwise, seed):
        assert "num_generations: 8" in contents
        assert "max_steps: -1" in contents
        assert "learning_rate: 1e-6" in contents
        assert "beta: 0.0" in contents
        assert "grpo_loss_type: dr_grpo" in contents
        assert "num_iterations: 1" in contents
        assert "max_grad_norm: 1.0" in contents
        assert "warmup_ratio: 0.0" in contents
        assert "eval_steps: 16" in contents
        assert 'save_strategy: "no"' in contents
    assert "maxent_alpha: 0.005" in entropy
    assert "maxent_alpha_lower_on_high_kl: false" in entropy
    assert "maxent_tau: 0.2" in listwise
    assert "seed_grpo_enabled: true" in seed
    assert "seed_grpo_alpha: 0.0417" in seed


def test_math_stable_quartet_presets_use_shared_stabilizers() -> None:
    repo_root = Path(__file__).resolve().parents[4]
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
    assert "max_completion_length: 1750" in base_math
    assert "reward_funcs:" in base_math
    assert "- accuracy" in base_math
    assert "maxent_length_normalize_ref: true" in base_math
    assert "maxent_length_normalize_policy: true" in base_math
    assert "beta: 0.08" in base_math
    assert "kl_target: 0.07" in base_math
    assert "greedy_eval_enabled: true" in base_math
    assert "eval_greedy_only_enabled: true" in base_math
    assert "truncate_completions_at_first_boxed_answer: true" in base_math
    assert "grpo_beta_controller_enabled: false" in base_math
    assert "maxent_beta_controller_enabled: false" in base_math
    for contents in (grpo, entropy, listwise, seed):
        assert "num_generations: 8" in contents
        assert "max_steps: -1" in contents
        assert "learning_rate: 1e-6" in contents
        assert "beta: 0.08" in contents
        assert "grpo_loss_type: dr_grpo" in contents
        assert "num_iterations: 1" in contents
        assert "grpo_beta_controller_enabled: false" in contents
        assert "maxent_beta_controller_enabled: false" in contents
        assert "max_grad_norm: 1.0" in contents
        assert "warmup_ratio: 0.0" in contents
        assert "eval_steps: 32" in contents
        assert 'save_strategy: "no"' in contents
    assert "maxent_alpha: 0.005" in entropy
    assert "maxent_alpha_lower_on_high_kl: true" in entropy
    assert "maxent_alpha_disable_outside_trust_zone: true" in entropy
    assert "maxent_tau: 0.2" in listwise
    assert "seed_grpo_enabled: true" in seed
    assert "seed_grpo_alpha: 0.0417" in seed
