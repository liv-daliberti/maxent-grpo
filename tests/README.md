# Tests layout

This repository uses a hybrid layout: subpackages for discovery + pytest markers for selection.
Markers are auto-applied in `tests/conftest.py` based on path and filename tokens.

## Structure

- `tests/cli/`: CLI entrypoints and console scripts
- `tests/config/`: configuration and recipe validation
- `tests/core/`: core utilities and data/model helpers
- `tests/evaluation/`: inference, evaluation, scoring
- `tests/generation/`: generation helpers (see `tests/generation/vllm/` for vLLM)
- `tests/ops/`: ops/tooling, perf, validation scripts
- `tests/pipelines/`: pipeline orchestration/integration
- `tests/rewards/`: reward logic (see `tests/rewards/weighting/` for weighting)
- `tests/runtime/`: runtime/deps/setup/logging helpers
- `tests/training/`: training loop, optim, metrics, and helpers

## Markers

- `cli`, `config`, `core`, `evaluation`, `generation`, `pipelines`, `runtime`, `ops`, `training`, `rewards`, `vllm`
- `logging`, `setup` are added from subdirectories or filename tokens
- `integration` is added for pipeline tests and files containing "integration"

Examples:
- `pytest -m "generation and vllm"`
- `pytest -m "training and not slow"`
- `pytest tests/cli -m "integration"`

## Migration map (old -> new)

| Old path | New path |
| --- | --- |
| `tests/test_cli_config_validation.py` | `tests/cli/test_cli_config_validation.py` |
| `tests/test_cli_console_scripts.py` | `tests/cli/test_cli_console_scripts.py` |
| `tests/test_cli_entrypoints.py` | `tests/cli/test_cli_entrypoints.py` |
| `tests/test_cli_generate.py` | `tests/cli/test_cli_generate.py` |
| `tests/test_cli_generate_cli.py` | `tests/cli/test_cli_generate_cli.py` |
| `tests/test_cli_generate_integration.py` | `tests/cli/test_cli_generate_integration.py` |
| `tests/test_cli_hydra_cli.py` | `tests/cli/test_cli_hydra_cli.py` |
| `tests/test_cli_hydra_smoke.py` | `tests/cli/test_cli_hydra_smoke.py` |
| `tests/test_cli_main_module.py` | `tests/cli/test_cli_main_module.py` |
| `tests/test_cli_training_entrypoints.py` | `tests/cli/test_cli_training_entrypoints.py` |
| `tests/test_cli_trl_and_optim.py` | `tests/cli/test_cli_trl_and_optim.py` |
| `tests/test_config_grpo_validation.py` | `tests/config/test_config_grpo_validation.py` |
| `tests/test_config_recipes.py` | `tests/config/test_config_recipes.py` |
| `tests/test_configs_validation.py` | `tests/config/test_configs_validation.py` |
| `tests/test_context_builder.py` | `tests/training/test_context_builder.py` |
| `tests/test_controller_objective.py` | `tests/training/test_controller_objective.py` |
| `tests/test_controller_optimizer.py` | `tests/training/test_controller_optimizer.py` |
| `tests/test_controller_resume.py` | `tests/training/test_controller_resume.py` |
| `tests/test_core_data_hub.py` | `tests/core/test_core_data_hub.py` |
| `tests/test_core_hub_import.py` | `tests/core/test_core_hub_import.py` |
| `tests/test_core_hub_model.py` | `tests/core/test_core_hub_model.py` |
| `tests/test_core_model.py` | `tests/core/test_core_model.py` |
| `tests/test_core_model_additional.py` | `tests/core/test_core_model_additional.py` |
| `tests/test_core_stubs.py` | `tests/core/test_core_stubs.py` |
| `tests/test_data.py` | `tests/core/test_data.py` |
| `tests/test_entrypoints.py` | `tests/cli/test_entrypoints.py` |
| `tests/test_eval_reward_funcs.py` | `tests/rewards/test_eval_reward_funcs.py` |
| `tests/test_evaluation.py` | `tests/evaluation/test_evaluation.py` |
| `tests/test_generate.py` | `tests/generation/test_generate.py` |
| `tests/test_generation_common.py` | `tests/generation/test_generation_common.py` |
| `tests/test_generation_common_additional.py` | `tests/generation/test_generation_common_additional.py` |
| `tests/test_generation_distributed.py` | `tests/generation/test_generation_distributed.py` |
| `tests/test_generation_helpers_additional.py` | `tests/generation/test_generation_helpers_additional.py` |
| `tests/test_generation_helpers_base.py` | `tests/generation/test_generation_helpers_base.py` |
| `tests/test_generation_helpers_coverage.py` | `tests/generation/test_generation_helpers_coverage.py` |
| `tests/test_generation_helpers_distilabel.py` | `tests/generation/test_generation_helpers_distilabel.py` |
| `tests/test_generation_helpers_missing_branches.py` | `tests/generation/test_generation_helpers_missing_branches.py` |
| `tests/test_generation_helpers_module.py` | `tests/generation/test_generation_helpers_module.py` |
| `tests/test_generation_helpers_more.py` | `tests/generation/test_generation_helpers_more.py` |
| `tests/test_generation_helpers_targets.py` | `tests/generation/test_generation_helpers_targets.py` |
| `tests/test_generation_helpers_unit.py` | `tests/generation/test_generation_helpers_unit.py` |
| `tests/test_generation_vllm_additional.py` | `tests/generation/vllm/test_generation_vllm_additional.py` |
| `tests/test_generation_vllm_distributed.py` | `tests/generation/vllm/test_generation_vllm_distributed.py` |
| `tests/test_generation_vllm_helper.py` | `tests/generation/vllm/test_generation_vllm_helper.py` |
| `tests/test_generation_vllm_requests.py` | `tests/generation/vllm/test_generation_vllm_requests.py` |
| `tests/test_generation_vllm_requests_additional.py` | `tests/generation/vllm/test_generation_vllm_requests_additional.py` |
| `tests/test_generation_vllm_requests_char_limit.py` | `tests/generation/vllm/test_generation_vllm_requests_char_limit.py` |
| `tests/test_generation_vllm_requests_prompt_limit.py` | `tests/generation/vllm/test_generation_vllm_requests_prompt_limit.py` |
| `tests/test_generation_vllm_sync_client_and_walk.py` | `tests/generation/vllm/test_generation_vllm_sync_client_and_walk.py` |
| `tests/test_generation_vllm_sync_paths.py` | `tests/generation/vllm/test_generation_vllm_sync_paths.py` |
| `tests/test_generation_vllm_unit.py` | `tests/generation/vllm/test_generation_vllm_unit.py` |
| `tests/test_generation_vllm_weight_sync.py` | `tests/generation/vllm/test_generation_vllm_weight_sync.py` |
| `tests/test_generation_vllm_weight_sync_additional.py` | `tests/generation/vllm/test_generation_vllm_weight_sync_additional.py` |
| `tests/test_generation_vllm_weight_sync_branches.py` | `tests/generation/vllm/test_generation_vllm_weight_sync_branches.py` |
| `tests/test_generation_vllm_weight_sync_new.py` | `tests/generation/vllm/test_generation_vllm_weight_sync_new.py` |
| `tests/test_generation_weight_sync.py` | `tests/generation/vllm/test_generation_weight_sync.py` |
| `tests/test_grpo_config.py` | `tests/config/test_grpo_config.py` |
| `tests/test_grpo_entrypoint.py` | `tests/cli/test_grpo_entrypoint.py` |
| `tests/test_grpo_entrypoints_unit.py` | `tests/cli/test_grpo_entrypoints_unit.py` |
| `tests/test_grpo_main.py` | `tests/cli/test_grpo_main.py` |
| `tests/test_grpo_prompt.py` | `tests/pipelines/test_grpo_prompt.py` |
| `tests/test_hub_utils.py` | `tests/core/test_hub_utils.py` |
| `tests/test_inference_init.py` | `tests/evaluation/test_inference_init.py` |
| `tests/test_inference_math.py` | `tests/evaluation/test_inference_math.py` |
| `tests/test_info_seed_eval.py` | `tests/evaluation/test_info_seed_eval.py` |
| `tests/test_infoseed_integration.py` | `tests/evaluation/test_infoseed_integration.py` |
| `tests/test_integration_pipelines_cpu.py` | `tests/pipelines/test_integration_pipelines_cpu.py` |
| `tests/test_loop_common.py` | `tests/training/test_loop_common.py` |
| `tests/test_maxent_grpo.py` | `tests/cli/test_maxent_grpo.py` |
| `tests/test_maxent_grpo_entry.py` | `tests/cli/test_maxent_grpo_entry.py` |
| `tests/test_model_utils.py` | `tests/core/test_model_utils.py` |
| `tests/test_ops_slurm_train.py` | `tests/ops/test_ops_slurm_train.py` |
| `tests/test_package_init.py` | `tests/core/test_package_init.py` |
| `tests/test_patches_trl.py` | `tests/training/test_patches_trl.py` |
| `tests/test_perf_microbench.py` | `tests/ops/test_perf_microbench.py` |
| `tests/test_pipelines_generation.py` | `tests/pipelines/test_pipelines_generation.py` |
| `tests/test_pipelines_inference_math.py` | `tests/pipelines/test_pipelines_inference_math.py` |
| `tests/test_pipelines_training_baseline.py` | `tests/pipelines/test_pipelines_training_baseline.py` |
| `tests/test_pipelines_training_maxent.py` | `tests/pipelines/test_pipelines_training_maxent.py` |
| `tests/test_recipes.py` | `tests/config/test_recipes.py` |
| `tests/test_reference_scoring_hang_safety.py` | `tests/evaluation/test_reference_scoring_hang_safety.py` |
| `tests/test_rewards.py` | `tests/rewards/test_rewards.py` |
| `tests/test_rewards_basic_extra.py` | `tests/rewards/test_rewards_basic_extra.py` |
| `tests/test_rewards_maxent_exports.py` | `tests/rewards/test_rewards_maxent_exports.py` |
| `tests/test_rewards_module.py` | `tests/rewards/test_rewards_module.py` |
| `tests/test_rewards_registry.py` | `tests/rewards/test_rewards_registry.py` |
| `tests/test_rewards_unit.py` | `tests/rewards/test_rewards_unit.py` |
| `tests/test_run_generation.py` | `tests/generation/test_run_generation.py` |
| `tests/test_run_generation_local.py` | `tests/generation/test_run_generation_local.py` |
| `tests/test_run_generation_vllm.py` | `tests/generation/vllm/test_run_generation_vllm.py` |
| `tests/test_run_helpers.py` | `tests/training/test_run_helpers.py` |
| `tests/test_run_helpers_stub_additional.py` | `tests/training/test_run_helpers_stub_additional.py` |
| `tests/test_run_logging.py` | `tests/runtime/logging/test_run_logging.py` |
| `tests/test_run_setup_reference.py` | `tests/helpers/run_setup_reference.py` |
| `tests/test_run_training_eval.py` | `tests/training/test_run_training_eval.py` |
| `tests/test_run_training_loop.py` | `tests/training/test_run_training_loop.py` |
| `tests/test_run_training_loss.py` | `tests/training/test_run_training_loss.py` |
| `tests/test_run_training_metrics.py` | `tests/training/test_run_training_metrics.py` |
| `tests/test_run_training_pipeline.py` | `tests/training/test_run_training_pipeline.py` |
| `tests/test_run_training_rewards.py` | `tests/training/test_run_training_rewards.py` |
| `tests/test_run_training_scoring.py` | `tests/training/test_run_training_scoring.py` |
| `tests/test_run_training_state.py` | `tests/training/test_run_training_state.py` |
| `tests/test_run_training_types.py` | `tests/training/test_run_training_types.py` |
| `tests/test_run_training_weighting.py` | `tests/training/test_run_training_weighting.py` |
| `tests/test_runtime_logging.py` | `tests/runtime/logging/test_runtime_logging.py` |
| `tests/test_runtime_setup_additional.py` | `tests/runtime/setup/test_runtime_setup_additional.py` |
| `tests/test_scoring.py` | `tests/evaluation/test_scoring.py` |
| `tests/test_scoring_additional.py` | `tests/evaluation/test_scoring_additional.py` |
| `tests/test_scoring_autocast_additional.py` | `tests/evaluation/test_scoring_autocast_additional.py` |
| `tests/test_setup_patch.py` | `tests/runtime/setup/test_setup_patch.py` |
| `tests/test_sitecustomize_env.py` | `tests/runtime/setup/test_sitecustomize_env.py` |
| `tests/test_tools_eval_math_delta.py` | `tests/ops/test_tools_eval_math_delta.py` |
| `tests/test_training_cli_eval.py` | `tests/cli/test_training_cli_eval.py` |
| `tests/test_training_cli_trl.py` | `tests/cli/test_training_cli_trl.py` |
| `tests/test_training_data.py` | `tests/training/test_training_data.py` |
| `tests/test_training_generation_distributed.py` | `tests/training/generation/test_training_generation_distributed.py` |
| `tests/test_training_generation_generator.py` | `tests/training/generation/test_training_generation_generator.py` |
| `tests/test_training_generation_helpers_comm.py` | `tests/training/generation/test_training_generation_helpers_comm.py` |
| `tests/test_training_generation_helpers_local.py` | `tests/training/generation/test_training_generation_helpers_local.py` |
| `tests/test_training_generation_helpers_more.py` | `tests/training/generation/test_training_generation_helpers_more.py` |
| `tests/test_training_generation_helpers_unit.py` | `tests/training/generation/test_training_generation_helpers_unit.py` |
| `tests/test_training_generation_helpers_wrappers.py` | `tests/training/generation/test_training_generation_helpers_wrappers.py` |
| `tests/test_training_generation_vllm_adapter_additional.py` | `tests/training/generation/vllm/test_training_generation_vllm_adapter_additional.py` |
| `tests/test_training_generation_vllm_adapter_fallback.py` | `tests/training/generation/vllm/test_training_generation_vllm_adapter_fallback.py` |
| `tests/test_training_generation_vllm_adapter_more.py` | `tests/training/generation/vllm/test_training_generation_vllm_adapter_more.py` |
| `tests/test_training_init.py` | `tests/training/test_training_init.py` |
| `tests/test_training_logging_runtime.py` | `tests/runtime/logging/test_training_logging_runtime.py` |
| `tests/test_training_loop_unit.py` | `tests/training/test_training_loop_unit.py` |
| `tests/test_training_loop_vllm_guard.py` | `tests/training/generation/vllm/test_training_loop_vllm_guard.py` |
| `tests/test_training_metrics_logging.py` | `tests/runtime/logging/test_training_metrics_logging.py` |
| `tests/test_training_metrics_unit.py` | `tests/training/test_training_metrics_unit.py` |
| `tests/test_training_optim.py` | `tests/training/test_training_optim.py` |
| `tests/test_training_optim_additional.py` | `tests/training/test_training_optim_additional.py` |
| `tests/test_training_optim_bounds.py` | `tests/training/test_training_optim_bounds.py` |
| `tests/test_training_pipeline_collect_stats.py` | `tests/pipelines/test_training_pipeline_collect_stats.py` |
| `tests/test_training_pipeline_edges.py` | `tests/pipelines/test_training_pipeline_edges.py` |
| `tests/test_training_pipeline_parity.py` | `tests/pipelines/test_training_pipeline_parity.py` |
| `tests/test_training_resume_and_cadence.py` | `tests/training/test_training_resume_and_cadence.py` |
| `tests/test_training_rewards_advantages.py` | `tests/rewards/test_training_rewards_advantages.py` |
| `tests/test_training_rewards_branches.py` | `tests/rewards/test_training_rewards_branches.py` |
| `tests/test_training_rewards_helpers.py` | `tests/rewards/test_training_rewards_helpers.py` |
| `tests/test_training_runtime_deepspeed.py` | `tests/runtime/test_training_runtime_deepspeed.py` |
| `tests/test_training_runtime_deps.py` | `tests/runtime/test_training_runtime_deps.py` |
| `tests/test_training_runtime_logging.py` | `tests/runtime/logging/test_training_runtime_logging.py` |
| `tests/test_training_runtime_logging_additional.py` | `tests/runtime/logging/test_training_runtime_logging_additional.py` |
| `tests/test_training_runtime_logging_env.py` | `tests/runtime/logging/test_training_runtime_logging_env.py` |
| `tests/test_training_runtime_prompts.py` | `tests/runtime/test_training_runtime_prompts.py` |
| `tests/test_training_runtime_setup.py` | `tests/runtime/setup/test_training_runtime_setup.py` |
| `tests/test_training_runtime_setup_additional.py` | `tests/runtime/setup/test_training_runtime_setup_additional.py` |
| `tests/test_training_runtime_torch_stub_additional.py` | `tests/runtime/test_training_runtime_torch_stub_additional.py` |
| `tests/test_training_runtime_torch_utils.py` | `tests/runtime/test_training_runtime_torch_utils.py` |
| `tests/test_training_types_structs.py` | `tests/training/test_training_types_structs.py` |
| `tests/test_trl_logging.py` | `tests/runtime/logging/test_trl_logging.py` |
| `tests/test_trl_patches.py` | `tests/training/test_trl_patches.py` |
| `tests/test_trl_weighting_logging.py` | `tests/runtime/logging/test_trl_weighting_logging.py` |
| `tests/test_types_logging.py` | `tests/runtime/logging/test_types_logging.py` |
| `tests/test_utils_fallbacks_imports.py` | `tests/core/test_utils_fallbacks_imports.py` |
| `tests/test_utils_hub.py` | `tests/core/test_utils_hub.py` |
| `tests/test_utils_stubs.py` | `tests/core/test_utils_stubs.py` |
| `tests/test_validate_logs.py` | `tests/ops/test_validate_logs.py` |
| `tests/test_validate_training_logs.py` | `tests/ops/test_validate_training_logs.py` |
| `tests/test_validation.py` | `tests/pipelines/test_validation.py` |
| `tests/test_vllm_patch.py` | `tests/generation/vllm/test_vllm_patch.py` |
| `tests/test_vllm_patch_additional.py` | `tests/generation/vllm/test_vllm_patch_additional.py` |
| `tests/test_vllm_server_patch.py` | `tests/generation/vllm/test_vllm_server_patch.py` |
| `tests/test_wandb_logging.py` | `tests/runtime/logging/test_wandb_logging.py` |
| `tests/test_weighting_logic.py` | `tests/rewards/weighting/test_weighting_logic.py` |
| `tests/test_weighting_logic_branches.py` | `tests/rewards/weighting/test_weighting_logic_branches.py` |
| `tests/test_weighting_loss.py` | `tests/rewards/weighting/test_weighting_loss.py` |
| `tests/test_weighting_loss_additional.py` | `tests/rewards/weighting/test_weighting_loss_additional.py` |
| `tests/test_weighting_loss_devices.py` | `tests/rewards/weighting/test_weighting_loss_devices.py` |
| `tests/test_weighting_types.py` | `tests/rewards/weighting/test_weighting_types.py` |
| `tests/test_zero_utils.py` | `tests/training/test_zero_utils.py` |
| `tests/test_zero_utils_additional.py` | `tests/training/test_zero_utils_additional.py` |
