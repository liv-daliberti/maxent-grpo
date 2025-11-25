"""
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import importlib
import types
from types import SimpleNamespace

import pytest

from maxent_grpo.training.types import PromptCacheEntry


def test_reward_computation_and_loss_accessors(training_stubs):
    rewards = importlib.import_module("training.types.rewards")

    advantage = rewards.AdvantageStats(grouped=[[0.1, 0.2]], samples=[0.1, 0.2])
    q_dist = rewards.QDistribution(grouped=[[0.3, 0.4]], samples=[0.3, 0.4])
    moments = rewards.RewardMoments(mean=1.5, std=0.5)
    pairs = rewards.PromptCompletionBatch(prompts=["p"], completions=["c"])
    rc = rewards.RewardComputation(
        total_utils=[1.0],
        per_reward_values={"r": [1.0]},
        advantage=advantage,
        pairs=pairs,
        q_distribution=q_dist,
        moments=moments,
    )
    assert rc.advantage_samples == [0.1, 0.2]
    assert rc.q_grouped == [[0.3, 0.4]]
    assert rc.train_reward_mean == 1.5
    assert rc.train_reward_std == 0.5

    scalars = rewards.LossScalarBundle(
        total_loss=3.0,
        policy_loss=1.0,
        clip_loss=0.25,
        kl_loss=0.5,
        weighted_kl_loss=0.75,
    )
    loss_outputs = rewards.LossOutputs(
        loss="stub_tensor",  # value unused in accessors
        scalars=scalars,
        log_ratio_train="log_ratio",
        denom_tok_tensor="denom",
    )
    assert loss_outputs.total_loss_scalar == 3.0
    assert loss_outputs.policy_loss_scalar == 1.0
    assert loss_outputs.clip_loss_scalar == 0.25
    assert loss_outputs.kl_loss_scalar == 0.5
    assert loss_outputs.weighted_kl_loss_scalar == 0.75

    ref = rewards.ReferenceLogprobs(
        ref_logp_sum="sum",
        ref_tok_counts="counts",
        ref_logp_sum_raw="raw",
        ref_logp_mean=0.0,
        avg_completion_tokens=1.0,
    )
    diag = rewards.BatchDiagnostics(
        kl_value=None,
        clip_ratio=0.1,
        clip_ratio_low_mean=0.0,
        clip_ratio_low_min=0.0,
        clip_ratio_high_mean=0.0,
        clip_ratio_high_max=0.0,
        clip_ratio_region_mean=0.0,
    )
    score_batch = rewards.PromptCacheEntry(
        input_ids=[1, 2, 3], attention_mask=[1, 1, 1]
    )
    assert score_batch.length == 3
    # ensure __all__ members are constructed without side effects
    _ = (ref, diag, score_batch)


@pytest.fixture()
def training_stubs(monkeypatch):
    """Install lightweight stubs for training deps before importing runtime."""
    from tests.helpers.run_setup_stubs import install_training_stubs

    return install_training_stubs(monkeypatch)


def test_generation_settings_passthrough(training_stubs):
    from maxent_grpo.training.run_helpers import (
        GenerationPenaltyConfig,
        VLLMClientConfig,
    )

    runtime = importlib.reload(importlib.import_module("training.types.runtime"))
    penalty = GenerationPenaltyConfig()
    vllm_cfg = VLLMClientConfig(
        url="http://localhost",
        rounds_cfg=1,
        retry_sleep=0.1,
        backfill_local=False,
        request_logprobs=False,
    )
    gs = runtime.GenerationSettings(
        max_prompt_len=1,
        max_completion_len=2,
        gen_temperature=0.5,
        gen_top_p=0.9,
        use_vllm=True,
        vllm=vllm_cfg,
        penalty=penalty,
    )

    gs.gen_top_k = 5
    gs.gen_best_of = 3
    gs.gen_frequency_penalty = 0.2
    gs.gen_presence_penalty = 0.4
    gs.gen_stop_sequences = ["</s>"]

    assert gs.gen_top_k == 5
    assert gs.gen_best_of == 3
    assert gs.gen_frequency_penalty == 0.2
    assert gs.gen_presence_penalty == 0.4
    assert gs.gen_stop_sequences == ["</s>"]
    # ensure we updated the underlying penalty object
    assert penalty.gen_top_k == 5
    assert penalty.gen_stop_sequences == ["</s>"]


def test_training_loop_context_properties(training_stubs):
    from maxent_grpo.training.run_helpers import VLLMClientConfig
    from maxent_grpo.training.weighting.types import (
        KlControllerSettings,
        QDistributionSettings,
        TauSchedule,
        WeightNormalizationSettings,
        WeightingSettings,
    )

    runtime = importlib.reload(importlib.import_module("training.types.runtime"))

    vllm_cfg = VLLMClientConfig(
        url="http://localhost",
        rounds_cfg=1,
        retry_sleep=0.1,
        backfill_local=False,
        request_logprobs=False,
    )
    generation = runtime.GenerationSettings(
        max_prompt_len=1,
        max_completion_len=2,
        gen_temperature=0.5,
        gen_top_p=0.9,
        use_vllm=True,
        vllm=vllm_cfg,
    )
    evaluation = runtime.EvaluationSettings(
        enabled=True, rows=[{"prompt": "p"}], batch_size=1, every_n_steps=10
    )
    schedule = runtime.OptimizationSchedule(
        num_epochs=1,
        num_generations=1,
        grad_accum_steps=1,
        max_grad_norm=1.0,
        steps_per_epoch=None,
        total_training_steps=1,
        warmup_steps=0,
    )
    optimizer_handle = runtime.OptimizerHandles(
        optimizer=runtime.Optimizer(),
        lr_scheduler=None,
        base_optimizer=runtime.Optimizer(),
        learning_rate=0.01,
    )
    optimization = runtime.OptimizationSettings(
        schedule=schedule, handles=optimizer_handle
    )
    weighting = WeightingSettings(
        tau=0.1,
        beta=0.2,
        normalization=WeightNormalizationSettings(denom=1.0, len_norm_ref=True),
        q_distribution=QDistributionSettings(temperature=1.0, epsilon=1e-6),
        tau_schedule=TauSchedule(
            target_entropy=None,
            learning_rate=0.01,
            minimum_value=0.0,
            maximum_value=1.0,
            warmup_steps=0,
        ),
        kl_controller=KlControllerSettings(target=0.1, horizon=10, step_size=0.5),
        train_grpo_objective=True,
    )
    batching = runtime.BatchingSettings(
        logprob_chunk_size=1,
        score_slice=1,
        prompt_length_cache_get=lambda _p: PromptCacheEntry([1], [1]),
    )
    scoring = runtime.ScoringSettings(
        weighting=weighting,
        clipping=runtime.ClipSettings(
            clip_range=0.1,
            use_clip_objective=False,
            clip_objective_coef=0.0,
            clip_adv_baseline=None,
        ),
        batching=batching,
    )
    controller = runtime.ControllerPaths(
        state_path=None, resume_from=None, overwrite_existing=False
    )
    loop_settings = runtime.LoopSettings(
        generation=generation,
        evaluation=evaluation,
        optimization=optimization,
        scoring=scoring,
        controller=controller,
    )
    runtime_handles = runtime.RuntimeHandles(
        accelerator=runtime.Accelerator(),
        model=runtime.PreTrainedModel(),
        tokenizer=runtime.PreTrainedTokenizer(),
        train_loader=runtime.DataLoader(),
        train_sampler=None,
        device=types.SimpleNamespace(),
        get_ref_model=lambda: "ref",
    )
    reward_spec = runtime.RewardSpec(reward_funcs=["f"], reward_weights=[1.0])
    ctx = runtime.TrainingLoopContext(
        runtime=runtime_handles,
        reward=reward_spec,
        settings=loop_settings,
        logging=SimpleNamespace(),
    )

    assert ctx.generation is generation
    assert ctx.evaluation is evaluation
    assert ctx.optimization is optimization
    assert ctx.scoring is scoring
    assert ctx.controller is controller
