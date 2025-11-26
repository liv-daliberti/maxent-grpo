import types

from maxent_grpo.config import GRPOConfig
from maxent_grpo.training.context_builder import apply_info_seed
from maxent_grpo.training.types import (
    BatchingSettings,
    ClipSettings,
    EvaluationSettings,
    GenerationSettings,
    ScoringSettings,
)
from maxent_grpo.training.weighting.types import (
    KlControllerSettings,
    QDistributionSettings,
    TauSchedule,
    WeightNormalizationSettings,
    WeightingSettings,
)


def _dummy_weighting() -> WeightingSettings:
    return WeightingSettings(
        tau=1.0,
        beta=0.0,
        normalization=WeightNormalizationSettings(denom=1.0, len_norm_ref=False),
        q_distribution=QDistributionSettings(temperature=1.0, epsilon=1e-6),
        tau_schedule=TauSchedule(
            target_entropy=None,
            learning_rate=0.0,
            minimum_value=0.0,
            maximum_value=0.0,
            warmup_steps=0,
        ),
        kl_controller=KlControllerSettings(target=0.0, horizon=1, step_size=0.0),
        train_grpo_objective=True,
    )


def test_apply_info_seed_maps_config():
    cfg = GRPOConfig(
        info_seed_enabled=True,
        info_seed_num_seeds=5,
        info_seed_lambda=0.2,
        info_seed_temperature=0.33,
        info_seed_loss_type="ce",
        info_seed_pooling="last",
        info_seed_prompt_template="<seed={seed}>",
    )
    generation = GenerationSettings(
        max_prompt_len=128,
        max_completion_len=64,
        gen_temperature=1.0,
        gen_top_p=0.95,
        use_vllm=False,
        vllm=types.SimpleNamespace(),
    )
    scoring = ScoringSettings(
        weighting=_dummy_weighting(),
        clipping=ClipSettings(
            clip_range=0.2,
            use_clip_objective=False,
            clip_objective_coef=1.0,
            clip_adv_baseline=None,
        ),
        batching=BatchingSettings(
            logprob_chunk_size=0,
            score_slice=0,
            prompt_length_cache_get=lambda s: types.SimpleNamespace(
                input_ids=[], attention_mask=[], length=0
            ),
        ),
    )
    evaluation = EvaluationSettings(
        enabled=False,
        rows=[],
        batch_size=1,
        every_n_steps=None,
    )
    generation, scoring, evaluation = apply_info_seed(
        generation, scoring, evaluation, cfg
    )
    assert generation.seed_augmentation is not None
    assert generation.seed_augmentation.num_seeds == 5
    assert generation.seed_augmentation.template == "<seed={seed}>"
    assert scoring.info_seed_lambda == 0.2
    assert scoring.info_seed_temperature == 0.33
    assert scoring.info_seed_loss_type == "ce"
    assert scoring.info_seed_pooling == "last"
    assert evaluation.seed_eval and evaluation.seed_eval["enabled"]
