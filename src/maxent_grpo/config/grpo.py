"""GRPO-specific configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .dataset import ScriptArguments, trl


@dataclass
class GRPOConfig(trl.GRPOConfig):
    """Additional knobs for GRPO runs (callbacks, benchmarks, etc)."""

    benchmarks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The benchmarks to run after training."},
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The callbacks to run during training."},
    )
    chat_template: Optional[str] = field(
        default=None,
        metadata={"help": "The chat template to use."},
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    num_completions_to_print: int = field(
        default=0,
        metadata={"help": "Number of completions to print."},
    )
    overwrite_hub_revision: bool = field(
        default=False,
        metadata={"help": "Whether to overwrite the Hub revision."},
    )
    push_to_hub_revision: bool = field(
        default=False,
        metadata={"help": "Whether to push to a Hub revision/branch."},
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use."},
    )
    wandb_log_unique_prompts: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to log the unique prompts to wandb. This will create a "
                "new run for each unique prompt."
            )
        },
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_run_group: Optional[str] = field(
        default=None,
        metadata={"help": ("The group to store runs under.")},
    )
    maxent_tau: float = field(
        default=0.0,
        metadata={"help": "Sequence-level entropy weight τ for MaxEnt-GRPO."},
    )
    maxent_q_temperature: float = field(
        default=1.0,
        metadata={
            "help": "Temperature applied when turning utilities into listwise q via softmax."
        },
    )
    maxent_q_epsilon: float = field(
        default=1e-6,
        metadata={
            "help": "Epsilon floor added to q for full support before normalization.",
        },
    )
    maxent_length_normalize_ref: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to length-normalize reference log-probs when computing weights."
            )
        },
    )
    maxent_logprob_chunk_size: int = field(
        default=0,
        metadata={
            "help": (
                "If >0, score reference/policy log-probs in mini-batches of this size "
                "to reduce activation memory."
            )
        },
    )
    maxent_target_weight_entropy: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "If set, automatically tune the MaxEnt entropy weight τ to keep the "
                "average per-prompt weight entropy near this target."
            )
        },
    )
    maxent_tau_lr: float = field(
        default=0.0,
        metadata={
            "help": "Learning rate used for automatic τ tuning (log-space).",
        },
    )
    maxent_tau_min: float = field(
        default=0.0,
        metadata={
            "help": "Lower bound for the learned sequence-level entropy weight τ."
        },
    )
    maxent_tau_max: float = field(
        default=0.0,
        metadata={
            "help": "Upper bound for the learned sequence-level entropy weight τ."
        },
    )
    maxent_tau_warmup_steps: int = field(
        default=-1,
        metadata={
            "help": (
                "Number of steps to keep τ on a warmup schedule before enabling automatic tuning. "
                "Set to a non-negative value to override, or -1 to match the learning-rate warmup."
            )
        },
    )
    maxent_use_clip_objective: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to blend in a PPO-style clipped objective on top of the MaxEnt loss."
            )
        },
    )
    maxent_clip_objective_coef: float = field(
        default=1.0,
        metadata={
            "help": "Scaling applied to the PPO-style clipped objective when enabled."
        },
    )
    maxent_clip_adv_baseline: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional baseline (e.g., 1/K) subtracted from the MaxEnt advantages before clipping."
            )
        },
    )
    train_grpo_objective: bool = field(
        default=False,
        metadata={
            "help": (
                "When true, disable the MaxEnt reference weighting term and train with the standard GRPO objective."
            )
        },
    )
    maxent_clip_range: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "If set, override the PPO clip range specifically for the MaxEnt objective "
                "(falls back to clip_range)."
            )
        },
    )
    init_kl_coeff: Optional[float] = field(
        default=None,
        metadata={
            "help": "Legacy alias for the reverse-KL coefficient β to keep recipes compatible.",
        },
    )
    init_kl_coef: Optional[float] = field(
        default=None,
        metadata={
            "help": "Single-f alias for init_kl_coeff used in some downstream tooling.",
        },
    )
    kl_target: float = field(
        default=0.0,
        metadata={
            "help": (
                "Target KL value for the β controller. Leave at zero to disable automatic adjustment."
            )
        },
    )
    kl_horizon: int = field(
        default=0,
        metadata={
            "help": "Horizon (in optimizer steps) for the β controller. Zero disables the controller.",
        },
    )
    kl_ctl_step_size: float = field(
        default=0.0,
        metadata={
            "help": "Maximum fractional change allowed per β controller update. Zero disables adaptation.",
        },
    )
    clip_range: float = field(
        default=0.2,
        metadata={
            "help": (
                "Legacy PPO clip range knob kept for compatibility; superseded by ppo_clip_range."
            )
        },
    )
    evaluation_strategy: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Alias for eval_strategy to remain compatible with standard transformers TrainingArguments."
            )
        },
    )
    gen_temperature: float = field(
        default=0.8,
        metadata={"help": "Sampling temperature used for candidate generation."},
    )
    gen_top_p: float = field(
        default=0.9,
        metadata={"help": "Top-p used for candidate generation."},
    )
    vllm_url: Optional[str] = field(
        default="http://localhost:8000/generate",
        metadata={"help": "Base URL for vLLM /generate when use_vllm is true."},
    )
    vllm_max_completion_rounds: int = field(
        default=0,
        metadata={
            "help": (
                "Maximum number of vLLM /generate attempts to top off missing "
                "completions. Set to 0 to match the requested num_generations."
            )
        },
    )
    vllm_retry_sleep: float = field(
        default=0.5,
        metadata={
            "help": "Seconds to sleep between vLLM completion retries when n outputs are missing."
        },
    )
    vllm_backfill_with_model: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to fall back to the local HF model.generate() to fill missing "
                "completions after exhausting vLLM retries."
            )
        },
    )
    vllm_return_logprobs: bool = field(
        default=True,
        metadata={
            "help": (
                "Request per-token logprobs from vLLM so the frozen reference model "
                "does not need to rescore completions."
            )
        },
    )
    vllm_best_of: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional best_of parameter forwarded to vLLM (matches TRL)."
        },
    )
    vllm_frequency_penalty: float = field(
        default=0.0,
        metadata={"help": "Frequency penalty applied during sampling (matches TRL)."},
    )
    vllm_presence_penalty: float = field(
        default=0.0,
        metadata={"help": "Presence penalty applied during sampling (matches TRL)."},
    )
    vllm_top_k: Optional[int] = field(
        default=None,
        metadata={"help": "Top-k sampling parameter forwarded to vLLM (matches TRL)."},
    )
    vllm_stop_sequences: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Stop sequences for vLLM; accepts JSON list or '||'-delimited string "
                "to match TRL's CLI."
            )
        },
    )

    def __post_init__(self) -> None:
        try:
            super().__post_init__()
        except AttributeError:
            pass
        eval_alias = getattr(self, "evaluation_strategy", None)
        if eval_alias not in (None, ""):
            try:
                from transformers.training_args import IntervalStrategy  # type: ignore

                eval_value = (
                    eval_alias
                    if isinstance(eval_alias, IntervalStrategy)
                    else IntervalStrategy(str(eval_alias))
                )
            except (ImportError, ModuleNotFoundError, ValueError):
                eval_value = eval_alias
            setattr(self, "eval_strategy", eval_value)


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """Script arguments for the GRPO training script."""

    reward_funcs: list[str] = field(
        default_factory=lambda: ["pure_accuracy_math"],
        metadata={
            "help": "List of reward functions. Allowed: 'pure_accuracy_math' only."
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for repetition penalty reward"},
    )
    dataset_prompt_column: str = field(
        default="problem",
        metadata={"help": "Column to use as prompts for training."},
    )
    dataset_solution_column: str = field(
        default="answer",
        metadata={"help": "Column to use as the gold solution/answer for training."},
    )
    eval_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Optional dataset to use exclusively for evaluation."},
    )
    eval_dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "Config name for the evaluation dataset when applicable."},
    )
    eval_dataset_split: str = field(
        default="validation",
        metadata={"help": "Split to load from the evaluation dataset."},
    )
    eval_dataset_prompt_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "Prompt column for the evaluation dataset (defaults to training column)."
        },
    )
    eval_dataset_solution_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "Answer column for the evaluation dataset (defaults to training column)."
        },
    )
    max_completion_len: int = field(
        default=16384,
        metadata={"help": "Maximum number of characters in completion."},
    )
    soft_punish_cache: int = field(
        default=4096,
        metadata={"help": "Minimum number of characters in completion."},
    )
    span_kl_target: float = field(
        default=0.05, metadata={"help": "per-token KL target"}
    )
    span_kl_beta0: float = field(default=0.12, metadata={"help": "initial KL coeff"})
    span_kl_horizon: int = field(
        default=10000, metadata={"help": "KL controller horizon"}
    )


__all__ = ["GRPOConfig", "GRPOScriptArguments"]
