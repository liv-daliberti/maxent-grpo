from __future__ import annotations

from pathlib import Path

import yaml

_OBJECTIVE_SPECIFIC_TRAINING_KEYS = {
    "beta",
    "objective",
    "maxent_alpha",
    "maxent_alpha_disable_outside_trust_zone",
    "maxent_alpha_kl_gain",
    "maxent_alpha_kl_threshold",
    "maxent_alpha_lower_on_high_kl",
    "maxent_alpha_raise_on_low_kl",
    "maxent_length_normalize_ref",
    "maxent_length_normalize_policy",
    "maxent_policy_entropy",
    "maxent_policy_entropy_mode",
    "maxent_q_epsilon",
    "maxent_q_temperature",
    "maxent_clip_objective_coef",
    "maxent_reference_logprobs_source",
    "maxent_reference_ema_enabled",
    "maxent_trl_reference_scoring",
    "maxent_tau",
    "maxent_use_clip_objective",
    "output_dir",
}


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs" / "recipes").exists():
            return parent
    raise RuntimeError("Unable to locate repository root")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    assert isinstance(payload, dict)
    return payload


def _merged_training_payload(root_key: str, payload: dict, repo: Path) -> dict:
    block = payload[root_key]
    merged = _load_yaml(repo / block["recipe"])
    merged.update(block.get("training", {}))
    return merged


def _assert_triplet(
    *,
    grpo_cfg: dict,
    entropy_cfg: dict,
    listwise_cfg: dict,
    shared_recipe: str,
    grpo_output_dir: str,
    entropy_output_dir: str,
    listwise_output_dir: str,
    explicit_num_generations: int | None = None,
) -> None:
    assert grpo_cfg["maxent"]["recipe"] == shared_recipe
    assert entropy_cfg["maxent"]["recipe"] == shared_recipe
    assert listwise_cfg["maxent"]["recipe"] == shared_recipe

    assert grpo_cfg["command"] == "train-maxent"
    assert entropy_cfg["command"] == "train-maxent"
    assert listwise_cfg["command"] == "train-maxent"

    assert grpo_cfg["maxent"]["training"]["objective"] == "grpo"
    assert grpo_cfg["maxent"]["training"]["controller_meta_enabled"] is False
    assert grpo_cfg["maxent"]["training"]["output_dir"] == grpo_output_dir

    assert entropy_cfg["maxent"]["training"]["objective"] == "maxent_entropy"
    assert entropy_cfg["maxent"]["training"]["maxent_alpha"] > 0.0
    assert entropy_cfg["maxent"]["training"]["controller_meta_enabled"] is False
    assert entropy_cfg["maxent"]["training"]["maxent_reference_ema_enabled"] is False
    assert entropy_cfg["maxent"]["training"]["output_dir"] == entropy_output_dir
    assert entropy_cfg["maxent"]["training"]["maxent_policy_entropy_mode"] == "exact"

    assert listwise_cfg["maxent"]["training"]["objective"] == "maxent_listwise"
    assert listwise_cfg["maxent"]["training"]["maxent_alpha"] == 0.0
    assert listwise_cfg["maxent"]["training"]["maxent_tau"] > 0.0
    assert listwise_cfg["maxent"]["training"]["controller_meta_enabled"] is False
    assert listwise_cfg["maxent"]["training"]["maxent_reference_ema_enabled"] is False
    assert listwise_cfg["maxent"]["training"]["output_dir"] == listwise_output_dir

    for cfg in (grpo_cfg, entropy_cfg, listwise_cfg):
        assert cfg["hydra"]["run"]["dir"]
        assert cfg["hydra"]["sweep"]["dir"]
        assert cfg["hydra"]["sweep"]["subdir"] == "${hydra.job.num}"
    if explicit_num_generations is not None:
        assert grpo_cfg["maxent"]["training"]["num_generations"] == explicit_num_generations
        assert (
            entropy_cfg["maxent"]["training"]["num_generations"] == explicit_num_generations
        )
        assert (
            listwise_cfg["maxent"]["training"]["num_generations"] == explicit_num_generations
        )


def _assert_triplet_only_differs_on_objective_specific_keys(
    root_key: str,
    *,
    grpo_cfg: dict,
    entropy_cfg: dict,
    listwise_cfg: dict,
    repo: Path,
) -> None:
    merged_triplet = [
        _merged_training_payload(root_key, cfg, repo)
        for cfg in (grpo_cfg, entropy_cfg, listwise_cfg)
    ]
    shared_keys = set.intersection(*(set(cfg) for cfg in merged_triplet))
    differing_keys = {
        key
        for key in shared_keys
        if not all(cfg[key] == merged_triplet[0][key] for cfg in merged_triplet[1:])
    }
    assert differing_keys <= _OBJECTIVE_SPECIFIC_TRAINING_KEYS


def test_math_objective_triplet_hydra_presets_are_explicit_and_aligned() -> None:
    repo = _repo_root()
    grpo_cfg = _load_yaml(repo / "configs/recipes/hydra/grpo_custom_math.yaml")
    entropy_cfg = _load_yaml(repo / "configs/recipes/hydra/maxent_entropy_math.yaml")
    listwise_cfg = _load_yaml(repo / "configs/recipes/hydra/maxent_listwise_math.yaml")
    _assert_triplet(
        grpo_cfg=grpo_cfg,
        entropy_cfg=entropy_cfg,
        listwise_cfg=listwise_cfg,
        shared_recipe="configs/recipes/Qwen2.5-1.5B-Instruct/grpo/config_math.yaml",
        grpo_output_dir="var/data/Qwen2.5-1.5B-Open-R1-GRPO-BASELINE-math-v1",
        entropy_output_dir="var/data/Qwen2.5-1.5B-Open-R1-MaxEnt-Entropy-math-v1",
        listwise_output_dir="var/data/Qwen2.5-1.5B-Open-R1-MaxEnt-Listwise-math-v1",
        explicit_num_generations=8,
    )
    _assert_triplet_only_differs_on_objective_specific_keys(
        "maxent",
        grpo_cfg=grpo_cfg,
        entropy_cfg=entropy_cfg,
        listwise_cfg=listwise_cfg,
        repo=repo,
    )
    merged_grpo = _merged_training_payload("maxent", grpo_cfg, repo)
    assert merged_grpo["gradient_accumulation_steps"] == 40
    assert merged_grpo["do_eval"] is False
    assert merged_grpo["eval_strategy"] == "no"
    assert merged_grpo["eval_on_start"] is True
    assert merged_grpo["eval_steps"] == 10
    merged_listwise = _merged_training_payload("maxent", listwise_cfg, repo)
    merged_entropy = _merged_training_payload("maxent", entropy_cfg, repo)
    assert merged_listwise["beta"] == 0.0
    assert merged_listwise["maxent_tau"] == 0.5
    assert merged_listwise["maxent_q_temperature"] == 2.0
    assert merged_listwise["maxent_use_clip_objective"] is True
    assert merged_listwise["maxent_clip_objective_coef"] == 1.0
    assert merged_listwise["maxent_reference_logprobs_source"] == "model"
    assert merged_listwise["maxent_trl_reference_scoring"] is True
    assert merged_listwise["maxent_length_normalize_ref"] is True
    assert merged_listwise["maxent_length_normalize_policy"] is True
    assert merged_entropy["maxent_alpha_lower_on_high_kl"] is True
    assert merged_entropy["maxent_alpha_disable_outside_trust_zone"] is True
    assert merged_entropy["maxent_alpha_kl_threshold"] == 0.07
    assert merged_entropy["maxent_alpha_kl_gain"] == 0.5


def test_math_stable_objective_triplet_hydra_presets_are_explicit_and_aligned() -> None:
    repo = _repo_root()
    grpo_cfg = _load_yaml(repo / "configs/recipes/hydra/grpo_custom_math_stable.yaml")
    entropy_cfg = _load_yaml(
        repo / "configs/recipes/hydra/maxent_entropy_math_stable.yaml"
    )
    listwise_cfg = _load_yaml(
        repo / "configs/recipes/hydra/maxent_listwise_math_stable.yaml"
    )
    _assert_triplet(
        grpo_cfg=grpo_cfg,
        entropy_cfg=entropy_cfg,
        listwise_cfg=listwise_cfg,
        shared_recipe="configs/recipes/Qwen2.5-1.5B-Instruct/grpo/config_math_stable.yaml",
        grpo_output_dir="var/data/Qwen2.5-1.5B-Open-R1-GRPO-BASELINE-math-stable-v1",
        entropy_output_dir="var/data/Qwen2.5-1.5B-Open-R1-MaxEnt-Entropy-math-stable-v1",
        listwise_output_dir="var/data/Qwen2.5-1.5B-Open-R1-MaxEnt-Listwise-math-stable-v1",
        explicit_num_generations=8,
    )
    _assert_triplet_only_differs_on_objective_specific_keys(
        "maxent",
        grpo_cfg=grpo_cfg,
        entropy_cfg=entropy_cfg,
        listwise_cfg=listwise_cfg,
        repo=repo,
    )
    merged_listwise = _merged_training_payload("maxent", listwise_cfg, repo)
    merged_grpo = _merged_training_payload("maxent", grpo_cfg, repo)
    assert merged_listwise["maxent_reference_logprobs_source"] == "model"
    assert merged_listwise["maxent_trl_reference_scoring"] is True
    assert merged_listwise["maxent_use_clip_objective"] is True
    assert merged_listwise["maxent_clip_objective_coef"] == 1.0
    assert merged_listwise["maxent_length_normalize_ref"] is True
    assert merged_listwise["maxent_length_normalize_policy"] is True
    assert merged_listwise["maxent_tau"] == 0.5
    assert merged_listwise["maxent_q_temperature"] == 2.0
    assert merged_grpo["gradient_accumulation_steps"] == 40
    assert merged_grpo["max_completion_length"] == 3000
    assert merged_grpo["dr_grpo_denominator_mode"] == "active_tokens"
    assert merged_grpo["reward_funcs"] == [
        "seed_paper_boxed_accuracy_math",
        "missing_boxed_answer_penalty_math",
    ]
    assert merged_grpo["eval_reward_funcs"] == ["seed_paper_boxed_accuracy_math"]
    assert merged_grpo["do_eval"] is False
    assert merged_grpo["eval_strategy"] == "no"
    assert merged_grpo["missing_boxed_answer_penalty"] == -0.05
    assert "<answer>" in merged_grpo["system_prompt"]
    assert "\\boxed{FINAL_ANSWER}" in merged_grpo["system_prompt"]
    assert "message['content']" in merged_grpo["chat_template"]
    assert merged_grpo["eval_on_start"] is True
    assert merged_grpo["eval_steps"] == 10
    assert merged_grpo["save_steps"] == 1
    assert merged_grpo["greedy_eval_enabled"] is True
    assert merged_grpo["eval_greedy_only_enabled"] is True
    assert merged_grpo["truncate_completions_at_first_boxed_answer"] is True


def test_math_fair_objective_quartet_hydra_presets_are_explicit_and_aligned() -> None:
    repo = _repo_root()
    grpo_cfg = _load_yaml(repo / "configs/recipes/hydra/grpo_custom_math_fair.yaml")
    entropy_cfg = _load_yaml(repo / "configs/recipes/hydra/maxent_entropy_math_fair.yaml")
    listwise_cfg = _load_yaml(repo / "configs/recipes/hydra/maxent_listwise_math_fair.yaml")
    seed_cfg = _load_yaml(repo / "configs/recipes/hydra/seed_grpo_math_fair.yaml")

    _assert_triplet(
        grpo_cfg=grpo_cfg,
        entropy_cfg=entropy_cfg,
        listwise_cfg=listwise_cfg,
        shared_recipe="configs/recipes/Qwen2.5-1.5B-Instruct/grpo/config_math_fair.yaml",
        grpo_output_dir="var/data/Qwen2.5-1.5B-Open-R1-GRPO-BASELINE-math-fair-v1",
        entropy_output_dir="var/data/Qwen2.5-1.5B-Open-R1-MaxEnt-Entropy-math-fair-v1",
        listwise_output_dir="var/data/Qwen2.5-1.5B-Open-R1-MaxEnt-Listwise-math-fair-v1",
        explicit_num_generations=8,
    )
    _assert_triplet_only_differs_on_objective_specific_keys(
        "maxent",
        grpo_cfg=grpo_cfg,
        entropy_cfg=entropy_cfg,
        listwise_cfg=listwise_cfg,
        repo=repo,
    )

    for cfg in (grpo_cfg, entropy_cfg, listwise_cfg, seed_cfg):
        assert cfg["command"] == "train-maxent"
        assert (
            cfg["maxent"]["recipe"]
            == "configs/recipes/Qwen2.5-1.5B-Instruct/grpo/config_math_fair.yaml"
        )
        assert cfg["maxent"]["training"]["gradient_accumulation_steps"] == 40
        assert cfg["maxent"]["training"]["eval_on_start"] is True
        assert cfg["maxent"]["training"]["eval_steps"] == 25
        assert cfg["maxent"]["training"]["seed_paper_eval_enabled"] is True
        assert cfg["maxent"]["training"]["seed_paper_eval_fail_on_error"] is True
        assert cfg["maxent"]["training"]["seed_paper_eval_template"] == "no"
        assert cfg["maxent"]["training"]["push_to_hub"] is False

    merged_grpo = _merged_training_payload("maxent", grpo_cfg, repo)
    merged_entropy = _merged_training_payload("maxent", entropy_cfg, repo)
    merged_listwise = _merged_training_payload("maxent", listwise_cfg, repo)
    merged_seed = _merged_training_payload("maxent", seed_cfg, repo)

    assert merged_grpo["prompt_template"] == "no"
    assert merged_grpo["system_prompt"] is None
    assert merged_grpo["chat_template"] is None
    assert merged_grpo["seed_paper_eval_template"] == "no"
    assert merged_grpo["reward_funcs"] == ["seed_paper_boxed_accuracy_math"]
    assert merged_grpo["eval_reward_funcs"] == ["seed_paper_boxed_accuracy_math"]
    assert merged_grpo["do_eval"] is False
    assert merged_grpo["eval_strategy"] == "no"
    assert merged_grpo["dr_grpo_denominator_mode"] == "fixed_max"
    assert "vllm_stop_sequences" not in merged_grpo
    assert "truncate_completions_at_first_boxed_answer" not in merged_grpo
    assert "missing_boxed_answer_penalty_math" not in merged_grpo.get("reward_funcs", [])

    assert merged_entropy["maxent_alpha"] == 0.005
    assert merged_entropy["beta"] == 0.0
    assert merged_entropy["maxent_alpha_lower_on_high_kl"] is True
    assert merged_entropy["maxent_alpha_disable_outside_trust_zone"] is True

    assert merged_listwise["beta"] == 0.08
    assert merged_listwise["maxent_tau"] == 0.5
    assert merged_listwise["maxent_q_temperature"] == 2.0
    assert merged_listwise["maxent_use_clip_objective"] is True
    assert merged_listwise["maxent_clip_objective_coef"] == 1.0
    assert merged_listwise["maxent_reference_logprobs_source"] == "model"
    assert merged_listwise["maxent_trl_reference_scoring"] is True
    assert merged_listwise["maxent_length_normalize_ref"] is True
    assert merged_listwise["maxent_length_normalize_policy"] is True

    assert merged_seed["seed_grpo_enabled"] is True
    assert merged_seed["seed_grpo_alpha"] == 0.0417
    assert merged_seed["seed_grpo_alpha_normalize_by_max_entropy"] is True
    assert merged_seed["seed_grpo_length_normalize_logprobs"] is True


def test_code_objective_triplet_hydra_presets_are_explicit_and_aligned() -> None:
    repo = _repo_root()
    grpo_cfg = _load_yaml(repo / "configs/recipes/hydra/grpo_custom_code_mbpp.yaml")
    entropy_cfg = _load_yaml(repo / "configs/recipes/hydra/maxent_entropy_code_mbpp.yaml")
    listwise_cfg = _load_yaml(repo / "configs/recipes/hydra/maxent_listwise_code_mbpp.yaml")
    _assert_triplet(
        grpo_cfg=grpo_cfg,
        entropy_cfg=entropy_cfg,
        listwise_cfg=listwise_cfg,
        shared_recipe="configs/recipes/Qwen2.5-0.5B-Instruct/grpo/config_code_mbpp.yaml",
        grpo_output_dir="var/data/Qwen2.5-0.5B-Open-R1-GRPO-code-mbpp-v1",
        entropy_output_dir="var/data/Qwen2.5-0.5B-Open-R1-MaxEnt-Entropy-code-mbpp-v1",
        listwise_output_dir="var/data/Qwen2.5-0.5B-Open-R1-MaxEnt-Listwise-code-mbpp-v1",
    )
    _assert_triplet_only_differs_on_objective_specific_keys(
        "maxent",
        grpo_cfg=grpo_cfg,
        entropy_cfg=entropy_cfg,
        listwise_cfg=listwise_cfg,
        repo=repo,
    )


def test_seed_grpo_math_hydra_preset_is_explicit_and_aligned() -> None:
    repo = _repo_root()
    seed_cfg = _load_yaml(repo / "configs/recipes/hydra/seed_grpo_math.yaml")
    assert seed_cfg["command"] == "train-maxent"
    assert (
        seed_cfg["maxent"]["recipe"]
        == "configs/recipes/Qwen2.5-1.5B-Instruct/grpo/config_math.yaml"
    )
    training = seed_cfg["maxent"]["training"]
    assert training["objective"] == "grpo"
    assert training["grpo_loss_type"] == "dr_grpo"
    assert training["num_generations"] == 8
    assert training["scale_rewards"] is False
    assert training["seed_grpo_enabled"] is True
    assert training["seed_grpo_alpha"] == 0.0417
    assert training["seed_grpo_alpha_normalize_by_max_entropy"] is True
    assert training["seed_grpo_length_normalize_logprobs"] is True
    assert (
        training["output_dir"] == "var/data/Qwen2.5-1.5B-Open-R1-SEED-GRPO-math-v1"
    )
    merged = _merged_training_payload("maxent", seed_cfg, repo)
    assert merged["gradient_accumulation_steps"] == 40
    assert merged["reward_funcs"] == ["seed_paper_boxed_accuracy_math"]
    assert merged["eval_reward_funcs"] == ["seed_paper_boxed_accuracy_math"]
    assert merged["prompt_template"] == "no"
    assert merged["system_prompt"] is None
    assert merged["chat_template"] is None
    assert seed_cfg["hydra"]["run"]["dir"]
    assert seed_cfg["hydra"]["sweep"]["dir"]
    assert seed_cfg["hydra"]["sweep"]["subdir"] == "${hydra.job.num}"


def test_seed_grpo_math_stable_hydra_preset_is_explicit_and_aligned() -> None:
    repo = _repo_root()
    seed_cfg = _load_yaml(repo / "configs/recipes/hydra/seed_grpo_math_stable.yaml")
    assert seed_cfg["command"] == "train-maxent"
    assert (
        seed_cfg["maxent"]["recipe"]
        == "configs/recipes/Qwen2.5-1.5B-Instruct/grpo/config_math_stable.yaml"
    )
    training = seed_cfg["maxent"]["training"]
    assert training["objective"] == "grpo"
    assert training["grpo_loss_type"] == "dr_grpo"
    assert training["num_generations"] == 8
    assert training["scale_rewards"] is False
    assert training["beta"] == 0.0
    assert training["grpo_beta_controller_enabled"] is False
    assert training["maxent_beta_controller_enabled"] is False
    assert training["seed_grpo_enabled"] is True
    assert training["seed_grpo_alpha"] == 0.0417
    assert training["seed_grpo_alpha_normalize_by_max_entropy"] is True
    assert training["seed_grpo_length_normalize_logprobs"] is True
    assert (
        training["output_dir"]
        == "var/data/Qwen2.5-1.5B-Open-R1-SEED-GRPO-math-stable-v1"
    )
    assert seed_cfg["hydra"]["run"]["dir"]
    assert seed_cfg["hydra"]["sweep"]["dir"]
    assert seed_cfg["hydra"]["sweep"]["subdir"] == "${hydra.job.num}"


def test_listwise_triplet_presets_keep_whole_prompt_groups_per_microbatch() -> None:
    repo = _repo_root()
    preset_specs = [
        ("maxent", _load_yaml(repo / "configs/recipes/hydra/maxent_listwise_math.yaml")),
        (
            "maxent",
            _load_yaml(repo / "configs/recipes/hydra/maxent_listwise_code_mbpp.yaml"),
        ),
    ]
    for root_key, payload in preset_specs:
        merged = _merged_training_payload(root_key, payload, repo)
        num_generations = int(merged["num_generations"])
        train_batch = int(merged["per_device_train_batch_size"])
        eval_batch = int(merged["per_device_eval_batch_size"])
        assert train_batch % num_generations == 0
        assert eval_batch % num_generations == 0


def test_entropy_recipe_files_are_explicit_about_variant() -> None:
    repo = _repo_root()
    recipe_paths = [
        repo / "configs/recipes/Qwen2.5-0.5B-Instruct/maxent-grpo/config_math.yaml",
        repo / "configs/recipes/Qwen2.5-0.5B-Instruct/maxent-grpo/config_code_mbpp.yaml",
        repo / "configs/recipes/Qwen2.5-1.5B-Instruct/maxent-grpo/config_math.yaml",
        repo / "configs/recipes/Qwen2.5-1.5B-Instruct/maxent-grpo/config_math_stable.yaml",
    ]
    for path in recipe_paths:
        payload = _load_yaml(path)
        assert payload["objective"] == "maxent_entropy"
