from __future__ import annotations

from pathlib import Path

import yaml

_OBJECTIVE_SPECIFIC_TRAINING_KEYS = {
    "objective",
    "maxent_alpha",
    "maxent_length_normalize_ref",
    "maxent_policy_entropy",
    "maxent_policy_entropy_mode",
    "maxent_q_epsilon",
    "maxent_q_temperature",
    "maxent_reference_ema_enabled",
    "maxent_tau",
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
        explicit_num_generations=2,
    )
    _assert_triplet_only_differs_on_objective_specific_keys(
        "maxent",
        grpo_cfg=grpo_cfg,
        entropy_cfg=entropy_cfg,
        listwise_cfg=listwise_cfg,
        repo=repo,
    )


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
    ]
    for path in recipe_paths:
        payload = _load_yaml(path)
        assert payload["objective"] == "maxent_entropy"
