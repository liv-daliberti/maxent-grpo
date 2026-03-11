"""Unit tests for command-default objective routing in hydra_cli."""

from __future__ import annotations

from types import SimpleNamespace

from maxent_grpo.cli.hydra_cli import (
    BaselineCommand,
    MaxentCommand,
    _apply_command_objective_default,
)


def test_baseline_command_defaults_to_grpo_without_recipe() -> None:
    training_args = SimpleNamespace(objective="maxent_entropy", train_grpo_objective=False)
    _apply_command_objective_default(
        training_args,
        command="train-baseline",
        command_cfg=BaselineCommand(),
        recipe_path=None,
    )
    assert training_args.objective == "grpo"
    assert training_args.train_grpo_objective is True


def test_maxent_command_defaults_to_maxent_without_recipe() -> None:
    training_args = SimpleNamespace(objective="grpo", train_grpo_objective=True)
    _apply_command_objective_default(
        training_args,
        command="train-maxent",
        command_cfg=MaxentCommand(),
        recipe_path=None,
    )
    assert training_args.objective == "maxent_entropy"
    assert training_args.train_grpo_objective is False


def test_explicit_override_is_preserved() -> None:
    training_args = SimpleNamespace(objective="maxent_listwise", train_grpo_objective=False)
    _apply_command_objective_default(
        training_args,
        command="train-baseline",
        command_cfg=BaselineCommand(training={"objective": "maxent_listwise"}),
        recipe_path=None,
    )
    assert training_args.objective == "maxent_listwise"
    assert training_args.train_grpo_objective is False
