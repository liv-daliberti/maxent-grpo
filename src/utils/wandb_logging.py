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

import os


def init_wandb_training(training_args):
    """Initialize Weights & Biases environment variables for a run.

    Exposes entity/project/group from ``training_args`` to the W&B backend via
    ``WANDB_*`` environment variables.

    :param training_args: Training configuration providing ``wandb_entity``,
        ``wandb_project``, and ``wandb_run_group`` fields.
    :type training_args: Any
    :returns: None
    :rtype: None
    """
    if training_args.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = training_args.wandb_entity
    if training_args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project
    if training_args.wandb_run_group is not None:
        os.environ["WANDB_RUN_GROUP"] = training_args.wandb_run_group
