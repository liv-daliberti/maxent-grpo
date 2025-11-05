#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 Liv d'Aliberti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import re
from concurrent.futures import Future

from transformers import AutoConfig

from huggingface_hub import (
    create_branch,
    create_repo,
    get_safetensors_metadata,
    list_repo_commits,
    list_repo_files,
    list_repo_refs,
    repo_exists,
    upload_folder,
)
from huggingface_hub.utils import HfHubHTTPError
from trl import GRPOConfig, SFTConfig


logger = logging.getLogger(__name__)


def push_to_hub_revision(
    training_args: SFTConfig | GRPOConfig, extra_ignore_patterns: list[str] | None = None
) -> Future:
    """Push a checkpoint directory to a branch on the Hub.

    :param training_args: Training config with Hub identifiers and output dir.
    :type training_args: SFTConfig | GRPOConfig
    :param extra_ignore_patterns: Additional filename patterns to ignore during upload.
    :type extra_ignore_patterns: list[str] | None
    :returns: Future that completes when the upload finishes.
    :rtype: concurrent.futures.Future
    """

    # Create a repo if it doesn't exist yet
    repo_url = create_repo(repo_id=training_args.hub_model_id, private=True, exist_ok=True)
    # Get initial commit to branch from
    initial_commit = list_repo_commits(training_args.hub_model_id)[-1]
    # Now create the branch we'll be pushing to
    create_branch(
        repo_id=training_args.hub_model_id,
        branch=training_args.hub_model_revision,
        revision=initial_commit.commit_id,
        exist_ok=True,
    )
    logger.info("Created target repo at %s", repo_url)
    logger.info(
        "Pushing to the Hub revision %s...", training_args.hub_model_revision
    )
    ignore_patterns = ["checkpoint-*", "*.pth"]
    if extra_ignore_patterns:
        ignore_patterns.extend(extra_ignore_patterns)
    future = upload_folder(
        repo_id=training_args.hub_model_id,
        folder_path=training_args.output_dir,
        revision=training_args.hub_model_revision,
        commit_message=f"Add {training_args.hub_model_revision} checkpoint",
        ignore_patterns=ignore_patterns,
        run_as_future=True,
    )
    logger.info(
        "Pushed to %s revision %s successfully!",
        repo_url,
        training_args.hub_model_revision,
    )

    return future


def check_hub_revision_exists(training_args: SFTConfig | GRPOConfig):
    """Validate whether a target Hub revision exists and is safe to write.

    :param training_args: Training config with Hub identifiers and flags.
    :type training_args: SFTConfig | GRPOConfig
    :returns: None
    :rtype: None
    :raises ValueError: If the revision exists and appears non‑empty without
        setting ``overwrite_hub_revision``.
    """
    if repo_exists(training_args.hub_model_id):
        if training_args.push_to_hub_revision is True:
            # First check if the revision exists
            revisions = [rev.name for rev in list_repo_refs(training_args.hub_model_id).branches]
            # If the revision exists, we next check it has a README file
            if training_args.hub_model_revision in revisions:
                repo_files = list_repo_files(
                    repo_id=training_args.hub_model_id,
                    revision=training_args.hub_model_revision,
                )
                if "README.md" in repo_files and training_args.overwrite_hub_revision is False:
                    raise ValueError(
                        f"Revision {training_args.hub_model_revision} already exists. "
                        "Use --overwrite_hub_revision to overwrite it."
                    )


def get_param_count_from_repo_id(repo_id: str) -> int:
    """Infer parameter count from Hub metadata or naming conventions.

    Attempts to read safetensors metadata; if unavailable, falls back to
    parsing strings like ``42m``, ``1.5b`` or products like ``8x7b``.

    :param repo_id: Hub repository ID.
    :type repo_id: str
    :returns: Best guess of total parameter count, or ``-1`` if unknown.
    :rtype: int
    """
    try:
        metadata = get_safetensors_metadata(repo_id)
        return list(metadata.parameter_count.values())[0]
    except (HfHubHTTPError, ValueError, KeyError):
        # Pattern to match products (like 8x7b) and single values (like 42m)
        pattern = r"((\d+(\.\d+)?)(x(\d+(\.\d+)?))?)([bm])"
        matches = re.findall(pattern, repo_id.lower())

        param_counts = []
        for _full_match, number1, _, _, number2, _, unit in matches:
            if number2:  # If there's a second number, it's a product
                number = float(number1) * float(number2)
            else:  # Otherwise, it's a single value
                number = float(number1)

            if unit == "b":
                number *= 1_000_000_000  # Convert to billion
            elif unit == "m":
                number *= 1_000_000  # Convert to million

            param_counts.append(number)

        if len(param_counts) > 0:
            # Return the largest number
            return int(max(param_counts))
        else:
            # Return -1 if no match found
            return -1


def get_gpu_count_for_vllm(
    model_name: str, revision: str = "main", num_gpus: int = 8
) -> int:
    """Choose a valid GPU count for vLLM tensor parallelism.

    vLLM requires that the number of attention heads and 64 are divisible by
    the tensor parallel size. This function decrements ``num_gpus`` until the
    constraints are satisfied.

    :param model_name: Model repository ID.
    :type model_name: str
    :param revision: Repo revision/branch.
    :type revision: str
    :param num_gpus: Starting number of GPUs available.
    :type num_gpus: int
    :returns: A compatible number of GPUs for vLLM.
    :rtype: int
    """
    config = AutoConfig.from_pretrained(model_name, revision=revision, trust_remote_code=True)
    # Get number of attention heads
    num_heads = config.num_attention_heads
    # Reduce num_gpus so that num_heads is divisible by num_gpus and 64 is divisible by num_gpus
    while num_heads % num_gpus != 0 or 64 % num_gpus != 0:
        logger.info(
            "Reducing num_gpus from %d to %d to make num_heads divisible by num_gpus",
            num_gpus,
            num_gpus - 1,
        )
        num_gpus -= 1
    return num_gpus
