#!/usr/bin/env python
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

"""Helpers for working with the Hugging Face Hub.

This module provides:

- Upload utilities to push a training output directory to a dedicated branch
  (revision) with basic safety checks.
- Small metadata helpers such as parameter count inference from a repo ID
  (via naming conventions or safetensors metadata) and choosing a valid GPU
  count for vLLM tensor parallelism.

"""

from __future__ import annotations

import logging
import re
from concurrent.futures import Future
from typing import Any, List, Optional, TYPE_CHECKING

try:  # pragma: no cover - optional dependency or incomplete install
    from transformers import AutoConfig
except Exception:  # pragma: no cover - allow runtimes without full transformers
    class AutoConfig:  # type: ignore[no-redef]
        """Fallback stub when transformers is missing or incomplete."""

        @staticmethod
        def from_pretrained(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("transformers is not installed or lacks AutoConfig")

if TYPE_CHECKING:  # pragma: no cover - import types without runtime dependency
    from huggingface_hub import (
        create_branch,
        create_repo,
        get_safetensors_metadata,
        list_repo_commits,
        list_repo_files,
        list_repo_refs,
        repo_exists,
        upload_folder,
        CommitInfo,
    )
    from huggingface_hub.errors import HfHubHTTPError
else:
    try:  # pragma: no cover - optional dependency
        from huggingface_hub import (
            create_branch,
            create_repo,
            get_safetensors_metadata,
            list_repo_commits,
            list_repo_files,
            list_repo_refs,
            repo_exists,
            upload_folder,
            CommitInfo,
        )
        from huggingface_hub.errors import HfHubHTTPError
    except ModuleNotFoundError:  # pragma: no cover - provide safe fallbacks for tests

        def create_branch(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("huggingface_hub is not installed")

        def create_repo(*_args: Any, **_kwargs: Any) -> str:
            raise RuntimeError("huggingface_hub is not installed")

        def get_safetensors_metadata(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("huggingface_hub is not installed")

        def list_repo_commits(*_args: Any, **_kwargs: Any) -> List[Any]:
            return []

        def list_repo_files(*_args: Any, **_kwargs: Any) -> List[str]:
            return []

        class _EmptyBranches:
            branches: List[Any] = []

        def list_repo_refs(*_args: Any, **_kwargs: Any) -> _EmptyBranches:
            return _EmptyBranches()

        def repo_exists(*_args: Any, **_kwargs: Any) -> bool:
            return False

        def upload_folder(*_args: Any, **_kwargs: Any) -> Future["CommitInfo"]:
            raise RuntimeError("huggingface_hub is not installed")

        class CommitInfo:
            commit_id = ""

        class HfHubHTTPError(Exception):
            pass


logger = logging.getLogger(__name__)


if TYPE_CHECKING:  # only for type checking; avoids runtime dependency
    from maxent_grpo.config import GRPOConfig


def push_to_hub_revision(
    training_args: "GRPOConfig",
    extra_ignore_patterns: Optional[List[str]] = None,
    *,
    include_checkpoints: bool = False,
) -> Future[CommitInfo]:
    """Push a checkpoint directory to a branch on the Hub.

    The helper will create the repository if missing, ensure the target branch
    exists (forked from the latest commit when possible), and upload the
    ``output_dir`` contents while ignoring common checkpoint artefacts. Uploads
    are executed asynchronously via ``run_as_future=True`` to avoid blocking
    training scripts.

    :param training_args: Training config with Hub identifiers (``hub_model_id``
        and ``hub_model_revision``) and the local ``output_dir`` to upload.
    :type training_args: GRPOConfig
    :param include_checkpoints: When True, do not ignore checkpoint-* folders.
    :type include_checkpoints: bool
    :param extra_ignore_patterns: Additional filename patterns to ignore during
        upload; appended to the default ``checkpoint-*`` and ``*.pth`` filters.
    :type extra_ignore_patterns: list[str] | None
    :returns: Future that completes when the upload finishes, resolving to the
        Hub commit metadata.
    :rtype: concurrent.futures.Future[huggingface_hub.CommitInfo]
    :raises ValueError: If ``hub_model_id`` is not set in ``training_args``.
    """
    if not training_args.hub_model_id:
        raise ValueError("hub_model_id must be set in training_args")

    revision = training_args.hub_model_revision or "main"
    output_dir = training_args.output_dir
    if not output_dir:
        raise ValueError("output_dir must be set in training_args")

    # Create a repo if it doesn't exist yet
    repo_url: str = create_repo(
        repo_id=training_args.hub_model_id, private=True, exist_ok=True
    )
    # Get initial commit to branch from (repo may be empty on first push)
    try:
        commits = list_repo_commits(training_args.hub_model_id)
        initial_commit = commits[-1] if commits else None
        base_rev: Optional[str] = (
            getattr(initial_commit, "commit_id", None) if initial_commit is not None else None
        )
    except HfHubHTTPError:
        # Fall back to default branch tip
        base_rev = None
    # Now create the branch we'll be pushing to
    create_branch(
        repo_id=training_args.hub_model_id,
        branch=revision,
        revision=base_rev,
        exist_ok=True,
    )
    logger.info("Created target repo at %s", repo_url)
    logger.info("Pushing to the Hub revision %s...", revision)
    ignore_patterns: List[str] = []
    if not include_checkpoints:
        ignore_patterns.extend(["checkpoint-*", "*.pth"])
    if extra_ignore_patterns:
        ignore_patterns.extend(extra_ignore_patterns)
    future: Future[CommitInfo] = upload_folder(
        repo_id=training_args.hub_model_id,
        folder_path=output_dir,
        revision=revision,
        commit_message=f"Add {revision} checkpoint",
        ignore_patterns=ignore_patterns,
        run_as_future=True,
    )
    logger.info(
        "Pushed to %s revision %s successfully!",
        repo_url,
        revision,
    )

    return future


def ensure_hf_repo_ready(training_args: "GRPOConfig") -> None:
    """Verify Hub credentials and provision the target repo/branch upfront.

    The helper is a best-effort preflight. When Hub access is not configured
    (or push is disabled), it returns early. Errors in network calls are
    surfaced as ``RuntimeError`` to avoid silent misconfiguration.

    :param training_args: Training config with Hub identifiers and push flags.
    :type training_args: GRPOConfig
    :returns: ``None``. The function exits early when Hub pushes are disabled.
    :rtype: None
    :raises RuntimeError: If the Hub preflight fails due to network or auth
        errors.
    """

    push_requested = bool(
        getattr(training_args, "push_to_hub", False)
        or getattr(training_args, "push_to_hub_revision", False)
    )
    if not push_requested:
        return
    repo_id = getattr(training_args, "hub_model_id", None)
    if not repo_id:
        logger.warning("push_to_hub requested but hub_model_id is unset; skipping preflight")
        return
    revision = getattr(training_args, "hub_model_revision", None) or "main"
    try:
        repo_url = create_repo(repo_id=repo_id, private=True, exist_ok=True)
        base_rev: Optional[str]
        try:
            base_rev = list_repo_commits(repo_id)[-1].commit_id
        except (IndexError, HfHubHTTPError):
            base_rev = None
        create_branch(repo_id=repo_id, branch=revision, revision=base_rev, exist_ok=True)
        logger.info("Verified Hugging Face repo %s (revision %s) is ready", repo_url, revision)
    except RuntimeError as exc:
        logger.warning("Skipping Hub preflight: %s", exc)
        return
    except (HfHubHTTPError, OSError, ValueError) as exc:  # pragma: no cover - network dependent
        raise RuntimeError(
            "Failed to preflight Hugging Face Hub access; check credentials/network"
        ) from exc


def check_hub_revision_exists(training_args: "GRPOConfig") -> None:
    """Validate whether a target Hub revision exists and is safe to write.

    The check avoids clobbering populated branches unless explicitly permitted
    via ``overwrite_hub_revision``. A README in the branch is treated as a
    signal that the branch has content.

    :param training_args: Training config with Hub identifiers and safety flags
        such as ``push_to_hub_revision`` and ``overwrite_hub_revision``.
    :type training_args: GRPOConfig
    :returns: ``None``. Raises if the target revision appears non-empty and
        overwriting is disallowed.
    :rtype: None
    :raises ValueError: If the revision exists and appears non-empty without
        setting ``overwrite_hub_revision``.
    """
    repo_id = getattr(training_args, "hub_model_id", None)
    if not repo_id:
        logger.warning(
            "push_to_hub_revision requested but hub_model_id is unset; skipping revision check"
        )
        return
    if repo_exists(repo_id):
        if training_args.push_to_hub_revision is True:
            # First check if the revision exists
            revisions = [rev.name for rev in list_repo_refs(repo_id).branches]
            # If the revision exists, we next check it has a README file
            if training_args.hub_model_revision in revisions:
                repo_files = list_repo_files(
                    repo_id=repo_id,
                    revision=training_args.hub_model_revision,
                )
                if (
                    "README.md" in repo_files
                    and training_args.overwrite_hub_revision is False
                ):
                    raise ValueError(
                        f"Revision {training_args.hub_model_revision} already exists. "
                        "Use --overwrite_hub_revision to overwrite it."
                    )


def get_param_count_from_repo_id(repo_id: Optional[str]) -> int:
    """Infer parameter count from naming conventions or Hub metadata.

    Prefers parsing strings like ``42m``, ``1.5b`` or products like ``8x7b``
    from the repo ID. Falls back to safetensors metadata when no pattern is
    found.

    :param repo_id: Hub repository ID.
    :type repo_id: str | None
    :returns: Best guess of total parameter count, or ``-1`` if unknown after
        attempting both pattern extraction and safetensors metadata lookup, or
        if ``repo_id`` is missing.
    :rtype: int
    """
    if not repo_id:
        return -1
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
        # Return the largest number from the string pattern
        return int(max(param_counts))

    # Fallback: try to read from Hub metadata
    try:  # pragma: no cover - behavior depends on environment
        metadata = get_safetensors_metadata(repo_id)
        return int(list(metadata.parameter_count.values())[0])
    except (HfHubHTTPError, ValueError, KeyError, TypeError):
        return -1


def get_gpu_count_for_vllm(
    model_name: Optional[str],
    revision: Optional[str] = "main",
    num_gpus: int = 8,
) -> int:
    """Choose a valid GPU count for vLLM tensor parallelism.

    vLLM requires that the number of attention heads and 64 are divisible by
    the tensor parallel size. This function decrements ``num_gpus`` until the
    constraints are satisfied.

    :param model_name: Model repository ID used to fetch the ``AutoConfig``.
    :type model_name: str | None
    :param revision: Repo revision/branch to inspect.
    :type revision: str | None
    :param num_gpus: Starting number of GPUs available; decremented until the
        constraints are satisfied.
    :type num_gpus: int
    :returns: A compatible number of GPUs for vLLM tensor parallelism.
    :rtype: int
    """
    if num_gpus <= 0:
        logger.warning("Invalid num_gpus=%d supplied; defaulting to 1", num_gpus)
        num_gpus = 1
    if not model_name:
        logger.warning("Missing model_name; using num_gpus=%d", num_gpus)
        return num_gpus
    safe_revision = revision or "main"
    try:
        config = AutoConfig.from_pretrained(
            model_name, revision=safe_revision, trust_remote_code=True
        )
    except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - network dependent
        logger.warning(
            "Unable to load config for %s (revision %s): %s; using num_gpus=%d",
            model_name,
            safe_revision,
            exc,
            num_gpus,
        )
        return num_gpus
    # Get number of attention heads
    num_heads = getattr(config, "num_attention_heads", None)
    if not isinstance(num_heads, int) or num_heads <= 0:
        logger.warning(
            "Unable to infer num_attention_heads for %s; using num_gpus=%d",
            model_name,
            num_gpus,
        )
        return max(1, num_gpus)
    # Reduce num_gpus so that num_heads is divisible by num_gpus and 64 is divisible by num_gpus
    while num_heads % num_gpus != 0 or 64 % num_gpus != 0:
        logger.info(
            "Reducing num_gpus from %d to %d to make num_heads divisible by num_gpus",
            num_gpus,
            num_gpus - 1,
        )
        num_gpus -= 1
    return num_gpus
