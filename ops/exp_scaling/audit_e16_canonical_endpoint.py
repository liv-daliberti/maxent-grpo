#!/usr/bin/env python3
"""Exact endpoint audit for E16's finite canonical action policies.

This generalizes the E14 27-leaf graph-coloring audit to position-specific
finite supports.  It enumerates every complete action, checks the entropy
chain rule independently, and teacher-forces every leaf.  Countdown support
is constructed only from its fixed ``(6, 3, 6)`` grammar; targets are read
only after construction, when the frozen grader classifies leaves.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = ROOT / "src"
AUDIT_SOURCE_ROOT = Path(
    os.environ.get("OAT_ZERO_E16_AUDIT_SOURCE_ROOT", str(DEFAULT_SRC))
).resolve()
if not AUDIT_SOURCE_ROOT.is_dir():
    raise RuntimeError(f"E16 audit source root does not exist: {AUDIT_SOURCE_ROOT}")
for import_root in (ROOT, AUDIT_SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from oat_drgrpo.canonical_actions import (  # noqa: E402
    canonical_action_strings_by_position,
    decode_canonical_action_response,
    enumerate_countdown_action_codes,
    resolve_canonical_action_space,
)
from oat_drgrpo.math_grader import (  # noqa: E402
    _canonical_countdown_expression_key,
    _modebench_answer_key,
    boxed_reward_fn,
)
from oat_drgrpo.templates import (  # noqa: E402
    apply_qwen_countdown_digits_template,
    apply_qwen_graph_digits_template,
)
from ops.exp_scaling.audit_e14_checkpoint import (  # noqa: E402
    DEFAULT_TOKENIZER_PATH,
    EXPECTED_TOKENIZER_FILES_HASH,
    EXPECTED_TOKENIZER_REVISION,
    EXPECTED_TOKENIZER_VOCAB_HASH,
    LOG_PROBABILITY_TOLERANCE,
    _fixed_size_same_length_batches,
    _mean,
    _right_padded_batch,
    _source_tree_hash,
    _weight_identity,
    tokenizer_files_hash,
    tokenizer_vocab_hash,
    validate_endpoint_checkpoint_tag,
)
from ops.exp_scaling.verify_e14_dataset import (  # noqa: E402
    EXPECTED_COMBINED_CONTENT_HASH as EXPECTED_GRAPH_CONTENT_HASH,
    _content_hash,
    validate_e14_rows,
)
from ops.exp_scaling.verify_e16_canonical_datasets import (  # noqa: E402
    EXPECTED_COUNTDOWN_COMBINED_CONTENT_HASH,
    validate_e16_countdown_rows,
)


TASKS = ("graph_coloring", "countdown")
ARMS = ("maxent", "maxent_control", "maxent_dual")
SMOKE_SEED = 9006
SMOKE_UPDATES = 32
SMOKE_PREFIXES = {
    "graph_coloring": "gce16_canonical_maxent_joint_smoke_v3",
    "countdown": "cde16_canonical_maxent_joint_smoke_v3",
}
PROBABILITY_TOLERANCE = 1e-5
ENTROPY_TOLERANCE = 1e-5
EXPECTED_ROWS = {
    "graph_coloring": (192, 96),
    "countdown": (384, 128),
}
EXPECTED_COUNTDOWN_CONTENT_HASH = EXPECTED_COUNTDOWN_COMBINED_CONTENT_HASH
EXPECTED_ACTION_TOKEN_IDS_BY_POSITION = {
    "graph_coloring": ((16, 17, 18),) * 3,
    "countdown": ((16, 17, 18, 19, 20, 21), (16, 17, 18), (16, 17, 18, 19, 20, 21)),
}


@dataclass(frozen=True)
class PolicyGeometry:
    """Task-independent description of a finite positional action tree."""

    task: str
    action_strings_by_position: tuple[tuple[str, ...], ...]

    @property
    def support_sizes(self) -> tuple[int, ...]:
        return tuple(len(support) for support in self.action_strings_by_position)

    @property
    def horizon(self) -> int:
        return len(self.action_strings_by_position)

    @property
    def leaf_count(self) -> int:
        return math.prod(self.support_sizes)

    @property
    def max_entropy(self) -> float:
        return math.log(self.leaf_count)

    @property
    def action_indices(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            itertools.product(*(range(size) for size in self.support_sizes))
        )

    @property
    def prefixes(self) -> tuple[tuple[int, ...], ...]:
        prefixes: list[tuple[int, ...]] = []
        for depth in range(self.horizon):
            prefixes.extend(
                itertools.product(
                    *(range(size) for size in self.support_sizes[:depth])
                )
            )
        return tuple(prefixes)

    @property
    def action_codes(self) -> tuple[str, ...]:
        return tuple(
            "".join(
                self.action_strings_by_position[position][index]
                for position, index in enumerate(action)
            )
            for action in self.action_indices
        )


@dataclass(frozen=True)
class ExactPolicyResult:
    """Exact complete-action quantities for one prompt."""

    geometry: PolicyGeometry
    leaf_log_probabilities: torch.Tensor
    leaf_probabilities: torch.Tensor
    leaf_entropy: float
    conditional_entropy: float
    probability_sum: float


@dataclass(frozen=True)
class CodecRecord:
    """One code and its deterministic decoded semantic identity."""

    action_indices: tuple[int, ...]
    code: str
    decoded_response: str
    semantic_key: str


def policy_geometry(task: str) -> PolicyGeometry:
    """Return a frozen graph or Countdown finite-tree geometry."""

    if task not in TASKS:
        raise ValueError(f"unsupported E16 canonical task: {task!r}")
    return PolicyGeometry(
        task=task,
        action_strings_by_position=canonical_action_strings_by_position(task),
    )


def public_decoder_reference(task: str, reference: str | Mapping[str, Any]) -> dict[str, Any]:
    """Strip labels before any codec or semantic-key operation.

    In particular, Countdown's target and any solution-derived metadata are
    deliberately absent from the returned mapping.
    """

    try:
        spec = json.loads(reference) if isinstance(reference, str) else dict(reference)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("canonical reference is malformed") from exc
    if spec.get("verifier") != task:
        raise ValueError(f"canonical reference is not a {task} row")
    if task == "countdown":
        try:
            numbers = [int(value) for value in spec["numbers"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("canonical Countdown numbers are malformed") from exc
        if len(numbers) != 3 or len(set(numbers)) != 3:
            raise ValueError(
                "canonical Countdown requires exactly three distinct operands"
            )
        return {"verifier": "countdown", "numbers": numbers}
    try:
        n = int(spec["n"])
        partial_colors = list(spec["partial_colors"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("canonical graph reference is malformed") from exc
    if len(partial_colors) != n or sum(value is None for value in partial_colors) != 3:
        raise ValueError("canonical graph reference must contain three hidden colors")
    return {
        "verifier": "graph_coloring",
        "n": n,
        "partial_colors": partial_colors,
    }


def _semantic_key(task: str, response: str, public_reference: Mapping[str, Any]) -> str:
    if task == "countdown":
        key = _canonical_countdown_expression_key(response, dict(public_reference))
    else:
        key = _modebench_answer_key(response, dict(public_reference))
    if key is None:
        raise ValueError(f"canonical {task} decoder emitted an unkeyable response")
    return str(key)


def validate_code_decoder_bijection(
    task: str, reference: str | Mapping[str, Any]
) -> tuple[CodecRecord, ...]:
    """Exhaustively validate code -> response -> semantic-key bijection."""

    geometry = policy_geometry(task)
    public_reference = public_decoder_reference(task, reference)
    if task == "countdown" and geometry.action_codes != enumerate_countdown_action_codes():
        raise RuntimeError("generic and canonical Countdown code enumerations disagree")
    records: list[CodecRecord] = []
    for action_indices, code in zip(
        geometry.action_indices, geometry.action_codes, strict=True
    ):
        decoded = decode_canonical_action_response(task, code, public_reference)
        repeated = decode_canonical_action_response(task, code, public_reference)
        if decoded != repeated:
            raise RuntimeError(f"canonical decoder is nondeterministic for code {code}")
        records.append(
            CodecRecord(
                action_indices=action_indices,
                code=code,
                decoded_response=decoded,
                semantic_key=_semantic_key(task, decoded, public_reference),
            )
        )
    fields = {
        "codes": [record.code for record in records],
        "decoded responses": [record.decoded_response for record in records],
        "semantic keys": [record.semantic_key for record in records],
    }
    for name, values in fields.items():
        if len(values) != geometry.leaf_count or len(set(values)) != geometry.leaf_count:
            raise ValueError(
                f"canonical {task} codec is not bijective over {name}: "
                f"unique={len(set(values))} expected={geometry.leaf_count}"
            )
    return tuple(records)


def _normalized_node_log_probabilities(
    node_log_probabilities: Mapping[
        tuple[int, ...], torch.Tensor | Sequence[float]
    ],
    geometry: PolicyGeometry,
) -> dict[tuple[int, ...], torch.Tensor]:
    expected = set(geometry.prefixes)
    observed = set(node_log_probabilities)
    if observed != expected:
        raise ValueError(
            "canonical prefix tree mismatch: "
            f"missing={sorted(expected - observed)} "
            f"extra={sorted(observed - expected)}"
        )
    normalized: dict[tuple[int, ...], torch.Tensor] = {}
    for prefix in geometry.prefixes:
        values = torch.as_tensor(
            node_log_probabilities[prefix], dtype=torch.float64
        ).detach().cpu()
        expected_size = geometry.support_sizes[len(prefix)]
        if values.shape != (expected_size,):
            raise ValueError(
                f"prefix {prefix} must have {expected_size} log probabilities"
            )
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError(f"prefix {prefix} has nonfinite probabilities")
        log_normalizer = float(torch.logsumexp(values, dim=0))
        if abs(log_normalizer) > PROBABILITY_TOLERANCE:
            raise ValueError(
                f"prefix {prefix} log probabilities are not normalized: "
                f"logsumexp={log_normalizer}"
            )
        normalized[prefix] = values
    return normalized


def enumerate_exact_policy(
    node_log_probabilities: Mapping[
        tuple[int, ...], torch.Tensor | Sequence[float]
    ],
    *,
    task: str,
) -> ExactPolicyResult:
    """Enumerate all leaves and verify both entropy identities exactly."""

    geometry = policy_geometry(task)
    nodes = _normalized_node_log_probabilities(node_log_probabilities, geometry)
    leaf_log_probabilities = torch.stack(
        [
            sum(
                nodes[action[:depth]][action[depth]]
                for depth in range(geometry.horizon)
            )
            for action in geometry.action_indices
        ]
    )
    leaf_probabilities = leaf_log_probabilities.exp()
    probability_sum = float(leaf_probabilities.sum())
    if not math.isfinite(probability_sum):
        raise FloatingPointError("enumerated canonical probability is nonfinite")
    if abs(probability_sum - 1.0) > PROBABILITY_TOLERANCE:
        raise ValueError(
            "enumerated canonical probabilities do not sum to one: "
            f"sum={probability_sum}"
        )
    leaf_entropy = float(
        -(leaf_probabilities * leaf_log_probabilities).sum()
    )
    conditional_entropy = 0.0
    for prefix in geometry.prefixes:
        prefix_log_probability = sum(
            nodes[prefix[:depth]][prefix[depth]]
            for depth in range(len(prefix))
        )
        prefix_probability = float(torch.exp(torch.as_tensor(prefix_log_probability)))
        local_logs = nodes[prefix]
        local_entropy = float(-(local_logs.exp() * local_logs).sum())
        conditional_entropy += prefix_probability * local_entropy
    if not math.isfinite(leaf_entropy) or not math.isfinite(conditional_entropy):
        raise FloatingPointError("canonical action entropy is nonfinite")
    if not (
        -ENTROPY_TOLERANCE
        <= leaf_entropy
        <= geometry.max_entropy + ENTROPY_TOLERANCE
    ):
        raise ValueError("canonical action entropy is outside its support bounds")
    if abs(leaf_entropy - conditional_entropy) > ENTROPY_TOLERANCE:
        raise ValueError(
            "leaf entropy disagrees with the conditional-entropy identity: "
            f"leaf={leaf_entropy} conditional={conditional_entropy}"
        )
    return ExactPolicyResult(
        geometry=geometry,
        leaf_log_probabilities=leaf_log_probabilities,
        leaf_probabilities=leaf_probabilities,
        leaf_entropy=leaf_entropy,
        conditional_entropy=conditional_entropy,
        probability_sum=probability_sum,
    )


def valid_policy_metrics(
    exact: ExactPolicyResult,
    grader_rewards: Sequence[int | float | bool],
    semantic_keys: Sequence[str],
) -> dict[str, Any]:
    """Compute exact valid probability and semantic-mode entropy.

    Leaf mass is aggregated by semantic key before ``H_valid`` is computed,
    so the metric remains correct even if a future codec deliberately admits
    representational aliases.  E16 separately requires a bijective codec.
    """

    leaf_count = exact.geometry.leaf_count
    rewards = torch.as_tensor(grader_rewards, dtype=torch.float64)
    if rewards.shape != (leaf_count,) or not bool(torch.isfinite(rewards).all()):
        raise ValueError(f"grader_rewards must be {leaf_count} finite binary values")
    if not bool(((rewards == 0.0) | (rewards == 1.0)).all()):
        raise ValueError("grader_rewards must be binary")
    if len(semantic_keys) != leaf_count or any(not str(key) for key in semantic_keys):
        raise ValueError(f"semantic_keys must contain {leaf_count} nonempty strings")

    reward_by_key: dict[str, int] = {}
    probability_by_key: dict[str, float] = defaultdict(float)
    valid_mask = rewards.to(torch.bool)
    for index, key_value in enumerate(semantic_keys):
        key = str(key_value)
        reward = int(rewards[index])
        if key in reward_by_key and reward_by_key[key] != reward:
            raise ValueError(f"semantic key {key!r} has conflicting grader rewards")
        reward_by_key[key] = reward
        probability_by_key[key] += float(exact.leaf_probabilities[index])
    valid_keys = sorted(key for key, reward in reward_by_key.items() if reward == 1)
    if not valid_keys:
        raise ValueError("the frozen data contract requires at least one valid action")

    p_valid = math.fsum(probability_by_key[key] for key in valid_keys)
    if not math.isfinite(p_valid) or p_valid <= 0.0 or p_valid > 1.0 + PROBABILITY_TOLERANCE:
        raise FloatingPointError(f"invalid exact valid probability: {p_valid}")
    q_semantic = {
        key: probability_by_key[key] / p_valid for key in valid_keys
    }
    q_sum = math.fsum(q_semantic.values())
    if abs(q_sum - 1.0) > PROBABILITY_TOLERANCE:
        raise ValueError(f"valid semantic policy is not normalized: {q_sum}")
    h_valid = -math.fsum(
        probability * math.log(probability)
        for probability in q_semantic.values()
        if probability > 0.0
    )
    n_eff_valid = math.exp(h_valid)

    valid_leaf_logs = exact.leaf_log_probabilities[valid_mask]
    log_p_valid = float(torch.logsumexp(valid_leaf_logs, dim=0))
    q_action_logs = valid_leaf_logs - log_p_valid
    q_action = q_action_logs.exp()
    h_valid_action = float(-(q_action * q_action_logs).sum())
    n_eff_valid_action = math.exp(h_valid_action)
    if abs(math.exp(log_p_valid) - p_valid) > PROBABILITY_TOLERANCE:
        raise ValueError("action and semantic valid probabilities disagree")
    if not all(
        math.isfinite(value)
        for value in (h_valid, n_eff_valid, h_valid_action, n_eff_valid_action)
    ):
        raise FloatingPointError("valid-policy entropy is nonfinite")
    if h_valid > math.log(len(valid_keys)) + ENTROPY_TOLERANCE:
        raise ValueError("semantic valid entropy is outside its support bounds")

    q_plus = torch.zeros(leaf_count, dtype=torch.float64)
    q_plus[valid_mask] = q_action
    return {
        "valid_action_count": int(valid_mask.sum()),
        "valid_semantic_key_count": len(valid_keys),
        "log_p_valid": log_p_valid,
        "p_valid": p_valid,
        "h_valid": h_valid,
        "n_eff_valid": n_eff_valid,
        "h_valid_action": h_valid_action,
        "n_eff_valid_action": n_eff_valid_action,
        "q_plus": [float(value) for value in q_plus],
        "valid_semantic_q_plus": q_semantic,
        "semantic_probability_mass": dict(sorted(probability_by_key.items())),
    }


def compare_tree_and_teacher_forced(
    node_log_probabilities: Mapping[
        tuple[int, ...], torch.Tensor | Sequence[float]
    ],
    teacher_forced_selected_log_probabilities: torch.Tensor
    | Sequence[Sequence[float]],
    *,
    task: str,
) -> dict[str, float]:
    """Cross-check leaf teacher forcing against the restricted prefix tree."""

    geometry = policy_geometry(task)
    nodes = _normalized_node_log_probabilities(node_log_probabilities, geometry)
    expected = torch.stack(
        [
            torch.stack(
                [
                    nodes[action[:depth]][action[depth]]
                    for depth in range(geometry.horizon)
                ]
            )
            for action in geometry.action_indices
        ]
    )
    observed = torch.as_tensor(
        teacher_forced_selected_log_probabilities, dtype=torch.float64
    ).detach().cpu()
    expected_shape = (geometry.leaf_count, geometry.horizon)
    if observed.shape != expected_shape:
        raise ValueError(
            "teacher-forced selected log probabilities must have shape "
            f"{expected_shape}"
        )
    if not bool(torch.isfinite(observed).all()):
        raise FloatingPointError("teacher-forced probabilities are nonfinite")
    per_token_max = float((expected - observed).abs().max())
    sequence_max = float(
        (expected.sum(dim=1) - observed.sum(dim=1)).abs().max()
    )
    if (
        per_token_max > LOG_PROBABILITY_TOLERANCE
        or sequence_max > LOG_PROBABILITY_TOLERANCE
    ):
        raise ValueError(
            "teacher-forced log probabilities disagree with the prefix tree: "
            f"per_token_max={per_token_max} sequence_max={sequence_max}"
        )
    return {
        "per_token_max_abs_error": per_token_max,
        "sequence_max_abs_error": sequence_max,
    }


@torch.inference_mode()
def infer_prefix_tree_log_probabilities(
    model: Any,
    prompt_token_ids: Sequence[Sequence[int]],
    *,
    task: str,
    action_token_ids_by_position: Sequence[Sequence[int]],
    pad_token_id: int,
    device: torch.device,
    batch_size: int,
) -> list[dict[tuple[int, ...], torch.Tensor]]:
    """Evaluate every finite-tree prefix with fixed three-action forwards."""

    geometry = policy_geometry(task)
    supports = tuple(
        tuple(int(token_id) for token_id in support)
        for support in action_token_ids_by_position
    )
    if tuple(map(len, supports)) != geometry.support_sizes:
        raise ValueError("token supports do not match canonical task geometry")
    records: list[tuple[int, tuple[int, ...], list[int]]] = []
    for prompt_index, prompt in enumerate(prompt_token_ids):
        if not prompt:
            raise ValueError(f"prompt {prompt_index} tokenized to an empty sequence")
        for prefix in geometry.prefixes:
            observed = [supports[pos][index] for pos, index in enumerate(prefix)]
            placeholders = [supports[pos][0] for pos in range(len(prefix), geometry.horizon)]
            records.append(
                (prompt_index, prefix, list(prompt) + observed + placeholders)
            )
    output: list[dict[tuple[int, ...], torch.Tensor]] = [
        {} for _ in prompt_token_ids
    ]
    for batch, real_count in _fixed_size_same_length_batches(
        records, sequence_index=2, batch_size=batch_size
    ):
        input_ids, attention_mask, _ = _right_padded_batch(
            [record[2] for record in batch],
            pad_token_id=pad_token_id,
            device=device,
        )
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=geometry.horizon + 1,
        ).logits
        if logits.ndim != 3 or logits.shape[1] != geometry.horizon + 1:
            raise RuntimeError("model did not honor canonical logits_to_keep")
        for row, (prompt_index, prefix, _) in enumerate(batch[:real_count]):
            depth = len(prefix)
            allowed = torch.tensor(supports[depth], dtype=torch.long, device=device)
            restricted = logits[row, depth].index_select(-1, allowed)
            logs = torch.log_softmax(restricted.float(), dim=-1)
            if not bool(torch.isfinite(logs).all()):
                raise FloatingPointError("model produced nonfinite prefix logits")
            output[prompt_index][prefix] = logs.double().cpu()
    return output


@torch.inference_mode()
def infer_teacher_forced_log_probabilities(
    model: Any,
    prompt_token_ids: Sequence[Sequence[int]],
    *,
    task: str,
    action_token_ids_by_position: Sequence[Sequence[int]],
    pad_token_id: int,
    device: torch.device,
    batch_size: int,
) -> list[torch.Tensor]:
    """Independently teacher-force every complete canonical action."""

    geometry = policy_geometry(task)
    supports = tuple(
        tuple(int(token_id) for token_id in support)
        for support in action_token_ids_by_position
    )
    if tuple(map(len, supports)) != geometry.support_sizes:
        raise ValueError("token supports do not match canonical task geometry")
    records: list[tuple[int, int, list[int]]] = []
    for prompt_index, prompt in enumerate(prompt_token_ids):
        if not prompt:
            raise ValueError(f"prompt {prompt_index} tokenized to an empty sequence")
        for action_index, action in enumerate(geometry.action_indices):
            action_ids = [supports[pos][index] for pos, index in enumerate(action)]
            records.append((prompt_index, action_index, list(prompt) + action_ids))
    output = [
        torch.empty((geometry.leaf_count, geometry.horizon), dtype=torch.float64)
        for _ in prompt_token_ids
    ]
    for batch, real_count in _fixed_size_same_length_batches(
        records, sequence_index=2, batch_size=batch_size
    ):
        input_ids, attention_mask, _ = _right_padded_batch(
            [record[2] for record in batch],
            pad_token_id=pad_token_id,
            device=device,
        )
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=geometry.horizon + 1,
        ).logits
        if logits.ndim != 3 or logits.shape[1] != geometry.horizon + 1:
            raise RuntimeError("model did not honor canonical logits_to_keep")
        for row, (prompt_index, action_index, _) in enumerate(batch[:real_count]):
            selected: list[torch.Tensor] = []
            action = geometry.action_indices[action_index]
            for position in range(geometry.horizon):
                allowed = torch.tensor(
                    supports[position], dtype=torch.long, device=device
                )
                restricted = logits[row, position].index_select(-1, allowed)
                logs = torch.log_softmax(restricted.float(), dim=-1)
                selected.append(logs[action[position]])
            selected_logs = torch.stack(selected)
            if not bool(torch.isfinite(selected_logs).all()):
                raise FloatingPointError("model produced nonfinite leaf logits")
            output[prompt_index][action_index] = selected_logs.double().cpu()
    return output


def _load_frozen_rows(
    task: str, data_root: Path
) -> tuple[list[dict[str, Any]], str]:
    from datasets import load_from_disk

    train_dataset = load_from_disk(str(data_root / "train"))["train"]
    eval_dataset = load_from_disk(str(data_root / "eval"))["multi_answer"]
    if task == "graph_coloring":
        train_rows = validate_e14_rows(list(train_dataset), split_tag="train")
        eval_rows = validate_e14_rows(
            list(eval_dataset), split_tag="multi_answer"
        )
        expected_hash = EXPECTED_GRAPH_CONTENT_HASH
    else:
        train_rows = validate_e16_countdown_rows(
            list(train_dataset), split_tag="train"
        )
        eval_rows = validate_e16_countdown_rows(
            list(eval_dataset), split_tag="eval"
        )
        expected_hash = EXPECTED_COUNTDOWN_CONTENT_HASH
    expected_train, expected_eval = EXPECTED_ROWS[task]
    if len(train_rows) != expected_train or len(eval_rows) != expected_eval:
        raise ValueError(
            f"E16 {task} requires frozen {expected_train}/{expected_eval} pools; "
            f"got {len(train_rows)}/{len(eval_rows)}"
        )
    content_hash = _content_hash({"eval": eval_rows, "train": train_rows})
    if content_hash != expected_hash:
        raise ValueError(
            f"E16 {task} data hash mismatch: expected={expected_hash} "
            f"observed={content_hash}"
        )
    return eval_rows, content_hash


def _format_prompt(task: str, problem: str) -> str:
    if task == "graph_coloring":
        return apply_qwen_graph_digits_template(problem)
    return apply_qwen_countdown_digits_template(problem)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_smoke_cell_identity(
    *,
    identity_path: Path,
    expected_identity_sha256: str,
    task: str,
    arm: str,
    seed: int,
    run_stamp: str,
    job_id: str,
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and return the exact prospective Stage-S cell identity."""

    if task not in TASKS or arm not in ARMS or seed != SMOKE_SEED:
        raise ValueError("endpoint audit is not an E16 Stage-S cell")
    if re.fullmatch(r"[1-9][0-9]*", str(job_id)) is None:
        raise ValueError("endpoint audit requires a positive numeric Slurm job ID")
    identity_path = identity_path.resolve()
    if not identity_path.is_file():
        raise FileNotFoundError(f"campaign identity does not exist: {identity_path}")
    observed_identity_sha256 = _sha256_file(identity_path)
    if (
        re.fullmatch(r"[0-9a-f]{64}", expected_identity_sha256) is None
        or observed_identity_sha256 != expected_identity_sha256
    ):
        raise ValueError("campaign identity does not match its reviewed SHA-256")
    try:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("campaign identity is not JSON") from error
    if not isinstance(identity, dict):
        raise ValueError("campaign identity is not an object")
    expected_prefix = SMOKE_PREFIXES[task]
    if (
        identity.get("schema") != "e16_canonical_campaign_identity_v1"
        or identity.get("protocol") != "E16"
        or identity.get("stage") != "smoke"
        or identity.get("task") != task
        or identity.get("prefix") != expected_prefix
        or identity.get("auto_resume") is not False
        or identity.get("watchdog_requeue") is not False
    ):
        raise ValueError("campaign identity has the wrong E16 Stage-S scope")
    task_config = identity.get("task_config")
    if (
        not isinstance(task_config, dict)
        or task_config.get("seeds") != [SMOKE_SEED]
        or task_config.get("stage") != "smoke"
        or task_config.get("task", {}).get("target_optimizer_updates")
        != SMOKE_UPDATES
        or set(task_config.get("arms", {})) != set(ARMS)
    ):
        raise ValueError("campaign identity does not describe the frozen smoke grid")
    expected_run_stamp = f"{expected_prefix}_{arm}_s{SMOKE_SEED}"
    if run_stamp != expected_run_stamp:
        raise ValueError("endpoint run stamp differs from task/arm/seed identity")
    run_dir = run_dir.resolve()
    if not run_dir.is_dir() or not run_dir.name.endswith(f"_{run_stamp}"):
        raise ValueError("endpoint run directory differs from the exact run stamp")
    protocol_path = Path(str(identity.get("protocol_path", ""))).resolve()
    protocol_sha256 = str(identity.get("protocol_sha256", ""))
    if (
        not protocol_path.is_file()
        or re.fullmatch(r"[0-9a-f]{64}", protocol_sha256) is None
        or _sha256_file(protocol_path) != protocol_sha256
    ):
        raise ValueError("campaign protocol file/hash binding is invalid")
    source_snapshot = Path(str(identity.get("source_snapshot", ""))).resolve()
    if source_snapshot != AUDIT_SOURCE_ROOT:
        raise ValueError("endpoint audit did not import the campaign source snapshot")
    if identity.get("source_snapshot_hash") != identity.get("source_hash"):
        raise ValueError("campaign source snapshot/hash binding is inconsistent")
    execution_surface_hash = str(identity.get("execution_surface_hash", ""))
    execution_snapshot_root = Path(
        str(identity.get("execution_snapshot_root", ""))
    ).resolve()
    execution_identity = identity.get("execution_identity")
    if (
        re.fullmatch(r"[0-9a-f]{64}", execution_surface_hash) is None
        or not execution_snapshot_root.is_dir()
        or not isinstance(execution_identity, dict)
        or execution_identity.get("schema")
        != "e16_execution_surface_identity_v1"
        or execution_identity.get("sha256") != execution_surface_hash
        or not isinstance(execution_identity.get("files"), list)
    ):
        raise ValueError("campaign execution-surface binding is missing")
    audit_relative_path = "ops/exp_scaling/audit_e16_canonical_endpoint.py"
    expected_audit_path = execution_snapshot_root / audit_relative_path
    if Path(__file__).resolve() != expected_audit_path.resolve():
        raise ValueError("endpoint audit was not executed from the frozen surface")
    execution_files = {
        str(record.get("path")): str(record.get("sha256"))
        for record in execution_identity["files"]
        if isinstance(record, dict)
        and set(record) == {"path", "sha256"}
    }
    if execution_files.get(audit_relative_path) != _sha256_file(expected_audit_path):
        raise ValueError("frozen endpoint-auditor file binding is invalid")
    binding = {
        "path": str(identity_path),
        "sha256": observed_identity_sha256,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "execution_snapshot_root": str(execution_snapshot_root),
        "execution_surface_hash": execution_surface_hash,
        "auditor": {
            "path": str(expected_audit_path.resolve()),
            "sha256": execution_files[audit_relative_path],
        },
    }
    cell = {
        "task": task,
        "arm": arm,
        "seed": seed,
        "run_stamp": run_stamp,
        "job_id": str(job_id),
    }
    return cell, binding


def audit_checkpoint(
    *,
    task: str,
    checkpoint: Path,
    tokenizer_path: Path,
    data_root: Path,
    output: Path,
    device_name: str,
    dtype_name: str,
    batch_size: int,
    expected_optimizer_updates: int,
    allow_terminal_alias: bool,
    arm: str,
    seed: int,
    run_stamp: str,
    job_id: str,
    run_dir: Path,
    identity_path: Path,
    expected_identity_sha256: str,
) -> dict[str, Any]:
    """Run and persist an exact E16 graph or Countdown endpoint audit."""

    geometry = policy_geometry(task)
    checkpoint = checkpoint.resolve()
    tokenizer_path = tokenizer_path.resolve()
    data_root = data_root.resolve()
    output = output.resolve()
    run_dir = run_dir.resolve()
    cell_identity, campaign_identity = _load_smoke_cell_identity(
        identity_path=identity_path,
        expected_identity_sha256=expected_identity_sha256,
        task=task,
        arm=arm,
        seed=seed,
        run_stamp=run_stamp,
        job_id=job_id,
        run_dir=run_dir,
    )
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"tokenizer does not exist: {tokenizer_path}")
    if batch_size <= 0 or expected_optimizer_updates <= 0:
        raise ValueError("batch size and expected updates must be positive")
    checkpoint_step, checkpoint_role = validate_endpoint_checkpoint_tag(
        checkpoint,
        expected_optimizer_updates=expected_optimizer_updates,
        allow_terminal_alias=allow_terminal_alias,
    )
    try:
        checkpoint_relative = checkpoint.relative_to(run_dir)
    except ValueError as error:
        raise ValueError("endpoint checkpoint is outside the exact run directory") from error
    checkpoint_parts = checkpoint_relative.parts
    if (
        len(checkpoint_parts) != 3
        or checkpoint_parts[1:] != ("saved_models", f"step_{SMOKE_UPDATES:05d}")
        or not checkpoint_parts[0].startswith("debug_")
        or checkpoint_step != SMOKE_UPDATES
        or checkpoint_role != "scheduled_update_boundary"
        or expected_optimizer_updates != SMOKE_UPDATES
        or allow_terminal_alias
    ):
        raise ValueError(
            "E16 Stage S requires the exact scheduled debug_*/saved_models/step_00032 checkpoint"
        )
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError("checkpoint is missing config.json")
    weight_files = list(checkpoint.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError("checkpoint contains no safetensors weights")
    eval_rows, content_hash = _load_frozen_rows(task, data_root)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if dtype_name == "auto":
        dtype_name = "bfloat16" if device.type == "cuda" else "float32"
    dtype_by_name = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in dtype_by_name:
        raise ValueError(f"unsupported dtype: {dtype_name}")
    if device.type == "cpu" and dtype_name == "float16":
        raise ValueError("float16 checkpoint inference is unsupported on CPU")

    tokenizer_identity_hash = tokenizer_files_hash(tokenizer_path)
    if tokenizer_identity_hash != EXPECTED_TOKENIZER_FILES_HASH:
        raise ValueError("E16 tokenizer file identity drifted")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), trust_remote_code=True, local_files_only=True
    )
    vocab_hash = tokenizer_vocab_hash(tokenizer)
    if vocab_hash != EXPECTED_TOKENIZER_VOCAB_HASH:
        raise ValueError("E16 tokenizer vocabulary drifted")
    action_space = resolve_canonical_action_space(tokenizer, task)
    expected_token_ids = EXPECTED_ACTION_TOKEN_IDS_BY_POSITION[task]
    if action_space.token_ids_by_position != expected_token_ids:
        raise ValueError(
            f"E16 {task} action token IDs drifted: "
            f"expected={expected_token_ids} "
            f"observed={action_space.token_ids_by_position}"
        )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")

    formatted_prompts = [
        _format_prompt(task, str(row["problem"])) for row in eval_rows
    ]
    prompt_token_ids = [
        list(tokenizer.encode(prompt, add_special_tokens=False))
        for prompt in formatted_prompts
    ]
    if any(not prompt for prompt in prompt_token_ids):
        raise ValueError("an E16 evaluation prompt tokenized to an empty sequence")

    weights_hash, weight_records = _weight_identity(weight_files)
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype_by_name[dtype_name],
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    trees = infer_prefix_tree_log_probabilities(
        model,
        prompt_token_ids,
        task=task,
        action_token_ids_by_position=action_space.token_ids_by_position,
        pad_token_id=int(pad_token_id),
        device=device,
        batch_size=batch_size,
    )
    teacher_forced = infer_teacher_forced_log_probabilities(
        model,
        prompt_token_ids,
        task=task,
        action_token_ids_by_position=action_space.token_ids_by_position,
        pad_token_id=int(pad_token_id),
        device=device,
        batch_size=batch_size,
    )

    prompts: list[dict[str, Any]] = []
    aggregates: dict[str, list[float]] = defaultdict(list)
    valid_counts: list[int] = []
    semantic_counts: list[int] = []
    for prompt_index, row in enumerate(eval_rows):
        exact = enumerate_exact_policy(trees[prompt_index], task=task)
        crosscheck = compare_tree_and_teacher_forced(
            trees[prompt_index], teacher_forced[prompt_index], task=task
        )
        codec = validate_code_decoder_bijection(task, str(row["answer"]))
        rewards: list[int] = []
        for record in codec:
            _, reward = boxed_reward_fn(record.decoded_response, str(row["answer"]))
            value = float(reward)
            if not math.isfinite(value) or value not in (0.0, 1.0):
                raise ValueError(f"eval[{prompt_index}] grader returned {reward!r}")
            rewards.append(int(value))
        metrics = valid_policy_metrics(
            exact, rewards, [record.semantic_key for record in codec]
        )
        declared = int(row["answer_mode_count"])
        if metrics["valid_semantic_key_count"] != declared:
            raise ValueError(
                f"eval[{prompt_index}] valid modes drifted: "
                f"declared={declared} audited={metrics['valid_semantic_key_count']}"
            )
        q_plus = metrics.pop("q_plus")
        leaves = []
        for leaf_index, record in enumerate(codec):
            action = geometry.action_indices[leaf_index]
            leaves.append(
                {
                    "action_code": record.code,
                    "action_token_ids": [
                        int(action_space.token_ids_by_position[pos][index])
                        for pos, index in enumerate(action)
                    ],
                    "decoded_response": record.decoded_response,
                    "semantic_key": record.semantic_key,
                    "grader_reward": rewards[leaf_index],
                    "leaf_log_probability": float(
                        exact.leaf_log_probabilities[leaf_index]
                    ),
                    "leaf_probability": float(exact.leaf_probabilities[leaf_index]),
                    "q_plus": q_plus[leaf_index],
                    "teacher_forced_selected_log_probabilities": [
                        float(value) for value in teacher_forced[prompt_index][leaf_index]
                    ],
                    "teacher_forced_sequence_log_probability": float(
                        teacher_forced[prompt_index][leaf_index].sum()
                    ),
                }
            )
        record = {
            "prompt_index": prompt_index,
            "problem_sha256": hashlib.sha256(
                str(row["problem"]).encode("utf-8")
            ).hexdigest(),
            "prompt_token_count": len(prompt_token_ids[prompt_index]),
            "declared_valid_mode_count": declared,
            "probability_sum": exact.probability_sum,
            "probability_sum_abs_error": abs(exact.probability_sum - 1.0),
            "exact_action_entropy": exact.leaf_entropy,
            "conditional_entropy": exact.conditional_entropy,
            "entropy_identity_abs_error": abs(
                exact.leaf_entropy - exact.conditional_entropy
            ),
            **metrics,
            "teacher_forced_vs_prefix_tree_per_token_max_abs_error": crosscheck[
                "per_token_max_abs_error"
            ],
            "teacher_forced_vs_prefix_tree_sequence_max_abs_error": crosscheck[
                "sequence_max_abs_error"
            ],
            "codec": {
                "code_count": len(codec),
                "decoded_response_count": len(
                    {item.decoded_response for item in codec}
                ),
                "semantic_key_count": len({item.semantic_key for item in codec}),
                "bijective": True,
                "label_fields_used_to_construct_support": [],
            },
            "leaves": leaves,
        }
        prompts.append(record)
        for key, value in {
            "exact_action_entropy": exact.leaf_entropy,
            "conditional_entropy": exact.conditional_entropy,
            "p_valid": metrics["p_valid"],
            "h_valid": metrics["h_valid"],
            "n_eff_valid": metrics["n_eff_valid"],
            "h_valid_action": metrics["h_valid_action"],
            "n_eff_valid_action": metrics["n_eff_valid_action"],
            "probability_sum_abs_error": record["probability_sum_abs_error"],
            "entropy_identity_abs_error": record["entropy_identity_abs_error"],
            "teacher_token_error": crosscheck["per_token_max_abs_error"],
            "teacher_sequence_error": crosscheck["sequence_max_abs_error"],
        }.items():
            aggregates[key].append(float(value))
        valid_counts.append(int(metrics["valid_action_count"]))
        semantic_counts.append(int(metrics["valid_semantic_key_count"]))

    result = {
        "schema": "e16_exact_canonical_endpoint_audit_v1",
        "status": "pass",
        "formulation": "e15_derived_direct_on_policy_canonical_maxent",
        "task": task,
        "cell_identity": cell_identity,
        "campaign_identity": campaign_identity,
        "run": {"path": str(run_dir)},
        "source": {
            "root": str(AUDIT_SOURCE_ROOT),
            "python_source_sha256": _source_tree_hash(AUDIT_SOURCE_ROOT),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "oat_step_tag": checkpoint_step,
            "optimizer_updates": expected_optimizer_updates,
            "role": checkpoint_role,
            "weights_manifest_sha256": weights_hash,
            "weight_files": weight_records,
        },
        "data": {
            "root": str(data_root),
            "combined_content_hash": content_hash,
            "eval_rows": len(eval_rows),
            "split": "multi_answer",
        },
        "runtime": {
            "device": str(device),
            "model_dtype": dtype_name,
            "batch_size": batch_size,
            "torch_version": torch.__version__,
        },
        "policy": {
            "action_strings_by_position": [
                list(support) for support in geometry.action_strings_by_position
            ],
            "action_token_ids_by_position": [
                list(support) for support in action_space.token_ids_by_position
            ],
            "support_sizes": list(geometry.support_sizes),
            "horizon": geometry.horizon,
            "prefix_count": len(geometry.prefixes),
            "leaf_count": geometry.leaf_count,
            "max_action_entropy_nats": geometry.max_entropy,
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_files_hash": tokenizer_identity_hash,
            "tokenizer_path": str(tokenizer_path),
            "tokenizer_revision": EXPECTED_TOKENIZER_REVISION,
            "tokenizer_vocab_hash": vocab_hash,
        },
        "tolerances": {
            "probability_sum_abs": PROBABILITY_TOLERANCE,
            "entropy_identity_abs_nats": ENTROPY_TOLERANCE,
            "teacher_forced_log_probability_abs": LOG_PROBABILITY_TOLERANCE,
        },
        "aggregate": {
            "prompt_count": len(prompts),
            "leaf_count": len(prompts) * geometry.leaf_count,
            "exact_action_entropy_mean": _mean(aggregates["exact_action_entropy"]),
            "conditional_entropy_mean": _mean(aggregates["conditional_entropy"]),
            "p_valid_mean": _mean(aggregates["p_valid"]),
            "p_valid_min": min(aggregates["p_valid"]),
            "h_valid_mean": _mean(aggregates["h_valid"]),
            "n_eff_valid_mean": _mean(aggregates["n_eff_valid"]),
            "h_valid_action_mean": _mean(aggregates["h_valid_action"]),
            "n_eff_valid_action_mean": _mean(aggregates["n_eff_valid_action"]),
            "valid_action_count_mean": _mean(
                [float(value) for value in valid_counts]
            ),
            "valid_semantic_key_count_mean": _mean(
                [float(value) for value in semantic_counts]
            ),
            "valid_semantic_key_count_histogram": {
                str(key): value
                for key, value in sorted(Counter(semantic_counts).items())
            },
            "codec_bijection_prompt_count": len(prompts),
            "probability_sum_max_abs_error": max(
                aggregates["probability_sum_abs_error"]
            ),
            "entropy_identity_max_abs_error": max(
                aggregates["entropy_identity_abs_error"]
            ),
            "teacher_forced_vs_prefix_tree_per_token_max_abs_error": max(
                aggregates["teacher_token_error"]
            ),
            "teacher_forced_vs_prefix_tree_sequence_max_abs_error": max(
                aggregates["teacher_sequence_error"]
            ),
        },
        "prompts": prompts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enumerate an E16 exact canonical endpoint policy."
    )
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-stamp", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--identity-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--expected-optimizer-updates", type=int, required=True)
    parser.add_argument("--allow-terminal-alias", action="store_true")
    args = parser.parse_args()
    result = audit_checkpoint(
        task=args.task,
        checkpoint=args.checkpoint,
        tokenizer_path=args.tokenizer,
        data_root=args.data_root,
        output=args.output,
        device_name=args.device,
        dtype_name=args.dtype,
        batch_size=args.batch_size,
        expected_optimizer_updates=args.expected_optimizer_updates,
        allow_terminal_alias=args.allow_terminal_alias,
        arm=args.arm,
        seed=args.seed,
        run_stamp=args.run_stamp,
        job_id=args.job_id,
        run_dir=args.run_dir,
        identity_path=args.identity,
        expected_identity_sha256=args.identity_sha256,
    )
    print(json.dumps(result["aggregate"], sort_keys=True, allow_nan=False))
    print(f"E16 exact endpoint audit passed: {args.output.resolve()}")


if __name__ == "__main__":
    main()
