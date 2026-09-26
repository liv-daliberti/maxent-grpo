#!/usr/bin/env python3
"""Exact endpoint audit for E14's 27 canonical graph-color actions.

The audit evaluates a saved Hugging Face checkpoint, not sampled generations.
For every frozen evaluation prompt it constructs the complete depth-three
restricted-policy tree over the one-token actions ``1``, ``2``, and ``3``.
It then independently teacher-forces every complete action to catch indexing,
padding, or serialization disagreements in the tree calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = ROOT / "src"
AUDIT_SOURCE_ROOT = Path(
    os.environ.get("OAT_ZERO_E14_AUDIT_SOURCE_ROOT", str(DEFAULT_SRC))
).resolve()
if not AUDIT_SOURCE_ROOT.is_dir():
    raise RuntimeError(f"E14 audit source root does not exist: {AUDIT_SOURCE_ROOT}")
for import_root in (ROOT, AUDIT_SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from oat_drgrpo.canonical_actions import (  # noqa: E402
    GRAPH_COLOR_ACTIONS,
    resolve_graph_color_action_token_ids,
)
from oat_drgrpo.math_grader import boxed_reward_fn  # noqa: E402
from oat_drgrpo.templates import apply_qwen_graph_digits_template  # noqa: E402
from ops.exp_scaling.verify_e14_dataset import (  # noqa: E402
    EXPECTED_COMBINED_CONTENT_HASH,
    _content_hash,
    validate_e14_rows,
)


ACTION_INDICES = tuple(itertools.product(range(3), repeat=3))
ACTION_STRINGS = tuple("".join(GRAPH_COLOR_ACTIONS[i] for i in a) for a in ACTION_INDICES)
PREFIXES = tuple(
    prefix
    for depth in range(3)
    for prefix in itertools.product(range(3), repeat=depth)
)
MAX_ACTION_ENTROPY = math.log(27.0)
PROBABILITY_TOLERANCE = 1e-5
ENTROPY_TOLERANCE = 1e-5
LOG_PROBABILITY_TOLERANCE = 5e-3
EXPECTED_EVAL_ROWS = 96
EXPECTED_TRAIN_ROWS = 192
EXPECTED_ACTION_TOKEN_IDS = (16, 17, 18)
EXPECTED_TOKENIZER_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DEFAULT_TOKENIZER_PATH = (
    ROOT
    / "var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct"
    / "snapshots"
    / EXPECTED_TOKENIZER_REVISION
)
EXPECTED_TOKENIZER_FILES_HASH = (
    "caa4fecabf4ddfe3d6678b909ca31e73337cfbfd9aa6befc935a9d1d90ca089d"
)
EXPECTED_TOKENIZER_VOCAB_HASH = (
    "698c955b0b438a535083d1771ef8b41069afba4b3d51a482558bdb68ea55e800"
)
TOKENIZER_IDENTITY_FILES = (
    "config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


@dataclass(frozen=True)
class ExactPolicyResult:
    """Exact finite-tree quantities for one prompt."""

    leaf_log_probabilities: torch.Tensor
    leaf_probabilities: torch.Tensor
    leaf_entropy: float
    conditional_entropy: float
    probability_sum: float


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_hash(source_root: Path) -> str:
    """Reproduce E14 launcher's nested source hash with logical repo paths."""

    files = sorted(
        (path for path in source_root.rglob("*.py") if path.is_file()),
        key=lambda path: path.relative_to(source_root).as_posix(),
    )
    if not files:
        raise ValueError(f"E14 audit source root has no Python files: {source_root}")
    outer = hashlib.sha256()
    for path in files:
        logical_path = ROOT / "src" / path.relative_to(source_root)
        outer.update(f"{_sha256_file(path)}  {logical_path}\n".encode())
    return outer.hexdigest()


def _weight_identity(paths: Iterable[Path]) -> tuple[str, list[dict[str, str]]]:
    digest = hashlib.sha256()
    records = []
    for path in sorted(paths, key=lambda item: item.name):
        file_hash = _sha256_file(path)
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(file_hash))
        records.append({"name": path.name, "sha256": file_hash})
    return digest.hexdigest(), records


def tokenizer_vocab_hash(tokenizer: Any) -> str:
    """Return the vocabulary identity used by the E14 runtime preflight."""

    return hashlib.sha256(
        json.dumps(
            sorted(
                (str(token), int(token_id))
                for token, token_id in tokenizer.get_vocab().items()
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def tokenizer_files_hash(tokenizer_path: Path) -> str:
    """Hash the exact frozen tokenizer/model-config identity files."""

    missing = [
        name for name in TOKENIZER_IDENTITY_FILES if not (tokenizer_path / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"E14 frozen tokenizer directory {tokenizer_path} is missing {missing}"
        )
    digest = hashlib.sha256()
    for name in TOKENIZER_IDENTITY_FILES:
        digest.update(name.encode("utf-8"))
        digest.update((tokenizer_path / name).read_bytes())
    return digest.hexdigest()


def _normalized_node_log_probabilities(
    node_log_probabilities: Mapping[tuple[int, ...], torch.Tensor | Sequence[float]],
) -> dict[tuple[int, ...], torch.Tensor]:
    expected = set(PREFIXES)
    observed = set(node_log_probabilities)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"canonical prefix tree mismatch: missing={missing} extra={extra}"
        )

    normalized: dict[tuple[int, ...], torch.Tensor] = {}
    for prefix in PREFIXES:
        values = torch.as_tensor(
            node_log_probabilities[prefix], dtype=torch.float64
        ).detach().cpu()
        if values.shape != (3,):
            raise ValueError(f"prefix {prefix} must have exactly three log probabilities")
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError(f"prefix {prefix} has nonfinite log probabilities")
        log_normalizer = float(torch.logsumexp(values, dim=0))
        if abs(log_normalizer) > PROBABILITY_TOLERANCE:
            raise ValueError(
                f"prefix {prefix} log probabilities are not normalized: "
                f"logsumexp={log_normalizer}"
            )
        normalized[prefix] = values
    return normalized


def enumerate_exact_policy(
    node_log_probabilities: Mapping[tuple[int, ...], torch.Tensor | Sequence[float]],
) -> ExactPolicyResult:
    """Enumerate all leaves and independently compute both entropy identities."""

    nodes = _normalized_node_log_probabilities(node_log_probabilities)
    leaf_logs = []
    for action in ACTION_INDICES:
        leaf_logs.append(
            sum(nodes[action[:depth]][action[depth]] for depth in range(3))
        )
    leaf_log_probabilities = torch.stack(leaf_logs)
    leaf_probabilities = leaf_log_probabilities.exp()
    probability_sum = float(leaf_probabilities.sum())
    if not math.isfinite(probability_sum):
        raise FloatingPointError("enumerated canonical probability sum is nonfinite")
    if abs(probability_sum - 1.0) > PROBABILITY_TOLERANCE:
        raise ValueError(
            "enumerated canonical probabilities do not sum to one: "
            f"sum={probability_sum}"
        )

    leaf_entropy = float(
        -(leaf_probabilities * leaf_log_probabilities).sum()
    )
    conditional_entropy = 0.0
    for prefix in PREFIXES:
        prefix_log_probability = sum(
            nodes[prefix[:depth]][prefix[depth]] for depth in range(len(prefix))
        )
        prefix_probability = float(torch.exp(torch.as_tensor(prefix_log_probability)))
        local_logs = nodes[prefix]
        local_entropy = float(-(local_logs.exp() * local_logs).sum())
        conditional_entropy += prefix_probability * local_entropy

    if not math.isfinite(leaf_entropy) or not math.isfinite(conditional_entropy):
        raise FloatingPointError("canonical action entropy is nonfinite")
    if not (-ENTROPY_TOLERANCE <= leaf_entropy <= MAX_ACTION_ENTROPY + ENTROPY_TOLERANCE):
        raise ValueError(f"canonical action entropy is outside [0, log(27)]: {leaf_entropy}")
    if abs(leaf_entropy - conditional_entropy) > ENTROPY_TOLERANCE:
        raise ValueError(
            "leaf entropy disagrees with the conditional-entropy identity: "
            f"leaf={leaf_entropy} conditional={conditional_entropy}"
        )
    return ExactPolicyResult(
        leaf_log_probabilities=leaf_log_probabilities,
        leaf_probabilities=leaf_probabilities,
        leaf_entropy=leaf_entropy,
        conditional_entropy=conditional_entropy,
        probability_sum=probability_sum,
    )


def valid_policy_metrics(
    exact: ExactPolicyResult,
    grader_rewards: Sequence[int | float | bool],
) -> dict[str, float | int | list[float]]:
    """Compute the preregistered valid-action conditional distribution."""

    rewards = torch.as_tensor(grader_rewards, dtype=torch.float64)
    if rewards.shape != (27,) or not bool(torch.isfinite(rewards).all()):
        raise ValueError("grader_rewards must be 27 finite binary values")
    if not bool(((rewards == 0.0) | (rewards == 1.0)).all()):
        raise ValueError("grader_rewards must be binary")
    valid_mask = rewards.to(torch.bool)
    valid_action_count = int(valid_mask.sum())
    if valid_action_count <= 0:
        raise ValueError("the frozen data contract requires at least one valid action")

    valid_leaf_logs = exact.leaf_log_probabilities[valid_mask]
    log_p_valid = float(torch.logsumexp(valid_leaf_logs, dim=0))
    if not math.isfinite(log_p_valid):
        raise FloatingPointError("valid-action log normalization is nonfinite")
    p_valid = math.exp(log_p_valid)
    if not math.isfinite(p_valid) or p_valid <= 0.0 or p_valid > 1.0 + PROBABILITY_TOLERANCE:
        raise FloatingPointError(f"invalid exact valid-action probability: {p_valid}")

    q_plus_log = valid_leaf_logs - log_p_valid
    q_plus = q_plus_log.exp()
    q_plus_sum = float(q_plus.sum())
    if abs(q_plus_sum - 1.0) > PROBABILITY_TOLERANCE:
        raise ValueError(f"valid-action conditional policy is not normalized: {q_plus_sum}")
    h_valid = float(-(q_plus * q_plus_log).sum())
    n_eff_valid = math.exp(h_valid)
    if not all(math.isfinite(value) for value in (h_valid, n_eff_valid)):
        raise FloatingPointError("valid-action entropy or effective support is nonfinite")
    if h_valid < -ENTROPY_TOLERANCE or h_valid > math.log(valid_action_count) + ENTROPY_TOLERANCE:
        raise ValueError("valid-action entropy is outside its finite-support bounds")

    q_plus_all = torch.zeros(27, dtype=torch.float64)
    q_plus_all[valid_mask] = q_plus
    return {
        "valid_action_count": valid_action_count,
        "log_p_valid": log_p_valid,
        "p_valid": p_valid,
        "h_valid": h_valid,
        "n_eff_valid": n_eff_valid,
        "q_plus": [float(value) for value in q_plus_all],
    }


def compare_tree_and_teacher_forced(
    node_log_probabilities: Mapping[tuple[int, ...], torch.Tensor | Sequence[float]],
    teacher_forced_selected_log_probabilities: torch.Tensor | Sequence[Sequence[float]],
) -> dict[str, float]:
    """Crosscheck full-action teacher forcing against the prefix-tree policy."""

    nodes = _normalized_node_log_probabilities(node_log_probabilities)
    expected = torch.stack(
        [
            torch.stack(
                [nodes[action[:depth]][action[depth]] for depth in range(3)]
            )
            for action in ACTION_INDICES
        ]
    )
    observed = torch.as_tensor(
        teacher_forced_selected_log_probabilities, dtype=torch.float64
    ).detach().cpu()
    if observed.shape != (27, 3):
        raise ValueError("teacher-forced selected log probabilities must have shape [27, 3]")
    if not bool(torch.isfinite(observed).all()):
        raise FloatingPointError("teacher-forced selected log probabilities are nonfinite")
    per_token_max = float((expected - observed).abs().max())
    sequence_max = float((expected.sum(dim=1) - observed.sum(dim=1)).abs().max())
    if per_token_max > LOG_PROBABILITY_TOLERANCE or sequence_max > LOG_PROBABILITY_TOLERANCE:
        raise ValueError(
            "teacher-forced log probabilities disagree with the restricted prefix tree: "
            f"per_token_max={per_token_max} sequence_max={sequence_max}"
        )
    return {
        "per_token_max_abs_error": per_token_max,
        "sequence_max_abs_error": sequence_max,
    }


def _right_padded_batch(
    sequences: Sequence[Sequence[int]], *, pad_token_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not sequences or any(len(sequence) <= 0 for sequence in sequences):
        raise ValueError("model inputs must be nonempty token sequences")
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    max_length = int(lengths.max())
    input_ids = torch.full(
        (len(sequences), max_length), int(pad_token_id), dtype=torch.long
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[row, :length] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[row, :length] = 1
    return input_ids.to(device), attention_mask.to(device), lengths.to(device)


def _fixed_size_same_length_batches(
    records: Sequence[tuple[Any, ...]],
    *,
    sequence_index: int,
    batch_size: int,
) -> Iterable[tuple[list[tuple[Any, ...]], int]]:
    """Batch equal-length inputs at a fixed batch dimension.

    The final microbatch in each length bucket is filled by duplicating its last
    real record.  Callers must discard rows at and beyond ``real_count``.  This
    makes the prefix-tree and teacher-forced forwards use the same batch shape,
    including their otherwise different 13- and 27-record remainders.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    by_length: dict[int, list[tuple[Any, ...]]] = defaultdict(list)
    for record in records:
        by_length[len(record[sequence_index])].append(record)
    for length in sorted(by_length):
        bucket = by_length[length]
        for start in range(0, len(bucket), batch_size):
            batch = list(bucket[start : start + batch_size])
            real_count = len(batch)
            batch.extend([batch[-1]] * (batch_size - real_count))
            yield batch, real_count


@torch.inference_mode()
def infer_prefix_tree_log_probabilities(
    model: Any,
    prompt_token_ids: Sequence[Sequence[int]],
    *,
    action_token_ids: Sequence[int],
    pad_token_id: int,
    device: torch.device,
    batch_size: int,
) -> list[dict[tuple[int, ...], torch.Tensor]]:
    """Evaluate the 13 prefixes with the leaf audit's exact forward shape.

    Every record has a full three-token continuation.  Unobserved suffix slots
    are filled with the first canonical support token and remain attended.  A
    causal model cannot use those future placeholders at the prefix position,
    while their presence makes sequence width, mask shape, logits slice, and
    batch dimension identical to complete-action teacher forcing.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    action_ids = tuple(int(value) for value in action_token_ids)
    if len(action_ids) != 3 or len(set(action_ids)) != 3:
        raise ValueError("E14 requires exactly three distinct action token ids")
    records: list[tuple[int, tuple[int, ...], list[int]]] = []
    for prompt_index, prompt in enumerate(prompt_token_ids):
        if not prompt:
            raise ValueError(f"prompt {prompt_index} tokenized to an empty sequence")
        for prefix in PREFIXES:
            observed_prefix = [action_ids[index] for index in prefix]
            support_placeholders = [action_ids[0]] * (3 - len(prefix))
            records.append(
                (
                    prompt_index,
                    prefix,
                    list(prompt) + observed_prefix + support_placeholders,
                )
            )

    output: list[dict[tuple[int, ...], torch.Tensor]] = [
        {} for _ in prompt_token_ids
    ]
    allowed = torch.tensor(action_ids, dtype=torch.long, device=device)
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
            logits_to_keep=4,
        ).logits
        if logits.ndim != 3 or logits.shape[1] != 4:
            raise RuntimeError(
                "checkpoint model did not honor logits_to_keep=4 for prefix audit"
            )
        prefix_depths = torch.tensor(
            [len(record[1]) for record in batch], dtype=torch.long, device=device
        )
        batch_rows = torch.arange(len(batch), dtype=torch.long, device=device)
        next_logits = logits[batch_rows, prefix_depths].index_select(-1, allowed)
        next_log_probabilities = torch.log_softmax(next_logits.float(), dim=-1)
        if not bool(torch.isfinite(next_log_probabilities).all()):
            raise FloatingPointError("model produced nonfinite restricted prefix logits")
        for row, (prompt_index, prefix, _) in enumerate(batch[:real_count]):
            output[prompt_index][prefix] = next_log_probabilities[row].double().cpu()
    return output


@torch.inference_mode()
def infer_teacher_forced_log_probabilities(
    model: Any,
    prompt_token_ids: Sequence[Sequence[int]],
    *,
    action_token_ids: Sequence[int],
    pad_token_id: int,
    device: torch.device,
    batch_size: int,
) -> list[torch.Tensor]:
    """Independently teacher-force all 27 complete actions per prompt."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    action_ids = tuple(int(value) for value in action_token_ids)
    if len(action_ids) != 3 or len(set(action_ids)) != 3:
        raise ValueError("E14 requires exactly three distinct action token ids")
    records: list[tuple[int, int, int, list[int]]] = []
    for prompt_index, prompt in enumerate(prompt_token_ids):
        if not prompt:
            raise ValueError(f"prompt {prompt_index} tokenized to an empty sequence")
        for action_index, action in enumerate(ACTION_INDICES):
            action_ids_for_leaf = [action_ids[index] for index in action]
            records.append(
                (
                    prompt_index,
                    action_index,
                    len(prompt),
                    list(prompt) + action_ids_for_leaf,
                )
            )

    output = [torch.empty((27, 3), dtype=torch.float64) for _ in prompt_token_ids]
    allowed = torch.tensor(action_ids, dtype=torch.long, device=device)
    for batch, real_count in _fixed_size_same_length_batches(
        records, sequence_index=3, batch_size=batch_size
    ):
        input_ids, attention_mask, _ = _right_padded_batch(
            [record[3] for record in batch],
            pad_token_id=pad_token_id,
            device=device,
        )
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=4,
        ).logits
        if logits.ndim != 3 or logits.shape[1] != 4:
            raise RuntimeError(
                "checkpoint model did not honor logits_to_keep=4 for leaf audit"
            )
        for row, (prompt_index, action_index, _, _) in enumerate(
            batch[:real_count]
        ):
            # The final four positions are: before a1, before a2, before a3,
            # and after a3. Only the first three score the canonical action.
            restricted_logits = logits[row, :3].index_select(-1, allowed)
            restricted_logs = torch.log_softmax(restricted_logits.float(), dim=-1)
            selected = torch.tensor(
                ACTION_INDICES[action_index], dtype=torch.long, device=device
            )
            selected_logs = restricted_logs.gather(
                -1, selected.unsqueeze(-1)
            ).squeeze(-1)
            if not bool(torch.isfinite(selected_logs).all()):
                raise FloatingPointError(
                    "model produced nonfinite teacher-forced action log probabilities"
                )
            output[prompt_index][action_index] = selected_logs.double().cpu()
    return output


def _load_frozen_rows(data_root: Path) -> tuple[list[dict[str, Any]], str]:
    from datasets import load_from_disk

    train = load_from_disk(str(data_root / "train"))["train"]
    evaluation = load_from_disk(str(data_root / "eval"))["multi_answer"]
    train_rows = validate_e14_rows(list(train), split_tag="train")
    eval_rows = validate_e14_rows(list(evaluation), split_tag="multi_answer")
    if len(train_rows) != EXPECTED_TRAIN_ROWS or len(eval_rows) != EXPECTED_EVAL_ROWS:
        raise ValueError(
            "E14 endpoint audit requires the frozen 192/96 pools; "
            f"got {len(train_rows)}/{len(eval_rows)}"
        )
    content_hash = _content_hash({"eval": eval_rows, "train": train_rows})
    if content_hash != EXPECTED_COMBINED_CONTENT_HASH:
        raise ValueError(
            "E14 endpoint data hash mismatch: "
            f"expected={EXPECTED_COMBINED_CONTENT_HASH} observed={content_hash}"
        )
    return eval_rows, content_hash


def _checkpoint_step(checkpoint: Path) -> int:
    match = re.fullmatch(r"step_(\d+)", checkpoint.name)
    if match is None:
        raise ValueError("E14 checkpoint directory must be named step_<number>")
    return int(match.group(1))


def validate_endpoint_checkpoint_tag(
    checkpoint: Path,
    *,
    expected_optimizer_updates: int,
    allow_terminal_alias: bool,
) -> tuple[int, str]:
    """Require the scheduled update boundary, unless its alias is explicit."""

    checkpoint_step = _checkpoint_step(checkpoint)
    if checkpoint_step == expected_optimizer_updates:
        return checkpoint_step, "scheduled_update_boundary"
    if allow_terminal_alias and checkpoint_step == expected_optimizer_updates + 1:
        return checkpoint_step, "forced_terminal_alias"
    alias_note = (
        f"; step_{expected_optimizer_updates + 1:05d} is accepted only with "
        "--allow-terminal-alias after byte-identity is independently established"
    )
    raise ValueError(
        "checkpoint is not the preregistered optimizer-update endpoint: "
        f"expected=step_{expected_optimizer_updates:05d} "
        f"observed={checkpoint.name}{alias_note}"
    )


def _mean(values: Sequence[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise FloatingPointError("cannot aggregate empty or nonfinite audit values")
    return math.fsum(values) / len(values)


def audit_checkpoint(
    *,
    checkpoint: Path,
    tokenizer_path: Path,
    data_root: Path,
    output: Path,
    device_name: str,
    dtype_name: str,
    batch_size: int,
    expected_optimizer_updates: int,
    allow_terminal_alias: bool,
) -> dict[str, Any]:
    """Run and persist the complete E14 endpoint audit."""

    # The scheduled save at step_00128 is the preregistered endpoint. OAT also
    # writes a byte-identical forced terminal alias at step_00129 after it
    # increments the loop counter, but that alias is opt-in rather than the
    # default scientific target.
    checkpoint = checkpoint.resolve()
    tokenizer_path = tokenizer_path.resolve()
    data_root = data_root.resolve()
    output = output.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint}")
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(
            f"frozen tokenizer directory does not exist: {tokenizer_path}"
        )
    if batch_size <= 0 or expected_optimizer_updates <= 0:
        raise ValueError("batch size and expected optimizer updates must be positive")
    checkpoint_step, checkpoint_role = validate_endpoint_checkpoint_tag(
        checkpoint,
        expected_optimizer_updates=expected_optimizer_updates,
        allow_terminal_alias=allow_terminal_alias,
    )
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError("checkpoint is missing config.json")
    weight_files = list(checkpoint.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError("checkpoint contains no safetensors weights")

    eval_rows, content_hash = _load_frozen_rows(data_root)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if dtype_name == "auto":
        dtype_name = "bfloat16" if device.type == "cuda" else "float32"
    dtype_by_name = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in dtype_by_name:
        raise ValueError(f"unsupported model dtype: {dtype_name}")
    if device.type == "cpu" and dtype_name == "float16":
        raise ValueError("float16 checkpoint inference is unsupported on CPU")
    model_dtype = dtype_by_name[dtype_name]
    weights_manifest_hash, weight_file_records = _weight_identity(weight_files)

    tokenizer_identity_hash = tokenizer_files_hash(tokenizer_path)
    if tokenizer_identity_hash != EXPECTED_TOKENIZER_FILES_HASH:
        raise ValueError(
            "E14 tokenizer file identity drifted: "
            f"expected={EXPECTED_TOKENIZER_FILES_HASH} "
            f"observed={tokenizer_identity_hash}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), trust_remote_code=True, local_files_only=True
    )
    vocab_hash = tokenizer_vocab_hash(tokenizer)
    if vocab_hash != EXPECTED_TOKENIZER_VOCAB_HASH:
        raise ValueError(
            "E14 checkpoint tokenizer vocabulary drifted: "
            f"expected={EXPECTED_TOKENIZER_VOCAB_HASH} observed={vocab_hash}"
        )
    action_token_ids = resolve_graph_color_action_token_ids(tokenizer)
    if action_token_ids != EXPECTED_ACTION_TOKEN_IDS:
        raise ValueError(
            f"E14 action token IDs drifted: expected={EXPECTED_ACTION_TOKEN_IDS} "
            f"observed={action_token_ids}"
        )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("checkpoint tokenizer has neither pad_token_id nor eos_token_id")

    formatted_prompts = [
        apply_qwen_graph_digits_template(str(row["problem"])) for row in eval_rows
    ]
    prompt_token_ids = [
        list(tokenizer.encode(prompt, add_special_tokens=False))
        for prompt in formatted_prompts
    ]
    if any(not prompt for prompt in prompt_token_ids):
        raise ValueError("an E14 evaluation prompt tokenized to an empty sequence")

    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    tree_log_probabilities = infer_prefix_tree_log_probabilities(
        model,
        prompt_token_ids,
        action_token_ids=action_token_ids,
        pad_token_id=int(pad_token_id),
        device=device,
        batch_size=batch_size,
    )
    teacher_forced = infer_teacher_forced_log_probabilities(
        model,
        prompt_token_ids,
        action_token_ids=action_token_ids,
        pad_token_id=int(pad_token_id),
        device=device,
        batch_size=batch_size,
    )

    prompt_records: list[dict[str, Any]] = []
    exact_entropies: list[float] = []
    conditional_entropies: list[float] = []
    p_valid_values: list[float] = []
    h_valid_values: list[float] = []
    n_eff_values: list[float] = []
    probability_sum_errors: list[float] = []
    entropy_identity_errors: list[float] = []
    token_crosscheck_errors: list[float] = []
    sequence_crosscheck_errors: list[float] = []
    valid_counts: list[int] = []

    for prompt_index, row in enumerate(eval_rows):
        exact = enumerate_exact_policy(tree_log_probabilities[prompt_index])
        crosscheck = compare_tree_and_teacher_forced(
            tree_log_probabilities[prompt_index], teacher_forced[prompt_index]
        )
        grader_rewards: list[int] = []
        for action in ACTION_STRINGS:
            _, reward = boxed_reward_fn(action, str(row["answer"]))
            reward_value = float(reward)
            if not math.isfinite(reward_value) or reward_value not in (0.0, 1.0):
                raise ValueError(
                    f"multi_answer[{prompt_index}] grader returned nonbinary {reward!r}"
                )
            grader_rewards.append(int(reward_value))
        metrics = valid_policy_metrics(exact, grader_rewards)
        declared_count = int(row["answer_mode_count"])
        if int(metrics["valid_action_count"]) != declared_count:
            raise ValueError(
                f"multi_answer[{prompt_index}] valid count drifted during audit: "
                f"declared={declared_count} audited={metrics['valid_action_count']}"
            )

        q_plus = metrics.pop("q_plus")
        leaves = []
        for action_index, action in enumerate(ACTION_STRINGS):
            leaves.append(
                {
                    "action": action,
                    "action_token_ids": [
                        int(action_token_ids[index]) for index in ACTION_INDICES[action_index]
                    ],
                    "grader_reward": grader_rewards[action_index],
                    "leaf_log_probability": float(exact.leaf_log_probabilities[action_index]),
                    "leaf_probability": float(exact.leaf_probabilities[action_index]),
                    "q_plus": float(q_plus[action_index]),
                    "teacher_forced_selected_log_probabilities": [
                        float(value) for value in teacher_forced[prompt_index][action_index]
                    ],
                    "teacher_forced_sequence_log_probability": float(
                        teacher_forced[prompt_index][action_index].sum()
                    ),
                }
            )

        probability_sum_error = abs(exact.probability_sum - 1.0)
        entropy_identity_error = abs(exact.leaf_entropy - exact.conditional_entropy)
        prompt_record = {
            "prompt_index": prompt_index,
            "problem_sha256": hashlib.sha256(
                str(row["problem"]).encode("utf-8")
            ).hexdigest(),
            "prompt_token_count": len(prompt_token_ids[prompt_index]),
            "declared_valid_action_count": declared_count,
            "valid_action_count": int(metrics["valid_action_count"]),
            "probability_sum": exact.probability_sum,
            "probability_sum_abs_error": probability_sum_error,
            "exact_action_entropy": exact.leaf_entropy,
            "conditional_entropy": exact.conditional_entropy,
            "entropy_identity_abs_error": entropy_identity_error,
            "p_valid": float(metrics["p_valid"]),
            "log_p_valid": float(metrics["log_p_valid"]),
            "h_valid": float(metrics["h_valid"]),
            "n_eff_valid": float(metrics["n_eff_valid"]),
            "teacher_forced_vs_prefix_tree_per_token_max_abs_error": crosscheck[
                "per_token_max_abs_error"
            ],
            "teacher_forced_vs_prefix_tree_sequence_max_abs_error": crosscheck[
                "sequence_max_abs_error"
            ],
            "leaves": leaves,
        }
        prompt_records.append(prompt_record)
        exact_entropies.append(exact.leaf_entropy)
        conditional_entropies.append(exact.conditional_entropy)
        p_valid_values.append(float(metrics["p_valid"]))
        h_valid_values.append(float(metrics["h_valid"]))
        n_eff_values.append(float(metrics["n_eff_valid"]))
        probability_sum_errors.append(probability_sum_error)
        entropy_identity_errors.append(entropy_identity_error)
        token_crosscheck_errors.append(crosscheck["per_token_max_abs_error"])
        sequence_crosscheck_errors.append(crosscheck["sequence_max_abs_error"])
        valid_counts.append(int(metrics["valid_action_count"]))

    result = {
        "schema": "e14_exact_endpoint_audit_v1",
        "status": "pass",
        "formulation": "canonical_graph_actions",
        "source": {
            "root": str(AUDIT_SOURCE_ROOT),
            "python_source_sha256": _source_tree_hash(AUDIT_SOURCE_ROOT),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "oat_step_tag": checkpoint_step,
            "optimizer_updates": expected_optimizer_updates,
            "role": checkpoint_role,
            "weights_manifest_sha256": weights_manifest_hash,
            "weight_files": weight_file_records,
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
            "actions": list(GRAPH_COLOR_ACTIONS),
            "action_token_ids": list(action_token_ids),
            "horizon": 3,
            "leaf_count": 27,
            "max_action_entropy_nats": MAX_ACTION_ENTROPY,
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
            "prompt_count": len(prompt_records),
            "leaf_count": len(prompt_records) * 27,
            "exact_action_entropy_mean": _mean(exact_entropies),
            "conditional_entropy_mean": _mean(conditional_entropies),
            "p_valid_mean": _mean(p_valid_values),
            "p_valid_min": min(p_valid_values),
            "h_valid_mean": _mean(h_valid_values),
            "n_eff_valid_mean": _mean(n_eff_values),
            "valid_action_count_mean": _mean([float(value) for value in valid_counts]),
            "valid_action_count_histogram": {
                str(key): value for key, value in sorted(Counter(valid_counts).items())
            },
            "probability_sum_max_abs_error": max(probability_sum_errors),
            "entropy_identity_max_abs_error": max(entropy_identity_errors),
            "teacher_forced_vs_prefix_tree_per_token_max_abs_error": max(
                token_crosscheck_errors
            ),
            "teacher_forced_vs_prefix_tree_sequence_max_abs_error": max(
                sequence_crosscheck_errors
            ),
        },
        "prompts": prompt_records,
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
        description="Enumerate E14's exact 27-action endpoint policy."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16")
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--expected-optimizer-updates", type=int, default=128)
    parser.add_argument(
        "--allow-terminal-alias",
        action="store_true",
        help=(
            "accept step_00129 for a 128-update audit only after its byte identity "
            "with the preregistered step_00128 target is independently established"
        ),
    )
    args = parser.parse_args()
    result = audit_checkpoint(
        checkpoint=args.checkpoint,
        tokenizer_path=args.tokenizer,
        data_root=args.data_root,
        output=args.output,
        device_name=args.device,
        dtype_name=args.dtype,
        batch_size=args.batch_size,
        expected_optimizer_updates=args.expected_optimizer_updates,
        allow_terminal_alias=args.allow_terminal_alias,
    )
    print(json.dumps(result["aggregate"], sort_keys=True, allow_nan=False))
    print(f"E14 exact endpoint audit passed: {args.output.resolve()}")


if __name__ == "__main__":
    main()
