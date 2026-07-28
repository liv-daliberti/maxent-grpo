#!/usr/bin/env python3
"""Materialize E49B's hard-but-solvable toy from immutable prior evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, load_from_disk


ROOT = Path(__file__).resolve().parents[2]
E39 = ROOT / "var/data/math12k_384_math500"
DEFAULT_OUTPUT = ROOT / "var/data/e49b_math_strategy_toy"
EXPECTED_TRAIN_ARROW = (
    "359defbf82b6e05a1fdddb3479ed689f8a607dc727814e73ebfe69b2ffdff8b8"
)
EXPECTED_EVAL_ARROW = (
    "2104f8f8eef09ce0bfc929e255f0f04c59311f1f3395bd293f1d03051c482cf7"
)
PRIOR_METRICS = {
    43: (
        ROOT
        / "var/data/xdr_qwen25_0p5b_instruct_grpo_"
        "mte39_math12k_384_semantic_entropy_05b_v1_grpo_s43/"
        "debug_job30065410/train_metrics.jsonl",
        "3e0ea068cbfd780ed3a940172d5cd649611df8e38ea0e37e0e0d3ceb3339a4ab",
    ),
    44: (
        ROOT
        / "var/data/xdr_qwen25_0p5b_instruct_grpo_"
        "mte39_math12k_384_semantic_entropy_05b_v1_grpo_s44/"
        "debug_job30065413/train_metrics.jsonl",
        "760482b0b4e553774927117284031a8510d982d65ebf8ac06d4b779c42d46ad3",
    ),
    45: (
        ROOT
        / "var/data/xdr_qwen25_0p5b_instruct_grpo_"
        "mte39_math12k_384_semantic_entropy_05b_v1_grpo_s45/"
        "debug_job30065416/train_metrics.jsonl",
        "b3e09d4fff395259998c2d81c97480aa6e32919be799a25cc772bb7271f8937b",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(item)))
    return digest.hexdigest()


def _prior_first_epoch_rewards() -> dict[int, dict[int, float]]:
    rewards: dict[int, dict[int, float]] = {}
    for seed, (path, expected_hash) in PRIOR_METRICS.items():
        if _sha256(path) != expected_hash:
            raise RuntimeError(f"prior seed-{seed} metrics identity mismatch")
        values = []
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if (
                "actor/rewards" in row
                and int(row.get("misc/prompt_epoch", -1)) == 0
            ):
                values.append(float(row["actor/rewards"]))
        if len(values) != 384:
            raise RuntimeError(
                f"prior seed-{seed} first epoch has {len(values)} groups"
            )
        order = torch.randperm(
            384,
            generator=torch.Generator().manual_seed(seed),
        ).tolist()
        for source_index, reward in zip(order, values, strict=True):
            rewards.setdefault(source_index, {})[seed] = reward
    if len(rewards) != 384 or any(len(values) != 3 for values in rewards.values()):
        raise RuntimeError("prior three-seed reward reconstruction is incomplete")
    return rewards


def materialize(output: Path) -> None:
    train_arrow = E39 / "train/train/data-00000-of-00001.arrow"
    eval_arrow = E39 / "eval/math/data-00000-of-00001.arrow"
    if _sha256(train_arrow) != EXPECTED_TRAIN_ARROW:
        raise RuntimeError("E39 training Arrow identity mismatch")
    if _sha256(eval_arrow) != EXPECTED_EVAL_ARROW:
        raise RuntimeError("E39 evaluation Arrow identity mismatch")

    full_train = load_from_disk(str(E39 / "train"))["train"]
    prior_rewards = _prior_first_epoch_rewards()
    eligible = []
    for source_index, by_seed in prior_rewards.items():
        row = full_train[source_index]
        mean_reward = sum(by_seed.values()) / len(by_seed)
        if (
            str(row["level"]) in {"3", "4"}
            and 0.10 <= mean_reward <= 0.55
            and sum(value > 0 for value in by_seed.values()) >= 2
        ):
            eligible.append(source_index)
    eligible.sort(
        key=lambda index: hashlib.sha256(
            (
                "e49b-support-toy-v1"
                + str(full_train[index]["unique_id"])
            ).encode("utf-8")
        ).hexdigest()
    )
    selected_indices = eligible[:50]
    if len(eligible) < 50 or len(selected_indices) != 50:
        raise RuntimeError("fewer than 50 prior-screened hard-solvable rows")
    toy_train = full_train.select(selected_indices)

    full_eval = load_from_disk(str(E39 / "eval"))["math"]
    eval_indices = []
    for difficulty in (3, 4):
        candidates = [
            index
            for index, value in enumerate(full_eval["difficulty"])
            if float(value) == float(difficulty)
        ]
        candidates.sort(
            key=lambda index: hashlib.sha256(
                (
                    f"e49b-eval-d{difficulty}-v1"
                    + str(full_eval[index]["problem"])
                    + str(full_eval[index]["answer"])
                ).encode("utf-8")
            ).hexdigest()
        )
        eval_indices.extend(candidates[:25])
    toy_eval = full_eval.select(eval_indices)
    if len(toy_eval) != 50:
        raise RuntimeError("MATH-500 lacks the matched level-3/4 evaluation rows")

    output.mkdir(parents=True, exist_ok=False)
    DatasetDict({"train": Dataset.from_dict(toy_train.to_dict())}).save_to_disk(
        str(output / "train")
    )
    DatasetDict({"math": Dataset.from_dict(toy_eval.to_dict())}).save_to_disk(
        str(output / "eval")
    )
    manifest = {
        "schema": "e49b_math_strategy_toy_v2",
        "source": "var/data/math12k_384_math500",
        "source_train_arrow_sha256": EXPECTED_TRAIN_ARROW,
        "source_eval_arrow_sha256": EXPECTED_EVAL_ARROW,
        "train_rows": 50,
        "selection": {
            "levels": [3, 4],
            "prior_mean_reward_interval": [0.10, 0.55],
            "positive_prior_seeds_at_least": 2,
            "eligible_rows": len(eligible),
            "tie_break": "sha256(e49b-support-toy-v1 || unique_id)",
            "prior_metric_sha256": {
                str(seed): expected_hash
                for seed, (_, expected_hash) in PRIOR_METRICS.items()
            },
        },
        "train_source_indices": selected_indices,
        "train_unique_ids": toy_train["unique_id"],
        "train_level_counts": {
            level: toy_train["level"].count(level)
            for level in sorted(set(toy_train["level"]))
        },
        "train_prior_reward": {
            str(index): {
                "mean": sum(prior_rewards[index].values()) / 3,
                "by_seed": {
                    str(seed): value
                    for seed, value in sorted(prior_rewards[index].items())
                },
            }
            for index in selected_indices
        },
        "train_subject_counts": {
            subject: toy_train["subject"].count(subject)
            for subject in sorted(set(toy_train["subject"]))
        },
        "eval_rows": 50,
        "eval_difficulty_counts": {"3": 25, "4": 25},
        "eval_source_indices": eval_indices,
        "train_tree_sha256": _tree_hash(output / "train"),
        "eval_tree_sha256": _tree_hash(output / "eval"),
    }
    (output / "MATERIALIZATION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def audit(output: Path) -> None:
    manifest = json.loads(
        (output / "MATERIALIZATION_MANIFEST.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema") != "e49b_math_strategy_toy_v2":
        raise RuntimeError("invalid E49 toy manifest")
    if _tree_hash(output / "train") != manifest["train_tree_sha256"]:
        raise RuntimeError("E49 toy training tree changed")
    if _tree_hash(output / "eval") != manifest["eval_tree_sha256"]:
        raise RuntimeError("E49 toy evaluation tree changed")
    train = load_from_disk(str(output / "train"))["train"]
    evaluation = load_from_disk(str(output / "eval"))["math"]
    if len(train) != 50 or len(evaluation) != 50:
        raise RuntimeError("E49 toy row-count mismatch")
    if list(train["unique_id"]) != manifest["train_unique_ids"]:
        raise RuntimeError("E49 toy training order changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        audit(args.output)
    else:
        materialize(args.output)
        audit(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
