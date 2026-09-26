#!/usr/bin/env python3
"""Train a shared PointMaze waypoint policy over dynamic legal supports."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any, Sequence


ROOT = Path(
    os.environ.get("OAT_ZERO_REPO_ROOT", Path(__file__).resolve().parents[1])
).resolve()
SRC = Path(os.environ.get("OAT_ZERO_SOURCE_ROOT", ROOT / "src")).resolve()
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oat_drgrpo.point_maze_waypoint_policy import (  # noqa: E402
    POINT_WAYPOINT_LABELS,
    POINT_WAYPOINT_PROMPT_FORMATS,
    convert_point_waypoint_prompt,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=88401)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument(
        "--prompt-format",
        choices=POINT_WAYPOINT_PROMPT_FORMATS,
        default="qwen_chatml",
    )
    return parser.parse_args()


def _restricted_loss(
    decision_logits: Any,
    *,
    supports: Sequence[Sequence[int]],
    selected_ids: Sequence[int],
    weights: Any,
) -> tuple[Any, Any]:
    import torch

    normalized = tuple(tuple(int(token) for token in row) for row in supports)
    if len(normalized) != int(decision_logits.shape[0]):
        raise ValueError("one dynamic support is required per SFT example")
    maximum = max(len(row) for row in normalized)
    support_ids = torch.empty(
        (len(normalized), maximum), dtype=torch.long, device="cuda"
    )
    support_mask = torch.zeros_like(support_ids, dtype=torch.bool)
    positions = []
    for index, (support, selected) in enumerate(zip(normalized, selected_ids)):
        if not support or len(support) != len(set(support)):
            raise ValueError("SFT action supports must be nonempty and unique")
        if int(selected) not in support:
            raise ValueError("SFT target lies outside its legal support")
        support_ids[index] = support[0]
        support_ids[index, : len(support)] = torch.tensor(
            support, dtype=torch.long, device="cuda"
        )
        support_mask[index, : len(support)] = True
        positions.append(support.index(int(selected)))
    restricted = decision_logits.gather(1, support_ids).float()
    restricted = restricted.masked_fill(~support_mask, -torch.inf)
    logprobs = torch.log_softmax(restricted, dim=-1)
    position_tensor = torch.tensor(positions, dtype=torch.long, device="cuda")
    nll = -logprobs[
        torch.arange(len(normalized), device="cuda"),
        position_tensor,
    ]
    loss = (nll * weights).mean()
    predictions = restricted.argmax(dim=-1)
    return loss, predictions.eq(position_tensor)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or (
        args.max_updates is not None and args.max_updates <= 0
    ):
        raise ValueError("waypoint SFT epochs and max updates must be positive")
    if args.output_model.exists() or args.output_receipt.exists():
        raise FileExistsError("fresh waypoint SFT outputs are required")
    if not args.protocol.is_file():
        raise FileNotFoundError(args.protocol)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("waypoint SFT requires a BF16-capable GPU")
    identity_path = args.data_root / "identity.json"
    examples_path = args.data_root / "examples.jsonl"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if (
        identity.get("status") != "pass"
        or identity.get("split") != "train_only"
        or identity.get("policy_interface") != "point_waypoint_v1"
        or identity.get("information_boundary", {}).get("dev_rows_materialized")
        or identity.get("information_boundary", {}).get("eval_rows_materialized")
        or _sha256(examples_path) != identity.get("hashes", {}).get("examples_sha256")
    ):
        raise ValueError("waypoint SFT data firewall or identity is invalid")
    examples = [
        json.loads(line)
        for line in examples_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not examples or len(examples) != int(identity["example_count"]):
        raise ValueError("waypoint SFT example count changed")
    episode_lengths = {
        str(row["episode_id"]): int(row["decision_count"])
        for row in identity["episodes"]
    }
    mean_episode_length = len(examples) / len(episode_lengths)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    label_token_ids = {}
    for label in POINT_WAYPOINT_LABELS:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1 or tokenizer.decode(ids) != label:
            raise RuntimeError(f"waypoint label {label!r} is not one exact token")
        label_token_ids[label] = int(ids[0])
    if len(set(label_token_ids.values())) != len(label_token_ids):
        raise RuntimeError("waypoint label token IDs are not unique")

    encoded = []
    for example in examples:
        rendered_prompt = convert_point_waypoint_prompt(
            str(example["prompt"]),
            prompt_format=args.prompt_format,
        )
        prompt_ids = tokenizer.encode(rendered_prompt, add_special_tokens=False)
        if not prompt_ids or len(prompt_ids) > args.max_length:
            raise ValueError("waypoint SFT prompt length is invalid")
        label = str(example["label"])
        allowed_labels = tuple(str(item) for item in example["allowed_labels"])
        if label not in allowed_labels or any(
            item not in label_token_ids for item in allowed_labels
        ):
            raise ValueError("waypoint SFT target/support labels are invalid")
        episode_id = str(example["episode_id"])
        encoded.append(
            {
                "input_ids": prompt_ids,
                "selected_id": label_token_ids[label],
                "support_ids": tuple(label_token_ids[item] for item in allowed_labels),
                "weight": mean_episode_length / episode_lengths[episode_id],
            }
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        fused=True,
    )
    batch_count = math.ceil(len(encoded) / args.batch_size)
    steps_per_epoch = math.ceil(batch_count / args.gradient_accumulation)
    planned_steps = args.epochs * steps_per_epoch
    if args.max_updates is not None and args.max_updates > planned_steps:
        raise ValueError("waypoint SFT max updates exceed the epoch budget")
    total_steps = (
        planned_steps if args.max_updates is None else int(args.max_updates)
    )

    def lr_scale(step: int) -> float:
        if step < args.warmup_steps:
            return float(step + 1) / max(float(args.warmup_steps), 1.0)
        remaining = total_steps - step - 1
        return max(
            float(remaining) / max(total_steps - args.warmup_steps, 1),
            0.0,
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)

    def collate(indices: Sequence[int]):
        rows = [encoded[index] for index in indices]
        maximum = max(len(row["input_ids"]) for row in rows)
        input_ids = torch.full(
            (len(rows), maximum),
            int(tokenizer.pad_token_id),
            dtype=torch.long,
        )
        attention = torch.zeros_like(input_ids)
        positions = []
        for index, row in enumerate(rows):
            length = len(row["input_ids"])
            input_ids[index, :length] = torch.tensor(row["input_ids"])
            attention[index, :length] = 1
            positions.append(length - 1)
        return (
            input_ids.cuda(non_blocking=True),
            attention.cuda(non_blocking=True),
            torch.tensor(positions, dtype=torch.long, device="cuda"),
            [row["support_ids"] for row in rows],
            [row["selected_id"] for row in rows],
            torch.tensor(
                [row["weight"] for row in rows],
                dtype=torch.float32,
                device="cuda",
            ),
        )

    updates = []
    update_index = 0
    for epoch in range(args.epochs):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        permutation = torch.randperm(len(encoded), generator=generator).tolist()
        batches = [
            permutation[start : start + args.batch_size]
            for start in range(0, len(permutation), args.batch_size)
        ]
        nominal_update_examples = args.batch_size * args.gradient_accumulation
        for group_start in range(0, len(batches), args.gradient_accumulation):
            if update_index >= total_steps:
                break
            group = batches[group_start : group_start + args.gradient_accumulation]
            optimizer.zero_grad(set_to_none=True)
            loss_sum = correct = examples_seen = 0.0
            for batch_indices in group:
                (
                    input_ids,
                    attention,
                    positions,
                    supports,
                    selected_ids,
                    weights,
                ) = collate(batch_indices)
                logits = model(input_ids=input_ids, attention_mask=attention).logits
                decision = logits[
                    torch.arange(len(batch_indices), device="cuda"),
                    positions,
                ]
                loss, matches = _restricted_loss(
                    decision,
                    supports=supports,
                    selected_ids=selected_ids,
                    weights=weights,
                )
                (loss * len(batch_indices) / nominal_update_examples).backward()
                loss_sum += float(loss.detach().item()) * len(batch_indices)
                correct += float(matches.sum().item())
                examples_seen += len(batch_indices)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(args.max_grad_norm)
            )
            optimizer.step()
            scheduler.step()
            update_index += 1
            updates.append(
                {
                    "update": update_index,
                    "epoch": epoch + 1,
                    "mean_loss": loss_sum / examples_seen,
                    "accuracy": correct / examples_seen,
                    "grad_norm": float(grad_norm.detach().item()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
            if update_index == 1 or update_index % 10 == 0:
                print(
                    f"[waypoint-sft] update={update_index}/{total_steps} "
                    f"loss={updates[-1]['mean_loss']:.6f} "
                    f"accuracy={updates[-1]['accuracy']:.3f}",
                    flush=True,
                )
    if update_index != total_steps:
        raise RuntimeError("waypoint SFT optimizer step count changed")

    model.eval()
    final_loss_sum = final_correct = 0.0
    with torch.no_grad():
        for start in range(0, len(encoded), args.batch_size):
            indices = list(range(start, min(start + args.batch_size, len(encoded))))
            (
                input_ids,
                attention,
                positions,
                supports,
                selected_ids,
                weights,
            ) = collate(indices)
            logits = model(input_ids=input_ids, attention_mask=attention).logits
            decision = logits[
                torch.arange(len(indices), device="cuda"),
                positions,
            ]
            loss, matches = _restricted_loss(
                decision,
                supports=supports,
                selected_ids=selected_ids,
                weights=weights,
            )
            final_loss_sum += float(loss.item()) * len(indices)
            final_correct += float(matches.sum().item())
    final_loss = final_loss_sum / len(encoded)
    final_accuracy = final_correct / len(encoded)
    if not math.isfinite(final_loss):
        raise RuntimeError("waypoint SFT final loss is nonfinite")

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        args.output_model, safe_serialization=True, max_shard_size="2GB"
    )
    tokenizer.save_pretrained(args.output_model)
    receipt = {
        "schema": "point-maze-waypoint-sft-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "decision": "eligible_for_waypoint_development_gate",
        "base_model": str(args.model.resolve()),
        "output_model": str(args.output_model.resolve()),
        "model_tree_sha256": _tree_sha256(args.output_model),
        "data_identity_sha256": _sha256(identity_path),
        "examples_sha256": _sha256(examples_path),
        "protocol_sha256": _sha256(args.protocol),
        "trainer_sha256": _sha256(Path(__file__).resolve()),
        "seed": args.seed,
        "epochs": args.epochs,
        "max_updates": args.max_updates,
        "steps_per_epoch": steps_per_epoch,
        "planned_optimizer_steps": planned_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "optimizer_steps": total_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "prompt_format": args.prompt_format,
        "label_token_ids": label_token_ids,
        "loss": "route-length-neutral-example-weighted-cross-entropy-over-dynamic-legal-support",
        "final_train_loss": final_loss,
        "final_train_accuracy": final_accuracy,
        "updates": updates,
        "information_boundary": {
            "train_only_examples": len(encoded),
            "dev_rows_loaded": False,
            "eval_rows_loaded": False,
            "online_reward_used": False,
        },
    }
    _atomic_json(args.output_receipt, receipt)
    print(
        f"[waypoint-sft] complete loss={final_loss:.6f} "
        f"accuracy={final_accuracy:.3f} output={args.output_model}",
        flush=True,
    )


if __name__ == "__main__":
    main()
