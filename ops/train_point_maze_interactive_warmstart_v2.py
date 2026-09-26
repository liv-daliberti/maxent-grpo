#!/usr/bin/env python3
"""PointMaze v2 interactive warm start (SFT on v2 certified route replays).

Same trainer as v1 with two cohort-specific assumptions relaxed: the frozen
644-example count and the update schedule tuned for it. The data firewall check
(train-only, no dev/eval loaded) and the examples-hash check are retained, since
those are what keep the warm start honest rather than merely v1-shaped.
"""

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
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = Path(os.environ.get("OAT_ZERO_SOURCE_ROOT", ROOT / "src")).resolve()
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oat_drgrpo.interactive_sft import restricted_action_cross_entropy  # noqa: E402
from oat_drgrpo.point_maze_interactive_policy import (  # noqa: E402
    POINT_POLICY_LABELS,
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
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
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
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=75201)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=7)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument(
        "--policy-interface",
        choices=("history_v1", "compact_state_v2", "velocity_state_v3"),
        default="history_v1",
    )
    parser.add_argument(
        "--frozen-short-balanced-v6",
        action="store_true",
        help="Use the prospectively frozen 3-epoch balanced velocity schedule.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_model.exists() or args.output_receipt.exists():
        raise FileExistsError("fresh PointMaze warm-start artifacts are required")
    expected_epochs = (
        3
        if args.frozen_short_balanced_v6
        else {
            "history_v1": 3,
            "compact_state_v2": 8,
            "velocity_state_v3": 12,
        }[args.policy_interface]
    )
    # v1 froze epochs/batch/accumulation to a schedule tuned for 644 examples.
    # v2 has ~77k, so the schedule is a parameter here; the integrity guards that
    # actually matter (data firewall, examples hash, identity count) are kept.
    if args.policy_interface != "velocity_state_v3":
        raise ValueError("PointMaze v2 warm start expects the velocity_state_v3 interface")
    if not torch_available():
        raise RuntimeError("PyTorch is unavailable")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("PointMaze warm start requires a BF16-capable GPU")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    identity_path = args.data_root / "identity.json"
    examples_path = args.data_root / "examples.jsonl"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if (
        identity.get("status") != "pass"
        or identity.get("split") != "train_only"
        or identity.get("information_boundary", {}).get("dev_dataset_loaded")
        or identity.get("information_boundary", {}).get("eval_dataset_loaded")
    ):
        raise ValueError("PointMaze warm-start data firewall is not passing")
    identity_interface = identity.get("policy_interface", "history_v1")
    if identity_interface != args.policy_interface:
        raise ValueError("PointMaze warm-start policy interface changed")
    if _sha256(examples_path) != identity["hashes"]["examples_sha256"]:
        raise ValueError("PointMaze warm-start examples hash changed")
    examples = [
        json.loads(line)
        for line in examples_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(examples) != identity["example_count"]:
        raise ValueError("PointMaze warm-start example count changed")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    label_token_ids = []
    for label in POINT_POLICY_LABELS:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1 or tokenizer.decode(token_ids) != label:
            raise RuntimeError(f"Point policy label {label!r} is not one token")
        label_token_ids.append(int(token_ids[0]))
    if len(set(label_token_ids)) != len(label_token_ids):
        raise RuntimeError("Point policy label token IDs are not unique")

    encoded = []
    for example in examples:
        prompt_ids = tokenizer.encode(
            str(example["prompt"]),
            add_special_tokens=False,
        )
        if not prompt_ids or len(prompt_ids) > args.max_length:
            raise ValueError("Point warm-start prompt length is invalid")
        label = str(example["label"])
        if label not in POINT_POLICY_LABELS:
            raise ValueError("Point warm-start label left the action support")
        encoded.append(
            {
                "input_ids": prompt_ids,
                "target": POINT_POLICY_LABELS.index(label),
            }
        )

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
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        fused=True,
    )
    batches_per_epoch = math.ceil(len(encoded) / args.batch_size)
    if batches_per_epoch % args.gradient_accumulation:
        raise ValueError("frozen batch schedule has a partial optimizer step")
    total_steps = (
        args.epochs * batches_per_epoch // args.gradient_accumulation
    )
    expected_steps = (
        69
        if args.frozen_short_balanced_v6
        else {
            "history_v1": 69,
            "compact_state_v2": 184,
            "velocity_state_v3": 276,
        }[args.policy_interface]
    )
    if total_steps != expected_steps:
        raise ValueError(
            f"frozen PointMaze warm start must have {expected_steps} updates"
        )

    def lr_scale(step_index: int) -> float:
        if step_index < args.warmup_steps:
            return float(step_index + 1) / float(args.warmup_steps)
        remaining = total_steps - (step_index + 1)
        denominator = max(total_steps - args.warmup_steps, 1)
        return max(float(remaining) / float(denominator), 0.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    action_ids = torch.tensor(label_token_ids, dtype=torch.long, device="cuda")

    def collate(
        indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = [encoded[index] for index in indices]
        maximum = max(len(row["input_ids"]) for row in rows)
        input_ids = torch.full(
            (len(rows), maximum),
            int(tokenizer.pad_token_id),
            dtype=torch.long,
        )
        attention = torch.zeros((len(rows), maximum), dtype=torch.long)
        positions = []
        targets = []
        for row_index, row in enumerate(rows):
            length = len(row["input_ids"])
            input_ids[row_index, :length] = torch.tensor(row["input_ids"])
            attention[row_index, :length] = 1
            positions.append(length - 1)
            targets.append(row["target"])
        return (
            input_ids.cuda(non_blocking=True),
            attention.cuda(non_blocking=True),
            torch.tensor(positions, dtype=torch.long, device="cuda"),
            torch.tensor(targets, dtype=torch.long, device="cuda"),
        )

    update_records = []
    optimizer.zero_grad(set_to_none=True)
    update_index = 0
    for epoch in range(args.epochs):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        permutation = torch.randperm(len(encoded), generator=generator).tolist()
        accumulated_loss = 0.0
        accumulated_correct = 0
        accumulated_examples = 0
        for batch_index, start in enumerate(
            range(0, len(permutation), args.batch_size),
            start=1,
        ):
            batch_indices = permutation[start : start + args.batch_size]
            input_ids, attention, positions, targets = collate(batch_indices)
            logits = model(input_ids=input_ids, attention_mask=attention).logits
            decision_logits = logits[
                torch.arange(len(batch_indices), device="cuda"),
                positions,
            ]
            loss = restricted_action_cross_entropy(
                decision_logits,
                action_token_ids=action_ids,
                target_action_indices=targets,
            )
            (loss / args.gradient_accumulation).backward()
            with torch.no_grad():
                predictions = decision_logits.index_select(1, action_ids).argmax(1)
                accumulated_correct += int(predictions.eq(targets).sum().item())
                accumulated_examples += len(batch_indices)
                accumulated_loss += float(loss.detach().item()) * len(batch_indices)
            if batch_index % args.gradient_accumulation == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(args.max_grad_norm),
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_index += 1
                update_records.append(
                    {
                        "update": update_index,
                        "epoch": epoch + 1,
                        "mean_loss_since_update": (
                            accumulated_loss / accumulated_examples
                        ),
                        "accuracy_since_update": (
                            accumulated_correct / accumulated_examples
                        ),
                        "grad_norm": float(grad_norm.detach().item()),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    }
                )
                accumulated_loss = 0.0
                accumulated_correct = 0
                accumulated_examples = 0
                if update_index == 1 or update_index % 10 == 0:
                    record = update_records[-1]
                    print(
                        "[point-warmstart-sft] "
                        f"update={update_index}/{total_steps} "
                        f"loss={record['mean_loss_since_update']:.6f} "
                        f"accuracy={record['accuracy_since_update']:.3f}",
                        flush=True,
                    )
    if update_index != total_steps:
        raise RuntimeError("Point warm-start optimizer-step count changed")

    model.eval()
    final_correct = 0
    final_loss_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(encoded), args.batch_size):
            indices = list(range(start, min(start + args.batch_size, len(encoded))))
            input_ids, attention, positions, targets = collate(indices)
            logits = model(input_ids=input_ids, attention_mask=attention).logits
            decision_logits = logits[
                torch.arange(len(indices), device="cuda"),
                positions,
            ]
            loss = restricted_action_cross_entropy(
                decision_logits,
                action_token_ids=action_ids,
                target_action_indices=targets,
            )
            final_loss_sum += float(loss.item()) * len(indices)
            final_correct += int(
                decision_logits.index_select(1, action_ids)
                .argmax(1)
                .eq(targets)
                .sum()
                .item()
            )
    final_loss = final_loss_sum / len(encoded)
    final_accuracy = final_correct / len(encoded)
    if not math.isfinite(final_loss):
        raise RuntimeError("Point warm-start final loss is nonfinite")

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        args.output_model,
        safe_serialization=True,
        max_shard_size="2GB",
    )
    tokenizer.save_pretrained(args.output_model)
    receipt = {
        "schema_version": (
            "point-maze-interactive-warmstart-sft-v1"
            if args.policy_interface == "history_v1"
            else (
                "point-maze-interactive-warmstart-sft-v2"
                if args.policy_interface == "compact_state_v2"
                else "point-maze-interactive-warmstart-sft-v3"
            )
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "decision": "eligible_for_unchanged_closed_loop_development_gate",
        "base_model": str(args.model.resolve()),
        "output_model": str(args.output_model.resolve()),
        "data_root": str(args.data_root.resolve()),
        "data_identity_sha256": _sha256(identity_path),
        "examples_sha256": _sha256(examples_path),
        "protocol_sha256": _sha256(args.protocol),
        "trainer_sha256": _sha256(Path(__file__).resolve()),
        "model_tree_sha256": _tree_sha256(args.output_model),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "optimizer_steps": total_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "max_grad_norm": args.max_grad_norm,
        "dtype": "bfloat16",
        "loss": "cross_entropy_renormalized_over_nine_action_label_tokens_only",
        "policy_interface": args.policy_interface,
        "frozen_short_balanced_v6": args.frozen_short_balanced_v6,
        "label_token_ids": dict(zip(POINT_POLICY_LABELS, label_token_ids)),
        "final_train_loss": final_loss,
        "final_train_accuracy": final_accuracy,
        "updates": update_records,
        "information_boundary": {
            "train_only_examples": len(encoded),
            "dev_dataset_loaded": False,
            "eval_dataset_loaded": False,
            "online_reward_used": False,
            "terminal_verifier_used_during_sft": False,
        },
    }
    _atomic_json(args.output_receipt, receipt)
    print(
        "[point-warmstart-sft] "
        f"status=pass final_loss={final_loss:.6f} "
        f"final_accuracy={final_accuracy:.3f} output={args.output_model}",
        flush=True,
    )


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


if __name__ == "__main__":
    main()
