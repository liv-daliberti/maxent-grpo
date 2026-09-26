#!/usr/bin/env python3
"""Train the frozen train-only AntMaze v13 restricted-action warm start."""

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

from oat_drgrpo.ant_maze_interactive_policy import ANT_POLICY_ACTIONS  # noqa: E402
from oat_drgrpo.interactive_sft import restricted_action_cross_entropy  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big")); digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=75313)
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=1536)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_model.exists() or args.output_receipt.exists():
        raise FileExistsError("fresh AntMaze v13 SFT artifacts required")
    if (args.epochs, args.batch_size, args.gradient_accumulation) != (64, 4, 4):
        raise ValueError("AntMaze v13 SFT schedule changed")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("AntMaze v13 SFT requires a BF16 GPU")
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    identity_path = args.data_root / "identity.json"
    examples_path = args.data_root / "examples.jsonl"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if (
        identity.get("status") != "pass"
        or identity.get("split") != "train_only"
        or identity.get("example_count") != 32
        or identity.get("episode_count") != 8
        or identity.get("information_boundary", {}).get("dev_dataset_loaded")
        or identity.get("information_boundary", {}).get("eval_dataset_loaded")
    ):
        raise ValueError("AntMaze v13 train-only data firewall failed")
    if sha(examples_path) != identity["hashes"]["examples_sha256"]:
        raise ValueError("AntMaze v13 examples hash changed")
    examples = [json.loads(line) for line in examples_path.read_text().splitlines() if line]
    if len(examples) != 32:
        raise ValueError("AntMaze v13 requires exactly 32 examples")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    action_token_ids = []
    for action in ANT_POLICY_ACTIONS:
        token_ids = tokenizer.encode(action, add_special_tokens=False)
        if len(token_ids) != 1 or tokenizer.decode(token_ids) != action:
            raise RuntimeError(f"Ant action {action!r} is not one exact token")
        action_token_ids.append(int(token_ids[0]))
    if len(set(action_token_ids)) != 8:
        raise RuntimeError("Ant action token IDs are not unique")

    encoded = []
    for example in examples:
        prompt_ids = tokenizer.encode(str(example["prompt"]), add_special_tokens=False)
        if not prompt_ids or len(prompt_ids) > args.max_length:
            raise ValueError("AntMaze v13 prompt length invalid")
        label = str(example["label"])
        if label not in ANT_POLICY_ACTIONS:
            raise ValueError("AntMaze v13 label left action support")
        encoded.append({"input_ids": prompt_ids, "target": ANT_POLICY_ACTIONS.index(label)})

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, fused=True
    )
    batches_per_epoch = math.ceil(len(encoded) / args.batch_size)
    if batches_per_epoch % args.gradient_accumulation:
        raise ValueError("AntMaze v13 has a partial optimizer step")
    total_steps = args.epochs * batches_per_epoch // args.gradient_accumulation
    if total_steps != 128:
        raise ValueError("AntMaze v13 must have exactly 128 optimizer updates")

    def lr_scale(step: int) -> float:
        if step < args.warmup_steps:
            return float(step + 1) / args.warmup_steps
        return max(float(total_steps - (step + 1)) / max(total_steps - args.warmup_steps, 1), 0.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    action_ids = torch.tensor(action_token_ids, dtype=torch.long, device="cuda")

    def collate(indices: list[int]):
        selected = [encoded[index] for index in indices]
        maximum = max(len(row["input_ids"]) for row in selected)
        input_ids = torch.full((len(selected), maximum), int(tokenizer.pad_token_id), dtype=torch.long)
        attention = torch.zeros((len(selected), maximum), dtype=torch.long)
        positions, targets = [], []
        for row_index, row in enumerate(selected):
            length = len(row["input_ids"])
            input_ids[row_index, :length] = torch.tensor(row["input_ids"])
            attention[row_index, :length] = 1
            positions.append(length - 1); targets.append(row["target"])
        return (
            input_ids.cuda(non_blocking=True), attention.cuda(non_blocking=True),
            torch.tensor(positions, device="cuda"), torch.tensor(targets, device="cuda"),
        )

    records = []
    optimizer.zero_grad(set_to_none=True)
    update = 0
    for epoch in range(args.epochs):
        permutation = torch.randperm(len(encoded), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        loss_sum = 0.0; correct = examples_seen = 0
        for batch_index, start in enumerate(range(0, len(permutation), args.batch_size), 1):
            indices = permutation[start : start + args.batch_size]
            input_ids, attention, positions, targets = collate(indices)
            logits = model(input_ids=input_ids, attention_mask=attention).logits
            decision = logits[torch.arange(len(indices), device="cuda"), positions]
            loss = restricted_action_cross_entropy(decision, action_token_ids=action_ids, target_action_indices=targets)
            (loss / args.gradient_accumulation).backward()
            with torch.no_grad():
                correct += int(decision.index_select(1, action_ids).argmax(1).eq(targets).sum())
                examples_seen += len(indices); loss_sum += float(loss.detach()) * len(indices)
            if batch_index % args.gradient_accumulation == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); update += 1
                records.append({
                    "update": update, "epoch": epoch + 1,
                    "mean_loss": loss_sum / examples_seen,
                    "accuracy": correct / examples_seen,
                    "grad_norm": float(grad_norm.detach()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                })
                loss_sum = 0.0; correct = examples_seen = 0
                if update == 1 or update % 16 == 0:
                    print(f"[ant-v13-sft] update={update}/128 loss={records[-1]['mean_loss']:.6f} accuracy={records[-1]['accuracy']:.3f}", flush=True)
    if update != 128:
        raise RuntimeError("AntMaze v13 optimizer update count changed")

    model.eval(); final_correct = 0; final_loss_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(encoded), args.batch_size):
            indices = list(range(start, min(start + args.batch_size, len(encoded))))
            input_ids, attention, positions, targets = collate(indices)
            decision = model(input_ids=input_ids, attention_mask=attention).logits[
                torch.arange(len(indices), device="cuda"), positions
            ]
            loss = restricted_action_cross_entropy(decision, action_token_ids=action_ids, target_action_indices=targets)
            final_loss_sum += float(loss) * len(indices)
            final_correct += int(decision.index_select(1, action_ids).argmax(1).eq(targets).sum())
    final_loss = final_loss_sum / len(encoded); final_accuracy = final_correct / len(encoded)
    if not math.isfinite(final_loss):
        raise RuntimeError("AntMaze v13 final loss is nonfinite")
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_model, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(args.output_model)
    atomic(args.output_receipt, {
        "schema_version": "ant-maze-interactive-warmstart-sft-v13",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass", "seed": args.seed, "epochs": args.epochs,
        "batch_size": args.batch_size, "gradient_accumulation": args.gradient_accumulation,
        "optimizer_updates": update, "example_count": len(encoded),
        "final_loss": final_loss, "final_accuracy": final_accuracy,
        "action_token_ids": dict(zip(ANT_POLICY_ACTIONS, action_token_ids)),
        "updates": records,
        "hashes": {
            "base_model_config_sha256": sha(args.model / "config.json"),
            "data_identity_sha256": sha(identity_path),
            "examples_sha256": sha(examples_path),
            "protocol_sha256": sha(args.protocol),
            "trainer_sha256": sha(Path(__file__).resolve()),
            "output_model_tree_sha256": tree_sha(args.output_model),
        },
        "information_boundary": {
            "train_examples_only": True, "dev_dataset_loaded": False,
            "eval_dataset_loaded": False, "online_reward_used": False,
        },
    })
    print(f"[ant-v13-sft] status=pass final_loss={final_loss:.6f} final_accuracy={final_accuracy:.3f}", flush=True)


if __name__ == "__main__":
    main()
