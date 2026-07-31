#!/usr/bin/env python3
"""Train the frozen 64-example ConstructiveCode v7 shared warm start."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any


SEED = 77201
EPOCHS = 4
EXAMPLES = 64
ACCUMULATION = 8
UPDATES = 32
MAX_LENGTH = 4096
LEARNING_RATE = 1e-5
TRAIN_IDS = {"327_B", "659_C", "1283_C", "1102_B"}
FORBIDDEN_IDS = {"359_B", "988_A", "1399_D", "361_B", "1294_C", "149_C"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(root: Path) -> str:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        records.append((path.relative_to(root).as_posix(), sha(path)))
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


def atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    return parser.parse_args()


def encode_assistant_only(
    tokenizer: Any, prompt: str, completion: str
) -> tuple[list[int], list[int]]:
    """Tokenize once and mask every token wholly contained in the prompt.

    Byte-pair tokenization need not be prefix-stable when text is concatenated.
    In particular, a completion-leading newline can merge with the prompt's
    final newline. Offset masking preserves that boundary token as a target
    because it contains completion text while masking every prompt-only token.
    """
    full_text = prompt + completion
    tokenized = tokenizer(
        full_text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )
    input_ids = tokenized["input_ids"]
    offsets = tokenized["offset_mapping"]
    boundary = len(prompt)
    if (
        len(input_ids) != len(offsets)
        or len(input_ids) > MAX_LENGTH
        or not input_ids
        or not offsets
        or offsets[-1][1] != len(full_text)
    ):
        raise RuntimeError("v7 SFT token offsets or maximum length drift")
    labels = [
        -100 if end <= boundary else token
        for token, (_, end) in zip(input_ids, offsets)
    ]
    if (
        all(label == -100 for label in labels)
        or any(
            label != -100 and end <= boundary
            for label, (_, end) in zip(labels, offsets)
        )
        or any(
            label == -100 and end > boundary
            for label, (_, end) in zip(labels, offsets)
        )
    ):
        raise RuntimeError("v7 SFT assistant-only loss boundary drift")
    return input_ids, labels


def main() -> None:
    args = parse_args()
    if args.output_model.exists() or args.output_receipt.exists():
        raise FileExistsError("fresh ConstructiveCode v7 SFT outputs required")
    for path in (
        args.base_model / "config.json", args.examples, args.manifest,
        args.protocol, args.identity,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = json.loads(args.manifest.read_text())
    rows = [json.loads(line) for line in args.examples.read_text().splitlines() if line]
    if (
        manifest.get("status") != "pass"
        or manifest.get("example_count") != EXAMPLES
        or manifest.get("train_problem_ids") != ["327_B", "659_C", "1283_C", "1102_B"]
        or manifest.get("development_problem_ids_loaded") != []
        or manifest.get("evaluation_problem_ids_loaded") != []
        or manifest.get("language_model_sampling") is not False
        or len(rows) != EXAMPLES
        or {row.get("source_problem_id") for row in rows} != TRAIN_IDS
        or any(row.get("source_problem_id") in FORBIDDEN_IDS for row in rows)
    ):
        raise RuntimeError("ConstructiveCode v7 SFT corpus crossed its frozen boundary")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("ConstructiveCode v7 SFT requires a BF16 GPU")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = []
    for row in rows:
        encoded.append(encode_assistant_only(
            tokenizer, row["prompt"], row["code"] + "<|im_end|>\n"
        ))
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
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
        model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999),
        eps=1e-8, weight_decay=0.0,
    )
    optimizer.zero_grad(set_to_none=True)
    records = []
    micro_losses = []
    update = 0
    for epoch in range(EPOCHS):
        order = list(range(EXAMPLES))
        random.Random(SEED + epoch).shuffle(order)
        for position, index in enumerate(order, start=1):
            input_ids, labels = encoded[index]
            inputs = torch.tensor([input_ids], dtype=torch.long, device="cuda")
            targets = torch.tensor([labels], dtype=torch.long, device="cuda")
            output = model(input_ids=inputs, labels=targets, use_cache=False)
            loss = output.loss
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite ConstructiveCode v7 SFT loss")
            float_loss = float(loss.detach().cpu())
            micro_losses.append(float_loss)
            (loss / ACCUMULATION).backward()
            if position % ACCUMULATION == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError("nonfinite ConstructiveCode v7 SFT gradient")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                record = {
                    "update": update,
                    "epoch": epoch + 1,
                    "mean_loss": sum(micro_losses[-ACCUMULATION:]) / ACCUMULATION,
                    "grad_norm": float(grad_norm.detach().cpu()),
                }
                records.append(record)
                print(
                    f"[constructive-v7-sft] update={update}/{UPDATES} "
                    f"loss={record['mean_loss']:.6f} grad={record['grad_norm']:.6f}",
                    flush=True,
                )
    if update != UPDATES or len(micro_losses) != EPOCHS * EXAMPLES:
        raise RuntimeError("ConstructiveCode v7 SFT schedule drift")
    if not all(math.isfinite(row["mean_loss"]) for row in records):
        raise RuntimeError("ConstructiveCode v7 SFT emitted nonfinite metrics")
    model.config.use_cache = True
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        args.output_model, safe_serialization=True, max_shard_size="5GB"
    )
    tokenizer.save_pretrained(args.output_model)
    payload = {
        "schema": "constructive-code-v7-train-only-sft-receipt-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "job_id": int(args.job_id),
        "seed": SEED,
        "epochs": EPOCHS,
        "example_count": EXAMPLES,
        "gradient_accumulation": ACCUMULATION,
        "optimizer_updates": update,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "assistant_only_loss": True,
        "base_model_tree_sha256": tree_hash(args.base_model),
        "output_model_tree_sha256": tree_hash(args.output_model),
        "examples_sha256": sha(args.examples),
        "manifest_sha256": sha(args.manifest),
        "protocol_sha256": sha(args.protocol),
        "identity_sha256": sha(args.identity),
        "train_problem_ids": sorted(TRAIN_IDS),
        "development_problem_ids_loaded": [],
        "evaluation_problem_ids_loaded": [],
        "first_update_loss": records[0]["mean_loss"],
        "final_update_loss": records[-1]["mean_loss"],
        "metrics": records,
    }
    atomic(args.output_receipt, payload)
    print(
        f"[constructive-v7-sft] complete updates={update} "
        f"final_loss={records[-1]['mean_loss']:.6f}", flush=True,
    )


if __name__ == "__main__":
    main()
