#!/usr/bin/env python3
"""Launch a PointMaze Tour run as a chain of one-hour chunks.

This cluster routes by requested walltime: `--time` of one hour or less lands in
the large `all` partition and starts immediately, anything longer is rerouted
into the contested cs/mltheory queues. A multi-pass run is therefore submitted
as one job per pass, each depending on the previous and resuming from its
checkpoint, so the whole run executes in the only lane with free capacity.

    python ops/exp_scaling/launch_point_maze_tour_chunked.py \
        --stage stage2 --arm control --seed 43 --passes 4
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "ops/slurm/point_maze_tour_chunk.slurm"
BASE_MODEL = (
    ROOT / "var/cache/huggingface/transformers"
    / "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots"
    / "7ae557604adf67be50417f59c2c2f167def9a775"
)
TRAIN_ROWS = 384


def submit(
    *,
    stage: str,
    arm: str,
    seed: int,
    passes: int,
    chunk_updates: int,
    eval_prompts: int,
    eval_interval: int,
    eval_split: str,
    prompt_format: str,
    data_root: Path,
    model: Path,
    depends_on: str | None,
) -> str:
    exports = ",".join(
        [
            "ALL",
            f"ROOT_DIR={ROOT}",
            f"OAT_ZERO_ARM={arm}",
            f"OAT_ZERO_SEED={seed}",
            f"OAT_ZERO_TOUR_STAGE={stage}",
            f"OAT_ZERO_TOUR_PASSES={passes}",
            f"OAT_ZERO_TOUR_MODEL={model}",
            f"OAT_ZERO_TOUR_CHUNK_UPDATES={chunk_updates}",
            f"OAT_ZERO_TOUR_EVAL_PROMPTS={eval_prompts}",
            f"OAT_ZERO_TOUR_EVAL_INTERVAL={eval_interval}",
            f"OAT_ZERO_TOUR_EVAL_SPLIT={eval_split}",
            f"OAT_ZERO_TOUR_DATA={data_root}",
            f"OAT_ZERO_TOUR_PROMPT_FORMAT={prompt_format}",
        ]
    )
    command = [
        "sbatch",
        "--parsable",
        f"--job-name=tour-{stage}-{arm[:3]}{seed}",
        "--partition=all",
        "--account=allcs",
        "--gres=gpu:a6000:1",
        "--cpus-per-task=4",
        "--mem=32G",
        # One hour or less is what keeps this in `all`; do not raise it.
        "--time=01:00:00",
        f"--export={exports}",
    ]
    if depends_on:
        # afterany, not afterok: a chunk killed by a node fault has still
        # written its checkpoint, and the next chunk resumes from it. afterok
        # would strand the chain on any transient failure.
        command.append(f"--dependency=afterany:{depends_on}")
    command.append(str(SCRIPT))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "sbatch failed")
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    # "semantic" is verified replay plus the fixed open-set semantic MaxEnt
    # term, the arm this domain was previously excluded from.
    parser.add_argument(
        "--arm", required=True, choices=("control", "replay", "semantic")
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--chunk-updates", type=int, default=TRAIN_ROWS)
    parser.add_argument("--eval-prompts", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=384)
    parser.add_argument("--eval-split", default="dev", choices=("dev", "eval"))
    parser.add_argument(
        "--prompt-format", default="qwen_chatml",
        choices=("qwen_chatml", "falcon3"),
        help="must match the model; a Falcon surface on Qwen weights runs silently wrong",
    )
    parser.add_argument("--model", type=Path, default=BASE_MODEL)
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "var/data/point_maze_tour_v1r1"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total_updates = args.passes * TRAIN_ROWS
    # Extending an existing run must only submit the chunks still outstanding;
    # submitting from zero queues jobs that find the run already terminal and
    # die on its receipt.
    done = 0
    checkpoint = (
        ROOT / "var/checkpoints"
        / f"tour_{args.stage}_{args.arm}_s{args.seed}" / "COMPLETE.json"
    )
    if checkpoint.is_file():
        import json

        done = int(json.loads(checkpoint.read_text())["update"])
    remaining = max(0, total_updates - done)
    chunks = -(-remaining // args.chunk_updates)
    resumed = f" (resuming from {done})" if done else ""
    print(
        f"{args.stage} {args.arm} s{args.seed}: {remaining} updates remaining "
        f"of {total_updates} in {chunks} chunk(s) of {args.chunk_updates}{resumed}"
    )
    if chunks == 0:
        print("  already complete; nothing to submit")
        return
    if args.dry_run:
        return
    previous: str | None = None
    ids: list[str] = []
    for index in range(chunks):
        job = submit(
            stage=args.stage,
            arm=args.arm,
            seed=args.seed,
            passes=args.passes,
            chunk_updates=args.chunk_updates,
            eval_prompts=args.eval_prompts,
            eval_interval=args.eval_interval,
            eval_split=args.eval_split,
            prompt_format=args.prompt_format,
            data_root=args.data_root,
            model=args.model,
            depends_on=previous,
        )
        note = " (starts now)" if index == 0 else f" (after {ids[index - 1]})"
        print(f"  chunk {index + 1}/{chunks} -> {job}{note}")
        ids.append(job)
        previous = job
    print(" ".join(ids))


if __name__ == "__main__":
    main()
