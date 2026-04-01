#!/usr/bin/env python
"""Run a tiny r1 eval probe with rollout-style vLLM settings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any
import urllib.request


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_CHECKPOINT_ROOT = (
    "/n/fs/similarity/maxent-grpo/var/data/"
    "drgrpo_1p5b_oat_parity_trl_r1_20260331_144832"
)
DEFAULT_DATASET_DIR = (
    "/n/fs/similarity/maxent-grpo/var/seed_paper_eval/external/"
    "SEED-GRPO/datasets/evaluation_suite"
)
DEFAULT_STOP_SEQUENCES = ["</answer>", "</answer>\n"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/n/fs/similarity/maxent-grpo"))
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--steps", default="200,300")
    parser.add_argument("--checkpoint-root", type=Path, default=Path(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dataset-dir", type=Path, default=Path(DEFAULT_DATASET_DIR))
    parser.add_argument("--task", default="aime")
    parser.add_argument("--prompt-start", type=int, default=0)
    parser.add_argument("--prompt-end", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    parser.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")
    return parser.parse_args()


def pick_free_port(preferred: int) -> int:
    def bindable(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                return False
            return True

    if preferred > 0 and bindable(preferred):
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_health(port: int, proc: subprocess.Popen[bytes], log_path: Path) -> None:
    health_url = f"http://127.0.0.1:{port}/health/"
    for _ in range(180):
        try:
            with urllib.request.urlopen(health_url, timeout=5):
                return
        except Exception:
            pass
        if proc.poll() is not None:
            tail = ""
            if log_path.exists():
                tail = "\n".join(log_path.read_text(errors="ignore").splitlines()[-120:])
            raise RuntimeError(
                f"vLLM exited during startup with code {proc.returncode}.\n{tail}"
            )
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {health_url}")


def run_eval_variant(
    *,
    root: Path,
    variant_dir: Path,
    base_model: str,
    dataset_dir: Path,
    port: int,
    task: str,
    prompt_start: int,
    prompt_end: int,
    stop_sequences: list[str] | None,
) -> dict[str, Any]:
    variant_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(root / "src/maxent_grpo/seed_paper_eval.py"),
        "--model-name",
        str(base_model),
        "--template",
        "r1",
        "--tasks",
        str(task),
        "--dataset-dir",
        str(dataset_dir),
        "--vllm-url",
        f"http://127.0.0.1:{port}/generate",
        "--vllm-batch-size",
        "1",
        "--n-samples",
        "1",
        "--max-test",
        "1",
        "--prompt-start",
        str(prompt_start),
        "--prompt-end",
        str(prompt_end),
        "--save-outputs",
        "--results-dir",
        str(variant_dir),
        "--vllm-use-rollout-token-guard",
    ]
    if stop_sequences:
        cmd.extend(["--vllm-stop-sequences", json.dumps(stop_sequences)])
    stdout_path = variant_dir / "eval_stdout.log"
    with stdout_path.open("wb") as handle:
        subprocess.run(
            cmd,
            cwd=str(root),
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )

    summary_candidates = sorted(variant_dir.glob("seed_paper_eval_*.summary.json"))
    if not summary_candidates:
        raise FileNotFoundError(f"No summary JSON found in {variant_dir}")
    summary = json.loads(summary_candidates[-1].read_text())

    output_path = variant_dir / "seed_paper_eval_outputs_single_n1.json"
    if not output_path.exists():
        raise FileNotFoundError(f"No single-output JSON found in {variant_dir}")
    outputs = json.loads(output_path.read_text())
    first_sample = outputs[0]["samples"][0]
    text = str(first_sample.get("text") or "")
    return {
        "summary_path": str(summary_candidates[-1]),
        "outputs_path": str(output_path),
        "result": summary.get("results", {}).get(task),
        "avg_len": summary.get("avg_lens", {}).get(task),
        "max_len": summary.get("max_lens", {}).get(task),
        "formatted_rate": summary.get("formatted", {}).get(task),
        "sample_reward": first_sample.get("reward"),
        "sample_formatted": first_sample.get("formatted"),
        "sample_token_count": first_sample.get("token_count"),
        "sample_text_head": text[:300],
        "sample_text_tail": text[-300:] if text else "",
    }


def terminate_process(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    steps = [int(part.strip()) for part in str(args.steps).split(",") if part.strip()]
    overview: dict[str, Any] = {"steps": {}, "task": args.task}

    for step in steps:
        step_dir = results_dir / f"step{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        model_path = args.checkpoint_root / f"checkpoint-{step}"
        port = pick_free_port(18000 + step)
        internal_port = pick_free_port(19000 + step)
        if internal_port == port:
            internal_port = pick_free_port(0)
        env = os.environ.copy()
        env["VLLM_DP_MASTER_PORT"] = str(internal_port)
        log_path = step_dir / "vllm_server.log"
        with log_path.open("wb") as log_handle:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(root / "tools/vllm_serve_compat.py"),
                    "--model",
                    str(model_path),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--tensor_parallel_size",
                    "1",
                    "--data_parallel_size",
                    "1",
                    "--max_model_len",
                    str(args.max_model_len),
                    "--dtype",
                    str(args.dtype),
                    "--enforce_eager",
                    "true" if args.enforce_eager else "false",
                ],
                cwd=str(root),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        try:
            wait_for_health(port, proc, log_path)
            overview["steps"][str(step)] = {
                "guarded": run_eval_variant(
                    root=root,
                    variant_dir=step_dir / "guarded",
                    base_model=args.base_model,
                    dataset_dir=args.dataset_dir,
                    port=port,
                    task=args.task,
                    prompt_start=args.prompt_start,
                    prompt_end=args.prompt_end,
                    stop_sequences=None,
                ),
                "guarded_stop": run_eval_variant(
                    root=root,
                    variant_dir=step_dir / "guarded_stop",
                    base_model=args.base_model,
                    dataset_dir=args.dataset_dir,
                    port=port,
                    task=args.task,
                    prompt_start=args.prompt_start,
                    prompt_end=args.prompt_end,
                    stop_sequences=DEFAULT_STOP_SEQUENCES,
                ),
            }
        finally:
            terminate_process(proc)

    overview_path = results_dir / "probe_overview.json"
    overview_path.write_text(json.dumps(overview, indent=2) + "\n")
    print(json.dumps({"overview_path": str(overview_path), "steps": steps}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
