#!/usr/bin/env python
"""
Launch a vLLM server (via TRL) and verify ``client_tag`` metadata is echoed.

Example:
    python tools/vllm_client_tag_smoke.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --client-tag rank-0 \
        --prompts "Hello, world?"

Pass ``--no-launch`` to skip spawning a server when one is already running.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from typing import Any, Iterable, Optional

import requests


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF repo or path for vLLM.")
    parser.add_argument("--revision", help="Optional HF revision to load.")
    parser.add_argument("--host", default="127.0.0.1", help="vLLM host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=29525, help="vLLM HTTP port (default: 29525).")
    parser.add_argument(
        "--group-port",
        type=int,
        default=None,
        help="Optional NCCL group port for tensor-parallel weight sync.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallelism passed to vllm_serve (default: 1).",
    )
    parser.add_argument("--dtype", default="float16", help="Model dtype (default: float16).")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="vLLM GPU utilization (default: 0.90).",
    )
    parser.add_argument(
        "--client-tag",
        default="smoke-rank-0",
        help="Tag forwarded via headers and verified in the response.",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["Hello, world?"],
        help="Prompts to send in the smoke request.",
    )
    parser.add_argument("--n", type=int, default=2, help="Completions per prompt (default: 2).")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for the smoke request (default: 0.8).",
    )
    parser.add_argument(
        "--return-logprobs",
        action="store_true",
        help="Request per-token logprob metadata and verify the response carries it.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=60.0,
        help="Timeout (seconds) for HTTP requests (default: 60).",
    )
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="Assume the vLLM server is already running; skip launching.",
    )
    parser.add_argument(
        "--server-log",
        default="var/artifacts/logs/vllm_client_tag_smoke.log",
        help="File to capture vLLM stdout/err when launching.",
    )
    parser.add_argument(
        "--keep-server",
        action="store_true",
        help="Do not terminate the launched server after the check.",
    )
    return parser.parse_args()


def _launch_vllm(args: argparse.Namespace) -> Optional[subprocess.Popen]:
    if args.no_launch:
        return None
    cmd = [
        sys.executable,
        "-m",
        "trl.scripts.vllm_serve",
        "--model",
        args.model,
        "--tensor_parallel_size",
        str(args.tensor_parallel_size),
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--gpu_memory_utilization",
        str(args.gpu_memory_utilization),
        "--enforce_eager",
        "false",
    ]
    if args.revision:
        cmd.extend(["--revision", args.revision])
    if args.group_port:
        cmd.extend(["--group-port", str(args.group_port)])
    log_path = os.path.abspath(args.server_log)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "ab")
    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("VLLM_USAGE_STATS_PATH", os.path.join("var", "cache", "vllm", "usage_stats.json"))
    env.setdefault("VLLM_NO_USAGE_STATS", "1")
    print(f"[smoke] Launching vLLM server:\n  {' '.join(cmd)}\n  log: {log_path}")
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
    return proc


def _wait_for_health(host: str, port: int, timeout: float) -> None:
    deadline = time.time() + timeout
    url = f"http://{host}:{port}/health"
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=5.0)
            if resp.status_code == 200:
                print("[smoke] vLLM /health OK.")
                return
        except requests.RequestException:
            pass
        time.sleep(2.0)
    raise RuntimeError(f"Timed out waiting for vLLM health at {url}")


def _extract_tags(
    payload: dict, keys: Iterable[str] = ("results", "outputs", "choices")
) -> set[str]:
    found = set()
    if isinstance(payload, dict):
        meta = payload.get("metadata")
        if isinstance(meta, dict) and meta.get("client_tag"):
            found.add(meta["client_tag"])
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                for entry in value:
                    found |= _extract_tags(entry, keys)
            elif isinstance(value, dict):
                found |= _extract_tags(value, keys)
    elif isinstance(payload, list):
        for entry in payload:
            found |= _extract_tags(entry, keys)
    return found


def _response_has_logprobs(payload: Any) -> bool:
    if isinstance(payload, dict):
        if any(
            key in payload
            for key in (
                "cumulative_logprob",
                "logprobs",
                "token_logprobs",
                "output_token_logprobs",
            )
        ):
            return True
        if isinstance(payload.get("metadata"), dict) and any(
            key in payload["metadata"]
            for key in ("logprob_sum", "token_logprobs", "raw_output")
        ):
            return True
        for key in ("results", "outputs", "choices"):
            value = payload.get(key)
            if _response_has_logprobs(value):
                return True
    elif isinstance(payload, list):
        return any(_response_has_logprobs(entry) for entry in payload)
    return False


def _post_generate(args: argparse.Namespace) -> dict:
    url = f"http://{args.host}:{args.port}/generate"
    payload = {
        "prompts": args.prompts,
        "n": args.n,
        "temperature": args.temperature,
    }
    if args.return_logprobs:
        payload["return_logprobs"] = True
    headers = {"X-VLLM-Client-Tag": args.client_tag}
    resp = requests.post(url, json=payload, headers=headers, timeout=args.request_timeout)
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    args = _parse_args()
    proc = None
    try:
        proc = _launch_vllm(args)
        _wait_for_health(args.host, args.port, timeout=300.0 if proc else 30.0)
        payload = _post_generate(args)
        tags = _extract_tags(payload)
        print(json.dumps(payload, indent=2)[:1000])
        if args.client_tag not in tags:
            raise RuntimeError(
                f"client_tag '{args.client_tag}' not found in response metadata: {sorted(tags)}"
            )
        if args.return_logprobs and not _response_has_logprobs(payload):
            raise RuntimeError(
                "Requested logprobs but response did not contain recognizable logprob metadata."
            )
        print(f"[smoke] Success: server echoed client_tag '{args.client_tag}'.")
        if args.return_logprobs:
            print("[smoke] Logprob metadata detected in response.")
        return 0
    finally:
        if proc and not args.keep_server:
            print("[smoke] Terminating vLLM server …")
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
