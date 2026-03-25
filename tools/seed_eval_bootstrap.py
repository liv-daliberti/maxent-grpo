"""Bootstrap the official SEED eval script with runtime-only overrides.

This keeps the official SEED script untouched while allowing cluster-specific
workarounds, such as disabling vLLM CPU swap cache allocation that hangs on
this environment with the paper-pinned vLLM 0.7.2 stack.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
import time
from pathlib import Path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--script", required=True, help="Path to evaluate_model.py")
    parser.add_argument(
        "--swap-space",
        type=float,
        default=None,
        help="Override vllm.LLM(..., swap_space=...) before the script runs.",
    )
    parser.add_argument(
        "--force-fast-grader",
        action="store_true",
        help="Force the SEED math graders to run with fast=True.",
    )
    parser.add_argument(
        "--strip-grader-input",
        action="store_true",
        help="Strip model outputs before passing them to the SEED math graders.",
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=None,
        help=(
            "Override the time-based seed used by evaluate_model.py by patching "
            "time.time_ns() before the script runs."
        ),
    )
    return parser.parse_known_args()


def maybe_patch_vllm_llm(swap_space: float | None) -> None:
    if swap_space is None:
        return

    import vllm

    orig_llm = vllm.LLM

    def patched_llm(*args, **kwargs):
        kwargs["swap_space"] = swap_space
        return orig_llm(*args, **kwargs)

    vllm.LLM = patched_llm


def maybe_patch_math_grader(force_fast: bool, strip_grader_input: bool) -> None:
    if not force_fast and not strip_grader_input:
        return

    import understand_r1_zero.math_grader as math_grader

    def wrap(fn):
        def patched(model_response, gt_answer, fast=False):
            if strip_grader_input and isinstance(model_response, str):
                model_response = model_response.strip()
            if force_fast:
                fast = True
            return fn(model_response, gt_answer, fast=fast)

        return patched

    math_grader.boxed_reward_fn = wrap(math_grader.boxed_reward_fn)
    math_grader.answer_tag_reward_fn = wrap(math_grader.answer_tag_reward_fn)
    math_grader.answer_tag_reward_fn_for_orz = wrap(
        math_grader.answer_tag_reward_fn_for_orz
    )


def maybe_patch_time_ns(seed: int | None) -> None:
    if seed is None:
        return

    def fixed_time_ns() -> int:
        return seed

    time.time_ns = fixed_time_ns


def main() -> int:
    args, passthrough = parse_args()
    script_path = Path(args.script).resolve()
    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    maybe_patch_vllm_llm(args.swap_space)
    maybe_patch_math_grader(args.force_fast_grader, args.strip_grader_input)
    maybe_patch_time_ns(args.sampling_seed)
    os.chdir(script_dir)
    sys.argv = [str(script_path), *passthrough]
    runpy.run_path(str(script_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
