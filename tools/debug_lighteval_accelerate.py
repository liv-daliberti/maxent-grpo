#!/usr/bin/env python
"""Debug wrapper around ``lighteval accelerate``.

This script mirrors the minimal CLI surface used by ``ops/slurm/evaluate.slurm``
but injects extra logging hooks into the LightEval ``Pipeline`` so we can see
which phase (tasks loading, model run, metrics) a job is stuck in.

Usage (from repo root, matching the Slurm script):

    python tools/debug_lighteval_accelerate.py \\
        "model_name=...,revision=...,batch_size=1,..." \\
        "lighteval|math_500|0|0" \\
        --output-dir var/artifacts/lighteval/checkpoint-600 \\
        --save-details \\
        --max-samples 1
"""

from __future__ import annotations

import argparse
import datetime as _dt
from typing import Optional


def _install_pipeline_debug_hooks() -> None:
    """Patch LightEval's Pipeline methods to emit phase markers.

    This is intentionally best-effort and only used for debugging; if any of the
    expected symbols are missing we simply return without raising.
    """

    try:
        from lighteval import pipeline as le_pipeline  # type: ignore[import]
    except Exception:  # pragma: no cover - optional / env-specific
        return

    def _wrap(obj, method_name: str, label: str | None = None, timeout_sec: float | None = None):
        orig = getattr(obj, method_name, None)
        if orig is None:
            return

        use_label = label or f"{getattr(obj, '__name__', obj)}.{method_name}"

        def wrapped(*args, **kwargs):  # type: ignore[override]
            cancel = None
            if timeout_sec:
                try:
                    import faulthandler

                    cancel = faulthandler.dump_traceback_later(timeout_sec, repeat=True)
                except Exception:
                    cancel = None
            ts = _dt.datetime.now().isoformat(timespec="seconds")
            print(f"[debug-lighteval][{ts}] ENTER {use_label}", flush=True)
            try:
                return orig(*args, **kwargs)
            finally:
                if cancel:
                    try:
                        import faulthandler

                        faulthandler.cancel_dump_traceback_later()
                    except Exception:
                        pass
                ts2 = _dt.datetime.now().isoformat(timespec="seconds")
                print(f"[debug-lighteval][{ts2}] EXIT  {use_label}", flush=True)

        setattr(obj, method_name, wrapped)

    Pipeline = getattr(le_pipeline, "Pipeline", None)
    if Pipeline is None:
        return

    # Key phases we care about for "hang" debugging.
    # Include early init helpers so we can pinpoint stalls before tasks load.
    for name in (
        "_init_parallelism_manager",
        "_init_model",
        "_init_random_seeds",
        "_init_tasks_and_requests",
        "_run_model",
        "_compute_metrics",
    ):
        _wrap(Pipeline, name)

    # Add lower-level hooks around model loading steps to catch hangs inside Transformers load.
    try:
        from lighteval.models import model_loader as _ml  # type: ignore[import]
        _wrap(_ml, "load_model")
    except Exception:
        pass

    try:
        from lighteval.models.transformers import transformers_model as _tm  # type: ignore[import]
        for _name in (
            "__init__",
            "_create_auto_model",
            "_create_auto_tokenizer",
            "init_model_parallel",
        ):
            _wrap(_tm.TransformersModel, _name, label=f"TransformersModel.{_name}")
        _wrap(
            _tm.TransformersModelConfig,
            "get_transformers_config",
            label="TransformersModelConfig.get_transformers_config",
        )
    except Exception:
        pass

    try:
        from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore[import]
        _wrap(AutoConfig, "from_pretrained", label="AutoConfig.from_pretrained")
        # Set a timeout so we dump stack traces if model load hangs.
        _wrap(
            AutoModelForCausalLM,
            "from_pretrained",
            label="AutoModelForCausalLM.from_pretrained",
            timeout_sec=300,
        )
    except Exception:
        pass


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Debug wrapper for `lighteval accelerate` that adds phase logging. "
            "Only a subset of the CLI surface used by this repo is implemented."
        )
    )
    parser.add_argument(
        "model_args",
        help=(
            "Model arguments string, e.g. "
            "model_name=...,revision=...,batch_size=1,generation_size=768,..."
        ),
    )
    parser.add_argument(
        "tasks",
        help="Comma-separated LightEval task spec, e.g. 'lighteval|math_500|0|0'",
    )
    parser.add_argument(
        "--output-dir",
        default="var/artifacts/lighteval",
        help="Output directory for evaluation results (default: var/artifacts/lighteval).",
    )
    parser.add_argument(
        "--save-details",
        action="store_true",
        help="Save per-sample details (forwarded to LightEval).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of samples per task.",
    )
    parser.add_argument(
        "--job-id",
        type=int,
        default=0,
        help="Optional job id for logging (forwarded to LightEval).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # Install debug hooks before importing the main entrypoint so that any
    # Pipeline construction uses the wrapped methods.
    _install_pipeline_debug_hooks()

    from lighteval.main_accelerate import accelerate  # type: ignore[import]

    print(
        "[debug-lighteval] Starting accelerate() "
        f"with max_samples={args.max_samples}, output_dir={args.output_dir}",
        flush=True,
    )
    results = accelerate(
        model_args=args.model_args,
        tasks=args.tasks,
        output_dir=args.output_dir,
        save_details=bool(args.save_details),
        max_samples=args.max_samples,
        job_id=args.job_id,
    )
    print("[debug-lighteval] accelerate() completed; results keys:", list(results.keys()), flush=True)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
