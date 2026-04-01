"""Trainer callback for official SEED paper-style eval against the live vLLM server."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

try:
    from transformers.trainer_callback import TrainerCallback
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional in tests
    TrainerCallback = object  # type: ignore[assignment]

from maxent_grpo.seed_paper_eval import (
    build_seed_paper_eval_payload,
    build_step0_wandb_payload,
    default_python_executable,
    default_workspace_dir,
)

LOG = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _slugify(raw: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", raw.strip())
    text = text.strip("-")
    return text or "unnamed-run"


def _resolve_results_root(training_args: Any) -> Path:
    configured = getattr(training_args, "seed_paper_eval_results_dir", None)
    if configured:
        return Path(str(configured)).expanduser().absolute()
    return _repo_root() / "var" / "artifacts" / "seed_paper_eval" / "live"


def _resolve_workspace_dir(training_args: Any) -> Path:
    configured = getattr(training_args, "seed_paper_eval_workspace_dir", None)
    if configured:
        return Path(str(configured)).expanduser().absolute()
    return default_workspace_dir(_repo_root())


def _resolve_step0_results_dir(training_args: Any) -> Path | None:
    configured = getattr(training_args, "seed_paper_eval_step0_results_dir", None)
    if configured:
        return Path(str(configured)).expanduser().absolute()
    env_value = os.environ.get("MAXENT_STEP0_PAPER_EVAL_RESULTS_DIR")
    if env_value and env_value.strip():
        return Path(env_value.strip()).expanduser().absolute()
    return None


def _resolve_python(training_args: Any) -> Path:
    configured = getattr(training_args, "seed_paper_eval_python", None)
    if configured:
        return Path(str(configured)).expanduser().absolute()
    return default_python_executable(_repo_root())


def _resolve_vllm_url() -> str | None:
    for key in ("MAXENT_VLLM_URL", "VLLM_URL"):
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()
    return None


def _env_int(keys: tuple[str, ...]) -> int | None:
    for key in keys:
        raw = os.environ.get(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        try:
            return int(text)
        except (TypeError, ValueError):
            continue
    return None


def _resolve_model_name_from_recipe_env() -> str:
    for key in ("GRPO_RECIPE_USED", "GRPO_RECIPE"):
        recipe_path = os.environ.get(key)
        if not recipe_path:
            continue
        try:
            content = Path(recipe_path).read_text(encoding="utf-8")
        except OSError:
            continue
        match = re.search(r"^model_name_or_path:\s*(.+?)\s*$", content, re.MULTILINE)
        if not match:
            continue
        value = match.group(1).strip().strip("'").strip('"')
        if value:
            return value
    return ""


def _resolve_model_name(training_args: Any) -> str:
    for key in ("model_name_or_path", "hub_model_id", "model_id"):
        value = getattr(training_args, key, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = _resolve_model_name_from_recipe_env()
    if value:
        return value
    for key in (
        "SEED_PAPER_EVAL_MODEL_NAME",
        "MAXENT_MODEL",
        "GRPO_MODEL",
        "MODEL",
    ):
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()
    return ""


def build_live_seed_paper_eval_command(
    training_args: Any,
    *,
    step: int,
) -> tuple[list[str], Path]:
    repo_root = _repo_root()
    run_name = str(
        getattr(training_args, "run_name", None)
        or getattr(training_args, "output_dir", None)
        or "unnamed-run"
    )
    results_dir = (
        _resolve_results_root(training_args)
        / _slugify(run_name)
        / f"step-{int(step):06d}"
    )
    command = [
        str(_resolve_python(training_args)),
        str(repo_root / "tools" / "seed_paper_eval.py"),
        "--model-name",
        _resolve_model_name(training_args),
        "--workspace-dir",
        str(_resolve_workspace_dir(training_args)),
        "--results-dir",
        str(results_dir),
        "--vllm-url",
        str(_resolve_vllm_url() or ""),
    ]
    template = getattr(training_args, "seed_paper_eval_template", None)
    if template:
        command.extend(["--template", str(template)])
    tasks = getattr(training_args, "seed_paper_eval_tasks", None)
    if tasks:
        command.extend(["--tasks", str(tasks)])
    max_test = getattr(training_args, "seed_paper_eval_max_test", None)
    if max_test is not None:
        command.extend(["--max-test", str(int(max_test))])
    batch_size = getattr(training_args, "seed_paper_eval_vllm_batch_size", None)
    if batch_size is not None:
        command.extend(["--vllm-batch-size", str(int(batch_size))])
    if bool(getattr(training_args, "seed_paper_reward_fast", False)):
        command.append("--seed-paper-reward-fast")
    if bool(getattr(training_args, "seed_paper_eval_pass_at_8_enabled", False)):
        command.append("--pass-at-8")
        pass_at_8_samples = getattr(
            training_args,
            "seed_paper_eval_pass_at_8_samples",
            None,
        )
        if pass_at_8_samples is not None:
            command.extend(["--pass-at-8-samples", str(int(pass_at_8_samples))])
        pass_at_8_temperature = getattr(
            training_args,
            "seed_paper_eval_pass_at_8_temperature",
            None,
        )
        if pass_at_8_temperature is not None:
            command.extend(
                ["--pass-at-8-temperature", str(float(pass_at_8_temperature))]
            )
        pass_at_8_top_p = getattr(
            training_args,
            "seed_paper_eval_pass_at_8_top_p",
            None,
        )
        if pass_at_8_top_p is not None:
            command.extend(["--pass-at-8-top-p", str(float(pass_at_8_top_p))])
    return command, results_dir


def _resolve_process_rank(training_args: Any) -> int:
    value = _env_int(("RANK", "SLURM_PROCID"))
    if value is not None:
        return max(0, value)
    raw = getattr(training_args, "process_index", None)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def _resolve_world_size(training_args: Any) -> int:
    value = _env_int(("WORLD_SIZE", "SLURM_NTASKS"))
    if value is not None:
        return max(1, value)
    raw = getattr(training_args, "world_size", None)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


def _coordination_dir(training_args: Any, *, step: int) -> Path:
    _, results_dir = build_live_seed_paper_eval_command(training_args, step=step)
    return results_dir / "_coord"


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _wait_for_rank_arrivals(
    coord_dir: Path,
    *,
    world_size: int,
    timeout_s: int,
) -> None:
    deadline = time.monotonic() + max(1, timeout_s)
    while time.monotonic() < deadline:
        arrivals = list(coord_dir.glob("arrived-rank*.json"))
        if len(arrivals) >= world_size:
            return
        time.sleep(1.0)
    raise TimeoutError(
        f"Timed out waiting for {world_size} ranks to enter live SEED eval under {coord_dir}"
    )


def _mark_rank_release(
    coord_dir: Path,
    *,
    rank: int,
    world_size: int,
    step: int,
) -> None:
    release_path = coord_dir / f"released-rank{rank:05d}.json"
    _write_json_atomic(
        release_path,
        {"rank": rank, "world_size": world_size, "pid": os.getpid(), "step": int(step)},
    )


def _wait_for_rank_releases(
    coord_dir: Path,
    *,
    world_size: int,
    timeout_s: int,
) -> None:
    deadline = time.monotonic() + max(1, timeout_s)
    while time.monotonic() < deadline:
        releases = list(coord_dir.glob("released-rank*.json"))
        if len(releases) >= world_size:
            return
        time.sleep(1.0)
    raise TimeoutError(
        f"Timed out waiting for {world_size} ranks to leave live SEED eval under {coord_dir}"
    )


def _wait_for_result_payload(
    result_path: Path,
    *,
    timeout_s: int,
) -> dict[str, object]:
    deadline = time.monotonic() + max(1, timeout_s)
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if result_path.exists():
            try:
                return json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                last_error = exc
        time.sleep(1.0)
    if last_error is not None:
        raise TimeoutError(
            f"Timed out waiting for a valid live SEED eval result payload at {result_path}: {last_error}"
        ) from last_error
    raise TimeoutError(f"Timed out waiting for live SEED eval result payload at {result_path}")


def _latest_summary_path(results_dir: Path) -> Path | None:
    summaries = sorted(results_dir.glob("*.summary.json"))
    return summaries[-1] if summaries else None


def _wandb_run() -> Any | None:
    try:
        import wandb
    except ImportError:  # pragma: no cover - optional dependency
        return None
    return getattr(wandb, "run", None)


def _log_summary_to_wandb(
    *,
    summary: dict[str, object],
    summary_path: Path,
    step: int,
) -> None:
    run = _wandb_run()
    if run is None:
        return
    payload: dict[str, object] = {}
    for prefix in ("seed_paper_eval_live", "paper_eval"):
        _define_metric_axis(run, prefix=prefix)
        payload.update(build_seed_paper_eval_payload(summary, prefix=prefix))
        payload[f"{prefix}/training_step"] = int(step)
        payload[f"{prefix}/ok"] = 1.0
    if payload:
        run.log(payload, commit=True)
        for key, value in payload.items():
            run.summary[key] = value
    for prefix in ("seed_paper_eval_live", "paper_eval"):
        run.summary[f"{prefix}/latest_summary_path"] = str(summary_path)
        run.summary[f"{prefix}/latest_step"] = int(step)
        run.summary[f"{prefix}/status"] = "ok"
    run.summary["paper_eval/source"] = "live"
    warning = summary.get("process_warning")
    if warning is not None:
        run.summary["seed_paper_eval_live/process_warning"] = str(warning)
        run.summary["paper_eval/process_warning"] = str(warning)


def _log_step0_summary_to_wandb(
    *,
    summary: dict[str, object],
    summary_path: Path,
) -> bool:
    run = _wandb_run()
    if run is None:
        return False
    payload: dict[str, object] = {}
    _define_metric_axis(run, prefix="step0_paper_eval")
    _define_metric_axis(run, prefix="paper_eval")
    payload.update(build_step0_wandb_payload(summary))
    payload.update(build_seed_paper_eval_payload(summary, prefix="paper_eval"))
    payload["step0_paper_eval/training_step"] = 0
    payload["step0_paper_eval/ok"] = 1.0
    payload["paper_eval/training_step"] = 0
    payload["paper_eval/ok"] = 1.0
    if payload:
        run.log(payload, commit=True)
        for key, value in payload.items():
            run.summary[key] = value
    run.summary["step0_paper_eval/summary_path"] = str(summary_path)
    run.summary["step0_paper_eval/latest_step"] = 0
    run.summary["step0_paper_eval/status"] = "ok"
    run.summary["paper_eval/summary_path"] = str(summary_path)
    run.summary["paper_eval/latest_step"] = 0
    run.summary["paper_eval/status"] = "ok"
    run.summary["paper_eval/source"] = "step0"
    warning = summary.get("process_warning")
    if warning is not None:
        run.summary["step0_paper_eval/process_warning"] = str(warning)
        run.summary["paper_eval/process_warning"] = str(warning)
    comparison = summary.get("expected_comparison")
    if isinstance(comparison, dict) and not bool(comparison.get("ok")):
        run.summary["step0_paper_eval/status"] = "expected_mismatch"
        run.summary["paper_eval/status"] = "expected_mismatch"
    return True


def _define_metric_axis(run: Any, *, prefix: str) -> None:
    define_metric = getattr(run, "define_metric", None)
    if not callable(define_metric):
        return
    step_key = f"{prefix}/training_step"
    define_metric(step_key)
    define_metric(f"{prefix}/*", step_metric=step_key)


def _sync_step0_summary_to_current_run(training_args: Any) -> bool:
    results_dir = _resolve_step0_results_dir(training_args)
    if results_dir is None:
        return False
    summary_path = _latest_summary_path(results_dir)
    if summary_path is None:
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return _log_step0_summary_to_wandb(summary=summary, summary_path=summary_path)


class SeedPaperEvalCallback(TrainerCallback):
    """Run the official SEED paper-style eval against the live vLLM server."""

    def __init__(self, training_args: Any) -> None:
        self.training_args = training_args
        self._seen_steps: set[int] = set()

    def _built_in_eval_enabled(self) -> bool:
        if not bool(getattr(self.training_args, "do_eval", False)):
            return False
        eval_strategy = getattr(self.training_args, "eval_strategy", None)
        if eval_strategy is None:
            return False
        text = str(eval_strategy).strip().lower()
        return text not in {"", "no", "none"}

    def _eval_steps(self) -> int:
        raw = getattr(self.training_args, "eval_steps", 0)
        try:
            return max(0, int(raw or 0))
        except (TypeError, ValueError):
            return 0

    def _run_live_eval_once(self, step: int) -> tuple[bool, str | None]:
        if step in self._seen_steps:
            return True, None
        vllm_url = _resolve_vllm_url()
        if not vllm_url:
            LOG.warning(
                "Skipping live SEED paper eval at step %s because no vLLM URL is available.",
                step,
            )
            return True, None
        command, results_dir = build_live_seed_paper_eval_command(
            self.training_args,
            step=step,
        )
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "0"
        env["WANDB_DISABLED"] = "true"
        timeout_s = int(
            getattr(self.training_args, "seed_paper_eval_timeout_s", 14400) or 14400
        )
        LOG.info(
            "Running live SEED paper eval | step=%s | results_dir=%s | cmd=%s",
            step,
            results_dir,
            " ".join(command),
        )
        try:
            subprocess.run(
                command,
                check=True,
                cwd=str(_repo_root()),
                env=env,
                text=True,
                timeout=timeout_s,
            )
            summary_path = _latest_summary_path(results_dir)
            if summary_path is None:
                raise RuntimeError(f"No summary JSON was produced under {results_dir}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            _log_summary_to_wandb(summary=summary, summary_path=summary_path, step=step)
            self._seen_steps.add(step)
            return True, None
        except Exception as exc:
            LOG.warning("Live SEED paper eval failed at step %s: %s", step, exc)
            run = _wandb_run()
            if run is not None:
                for prefix in ("seed_paper_eval_live", "paper_eval"):
                    _define_metric_axis(run, prefix=prefix)
                run.log(
                    {
                        "seed_paper_eval_live/training_step": int(step),
                        "seed_paper_eval_live/ok": 0.0,
                        "paper_eval/training_step": int(step),
                        "paper_eval/ok": 0.0,
                    },
                    commit=True,
                )
                run.summary["seed_paper_eval_live/last_error"] = str(exc)
                run.summary["seed_paper_eval_live/latest_step"] = int(step)
                run.summary["seed_paper_eval_live/status"] = "failed"
                run.summary["paper_eval/last_error"] = str(exc)
                run.summary["paper_eval/latest_step"] = int(step)
                run.summary["paper_eval/status"] = "failed"
                run.summary["paper_eval/source"] = "live"
            return False, str(exc)

    def _run_live_eval(self, step: int) -> None:
        rank = _resolve_process_rank(self.training_args)
        world_size = _resolve_world_size(self.training_args)
        timeout_s = int(
            getattr(self.training_args, "seed_paper_eval_timeout_s", 14400) or 14400
        )
        fail_on_error = bool(
            getattr(self.training_args, "seed_paper_eval_fail_on_error", False)
        )
        if world_size <= 1:
            ok, error = self._run_live_eval_once(step)
            if not ok and fail_on_error:
                raise RuntimeError(
                    f"Live SEED paper eval failed at step {step}: {error}"
                )
            return

        coord_dir = _coordination_dir(self.training_args, step=step)
        coord_dir.mkdir(parents=True, exist_ok=True)
        arrival_path = coord_dir / f"arrived-rank{rank:05d}.json"
        _write_json_atomic(
            arrival_path,
            {"rank": rank, "world_size": world_size, "pid": os.getpid(), "step": int(step)},
        )
        result_path = coord_dir / "result.json"

        ok = False
        error: str | None = None
        if rank == 0:
            _wait_for_rank_arrivals(coord_dir, world_size=world_size, timeout_s=timeout_s)
            ok, error = self._run_live_eval_once(step)
            _write_json_atomic(
                result_path,
                {
                    "ok": bool(ok),
                    "error": error,
                    "rank": rank,
                    "step": int(step),
                    "world_size": world_size,
                },
            )
        else:
            payload = _wait_for_result_payload(result_path, timeout_s=timeout_s + 300)
            ok = bool(payload.get("ok"))
            error_obj = payload.get("error")
            error = str(error_obj) if error_obj is not None else None

        # Hold every rank here until all peers have fully observed the eval result
        # and are ready to return to the trainer. Without this exit rendezvous,
        # rank 0 can enter checkpoint save collectives while other ranks are still
        # unwinding the eval callback, which can deadlock NCCL.
        _mark_rank_release(coord_dir, rank=rank, world_size=world_size, step=step)
        _wait_for_rank_releases(coord_dir, world_size=world_size, timeout_s=timeout_s + 300)
        if ok:
            self._seen_steps.add(step)
            return
        if fail_on_error:
            raise RuntimeError(
                f"Live SEED paper eval failed at step {step}: {error}"
            )

    def on_train_begin(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> Any:
        _ = args
        _ = kwargs
        if not bool(getattr(self.training_args, "seed_paper_eval_enabled", False)):
            return control
        if self._built_in_eval_enabled():
            return control
        trigger_on_start = getattr(
            self.training_args,
            "seed_paper_eval_on_start",
            getattr(self.training_args, "eval_on_start", False),
        )
        if bool(trigger_on_start):
            step = int(getattr(state, "global_step", 0) or 0)
            synced_step0 = False
            if step == 0:
                step0_results_dir = _resolve_step0_results_dir(self.training_args)
                if step0_results_dir is not None and _latest_summary_path(step0_results_dir) is not None:
                    synced_step0 = True
                    self._seen_steps.add(0)
                    if bool(getattr(state, "is_world_process_zero", True)):
                        _sync_step0_summary_to_current_run(self.training_args)
            if not synced_step0:
                self._run_live_eval(step)
        return control

    def on_step_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> Any:
        _ = args
        _ = kwargs
        if not bool(getattr(self.training_args, "seed_paper_eval_enabled", False)):
            return control
        if self._built_in_eval_enabled():
            return control
        step = int(getattr(state, "global_step", 0) or 0)
        eval_steps = self._eval_steps()
        if step > 0 and eval_steps > 0 and step % eval_steps == 0:
            self._run_live_eval(step)
        return control

    def on_evaluate(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> Any:
        _ = args
        _ = kwargs
        if not bool(getattr(self.training_args, "seed_paper_eval_enabled", False)):
            return control
        if not self._built_in_eval_enabled():
            return control
        step = int(getattr(state, "global_step", 0) or 0)
        self._run_live_eval(step)
        return control

    def on_train_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> Any:
        _ = args
        _ = kwargs
        if not bool(getattr(self.training_args, "seed_paper_eval_enabled", False)):
            return control
        if self._built_in_eval_enabled():
            return control
        step = int(getattr(state, "global_step", 0) or 0)
        if step > 0:
            self._run_live_eval(step)
        return control


__all__ = [
    "SeedPaperEvalCallback",
    "build_live_seed_paper_eval_command",
]
