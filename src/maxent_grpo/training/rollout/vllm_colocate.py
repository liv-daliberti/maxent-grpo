"""In-process (colocated) vLLM generation helpers for the custom loop."""

from __future__ import annotations

import atexit
import faulthandler
import inspect
import logging
import multiprocessing as mp
import os
import socket
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

from maxent_grpo.patches.vllm import VLLMLogprobResult
from maxent_grpo.training.runtime.prompts import _truncate_prompt, PROMPT_CHAR_LIMIT
from maxent_grpo.utils.imports import optional_import

LOG = logging.getLogger(__name__)


def _parse_log_level(raw: str) -> Optional[int]:
    value = raw.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return logging._nameToLevel.get(value.upper())


def _configure_colocate_logging() -> None:
    raw = os.getenv("MAXENT_VLLM_COLOCATE_LOG_LEVEL")
    if raw:
        level = _parse_log_level(raw)
        if isinstance(level, int):
            LOG.setLevel(level)
        else:
            LOG.warning("Invalid MAXENT_VLLM_COLOCATE_LOG_LEVEL=%r; ignoring.", raw)
        return
    # Default to warnings-only to avoid excessive colocate logging once stable.
    LOG.setLevel(logging.WARNING)


_configure_colocate_logging()


def _env_float(name: str) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        LOG.debug("Invalid %s=%r; ignoring.", name, raw)
        return None


def _env_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        LOG.debug("Invalid %s=%r; ignoring.", name, raw)
        return None


def _filter_kwargs(callable_obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values()):
        return kwargs
    return {key: val for key, val in kwargs.items() if key in sig.parameters}


def _resolve_model_id(ctx: Any) -> Optional[str]:
    for key in (
        "vllm_model_id",
        "served_model_id",
        "model_name",
        "model_id",
        "hub_model_id",
        "model_name_or_path",
    ):
        value = getattr(ctx, key, None)
        if isinstance(value, str) and value:
            return value
    model = getattr(ctx, "model", None)
    if model is not None:
        name = getattr(model, "name_or_path", None)
        if isinstance(name, str) and name:
            return name
        cfg = getattr(model, "config", None)
        cfg_name = getattr(cfg, "name_or_path", None) or getattr(cfg, "_name_or_path", None)
        if isinstance(cfg_name, str) and cfg_name:
            return cfg_name
    training_args = getattr(ctx, "training_args", None)
    if training_args is not None:
        for key in ("model_name_or_path", "hub_model_id", "model_id"):
            value = getattr(training_args, key, None)
            if isinstance(value, str) and value:
                return value
    return None


def _env_bool(name: str) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    LOG.debug("Invalid %s=%r; ignoring.", name, raw)
    return None


def _resolve_dtype(ctx: Any) -> Optional[str]:
    training_args = getattr(ctx, "training_args", None)
    if training_args is None:
        return None
    if getattr(training_args, "fp16", False):
        return "float16"
    if getattr(training_args, "bf16", False):
        return "bfloat16"
    return None


def _init_mode() -> str:
    raw = os.getenv("MAXENT_VLLM_COLOCATE_INIT_MODE", "").strip().lower()
    if raw in {"subprocess", "process", "proc"}:
        return "subprocess"
    if raw in {"thread", "async", "background"}:
        return "thread"
    if raw in {"blocking", "sync", "foreground"}:
        return "blocking"
    return "auto"


def _dist_initialized() -> bool:
    try:
        import torch

        dist = getattr(torch, "distributed", None)
        return bool(
            dist
            and hasattr(dist, "is_available")
            and hasattr(dist, "is_initialized")
            and dist.is_available()
            and dist.is_initialized()
        )
    except Exception:
        return False


def _log_env_snapshot(keys: List[str]) -> None:
    items: List[str] = []
    for key in keys:
        value = os.getenv(key)
        if value is None:
            continue
        items.append(f"{key}={value}")
    if items:
        LOG.info("vLLM colocate env snapshot | %s", " ".join(items))


def _log_torch_snapshot() -> None:
    try:
        import torch
    except Exception as exc:
        LOG.info("vLLM colocate torch snapshot skipped | error=%s", exc)
        return
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False
    device_idx = None
    device_name = None
    total_mem = None
    free_mem = None
    try:
        if cuda_available:
            device_idx = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device_idx)
            if hasattr(torch.cuda, "mem_get_info"):
                free_mem, total_mem = torch.cuda.mem_get_info(device_idx)
    except Exception:
        pass
    LOG.info(
        "vLLM colocate torch | cuda_available=%s device=%s name=%s free_mem=%s total_mem=%s",
        cuda_available,
        device_idx,
        device_name,
        free_mem,
        total_mem,
    )


def _log_process_snapshot() -> None:
    try:
        pid = os.getpid()
    except Exception:
        pid = None
    try:
        host = socket.gethostname()
    except Exception:
        host = None
    LOG.info(
        "vLLM colocate process | pid=%s thread=%s host=%s",
        pid,
        threading.current_thread().name,
        host,
    )


def _log_runtime_snapshot() -> None:
    try:
        python_version = sys.version.replace("\n", " ")
    except Exception:
        python_version = None
    LOG.info(
        "vLLM colocate runtime | python=%s executable=%s",
        python_version,
        getattr(sys, "executable", None),
    )


def _extract_logprob_sequence(raw: Any) -> Optional[List[float]]:
    if raw is None:
        return None
    if isinstance(raw, list):
        cleaned: List[float] = []
        for entry in raw:
            if entry is None:
                continue
            if isinstance(entry, (int, float)):
                cleaned.append(float(entry))
                continue
            if isinstance(entry, dict) and entry:
                val = next(iter(entry.values()))
            else:
                val = entry
            if isinstance(val, dict):
                val = val.get("logprob", val.get("log_prob"))
            elif hasattr(val, "logprob"):
                val = getattr(val, "logprob", None)
            if isinstance(val, (int, float)):
                cleaned.append(float(val))
        return cleaned if cleaned else None
    return None


def _sum_logprobs(values: Optional[List[float]]) -> Optional[float]:
    if not values:
        return None
    total = 0.0
    for val in values:
        total += float(val)
    return total


def _coerce_logprob_payload(
    payload: Optional[List[List[Optional[Dict[str, Any]]]]],
) -> Optional[List[List[Optional[VLLMLogprobResult]]]]:
    if payload is None:
        return None
    converted: List[List[Optional[VLLMLogprobResult]]] = []
    for group in payload:
        group_converted: List[Optional[VLLMLogprobResult]] = []
        for entry in group:
            if entry is None:
                group_converted.append(None)
                continue
            if isinstance(entry, VLLMLogprobResult):
                group_converted.append(entry)
                continue
            if isinstance(entry, dict):
                group_converted.append(
                    VLLMLogprobResult(
                        logprob_sum=entry.get("logprob_sum"),
                        token_count=entry.get("token_count"),
                        token_logprobs=entry.get("token_logprobs"),
                        raw_output=entry.get("raw_output"),
                    )
                )
                continue
            group_converted.append(None)
        converted.append(group_converted)
    return converted


def _outputs_to_payload(
    outputs: Any, want_logprobs: bool
) -> Tuple[List[List[str]], Optional[List[List[Optional[Dict[str, Any]]]]]]:
    grouped: List[List[str]] = []
    grouped_meta: Optional[List[List[Optional[Dict[str, Any]]]]] = [] if want_logprobs else None
    for output in outputs:
        seqs = getattr(output, "outputs", None) or []
        group: List[str] = []
        meta_group: Optional[List[Optional[Dict[str, Any]]]] = [] if grouped_meta is not None else None
        for seq in seqs:
            text = getattr(seq, "text", None)
            if text is None:
                text = str(getattr(seq, "text", ""))
            group.append(text)
            if meta_group is not None:
                logprob_sum = getattr(seq, "cumulative_logprob", None)
                token_ids = getattr(seq, "token_ids", None) or getattr(
                    seq, "output_token_ids", None
                )
                token_count = len(token_ids) if token_ids is not None else None
                token_logprobs = _extract_logprob_sequence(getattr(seq, "logprobs", None))
                if logprob_sum is None:
                    logprob_sum = _sum_logprobs(token_logprobs)
                meta_group.append(
                    {
                        "logprob_sum": logprob_sum,
                        "token_count": token_count,
                        "token_logprobs": token_logprobs,
                    }
                )
        grouped.append(group)
        if meta_group is not None:
            grouped_meta.append(meta_group)
    return grouped, grouped_meta


def _vllm_colocate_worker(conn: Any) -> None:
    """Subprocess worker for vLLM colocate init/generate."""
    try:
        def _send_log(message: str) -> None:
            try:
                conn.send({"type": "log", "message": message})
            except Exception:
                pass

        try:
            _send_log(
                f"worker boot | pid={os.getpid()} python={sys.version.replace(os.linesep, ' ')}"
            )
        except Exception:
            pass
        init_msg = conn.recv()
        if not isinstance(init_msg, dict) or init_msg.get("type") != "init":
            conn.send({"ok": False, "error": "Invalid init payload"})
            return
        model_id = init_msg.get("model_id")
        llm_kwargs = init_msg.get("llm_kwargs") or {}
        request_logprobs_default = bool(init_msg.get("request_logprobs", False))
        _send_log(
            f"worker init payload | model_id={model_id} request_logprobs={request_logprobs_default} llm_kwargs={llm_kwargs}"
        )

        # Isolate the worker from the training process' distributed context.
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["SLURM_LOCALID"] = "0"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        os.environ["VLLM_DP_SIZE"] = "1"
        os.environ["VLLM_DP_RANK"] = "0"
        os.environ["VLLM_DP_RANK_LOCAL"] = "0"
        _send_log("worker env override | RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 SLURM_LOCALID=0")
        master_addr = os.getenv("MAXENT_VLLM_COLOCATE_MASTER_ADDR") or "127.0.0.1"
        master_port = os.getenv("MAXENT_VLLM_COLOCATE_MASTER_PORT")
        if not master_port:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind((master_addr, 0))
                master_port = str(sock.getsockname()[1])
            except Exception:
                master_port = "29501"
            finally:
                try:
                    sock.close()
                except Exception:
                    pass
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["VLLM_DP_MASTER_IP"] = master_addr
        os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)
        _send_log(
            f"worker dist env | MASTER_ADDR={master_addr} MASTER_PORT={master_port} LOCAL_WORLD_SIZE=1"
        )
        preinit = _env_bool("MAXENT_VLLM_COLOCATE_PREINIT_DIST")
        if preinit is None:
            preinit = True
        if preinit:
            try:
                import torch

                dist = getattr(torch, "distributed", None)
                if dist is not None and dist.is_available() and not dist.is_initialized():
                    init_method = os.getenv("MAXENT_VLLM_COLOCATE_INIT_METHOD")
                    if not init_method:
                        store_dir = os.getenv("MAXENT_VLLM_COLOCATE_STORE_DIR") or tempfile.gettempdir()
                        fd, path = tempfile.mkstemp(prefix="vllm-init-", dir=store_dir)
                        os.close(fd)
                        init_method = f"file://{path}"
                    _send_log(
                        f"worker dist preinit | method={init_method} backend=nccl"
                    )
                    dist.init_process_group(
                        backend="nccl",
                        init_method=init_method,
                        world_size=1,
                        rank=0,
                    )
                    _send_log("worker dist preinit done | initialized=True")
            except Exception as exc:
                _send_log(f"worker dist preinit failed | error={exc}")

        # If a specific CUDA device was requested, remap visibility so vLLM
        # only sees that GPU (and use cuda:0 inside the worker).
        device = llm_kwargs.get("device")
        if isinstance(device, str) and device.startswith("cuda:"):
            idx = device.split(":", 1)[1]
            if idx.isdigit():
                os.environ["CUDA_VISIBLE_DEVICES"] = idx
                llm_kwargs = dict(llm_kwargs)
                llm_kwargs["device"] = "cuda:0"
                _send_log(f"worker CUDA remap | CUDA_VISIBLE_DEVICES={idx} device=cuda:0")

        _send_log("worker import vllm start")
        vllm_mod = optional_import("vllm")
        if vllm_mod is None:
            raise RuntimeError("vllm is not installed")
        llm_cls = getattr(vllm_mod, "LLM", None)
        if llm_cls is None:
            raise RuntimeError("vllm.LLM is unavailable")
        _send_log(
            f"worker vllm module | version={getattr(vllm_mod, '__version__', None)} path={getattr(vllm_mod, '__file__', None)}"
        )
        llm_kwargs = _filter_kwargs(llm_cls, llm_kwargs)
        device = llm_kwargs.get("device")
        if isinstance(device, str) and device.startswith("cuda:"):
            try:
                import torch

                torch.cuda.set_device(int(device.split(":")[1]))
                _send_log(f"worker torch cuda set_device | device={device}")
            except Exception:
                pass
        try:
            import torch

            torch_version = getattr(torch, "__version__", None)
            cuda_version = getattr(torch, "version", None)
            cuda_version = getattr(cuda_version, "cuda", None) if cuda_version is not None else None
            cudnn_version = None
            if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
                cudnn_version = getattr(torch.backends.cudnn, "version", None)
                if callable(cudnn_version):
                    cudnn_version = cudnn_version()
            _send_log(
                f"worker torch | version={torch_version} cuda_version={cuda_version} cudnn={cudnn_version}"
            )
        except Exception:
            pass
        try:
            import torch

            if torch.cuda.is_available() and hasattr(torch.cuda, "mem_get_info"):
                free_mem, total_mem = torch.cuda.mem_get_info()
                _send_log(f"worker cuda mem pre-init | free_mem={free_mem} total_mem={total_mem}")
        except Exception:
            pass
        _send_log(f"worker LLM init start | model={model_id} kwargs={llm_kwargs}")
        stack_interval = _env_float("MAXENT_VLLM_COLOCATE_INIT_STACK_S")
        watchdog_stop = threading.Event()

        def _stack_watchdog() -> None:
            if not stack_interval or stack_interval <= 0:
                return
            while not watchdog_stop.wait(stack_interval):
                _send_log("worker LLM init still running; dumping stack traces")
                try:
                    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                except Exception:
                    pass

        watchdog = None
        if stack_interval and stack_interval > 0:
            watchdog = threading.Thread(
                target=_stack_watchdog, name="vllm-colocate-init-watchdog", daemon=True
            )
            watchdog.start()
        started = time.time()
        llm = llm_cls(model=model_id, **llm_kwargs)
        watchdog_stop.set()
        if watchdog is not None:
            try:
                watchdog.join(timeout=1.0)
            except Exception:
                pass
        elapsed = time.time() - started
        _send_log(f"worker LLM init done | elapsed_s={elapsed:.2f}")
        try:
            import torch

            if torch.cuda.is_available() and hasattr(torch.cuda, "mem_get_info"):
                free_mem, total_mem = torch.cuda.mem_get_info()
                _send_log(f"worker cuda mem post-init | free_mem={free_mem} total_mem={total_mem}")
        except Exception:
            pass
        params_cls = getattr(vllm_mod, "SamplingParams", None)
        if params_cls is None:
            raise RuntimeError("vllm.SamplingParams is unavailable")
        _send_log("worker SamplingParams available")
        conn.send({"ok": True})

        while True:
            msg = conn.recv()
            if not isinstance(msg, dict):
                continue
            if msg.get("type") == "shutdown":
                break
            if msg.get("type") != "generate":
                continue
            prompts = msg.get("prompts") or []
            params_kwargs = msg.get("params_kwargs") or {}
            request_logprobs = bool(msg.get("request_logprobs", request_logprobs_default))
            params_kwargs = _filter_kwargs(params_cls, params_kwargs)
            params = params_cls(**params_kwargs)
            outputs = llm.generate(prompts, params)
            grouped, grouped_meta = _outputs_to_payload(outputs, request_logprobs)
            conn.send({"ok": True, "grouped": grouped, "meta": grouped_meta})
    except Exception as exc:
        try:
            conn.send(
                {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


class ColocateVLLMEngine:
    """Lazy vLLM engine wrapper used for colocated generation."""

    def __init__(self, ctx: Any, fallback_generate: Any) -> None:
        self.ctx = ctx
        self._fallback_generate = fallback_generate
        self._llm: Any = None
        self._init_failed = False
        self._worker_proc: Optional[mp.Process] = None
        self._worker_conn: Optional[Any] = None

    def _resolve_init_mode(self) -> str:
        mode = _init_mode()
        if mode == "auto":
            return "subprocess" if _dist_initialized() else "thread"
        return mode

    def _configure_vllm_env(self) -> None:
        backend_override = os.getenv("MAXENT_VLLM_COLOCATE_ATTENTION_BACKEND")
        if backend_override:
            os.environ["VLLM_ATTENTION_BACKEND"] = backend_override
            LOG.info(
                "vLLM colocate attention backend override | backend=%s",
                backend_override,
            )
        if os.getenv("VLLM_ENABLE_V1_MULTIPROCESSING") is None:
            mp_override = _env_bool("MAXENT_VLLM_COLOCATE_V1_MULTIPROCESSING")
            if mp_override is None or not mp_override:
                os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
                LOG.info(
                    "Disabling vLLM V1 multiprocessing for colocate. "
                    "Set MAXENT_VLLM_COLOCATE_V1_MULTIPROCESSING=1 to re-enable."
                )
            else:
                os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
        if os.getenv("VLLM_USE_V1") is None:
            force_v0 = _env_bool("MAXENT_VLLM_COLOCATE_FORCE_V0")
            if force_v0:
                os.environ["VLLM_USE_V1"] = "0"
                LOG.info(
                    "Forcing vLLM V0 engine for colocate "
                    "(MAXENT_VLLM_COLOCATE_FORCE_V0=1)."
                )
        LOG.info(
            "vLLM colocate env | VLLM_USE_V1=%s VLLM_ENABLE_V1_MULTIPROCESSING=%s "
            "CUDA_VISIBLE_DEVICES=%s LOCAL_RANK=%s SLURM_LOCALID=%s",
            os.getenv("VLLM_USE_V1"),
            os.getenv("VLLM_ENABLE_V1_MULTIPROCESSING"),
            os.getenv("CUDA_VISIBLE_DEVICES"),
            os.getenv("LOCAL_RANK"),
            os.getenv("SLURM_LOCALID"),
        )

    def _build_llm_kwargs(self) -> Dict[str, Any]:
        llm_kwargs: Dict[str, Any] = {}
        dtype_override = _resolve_dtype(self.ctx)
        if dtype_override:
            llm_kwargs["dtype"] = dtype_override
        gpu_util = _env_float("MAXENT_VLLM_COLOCATE_GPU_UTIL")
        if gpu_util is not None:
            llm_kwargs["gpu_memory_utilization"] = gpu_util
        tp_size = _env_int("MAXENT_VLLM_COLOCATE_TP")
        if tp_size is not None:
            llm_kwargs["tensor_parallel_size"] = tp_size
        max_model_len = _env_int("MAXENT_VLLM_COLOCATE_MAX_MODEL_LEN")
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        enforce_eager = _env_bool("MAXENT_VLLM_COLOCATE_ENFORCE_EAGER")
        if enforce_eager is None:
            enforce_eager = True
        llm_kwargs["enforce_eager"] = enforce_eager
        device_override = os.getenv("MAXENT_VLLM_COLOCATE_DEVICE")
        if device_override:
            llm_kwargs["device"] = device_override
        else:
            local_rank = os.getenv("LOCAL_RANK") or os.getenv("SLURM_LOCALID")
            if local_rank is not None and str(local_rank).isdigit():
                llm_kwargs["device"] = f"cuda:{int(local_rank)}"
            else:
                try:
                    import torch

                    if torch.cuda.is_available():
                        llm_kwargs["device"] = f"cuda:{torch.cuda.current_device()}"
                except Exception:
                    pass
        trust_remote = getattr(self.ctx, "trust_remote_code", None)
        if trust_remote is None:
            training_args = getattr(self.ctx, "training_args", None)
            trust_remote = getattr(training_args, "trust_remote_code", None)
        if trust_remote is not None:
            llm_kwargs["trust_remote_code"] = bool(trust_remote)
        return llm_kwargs

    def _build_llm_spec(self) -> Tuple[str, Dict[str, Any]]:
        LOG.info("vLLM colocate build spec start")
        _log_process_snapshot()
        _log_runtime_snapshot()
        _log_env_snapshot(
            [
                "CUDA_VISIBLE_DEVICES",
                "CUDA_DEVICE_ORDER",
                "LOCAL_RANK",
                "RANK",
                "WORLD_SIZE",
                "SLURM_LOCALID",
                "SLURM_PROCID",
                "SLURM_NTASKS",
                "MASTER_ADDR",
                "MASTER_PORT",
                "NCCL_P2P_DISABLE",
                "NCCL_IB_DISABLE",
                "NCCL_SOCKET_IFNAME",
                "NCCL_DEBUG",
                "NCCL_DEBUG_SUBSYS",
                "VLLM_USE_V1",
                "VLLM_ENABLE_V1_MULTIPROCESSING",
                "VLLM_ATTENTION_BACKEND",
                "VLLM_LOGGING_LEVEL",
                "MAXENT_VLLM_COLOCATE_GPU_UTIL",
                "MAXENT_VLLM_COLOCATE_TP",
                "MAXENT_VLLM_COLOCATE_MAX_MODEL_LEN",
                "MAXENT_VLLM_COLOCATE_ENFORCE_EAGER",
                "MAXENT_VLLM_COLOCATE_FORCE_V0",
                "MAXENT_VLLM_COLOCATE_DEVICE",
                "MAXENT_VLLM_COLOCATE_INIT_TIMEOUT_S",
                "MAXENT_VLLM_COLOCATE_INIT_MODE",
                "MAXENT_VLLM_COLOCATE_INIT_HEARTBEAT_S",
                "MAXENT_VLLM_COLOCATE_INIT_STACK_S",
                "MAXENT_VLLM_COLOCATE_MASTER_ADDR",
                "MAXENT_VLLM_COLOCATE_MASTER_PORT",
                "MAXENT_VLLM_COLOCATE_PREINIT_DIST",
                "MAXENT_VLLM_COLOCATE_INIT_METHOD",
                "MAXENT_VLLM_COLOCATE_STORE_DIR",
                "MAXENT_VLLM_COLOCATE_ATTENTION_BACKEND",
                "MAXENT_VLLM_COLOCATE_V1_MULTIPROCESSING",
            ]
        )
        _log_torch_snapshot()
        self._configure_vllm_env()
        model_id = _resolve_model_id(self.ctx)
        if not model_id:
            raise RuntimeError("Unable to resolve model ID for vLLM colocate")
        LOG.info("vLLM colocate resolved model_id=%s", model_id)
        llm_kwargs = self._build_llm_kwargs()
        LOG.info("vLLM colocate llm_kwargs pre-filter | %s", llm_kwargs)
        LOG.info(
            "vLLM colocate device selection | device=%s torch_device=%s",
            llm_kwargs.get("device", "auto"),
            getattr(getattr(self, "ctx", None), "device", None),
        )
        LOG.info(
            "vLLM colocate compile mode | enforce_eager=%s",
            llm_kwargs.get("enforce_eager"),
        )
        LOG.info(
            "vLLM colocate init | model=%s | dtype=%s | tp=%s | gpu_util=%s",
            model_id,
            llm_kwargs.get("dtype", "default"),
            llm_kwargs.get("tensor_parallel_size", "auto"),
            llm_kwargs.get("gpu_memory_utilization", "default"),
        )
        return model_id, llm_kwargs

    def _build_llm(self) -> Any:
        model_id, llm_kwargs = self._build_llm_spec()
        LOG.info("vLLM colocate import vllm start")
        vllm_mod = optional_import("vllm")
        if vllm_mod is None:
            raise RuntimeError("vllm is not installed")
        LOG.info(
            "vLLM colocate vllm module | version=%s path=%s",
            getattr(vllm_mod, "__version__", None),
            getattr(vllm_mod, "__file__", None),
        )
        llm_cls = getattr(vllm_mod, "LLM", None)
        if llm_cls is None:
            raise RuntimeError("vllm.LLM is unavailable")
        pre_filter = dict(llm_kwargs)
        llm_kwargs = _filter_kwargs(llm_cls, llm_kwargs)
        removed = sorted(set(pre_filter) - set(llm_kwargs))
        if removed:
            LOG.info("vLLM colocate llm_kwargs filtered | removed=%s", removed)
        LOG.info("vLLM colocate LLM init start | model=%s kwargs=%s", model_id, llm_kwargs)
        started = time.time()
        llm = llm_cls(model=model_id, **llm_kwargs)
        elapsed = time.time() - started
        LOG.info("vLLM colocate LLM init done | elapsed_s=%.2f", elapsed)
        return llm

    def _shutdown_worker(self) -> None:
        conn = getattr(self, "_worker_conn", None)
        proc = getattr(self, "_worker_proc", None)
        self._worker_conn = None
        self._worker_proc = None
        if conn is not None:
            try:
                conn.send({"type": "shutdown"})
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
        if proc is not None:
            try:
                if proc.is_alive():
                    proc.terminate()
            except Exception:
                pass
            try:
                proc.join(timeout=5)
            except Exception:
                pass

    def _ensure_worker(self) -> None:
        if self._worker_conn is not None and self._worker_proc is not None:
            if self._worker_proc.is_alive():
                return
            self._shutdown_worker()
        if self._init_failed:
            raise RuntimeError("vLLM colocate initialization previously failed")

        timeout_s = _env_float("MAXENT_VLLM_COLOCATE_INIT_TIMEOUT_S")
        if timeout_s is None:
            timeout_s = 0.0
        LOG.info(
            "vLLM colocate init start | timeout_s=%.1f | mode=subprocess",
            float(timeout_s),
        )
        model_id, llm_kwargs = self._build_llm_spec()
        ctx = mp.get_context("spawn")
        try:
            start_method = ctx.get_start_method()
        except Exception:
            start_method = "spawn"
        LOG.info("vLLM colocate subprocess | start_method=%s", start_method)
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(
            target=_vllm_colocate_worker,
            args=(child_conn,),
            name="vllm-colocate-worker",
            daemon=True,
        )
        proc.start()
        LOG.info("vLLM colocate subprocess started | pid=%s", proc.pid)
        child_conn.close()
        init_payload = {
            "type": "init",
            "model_id": model_id,
            "llm_kwargs": llm_kwargs,
            "request_logprobs": bool(getattr(self.ctx, "vllm_request_logprobs", False)),
        }
        parent_conn.send(init_payload)
        start = time.time()
        heartbeat_s = _env_float("MAXENT_VLLM_COLOCATE_INIT_HEARTBEAT_S")
        if heartbeat_s is None or heartbeat_s <= 0:
            heartbeat_s = 30.0
        init_resp: Any = None
        while True:
            elapsed = time.time() - start
            if timeout_s and timeout_s > 0 and elapsed >= timeout_s:
                self._init_failed = True
                LOG.warning(
                    "vLLM colocate subprocess init timed out | elapsed_s=%.1f alive=%s exitcode=%s",
                    elapsed,
                    proc.is_alive(),
                    proc.exitcode,
                )
                self._shutdown_worker()
                raise RuntimeError(
                    f"vLLM colocate init timed out after {timeout_s:.1f}s"
                )
            poll_timeout = heartbeat_s
            if timeout_s and timeout_s > 0:
                remaining = max(timeout_s - elapsed, 0.0)
                poll_timeout = min(heartbeat_s, remaining)
            if parent_conn.poll(poll_timeout):
                init_resp = parent_conn.recv()
                if isinstance(init_resp, dict) and init_resp.get("type") == "log":
                    LOG.info("vLLM colocate worker | %s", init_resp.get("message"))
                    init_resp = None
                    continue
                break
            LOG.info(
                "vLLM colocate init waiting on subprocess | elapsed_s=%.1f alive=%s exitcode=%s",
                elapsed,
                proc.is_alive(),
                proc.exitcode,
            )
            if not proc.is_alive() and not parent_conn.poll(0.0):
                init_resp = {
                    "ok": False,
                    "error": f"vLLM colocate subprocess exited (exitcode={proc.exitcode})",
                }
                break
        if init_resp is None:
            self._init_failed = True
            self._shutdown_worker()
            raise RuntimeError("vLLM colocate subprocess init produced no response")
        if not isinstance(init_resp, dict) or not init_resp.get("ok"):
            self._init_failed = True
            self._shutdown_worker()
            err = init_resp.get("error") if isinstance(init_resp, dict) else None
            tb = init_resp.get("traceback") if isinstance(init_resp, dict) else None
            if tb:
                LOG.warning("vLLM colocate subprocess init traceback:\n%s", tb)
            raise RuntimeError(err or "vLLM colocate subprocess init failed")
        self._worker_conn = parent_conn
        self._worker_proc = proc
        atexit.register(self._shutdown_worker)

    def _build_llm_with_timeout(self) -> Any:
        timeout_s = _env_float("MAXENT_VLLM_COLOCATE_INIT_TIMEOUT_S")
        if timeout_s is None or timeout_s <= 0:
            return self._build_llm()

        mode = self._resolve_init_mode()
        if mode == "subprocess":
            self._ensure_worker()
            return None
        LOG.info("vLLM colocate init start | timeout_s=%.1f | mode=%s", timeout_s, mode)
        if mode == "blocking":
            started = time.time()
            llm = self._build_llm()
            elapsed = time.time() - started
            if elapsed > timeout_s:
                LOG.warning(
                    "vLLM colocate init exceeded timeout (%.1fs > %.1fs). "
                    "Set MAXENT_VLLM_COLOCATE_INIT_MODE=thread to enforce a hard timeout.",
                    elapsed,
                    timeout_s,
                )
            return llm

        result: Dict[str, Any] = {}
        error: Dict[str, Exception] = {}
        done = threading.Event()

        def _runner() -> None:
            try:
                result["llm"] = self._build_llm()
            except Exception as exc:
                error["exc"] = exc
            finally:
                done.set()

        thread = threading.Thread(
            target=_runner,
            name="vllm-colocate-init",
            daemon=True,
        )
        thread.start()
        heartbeat_s = _env_float("MAXENT_VLLM_COLOCATE_INIT_HEARTBEAT_S")
        if heartbeat_s is None or heartbeat_s <= 0:
            heartbeat_s = 30.0
        start = time.time()
        while True:
            elapsed = time.time() - start
            remaining = timeout_s - elapsed
            if remaining <= 0:
                LOG.warning(
                    "vLLM colocate init timed out after %.1fs; disabling colocate engine.",
                    timeout_s,
                )
                LOG.warning(
                    "vLLM colocate init thread still running | elapsed_s=%.1f alive=%s",
                    elapsed,
                    thread.is_alive(),
                )
                raise RuntimeError(
                    f"vLLM colocate init timed out after {timeout_s:.1f}s"
                )
            wait_s = min(heartbeat_s, remaining)
            if done.wait(wait_s):
                break
            LOG.info(
                "vLLM colocate init still running | elapsed_s=%.1f thread_alive=%s",
                elapsed,
                thread.is_alive(),
            )
        if "exc" in error:
            raise error["exc"]
        return result["llm"]

    def _get_llm(self) -> Any:
        if self._llm is not None:
            return self._llm
        if self._init_failed:
            raise RuntimeError("vLLM colocate initialization previously failed")
        if self._resolve_init_mode() == "subprocess":
            raise RuntimeError("vLLM colocate subprocess mode is active")
        try:
            self._llm = self._build_llm_with_timeout()
        except Exception as exc:
            self._init_failed = True
            raise RuntimeError(str(exc)) from exc
        return self._llm

    def _build_sampling_params_kwargs(self, request_count: int) -> Dict[str, Any]:
        stop_sequences = (
            self.ctx.gen_stop_sequences
            if getattr(self.ctx, "gen_stop_sequences", None) is not None
            else getattr(self.ctx, "vllm_stop_sequences", None)
        )
        top_k = (
            self.ctx.gen_top_k
            if getattr(self.ctx, "gen_top_k", None) is not None
            else getattr(self.ctx, "vllm_top_k", None)
        )
        if top_k is None or top_k == 0:
            top_k = -1
        best_of = (
            self.ctx.gen_best_of
            if getattr(self.ctx, "gen_best_of", None) is not None
            else getattr(self.ctx, "vllm_best_of", None)
        )
        params_kwargs: Dict[str, Any] = {
            "temperature": self.ctx.gen_temperature,
            "top_p": self.ctx.gen_top_p,
            "top_k": top_k,
            "max_tokens": self.ctx.max_completion_len,
            "n": int(request_count),
            "best_of": best_of,
            "frequency_penalty": getattr(self.ctx, "gen_frequency_penalty", 0.0),
            "presence_penalty": getattr(self.ctx, "gen_presence_penalty", 0.0),
            "stop": stop_sequences,
            "logit_bias": getattr(self.ctx, "vllm_logit_bias", None),
            "guided_json": getattr(self.ctx, "vllm_guided_json", None),
            "guided_regex": getattr(self.ctx, "vllm_guided_regex", None),
        }
        if bool(getattr(self.ctx, "vllm_request_logprobs", False)):
            params_kwargs["logprobs"] = 1
        return params_kwargs

    def _build_sampling_params(self, request_count: int) -> Any:
        vllm_mod = optional_import("vllm")
        if vllm_mod is None:
            raise RuntimeError("vllm is not installed")
        params_cls = getattr(vllm_mod, "SamplingParams", None)
        if params_cls is None:
            raise RuntimeError("vllm.SamplingParams is unavailable")
        params_kwargs = self._build_sampling_params_kwargs(request_count)
        params_kwargs = _filter_kwargs(params_cls, params_kwargs)
        return params_cls(**params_kwargs)

    def _record_latency(self, latency_ms: float) -> None:
        stats = getattr(self.ctx, "generation_stats", None)
        if not isinstance(stats, dict):
            return
        stats["vllm_last_latency_ms"] = float(latency_ms)
        stats["vllm_latency_total_ms"] = float(stats.get("vllm_latency_total_ms", 0.0)) + float(
            latency_ms
        )
        stats["vllm_latency_calls"] = int(stats.get("vllm_latency_calls", 0)) + 1

    def request_batch(
        self,
        prompts: List[str],
        request_count: int,
    ) -> Tuple[
        Optional[List[List[str]]],
        Optional[List[List[Optional[VLLMLogprobResult]]]],
    ]:
        if not prompts:
            return [], None
        char_limit = getattr(self.ctx, "prompt_char_limit", None)
        if char_limit is None:
            char_limit = PROMPT_CHAR_LIMIT
        tokenizer = getattr(self.ctx, "tokenizer", None)
        max_tokens = getattr(self.ctx, "max_prompt_len", None)
        truncated = [
            _truncate_prompt(
                prompt,
                char_limit,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
            )
            for prompt in prompts
        ]

        request_logprobs = bool(getattr(self.ctx, "vllm_request_logprobs", False))
        if self._resolve_init_mode() == "subprocess":
            try:
                self._ensure_worker()
                params_kwargs = self._build_sampling_params_kwargs(request_count)
            except Exception as exc:
                LOG.warning("vLLM colocate unavailable (%s); falling back to local generation.", exc)
                return self._fallback_generate(truncated, request_count, None)

            start = time.time()
            try:
                LOG.info(
                    "vLLM colocate generate (subprocess) | prompts=%d request_count=%d",
                    len(truncated),
                    request_count,
                )
                conn = self._worker_conn
                if conn is None:
                    raise RuntimeError("vLLM colocate worker is unavailable")
                conn.send(
                    {
                        "type": "generate",
                        "prompts": truncated,
                        "params_kwargs": params_kwargs,
                        "request_logprobs": request_logprobs,
                    }
                )
                resp = conn.recv()
                if not isinstance(resp, dict) or not resp.get("ok"):
                    err = resp.get("error") if isinstance(resp, dict) else None
                    raise RuntimeError(err or "vLLM colocate subprocess request failed")
                grouped = resp.get("grouped") or []
                grouped_meta = _coerce_logprob_payload(resp.get("meta"))
            except Exception as exc:
                LOG.warning(
                    "vLLM colocate generate failed (%s); falling back to local generation.",
                    exc,
                )
                return self._fallback_generate(truncated, request_count, None)
            latency_ms = (time.time() - start) * 1000.0
            LOG.info("vLLM colocate generate done | latency_ms=%.2f", latency_ms)
            self._record_latency(latency_ms)
        else:
            try:
                llm = self._get_llm()
                params = self._build_sampling_params(request_count)
            except Exception as exc:
                LOG.warning("vLLM colocate unavailable (%s); falling back to local generation.", exc)
                return self._fallback_generate(truncated, request_count, None)

            start = time.time()
            try:
                LOG.info(
                    "vLLM colocate generate | prompts=%d request_count=%d",
                    len(truncated),
                    request_count,
                )
                outputs = llm.generate(truncated, params)
                grouped, grouped_meta_payload = _outputs_to_payload(outputs, request_logprobs)
                grouped_meta = _coerce_logprob_payload(grouped_meta_payload)
            except Exception as exc:
                LOG.warning(
                    "vLLM colocate generate failed (%s); falling back to local generation.",
                    exc,
                )
                return self._fallback_generate(truncated, request_count, None)
            latency_ms = (time.time() - start) * 1000.0
            LOG.info("vLLM colocate generate done | latency_ms=%.2f", latency_ms)
            self._record_latency(latency_ms)

        if len(grouped) != len(prompts):
            LOG.warning(
                "vLLM colocate returned %d groups for %d prompts; falling back to local generation.",
                len(grouped),
                len(prompts),
            )
            return self._fallback_generate(truncated, request_count, None)
        return grouped, grouped_meta


__all__ = ["ColocateVLLMEngine"]
