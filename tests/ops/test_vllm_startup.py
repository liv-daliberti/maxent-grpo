from __future__ import annotations

from maxent_grpo.ops.vllm_startup import (
    StartupStatus,
    classify_vllm_startup_log,
    should_trigger_v0_fallback,
)


def test_classify_vllm_startup_healthy() -> None:
    log_text = """
INFO:     Started server process [123]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
"""
    assert classify_vllm_startup_log(log_text) is StartupStatus.HEALTHY


def test_classify_vllm_startup_core_engine_stall() -> None:
    log_text = """
INFO:     Started server process [123]
INFO:     Waiting for application startup.
INFO 02-24 00:15:44 [gpu_model_runner.py:1801] Model loading took 0.9277 GiB and 0.626989 seconds
DEBUG 02-24 00:15:52 [utils.py:475] Waiting for 1 local, 0 remote core engine proc(s) to start.
DEBUG 02-24 00:16:02 [utils.py:475] Waiting for 1 local, 0 remote core engine proc(s) to start.
DEBUG 02-24 00:16:12 [utils.py:475] Waiting for 1 local, 0 remote core engine proc(s) to start.
"""
    assert classify_vllm_startup_log(log_text, stall_threshold=3) is StartupStatus.CORE_ENGINE_STALL


def test_classify_vllm_startup_post_init_stall_signature() -> None:
    log_text = """
INFO:     Started server process [123]
INFO:     Waiting for application startup.
INFO 02-24 00:15:44 [gpu_model_runner.py:1801] Model loading took 0.9277 GiB and 0.626989 seconds
INFO 02-24 00:15:49 [kv_cache_utils.py:716] GPU KV cache size: 6,021,280 tokens
INFO 02-24 00:15:49 [kv_cache_utils.py:720] Maximum concurrency for 4,864 tokens per request: 1237.93x
"""
    assert classify_vllm_startup_log(log_text) is StartupStatus.CORE_ENGINE_STALL


def test_should_trigger_v0_fallback_requires_min_attempts() -> None:
    log_text = """
INFO:     Started server process [123]
INFO:     Waiting for application startup.
INFO 02-24 00:15:44 [gpu_model_runner.py:1801] Model loading took 0.9277 GiB and 0.626989 seconds
DEBUG 02-24 00:15:52 [utils.py:475] Waiting for 1 local, 0 remote core engine proc(s) to start.
DEBUG 02-24 00:16:02 [utils.py:475] Waiting for 1 local, 0 remote core engine proc(s) to start.
DEBUG 02-24 00:16:12 [utils.py:475] Waiting for 1 local, 0 remote core engine proc(s) to start.
"""
    assert not should_trigger_v0_fallback(log_text, attempt=10, min_attempts=20)
    assert should_trigger_v0_fallback(log_text, attempt=20, min_attempts=20)


def test_classify_vllm_startup_error() -> None:
    log_text = """
INFO:     Started server process [123]
ERROR 02-24 00:15:44 [core.py:654] Invocation of collective_rpc method failed
Traceback (most recent call last):
RuntimeError: NCCL error: invalid usage
"""
    assert classify_vllm_startup_log(log_text) is StartupStatus.ERROR
