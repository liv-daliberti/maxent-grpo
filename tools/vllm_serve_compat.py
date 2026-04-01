import importlib
import importlib.util
import inspect
import json
import logging
import os
from typing import Any, Optional


_VLLM_BATCH_UPDATE_PREFIX = "__maxent_vllm_batch__:"
_LOG = logging.getLogger(__name__)
_LEGACY_WEIGHT_SYNC_WORKER_NAME = "LegacyWeightSyncWorker"

os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")


def _patch_transformers_utils() -> None:
    """Backfill TRL-expected helpers missing from older Transformers builds."""

    try:
        transformers_utils = importlib.import_module("transformers.utils")
    except Exception:
        return
    if hasattr(transformers_utils, "is_rich_available"):
        return

    def _is_rich_available() -> bool:
        return importlib.util.find_spec("rich") is not None

    setattr(transformers_utils, "is_rich_available", _is_rich_available)


def _engine_arg_fields() -> set[str]:
    try:
        arg_utils = importlib.import_module("vllm.engine.arg_utils")
    except Exception:
        return set()
    engine_args_cls = getattr(arg_utils, "EngineArgs", None)
    fields = getattr(engine_args_cls, "__dataclass_fields__", None)
    if not isinstance(fields, dict):
        return set()
    return set(fields)


def _llm_supports_kwarg(name: str) -> bool:
    return name in _engine_arg_fields()


def _legacy_weight_sync_worker_cls(vllm_serve: Any) -> type:
    cached = globals().get(_LEGACY_WEIGHT_SYNC_WORKER_NAME)
    if isinstance(cached, type):
        return cached

    worker_mod = importlib.import_module("vllm.worker.worker")
    base_worker_cls = getattr(worker_mod, "Worker", None)
    extension_cls = getattr(vllm_serve, "WeightSyncWorkerExtension", None)
    if base_worker_cls is None or extension_cls is None:
        raise RuntimeError("Failed to construct legacy weight-sync worker class")

    legacy_cls = type(
        _LEGACY_WEIGHT_SYNC_WORKER_NAME,
        (extension_cls, base_worker_cls),
        {},
    )
    legacy_cls.__module__ = __name__
    globals()[_LEGACY_WEIGHT_SYNC_WORKER_NAME] = legacy_cls
    return legacy_cls


def _guided_decoding_kwargs(guided_decoding: Any) -> dict[str, Any]:
    kwargs = dict(getattr(guided_decoding, "kwargs", {}) or {})
    for name in (
        "json",
        "regex",
        "choice",
        "grammar",
        "json_object",
        "disable_fallback",
        "disable_any_whitespace",
        "disable_additional_properties",
        "whitespace_pattern",
        "structural_tag",
    ):
        if name not in kwargs:
            value = getattr(guided_decoding, name, None)
            if value is not None:
                kwargs[name] = value
    return kwargs


def _patch_guided_decoding() -> None:
    sampling_params = importlib.import_module("vllm.sampling_params")
    if hasattr(sampling_params, "GuidedDecodingParams"):
        return

    structured_outputs_cls = getattr(sampling_params, "StructuredOutputsParams", None)
    if structured_outputs_cls is None:
        return

    class GuidedDecodingParams:
        def __init__(self, backend: Optional[str] = None, **kwargs: Any):
            self.backend = backend
            self.kwargs = dict(kwargs)
            for key, value in kwargs.items():
                setattr(self, key, value)

    sampling_params.GuidedDecodingParams = GuidedDecodingParams


def _patch_sampling_params_factory(module: Any) -> None:
    if getattr(module, "_maxent_guided_decoding_patch", False):
        return

    sampling_params_mod = importlib.import_module("vllm.sampling_params")
    guided_cls = getattr(sampling_params_mod, "GuidedDecodingParams", None)
    structured_outputs_cls = getattr(sampling_params_mod, "StructuredOutputsParams", None)
    if guided_cls is None or structured_outputs_cls is None:
        return

    original_sampling_params = getattr(module, "SamplingParams", None)
    if original_sampling_params is None:
        return

    def _guided_to_structured_outputs(guided_decoding: Any) -> Any:
        if guided_decoding is None:
            return None
        if isinstance(guided_decoding, structured_outputs_cls):
            return guided_decoding
        structured = structured_outputs_cls(
            **_guided_decoding_kwargs(guided_decoding)
        )
        backend = getattr(guided_decoding, "backend", None)
        if backend is not None:
            try:
                setattr(structured, "_backend", backend)
            except Exception:
                pass
        return structured

    def _compat_sampling_params(*args: Any, **kwargs: Any) -> Any:
        if "guided_decoding" in kwargs and "structured_outputs" not in kwargs:
            kwargs["structured_outputs"] = _guided_to_structured_outputs(
                kwargs.pop("guided_decoding")
            )
        else:
            kwargs.pop("guided_decoding", None)
        return original_sampling_params(*args, **kwargs)

    setattr(module, "GuidedDecodingParams", guided_cls)
    setattr(module, "SamplingParams", _compat_sampling_params)
    setattr(module, "_maxent_guided_decoding_patch", True)


def _patch_get_open_port() -> None:
    utils = importlib.import_module("vllm.utils")
    if hasattr(utils, "get_open_port"):
        return
    network_utils = importlib.import_module("vllm.utils.network_utils")
    utils.get_open_port = network_utils.get_open_port


def _import_builtin_weight_transfer() -> Optional[type]:
    if os.getenv("MAXENT_DISABLE_BUILTIN_WEIGHT_TRANSFER", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None
    try:
        nccl_engine_mod = importlib.import_module(
            "vllm.distributed.weight_transfer.nccl_engine"
        )
        gpu_worker_mod = importlib.import_module("vllm.v1.worker.gpu_worker")
    except Exception:
        return None

    engine_cls = getattr(nccl_engine_mod, "NCCLWeightTransferEngine", None)
    if engine_cls is None:
        return None
    worker_cls = getattr(gpu_worker_mod, "GPUWorker", None)
    if worker_cls is not None:
        if not callable(getattr(worker_cls, "init_weight_transfer_engine", None)):
            return None
        if not callable(getattr(worker_cls, "update_weights", None)):
            return None
    return engine_cls


def _build_update_request_model() -> Any:
    pydantic_mod = importlib.import_module("pydantic")
    base_model = getattr(pydantic_mod, "BaseModel")

    class _UpdateRequest(base_model):
        name: Optional[str] = None
        dtype: Optional[str] = None
        shape: Optional[list[int]] = None
        names: Optional[list[str]] = None
        dtypes: Optional[list[str]] = None
        shapes: Optional[list[list[int]]] = None

    return _UpdateRequest


def _build_generate_request_model() -> Any:
    pydantic_mod = importlib.import_module("pydantic")
    base_model = getattr(pydantic_mod, "BaseModel")

    class _GenerateRequest(base_model):
        prompts: list[str]
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        frequency_penalty: float = 0.0
        presence_penalty: float = 0.0
        best_of: Optional[int] = None
        stop: Optional[list[str]] = None
        logit_bias: Optional[dict[str, float]] = None
        allowed_token_ids: Optional[list[int]] = None
        blocked_token_ids: Optional[list[int]] = None
        guided_decoding_regex: Optional[str] = None
        guided_regex: Optional[str] = None
        guided_json: Optional[str] = None
        logprobs: Optional[int] = None
        return_logprobs: bool = False
        stream: bool = False
        request_id: Optional[str] = None
        client_tag: Optional[str] = None

    return _GenerateRequest


def _build_generate_response_model() -> Any:
    pydantic_mod = importlib.import_module("pydantic")
    base_model = getattr(pydantic_mod, "BaseModel")

    class _GenerateResponse(base_model):
        completion_ids: list[list[int]]
        logprobs: Optional[list[Any]] = None
        cumulative_logprobs: Optional[list[Optional[float]]] = None

    return _GenerateResponse


def _normalize_update_request(request: Any) -> tuple[list[str], list[str], list[tuple[int, ...]]]:
    names = getattr(request, "names", None)
    dtypes = getattr(request, "dtypes", None)
    shapes = getattr(request, "shapes", None)
    if names is not None or dtypes is not None or shapes is not None:
        if not isinstance(names, list) or not isinstance(dtypes, list) or not isinstance(shapes, list):
            raise ValueError("Batched weight update requires names, dtypes, and shapes lists.")
        if not (len(names) == len(dtypes) == len(shapes)):
            raise ValueError("Batched weight update payload lengths must match.")
        return (
            [str(name) for name in names],
            [str(dtype) for dtype in dtypes],
            [tuple(shape) for shape in shapes],
        )
    name = getattr(request, "name", None)
    dtype = getattr(request, "dtype", None)
    shape = getattr(request, "shape", None)
    if name is None or dtype is None or shape is None:
        raise ValueError("Weight update requires name/dtype/shape.")
    if isinstance(name, str) and name.startswith(_VLLM_BATCH_UPDATE_PREFIX):
        payload = json.loads(name[len(_VLLM_BATCH_UPDATE_PREFIX):])
        names = payload.get("names")
        dtypes = payload.get("dtypes")
        shapes = payload.get("shapes")
        if not isinstance(names, list) or not isinstance(dtypes, list) or not isinstance(shapes, list):
            raise ValueError("Encoded batched weight update payload is invalid.")
        if not (len(names) == len(dtypes) == len(shapes)):
            raise ValueError("Encoded batched weight update payload lengths must match.")
        return (
            [str(item) for item in names],
            [str(item) for item in dtypes],
            [tuple(item) for item in shapes],
        )
    return [str(name)], [str(dtype)], [tuple(shape)]


def _normalize_logit_bias(raw: Any) -> Optional[dict[int, float]]:
    if not isinstance(raw, dict):
        return None
    normalized: dict[int, float] = {}
    for key, value in raw.items():
        try:
            normalized[int(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized or None


def _normalize_allowed_token_ids(raw: Any) -> Optional[list[int]]:
    if not isinstance(raw, list):
        return None
    normalized: list[int] = []
    for value in raw:
        try:
            token_id = int(value)
        except (TypeError, ValueError):
            continue
        if token_id >= 0:
            normalized.append(token_id)
    return normalized or None


def _normalize_blocked_token_ids(raw: Any) -> Optional[list[int]]:
    if not isinstance(raw, list):
        return None
    normalized: list[int] = []
    for value in raw:
        try:
            token_id = int(value)
        except (TypeError, ValueError):
            continue
        if token_id >= 0:
            normalized.append(token_id)
    return normalized or None


def _blocked_tail_to_allowed_token_ids(
    blocked_token_ids: Optional[list[int]],
) -> Optional[list[int]]:
    """Translate a contiguous blocked tail into native allowed_token_ids.

    vLLM 0.7.2 supports ``allowed_token_ids`` natively, but there is no
    built-in ``blocked_token_ids`` or token-id bad-words hook. Our server-mode
    trainer sends the blocked tail as ``range(tokenizer_limit, model_limit)``.
    Convert that into the equivalent allowed prefix ``[0, tokenizer_limit)`` so
    the engine enforces the boundary via its real logits-processor path.
    """

    if not blocked_token_ids:
        return None
    blocked = sorted(set(int(token_id) for token_id in blocked_token_ids))
    if not blocked:
        return None
    start = blocked[0]
    end = blocked[-1]
    if start <= 0:
        raise ValueError("blocked_token_ids tail must start above 0")
    expected = list(range(start, end + 1))
    if blocked != expected:
        raise ValueError(
            "blocked_token_ids must form one contiguous tail range to derive "
            "allowed_token_ids safely"
        )
    return list(range(start))


def _extract_output_logprobs(output: Any) -> Optional[list[float]]:
    raw = getattr(output, "logprobs", None)
    if raw is None:
        return None
    if isinstance(raw, dict):
        raw = raw.get("token_logprobs")
    if not isinstance(raw, list):
        return None
    cleaned: list[float] = []
    for item in raw:
        try:
            if isinstance(item, dict):
                value = item.get("logprob")
            else:
                value = getattr(item, "logprob", item)
            if value is None:
                continue
            cleaned.append(float(value))
        except (TypeError, ValueError):
            continue
    return cleaned or None


def _patch_blocking_init_communicator() -> Any:
    _patch_transformers_utils()
    vllm_serve = importlib.import_module("trl.scripts.vllm_serve")
    _patch_sampling_params_factory(vllm_serve)
    uvicorn_mod = importlib.import_module("uvicorn")
    original_run = getattr(uvicorn_mod, "run")
    if getattr(original_run, "_maxent_blocking_init_patch", False):
        return vllm_serve

    from fastapi.routing import APIRoute
    builtin_weight_transfer = _import_builtin_weight_transfer()
    update_request_model = _build_update_request_model()
    generate_request_model = _build_generate_request_model()
    generate_response_model = _build_generate_response_model()

    def _rebuild_route(
        route: APIRoute,
        endpoint: Any,
        path: str | None = None,
        response_model: Any | None = None,
    ) -> None:
        APIRoute.__init__(
            route,
            path or route.path,
            endpoint,
            response_model=response_model if response_model is not None else route.response_model,
            status_code=route.status_code,
            tags=route.tags,
            dependencies=route.dependencies,
            summary=route.summary,
            description=route.description,
            response_description=route.response_description,
            responses=route.responses,
            deprecated=route.deprecated,
            name=route.name,
            methods=sorted(route.methods or []),
            operation_id=route.operation_id,
            response_model_include=route.response_model_include,
            response_model_exclude=route.response_model_exclude,
            response_model_by_alias=route.response_model_by_alias,
            response_model_exclude_unset=route.response_model_exclude_unset,
            response_model_exclude_defaults=route.response_model_exclude_defaults,
            response_model_exclude_none=route.response_model_exclude_none,
            include_in_schema=route.include_in_schema,
            response_class=route.response_class,
            dependency_overrides_provider=route.dependency_overrides_provider,
            callbacks=route.callbacks,
            openapi_extra=route.openapi_extra,
            generate_unique_id_function=route.generate_unique_id_function,
        )

    def _make_generate_endpoint(
        endpoint: Any,
        connections: list[Any],
        script_args: Any,
    ) -> Any:
        async def _patched_generate(request: Any) -> dict[str, Any]:
            guided_regex = getattr(request, "guided_regex", None) or getattr(
                request, "guided_decoding_regex", None
            )
            guided_json = getattr(request, "guided_json", None)
            guided_decoding = None
            guided_cls = getattr(vllm_serve, "GuidedDecodingParams", None)
            if guided_cls is not None:
                if guided_json is not None:
                    guided_decoding = guided_cls(backend="outlines", json=guided_json)
                elif guided_regex is not None:
                    guided_decoding = guided_cls(backend="outlines", regex=guided_regex)

            sampling_kwargs: dict[str, Any] = {
                "n": int(getattr(request, "n", 1)),
                "repetition_penalty": float(getattr(request, "repetition_penalty", 1.0)),
                "temperature": float(getattr(request, "temperature", 1.0)),
                "top_p": float(getattr(request, "top_p", 1.0)),
                "top_k": int(getattr(request, "top_k", -1)),
                "min_p": float(getattr(request, "min_p", 0.0)),
                "max_tokens": int(getattr(request, "max_tokens", 16)),
                "frequency_penalty": float(getattr(request, "frequency_penalty", 0.0)),
                "presence_penalty": float(getattr(request, "presence_penalty", 0.0)),
                "guided_decoding": guided_decoding,
            }
            best_of = getattr(request, "best_of", None)
            if best_of is not None:
                sampling_kwargs["best_of"] = int(best_of)
            stop = getattr(request, "stop", None)
            if stop:
                sampling_kwargs["stop"] = list(stop)
            logit_bias = _normalize_logit_bias(getattr(request, "logit_bias", None))
            if logit_bias:
                sampling_kwargs["logit_bias"] = logit_bias
            allowed_token_ids = _normalize_allowed_token_ids(
                getattr(request, "allowed_token_ids", None)
            )
            blocked_token_ids = _normalize_blocked_token_ids(
                getattr(request, "blocked_token_ids", None)
            )
            if blocked_token_ids:
                derived_allowed = _blocked_tail_to_allowed_token_ids(
                    blocked_token_ids
                )
                if allowed_token_ids:
                    blocked_set = set(blocked_token_ids)
                    merged_allowed = [
                        token_id
                        for token_id in allowed_token_ids
                        if token_id not in blocked_set
                    ]
                    if len(merged_allowed) != len(allowed_token_ids):
                        _LOG.warning(
                            "Filtered %d blocked token ids out of allowed_token_ids on /generate.",
                            len(allowed_token_ids) - len(merged_allowed),
                        )
                    allowed_token_ids = merged_allowed
                else:
                    allowed_token_ids = derived_allowed
                    _LOG.warning(
                        "Translated blocked_token_ids tail (%d blocked) into allowed_token_ids (%d allowed) on /generate.",
                        len(blocked_token_ids),
                        len(allowed_token_ids or []),
                    )
            if allowed_token_ids:
                sampling_kwargs["allowed_token_ids"] = allowed_token_ids
            logprobs = getattr(request, "logprobs", None)
            return_logprobs = bool(getattr(request, "return_logprobs", False))
            if logprobs is not None or return_logprobs:
                sampling_kwargs["logprobs"] = int(logprobs or 1)

            sampling_params = vllm_serve.SamplingParams(**sampling_kwargs)
            chunked_prompts = vllm_serve.chunk_list(
                request.prompts, script_args.data_parallel_size
            )
            for connection, prompts in zip(connections, chunked_prompts):
                if not prompts:
                    prompts = ["<placeholder>"]
                kwargs = {"prompts": prompts, "sampling_params": sampling_params}
                connection.send({"type": "call", "method": "generate", "kwargs": kwargs})

            all_outputs = [connection.recv() for connection in connections]
            all_outputs = [
                output for output, prompts in zip(all_outputs, chunked_prompts) if prompts
            ]
            completion_ids: list[list[int]] = []
            flat_logprobs: list[Optional[list[float]]] = []
            flat_cumulative: list[Optional[float]] = []
            have_logprobs = False
            for outputs in importlib.import_module("itertools").chain.from_iterable(all_outputs):
                for output in outputs.outputs:
                    completion_ids.append(list(output.token_ids))
                    token_logprobs = _extract_output_logprobs(output)
                    cumulative = getattr(output, "cumulative_logprob", None)
                    if token_logprobs is not None or cumulative is not None:
                        have_logprobs = True
                    flat_logprobs.append(token_logprobs)
                    try:
                        flat_cumulative.append(None if cumulative is None else float(cumulative))
                    except (TypeError, ValueError):
                        flat_cumulative.append(None)

            response: dict[str, Any] = {"completion_ids": completion_ids}
            if have_logprobs:
                if any(item is not None for item in flat_logprobs):
                    response["logprobs"] = flat_logprobs
                if any(item is not None for item in flat_cumulative):
                    response["cumulative_logprobs"] = flat_cumulative
            return response

        _patched_generate.__name__ = getattr(endpoint, "__name__", "generate")
        _patched_generate.__doc__ = getattr(endpoint, "__doc__", None)
        _patched_generate.__signature__ = inspect.Signature(
            parameters=[
                inspect.Parameter(
                    "request",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=generate_request_model,
                )
            ],
            return_annotation=generate_response_model,
        )
        _patched_generate.__annotations__ = {
            "request": generate_request_model,
            "return": generate_response_model,
        }
        setattr(_patched_generate, "_maxent_generate_patch", True)
        return _patched_generate

    def _make_init_endpoint(
        endpoint: Any,
        connections: list[Any],
        script_args: Any,
        request_annotation: Any,
    ) -> Any:
        async def _patched_init_communicator(request: Any) -> dict[str, str]:
            world_size = script_args.tensor_parallel_size * script_args.data_parallel_size + 1
            if builtin_weight_transfer is None:
                command = {
                    "type": "call",
                    "method": "collective_rpc",
                    "kwargs": {
                        "method": "init_communicator",
                        "args": (request.host, request.port, world_size),
                    },
                }
            else:
                command = {
                    "type": "fire_and_forget_with_ack",
                    "method": "init_weight_transfer_engine",
                    "args": (
                        {
                            "init_info": {
                                "master_address": request.host,
                                "master_port": request.port,
                                "rank_offset": 1,
                                "world_size": world_size,
                            }
                        },
                    ),
                }
                _LOG.info(
                    "vLLM built-in init accepted | host=%s port=%s world_size=%d",
                    request.host,
                    request.port,
                    world_size,
                )
            for connection in connections:
                connection.send(command)
            for connection in connections:
                connection.recv()
            return {"message": "Communicator initialized"}

        _patched_init_communicator.__name__ = getattr(endpoint, "__name__", "init_communicator")
        _patched_init_communicator.__doc__ = getattr(endpoint, "__doc__", None)
        if request_annotation is not inspect.Signature.empty:
            _patched_init_communicator.__signature__ = inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        "request",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=request_annotation,
                    )
                ],
                return_annotation=dict[str, str],
            )
            _patched_init_communicator.__annotations__ = {
                "request": request_annotation,
                "return": dict[str, str],
            }
        setattr(_patched_init_communicator, "_maxent_blocking_init_patch", True)
        return _patched_init_communicator

    def _make_update_endpoint(
        endpoint: Any,
        connections: list[Any],
        request_annotation: Any,
    ) -> Any:
        async def _patched_update_named_param(request: Any) -> dict[str, str]:
            names, dtypes, shapes = _normalize_update_request(request)
            if builtin_weight_transfer is None:
                if len(names) != 1:
                    raise ValueError("Legacy vLLM update path only supports single-parameter updates.")
                dtype = getattr(importlib.import_module("torch"), dtypes[0].split(".")[-1])
                command = {
                    "type": "fire_and_forget_with_ack",
                    "method": "collective_rpc",
                    "kwargs": {
                        "method": "update_named_param",
                        "args": (names[0], dtype, shapes[0]),
                    },
                }
            else:
                command = {
                    "type": "call",
                    "method": "update_weights",
                    "args": (
                        {
                            "update_info": {
                                "names": names,
                                "dtype_names": [
                                    dtype.split(".")[-1] for dtype in dtypes
                                ],
                                "shapes": [list(shape) for shape in shapes],
                                "packed": True,
                                "is_checkpoint_format": True,
                            }
                        },
                    ),
                }
                _LOG.info(
                    "vLLM built-in update requested | params=%d packed=true checkpoint=true",
                    len(names),
                )
            for connection in connections:
                connection.send(command)
            responses = [connection.recv() for connection in connections]
            if builtin_weight_transfer is not None:
                # We only care that the worker-side update completed without raising.
                _ = responses
            return {"message": "Workers ready for named parameter update"}

        _patched_update_named_param.__name__ = getattr(endpoint, "__name__", "update_named_param")
        _patched_update_named_param.__doc__ = getattr(endpoint, "__doc__", None)
        request_model = update_request_model
        if request_model is inspect.Signature.empty:
            request_model = request_annotation
        _patched_update_named_param.__signature__ = inspect.Signature(
            parameters=[
                inspect.Parameter(
                    "request",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=request_model,
                )
            ],
            return_annotation=dict[str, str],
        )
        _patched_update_named_param.__annotations__ = {
            "request": request_model,
            "return": dict[str, str],
        }
        setattr(_patched_update_named_param, "_maxent_worker_ack_patch", True)
        return _patched_update_named_param

    def _patch_app(app: Any) -> None:
        routes = getattr(app, "routes", None)
        if not isinstance(routes, list):
            return
        for route in routes:
            if not isinstance(route, APIRoute):
                continue
            endpoint = getattr(route, "endpoint", None)
            if endpoint is None:
                continue
            closure = inspect.getclosurevars(endpoint).nonlocals
            connections = closure.get("connections")
            if not isinstance(connections, list):
                continue
            request_param = inspect.signature(endpoint).parameters.get("request")
            request_annotation = (
                request_param.annotation
                if request_param is not None
                else inspect.Signature.empty
            )

            if getattr(route, "path", None) == "/generate/":
                script_args = closure.get("script_args")
                if script_args is None:
                    continue
                if getattr(endpoint, "_maxent_generate_patch", False):
                    continue
                _rebuild_route(
                    route,
                    _make_generate_endpoint(endpoint, connections, script_args),
                    response_model=generate_response_model,
                )
                continue

            if getattr(route, "path", None) == "/init_communicator/":
                script_args = closure.get("script_args")
                if script_args is None:
                    continue
                if getattr(endpoint, "_maxent_blocking_init_patch", False):
                    continue
                _rebuild_route(
                    route,
                    _make_init_endpoint(endpoint, connections, script_args, request_annotation),
                )
                continue

            if getattr(route, "path", None) == "/update_named_param/":
                if getattr(endpoint, "_maxent_worker_ack_patch", False):
                    continue
                _rebuild_route(
                    route,
                    _make_update_endpoint(endpoint, connections, request_annotation),
                )

    def _patched_run(app: Any, *args: Any, **kwargs: Any) -> Any:
        _patch_app(app)
        return original_run(app, *args, **kwargs)

    setattr(_patched_run, "_maxent_blocking_init_patch", True)
    uvicorn_mod.run = _patched_run
    if hasattr(vllm_serve, "uvicorn"):
        setattr(vllm_serve.uvicorn, "run", _patched_run)
    return vllm_serve


def _patch_worker_ack(vllm_serve: Any) -> None:
    original_worker = getattr(vllm_serve, "llm_worker", None)
    if original_worker is None or getattr(original_worker, "_maxent_ack_patch", False):
        return

    llm_cls = getattr(vllm_serve, "LLM", None)
    if llm_cls is None:
        raise RuntimeError("Failed to locate vLLM LLM class for worker patch")
    builtin_weight_transfer = _import_builtin_weight_transfer()
    supports_worker_extension_cls = _llm_supports_kwarg("worker_extension_cls")
    supports_worker_cls = _llm_supports_kwarg("worker_cls")

    def _patched_llm_worker(
        script_args: Any,
        data_parallel_rank: int,
        master_port: int,
        connection: Any,
    ) -> None:
        vllm_serve.os.environ["VLLM_DP_RANK"] = str(data_parallel_rank)
        vllm_serve.os.environ["VLLM_DP_RANK_LOCAL"] = str(data_parallel_rank)
        vllm_serve.os.environ["VLLM_DP_SIZE"] = str(script_args.data_parallel_size)
        vllm_serve.os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

        llm_kwargs = {
            "model": script_args.model,
            "revision": script_args.revision,
            "tensor_parallel_size": script_args.tensor_parallel_size,
            "gpu_memory_utilization": script_args.gpu_memory_utilization,
            "enforce_eager": script_args.enforce_eager,
            "dtype": script_args.dtype,
            "enable_prefix_caching": script_args.enable_prefix_caching,
            "kv_cache_dtype": script_args.kv_cache_dtype,
            "max_model_len": script_args.max_model_len,
        }
        if builtin_weight_transfer is None:
            if supports_worker_extension_cls:
                llm_kwargs["worker_extension_cls"] = (
                    "trl.scripts.vllm_serve.WeightSyncWorkerExtension"
                )
            elif supports_worker_cls:
                llm_kwargs["worker_cls"] = _legacy_weight_sync_worker_cls(vllm_serve)
            else:
                raise RuntimeError(
                    "vLLM runtime does not support either worker_extension_cls or worker_cls"
                )
        else:
            llm_kwargs["weight_transfer_config"] = {"backend": "nccl"}

        llm = llm_cls(**llm_kwargs)

        connection.send({"status": "ready"})

        while True:
            try:
                command = connection.recv()
            except KeyboardInterrupt:
                if builtin_weight_transfer is None:
                    llm.collective_rpc(method="close_communicator")
                break

            command_type = command["type"]
            if command_type in ["call", "fire_and_forget", "fire_and_forget_with_ack"]:
                method_name = command["method"]
                args, kwargs = command.get("args", ()), command.get("kwargs", {})
                method = getattr(llm, method_name)
                if command_type == "fire_and_forget_with_ack":
                    connection.send({"status": "accepted"})
                result = method(*args, **kwargs)
                if command_type == "call":
                    connection.send(result)
            elif command_type == "shutdown":
                break

    setattr(_patched_llm_worker, "_maxent_ack_patch", True)
    setattr(vllm_serve, "llm_worker", _patched_llm_worker)


def _patch_server_update_barrier(vllm_serve: Any) -> None:
    if _import_builtin_weight_transfer() is not None:
        return
    worker_ext = getattr(vllm_serve, "WeightSyncWorkerExtension", None)
    if worker_ext is None or getattr(worker_ext, "_maxent_skip_barrier_patch", False):
        return

    torch_mod = importlib.import_module("torch")

    def _patched_update_named_param(
        self: Any,
        name: str,
        dtype: Any,
        shape: Any,
    ) -> None:
        if self.pynccl_comm is None:
            raise RuntimeError(
                "Communicator not initialized. Call `init_communicator` first."
            )

        weight = torch_mod.empty(shape, dtype=dtype, device=self.device)
        self.pynccl_comm.broadcast(weight, src=self.client_rank)
        torch_mod.cuda.current_stream(device=weight.device).synchronize()
        self.model_runner.model.load_weights(weights=[(name, weight)])

    setattr(worker_ext, "update_named_param", _patched_update_named_param)
    setattr(worker_ext, "_maxent_skip_barrier_patch", True)


if __name__ == "__main__":
    _patch_guided_decoding()
    _patch_get_open_port()
    vllm_serve = _patch_blocking_init_communicator()
    _patch_worker_ack(vllm_serve)
    _patch_server_update_barrier(vllm_serve)
    parser = vllm_serve.make_parser()
    (script_args,) = parser.parse_args_and_config()
    vllm_serve.main(script_args)
