from __future__ import annotations

import json

import pytest

from maxent_grpo.patches import vllm_server


def _run_asgi_json(
    app,
    *,
    method: str = "POST",
    path: str = "/generate",
    headers: dict[str, str] | None = None,
    payload: dict | None = None,
) -> tuple[int, dict[str, str], bytes]:
    """Invoke an ASGI app and return (status, headers, body_bytes).

    This avoids Starlette's TestClient, which relies on cross-thread asyncio
    wakeups that are not available in some sandboxed/HPC environments.
    """

    import asyncio

    body = json.dumps(payload or {}).encode("utf-8")
    header_items = [(b"content-type", b"application/json")]
    for key, value in (headers or {}).items():
        header_items.append((key.lower().encode("latin-1"), value.encode("latin-1")))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "scheme": "http",
        "method": method,
        "path": path,
        "raw_path": path.encode("ascii", errors="ignore"),
        "query_string": b"",
        "root_path": "",
        "headers": header_items,
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
    }

    messages: list[dict] = []
    request_sent = False

    async def receive():
        nonlocal request_sent
        if request_sent:
            return {"type": "http.disconnect"}
        request_sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)

    asyncio.run(app(scope, receive, send))
    start = next(msg for msg in messages if msg.get("type") == "http.response.start")
    status = int(start.get("status", 200))
    resp_headers: dict[str, str] = {}
    for key, value in start.get("headers") or []:
        resp_headers[key.decode("latin-1").lower()] = value.decode("latin-1")
    chunks = [
        msg.get("body", b"")
        for msg in messages
        if msg.get("type") == "http.response.body"
    ]
    return status, resp_headers, b"".join(chunks)


def test_propagate_client_tag_updates_results_and_outputs():
    payload = {
        "results": [
            {
                "outputs": [
                    {"text": "foo"},
                    {"metadata": {"client_tag": "old"}, "text": "bar"},
                ]
            }
        ]
    }
    changed = vllm_server._propagate_client_tag(payload, "rank-1")
    assert changed is True
    assert payload["results"][0]["metadata"]["client_tag"] == "rank-1"
    assert payload["results"][0]["outputs"][0]["metadata"]["client_tag"] == "rank-1"
    assert payload["results"][0]["outputs"][1]["metadata"]["client_tag"] == "rank-1"


def test_propagate_client_tag_handles_choice_schema():
    payload = {
        "choices": [
            {"message": {"role": "assistant", "content": "hola"}},
            {"metadata": {"client_tag": "rank-1"}},
        ]
    }
    changed = vllm_server._propagate_client_tag(payload, "rank-2")
    assert changed is True
    assert payload["choices"][0]["metadata"]["client_tag"] == "rank-2"
    assert payload["choices"][1]["metadata"]["client_tag"] == "rank-2"


def test_fastapi_middleware_rewrites_generate_response():
    fastapi = pytest.importorskip("fastapi")

    app = fastapi.FastAPI()

    @app.post("/generate")
    async def generate():
        return {"results": [{"outputs": [{"text": "foo"}]}]}

    installed = vllm_server.install_vllm_client_tag_middleware(app)
    assert installed is True

    status, _headers, body = _run_asgi_json(
        app,
        headers={"X-VLLM-Client-Tag": "rank-9"},
        payload={"prompts": ["x"]},
    )
    assert status == 200
    payload = json.loads(body)
    result = payload["results"][0]
    assert result["metadata"]["client_tag"] == "rank-9"
    assert result["outputs"][0]["metadata"]["client_tag"] == "rank-9"


def test_fastapi_middleware_handles_streaming_body():
    fastapi = pytest.importorskip("fastapi")
    StreamingResponse = pytest.importorskip("starlette.responses").StreamingResponse

    app = fastapi.FastAPI()

    @app.post("/generate")
    async def generate():
        payload = {"results": [{"outputs": [{"text": "foo"}]}]}

        async def _chunks():
            data = json.dumps(payload).encode("utf-8")
            mid = len(data) // 2
            yield data[:mid]
            yield data[mid:]

        return StreamingResponse(_chunks(), media_type="application/json")

    install_cnt = vllm_server.install_vllm_client_tag_middleware(app)
    assert install_cnt is True

    status, _headers, body = _run_asgi_json(
        app,
        headers={"X-VLLM-Client-Tag": "rank-stream"},
        payload={"prompts": ["x"]},
    )
    assert status == 200
    payload = json.loads(body)
    assert payload["results"][0]["metadata"]["client_tag"] == "rank-stream"


def test_fastapi_middleware_stream_param():
    fastapi = pytest.importorskip("fastapi")
    StreamingResponse = pytest.importorskip("starlette.responses").StreamingResponse

    app = fastapi.FastAPI()

    @app.post("/generate")
    async def generate():
        payload = {"results": [{"outputs": [{"text": "foo"}]}]}

        async def _chunks():
            data = json.dumps(payload).encode("utf-8")
            yield data[: max(1, len(data) // 3)]
            yield data[max(1, len(data) // 3) :]

        return StreamingResponse(_chunks(), media_type="application/json")

    assert vllm_server.install_vllm_client_tag_middleware(app)
    status, _headers, body = _run_asgi_json(
        app,
        headers={"X-VLLM-Client-Tag": "rank-stream-ctx"},
        payload={"prompts": ["x"]},
    )
    assert status == 200
    payload = json.loads(body)
    assert payload["results"][0]["metadata"]["client_tag"] == "rank-stream-ctx"
