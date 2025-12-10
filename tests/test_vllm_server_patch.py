import json

import pytest

from maxent_grpo.patches import vllm_server


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
    TestClient = pytest.importorskip("fastapi.testclient").TestClient

    app = fastapi.FastAPI()

    @app.post("/generate")
    async def generate():
        return {"results": [{"outputs": [{"text": "foo"}]}]}

    installed = vllm_server.install_vllm_client_tag_middleware(app)
    assert installed is True

    client = TestClient(app)
    headers = {"X-VLLM-Client-Tag": "rank-9"}
    resp = client.post("/generate", json={"prompts": ["x"]}, headers=headers)
    payload = resp.json()
    result = payload["results"][0]
    assert result["metadata"]["client_tag"] == "rank-9"
    assert result["outputs"][0]["metadata"]["client_tag"] == "rank-9"


def test_fastapi_middleware_handles_streaming_body():
    fastapi = pytest.importorskip("fastapi")
    TestClient = pytest.importorskip("fastapi.testclient").TestClient
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

    client = TestClient(app)
    resp = client.post(
        "/generate",
        json={"prompts": ["x"]},
        headers={"X-VLLM-Client-Tag": "rank-stream"},
    )
    payload = resp.json()
    assert payload["results"][0]["metadata"]["client_tag"] == "rank-stream"


def test_fastapi_middleware_stream_param():
    fastapi = pytest.importorskip("fastapi")
    TestClient = pytest.importorskip("fastapi.testclient").TestClient
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
    client = TestClient(app)
    with client.stream(
        "POST",
        "/generate",
        json={"prompts": ["x"]},
        headers={"X-VLLM-Client-Tag": "rank-stream-ctx"},
    ) as resp:
        iter_content = getattr(resp, "iter_content", None)
        if callable(iter_content):
            chunks = iter_content()
        else:
            iter_bytes = getattr(resp, "iter_bytes", None)
            if callable(iter_bytes):
                chunks = iter_bytes()
            else:  # pragma: no cover - fallback for unexpected clients
                chunks = [resp.read()]
        body = b"".join(chunks)
    payload = json.loads(body)
    assert payload["results"][0]["metadata"]["client_tag"] == "rank-stream-ctx"
