"""FastAPI adapter that exposes vLLM endpoints through an Ollama-compatible API."""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
from typing import Any
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import (
    ADAPTER_EXPOSED_MODEL_NAME,
    ADAPTER_MODEL_ARCHITECTURE,
    ADAPTER_MODEL_CONTEXT_LENGTH,
    ADAPTER_HOST,
    ADAPTER_PORT,
    OLLAMA_VERSION,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    VLLM_MODEL_NAME,
    VLLM_REQUEST_TIMEOUT,
)
from models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingsRequest,
    EmbeddingsResponse,
    GenerateRequest,
    GenerateResponse,
    ModelDetails,
    ModelInfo,
    ProcessModelInfo,
    PsResponse,
    PullRequest,
    PullResponse,
    ShowRequest,
    ShowResponse,
    TagsResponse,
)

NDJSON_MEDIA_TYPE = "application/x-ndjson"
SSE_MEDIA_TYPE = "text/event-stream"
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
COMPLETIONS_PATH = "/v1/completions"
EMBEDDINGS_PATH = "/v1/embeddings"
MODELS_PATH = "/v1/models"
DEFAULT_DIGEST = "sha256:0000000000000000"
UPSTREAM_OWNER = "vllm-ollama-adapter"


class UpstreamProxyError(Exception):
    """Raised when the upstream vLLM API returns an error or cannot be reached.

    Attributes:
        status_code: HTTP status code to return to the client (502/503 for proxy errors,
            or the upstream status for other errors).
        message: Human-readable error description.
    """

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(message)


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """Manage FastAPI application lifecycle: initialize HTTP client on startup, cleanup on shutdown.
    
    Creates an authenticated httpx.AsyncClient for proxying requests to the upstream vLLM API.
    The client is stored in app.state and automatically closed when the application shuts down.
    """
    headers = {"Authorization": f"Bearer {VLLM_API_KEY}"} if VLLM_API_KEY else None
    fastapi_app.state.http_client = httpx.AsyncClient(
        base_url=VLLM_BASE_URL,
        timeout=VLLM_REQUEST_TIMEOUT,
        headers=headers,
    )
    try:
        yield
    finally:
        await fastapi_app.state.http_client.aclose()


app = FastAPI(title="vLLM-to-Ollama Adapter", version=OLLAMA_VERSION, lifespan=lifespan)


@app.exception_handler(UpstreamProxyError)
async def handle_upstream_proxy_error(_: Request, exc: UpstreamProxyError):
    """Handle UpstreamProxyError by returning a JSON error response."""
    return JSONResponse({"error": exc.message}, status_code=exc.status_code)


def _http_client() -> httpx.AsyncClient:
    return app.state.http_client


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _model_family(model_name: str) -> str:
    if ADAPTER_MODEL_ARCHITECTURE:
        return ADAPTER_MODEL_ARCHITECTURE
    if not model_name:
        return "unknown"
    return model_name.split("/")[-1].split("-")[0].lower()


def _build_model_details(model_name: str) -> ModelDetails:
    family = _model_family(model_name)
    return ModelDetails(
        parent_model="",
        format="gguf",
        family=family,
        families=[family],
        parameter_size="unknown",
        quantization_level="unknown",
    )


def _build_model_info(model_name: str) -> ModelInfo:
    return ModelInfo(
        name=model_name,
        model=model_name,
        modified_at=_now_iso(),
        size=0,
        digest=DEFAULT_DIGEST,
        details=_build_model_details(model_name),
    )


def _build_process_model(model_name: str) -> ProcessModelInfo:
    return ProcessModelInfo(
        name=model_name,
        model=model_name,
        size=0,
        digest=DEFAULT_DIGEST,
        details=_build_model_details(model_name),
        context_length=ADAPTER_MODEL_CONTEXT_LENGTH,
        expires_at=_now_iso(),
        size_vram=0,
    )


def _build_openai_model_card(
    model_name: str, source: dict[str, Any] | None = None
) -> dict[str, Any]:
    card = dict(source or {})
    card["id"] = model_name
    card.setdefault("object", "model")
    card.setdefault("created", 0)
    card.setdefault("owned_by", UPSTREAM_OWNER)
    return card


def _resolve_public_model(model: str) -> str:
    return ADAPTER_EXPOSED_MODEL_NAME or VLLM_MODEL_NAME or model


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _normalize_openai_model_cards(data: dict[str, Any]) -> list[dict[str, Any]]:
    source_cards = [item for item in data.get("data", []) if isinstance(item, dict)]
    preferred_model = _resolve_public_model("")

    if preferred_model:
        first_source = source_cards[0] if source_cards else None
        return [_build_openai_model_card(preferred_model, first_source)]

    cards: list[dict[str, Any]] = []

    for source in source_cards:
        model_name = str(source.get("id", "")).strip()
        if model_name:
            cards.append(_build_openai_model_card(model_name, source))

    if not cards:
        fallback_name = _resolve_public_model("") or _resolve_model("")
        cards.append(_build_openai_model_card(fallback_name))

    deduped_names = _dedupe_strings([str(card.get("id", "")).strip() for card in cards])
    first_by_name = {
        str(card.get("id", "")).strip(): card
        for card in cards
        if str(card.get("id", "")).strip()
    }
    return [first_by_name[name] for name in deduped_names]


def _extract_first_string(payload: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _extract_nested_error_message(payload: dict[str, Any]) -> str | None:
    error = payload.get("error")
    if isinstance(error, str) and error.strip():
        return error
    if isinstance(error, dict):
        return _extract_first_string(error, ("message", "detail"))
    return None


def _extract_error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        text = response.text.strip()
        return text or f"Upstream vLLM request failed with HTTP {response.status_code}."

    if isinstance(payload, dict):
        return (
            _extract_nested_error_message(payload)
            or _extract_first_string(payload, ("message", "detail"))
            or f"Upstream vLLM request failed with HTTP {response.status_code}."
        )

    return f"Upstream vLLM request failed with HTTP {response.status_code}."


def _raise_request_error(exc: httpx.RequestError) -> None:
    raise UpstreamProxyError(503, f"Unable to reach vLLM at {VLLM_BASE_URL}: {exc}") from exc


def _raise_response_error(response: httpx.Response) -> None:
    status_code = 502 if response.status_code >= 500 else response.status_code
    raise UpstreamProxyError(status_code, _extract_error_message(response))


async def _request_upstream_json(
    method: str, path: str, json_body: dict[str, Any] | None = None
) -> dict[str, Any]:
    try:
        response = await _http_client().request(method, path, json=json_body)
    except httpx.RequestError as exc:
        _raise_request_error(exc)

    if response.is_error:
        _raise_response_error(response)

    try:
        return response.json()
    except ValueError as exc:
        raise UpstreamProxyError(502, f"Upstream vLLM returned invalid JSON for {path}.") from exc


async def _open_upstream_stream(
    method: str, path: str, json_body: dict[str, Any] | None = None
) -> httpx.Response:
    request = _http_client().build_request(method, path, json=json_body)
    try:
        response = await _http_client().send(request, stream=True)
    except httpx.RequestError as exc:
        _raise_request_error(exc)

    if response.is_error:
        await response.aread()
        try:
            _raise_response_error(response)
        finally:
            await response.aclose()

    return response


async def _iter_sse_lines(response: httpx.Response):
    try:
        async for line in response.aiter_lines():
            if line:
                yield line
    finally:
        await response.aclose()


def _extract_sse_payload(line: str) -> str | None:
    if not line.startswith("data:"):
        return None
    return line[5:].lstrip()


def _extract_extra_fields(
    payload: dict[str, Any], excluded_keys: set[str]
) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in excluded_keys and value is not None
    }


def _message_from_payload(payload: dict[str, Any], default_role: str = "assistant") -> ChatMessage:
    message = {
        "role": payload.get("role") or default_role,
        "content": payload.get("content"),
        **_extract_extra_fields(payload, {"role", "content"}),
    }
    return ChatMessage.model_validate(message)


def _extract_options(options: dict | None) -> dict:
    if not options:
        return {}
    mapping = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "num_predict": "max_tokens",
        "stop": "stop",
        "seed": "seed",
        "repeat_penalty": "repetition_penalty",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
    }
    params: dict = {}
    for ollama_key, openai_key in mapping.items():
        if ollama_key in options:
            params[openai_key] = options[ollama_key]
    return params


def _resolve_model(model: str) -> str:
    return VLLM_MODEL_NAME or model


@app.get("/")
async def health():
    """Return a simple health-check response."""
    return JSONResponse({"status": "ok"})


@app.get("/api/version")
async def version():
    """Return the current Ollama API version."""
    return JSONResponse({"version": OLLAMA_VERSION})


@app.get("/api/tags")
async def list_models():
    """List available models (proxied to vLLM models endpoint)."""
    try:
        data = await _request_upstream_json("GET", MODELS_PATH)
        cards = _normalize_openai_model_cards(data)
        models = [_build_model_info(str(card["id"])) for card in cards]
    except UpstreamProxyError:
        models = [_build_model_info(_resolve_public_model("unknown"))]
    return TagsResponse(models=models).model_dump()


@app.get("/api/ps")
async def list_running_models():
    """List running models (proxied to vLLM models endpoint)."""
    try:
        data = await _request_upstream_json("GET", MODELS_PATH)
        cards = _normalize_openai_model_cards(data)
        models = [_build_process_model(str(card["id"])) for card in cards]
    except UpstreamProxyError:
        models = [_build_process_model(_resolve_public_model("unknown"))]
    return PsResponse(models=models).model_dump()


@app.post("/api/show")
async def show_model(req: ShowRequest):
    """Return metadata and details for a model."""
    public_model_name = _resolve_public_model(req.name)
    upstream_model_name = _resolve_model(req.name)
    arch = _model_family(public_model_name)
    return ShowResponse(
        modelfile=f"FROM {public_model_name}",
        parameters="",
        template="",
        details=_build_model_details(public_model_name),
        model_info={
            "general.architecture": arch,
            "general.basename": public_model_name,
            f"{arch}.context_length": ADAPTER_MODEL_CONTEXT_LENGTH,
            "general.file_type": 2,
            "general.parameter_count": 0,
            "general.quantization_version": 2,
        },
        capabilities=["completion", "tools","vision"],
        remote_model=upstream_model_name,
    ).model_dump()


@app.post("/api/pull")
async def pull_model(req: PullRequest):
    """Simulate pulling a model (no-op for vLLM proxy)."""
    if req.stream:

        async def _stream():
            yield json.dumps(PullResponse(status="pulling manifest").model_dump()) + "\n"
            yield json.dumps(PullResponse(status="success").model_dump()) + "\n"

        return StreamingResponse(_stream(), media_type=NDJSON_MEDIA_TYPE)
    return PullResponse(status="success").model_dump()


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    """Generate a completion for the given prompt."""
    public_model = _resolve_public_model(req.model)
    model = _resolve_model(req.model)
    params = _extract_options(req.options)
    extra = req.model_extra or {}

    prompt = req.prompt
    if req.system and not req.raw:
        prompt = f"{req.system}\n\n{prompt}"

    payload = {
        **_extract_extra_fields(extra, {"options"}),
        "model": model,
        "prompt": prompt,
        "stream": req.stream,
        **params,
    }

    if req.stream:
        response = await _open_upstream_stream(
            "POST", COMPLETIONS_PATH, json_body=payload
        )
        return StreamingResponse(
            _stream_generate(response, public_model), media_type=NDJSON_MEDIA_TYPE
        )

    data = await _request_upstream_json("POST", COMPLETIONS_PATH, json_body=payload)

    text = data["choices"][0]["text"]
    usage = data.get("usage", {})
    finish_reason = data["choices"][0].get("finish_reason")

    return GenerateResponse(
        model=public_model,
        created_at=_now_iso(),
        response=text,
        done=True,
        done_reason=finish_reason,
        prompt_eval_count=usage.get("prompt_tokens"),
        eval_count=usage.get("completion_tokens"),
    ).model_dump()


async def _stream_generate(response: httpx.Response, model: str):
    """Stream generate responses from an upstream SSE response."""
    finish_reason: str | None = None
    async for line in _iter_sse_lines(response):
        payload = _extract_sse_payload(line)
        if payload is None:
            continue
        if payload == "[DONE]":
            break

        chunk = json.loads(payload)
        choice = chunk["choices"][0]
        text = choice.get("text", "")
        finish_reason = choice.get("finish_reason") or finish_reason

        if text:
            out = GenerateResponse(
                model=model, created_at=_now_iso(), response=text, done=False
            )
            yield json.dumps(out.model_dump(exclude_none=True)) + "\n"

    final = GenerateResponse(
        model=model,
        created_at=_now_iso(),
        response="",
        done=True,
        done_reason=finish_reason,
    )
    yield json.dumps(final.model_dump()) + "\n"


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Generate a chat completion for the given messages."""
    public_model = _resolve_public_model(req.model)
    model = _resolve_model(req.model)
    params = _extract_options(req.options)
    extra = req.model_extra or {}

    messages = [message.model_dump(exclude_none=True) for message in req.messages]
    payload = {
        **_extract_extra_fields(extra, {"options"}),
        "model": model,
        "messages": messages,
        "stream": req.stream,
        **params,
    }

    if req.stream:
        response = await _open_upstream_stream("POST", CHAT_COMPLETIONS_PATH, json_body=payload)
        return StreamingResponse(_stream_chat(response, public_model), media_type=NDJSON_MEDIA_TYPE)

    data = await _request_upstream_json("POST", CHAT_COMPLETIONS_PATH, json_body=payload)

    choice = data["choices"][0]
    msg = choice["message"]
    usage = data.get("usage", {})

    return ChatResponse(
        model=public_model,
        created_at=_now_iso(),
        message=_message_from_payload(msg),
        done=True,
        done_reason=choice.get("finish_reason"),
        prompt_eval_count=usage.get("prompt_tokens"),
        eval_count=usage.get("completion_tokens"),
    ).model_dump()


async def _stream_chat(response: httpx.Response, model: str):
    """Stream chat responses from an upstream SSE response."""
    finish_reason: str | None = None
    async for line in _iter_sse_lines(response):
        payload = _extract_sse_payload(line)
        if payload is None:
            continue
        if payload == "[DONE]":
            break

        chunk = json.loads(payload)
        choice = chunk["choices"][0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason") or finish_reason

        if delta:
            out = ChatResponse(
                model=model,
                created_at=_now_iso(),
                message=_message_from_payload(delta),
                done=False,
            )
            yield json.dumps(out.model_dump(exclude_none=True)) + "\n"

    final = ChatResponse(
        model=model,
        created_at=_now_iso(),
        message=ChatMessage(role="assistant"),
        done=True,
        done_reason=finish_reason,
    )
    yield json.dumps(final.model_dump()) + "\n"


@app.post("/api/embeddings")
@app.post("/api/embed")
async def embeddings(req: EmbeddingsRequest):
    """Generate embeddings for the given input text."""
    model = _resolve_model(req.model)
    input_text = req.prompt or req.input or ""

    if isinstance(input_text, list):
        inputs = input_text
    else:
        inputs = [input_text]

    payload = {**(req.model_extra or {}), "model": model, "input": inputs}
    data = await _request_upstream_json("POST", EMBEDDINGS_PATH, json_body=payload)

    embeddings_list = [item["embedding"] for item in data.get("data", [])]

    if len(embeddings_list) == 1:
        return EmbeddingsResponse(embedding=embeddings_list[0]).model_dump()
    return EmbeddingsResponse(embeddings=embeddings_list).model_dump()


@app.post("/v1/chat/completions")
async def openai_chat_passthrough(request: Request):
    """Proxy OpenAI-compatible chat completions to vLLM."""
    body = await request.json()
    body["model"] = _resolve_model(body.get("model", ""))
    stream = body.get("stream", False)

    if stream:
        response = await _open_upstream_stream("POST", CHAT_COMPLETIONS_PATH, json_body=body)

        async def _proxy_stream():
            async for line in _iter_sse_lines(response):
                yield line + "\n"

        return StreamingResponse(_proxy_stream(), media_type=SSE_MEDIA_TYPE)

    return JSONResponse(await _request_upstream_json("POST", CHAT_COMPLETIONS_PATH, json_body=body))


@app.post("/v1/completions")
async def openai_completions_passthrough(request: Request):
    """Proxy OpenAI-compatible completions to vLLM."""
    body = await request.json()
    body["model"] = _resolve_model(body.get("model", ""))
    stream = body.get("stream", False)

    if stream:
        response = await _open_upstream_stream("POST", COMPLETIONS_PATH, json_body=body)

        async def _proxy_stream():
            async for line in _iter_sse_lines(response):
                yield line + "\n"

        return StreamingResponse(_proxy_stream(), media_type=SSE_MEDIA_TYPE)

    return JSONResponse(await _request_upstream_json("POST", COMPLETIONS_PATH, json_body=body))


@app.post("/v1/embeddings")
async def openai_embeddings_passthrough(request: Request):
    """Proxy OpenAI-compatible embeddings to vLLM."""
    body = await request.json()
    body["model"] = _resolve_model(body.get("model", ""))
    return JSONResponse(await _request_upstream_json("POST", EMBEDDINGS_PATH, json_body=body))


@app.get("/v1/models")
async def openai_models_passthrough():
    """Proxy OpenAI-compatible models list to vLLM."""
    try:
        data = await _request_upstream_json("GET", MODELS_PATH)
        payload = {
            "object": data.get("object", "list"),
            "data": _normalize_openai_model_cards(data),
        }
    except UpstreamProxyError:
        payload = {
            "object": "list",
            "data": [_build_openai_model_card(_resolve_public_model("unknown"))],
        }
    return JSONResponse(payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=ADAPTER_HOST, port=ADAPTER_PORT)
