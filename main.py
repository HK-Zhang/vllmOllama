from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import ADAPTER_HOST, ADAPTER_PORT, OLLAMA_VERSION, VLLM_BASE_URL, VLLM_MODEL_NAME
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
    PullRequest,
    PullResponse,
    ShowRequest,
    ShowResponse,
    TagsResponse,
)

app = FastAPI(title="vLLM-to-Ollama Adapter", version=OLLAMA_VERSION)

http_client = httpx.AsyncClient(base_url=VLLM_BASE_URL, timeout=120.0)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _extract_options(options: dict | None) -> dict:
    """Map Ollama options to OpenAI-compatible parameters."""
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
    """Use configured model name, falling back to the request model field."""
    return VLLM_MODEL_NAME or model


# --- Health / Version ---


@app.get("/")
async def health():
    return JSONResponse({"status": "ok"})


@app.get("/api/version")
async def version():
    return JSONResponse({"version": OLLAMA_VERSION})


# --- List Models ---


@app.get("/api/tags")
async def list_models():
    try:
        resp = await http_client.get("/v1/models")
        resp.raise_for_status()
        data = resp.json()
        models = []
        for m in data.get("data", []):
            models.append(
                ModelInfo(
                    name=m["id"],
                    model=m["id"],
                    modified_at=_now_iso(),
                    size=0,
                    digest="sha256:0000000000000000",
                    details=ModelDetails(
                        parent_model="",
                        format="gguf",
                        family=m["id"].split("/")[0] if "/" in m["id"] else m["id"],
                        parameter_size="unknown",
                        quantization_level="unknown",
                    ),
                )
            )
        return TagsResponse(models=models).model_dump()
    except httpx.HTTPError:
        model_name = _resolve_model("")
        fallback = ModelInfo(
            name=model_name,
            model=model_name,
            modified_at=_now_iso(),
            size=0,
            digest="sha256:0000000000000000",
            details=ModelDetails(family=model_name, parameter_size="unknown", quantization_level="unknown"),
        )
        return TagsResponse(models=[fallback]).model_dump()


# --- Show Model ---


@app.post("/api/show")
async def show_model(req: ShowRequest):
    model_name = _resolve_model(req.name)
    return ShowResponse(
        modelfile=f"FROM {model_name}",
        parameters="",
        template="",
        details=ModelDetails(
            parent_model="",
            format="gguf",
            family=model_name,
            parameter_size="unknown",
            quantization_level="unknown",
        ),
        model_info={"general.architecture": model_name},
    ).model_dump()


# --- Pull Model (no-op) ---


@app.post("/api/pull")
async def pull_model(req: PullRequest):
    if req.stream:

        async def _stream():
            yield json.dumps(PullResponse(status="pulling manifest").model_dump()) + "\n"
            yield json.dumps(PullResponse(status="success").model_dump()) + "\n"

        return StreamingResponse(_stream(), media_type="application/x-ndjson")
    return PullResponse(status="success").model_dump()


# --- Generate (Completion) ---


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    model = _resolve_model(req.model)
    params = _extract_options(req.options)

    prompt = req.prompt
    if req.system:
        prompt = f"{req.system}\n\n{prompt}"

    payload = {"model": model, "prompt": prompt, "stream": req.stream, **params}

    if req.stream:
        return StreamingResponse(_stream_generate(payload, model), media_type="application/x-ndjson")

    resp = await http_client.post("/v1/completions", json=payload)
    resp.raise_for_status()
    data = resp.json()

    text = data["choices"][0]["text"]
    usage = data.get("usage", {})

    return GenerateResponse(
        model=model,
        created_at=_now_iso(),
        response=text,
        done=True,
        prompt_eval_count=usage.get("prompt_tokens"),
        eval_count=usage.get("completion_tokens"),
    ).model_dump()


async def _stream_generate(payload: dict, model: str):
    async with http_client.stream("POST", "/v1/completions", json=payload) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            chunk_str = line[6:]
            if chunk_str.strip() == "[DONE]":
                break
            chunk = json.loads(chunk_str)
            text = chunk["choices"][0].get("text", "")
            out = GenerateResponse(model=model, created_at=_now_iso(), response=text, done=False)
            yield json.dumps(out.model_dump()) + "\n"

    final = GenerateResponse(model=model, created_at=_now_iso(), response="", done=True)
    yield json.dumps(final.model_dump()) + "\n"


# --- Chat ---


@app.post("/api/chat")
async def chat(req: ChatRequest):
    model = _resolve_model(req.model)
    params = _extract_options(req.options)

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    payload = {"model": model, "messages": messages, "stream": req.stream, **params}

    if req.stream:
        return StreamingResponse(_stream_chat(payload, model), media_type="application/x-ndjson")

    resp = await http_client.post("/v1/chat/completions", json=payload)
    resp.raise_for_status()
    data = resp.json()

    choice = data["choices"][0]
    msg = choice["message"]
    usage = data.get("usage", {})

    return ChatResponse(
        model=model,
        created_at=_now_iso(),
        message=ChatMessage(role=msg["role"], content=msg["content"]),
        done=True,
        prompt_eval_count=usage.get("prompt_tokens"),
        eval_count=usage.get("completion_tokens"),
    ).model_dump()


async def _stream_chat(payload: dict, model: str):
    async with http_client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            chunk_str = line[6:]
            if chunk_str.strip() == "[DONE]":
                break
            chunk = json.loads(chunk_str)
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            out = ChatResponse(
                model=model,
                created_at=_now_iso(),
                message=ChatMessage(role="assistant", content=content),
                done=False,
            )
            yield json.dumps(out.model_dump()) + "\n"

    final = ChatResponse(model=model, created_at=_now_iso(), message=ChatMessage(role="assistant", content=""), done=True)
    yield json.dumps(final.model_dump()) + "\n"


# --- Embeddings ---


@app.post("/api/embeddings")
@app.post("/api/embed")
async def embeddings(req: EmbeddingsRequest):
    model = _resolve_model(req.model)
    input_text = req.prompt or req.input or ""

    if isinstance(input_text, list):
        inputs = input_text
    else:
        inputs = [input_text]

    payload = {"model": model, "input": inputs}
    resp = await http_client.post("/v1/embeddings", json=payload)
    resp.raise_for_status()
    data = resp.json()

    embeddings_list = [item["embedding"] for item in data.get("data", [])]

    if len(embeddings_list) == 1:
        return EmbeddingsResponse(embedding=embeddings_list[0]).model_dump()
    return EmbeddingsResponse(embeddings=embeddings_list).model_dump()


# --- Copilot compatibility: OpenAI passthrough ---
# GitHub Copilot may also hit OpenAI-compatible endpoints directly.


@app.post("/v1/chat/completions")
async def openai_chat_passthrough(request: Request):
    body = await request.json()
    body["model"] = _resolve_model(body.get("model", ""))
    stream = body.get("stream", False)

    if stream:

        async def _proxy_stream():
            async with http_client.stream("POST", "/v1/chat/completions", json=body) as resp:
                async for line in resp.aiter_lines():
                    yield line + "\n"

        return StreamingResponse(_proxy_stream(), media_type="text/event-stream")

    resp = await http_client.post("/v1/chat/completions", json=body)
    resp.raise_for_status()
    return JSONResponse(resp.json())


@app.post("/v1/completions")
async def openai_completions_passthrough(request: Request):
    body = await request.json()
    body["model"] = _resolve_model(body.get("model", ""))
    stream = body.get("stream", False)

    if stream:

        async def _proxy_stream():
            async with http_client.stream("POST", "/v1/completions", json=body) as resp:
                async for line in resp.aiter_lines():
                    yield line + "\n"

        return StreamingResponse(_proxy_stream(), media_type="text/event-stream")

    resp = await http_client.post("/v1/completions", json=body)
    resp.raise_for_status()
    return JSONResponse(resp.json())


@app.get("/v1/models")
async def openai_models_passthrough():
    resp = await http_client.get("/v1/models")
    resp.raise_for_status()
    return JSONResponse(resp.json())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=ADAPTER_HOST, port=ADAPTER_PORT)
