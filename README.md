# vLLM-to-Ollama Adapter

A lightweight FastAPI adapter that exposes a vLLM service as an Ollama-compatible API. It is designed for GitHub Copilot and other clients that expect either the Ollama API surface or OpenAI-compatible endpoints.

## What improved for GitHub Copilot

- Preserves extra chat and completion fields instead of dropping them, which improves compatibility with Copilot features that send OpenAI-style options such as `tools`, `tool_choice`, or message metadata.
- Normalizes model discovery so the configured `VLLM_MODEL_NAME` is exposed consistently through both `/api/tags` and `/v1/models`.
- Adds `/api/ps` and `/v1/embeddings`, which closes common discovery and embeddings gaps for Ollama-style and OpenAI-style clients.
- Exposes the configured context length through both `/api/show` and `/api/ps`, which helps clients avoid falling back to conservative defaults such as ~32K.
- Supports a public model alias that is separate from the upstream vLLM model ID, which helps when Copilot caches Ollama model metadata by model name.
- Returns token usage for non-streaming calls and includes usage metrics in the final Ollama-style streaming chunk when the upstream stream provides them.
- Returns cleaner upstream failures instead of raw FastAPI stack traces when vLLM is unavailable or returns an error.
- Supports upstream bearer authentication and a configurable upstream timeout.

## Architecture

```text
GitHub Copilot / Client
        │
        ▼
┌─────────────────────┐
│  vLLM-Ollama Adapter│  (FastAPI, port 11434)
│  Ollama API compat  │
└────────┬────────────┘
         │  HTTP (OpenAI-compatible)
         ▼
┌─────────────────────┐
│     vLLM Server     │  (port 8000)
└─────────────────────┘
```

## Quick Start

### Local

```bash
pip install -r requirements.txt
VLLM_BASE_URL=http://localhost:8000 VLLM_MODEL_NAME=my-model python main.py
```

You can also create a `.env` file in the project root; `python-dotenv` loads it automatically when the app starts.

### Docker

```bash
docker build -t vllm-ollama-adapter .
docker run -p 11434:11434 \
  -e VLLM_BASE_URL=http://host.docker.internal:8000 \
  -e VLLM_MODEL_NAME=my-model \
  vllm-ollama-adapter
```

## Configuration

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `VLLM_BASE_URL` | `http://localhost:8000` | vLLM server base URL |
| `VLLM_MODEL_NAME` | empty | Upstream vLLM model ID used for proxied requests |
| `ADAPTER_EXPOSED_MODEL_NAME` | empty | Optional public model name exposed to Copilot and other clients |
| `VLLM_API_KEY` | empty | Optional bearer token forwarded to the upstream vLLM server |
| `VLLM_REQUEST_TIMEOUT` | `120` | Timeout in seconds for upstream vLLM requests |
| `ADAPTER_MODEL_ARCHITECTURE` | empty | Optional override for the Ollama architecture token reported in `/api/show` |
| `ADAPTER_MODEL_CONTEXT_LENGTH` | `262144` | Context length reported to Ollama-compatible clients |
| `ADAPTER_HOST` | `0.0.0.0` | Adapter listen address |
| `ADAPTER_PORT` | `11434` | Adapter listen port (Ollama default) |

## API Endpoints

### Ollama-Compatible Endpoints

#### `GET /`

Health check.

**Response:**

```json
{"status": "ok"}
```

---

#### `GET /api/version`

Returns the emulated Ollama version.

**Response:**

```json
{"version": "0.6.4"}
```

---

#### `GET /api/tags`

List available models. Queries the vLLM `/v1/models` endpoint and returns results in Ollama format.

If `ADAPTER_EXPOSED_MODEL_NAME` is set, the adapter exposes only that public model name. This is useful for forcing Copilot to register a fresh model entry after metadata changes.

**Response:**

```json
{
  "models": [
    {
      "name": "my-model",
      "model": "my-model",
      "modified_at": "2025-01-01T00:00:00.000000Z",
      "size": 0,
      "digest": "sha256:0000000000000000",
      "details": {
        "parent_model": "",
        "format": "gguf",
        "family": "my-model",
        "parameter_size": "unknown",
        "quantization_level": "unknown"
      }
    }
  ]
}
```

---

#### `POST /api/show`

Show model information.

**Request body:**

```json
{"name": "my-model"}
```

**Response:**

```json
{
  "modelfile": "FROM my-model",
  "parameters": "",
  "template": "",
  "details": {
    "parent_model": "",
    "format": "gguf",
    "family": "my-model",
    "parameter_size": "unknown",
    "quantization_level": "unknown"
  }
}
```

---

#### `GET /api/ps`

List models that appear loaded from the adapter's perspective. This improves compatibility with clients that probe Ollama's running-models endpoint before starting a chat.

**Response:**

```json
{
  "models": [
    {
      "name": "my-model",
      "model": "my-model",
      "size": 0,
      "digest": "sha256:0000000000000000",
      "details": {
        "parent_model": "",
        "format": "gguf",
        "family": "my",
        "families": ["my"],
        "parameter_size": "unknown",
        "quantization_level": "unknown"
      },
      "context_length": 262144,
      "expires_at": "2026-01-01T00:00:00.000000Z",
      "size_vram": 0
    }
  ]
}
```

---

#### `POST /api/pull`

Simulated model pull (no-op since vLLM already serves the model). Returns success immediately.

**Request body:**

```json
{"name": "my-model", "stream": false}
```

**Response:**

```json
{"status": "success"}
```

---

#### `POST /api/generate`

Generate a text completion. Supports streaming (NDJSON) and non-streaming modes.

**Request body:**

```json
{
  "model": "my-model",
  "prompt": "Why is the sky blue?",
  "stream": false,
  "options": {
    "temperature": 0.7,
    "num_predict": 256
  }
}
```

**Response (non-streaming):**

```json
{
  "model": "my-model",
  "created_at": "2025-01-01T00:00:00.000000Z",
  "response": "The sky appears blue because...",
  "done": true,
  "prompt_eval_count": 12,
  "eval_count": 64
}
```

**Response (streaming):** Newline-delimited JSON objects with `done: false` until the final object with `done: true`.

The final streaming object includes token usage metrics such as `prompt_eval_count` and `eval_count` when the upstream vLLM stream provides usage data.

---

#### `POST /api/chat`

Chat completion with message history. Supports streaming (NDJSON) and non-streaming modes.

Extra request fields are forwarded upstream, so OpenAI-style Copilot options such as `tools`, `tool_choice`, `response_format`, and extra message fields can pass through instead of being discarded.

**Request body:**

```json
{
  "model": "my-model",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": false,
  "options": {
    "temperature": 0.7
  }
}
```

**Response (non-streaming):**

```json
{
  "model": "my-model",
  "created_at": "2025-01-01T00:00:00.000000Z",
  "message": {"role": "assistant", "content": "Hello! How can I help you?"},
  "done": true,
  "prompt_eval_count": 20,
  "eval_count": 8
}
```

**Response (streaming):** The final streamed JSON object includes `prompt_eval_count` and `eval_count` when the upstream stream provides usage data.

---

#### `POST /api/embeddings` | `POST /api/embed`

Generate embeddings for text input.

**Request body:**

```json
{"model": "my-model", "prompt": "The quick brown fox"}
```

**Response:**

```json
{"embedding": [0.123, -0.456, ...]}
```

---

### OpenAI-Compatible Passthrough Endpoints

These endpoints proxy directly to the vLLM server, providing compatibility with clients that use the OpenAI API format (including GitHub Copilot's OpenAI mode).

| Endpoint | Method | Description |
| -------- | ------ | ----------- |
| `/v1/chat/completions` | POST | Chat completions (streaming supported) |
| `/v1/completions` | POST | Text completions (streaming supported) |
| `/v1/embeddings` | POST | Embeddings passthrough |
| `/v1/models` | GET | List models |

---

### Options Mapping

Ollama `options` fields are mapped to OpenAI-compatible parameters:

| Ollama Option | OpenAI Parameter |
| ------------- | ---------------- |
| `temperature` | `temperature` |
| `top_p` | `top_p` |
| `top_k` | `top_k` |
| `num_predict` | `max_tokens` |
| `stop` | `stop` |
| `seed` | `seed` |
| `repeat_penalty` | `repetition_penalty` |
| `frequency_penalty` | `frequency_penalty` |
| `presence_penalty` | `presence_penalty` |

## GitHub Copilot Integration

Configure GitHub Copilot to use this adapter as an Ollama endpoint:

1. Start the adapter pointing at your vLLM server.
2. Set `VLLM_MODEL_NAME` to the real upstream vLLM model ID.
3. If you want Copilot to see a friendlier name or force a fresh registration, set `ADAPTER_EXPOSED_MODEL_NAME`.
4. If Copilot still shows the wrong context window, set `ADAPTER_MODEL_ARCHITECTURE` to the exact Ollama architecture token for your model and `ADAPTER_MODEL_CONTEXT_LENGTH` to the real value.
5. In VS Code settings, configure the Ollama endpoint URL to `http://localhost:11434`.
6. Reload the VS Code window after changing the exposed model name or context metadata so Copilot drops any stale cached Ollama model info.
7. The adapter exposes both Ollama-native and OpenAI-compatible endpoints, so it works whether Copilot probes Ollama routes, OpenAI routes, or both.

### Recommended `.env` for Copilot

```env
VLLM_BASE_URL=http://localhost:8000
VLLM_MODEL_NAME=my-model
# Optional public alias for clients such as Copilot
ADAPTER_EXPOSED_MODEL_NAME=my-model-262k
# Optional when your upstream vLLM server requires auth
VLLM_API_KEY=
# Optional when prompts or context windows are large
VLLM_REQUEST_TIMEOUT=120
# Optional when the model name does not imply the right Ollama architecture token
ADAPTER_MODEL_ARCHITECTURE=llama
ADAPTER_MODEL_CONTEXT_LENGTH=262144
ADAPTER_HOST=0.0.0.0
ADAPTER_PORT=11434
```

### Limitations

- Ollama metadata such as quantization, file type, and parameter count is synthetic because vLLM does not expose equivalent details in the same shape.
- If `ADAPTER_MODEL_ARCHITECTURE` is left empty, the adapter derives an architecture token from the model name. For unusual model names, that heuristic may be too loose for client-specific metadata parsing.
- VS Code's Copilot Ollama provider fetches `/api/tags` first and then usually calls `/api/show` per model, but it also caches model info by base URL and model ID during the session. If you only see `/api/tags`, you may be hitting the cache.
- The adapter forwards tool-related request fields and tool-call responses, but model-side tool behavior still depends on what the upstream vLLM model actually supports.

## Development

```bash
pip install -r requirements.txt
python main.py
```

The server starts on port 11434 by default. FastAPI auto-generates interactive docs at `/docs` (Swagger UI) and `/redoc`.
