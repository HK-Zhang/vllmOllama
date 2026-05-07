# vLLM-to-Ollama Adapter

A lightweight FastAPI adapter that exposes a vLLM service as an Ollama-compatible API. Designed for use with GitHub Copilot and other tools that consume the Ollama protocol.

## Architecture

```
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
|----------|---------|-------------|
| `VLLM_BASE_URL` | `http://localhost:8000` | vLLM server base URL |
| `VLLM_MODEL_NAME` | `default-model` | Model name served by vLLM |
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
{"version": "0.6.2"}
```

---

#### `GET /api/tags`

List available models. Queries the vLLM `/v1/models` endpoint and returns results in Ollama format.

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

---

#### `POST /api/chat`

Chat completion with message history. Supports streaming (NDJSON) and non-streaming modes.

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
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (streaming supported) |
| `/v1/completions` | POST | Text completions (streaming supported) |
| `/v1/models` | GET | List models |

---

### Options Mapping

Ollama `options` fields are mapped to OpenAI-compatible parameters:

| Ollama Option | OpenAI Parameter |
|---------------|------------------|
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
2. In VS Code settings, configure the Ollama endpoint URL to `http://localhost:11434`.
3. The adapter exposes both Ollama-native and OpenAI-compatible endpoints, so it works regardless of which protocol Copilot uses.

## Development

```bash
pip install -r requirements.txt
python main.py
```

The server starts on port 11434 by default. FastAPI auto-generates interactive docs at `/docs` (Swagger UI) and `/redoc`.
