"""Pydantic models for Ollama-compatible API request and response payloads."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class CompatibleBaseModel(BaseModel):
    """Base model that allows extra fields for compatibility."""

    model_config = ConfigDict(extra="allow")


class GenerateRequest(CompatibleBaseModel):
    """Request payload for the /api/generate endpoint."""

    model: str = ""
    prompt: str
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[list[int]] = None
    stream: bool = True
    raw: bool = False
    options: Optional[dict] = None
    keep_alive: Optional[str] = None


class GenerateResponse(CompatibleBaseModel):
    """Response payload for the /api/generate endpoint."""

    model: str
    created_at: str
    response: str
    done: bool
    done_reason: Optional[str] = None
    context: Optional[list[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class ChatMessage(CompatibleBaseModel):
    """A single message in a chat conversation."""

    role: str
    content: Optional[str] = None
    images: Optional[list[str]] = None


class ChatRequest(CompatibleBaseModel):
    """Request payload for the /api/chat endpoint."""

    model: str = ""
    messages: list[ChatMessage]
    stream: bool = True
    options: Optional[dict] = None
    keep_alive: Optional[str] = None


class ChatResponse(CompatibleBaseModel):
    """Response payload for the /api/chat endpoint."""

    model: str
    created_at: str
    message: Optional[ChatMessage] = None
    done: bool
    done_reason: Optional[str] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class EmbeddingsRequest(CompatibleBaseModel):
    """Request payload for the /api/embeddings endpoint."""

    model: str = ""
    prompt: Optional[str] = None
    input: Optional[str | list[str]] = None


class EmbeddingsResponse(CompatibleBaseModel):
    """Response payload for the /api/embeddings endpoint."""

    embedding: Optional[list[float]] = None
    embeddings: Optional[list[list[float]]] = None


class ModelDetails(BaseModel):
    """Detailed metadata about a model."""

    parent_model: str = ""
    format: str = "gguf"
    family: str = ""
    families: list[str] = Field(default_factory=list)
    parameter_size: str = ""
    quantization_level: str = ""


class ModelInfo(BaseModel):
    """Information about a single model for listing endpoints."""

    name: str
    model: str
    modified_at: str
    size: int
    digest: str
    details: ModelDetails


class TagsResponse(BaseModel):
    """Response payload for the /api/tags endpoint."""

    models: list[ModelInfo]


class ProcessModelInfo(BaseModel):
    """Information about a running model for the /api/ps endpoint."""

    name: str
    model: str
    size: int
    digest: str
    details: ModelDetails
    context_length: int
    expires_at: str
    size_vram: int = 0


class PsResponse(BaseModel):
    """Response payload for the /api/ps endpoint."""

    models: list[ProcessModelInfo]


class ShowRequest(CompatibleBaseModel):
    """Request payload for the /api/show endpoint."""

    name: Optional[str] = None
    model: Optional[str] = None
    verbose: Optional[bool] = False

    @model_validator(mode="after")
    def _resolve_name(self):
        self.name = self.name or self.model or ""
        return self


class ShowResponse(CompatibleBaseModel):
    """Response payload for the /api/show endpoint."""

    modelfile: str = ""
    parameters: str = ""
    template: str = ""
    details: ModelDetails
    model_info: Optional[dict] = None
    capabilities: list[str] = Field(default_factory=lambda: ["completion"])


class PullRequest(CompatibleBaseModel):
    """Request payload for the /api/pull endpoint."""

    name: str
    insecure: bool = False
    stream: bool = True


class PullResponse(CompatibleBaseModel):
    """Response payload for the /api/pull endpoint."""

    status: str
    digest: Optional[str] = None
    total: Optional[int] = None
    completed: Optional[int] = None
