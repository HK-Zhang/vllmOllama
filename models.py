"""Pydantic models for Ollama-compatible API request and response payloads."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class CompatibleBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class GenerateRequest(CompatibleBaseModel):
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
    role: str
    content: Optional[str] = None
    images: Optional[list[str]] = None


class ChatRequest(CompatibleBaseModel):
    model: str = ""
    messages: list[ChatMessage]
    stream: bool = True
    options: Optional[dict] = None
    keep_alive: Optional[str] = None


class ChatResponse(CompatibleBaseModel):
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
    model: str = ""
    prompt: Optional[str] = None
    input: Optional[str | list[str]] = None


class EmbeddingsResponse(CompatibleBaseModel):
    embedding: Optional[list[float]] = None
    embeddings: Optional[list[list[float]]] = None


class ModelDetails(BaseModel):
    parent_model: str = ""
    format: str = "gguf"
    family: str = ""
    families: list[str] = Field(default_factory=list)
    parameter_size: str = ""
    quantization_level: str = ""


class ModelInfo(BaseModel):
    name: str
    model: str
    modified_at: str
    size: int
    digest: str
    details: ModelDetails


class TagsResponse(BaseModel):
    models: list[ModelInfo]


class ProcessModelInfo(BaseModel):
    name: str
    model: str
    size: int
    digest: str
    details: ModelDetails
    context_length: int
    expires_at: str
    size_vram: int = 0


class PsResponse(BaseModel):
    models: list[ProcessModelInfo]


class ShowRequest(CompatibleBaseModel):
    name: Optional[str] = None
    model: Optional[str] = None
    verbose: Optional[bool] = False

    @model_validator(mode="after")
    def _resolve_name(self):
        self.name = self.name or self.model or ""
        return self


class ShowResponse(CompatibleBaseModel):
    modelfile: str = ""
    parameters: str = ""
    template: str = ""
    details: ModelDetails
    model_info: Optional[dict] = None
    capabilities: list[str] = Field(default_factory=lambda: ["completion"])


class PullRequest(CompatibleBaseModel):
    name: str
    insecure: bool = False
    stream: bool = True


class PullResponse(CompatibleBaseModel):
    status: str
    digest: Optional[str] = None
    total: Optional[int] = None
    completed: Optional[int] = None
