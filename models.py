from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, model_validator


class GenerateRequest(BaseModel):
    model: str
    prompt: str
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[list[int]] = None
    stream: bool = True
    raw: bool = False
    options: Optional[dict] = None
    keep_alive: Optional[str] = None


class GenerateResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool
    context: Optional[list[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class ChatMessage(BaseModel):
    role: str
    content: str
    images: Optional[list[str]] = None


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = True
    options: Optional[dict] = None
    keep_alive: Optional[str] = None


class ChatResponse(BaseModel):
    model: str
    created_at: str
    message: Optional[ChatMessage] = None
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class EmbeddingsRequest(BaseModel):
    model: str
    prompt: Optional[str] = None
    input: Optional[str | list[str]] = None


class EmbeddingsResponse(BaseModel):
    embedding: Optional[list[float]] = None
    embeddings: Optional[list[list[float]]] = None


class ModelDetails(BaseModel):
    parent_model: str = ""
    format: str = "gguf"
    family: str = ""
    families: Optional[list[str]] = None
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


class ShowRequest(BaseModel):
    name: Optional[str] = None
    model: Optional[str] = None
    verbose: Optional[bool] = False

    @model_validator(mode="after")
    def _resolve_name(self):
        self.name = self.name or self.model or ""
        return self


class ShowResponse(BaseModel):
    modelfile: str = ""
    parameters: str = ""
    template: str = ""
    details: ModelDetails
    model_info: Optional[dict] = None


class PullRequest(BaseModel):
    name: str
    insecure: bool = False
    stream: bool = True


class PullResponse(BaseModel):
    status: str
    digest: Optional[str] = None
    total: Optional[int] = None
    completed: Optional[int] = None
