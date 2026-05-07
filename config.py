"""Configuration settings loaded from environment variables."""

import os

from dotenv import load_dotenv


load_dotenv()

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "").strip()
ADAPTER_EXPOSED_MODEL_NAME = os.getenv("ADAPTER_EXPOSED_MODEL_NAME", "").strip()
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "").strip()
VLLM_REQUEST_TIMEOUT = float(os.getenv("VLLM_REQUEST_TIMEOUT", "120"))
ADAPTER_MODEL_ARCHITECTURE = os.getenv("ADAPTER_MODEL_ARCHITECTURE", "").strip().lower()
ADAPTER_MODEL_CONTEXT_LENGTH = int(os.getenv("ADAPTER_MODEL_CONTEXT_LENGTH", "262144"))
ADAPTER_HOST = os.getenv("ADAPTER_HOST", "0.0.0.0")
ADAPTER_PORT = int(os.getenv("ADAPTER_PORT", "11434"))
OLLAMA_VERSION = "0.6.4"
