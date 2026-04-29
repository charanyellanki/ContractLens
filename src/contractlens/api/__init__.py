"""API module for FastAPI backend service."""

from contractlens.api.routes import router
from contractlens.api.schemas import (
    ExtractRequest,
    ExtractResponse,
    EvaluateRequest,
    EvaluateResponse,
)

__all__ = [
    "router",
    "ExtractRequest",
    "ExtractResponse",
    "EvaluateRequest",
    "EvaluateResponse",
]