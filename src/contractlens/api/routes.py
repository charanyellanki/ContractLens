"""FastAPI routes for ContractLens API."""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from contractlens.models import ClauseCategory, Contract


router = APIRouter(prefix="/api/v1", tags=["contractlens"])


# Request/Response schemas
class ExtractRequest(BaseModel):
    """Request for clause extraction."""

    contract_text: str = Field(..., description="Full contract text")
    categories: Optional[list[str]] = Field(
        None, description="Clause categories to extract (default: all)"
    )
    model: str = Field("gpt-4o-mini", description="LLM model to use")


class ClauseSpan(BaseModel):
    """A extracted clause span."""

    text: str
    start_char: int
    end_char: int
    category: str
    confidence: float


class ExtractResponse(BaseModel):
    """Response for clause extraction."""

    contract_id: str
    clauses: list[ClauseSpan]
    model_used: str
    extraction_time_ms: float
    cost_usd: float


class EvaluateRequest(BaseModel):
    """Request for evaluation."""

    contract_text: str
    ground_truth: list[dict]
    model: str = "gpt-4o-mini"


class EvaluateResponse(BaseModel):
    """Response for evaluation."""

    precision: float
    recall: float
    f1: float
    latency_ms: float
    cost_usd: float
    category_results: dict


@router.post("/extract", response_model=ExtractResponse)
async def extract_clauses(request: ExtractRequest) -> ExtractResponse:
    """Extract clauses from a contract."""
    # TODO: Implement actual extraction
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_extraction(request: EvaluateRequest) -> EvaluateResponse:
    """Evaluate extraction against ground truth."""
    # TODO: Implement actual evaluation
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "service": "contractlens"}


@router.get("/categories")
async def list_categories() -> dict:
    """List all available clause categories."""
    return {
        "categories": [cat.value for cat in ClauseCategory]
    }