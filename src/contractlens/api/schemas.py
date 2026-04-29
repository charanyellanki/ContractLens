"""API schemas for request/response models."""

from typing import Optional

from pydantic import BaseModel, Field


class ExtractRequest(BaseModel):
    """Request for clause extraction."""

    contract_text: str = Field(..., description="Full contract text to analyze")
    categories: Optional[list[str]] = Field(
        None,
        description="Specific clause categories to extract (default: all 41 CUAD categories)",
    )
    model: str = Field(
        "gpt-4o-mini",
        description="LLM model to use for extraction",
    )
    include_verification: bool = Field(
        True,
        description="Whether to verify extracted spans against source text",
    )


class ExtractResponse(BaseModel):
    """Response for clause extraction."""

    contract_id: str = Field(..., description="Unique identifier for the contract")
    clauses: list[dict] = Field(..., description="Extracted clauses")
    model_used: str = Field(..., description="LLM model used")
    extraction_time_ms: float = Field(..., description="Total extraction time")
    cost_usd: float = Field(..., description="Total cost in USD")
    verified: bool = Field(..., description="Whether verification was performed")


class EvaluateRequest(BaseModel):
    """Request for evaluation."""

    contract_text: str = Field(..., description="Contract text")
    ground_truth: list[dict] = Field(..., description="Ground truth clauses")
    model: str = Field("gpt-4o-mini", description="Model to evaluate")
    categories: Optional[list[str]] = Field(
        None, description="Categories to evaluate (default: all)"
    )


class EvaluateResponse(BaseModel):
    """Response for evaluation."""

    precision: float
    recall: float
    f1: float
    latency_ms: float
    cost_usd: float
    category_results: dict
    error_distribution: dict