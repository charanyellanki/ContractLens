"""LangGraph state machine for ContractLens orchestration."""

from dataclasses import dataclass, field
from typing import Optional

from contractlens.models import (
    Clause,
    ClauseCategory,
    Contract,
    ExtractionResult,
    Span,
    VerificationResult,
    VerificationStatus,
)


@dataclass
class ContractLensState:
    """State for the ContractLens LangGraph workflow."""

    # Input
    contract: Optional[Contract] = None
    target_categories: list[ClauseCategory] = field(default_factory=list)
    
    # Retrieval state
    retrieved_chunks: list[str] = field(default_factory=list)
    
    # Extraction state
    extracted_spans: dict[ClauseCategory, list[Span]] = field(default_factory=dict)
    extraction_attempts: int = 0
    
    # Verification state
    verification_results: list[VerificationResult] = field(default_factory=list)
    verified_clauses: list[Clause] = field(default_factory=list)
    rejected_spans: list[Span] = field(default_factory=list)
    
    # Retry state
    retry_count: int = 0
    max_retries: int = 2
    should_retry: bool = False
    
    # Output
    extraction_result: Optional[ExtractionResult] = None
    error: Optional[str] = None
    
    # Metadata
    model_used: str = "gpt-4o-mini"
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0


def create_initial_state(
    contract: Contract,
    categories: list[ClauseCategory],
    model: str = "gpt-4o-mini",
    max_retries: int = 2,
) -> ContractLensState:
    """Create initial state for a contract processing workflow."""
    return ContractLensState(
        contract=contract,
        target_categories=categories,
        model_used=model,
        max_retries=max_retries,
    )


def has_unverified_spans(state: ContractLensState) -> bool:
    """Check if there are spans that need verification."""
    for category, spans in state.extracted_spans.items():
        for span in spans:
            # Check if this span has been verified
            verified = any(
                v.span.start_char == span.start_char and v.span.end_char == span.end_char
                for v in state.verification_results
            )
            if not verified:
                return True
    return False


def get_unverified_spans(state: ContractLensState) -> list[tuple[Span, ClauseCategory]]:
    """Get all spans that haven't been verified yet."""
    unverified: list[tuple[Span, ClauseCategory]] = []
    for category, spans in state.extracted_spans.items():
        for span in spans:
            # Check if already verified
            verified = any(
                v.span.start_char == span.start_char 
                and v.span.end_char == span.end_char
                and v.span.category == span.category
                for v in state.verification_results
            )
            if not verified:
                unverified.append((span, category))
    return unverified


def should_continue_retrieval(state: ContractLensState) -> bool:
    """Determine if retrieval should continue."""
    return state.contract is not None and not state.retrieved_chunks


def should_continue_extraction(state: ContractLensState) -> bool:
    """Determine if extraction should continue."""
    return bool(state.retrieved_chunks) and not state.extracted_spans


def should_continue_verification(state: ContractLensState) -> bool:
    """Determine if verification should continue."""
    return bool(state.extracted_spans) and has_unverified_spans(state)


def should_retry_extraction(state: ContractLensState) -> bool:
    """Determine if extraction should be retried."""
    if state.retry_count >= state.max_retries:
        return False
    
    # Check if we have too many rejected spans
    rejected_count = len(state.rejected_spans)
    total_extracted = sum(len(spans) for spans in state.extracted_spans.values())
    
    if total_extracted > 0 and rejected_count / total_extracted > 0.5:
        return True
    
    return False


def is_complete(state: ContractLensState) -> bool:
    """Check if the workflow is complete."""
    return (
        state.extraction_result is not None
        or state.error is not None
        or (state.retry_count >= state.max_retries and not has_unverified_spans(state))
    )