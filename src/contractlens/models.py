"""Pydantic models for ContractLens data contracts."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ClauseCategory(str, Enum):
    """CUAD clause categories (41 categories)."""

    # Confidentiality & Non-Disclosure
    CONFIDENTIALITY = "Confidentiality"
    NON_DISCLOSURE = "Non-Disclosure"
    NON_DISCLOSURE_AGREEMENT = "Non-Disclosure Agreement"
    
    # Intellectual Property
    IP_OWNERSHIP = "IP Ownership"
    LICENSE_GRANT = "License Grant"
    PATENT_RIGHTS = "Patent Rights"
    COPYRIGHT = "Copyright"
    TRADEMARK = "Trademark"
    
    # Liability & Indemnification
    LIMITATION_OF_LIABILITY = "Limitation of Liability"
    INDEMNIFICATION = "Indemnification"
    INDEMNIFICATION_CAP = "Indemnification Cap"
    MUTUAL_INDEMNIFICATION = "Mutual Indemnification"
    
    # Termination & Survival
    TERMINATION = "Termination"
    TERMINATION_FOR_CONVENIENCE = "Termination for Convenience"
    TERMINATION_FOR_CAUSE = "Termination for Cause"
    SURVIVAL = "Survival"
    
    # Payment & Compensation
    PAYMENT_TERMS = "Payment Terms"
    PRICE_AND_PAYMENT = "Price and Payment"
    TAXES = "Taxes"
    EXPENSES = "Expenses"
    INVOICING = "Invoicing"
    
    # Warranties & Representations
    WARRANTY = "Warranty"
    WARRANTIES = "Warranties"
    REPRESENTATIONS = "Representations"
    DISCLAIMER = "Disclaimer"
    
    # Force Majeure & Governing Law
    FORCE_MAJEURE = "Force Majeure"
    GOVERNING_LAW = "Governing Law"
    JURISDICTION = "Jurisdiction"
    VENUE = "Venue"
    
    # Assignment & Amendments
    ASSIGNMENT = "Assignment"
    AMENDMENT = "Amendment"
    WAIVER = "Waiver"
    
    # Notices & Communication
    NOTICES = "Notices"
    ENTIRE_AGREEMENT = "Entire Agreement"
    SEVERABILITY = "Severability"
    
    # Relationship & Compliance
    RELATIONSHIP = "Relationship"
    COMPLIANCE = "Compliance"
    REGULATORY = "Regulatory"
    EXPORT_CONTROL = "Export Control"
    ANTI_CORRUPTION = "Anti-Corruption"
    
    # Other
    NON_COMPETE = "Non-Compete"
    NON_SOLICITATION = "Non-Solicitation"
    INSURANCE = "Insurance"
    SECURITY = "Security"
    DATA_PROTECTION = "Data Protection"
    PRIVACY = "Privacy"


class Span(BaseModel):
    """A character span within a text."""

    start_char: int = Field(..., description="Start character offset (0-indexed)")
    end_char: int = Field(..., description="End character offset (exclusive)")
    text: str = Field(..., description="The actual text content of the span")
    category: ClauseCategory = Field(..., description="Clause category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")

    @field_validator("end_char")
    @classmethod
    def end_must_exceed_start(cls, v: int, info: "Span") -> int:
        if "start_char" in info.data and v <= info.data["start_char"]:
            msg = "end_char must be greater than start_char"
            raise ValueError(msg)
        return v


class Clause(BaseModel):
    """A clause extracted from a contract."""

    clause_id: str = Field(..., description="Unique clause identifier")
    spans: list[Span] = Field(..., description="List of spans comprising this clause")
    source_contract_id: str = Field(..., description="ID of the source contract")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("spans")
    @classmethod
    def spans_not_empty(cls, v: list[Span]) -> list[Span]:
        if not v:
            raise ValueError("At least one span is required")
        return v


class Contract(BaseModel):
    """A legal contract document."""

    contract_id: str = Field(..., description="Unique contract identifier")
    title: str = Field(..., description="Contract title")
    text: str = Field(..., description="Full contract text")
    file_path: Optional[str] = Field(None, description="Source file path if applicable")
    metadata: dict[str, str] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ExtractionResult(BaseModel):
    """Result of clause extraction from a contract."""

    result_id: str = Field(..., description="Unique extraction result identifier")
    contract_id: str = Field(..., description="ID of the contract processed")
    clauses: list[Clause] = Field(..., description="Extracted clauses")
    model_used: str = Field(..., description="LLM model used for extraction")
    extraction_time_ms: float = Field(..., ge=0.0, description="Extraction time in milliseconds")
    retry_count: int = Field(default=0, description="Number of retries attempted")
    error: Optional[str] = Field(None, description="Error message if extraction failed")


class VerificationStatus(str, Enum):
    """Status of span verification."""

    VERIFIED = "verified"
    REJECTED = "rejected"
    PENDING = "pending"


class VerificationResult(BaseModel):
    """Result of verifying a clause span against source text."""

    result_id: str = Field(..., description="Unique verification result identifier")
    span: Span = Field(..., description="The span being verified")
    source_text: str = Field(..., description="The source text checked against")
    status: VerificationStatus = Field(..., description="Verification status")
    verification_quote: Optional[str] = Field(
        None, description="Quote from source text supporting the verification"
    )
    reasoning: str = Field(..., description="Explanation of verification decision")
    model_used: str = Field(..., description="LLM model used for verification")
    verification_time_ms: float = Field(..., ge=0.0)
    cost_usd: float = Field(..., ge=0.0, description="Cost of verification in USD")


class EvaluationResult(BaseModel):
    """Result of evaluating extraction against ground truth."""

    result_id: str = Field(..., description="Unique evaluation result identifier")
    contract_id: str = Field(..., description="ID of the contract evaluated")
    category: ClauseCategory = Field(..., description="Clause category evaluated")
    
    # Metrics
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1: float = Field(..., ge=0.0, le=1.0)
    
    # Details
    true_positives: int = Field(..., ge=0, description="Correctly identified spans")
    false_positives: int = Field(..., ge=0, description="Incorrectly identified spans")
    false_negatives: int = Field(..., ge=0, description="Missed ground truth spans")
    
    # Context
    model_used: str = Field(..., description="LLM model evaluated")
    evaluation_time_ms: float = Field(..., ge=0.0)


class ErrorCategory(str, Enum):
    """Categories for error taxonomy."""

    # Extraction Errors
    NO_EXTRACTION = "no_extraction"
    PARTIAL_EXTRACTION = "partial_extraction"
    WRONG_CATEGORY = "wrong_category"
    OFFSET_ERROR = "offset_error"
    
    # Verification Errors
    VERIFICATION_FAILED = "verification_failed"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    
    # Retrieval Errors
    RETRIEVAL_MISSED = "retrieval_missed"
    RETRIEVAL_NOISE = "retrieval_noise"
    RERANK_FAIL = "rerank_fail"
    
    # Model Errors
    MODEL_TIMEOUT = "model_timeout"
    MODEL_ERROR = "model_error"
    RATE_LIMIT = "rate_limit"


class ErrorRecord(BaseModel):
    """Record of an error encountered during processing."""

    error_id: str = Field(..., description="Unique error identifier")
    error_category: ErrorCategory = Field(..., description="Category of the error")
    contract_id: str = Field(..., description="Contract where error occurred")
    category: Optional[ClauseCategory] = Field(None, description="Related clause category")
    message: str = Field(..., description="Error message")
    details: dict[str, str] = Field(default_factory=dict, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Type alias for forward references
SpanDict = dict[str, str | int | ClauseCategory]