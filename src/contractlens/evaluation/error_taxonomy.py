"""Error taxonomy for failure mode analysis."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from contractlens.models import ClauseCategory, ErrorCategory, ErrorRecord


class ErrorTaxonomy:
    """Structured error taxonomy for ContractLens failures."""

    def __init__(self) -> None:
        self._errors: list[ErrorRecord] = []
        self._category_counts: dict[ErrorCategory, int] = {}

    def record_error(
        self,
        category: ErrorCategory,
        contract_id: str,
        message: str,
        clause_category: Optional[ClauseCategory] = None,
        details: Optional[dict[str, str]] = None,
    ) -> None:
        """Record an error encountered during processing."""
        import uuid

        error = ErrorRecord(
            error_id=str(uuid.uuid4()),
            error_category=category,
            contract_id=contract_id,
            category=clause_category,
            message=message,
            details=details or {},
            timestamp=datetime.utcnow(),
        )
        
        self._errors.append(error)
        self._category_counts[category] = self._category_counts.get(category, 0) + 1

    def get_errors_by_category(self, category: ErrorCategory) -> list[ErrorRecord]:
        """Get all errors of a specific category."""
        return [e for e in self._errors if e.error_category == category]

    def get_error_distribution(self) -> dict[str, int]:
        """Get distribution of errors by category."""
        return {cat.value: count for cat, count in self._category_counts.items()}

    def get_top_errors(self, n: int = 5) -> list[tuple[ErrorCategory, int]]:
        """Get the top N most common errors."""
        sorted_errors = sorted(
            self._category_counts.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_errors[:n]

    def get_contract_errors(self, contract_id: str) -> list[ErrorRecord]:
        """Get all errors for a specific contract."""
        return [e for e in self._errors if e.contract_id == contract_id]

    def get_category_error_rate(
        self, contract_id: str, clause_category: ClauseCategory
    ) -> float:
        """Get error rate for a specific clause category."""
        category_errors = self.get_errors_by_category(ErrorCategory.WRONG_CATEGORY)
        relevant = [e for e in category_errors if e.category == clause_category]
        return len(relevant) / max(1, len([e for e in self._errors if e.contract_id == contract_id]))

    def clear(self) -> None:
        """Clear all recorded errors."""
        self._errors.clear()
        self._category_counts.clear()

    def __len__(self) -> int:
        return len(self._errors)

    def __repr__(self) -> str:
        return f"ErrorTaxonomy(errors={len(self._errors)}, categories={len(self._category_counts)})"


# Predefined error patterns
ERROR_PATTERNS = {
    "offset_mismatch": ErrorCategory.OFFSET_ERROR,
    "span_out_of_bounds": ErrorCategory.OFFSET_ERROR,
    "negative_offsets": ErrorCategory.OFFSET_ERROR,
    "verification_rejected": ErrorCategory.VERIFICATION_FAILED,
    "no_source_match": ErrorCategory.VERIFICATION_FAILED,
    "hallucinated_clause": ErrorCategory.FALSE_POSITIVE,
    "missing_clause": ErrorCategory.FALSE_NEGATIVE,
    "wrong_category_label": ErrorCategory.WRONG_CATEGORY,
    "empty_extraction": ErrorCategory.NO_EXTRACTION,
    "partial_extraction": ErrorCategory.PARTIAL_EXTRACTION,
    "retrieval_empty": ErrorCategory.RETRIEVAL_MISSED,
    "retrieval_irrelevant": ErrorCategory.RETRIEVAL_NOISE,
    "model_timeout": ErrorCategory.MODEL_TIMEOUT,
    "rate_limit_exceeded": ErrorCategory.RATE_LIMIT,
}


def classify_error(message: str) -> ErrorCategory:
    """Classify an error message into the appropriate category."""
    message_lower = message.lower()
    
    for pattern, category in ERROR_PATTERNS.items():
        if pattern in message_lower:
            return category
    
    # Default to model error
    return ErrorCategory.MODEL_ERROR