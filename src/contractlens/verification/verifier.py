"""Span verifier using LLM-as-judge."""

import json
import logging
import uuid
from typing import Optional

from contractlens.llm import LLMWrapper, get_llm_wrapper
from contractlens.models import (
    Span,
    VerificationResult,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


VERIFICATION_PROMPT = """You are a legal contract verification expert. Your task is to verify whether a extracted clause span is literally grounded in the source contract text.

Extracted clause:
"{span_text}"

Source contract text:
"{source_text}"

Determine if the extracted clause is:
1. VERIFIED: The clause text is literally present (or near-literal with minor formatting changes) in the source text
2. REJECTED: The clause text is NOT present in the source text - it may be a hallucination or misreading

Return your response as JSON:
{{
  "status": "verified" or "rejected",
  "verification_quote": "the exact quote from source text that supports the verification (if verified)",
  "reasoning": "explanation of your verification decision"
}}

If verified, provide the exact quote from the source text that matches or nearly matches the extracted clause.
If rejected, explain why the extracted clause cannot be found in the source text.
"""


class SpanVerifier:
    """Verifies extracted spans against source text."""

    def __init__(
        self,
        llm_wrapper: Optional[LLMWrapper] = None,
        default_model: str = "gpt-4o-mini",
    ) -> None:
        self.llm = llm_wrapper or get_llm_wrapper()
        self.default_model = default_model

    def verify(
        self,
        span: Span,
        source_text: str,
        model: Optional[str] = None,
    ) -> VerificationResult:
        """Verify a single span against source text."""
        prompt = VERIFICATION_PROMPT.format(
            span_text=span.text,
            source_text=source_text[:8000],  # Truncate to avoid token limits
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            result = self.llm.complete(
                messages=messages,
                model=model or self.default_model,
                temperature=0.0,
                max_tokens=1000,
                response_format={"type": "json_object"},
            )

            verification = self._parse_verification_response(
                result.content, span, source_text, result.model, result.latency_ms, result.cost_usd
            )
            logger.info(
                f"Verification {verification.status.value}: "
                f"{span.category.value} (confidence: {span.confidence:.2f})"
            )
            return verification

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            raise

    def _parse_verification_response(
        self,
        response: str,
        span: Span,
        source_text: str,
        model: str,
        latency_ms: float,
        cost_usd: float,
    ) -> VerificationResult:
        """Parse LLM verification response."""
        try:
            data = json.loads(response)
            status = VerificationStatus(data.get("status", "pending"))
            
            return VerificationResult(
                result_id=str(uuid.uuid4()),
                span=span,
                source_text=source_text[:500],  # Store snippet for reference
                status=status,
                verification_quote=data.get("verification_quote"),
                reasoning=data.get("reasoning", ""),
                model_used=model,
                verification_time_ms=latency_ms,
                cost_usd=cost_usd,
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse verification response: {e}")
            return VerificationResult(
                result_id=str(uuid.uuid4()),
                span=span,
                source_text=source_text[:500],
                status=VerificationStatus.PENDING,
                reasoning=f"Failed to parse response: {e}",
                model_used=model,
                verification_time_ms=latency_ms,
                cost_usd=cost_usd,
            )

    def verify_batch(
        self,
        spans: list[Span],
        source_text: str,
        model: Optional[str] = None,
    ) -> list[VerificationResult]:
        """Verify multiple spans."""
        results: list[VerificationResult] = []
        for span in spans:
            result = self.verify(span, source_text, model)
            results.append(result)
        return results