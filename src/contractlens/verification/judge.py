"""Verification judge - alternate implementation using different prompt strategy."""

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


JUDGE_PROMPT = """You are an expert judge evaluating contract clause extractions.

Given:
- Extracted clause: "{span_text}"
- Source contract: "{source_text}"

Evaluate whether the extraction is VALID (grounded in source) or INVALID (hallucinated/misread).

Output JSON:
{{
  "valid": true or false,
  "quote": "exact matching text from source if valid",
  "explanation": "brief reasoning"
}}
"""


class VerificationJudge:
    """Alternative verification using judge-style prompts."""

    def __init__(
        self,
        llm_wrapper: Optional[LLMWrapper] = None,
        default_model: str = "gpt-4o",
    ) -> None:
        self.llm = llm_wrapper or get_llm_wrapper()
        self.default_model = default_model

    def judge(
        self,
        span: Span,
        source_text: str,
        model: Optional[str] = None,
    ) -> VerificationResult:
        """Judge a span's validity."""
        prompt = JUDGE_PROMPT.format(
            span_text=span.text,
            source_text=source_text[:8000],
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            result = self.llm.complete(
                messages=messages,
                model=model or self.default_model,
                temperature=0.0,
                max_tokens=500,
                response_format={"type": "json_object"},
            )

            return self._parse_judge_response(
                result.content, span, source_text, result.model, result.latency_ms, result.cost_usd
            )

        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            raise

    def _parse_judge_response(
        self,
        response: str,
        span: Span,
        source_text: str,
        model: str,
        latency_ms: float,
        cost_usd: float,
    ) -> VerificationResult:
        """Parse judge response."""
        try:
            data = json.loads(response)
            valid = data.get("valid", False)
            status = VerificationStatus.VERIFIED if valid else VerificationStatus.REJECTED
            
            return VerificationResult(
                result_id=str(uuid.uuid4()),
                span=span,
                source_text=source_text[:500],
                status=status,
                verification_quote=data.get("quote"),
                reasoning=data.get("explanation", ""),
                model_used=model,
                verification_time_ms=latency_ms,
                cost_usd=cost_usd,
            )

        except json.JSONDecodeError:
            return VerificationResult(
                result_id=str(uuid.uuid4()),
                span=span,
                source_text=source_text[:500],
                status=VerificationStatus.PENDING,
                reasoning="Failed to parse judge response",
                model_used=model,
                verification_time_ms=latency_ms,
                cost_usd=cost_usd,
            )