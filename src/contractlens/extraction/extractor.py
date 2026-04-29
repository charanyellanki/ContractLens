"""Clause extractor using LLM."""

import json
import logging
import uuid
from typing import Optional

from contractlens.llm import LLMWrapper, get_llm_wrapper
from contractlens.models import ClauseCategory, ExtractionResult, Span
from contractlens.extraction.prompts import get_extraction_prompt

logger = logging.getLogger(__name__)


class ClauseExtractor:
    """Extracts clauses from contracts using LLM."""

    def __init__(
        self,
        llm_wrapper: Optional[LLMWrapper] = None,
        default_model: str = "gpt-4o-mini",
    ) -> None:
        self.llm = llm_wrapper or get_llm_wrapper()
        self.default_model = default_model

    def extract(
        self,
        contract_text: str,
        category: ClauseCategory,
        context: Optional[str] = None,
        model: Optional[str] = None,
    ) -> list[Span]:
        """Extract clauses for a specific category."""
        prompt = get_extraction_prompt(category).format(
            contract_text=contract_text[:8000],  # Truncate to avoid token limits
            context=context or "",
        )

        messages = [{"role": "user", "content": prompt}]
        
        try:
            result = self.llm.complete(
                messages=messages,
                model=model or self.default_model,
                temperature=0.0,
                max_tokens=4000,
                response_format={"type": "json_object"},
            )

            # Parse the JSON response
            spans = self._parse_extraction_response(result.content, category)
            logger.info(
                f"Extracted {len(spans)} {category.value} clauses "
                f"using {result.model}"
            )
            return spans

        except Exception as e:
            logger.error(f"Extraction failed for {category.value}: {e}")
            raise

    def _parse_extraction_response(
        self, response: str, category: ClauseCategory
    ) -> list[Span]:
        """Parse LLM response into Span objects."""
        try:
            data = json.loads(response)
            if not isinstance(data, list):
                data = data.get("clauses", [])

            spans: list[Span] = []
            for item in data:
                if isinstance(item, dict):
                    spans.append(
                        Span(
                            start_char=item.get("start_char", 0),
                            end_char=item.get("end_char", 0),
                            text=item.get("text", ""),
                            category=category,
                            confidence=item.get("confidence", 0.5),
                        )
                    )
            return spans

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse extraction response: {e}")
            return []

    def extract_all_categories(
        self,
        contract_text: str,
        categories: list[ClauseCategory],
        context: Optional[str] = None,
        model: Optional[str] = None,
    ) -> dict[ClauseCategory, list[Span]]:
        """Extract clauses for all specified categories."""
        results: dict[ClauseCategory, list[Span]] = {}
        
        for category in categories:
            try:
                spans = self.extract(contract_text, category, context, model)
                results[category] = spans
            except Exception as e:
                logger.error(f"Failed to extract {category.value}: {e}")
                results[category] = []

        return results