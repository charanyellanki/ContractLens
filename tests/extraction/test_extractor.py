"""Tests for extraction module."""

import pytest

from contractlens.extraction.extractor import ClauseExtractor
from contractlens.extraction.prompts import (
    EXTRACTION_PROMPTS,
    get_extraction_prompt,
)
from contractlens.models import ClauseCategory, Span


class TestExtractionPrompts:
    """Tests for extraction prompts."""

    def test_all_categories_have_prompts(self):
        """Test all 41 categories have prompts."""
        for category in ClauseCategory:
            assert category in EXTRACTION_PROMPTS

    def test_get_extraction_prompt(self):
        """Test getting extraction prompt for category."""
        prompt = get_extraction_prompt(ClauseCategory.TERMINATION)
        
        assert "Termination" in prompt
        assert "{contract_text}" not in prompt  # Should be formatted

    def test_prompt_contains_required_fields(self):
        """Test prompt contains required output fields."""
        prompt = get_extraction_prompt(ClauseCategory.CONFIDENTIALITY)
        
        assert "text" in prompt
        assert "start_char" in prompt
        assert "end_char" in prompt
        assert "confidence" in prompt


class TestClauseExtractor:
    """Tests for ClauseExtractor."""

    def test_extractor_initialization(self):
        """Test extractor initialization."""
        extractor = ClauseExtractor(default_model="gpt-4o-mini")
        
        assert extractor.default_model == "gpt-4o-mini"

    def test_parse_extraction_response_empty(self):
        """Test parsing empty response."""
        extractor = ClauseExtractor()
        result = extractor._parse_extraction_response("[]", ClauseCategory.TERMINATION)
        
        assert result == []

    def test_parse_extraction_response_valid(self):
        """Test parsing valid JSON response."""
        extractor = ClauseExtractor()
        response = '[{"text": "Termination clause", "start_char": 0, "end_char": 18, "confidence": 0.9}]'
        
        result = extractor._parse_extraction_response(
            response, ClauseCategory.TERMINATION
        )
        
        assert len(result) == 1
        assert result[0].text == "Termination clause"
        assert result[0].category == ClauseCategory.TERMINATION

    def test_parse_extraction_response_invalid(self):
        """Test parsing invalid JSON."""
        extractor = ClauseExtractor()
        result = extractor._parse_extraction_response(
            "not valid json", ClauseCategory.TERMINATION
        )
        
        assert result == []