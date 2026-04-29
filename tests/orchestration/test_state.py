"""Tests for orchestration module."""

import pytest

from contractlens.models import ClauseCategory, Contract
from contractlens.orchestration.state import (
    ContractLensState,
    create_initial_state,
    get_unverified_spans,
    has_unverified_spans,
    is_complete,
)


class TestContractLensState:
    """Tests for ContractLensState."""

    def test_create_initial_state(self):
        """Test creating initial state."""
        contract = Contract(
            contract_id="test_1",
            title="Test Contract",
            text="Contract text here",
        )
        categories = [ClauseCategory.TERMINATION, ClauseCategory.CONFIDENTIALITY]
        
        state = create_initial_state(contract, categories, "gpt-4o-mini", 2)
        
        assert state.contract == contract
        assert state.target_categories == categories
        assert state.model_used == "gpt-4o-mini"
        assert state.max_retries == 2

    def test_has_unverified_spans_empty(self):
        """Test has_unverified_spans with no spans."""
        state = ContractLensState()
        
        assert has_unverified_spans(state) is False

    def test_has_unverified_spans_with_spans(self):
        """Test has_unverified_spans with extracted spans."""
        from contractlens.models import Span
        
        state = ContractLensState()
        state.extracted_spans = {
            ClauseCategory.TERMINATION: [
                Span(
                    start_char=0,
                    end_char=10,
                    text="Test",
                    category=ClauseCategory.TERMINATION,
                    confidence=0.9,
                )
            ]
        }
        
        assert has_unverified_spans(state) is True

    def test_get_unverified_spans(self):
        """Test getting unverified spans."""
        from contractlens.models import Span
        
        state = ContractLensState()
        span1 = Span(
            start_char=0,
            end_char=10,
            text="Test 1",
            category=ClauseCategory.TERMINATION,
            confidence=0.9,
        )
        span2 = Span(
            start_char=20,
            end_char=30,
            text="Test 2",
            category=ClauseCategory.CONFIDENTIALITY,
            confidence=0.9,
        )
        
        state.extracted_spans = {
            ClauseCategory.TERMINATION: [span1],
            ClauseCategory.CONFIDENTIALITY: [span2],
        }
        
        unverified = get_unverified_spans(state)
        
        assert len(unverified) == 2

    def test_is_complete_with_result(self):
        """Test is_complete with extraction result."""
        from contractlens.models import ExtractionResult
        
        state = ContractLensState()
        state.extraction_result = ExtractionResult(
            result_id="test",
            contract_id="test",
            clauses=[],
            model_used="gpt-4o-mini",
            extraction_time_ms=100.0,
        )
        
        assert is_complete(state) is True

    def test_is_complete_with_error(self):
        """Test is_complete with error."""
        state = ContractLensState(error="Test error")
        
        assert is_complete(state) is True