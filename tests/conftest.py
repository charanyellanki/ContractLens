"""Pytest configuration and fixtures."""

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def sample_contract_text():
    """Sample contract text for testing."""
    return """
    CONFIDENTIALITY AGREEMENT
    
    This Confidentiality Agreement ("Agreement") is entered into as of January 1, 2024.
    
    1. CONFIDENTIALITY
    Each party agrees to keep confidential all Confidential Information received from the other party.
    
    2. TERM
    This Agreement shall remain in effect for a period of two (2) years from the date hereof.
    
    3. INDEMNIFICATION
    Each party shall indemnify and hold harmless the other party from any damages arising from breach.
    """


@pytest.fixture
def mock_llm_wrapper():
    """Mock LLM wrapper for testing."""
    from contractlens.llm import LLMWrapper, LLMCallResult, TokenUsage
    
    wrapper = MagicMock(spec=LLMWrapper)
    
    # Mock a successful call
    wrapper.complete.return_value = LLMCallResult(
        content='[{"text": "clause", "start_char": 0, "end_char": 5, "confidence": 0.9}]',
        model="gpt-4o-mini",
        token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        latency_ms=100.0,
        cost_usd=0.001,
    )
    
    return wrapper


@pytest.fixture
def sample_spans():
    """Sample spans for testing."""
    from contractlens.models import Span, ClauseCategory
    
    return [
        Span(
            start_char=100,
            end_char=200,
            text="Confidentiality clause text",
            category=ClauseCategory.CONFIDENTIALITY,
            confidence=0.95,
        ),
        Span(
            start_char=300,
            end_char=400,
            text="Termination clause text",
            category=ClauseCategory.TERMINATION,
            confidence=0.90,
        ),
    ]


@pytest.fixture
def sample_contract():
    """Sample contract for testing."""
    from contractlens.models import Contract
    
    return Contract(
        contract_id="test_contract_1",
        title="Test NDA",
        text="Sample contract text for testing.",
    )