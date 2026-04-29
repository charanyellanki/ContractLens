"""Tests for API module."""

import pytest

from contractlens.api.routes import router
from contractlens.api.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    ExtractRequest,
    ExtractResponse,
)


class TestAPI Schemas:
    """Tests for API schemas."""

    def test_extract_request_valid(self):
        """Test ExtractRequest validation."""
        request = ExtractRequest(
            contract_text="Test contract",
            model="gpt-4o-mini",
        )
        
        assert request.contract_text == "Test contract"
        assert request.model == "gpt-4o-mini"
        assert request.include_verification is True

    def test_extract_request_with_categories(self):
        """Test ExtractRequest with specific categories."""
        request = ExtractRequest(
            contract_text="Test contract",
            categories=["Termination", "Confidentiality"],
        )
        
        assert len(request.categories) == 2

    def test_evaluate_request_valid(self):
        """Test EvaluateRequest validation."""
        request = EvaluateRequest(
            contract_text="Test contract",
            ground_truth=[{"text": "clause", "start": 0, "end": 5}],
            model="gpt-4o-mini",
        )
        
        assert request.contract_text == "Test contract"
        assert len(request.ground_truth) == 1

    def test_extract_response(self):
        """Test ExtractResponse."""
        response = ExtractResponse(
            contract_id="test_1",
            clauses=[],
            model_used="gpt-4o-mini",
            extraction_time_ms=100.0,
            cost_usd=0.01,
            verified=True,
        )
        
        assert response.contract_id == "test_1"
        assert response.verified is True


class TestAPIRoutes:
    """Tests for API routes."""

    def test_router_has_routes(self):
        """Test router has expected routes."""
        routes = [r.path for r in router.routes]
        
        assert "/extract" in routes
        assert "/evaluate" in routes
        assert "/health" in routes
        assert "/categories" in routes