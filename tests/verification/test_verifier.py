"""Tests for verification module."""

import pytest

from contractlens.models import ClauseCategory, Span, VerificationStatus
from contractlens.verification.verifier import SpanVerifier
from contractlens.verification.judge import VerificationJudge


class TestSpanVerifier:
    """Tests for SpanVerifier."""

    def test_verifier_initialization(self):
        """Test verifier initialization."""
        verifier = SpanVerifier(default_model="gpt-4o-mini")
        
        assert verifier.default_model == "gpt-4o-mini"

    def test_parse_verification_response_valid(self):
        """Test parsing valid verification response."""
        verifier = SpanVerifier()
        span = Span(
            start_char=0,
            end_char=10,
            text="Test clause",
            category=ClauseCategory.TERMINATION,
            confidence=0.9,
        )
        
        response = '''{
            "status": "verified",
            "verification_quote": "Test clause",
            "reasoning": "Exact match found"
        }'''
        
        result = verifier._parse_verification_response(
            response, span, "source text", "gpt-4o-mini", 100.0, 0.01
        )
        
        assert result.status == VerificationStatus.VERIFIED
        assert result.verification_quote == "Test clause"

    def test_parse_verification_response_rejected(self):
        """Test parsing rejected verification."""
        verifier = SpanVerifier()
        span = Span(
            start_char=0,
            end_char=10,
            text="Fake clause",
            category=ClauseCategory.TERMINATION,
            confidence=0.9,
        )
        
        response = '''{
            "status": "rejected",
            "verification_quote": null,
            "reasoning": "Not found in source"
        }'''
        
        result = verifier._parse_verification_response(
            response, span, "source text", "gpt-4o-mini", 100.0, 0.01
        )
        
        assert result.status == VerificationStatus.REJECTED


class TestVerificationJudge:
    """Tests for VerificationJudge."""

    def test_judge_initialization(self):
        """Test judge initialization."""
        judge = VerificationJudge(default_model="gpt-4o")
        
        assert judge.default_model == "gpt-4o"

    def test_parse_judge_response_valid(self):
        """Test parsing valid judge response."""
        judge = VerificationJudge()
        span = Span(
            start_char=0,
            end_char=10,
            text="Test clause",
            category=ClauseCategory.TERMINATION,
            confidence=0.9,
        )
        
        response = '''{
            "valid": true,
            "quote": "Test clause",
            "explanation": "Found in source"
        }'''
        
        result = judge._parse_judge_response(
            response, span, "source text", "gpt-4o", 100.0, 0.01
        )
        
        assert result.status == VerificationStatus.VERIFIED

    def test_parse_judge_response_invalid(self):
        """Test parsing invalid judge response."""
        judge = VerificationJudge()
        span = Span(
            start_char=0,
            end_char=10,
            text="Fake clause",
            category=ClauseCategory.TERMINATION,
            confidence=0.9,
        )
        
        response = '''{
            "valid": false,
            "quote": null,
            "explanation": "Not found"
        }'''
        
        result = judge._parse_judge_response(
            response, span, "source text", "gpt-4o", 100.0, 0.01
        )
        
        assert result.status == VerificationStatus.REJECTED