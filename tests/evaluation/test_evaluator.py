"""Tests for evaluation module."""

import pytest

from contractlens.evaluation.evaluator import Evaluator
from contractlens.evaluation.error_taxonomy import ErrorTaxonomy, classify_error
from contractlens.evaluation.metrics import (
    compute_char_overlap,
    compute_span_f1,
    is_partial_match,
)
from contractlens.models import ClauseCategory, ErrorCategory, Span


class TestMetrics:
    """Tests for metrics computation."""

    def test_compute_char_overlap(self):
        """Test character overlap computation."""
        overlap, pred_cov, gt_cov = compute_char_overlap(0, 10, 5, 15)
        
        assert overlap == 5
        assert pred_cov == 0.5
        assert gt_cov == 0.5

    def test_compute_char_overlap_no_overlap(self):
        """Test with no overlap."""
        overlap, pred_cov, gt_cov = compute_char_overlap(0, 10, 20, 30)
        
        assert overlap == 0
        assert pred_cov == 0.0
        assert gt_cov == 0.0

    def test_is_partial_match(self):
        """Test partial match detection."""
        pred = Span(
            start_char=0,
            end_char=10,
            text="Test clause",
            category=ClauseCategory.TERMINATION,
            confidence=0.9,
        )
        gt = Span(
            start_char=0,
            end_char=20,
            text="Test clause with more text",
            category=ClauseCategory.TERMINATION,
            confidence=0.9,
        )
        
        assert is_partial_match(pred, gt) is True

    def test_compute_span_f1_perfect_match(self):
        """Test F1 with perfect match."""
        pred = [
            Span(
                start_char=0,
                end_char=10,
                text="Termination",
                category=ClauseCategory.TERMINATION,
                confidence=0.9,
            )
        ]
        gt = [
            Span(
                start_char=0,
                end_char=10,
                text="Termination",
                category=ClauseCategory.TERMINATION,
                confidence=0.9,
            )
        ]
        
        tp, fp, fn, precision, recall, f1 = compute_span_f1(
            pred, gt, ClauseCategory.TERMINATION
        )
        
        assert tp == 1
        assert fp == 0
        assert fn == 0
        assert f1 == 1.0

    def test_compute_span_f1_no_predictions(self):
        """Test F1 with no predictions."""
        pred: list[Span] = []
        gt = [
            Span(
                start_char=0,
                end_char=10,
                text="Termination",
                category=ClauseCategory.TERMINATION,
                confidence=0.9,
            )
        ]
        
        tp, fp, fn, precision, recall, f1 = compute_span_f1(
            pred, gt, ClauseCategory.TERMINATION
        )
        
        assert tp == 0
        assert fp == 0
        assert fn == 1
        assert f1 == 0.0


class TestErrorTaxonomy:
    """Tests for error taxonomy."""

    def test_record_error(self):
        """Test recording an error."""
        taxonomy = ErrorTaxonomy()
        
        taxonomy.record_error(
            category=ErrorCategory.OFFSET_ERROR,
            contract_id="test_1",
            message="Offset mismatch",
            clause_category=ClauseCategory.TERMINATION,
        )
        
        assert len(taxonomy) == 1

    def test_get_error_distribution(self):
        """Test getting error distribution."""
        taxonomy = ErrorTaxonomy()
        
        taxonomy.record_error(ErrorCategory.OFFSET_ERROR, "c1", "error 1")
        taxonomy.record_error(ErrorCategory.OFFSET_ERROR, "c2", "error 2")
        taxonomy.record_error(ErrorCategory.VERIFICATION_FAILED, "c3", "error 3")
        
        dist = taxonomy.get_error_distribution()
        
        assert dist["offset_error"] == 2
        assert dist["verification_failed"] == 1

    def test_get_top_errors(self):
        """Test getting top errors."""
        taxonomy = ErrorTaxonomy()
        
        taxonomy.record_error(ErrorCategory.OFFSET_ERROR, "c1", "e1")
        taxonomy.record_error(ErrorCategory.OFFSET_ERROR, "c2", "e2")
        taxonomy.record_error(ErrorCategory.VERIFICATION_FAILED, "c3", "e3")
        
        top = taxonomy.get_top_errors(2)
        
        assert len(top) == 2
        assert top[0][0] == ErrorCategory.OFFSET_ERROR

    def test_classify_error(self):
        """Test error classification."""
        assert classify_error("offset mismatch") == ErrorCategory.OFFSET_ERROR
        assert classify_error("verification rejected") == ErrorCategory.VERIFICATION_FAILED
        assert classify_error("unknown error") == ErrorCategory.MODEL_ERROR


class TestEvaluator:
    """Tests for Evaluator."""

    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = Evaluator(model="gpt-4o-mini")
        
        assert evaluator.model == "gpt-4o-mini"
        assert len(evaluator.categories) == 41

    def test_evaluate_contract(self):
        """Test evaluating a single contract."""
        evaluator = Evaluator(model="gpt-4o-mini")
        
        contract_text = "This is a test contract with a termination clause."
        ground_truth = [
            Span(
                start_char=32,
                end_char=47,
                text="termination clause",
                category=ClauseCategory.TERMINATION,
                confidence=1.0,
            )
        ]
        
        results = evaluator.evaluate_contract(contract_text, ground_truth, "test_1")
        
        assert len(results) == 41  # All categories