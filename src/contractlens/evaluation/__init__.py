"""Evaluation module for metrics computation and error analysis."""

from contractlens.evaluation.evaluator import Evaluator
from contractlens.evaluation.metrics import compute_span_f1, compute_category_metrics
from contractlens.evaluation.error_taxonomy import ErrorTaxonomy

__all__ = ["Evaluator", "compute_span_f1", "compute_category_metrics", "ErrorTaxonomy"]