"""Span-level F1 computation."""

from dataclasses import dataclass
from typing import Optional

from contractlens.models import ClauseCategory, Span


@dataclass
class SpanMatch:
    """A match between predicted and ground truth spans."""

    predicted: Span
    ground_truth: Span
    overlap_chars: int
    precision: float
    recall: float


def compute_char_overlap(
    pred_start: int,
    pred_end: int,
    gt_start: int,
    gt_end: int,
) -> tuple[int, int, int]:
    """Compute character overlap between two spans.
    
    Returns:
        Tuple of (overlap_chars, pred_coverage, gt_coverage)
    """
    overlap_start = max(pred_start, gt_start)
    overlap_end = min(pred_end, gt_end)
    
    overlap_chars = max(0, overlap_end - overlap_start)
    pred_coverage = overlap_chars / max(1, pred_end - pred_start)
    gt_coverage = overlap_chars / max(1, gt_end - gt_start)
    
    return overlap_chars, pred_coverage, gt_coverage


def is_partial_match(
    pred: Span,
    gt: Span,
    min_overlap_ratio: float = 0.5,
) -> bool:
    """Check if predicted span is a partial match to ground truth."""
    overlap_chars, pred_coverage, gt_coverage = compute_char_overlap(
        pred.start_char, pred.end_char,
        gt.start_char, gt.end_char,
    )
    
    # Consider it a match if either span covers >50% of the other
    return pred_coverage >= min_overlap_ratio or gt_coverage >= min_overlap_ratio


def compute_span_f1(
    predicted_spans: list[Span],
    ground_truth_spans: list[Span],
    category: ClauseCategory,
    partial_credit: bool = True,
) -> tuple[int, int, int, float, float, float]:
    """Compute span-level metrics.
    
    Args:
        predicted_spans: List of predicted clause spans
        ground_truth_spans: List of ground truth clause spans
        category: The clause category being evaluated
        partial_credit: Whether to give partial credit for overlapping spans
    
    Returns:
        Tuple of (true_positives, false_positives, false_negatives, precision, recall, f1)
    """
    if not ground_truth_spans:
        # No ground truth - all predictions are false positives
        return 0, len(predicted_spans), 0, 0.0, 0.0, 0.0
    
    if not predicted_spans:
        # No predictions - all ground truth are missed
        return 0, 0, len(ground_truth_spans), 0.0, 0.0, 0.0
    
    # Filter to matching category
    pred_filtered = [p for p in predicted_spans if p.category == category]
    gt_filtered = [g for g in ground_truth_spans if g.category == category]
    
    if not pred_filtered or not gt_filtered:
        if pred_filtered:
            return 0, len(pred_filtered), 0, 0.0, 0.0, 0.0
        if gt_filtered:
            return 0, 0, len(gt_filtered), 0.0, 0.0, 0.0
        return 0, 0, 0, 0.0, 0.0, 0.0
    
    # Find matches
    matched_preds: set[int] = set()
    matched_gts: set[int] = set()
    
    for i, pred in enumerate(pred_filtered):
        for j, gt in enumerate(gt_filtered):
            if j in matched_gts:
                continue
            
            if partial_credit and is_partial_match(pred, gt):
                matched_preds.add(i)
                matched_gts.add(j)
            elif pred.text.strip() == gt.text.strip():
                # Exact match
                matched_preds.add(i)
                matched_gts.add(j)
    
    true_positives = len(matched_preds)
    false_positives = len(pred_filtered) - true_positives
    false_negatives = len(gt_filtered) - len(matched_gts)
    
    # Compute metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return true_positives, false_positives, false_negatives, precision, recall, f1


def compute_category_metrics(
    predicted_spans: list[Span],
    ground_truth_spans: list[Span],
    category: ClauseCategory,
) -> dict[str, float]:
    """Compute metrics for a single category."""
    tp, fp, fn, precision, recall, f1 = compute_span_f1(
        predicted_spans, ground_truth_spans, category
    )
    
    return {
        "category": category.value,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }