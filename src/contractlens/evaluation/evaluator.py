"""Main evaluator class for ContractLens."""

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from contractlens.evaluation.error_taxonomy import ErrorTaxonomy
from contractlens.evaluation.metrics import compute_span_f1
from contractlens.extraction import ClauseExtractor
from contractlens.models import (
    ClauseCategory,
    Contract,
    EvaluationResult,
    Span,
)
from contractlens.verification import SpanVerifier

logger = logging.getLogger(__name__)


@dataclass
class EvaluationSummary:
    """Summary of evaluation results."""

    total_contracts: int
    total_clauses: int
    overall_precision: float
    overall_recall: float
    overall_f1: float
    avg_latency_ms: float
    avg_cost_usd: float
    category_results: dict[str, dict[str, float]]
    error_taxonomy: ErrorTaxonomy


class Evaluator:
    """Main evaluator for ContractLens clause extraction."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        categories: Optional[list[ClauseCategory]] = None,
    ) -> None:
        self.model = model
        self.categories = categories or list(ClauseCategory)
        self.extractor = ClauseExtractor(default_model=model)
        self.verifier = SpanVerifier(default_model=model)
        self.error_taxonomy = ErrorTaxonomy()

    def evaluate_contract(
        self,
        contract_text: str,
        ground_truth_spans: list[Span],
        contract_id: Optional[str] = None,
    ) -> list[EvaluationResult]:
        """Evaluate extraction on a single contract."""
        contract_id = contract_id or str(uuid.uuid4())
        contract = Contract(
            contract_id=contract_id,
            title="Evaluation Contract",
            text=contract_text,
        )

        results: list[EvaluationResult] = []

        # Extract clauses for each category
        for category in self.categories:
            try:
                start_time = time.perf_counter()
                
                extracted = self.extractor.extract(
                    contract_text=contract_text,
                    category=category,
                    model=self.model,
                )
                
                latency_ms = (time.perf_counter() - start_time) * 1000

                # Compute metrics
                tp, fp, fn, precision, recall, f1 = compute_span_f1(
                    extracted, ground_truth_spans, category
                )

                result = EvaluationResult(
                    result_id=str(uuid.uuid4()),
                    contract_id=contract_id,
                    category=category,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    true_positives=tp,
                    false_positives=fp,
                    false_negatives=fn,
                    model_used=self.model,
                    evaluation_time_ms=latency_ms,
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Evaluation failed for {category.value}: {e}")
                self.error_taxonomy.record_error(
                    category="model_error",
                    contract_id=contract_id,
                    message=str(e),
                    clause_category=category,
                )

        return results

    def evaluate_batch(
        self,
        contracts: list[tuple[str, list[Span]]],
    ) -> EvaluationSummary:
        """Evaluate on a batch of contracts."""
        all_results: list[EvaluationResult] = []
        total_latency = 0.0
        total_cost = 0.0

        for contract_text, ground_truth in contracts:
            results = self.evaluate_contract(contract_text, ground_truth)
            all_results.extend(results)
            total_latency += sum(r.evaluation_time_ms for r in results)
            # TODO: Add cost tracking

        # Aggregate results
        category_agg: dict[str, dict[str, float]] = {}
        
        for result in all_results:
            cat = result.category.value
            if cat not in category_agg:
                category_agg[cat] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "count": 0,
                }
            
            cat_data = category_agg[cat]
            cat_data["precision"] += result.precision
            cat_data["recall"] += result.recall
            cat_data["f1"] += result.f1
            cat_data["count"] += 1

        # Average per category
        for cat in category_agg:
            count = category_agg[cat]["count"]
            if count > 0:
                category_agg[cat]["precision"] /= count
                category_agg[cat]["recall"] /= count
                category_agg[cat]["f1"] /= count

        # Overall metrics
        overall_precision = sum(r.precision for r in all_results) / len(all_results) if all_results else 0.0
        overall_recall = sum(r.recall for r in all_results) / len(all_results) if all_results else 0.0
        overall_f1 = sum(r.f1 for r in all_results) / len(all_results) if all_results else 0.0

        return EvaluationSummary(
            total_contracts=len(contracts),
            total_clauses=len(all_results),
            overall_precision=overall_precision,
            overall_recall=overall_recall,
            overall_f1=overall_f1,
            avg_latency_ms=total_latency / max(1, len(contracts)),
            avg_cost_usd=total_cost / max(1, len(contracts)),
            category_results=category_agg,
            error_taxonomy=self.error_taxonomy,
        )