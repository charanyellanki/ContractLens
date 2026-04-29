"""Metrics collection for ContractLens."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from contractlens.models import ClauseCategory


@dataclass
class ExtractionMetrics:
    """Metrics for a single extraction operation."""

    operation_id: str
    timestamp: datetime
    model: str
    category: ClauseCategory
    num_spans_extracted: int
    latency_ms: float
    cost_usd: float
    success: bool
    error: Optional[str] = None


@dataclass
class VerificationMetrics:
    """Metrics for verification operations."""

    operation_id: str
    timestamp: datetime
    model: str
    num_spans_verified: int
    num_verified: int
    num_rejected: int
    latency_ms: float
    cost_usd: float


class MetricsCollector:
    """Collects and aggregates metrics across ContractLens operations."""

    def __init__(self) -> None:
        self._extraction_metrics: list[ExtractionMetrics] = []
        self._verification_metrics: list[VerificationMetrics] = []

    def record_extraction(
        self,
        metrics: ExtractionMetrics,
    ) -> None:
        """Record extraction metrics."""
        self._extraction_metrics.append(metrics)

    def record_verification(
        self,
        metrics: VerificationMetrics,
    ) -> None:
        """Record verification metrics."""
        self._verification_metrics.append(metrics)

    def get_extraction_summary(self) -> dict:
        """Get summary of extraction metrics."""
        if not self._extraction_metrics:
            return {
                "total_operations": 0,
                "successful_operations": 0,
                "failed_operations": 0,
                "total_spans_extracted": 0,
                "total_latency_ms": 0.0,
                "total_cost_usd": 0.0,
                "avg_latency_ms": 0.0,
                "avg_cost_usd": 0.0,
            }

        successful = [m for m in self._extraction_metrics if m.success]
        failed = [m for m in self._extraction_metrics if not m.success]

        return {
            "total_operations": len(self._extraction_metrics),
            "successful_operations": len(successful),
            "failed_operations": len(failed),
            "total_spans_extracted": sum(m.num_spans_extracted for m in self._extraction_metrics),
            "total_latency_ms": sum(m.latency_ms for m in self._extraction_metrics),
            "total_cost_usd": sum(m.cost_usd for m in self._extraction_metrics),
            "avg_latency_ms": sum(m.latency_ms for m in self._extraction_metrics)
            / len(self._extraction_metrics),
            "avg_cost_usd": sum(m.cost_usd for m in self._extraction_metrics)
            / len(self._extraction_metrics),
        }

    def get_verification_summary(self) -> dict:
        """Get summary of verification metrics."""
        if not self._verification_metrics:
            return {
                "total_operations": 0,
                "total_spans_verified": 0,
                "total_verified": 0,
                "total_rejected": 0,
                "verification_rate": 0.0,
                "total_latency_ms": 0.0,
                "total_cost_usd": 0.0,
            }

        total_verified = sum(m.num_verified for m in self._verification_metrics)
        total_rejected = sum(m.num_rejected for m in self._verification_metrics)
        total_spans = total_verified + total_rejected

        return {
            "total_operations": len(self._verification_metrics),
            "total_spans_verified": total_spans,
            "total_verified": total_verified,
            "total_rejected": total_rejected,
            "verification_rate": total_verified / total_spans if total_spans > 0 else 0.0,
            "total_latency_ms": sum(m.latency_ms for m in self._verification_metrics),
            "total_cost_usd": sum(m.cost_usd for m in self._verification_metrics),
        }

    def get_model_performance(self) -> dict:
        """Get performance metrics grouped by model."""
        models: dict[str, dict] = {}

        for metric in self._extraction_metrics:
            if metric.model not in models:
                models[metric.model] = {
                    "operations": 0,
                    "total_latency_ms": 0.0,
                    "total_cost_usd": 0.0,
                    "spans_extracted": 0,
                }

            m = models[metric.model]
            m["operations"] += 1
            m["total_latency_ms"] += metric.latency_ms
            m["total_cost_usd"] += metric.cost_usd
            m["spans_extracted"] += metric.num_spans_extracted

        # Compute averages
        for model in models:
            m = models[model]
            ops = m["operations"]
            m["avg_latency_ms"] = m["total_latency_ms"] / ops
            m["avg_cost_usd"] = m["total_cost_usd"] / ops

        return models

    def clear(self) -> None:
        """Clear all metrics."""
        self._extraction_metrics.clear()
        self._verification_metrics.clear()