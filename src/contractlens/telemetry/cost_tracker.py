"""Cost tracking for LLM calls."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from contractlens.llm import LLMCallResult


@dataclass
class CallRecord:
    """Record of a single LLM call."""

    call_id: str
    timestamp: datetime
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float
    operation: str  # extraction, verification, etc.


class CostTracker:
    """Tracks costs and token usage across ContractLens operations."""

    def __init__(self) -> None:
        self._records: list[CallRecord] = []
        self._operation_costs: dict[str, float] = {}
        self._operation_tokens: dict[str, int] = {}

    def record_call(
        self,
        call_result: LLMCallResult,
        operation: str,
    ) -> None:
        """Record a completed LLM call."""
        record = CallRecord(
            call_id=f"{operation}_{len(self._records)}",
            timestamp=call_result.timestamp,
            model=call_result.model,
            prompt_tokens=call_result.token_usage.prompt_tokens,
            completion_tokens=call_result.token_usage.completion_tokens,
            latency_ms=call_result.latency_ms,
            cost_usd=call_result.cost_usd,
            operation=operation,
        )
        self._records.append(record)

        # Aggregate by operation
        self._operation_costs[operation] = (
            self._operation_costs.get(operation, 0.0) + call_result.cost_usd
        )
        self._operation_tokens[operation] = (
            self._operation_tokens.get(operation, 0)
            + call_result.token_usage.total_tokens
        )

    def get_total_cost(self) -> float:
        """Get total cost across all operations."""
        return sum(r.cost_usd for r in self._records)

    def get_total_tokens(self) -> int:
        """Get total tokens used across all operations."""
        return sum(r.prompt_tokens + r.completion_tokens for r in self._records)

    def get_operation_cost(self, operation: str) -> float:
        """Get cost for a specific operation."""
        return self._operation_costs.get(operation, 0.0)

    def get_operation_tokens(self, operation: str) -> int:
        """Get tokens for a specific operation."""
        return self._operation_tokens.get(operation, 0)

    def get_average_latency(self, operation: Optional[str] = None) -> float:
        """Get average latency for operations."""
        if operation:
            ops = [r for r in self._records if r.operation == operation]
        else:
            ops = self._records

        if not ops:
            return 0.0

        return sum(r.latency_ms for r in ops) / len(ops)

    def get_cost_per_contract(self, num_contracts: int) -> float:
        """Calculate average cost per contract."""
        if num_contracts == 0:
            return 0.0
        return self.get_total_cost() / num_contracts

    def get_summary(self) -> dict:
        """Get a summary of all costs and usage."""
        return {
            "total_cost_usd": self.get_total_cost(),
            "total_tokens": self.get_total_tokens(),
            "total_prompt_tokens": sum(r.prompt_tokens for r in self._records),
            "total_completion_tokens": sum(r.completion_tokens for r in self._records),
            "operation_costs": self._operation_costs.copy(),
            "operation_tokens": self._operation_tokens.copy(),
            "average_latency_ms": self.get_average_latency(),
            "total_calls": len(self._records),
        }

    def clear(self) -> None:
        """Clear all records."""
        self._records.clear()
        self._operation_costs.clear()
        self._operation_tokens.clear()