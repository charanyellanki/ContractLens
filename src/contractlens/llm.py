"""LiteLLM wrapper with cost and latency logging."""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from litellm import acompletion, completion

from contractlens.models import ClauseCategory

logger = logging.getLogger(__name__)


# Cost per 1M tokens (as of 2024) - USD
LLM_COSTS: dict[str, dict[str, float]] = {
    "gpt-4o": {
        "input": 5.00,    # $5.00 per 1M input tokens
        "output": 15.00,  # $15.00 per 1M output tokens
    },
    "gpt-4o-mini": {
        "input": 0.15,    # $0.15 per 1M input tokens
        "output": 0.60,   # $0.60 per 1M output tokens
    },
    "gpt-4-turbo": {
        "input": 10.00,
        "output": 30.00,
    },
    "claude-3-opus": {
        "input": 15.00,
        "output": 75.00,
    },
    "claude-3-sonnet": {
        "input": 3.00,
        "output": 15.00,
    },
    "claude-3-haiku": {
        "input": 0.25,
        "output": 1.25,
    },
    # Placeholder for LoRA-fine-tuned Llama 3 8B
    "llama-3-8b-lora": {
        "input": 0.00,    # Self-hosted, no API cost
        "output": 0.00,
    },
}


@dataclass
class TokenUsage:
    """Token usage from an LLM call."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class LLMCallResult:
    """Result of an LLM call with telemetry."""

    content: str
    model: str
    token_usage: TokenUsage
    latency_ms: float
    cost_usd: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class LLMWrapper:
    """LiteLLM wrapper with cost and latency tracking."""

    def __init__(self, default_model: str = "gpt-4o-mini") -> None:
        self.default_model = default_model
        self._call_history: list[LLMCallResult] = []

    def _calculate_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Calculate cost in USD based on token usage."""
        costs = LLM_COSTS.get(model, {"input": 0.0, "output": 0.0})
        input_cost = (prompt_tokens / 1_000_000) * costs["input"]
        output_cost = (completion_tokens / 1_000_000) * costs["output"]
        return input_cost + output_cost

    def _extract_tokens(self, response: Any) -> TokenUsage:
        """Extract token usage from LiteLLM response."""
        usage = response.usage
        return TokenUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )

    def complete(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMCallResult:
        """Make a synchronous LLM call with telemetry."""
        model = model or self.default_model
        start_time = time.perf_counter()

        try:
            response = completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000
            token_usage = self._extract_tokens(response)
            content = response.choices[0].message.content or ""
            cost_usd = self._calculate_cost(
                model, token_usage.prompt_tokens, token_usage.completion_tokens
            )

            result = LLMCallResult(
                content=content,
                model=model,
                token_usage=token_usage,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
            )
            self._call_history.append(result)
            logger.info(
                f"LLM call completed: model={model}, "
                f"tokens={token_usage.total_tokens}, "
                f"latency={latency_ms:.0f}ms, cost=${cost_usd:.4f}"
            )
            return result

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"LLM call failed: model={model}, error={e}")
            raise

    async def acomplete(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMCallResult:
        """Make an asynchronous LLM call with telemetry."""
        model = model or self.default_model
        start_time = time.perf_counter()

        try:
            response = await acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000
            token_usage = self._extract_tokens(response)
            content = response.choices[0].message.content or ""
            cost_usd = self._calculate_cost(
                model, token_usage.prompt_tokens, token_usage.completion_tokens
            )

            result = LLMCallResult(
                content=content,
                model=model,
                token_usage=token_usage,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
            )
            self._call_history.append(result)
            logger.info(
                f"Async LLM call completed: model={model}, "
                f"tokens={token_usage.total_tokens}, "
                f"latency={latency_ms:.0f}ms, cost=${cost_usd:.4f}"
            )
            return result

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Async LLM call failed: model={model}, error={e}")
            raise

    def get_call_history(self) -> list[LLMCallResult]:
        """Get the history of LLM calls."""
        return self._call_history.copy()

    def get_total_cost(self) -> float:
        """Get total cost of all calls in USD."""
        return sum(call.cost_usd for call in self._call_history)

    def get_total_tokens(self) -> int:
        """Get total tokens used across all calls."""
        return sum(call.token_usage.total_tokens for call in self._call_history)

    def get_average_latency(self) -> float:
        """Get average latency in milliseconds."""
        if not self._call_history:
            return 0.0
        return sum(call.latency_ms for call in self._call_history) / len(
            self._call_history
        )

    def reset_history(self) -> None:
        """Reset the call history."""
        self._call_history.clear()


# Global wrapper instance
_default_wrapper: Optional[LLMWrapper] = None


def get_llm_wrapper() -> LLMWrapper:
    """Get the global LLM wrapper instance."""
    global _default_wrapper
    if _default_wrapper is None:
        _default_wrapper = LLMWrapper()
    return _default_wrapper