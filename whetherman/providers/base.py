"""Provider-neutral types for OpenAI-compatible Chat Completions backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.cached_tokens + other.cached_tokens,
        )


@dataclass
class ChatResult:
    """One completed chat call, normalized across providers."""

    message: dict[str, Any]  # assistant message: role, content, optional tool_calls
    usage: Usage
    latency_ms: float
    cost_usd: float
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return self.message.get("tool_calls") or []


@dataclass
class Pricing:
    """USD per million tokens. Cached input falls back to the input price."""

    input_per_m: float = 0.0
    output_per_m: float = 0.0
    cached_input_per_m: float | None = None

    def cost(self, u: Usage) -> float:
        cached_price = self.input_per_m if self.cached_input_per_m is None else self.cached_input_per_m
        uncached = max(u.prompt_tokens - u.cached_tokens, 0)
        return (
            uncached * self.input_per_m
            + u.cached_tokens * cached_price
            + u.completion_tokens * self.output_per_m
        ) / 1e6
