"""OpenAI-compatible Chat Completions client with per-provider quirks.

Supported backends are declared in PROVIDERS. Each entry pins the base URL, the env var that
holds the API key, pricing, and any request-body extras the provider needs.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from whetherman.providers.base import ChatResult, Pricing, Usage


@dataclass
class ProviderSpec:
    name: str
    base_url: str
    api_key_env: str | None = None
    pricing: dict[str, Pricing] = field(default_factory=dict)  # by model id
    extra_body: dict[str, Any] = field(default_factory=dict)
    supports_seed: bool = True
    timeout_s: float = 180.0  # read timeout; must exceed the slowest legitimate full-length generation


PROVIDERS: dict[str, ProviderSpec] = {
    # DigitalOcean Gradient serverless inference. Prices are per 1M tokens; reverify in MODELS.md.
    "digitalocean": ProviderSpec(
        name="digitalocean",
        base_url="https://inference.do-ai.run/v1",
        api_key_env="DIGITALOCEAN_INFERENCE_KEY",
        pricing={
            "openai-gpt-oss-120b": Pricing(input_per_m=0.10, output_per_m=0.70),
            "openai-gpt-oss-20b": Pricing(input_per_m=0.05, output_per_m=0.45),
        },
    ),
    # Local Ollama. Free; we disable Qwen3.5 thinking so the agent answers directly.
    "ollama": ProviderSpec(
        name="ollama",
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        extra_body={"reasoning_effort": "none"},
        timeout_s=900.0,  # a small local model can take minutes to emit max_tokens; that is agent behavior
    ),
    # Any other OpenAI-compatible endpoint serving an open-weight model.
    "openai_compat": ProviderSpec(
        name="openai_compat",
        base_url=os.environ.get("WHETHERMAN_AGENT_BASE_URL", "http://localhost:8000/v1"),
        api_key_env="WHETHERMAN_AGENT_API_KEY",
    ),
}


def provider_for_url(base_url: str) -> ProviderSpec:
    for spec in PROVIDERS.values():
        if base_url.rstrip("/") == spec.base_url.rstrip("/"):
            return spec
    if ":11434" in base_url:
        return ProviderSpec(name="ollama", base_url=base_url, extra_body=PROVIDERS["ollama"].extra_body,
                            timeout_s=PROVIDERS["ollama"].timeout_s)
    return ProviderSpec(name="openai_compat", base_url=base_url, api_key_env="WHETHERMAN_AGENT_API_KEY")


def parse_usage(raw: dict[str, Any]) -> Usage:
    u = raw.get("usage") or {}
    details = u.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") or u.get("cached_tokens") or 0
    return Usage(
        prompt_tokens=int(u.get("prompt_tokens") or 0),
        completion_tokens=int(u.get("completion_tokens") or 0),
        cached_tokens=int(cached),
    )


def normalize_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Reduce a provider's assistant message to the standard fields we send back."""
    out: dict[str, Any] = {"role": "assistant", "content": msg.get("content")}
    calls = []
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        args = fn.get("arguments", "{}")
        if not isinstance(args, str):
            args = json.dumps(args)
        calls.append(
            {
                "id": tc.get("id") or f"call_{i}",
                "type": "function",
                "function": {"name": fn.get("name", ""), "arguments": args},
            }
        )
    if calls:
        out["tool_calls"] = calls
    return out


RETRY_STATUS = {403, 408, 409, 429, 500, 502, 503, 504}


class ChatClient:
    """Thin synchronous client. One instance per (provider, model)."""

    def __init__(
        self,
        model: str,
        provider: str | ProviderSpec = "ollama",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 8,
    ) -> None:
        self.max_retries = max_retries
        spec = PROVIDERS[provider] if isinstance(provider, str) else provider
        if base_url:
            spec = ProviderSpec(**{**spec.__dict__, "base_url": base_url})
        self.spec = spec
        self.model = model
        key = api_key or (os.environ.get(spec.api_key_env) if spec.api_key_env else None)
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self._http = httpx.Client(
            base_url=spec.base_url, headers=headers, timeout=timeout or spec.timeout_s, transport=transport
        )
        self.pricing = spec.pricing.get(model, Pricing())

    def build_body(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        **params: Any,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "messages": messages, **self.spec.extra_body}
        if tools:
            body["tools"] = tools
            if tool_choice is not None:
                body["tool_choice"] = tool_choice
        if not self.spec.supports_seed:
            params.pop("seed", None)
        body.update({k: v for k, v in params.items() if v is not None})
        return body

    def post_raw(self, body: dict[str, Any]) -> tuple[dict[str, Any], float]:
        """POST with retries on rate limits, transient 403/5xx and timeouts. Latency is the final attempt's."""
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                r = self._http.post("/chat/completions", json=body)
            except httpx.TransportError:
                if attempt == self.max_retries:
                    raise
                time.sleep(min(2**attempt, 60) * (1 + random.random()))
                continue
            latency = (time.perf_counter() - t0) * 1000
            if r.status_code in RETRY_STATUS and attempt < self.max_retries:
                ra = r.headers.get("retry-after", "")
                wait = float(ra) if ra.replace(".", "", 1).isdigit() else min(2**attempt, 60)
                time.sleep(wait * (1 + random.random()))
                continue
            r.raise_for_status()
            return r.json(), latency
        raise RuntimeError("unreachable")

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        temperature: float = 0.0,
        seed: int | None = 0,
        **params: Any,
    ) -> ChatResult:
        body = self.build_body(
            messages, tools, tool_choice, temperature=temperature, seed=seed, **params
        )
        raw, latency = self.post_raw(body)
        choice = (raw.get("choices") or [{}])[0]
        usage = parse_usage(raw)
        return ChatResult(
            message=normalize_message(choice.get("message") or {}),
            usage=usage,
            latency_ms=latency,
            cost_usd=self.pricing.cost(usage),
            finish_reason=choice.get("finish_reason"),
            raw=raw,
        )

    def close(self) -> None:
        self._http.close()
