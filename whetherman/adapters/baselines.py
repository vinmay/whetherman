"""Baseline routers sharing the adapter interface: rules-only control and a prompted LLM router."""

from __future__ import annotations

import json
import math
import re
import time
from typing import Any

import numpy as np

from whetherman.labels import LABELS

_IMPOSSIBLE = re.compile(r"\b(deploy|book|merge|order|phone|call|text|delete|transfer|reboot|restart|"
                         r"roll back|approve|post)\b", re.I)
_VAGUE = re.compile(r"\b(that|the thing|i mentioned|i told you|there|my total|the file)\b", re.I)
_STOP = {"the", "a", "an", "of", "to", "in", "and", "is", "for", "what", "me", "my", "it", "on", "with"}


def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", (s or "").lower()) if w not in _STOP and len(w) > 2}


def _onehot(label: str, conf: float = 0.9) -> list[float]:
    p = np.full(4, (1 - conf) / 3)
    p[LABELS.index(label)] = conf
    return np.log(p).tolist()


class RulesRouter:
    """Hand-written control: no learning, no agent information."""

    placement = "cpu"

    def format(self, d: dict[str, Any]) -> dict[str, Any]:
        return d

    def predict(self, d: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        msgs = d["messages"]
        last = msgs[-1]
        user = next((m.get("content") or "" for m in reversed(msgs) if m["role"] == "user"), "")
        tool_words = set()
        for t in d.get("tools") or []:
            f = t.get("function", t)
            tool_words |= _words(f.get("name", "").replace("_", " ") + " " + f.get("description", ""))
        if last["role"] == "tool":
            label = "ANSWER"  # a result just arrived: assume it is enough
        elif _IMPOSSIBLE.search(user) and not (_words(user) & tool_words):
            label = "DECLINE"
        elif _VAGUE.search(user) and len(user.split()) < 9:
            label = "ASK"
        elif not (_words(user) & tool_words):
            label = "ANSWER"
        else:
            label = "ACT"
        return {"label": label, "logits": _onehot(label), "probs": np.exp(_onehot(label)).tolist(),
                "latency_ms": (time.perf_counter() - t0) * 1000}


ROUTER_PROMPT = """You decide, for an AI agent, whether its NEXT step needs a tool call.
Read the conversation and the available tools, then reply with exactly one word:
ACT - the agent needs to call a tool now to complete this step reliably
ANSWER - the agent can reply reliably without a tool (including: the tool results so far are already enough)
ASK - required information is missing from the user; the agent should ask a clarifying question
DECLINE - no available tool can do what is needed and it cannot be meaningfully answered without one
Reply with only the word."""


def serialize(d: dict[str, Any], max_result_chars: int = 600) -> str:
    tools = []
    for t in d.get("tools") or []:
        f = t.get("function", t)
        params = list(((f.get("parameters") or {}).get("properties") or {}).keys())
        tools.append(f"- {f.get('name')}({', '.join(params)}): {f.get('description', '')}")
    lines = []
    for m in d["messages"]:
        if m["role"] == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                lines.append(f"ASSISTANT called {tc['function']['name']}({tc['function'].get('arguments', '')})")
        elif m["role"] == "tool":
            c = m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content"))
            lines.append(f"TOOL RESULT: {c[:max_result_chars]}")
        else:
            lines.append(f"{m['role'].upper()}: {m.get('content') or ''}")
    return "AVAILABLE TOOLS:\n" + "\n".join(tools) + "\n\nCONVERSATION:\n" + "\n".join(lines) + \
        "\n\nDecision for the agent's next step:"


class PromptedRouter:
    """Prompted LLM router over an OpenAI-compatible endpoint (default: Qwen3.5-4B via Ollama)."""

    def __init__(self, client) -> None:
        self.client = client
        self.placement = f"{client.spec.name}:{client.model}"

    def format(self, d: dict[str, Any]) -> dict[str, Any]:
        return {"messages": [{"role": "system", "content": ROUTER_PROMPT},
                             {"role": "user", "content": serialize(d)}]}

    def predict(self, f: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        body = self.client.build_body(f["messages"], temperature=0.0, seed=0, max_tokens=4, logprobs=True,
                                      top_logprobs=10)
        raw, _ = self.client.post_raw(body)
        ch = raw["choices"][0]
        text = (ch["message"].get("content") or "").strip().upper()
        scores = {k: -30.0 for k in LABELS}
        top = (((ch.get("logprobs") or {}).get("content") or [{}])[0] or {}).get("top_logprobs") or []
        for alt in top:
            tok = alt.get("token", "").strip().upper()
            for k in LABELS:
                if tok and (k.startswith(tok) and len(tok) >= 2 or tok.startswith(k)):
                    scores[k] = max(scores[k], alt["logprob"])
        if all(v == -30.0 for v in scores.values()):  # no logprobs: parse the text
            label = next((k for k in LABELS if text.startswith(k[:3])), "ACT")
            logits = _onehot(label, 0.97)
        else:
            logits = [scores[k] for k in LABELS]
        p = np.exp(np.array(logits) - max(logits))
        p /= p.sum()
        return {"label": LABELS[int(p.argmax())], "logits": logits, "probs": p.tolist(),
                "latency_ms": (time.perf_counter() - t0) * 1000, "text": text,
                "has_logprobs": not math.isclose(sum(np.exp(logits)), 1.0) or bool(top)}
