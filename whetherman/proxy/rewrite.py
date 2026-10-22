"""Request routing and rewriting for the supported Chat Completions subset.

Supported: non-streaming requests with tools present and tool_choice "auto" or unset.
Everything else is forwarded unchanged.
"""

from __future__ import annotations

import copy
from typing import Any

ASK_INSTRUCTION = (
    "Before doing anything else, ask the user exactly one short clarifying question to get the "
    "information that is missing from their request. Do not answer yet."
)
DECLINE_INSTRUCTION = (
    "None of your available tools can do what the user is asking. Briefly explain that limitation "
    "and, if possible, suggest what the user could do instead. Do not pretend to perform the action."
)
ANSWER_INSTRUCTION = (
    "Answer the user now using only the information already in this conversation. Do not call any tools."
)
TOOL_KEYS = ("tools", "tool_choice", "parallel_tool_calls", "functions", "function_call")


def route_reason(body: Any) -> str | None:
    """None if this request should be gated; otherwise why it is forwarded transparently."""
    if not isinstance(body, dict):
        return "malformed"
    if body.get("stream"):
        return "stream"
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs or not all(isinstance(m, dict) and "role" in m for m in msgs):
        return "malformed"
    tools = body.get("tools")
    if not isinstance(tools, list) or not tools:
        return "no_tools"
    tc = body.get("tool_choice")
    if tc not in (None, "auto"):
        return "forced_tool_choice"
    if any(isinstance(m.get("content"), list) and any(
            isinstance(p, dict) and p.get("type") not in ("text", None) for p in m["content"]) for m in msgs):
        return "non_text_content"
    if body.get("n", 1) != 1:
        return "n_gt_1"
    return None


def decision_input(body: dict[str, Any]) -> dict[str, Any]:
    msgs = body["messages"]
    return {"messages": msgs, "tools": body["tools"],
            "current_step": sum(1 for m in msgs if m.get("role") == "assistant")}


def _with_instruction(msgs: list[dict[str, Any]], instruction: str) -> list[dict[str, Any]]:
    """Add the instruction to the leading system message (or insert one); roles stay ordered."""
    msgs = copy.deepcopy(msgs)
    if msgs and msgs[0].get("role") in ("system", "developer") and isinstance(msgs[0].get("content"), str):
        msgs[0]["content"] = msgs[0]["content"].rstrip() + "\n\n" + instruction
    else:
        msgs.insert(0, {"role": "system", "content": instruction})
    return msgs


def rewrite(body: dict[str, Any], decision: str) -> dict[str, Any]:
    """Return the request to forward for a decision. ACT is the identity.

    Non-ACT removes every tool-related parameter (so tool_choice stays consistent) and keeps
    all messages, tool-call ids, tool results and other parameters untouched."""
    if decision == "ACT":
        return body
    out = {k: copy.deepcopy(v) for k, v in body.items() if k not in TOOL_KEYS}
    if decision == "ASK":
        out["messages"] = _with_instruction(body["messages"], ASK_INSTRUCTION)
    elif decision == "DECLINE":
        out["messages"] = _with_instruction(body["messages"], DECLINE_INSTRUCTION)
    elif decision == "ANSWER":
        out["messages"] = _with_instruction(body["messages"], ANSWER_INSTRUCTION)
    else:
        return body  # unknown decision: fail open
    return out


def has_tool_calls(response: Any) -> bool:
    """True if a Chat Completions response asks for a tool call (used to detect non-compliance)."""
    try:
        msg = (response.get("choices") or [{}])[0].get("message") or {}
    except AttributeError:
        return False
    return bool(msg.get("tool_calls") or msg.get("function_call"))


def schema_tokens_estimate(tools: list[dict[str, Any]]) -> int:
    """Rough prompt-token size of tool schemas (chars/4); exact savings come from provider usage."""
    import json

    return len(json.dumps(tools)) // 4
