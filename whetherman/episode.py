"""Run one agent episode on a task: model call -> tool execution loop -> oracle check."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from whetherman.providers.base import ChatResult, Usage
from whetherman.tasks import Task, check
from whetherman.tools.registry import ToolEnv, load_all

AGENT_SYSTEM = "You are a helpful assistant. Be concise and give the final answer directly."


class Agent(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
             tool_choice: Any = None, **kw: Any) -> ChatResult: ...


@dataclass
class StepRecord:
    tools_offered: bool
    tool_calls: list[str]
    latency_ms: float
    usage: Usage
    cost_usd: float
    gate: dict[str, Any] | None = None  # decision info when run through a gate


@dataclass
class Episode:
    task_id: str
    with_tools: bool
    success: bool
    final_text: str | None
    steps: list[StepRecord] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)
    wall_ms: float = 0.0
    error: str | None = None

    @property
    def n_tool_calls(self) -> int:
        return len(self.tool_calls)

    @property
    def usage(self) -> Usage:
        u = Usage()
        for s in self.steps:
            u = u + s.usage
        return u

    @property
    def cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.steps)

    def summary(self) -> dict[str, Any]:
        u = self.usage
        return {
            "task_id": self.task_id, "with_tools": self.with_tools, "success": self.success,
            "n_tool_calls": self.n_tool_calls, "tools_called": [c["name"] for c in self.tool_calls],
            "n_model_calls": len(self.steps), "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens, "cached_tokens": u.cached_tokens,
            "cost_usd": self.cost_usd, "wall_ms": self.wall_ms,
            "model_ms": sum(s.latency_ms for s in self.steps), "error": self.error,
            "final_text": (self.final_text or "")[:500],
            "gate": [s.gate for s in self.steps if s.gate],
        }


def initial_messages(task: Task, system: str | None = AGENT_SYSTEM) -> list[dict[str, Any]]:
    msgs = copy.deepcopy(task.messages)
    if system and not any(m["role"] == "system" for m in msgs):
        msgs.insert(0, {"role": "system", "content": system})
    return msgs


def run_episode(agent: Agent, task: Task, with_tools: bool = True, max_steps: int = 8,
                system: str | None = AGENT_SYSTEM, **chat_kw: Any) -> Episode:
    load_all()
    env = ToolEnv(tool_names=list(task.tools), context=copy.deepcopy(task.context))
    tools = env.schemas() if with_tools and task.tools else None
    msgs = initial_messages(task, system)
    ep = Episode(task_id=task.id, with_tools=with_tools, success=False, final_text=None)
    t0 = time.perf_counter()
    try:
        for _ in range(max_steps):
            r = agent.chat(msgs, tools=tools, **chat_kw)
            ep.steps.append(StepRecord(
                tools_offered=tools is not None, tool_calls=[c["function"]["name"] for c in r.tool_calls],
                latency_ms=r.latency_ms, usage=r.usage, cost_usd=r.cost_usd,
                gate=getattr(r, "gate", None),
            ))
            msgs.append(r.message)
            if not r.tool_calls:
                ep.final_text = r.message.get("content")
                break
            for tc in r.tool_calls:
                out = env.execute(tc["function"]["name"], tc["function"]["arguments"])
                ep.tool_calls.append({"name": tc["function"]["name"],
                                      "arguments": tc["function"]["arguments"]})
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
    except Exception as e:  # record and fail the episode; never crash a batch
        ep.error = f"{type(e).__name__}: {e}"[:500]
    ep.wall_ms = (time.perf_counter() - t0) * 1000
    ep.state = env.state
    ep.messages = msgs
    ep.success = ep.error is None and check(task.oracle, ep.final_text, env.state)
    return ep
