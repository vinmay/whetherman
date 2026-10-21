"""Executable tool registry shared by labeling, calibration probes, E3 and the A/B harness.

Every tool is deterministic and offline-safe: real implementations for pure computations and
bundled fixtures, simulators for anything with side effects or network dependence.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]  # fn(env, **kwargs) -> JSON-serializable result

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


REGISTRY: dict[str, Tool] = {}


def tool(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None):
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        REGISTRY[name] = Tool(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": properties,
                "required": required if required is not None else list(properties),
            },
            fn=fn,
        )
        return fn

    return deco


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    result: Any
    ok: bool


@dataclass
class ToolEnv:
    """Per-episode tool state. `context` carries task parameters (e.g. a retrieval corpus)."""

    tool_names: list[str]
    context: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=lambda: {"emails": [], "issues": []})
    calls: list[ToolCallRecord] = field(default_factory=list)

    def schemas(self) -> list[dict[str, Any]]:
        return [REGISTRY[n].schema() for n in self.tool_names]

    def execute(self, name: str, arguments: str | dict[str, Any]) -> str:
        """Run a tool call and return the string content of the tool message."""
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except (json.JSONDecodeError, ValueError) as e:
            rec = ToolCallRecord(name, {"_raw": str(arguments)}, f"invalid arguments: {e}", False)
            self.calls.append(rec)
            return json.dumps({"error": rec.result})
        if name not in self.tool_names or name not in REGISTRY:
            rec = ToolCallRecord(name, args, f"unknown tool: {name}", False)
            self.calls.append(rec)
            return json.dumps({"error": rec.result})
        try:
            result = REGISTRY[name].fn(self, **args)
            ok = True
        except TypeError as e:
            result, ok = {"error": f"bad arguments: {e}"}, False
        except Exception as e:  # tool errors are returned to the agent, not raised
            result, ok = {"error": f"{type(e).__name__}: {e}"}, False
        self.calls.append(ToolCallRecord(name, copy.deepcopy(args), result, ok))
        return result if isinstance(result, str) else json.dumps(result, default=str)


def load_all() -> dict[str, Tool]:
    """Import tool modules so their @tool decorators register."""
    from whetherman.tools import dev, w2t  # noqa: F401

    return REGISTRY
