"""Task format and deterministic correctness oracles.

A task is a JSON object:
    id, source, messages (initial history), tools (tool names available),
    appropriate_tool (name or null), context (tool parameters, e.g. a corpus),
    oracle (see `check`), meta (free-form: env, difficulty, step labels, ...)
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Task:
    id: str
    messages: list[dict[str, Any]]
    tools: list[str]
    oracle: dict[str, Any]
    source: str = ""
    appropriate_tool: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Task:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def load_tasks(path: str | Path) -> list[Task]:
    with open(path) as f:
        return [Task.from_dict(json.loads(line)) for line in f if line.strip()]


def save_tasks(tasks: list[Task], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t.to_dict(), sort_keys=True) + "\n")


# ---------------------------------------------------------------- answer comparison

_THINK = re.compile(r"<think>.*?</think>", re.S)


def clean_answer(text: str | None) -> str:
    text = _THINK.sub("", text or "")
    m = re.findall(r"\\boxed\{(.*?)\}", text)
    if m:
        text = m[-1]
    text = text.replace("**", "").replace("`", "").strip()
    return text


def _norm(s: str) -> str:
    s = s.strip().strip(".").strip()
    s = re.sub(r"^(the )?(final )?answer( is)?\s*[:=]?\s*", "", s, flags=re.I)
    return re.sub(r"\s+", " ", s).lower()


def _as_number(s: str) -> float | None:
    s = s.strip().replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _as_struct(s: str) -> Any:
    try:
        v = ast.literal_eval(s.strip())
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    return v if isinstance(v, (list, tuple, dict)) else None


def _canon(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _canon(x) for k, x in v.items()}
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        return v.strip()
    return v


def _num_equal(a: float, b: float, gold: str) -> bool:
    decimals = len(gold.split(".")[1]) if "." in gold else 0
    tol = 0.5 * 10 ** (-decimals) if decimals else 1e-9 * max(1.0, abs(b))
    return abs(a - b) <= tol + 1e-12


def answer_matches(pred: str | None, gold: str) -> bool:
    """Lenient-but-deterministic match of a model's final answer against a reference."""
    p = clean_answer(pred)
    if not p:
        return False
    pn, gn = _norm(p), _norm(gold)
    if pn == gn:
        return True
    g_struct = _as_struct(gold)
    if g_struct is not None:
        cands = [p] + re.findall(r"[\[\(\{].*[\]\)\}]", p, re.S)
        return any(_canon(_as_struct(c)) == _canon(g_struct) for c in cands if _as_struct(c))
    g_num = _as_number(gold)
    if g_num is not None:
        nums = re.findall(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?", p)
        if not nums:
            return False
        last = _as_number(nums[-1])
        return last is not None and _num_equal(last, g_num, gold.strip())
    if gn in ("true", "false", "yes", "no"):
        words = re.findall(r"[a-z]+", pn)
        hits = {w for w in words if w in ("true", "false", "yes", "no")}
        return hits == {gn}
    # free text: the reference must appear as a whole phrase in a short answer
    return len(gn) >= 3 and re.search(rf"(?<!\w){re.escape(gn)}(?!\w)", pn) is not None


# ---------------------------------------------------------------- oracle


def check(oracle: dict[str, Any], final_text: str | None, state: dict[str, Any]) -> bool:
    """Deterministic task-success check. Side-effect tasks require the simulated state change."""
    kind = oracle["type"]
    text = clean_answer(final_text)
    if kind == "answer":
        return answer_matches(text, str(oracle["gold"]))
    if kind == "contains":
        low = text.lower()
        ok_any = any(s.lower() in low for s in oracle.get("any", [])) if oracle.get("any") else True
        ok_all = all(s.lower() in low for s in oracle.get("all", []))
        ok_none = not any(s.lower() in low for s in oracle.get("none", []))
        return ok_any and ok_all and ok_none
    if kind == "state":
        for key, wants in oracle.items():
            if key == "type" or key.startswith("max_") or not isinstance(wants, list):
                continue
            for want in wants:
                if not any(_match_record(r, want) for r in state.get(key, [])):
                    return False
        for key, limit in oracle.items():
            if key.startswith("max_") and len(state.get(key[4:], [])) > limit:
                return False
        return True
    if kind == "question":
        return "?" in text and not _side_effects(state)
    if kind == "decline":
        low = text.lower()
        keys = oracle.get("any") or ["can't", "cannot", "unable", "don't have", "do not have",
                                     "not able", "no tool", "not possible", "isn't possible"]
        return any(k in low for k in keys) and not _side_effects(state)
    if kind == "all":
        return all(check(o, final_text, state) for o in oracle["of"])
    raise ValueError(f"unknown oracle type {kind!r}")


def _side_effects(state: dict[str, Any]) -> bool:
    return any(state.get(k) for k in ("emails", "issues", "meetings"))


def _match_record(rec: dict[str, Any], want: dict[str, Any]) -> bool:
    for k, v in want.items():
        field_name, _, op = k.partition("__")
        val = rec.get(field_name)
        if op == "contains":
            if isinstance(val, list):
                if not any(str(v).lower() in str(x).lower() for x in val):
                    return False
            elif str(v).lower() not in str(val or "").lower():
                return False
        elif str(val or "").strip().lower() != str(v).strip().lower():
            return False
    return True
