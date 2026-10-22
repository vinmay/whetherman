"""Append-only spend log at eval/results/spend.json."""

from __future__ import annotations

import json
import time
from pathlib import Path

SPEND = Path(__file__).resolve().parent / "results" / "spend.json"
CAPS = {"hosted": 10.0, "tinker": 10.0}


def load() -> dict:
    if SPEND.exists():
        return json.loads(SPEND.read_text())
    return {"caps_usd": CAPS, "entries": []}


def total(kind: str = "hosted") -> float:
    return sum(e["usd"] for e in load()["entries"] if e["kind"] == kind)


def record(kind: str, what: str, usd: float, **extra) -> None:
    d = load()
    d["caps_usd"] = CAPS
    d["entries"].append({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, "what": what,
                         "usd": round(usd, 6), **extra})
    d["totals_usd"] = {k: round(sum(e["usd"] for e in d["entries"] if e["kind"] == k), 6)
                       for k in {e["kind"] for e in d["entries"]}}
    SPEND.parent.mkdir(parents=True, exist_ok=True)
    SPEND.write_text(json.dumps(d, indent=1))


def check_cap(kind: str, planned: float) -> None:
    if total(kind) + planned > CAPS[kind]:
        raise SystemExit(f"spend cap: {kind} spent {total(kind):.2f} + planned {planned:.2f} "
                         f"> cap {CAPS[kind]:.2f}; stopping for owner approval")
