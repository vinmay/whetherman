"""Build labeled decision points (exact history at the decision moment) for every eval set."""

from __future__ import annotations

import json
from pathlib import Path

from whetherman.tools.registry import REGISTRY, load_all

FIX = Path(__file__).resolve().parents[1] / "data" / "fixtures"
load_all()


def _read(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.open()] if p.exists() else []


def _schemas(names: list[str]) -> list[dict]:
    return [REGISTRY[n].schema() for n in names]


def e1(split: str) -> list[dict]:
    return [{**r, "multi_step": False, "unseen_schema": False} for r in _read(FIX / "e1" / f"{split}.jsonl")]


def labeled_tasks(task_file: Path, agent: str, label_file: str) -> list[dict]:
    """Executable tasks -> first-step decision points with this agent's usable labels."""
    labels = {r["task_id"]: r for r in _read(FIX / "labels" / agent / label_file)}
    out = []
    for t in _read(task_file):
        lab = labels.get(t["id"])
        if lab is None or lab["label"] not in ("ACT", "ANSWER"):
            continue
        out.append({"id": t["id"], "source": t["source"], "messages": t["messages"],
                    "tools": _schemas(t["tools"]), "label": lab["label"],
                    "multi_step": bool(t["meta"].get("multi_step")), "unseen_schema": False,
                    "meta": t["meta"]})
    return out


def ambiguous_count(agent: str, label_file: str) -> dict:
    rows = _read(FIX / "labels" / agent / label_file)
    return {"labeled": len(rows), "ambiguous": sum(r["label"] == "ambiguous" for r in rows)}


def e2(split: str, agent: str) -> list[dict]:
    return labeled_tasks(FIX / "w2t" / f"{split}.jsonl", agent, f"{split}.jsonl")


def probe(agent: str) -> list[dict]:
    return (labeled_tasks(FIX / "w2t" / "probe.jsonl", agent, "probe.jsonl")
            + labeled_tasks(FIX / "probe_pack" / "probe_dev.jsonl", agent, "probe_dev.jsonl"))


def e3_capability_tasks() -> list[dict]:
    """E3 capability points as executable tasks (history prefix + oracle) for label_necessity."""
    out = []
    for p in _read(FIX / "e3" / "e3.jsonl"):
        if p["label_mode"] == "capability":
            out.append({"id": p["id"], "source": "e3", "messages": p["messages"], "tools": p["tools"],
                        "oracle": p["oracle"], "appropriate_tool": None, "context": {},
                        "meta": {"multi_step": p["step"] > 0, "conv_id": p["conv_id"]}})
    return out


def e3(agent: str | None) -> list[dict]:
    """Fixed-label points plus (if agent given) that agent's usable capability labels."""
    cap_file = FIX / "labels" / str(agent) / "e3_capability_tasks.jsonl"
    caps = {r["task_id"]: r["label"] for r in _read(cap_file)} if agent else {}
    pts = _read(FIX / "e3" / "e3.jsonl")
    multi = {q["conv_id"] for q in pts if q["step"] > 0}
    out = []
    for p in pts:
        lab = p["label"] or caps.get(p["id"])
        if lab not in ("ACT", "ANSWER", "ASK", "DECLINE"):
            continue
        out.append({"id": p["id"], "source": "e3", "messages": p["messages"], "tools": _schemas(p["tools"]),
                    "label": lab, "label_mode": p["label_mode"], "multi_step": p["conv_id"] in multi,
                    "step": p["step"], "unseen_schema": p["unseen_schema"]})
    return out
