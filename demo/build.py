"""Bundle real A/B runs and cached routing outputs into demo/data.js for the demo page (demo/index.html).

uv run python -m demo.build [--suite ab_dev] [--tag _v3]

Nothing here runs a model: episodes come from eval/results/ab/, router outputs from the ladder's cache
(eval/results/routing/<checkpoint>/), probabilities from the shipped profiles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval import points
from whetherman.calibrate.adjust import softmax
from whetherman.labels import LABELS
from whetherman.policy.gate import Profile

ROOT = Path(__file__).resolve().parents[1]
AB = ROOT / "eval" / "results" / "ab"
OUT = ROOT / "demo" / "data.js"
AGENTS = ["L", "S"]
AGENT_NAMES = {"L": "gpt-oss-120b (DigitalOcean)", "S": "Qwen3.5-4B (local Ollama)"}


def read(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.open()] if p.exists() else []


def first_user(messages: list[dict]) -> str:
    return next((m["content"] for m in messages if m["role"] == "user" and isinstance(m.get("content"), str)), "")


def episodes(agent: str, suite: str, tag: str) -> list[dict]:
    rows = read(AB / f"{suite}_{agent}{tag}.jsonl")
    tasks = {t["id"]: t for t in read(ROOT / "data" / "fixtures" / "ab" / f"{suite}.jsonl")}
    verdicts = {(x["task_id"], x["repeat"]): x["verdict"] for x in read(AB / f"{suite}_{agent}{tag}.replay.jsonl")
                if x["config"] == "adapted"}
    base = {(r["task_id"], r["repeat"]): r for r in rows if r["config"] == "none"}
    out = []
    for r in rows:
        if r["config"] != "adapted":
            continue
        b = base.get((r["task_id"], r["repeat"]))
        out.append({
            "task_id": r["task_id"], "repeat": r["repeat"], "family": r.get("family"),
            "multi_step": r.get("multi_step", False), "prompt": first_user(tasks[r["task_id"]]["messages"]),
            "success": r["success"], "tool_calls": r["n_tool_calls"], "tools_called": r["tools_called"],
            "wall_s": round(r["wall_ms"] / 1000, 2), "final_text": r["final_text"][:300],
            "steps": [{"decision": g["decision"], "raw_label": g.get("raw_label"),
                       "act_prob": None if g.get("act_prob") is None else round(g["act_prob"], 4),
                       "tau": g.get("tau"), "gate_ms": round(g.get("latency_ms") or 0, 1),
                       "tools_removed": g.get("tools_removed", False),
                       "noncompliant": bool(g.get("noncompliant")), "error": g.get("error")}
                      for g in r.get("gate") or []],
            "replay_verdict": verdicts.get((r["task_id"], r["repeat"])),
            "baseline": None if b is None else {"success": b["success"], "tool_calls": b["n_tool_calls"],
                                                "tools_called": b["tools_called"],
                                                "wall_s": round(b["wall_ms"] / 1000, 2)},
        })
    return out


def cached(checkpoint: str, set_name: str) -> tuple[dict, dict]:
    logits, hidden = {}, {}
    for r in read(ROOT / "eval" / "results" / "routing" / checkpoint / f"{set_name}.jsonl"):
        logits[r["id"]] = r["logits"]
        if "hidden" in r:
            hidden[r["id"]] = r["hidden"]
    return logits, hidden


def routed_points(profiles: dict[str, Profile], checkpoint: str) -> list[dict]:
    """E2-train held-out points (never trained on): each agent's label and P(ACT) under each profile."""
    logits, hidden = cached(checkpoint, "e2_train_val")
    labels = {ag: {p["id"]: p for p in points.e2("e2_train", ag)} for ag in AGENTS}
    ids = sorted(set(labels["L"]) | set(labels["S"]))
    out = []
    for i in ids:
        if i not in hidden:
            continue
        p = labels["L"].get(i) or labels["S"][i]
        row = {"id": i, "prompt": first_user(p["messages"])[:400], "tools": [t["function"]["name"] for t in p["tools"]],
               "env": p["meta"].get("env"), "label": {ag: labels[ag][i]["label"] if i in labels[ag] else None
                                                    for ag in AGENTS}, "probs": {}}
        for ag, prof in profiles.items():
            row["probs"][ag] = np.round(prof.adjust.probs(np.array(logits[i]), np.array(hidden[i])), 4).tolist()
        out.append(row)
    return out


def four_decisions(checkpoint: str) -> list[dict]:
    """One correctly routed E1-val example per decision (shared backbone, default policy)."""
    logits, _ = cached(checkpoint, "e1_val")
    pts = {p["id"]: p for p in points.e1("val")}
    out, seen = [], set()
    for i, lg in logits.items():
        p = pts.get(i)
        if p is None or p["label"] in seen:
            continue
        pr = softmax(np.array(lg))
        if LABELS[int(pr.argmax())] == p["label"] and pr.max() > 0.9:
            seen.add(p["label"])
            out.append({"id": i, "label": p["label"], "prompt": first_user(p["messages"])[:300],
                        "tools": [t["function"]["name"] for t in p["tools"]], "probs": np.round(pr, 4).tolist()})
    return sorted(out, key=lambda x: LABELS.index(x["label"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="ab_dev")
    ap.add_argument("--tag", default="_v3", help="A/B file suffix: eval/results/ab/<suite>_<agent><tag>.jsonl")
    a = ap.parse_args()
    profiles = {ag: Profile.load(ROOT / "profiles" / f"{ag}.json") for ag in AGENTS}
    ck = Path(profiles["L"].backbone).name
    data = {
        "suite": a.suite, "checkpoint": ck, "labels": LABELS,
        "agents": {ag: {"name": AGENT_NAMES[ag], "tau": profiles[ag].tau, "episodes": episodes(ag, a.suite, a.tag)}
                   for ag in AGENTS},
        "points": routed_points(profiles, ck), "four_decisions": four_decisions(ck),
    }
    # E1 has no ANSWER class: take the ANSWER example from a held-out E2 point L's profile skips correctly
    tl = profiles["L"].tau
    ans = next((p for p in data["points"] if p["label"]["L"] == "ANSWER" and max(p["probs"]["L"][1:]) > tl
                and int(np.argmax(p["probs"]["L"])) == LABELS.index("ANSWER")), None)
    if ans:
        data["four_decisions"].insert(1, {"id": ans["id"], "label": "ANSWER", "prompt": ans["prompt"][:300],
                                          "tools": ans["tools"], "probs": ans["probs"]["L"], "profile": "L"})
    OUT.write_text("window.DEMO = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KiB): "
          + ", ".join(f"{ag} {len(d['episodes'])} episodes" for ag, d in data["agents"].items())
          + f", {len(data['points'])} routed points")


if __name__ == "__main__":
    main()
