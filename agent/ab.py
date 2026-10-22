"""End-to-end A/B: no gate vs Whetherman (adapted profile), per agent.

uv run python -m agent.ab --agent S --suite ab_dev --configs none adapted --repeats 2
  -> eval/results/ab/<suite>_<agent>.jsonl   one row per (task, config, repeat) episode
  -> eval/results/ab/<suite>_<agent>.replay.jsonl   pass->fail replay adjudication

Order control: configuration order is shuffled per (task, repeat) with a fixed seed, so prompt
caching cannot systematically favor one configuration. Cached-token usage is recorded per call.
Trajectories are never aligned step by step: success comes from task oracles only.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from eval import spend
from whetherman.agents import agent_config, make_client
from whetherman.episode import run_episode
from whetherman.policy.gate import Gate, Profile
from whetherman.proxy.server import GatedClient
from whetherman.tasks import load_tasks

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "results" / "ab"
SEED = 2026


def build_gate(config: str, agent: str, device: str, routers: dict, profiles: str = "profiles") -> Gate | None:
    if config == "none":
        return None
    from whetherman.adapters.functiongemma import FunctionGemmaRouter

    profile_for = {"adapted": f"{profiles}/{agent}.json", "shared": None, "swap": None}
    if config.startswith("adapted"):
        prof = Profile.load(ROOT / profile_for["adapted"])
    elif config == "shared":  # same backbone as the agent's profile, identity adjustment, default tau 0.5
        prof = Profile(agent="shared", tau=0.5, backbone=Profile.load(ROOT / profile_for["adapted"]).backbone)
    elif config.startswith("profile:"):  # profile swap, e.g. profile:L
        prof = Profile.load(ROOT / f"{profiles}/{config.split(':', 1)[1]}.json")
    elif config == "rules":
        from whetherman.adapters.baselines import RulesRouter

        return Gate(RulesRouter(), Profile(agent="rules", tau=0.5))
    elif config == "qwen_prompted":  # P1: prompted Qwen3.5-4B router (local Ollama), default policy
        from whetherman.adapters.baselines import PromptedRouter
        from whetherman.providers.openai_compat import ChatClient

        return Gate(PromptedRouter(ChatClient("qwen3.5:4b", "ollama")), Profile(agent="qwen_prompted", tau=0.5),
                    timeout_ms=30000)
    else:
        raise ValueError(config)
    key = (prof.backbone, device)
    if key not in routers:
        routers[key] = FunctionGemmaRouter(checkpoint=ROOT / prof.backbone, device=device)
    return Gate(routers[key], prof, timeout_ms=5000)


def replay_adjudicate(client, task, ep_row: dict[str, Any], max_steps: int, n: int = 2) -> dict[str, Any]:
    """For a pass->fail task: replay from each intervention point with ACT forced (ungated).
    A restored success is evidence the intervention caused the failure, not proof."""
    msgs = ep_row["messages"]
    gates = ep_row["gate_steps"]
    # the i-th model call sees the history before the i-th assistant message
    assistant_idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    results = []
    for step, g in enumerate(gates):
        if not g or g.get("decision") in (None, "ACT") or step >= len(assistant_idx):
            continue
        prefix = msgs[: assistant_idx[step]]
        outs = []
        for r in range(n):
            ep = run_episode(client, task, with_tools=True, max_steps=max_steps, prefix=prefix, seed=500 + r)
            outs.append({"success": ep.success, "n_tool_calls": ep.n_tool_calls})
        results.append({"step": step, "decision": g["decision"], "replays": outs})
    restored = any(o["success"] for r in results for o in r["replays"])
    return {"task_id": task.id, "repeat": ep_row["repeat"], "points": results,
            "verdict": "restored_with_act (evidence, not proof)" if restored else
            ("no_intervention_found" if not results else "unresolved")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--suite", default="ab_dev")
    ap.add_argument("--configs", nargs="+", default=["none", "adapted"])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--est-usd", type=float, default=0.0)
    ap.add_argument("--profiles", default="profiles", help="directory holding <agent>.json profiles")
    a = ap.parse_args()

    tasks = load_tasks(ROOT / "data" / "fixtures" / "ab" / f"{a.suite}.jsonl")[: a.limit]
    hosted = agent_config(a.agent)["provider"] != "ollama"
    if hosted:
        spend.check_cap("hosted", a.est_usd)
    routers: dict = {}
    gates = {c: build_gate(c, a.agent, a.device, routers, a.profiles) for c in a.configs}
    base = make_client(a.agent)
    clients = {c: GatedClient(base, g, profile_name=c) for c, g in gates.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{a.suite}_{a.agent}{a.tag}.jsonl"
    done = set()
    if out.exists():
        done = {(r["task_id"], r["config"], r["repeat"]) for r in map(json.loads, out.open())}
    rng = random.Random(SEED)
    t0, cost = time.time(), 0.0
    with out.open("a") as f:
        for ti, task in enumerate(tasks):
            for rep in range(a.repeats):
                order = list(a.configs)
                rng.shuffle(order)
                for pos, cfg in enumerate(order):
                    if (task.id, cfg, rep) in done:
                        continue
                    ep = run_episode(clients[cfg], task, with_tools=True, max_steps=a.max_steps, seed=rep)
                    row = {**ep.summary(), "config": cfg, "repeat": rep, "order_pos": pos, "agent": a.agent,
                           "suite": a.suite, "multi_step": bool(task.meta.get("multi_step")),
                           "family": task.meta.get("family") or task.meta.get("env"),
                           "gate_steps": [s.gate for s in ep.steps],
                           "step_latency_ms": [s.latency_ms for s in ep.steps],
                           "step_cached": [s.usage.cached_tokens for s in ep.steps],
                           "step_prompt": [s.usage.prompt_tokens for s in ep.steps],
                           "messages": ep.messages, "state": ep.state}
                    f.write(json.dumps(row, default=str) + "\n")
                    f.flush()
                    cost += ep.cost_usd
            print(f"[{a.agent}] {ti + 1}/{len(tasks)} {time.time() - t0:.0f}s ${cost:.4f}", flush=True)
    if hosted and cost:
        spend.record("hosted", f"A/B {a.suite} {a.agent} {'+'.join(a.configs)}", cost)

    # pass->fail adjudication (per repeat pair: none vs each gated config)
    rows = [json.loads(x) for x in out.open()]
    by = {(r["task_id"], r["config"], r["repeat"]): r for r in rows}
    adj = []
    for task in tasks:
        for rep in range(a.repeats):
            ung = by.get((task.id, "none", rep))
            for cfg in a.configs:
                g = by.get((task.id, cfg, rep))
                if cfg == "none" or not ung or not g:
                    continue
                if ung["success"] and not g["success"]:
                    res = replay_adjudicate(base, task, g, a.max_steps)
                    res["config"] = cfg
                    adj.append(res)
    (OUT / f"{a.suite}_{a.agent}{a.tag}.replay.jsonl").write_text("".join(json.dumps(x) + "\n" for x in adj))
    print(f"adjudicated {len(adj)} pass->fail cases")


if __name__ == "__main__":
    main()
