"""Whetherman per-decision latency, p50/p95, CPU and GPU, across four input scenarios.

uv run python -m eval.latency --device cpu --checkpoint none   (untuned backbone; "laya" = Laya comparison)
-> eval/results/latency_<tag>_<device>.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from whetherman.tools.registry import REGISTRY, load_all

ROOT = Path(__file__).resolve().parents[1]
load_all()
DEV8 = ["read_file", "search_repo", "run_tests", "web_search", "get_weather", "calculator", "send_email",
        "create_issue"]


def tc(i, name, args):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def scenarios() -> dict[str, dict]:
    s = {}
    s["short_2tools"] = {"messages": [{"role": "user", "content": "What's the weather in Lisbon?"}],
                         "tools": [REGISTRY[n].schema() for n in ("get_weather", "calculator")]}
    hist = [{"role": "system", "content": "You are a developer assistant. Be concise."}]
    for i in range(4):
        hist += [{"role": "user", "content": f"Question {i}: can you explain how the cart total is computed, "
                  "including tax and shipping rules, in a couple of sentences?"},
                 {"role": "assistant", "content": "The total applies the discount, then tax, then adds shipping "
                  "below the free-shipping threshold. " * 2}]
    hist.append({"role": "user", "content": "And which file holds the tax rate?"})
    s["medium_8tools"] = {"messages": hist, "tools": [REGISTRY[n].schema() for n in DEV8]}
    ms = [{"role": "user", "content": "Run the tests and tell me why the failing one fails."},
          tc(1, "run_tests", {}), {"role": "tool", "tool_call_id": "c1", "content": json.dumps(
              {"exit_code": 1, "output": "....F.\nFAILED tests/test_cart.py::test_loyal_discount - assert 107.12 == "
               "94.38\n1 failed, 5 passed in 0.04s" + " " * 200})},
          tc(2, "read_file", {"path": "shopcart/pricing.py"}),
          {"role": "tool", "tool_call_id": "c2", "content": json.dumps(
              {"content": (ROOT / "data/fixtures/probe_pack/fixture_repo/shopcart/pricing.py").read_text()})}]
    s["multistep_2results"] = {"messages": ms, "tools": [REGISTRY[n].schema() for n in DEV8]}
    long = [{"role": "system", "content": "You are a developer assistant."}]
    for i in range(40):
        long += [{"role": "user", "content": f"Step {i}: " + "please review this paragraph carefully. " * 12},
                 {"role": "assistant", "content": "Reviewed. " + "The paragraph is fine. " * 10}]
    long.append({"role": "user", "content": "Now what's the weather in Oslo?"})
    s["long_near_budget"] = {"messages": long, "tools": [REGISTRY[n].schema() for n in DEV8]}
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--checkpoint", default="none")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    import torch

    from whetherman.adapters.functiongemma import FunctionGemmaRouter

    if a.device == "cpu":
        torch.set_num_threads(8)
    if a.checkpoint == "laya":  # P2 comparison router
        from whetherman.adapters.laya import LayaRouter

        r = LayaRouter(head=ROOT / "checkpoints" / "laya-head", device=a.device)
    else:
        r = FunctionGemmaRouter(checkpoint=None if a.checkpoint == "none" else ROOT / a.checkpoint, device=a.device)
    if a.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    res = {"device": a.device, "checkpoint": a.checkpoint, "threads": torch.get_num_threads(),
           "max_input_tokens": getattr(r, "max_input_tokens", getattr(r, "max_len", None)), "scenarios": {}}
    for name, d in scenarios().items():
        lat, fmt = [], []
        f = None
        for i in range(a.n + 3):
            t0 = time.perf_counter()
            f = r.format(d)
            t1 = time.perf_counter()
            r.predict(f)
            t2 = time.perf_counter()
            if i >= 3:  # warmup
                fmt.append((t1 - t0) * 1000)
                lat.append((t2 - t0) * 1000)
        res["scenarios"][name] = {"n": a.n, "p50_ms": float(np.percentile(lat, 50)),
                                  "p95_ms": float(np.percentile(lat, 95)),
                                  "format_p50_ms": float(np.percentile(fmt, 50)),
                                  "input_tokens": f["n_tokens"], "truncated": f["truncated"]}
        print(name, res["scenarios"][name], flush=True)
    if a.device == "cuda":
        res["peak_vram_mib"] = torch.cuda.max_memory_allocated() / 2**20
    tag = a.tag or ("untuned" if a.checkpoint == "none" else Path(a.checkpoint).name)
    out = ROOT / "eval" / "results" / f"latency_{tag}_{a.device}.json"
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
