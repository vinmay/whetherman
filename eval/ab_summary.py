"""Summarize an A/B run: success with Wilson CIs, tool calls, tokens, cost, latency, interventions.

uv run python -m eval.ab_summary eval/results/ab/ab_dev_L_v3.jsonl  -> prints a table, writes <file>.summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from whetherman.labeling import wilson


def summarize(rows: list[dict], replays: list[dict]) -> dict:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["config"]].append(r)
    out: dict = {"configs": {}, "per_family": {}}
    for c, rs in by.items():
        n, k = len(rs), sum(r["success"] for r in rs)
        gates = [g for r in rs for g in (r.get("gate") or [])]
        out["configs"][c] = {
            "episodes": n, "success": k, "success_rate": k / n, "success_ci": list(wilson(k, n)),
            "tool_calls_mean": float(np.mean([r["n_tool_calls"] for r in rs])),
            "model_calls_mean": float(np.mean([r["n_model_calls"] for r in rs])),
            "tokens_mean": float(np.mean([r["prompt_tokens"] + r["completion_tokens"] for r in rs])),
            "cost_usd": float(sum(r["cost_usd"] for r in rs)),
            "wall_s_median": float(np.median([r["wall_ms"] for r in rs]) / 1000),
            "gate_steps": len(gates),
            "interventions": sum(g.get("decision") not in (None, "ACT") for g in gates),
            "noncompliant_fallbacks": sum(bool(g.get("noncompliant")) for g in gates),
            "gate_errors": dict(Counter(g["error"] for g in gates if g.get("error"))),
            "gate_ms_median": float(np.median([g["latency_ms"] for g in gates if g.get("latency_ms")]))
            if any(g.get("latency_ms") for g in gates) else None,
            "replay_verdicts": dict(Counter(x["verdict"] for x in replays if x["config"] == c)),
        }
        fam: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for r in rs:
            fam[r.get("family") or "?"][0] += 1
            fam[r.get("family") or "?"][1] += int(r["success"])
        out["per_family"][c] = {f: {"n": v[0], "success": v[1]} for f, v in sorted(fam.items())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    a = ap.parse_args()
    p = Path(a.path)
    rows = [json.loads(x) for x in p.open()]
    rp = p.with_suffix(".replay.jsonl")
    replays = [json.loads(x) for x in rp.open()] if rp.exists() else []
    s = summarize(rows, replays)
    p.with_suffix(".summary.json").write_text(json.dumps(s, indent=1))
    for c, m in s["configs"].items():
        lo, hi = m["success_ci"]
        print(f"{c:12s} success {m['success']}/{m['episodes']} ({m['success_rate']:.2f}, CI {lo:.2f}-{hi:.2f}) "
              f"tools {m['tool_calls_mean']:.2f} tokens {m['tokens_mean']:.0f} ${m['cost_usd']:.4f} "
              f"wall {m['wall_s_median']:.1f}s interventions {m['interventions']}/{m['gate_steps']} "
              f"noncompliant {m['noncompliant_fallbacks']} replay {m['replay_verdicts']}")


if __name__ == "__main__":
    main()
