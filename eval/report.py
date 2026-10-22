"""Build the results tables (markdown) from result files, so no number is copied by hand.

uv run python -m eval.report --phase final   -> eval/results/REPORT_final.md
uv run python -m eval.report --phase dev --ab-tag _v3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "eval" / "results"
CONFIG_NAMES = {"none": "No gate", "adapted": "**Whetherman (adapted)**", "shared": "Shared backbone, no profile",
                "rules": "Rules-only control", "qwen_prompted": "Prompted Qwen3.5-4B router"}


def load(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def ci(m: dict) -> str:
    lo, hi = m["success_ci"] if "success_ci" in m else m["ci95"]
    return f"[{lo:.2f}, {hi:.2f}]"


def frac(m: dict) -> str:
    return f"{m['k']}/{m['n']} {ci(m)}" if m["n"] else "–"


def cfg_name(c: str) -> str:
    return CONFIG_NAMES.get(c, f"Profile swap ({c.split(':')[1]}'s profile)" if c.startswith("profile:") else c)


def ab_section(agent: str, s: dict, profile: dict | None) -> list[str]:
    out = [f"### Agent {agent}: end-to-end A/B ({s['configs']['none']['episodes']} episodes per configuration)", "",
           "Headline excludes guessable two-choice tasks (DATA.md). Tokens, cost and wall time are over all tasks.", "",
           "| Configuration | Success (headline) | 95% CI | Tool calls / episode | Tokens / episode | Cost | "
           "Median wall | Gate skips | Non-compliant | Replay: restored with ACT |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    order = ["none", "adapted"] + [c for c in s["configs"] if c not in ("none", "adapted")]
    for c in order:
        if c not in s["configs"]:
            continue
        m, h = s["configs"][c], s["headline"][c]
        restored = sum(v for k, v in m["replay_verdicts"].items() if k.startswith("restored"))
        out.append(f"| {cfg_name(c)} | {h['success']}/{h['episodes']} | {ci(h)} | {h['tool_calls_mean']:.2f} | "
                   f"{m['tokens_mean']:.0f} | ${m['cost_usd']:.4f} | {m['wall_s_median']:.1f}s | "
                   f"{m['interventions']}/{m['gate_steps']} | {m['noncompliant_fallbacks']} | {restored} |")
    g = s["guessable"]
    if g.get("none", {}).get("episodes"):
        out += ["", "Guessable group (reported separately): " + ", ".join(
            f"{cfg_name(c)} {g[c]['success']}/{g[c]['episodes']}" for c in order if c in g and g[c]["episodes"])]
    n, a = s["configs"]["none"], s["configs"].get("adapted")
    if a:
        hn, ha = s["headline"]["none"], s["headline"]["adapted"]
        dc = (ha["tool_calls_mean"] - hn["tool_calls_mean"]) / hn["tool_calls_mean"] if hn["tool_calls_mean"] else 0
        dt = (a["tokens_mean"] - n["tokens_mean"]) / n["tokens_mean"] if n["tokens_mean"] else 0
        out += ["", f"Whetherman vs no gate: tool calls {dc:+.0%} (headline), tokens {dt:+.0%}, "
                    f"success {ha['success'] - hn['success']:+d} of {hn['episodes']}."]
        if profile and n["cost_usd"] == 0:
            mins = profile["meta"]["calibration_seconds"] / 60
            out.append(f"Local agent (no dollar cost). Calibration took {mins:.0f} min of wall time; "
                       "savings are calls, tokens and time.")
        elif profile and n["cost_usd"] > a["cost_usd"]:
            per_ep = (n["cost_usd"] - a["cost_usd"]) / n["episodes"]
            cal = profile["meta"]["calibration_usd"]
            out.append(f"Calibration cost ${cal:.2f}; saved ${per_ep:.5f} per episode on this suite; "
                       f"break-even after ~{cal / per_ep:,.0f} episodes.")
        elif profile:
            out.append(f"Calibration cost ${profile['meta']['calibration_usd']:.2f}; no dollar saving on this suite, "
                       "so it is not amortized.")
    return out + [""]


def ladder_section(lad: dict) -> list[str]:
    out = ["### Offline routing (fixed histories)", "",
           "False skip = the gate withholds tools from a decision point that needed them (that agent's label). "
           "Correct skip = non-ACT on a point the agent answers without tools.", ""]
    for scope, d in [("Agent-agnostic", lad["fixed"])] + [(f"Agent {k}", v) for k, v in lad["per_agent"].items()]:
        out += [f"**{scope}**", "", "| Router / set | n | Macro-F1 | False skips | Correct skips | τ |",
                "|---|---|---|---|---|---|"]
        for name, r in d.items():
            m = r["all"]
            out.append(f"| {name} | {m['n']} | {m['macro_f1']:.3f} | {frac(m['false_skip'])} | "
                       f"{frac(m['correct_suppression'])} | {r.get('tau', '')} |")
        out.append("")
    return out


def external_section() -> list[str]:
    out = []
    w = load(ROOT / "eval" / "external" / "when2call" / "summary.json")
    if w:
        out += [f"### When2Call official MCQ test (n={w['n']}, NVIDIA/When2Call @ {w['commit'][:7]}, self-run)", "",
                "Classifier class mapped to the MCQ option; official metric code. acc_norm not applicable.", "",
                "| Router | Macro-F1 | Accuracy | Tool hallucination rate |", "|---|---|---|---|"]
        for rn, m in w["routers"].items():
            out.append(f"| {rn} | {m['macro_f1']:.3f} | {m['acc']:.3f} | {m['hallucination_rate']:.3f} |")
        out.append("")
    b = load(ROOT / "eval" / "external" / "bfcl" / "summary.json")
    if b:
        out += [f"### BFCL relevance / irrelevance subsets (gorilla @ {b['commit'][:7]}, self-run, not a leaderboard "
                "result)", "", f"Overlap with training text: {b['overlap_with_training']}.", "",
                "| Router | " + " | ".join(b["n"]) + " |", "|---" * (len(b["n"]) + 1) + "|"]
        for rn, r in b["routers"].items():
            out.append(f"| {rn} | " + " | ".join(f"{r[s]['all']['pass']}/{r[s]['all']['n']} "
                                                f"({r[s]['all']['rate']:.2f})" for s in b["n"]) + " |")
        out.append("")
    return out


def latency_section(tags: list[str]) -> list[str]:
    out = []
    for t in tags:
        lat = load(RES / f"latency_{t}.json")
        if not lat:
            continue
        out += [f"### Router latency ({lat['device']}, {Path(lat['checkpoint']).name})", "",
                "| Scenario | Input tokens | p50 | p95 |", "|---|---|---|---|"]
        for name, m in lat["scenarios"].items():
            out.append(f"| {name} | {m['input_tokens']} | {m['p50_ms']:.0f} ms | {m['p95_ms']:.0f} ms |")
        out.append("")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "final"], default="final")
    ap.add_argument("--ab-tag", default="")
    ap.add_argument("--agents", nargs="+", default=["L", "S", "U"])
    ap.add_argument("--latency", nargs="*", default=["v3_cuda_cuda", "v3_cpu_cpu"])
    a = ap.parse_args()
    suite = "ab_final" if a.phase == "final" else "ab_dev"
    lines = [f"# Results ({a.phase})", "", "Generated by `eval/report.py` from the files in eval/results/.", ""]
    for ag in a.agents:
        s = load(RES / "ab" / f"{suite}_{ag}{a.ab_tag}.summary.json")
        if s:
            lines += ab_section(ag, s, load(ROOT / "profiles" / f"{ag}.json"))
    lad = load(RES / f"ladder_{a.phase}_shared-v3.json")
    if lad:
        lines += ladder_section(lad)
    if a.phase == "final":
        lines += external_section()
    lines += latency_section(a.latency)
    spend = load(RES / "spend.json")
    if spend:
        lines += [f"Hosted spend to date: ${spend['totals_usd'].get('hosted', 0):.2f} "
                  f"(cap ${spend['caps_usd']['hosted']:.0f}).", ""]
    out = RES / f"REPORT_{a.phase}.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
