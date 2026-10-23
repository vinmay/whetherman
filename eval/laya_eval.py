"""Laya vs FunctionGemma, offline only (P2 comparison; Laya is never the shipped default).

uv run python -m eval.laya_eval --phase dev     # fits profiles/laya/{L,S}.json, dev rows
uv run python -m eval.laya_eval --phase final   # held-out rows; run once, after the Laya freeze record

Profiles use the same per-agent procedure as the shipped ones: the same probe points (exactly the ids the
shipped profiles were fitted on), the same adaptation/threshold split, feature head and tau rule.
Rows mirror eval/ladder.py: default policy (tau 0.5) and adapted; metrics include false skips with CIs,
macro-F1, ECE and the single/multi-step and seen/unseen-schema breakdowns. Truncation (inputs over Laya's
512-token limit) is reported per set from the routing cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval import ladder, points
from whetherman.calibrate.adjust import VectorScaling
from whetherman.calibrate.run import fit_profile
from whetherman.policy.gate import Profile

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "profiles" / "laya"
AGENTS = ["L", "S"]


def truncation(set_name: str, ids: list[str]) -> dict:
    rows = {}
    path = ladder.RAW / "laya" / f"{set_name}.jsonl"
    for r in map(json.loads, path.open()):
        rows[r["id"]] = r.get("truncated", False)
    k = sum(bool(rows.get(i)) for i in ids)
    return {"over_limit": k, "n": len(ids), "rate": k / len(ids) if ids else None}


def fit_profiles(device: str, cache: dict) -> dict[str, Profile]:
    out = {}
    PROFILES.mkdir(parents=True, exist_ok=True)
    for ag in AGENTS:
        shipped = Profile.load(ROOT / "profiles" / f"{ag}.json").meta["fit"]
        ids = set(shipped["adapt_ids"]) | set(shipped["threshold_ids"])
        pts = [p for p in points.probe(ag) if p["id"] in ids]
        raw = ladder.raw_outputs("laya", f"probe_{ag}", pts, device, cache)
        logits = np.array([raw[p["id"]] for p in pts])
        hidden = np.array([raw.hidden[p["id"]] for p in pts])
        prof, report = fit_profile(ag, pts, logits, "checkpoints/laya-lora", hidden=hidden)
        same = set(report["adapt_ids"]) == set(shipped["adapt_ids"]) and \
            set(report["threshold_ids"]) == set(shipped["threshold_ids"])
        if not same:
            raise SystemExit(f"{ag}: Laya profile split differs from the shipped profile's")
        prof.meta = {"fit": report, "router": "laya", "probe_ids_from": f"profiles/{ag}.json"}
        prof.save(PROFILES / f"{ag}.json")
        print(f"[{ag}] Laya profile: {len(pts)} probe points, tau {prof.tau}", flush=True)
        out[ag] = prof
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "final"], default="dev")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    cache: dict = {}
    ident = VectorScaling()
    if a.phase == "dev":
        profiles = fit_profiles(a.device, cache)
        fixed = {"e1_val": points.e1("val")}
        trained = ladder.trained_e2_ids("checkpoints/shared-v3")
        per_agent = {ag: {"e2_train_val": [p for p in points.e2("e2_train", ag) if p["id"] not in trained]}
                     for ag in AGENTS}
    else:
        profiles = {ag: Profile.load(PROFILES / f"{ag}.json") for ag in AGENTS}
        fixed = {"e1_test": points.e1("test")}
        dis = ladder.disagreement_ids("e2_test")
        per_agent = {ag: {"e2_test": [dict(p, disagree=p["id"] in dis) for p in points.e2("e2_test", ag)],
                          "multihop_test": points.e2("multihop_test", ag), "e3": points.e3(ag)}
                     for ag in AGENTS}
    res: dict = {"phase": a.phase, "router": "laya option (b): LoRA-fine-tuned encoder + 4-class head", "fixed": {}, "per_agent": {},
                 "truncation": {}, "profiles": {ag: p.tau for ag, p in profiles.items()}}
    for sname, pts in fixed.items():
        raw = ladder.raw_outputs("laya", sname, pts, a.device, cache)
        res["fixed"][f"laya/{sname}"] = ladder.evaluate(pts, raw, ident, 0.5)
        res["truncation"][sname] = truncation(sname, [p["id"] for p in pts])
    for ag, sets in per_agent.items():
        for sname, pts in sets.items():
            raw = ladder.raw_outputs("laya", sname, pts, a.device, cache)
            res["per_agent"].setdefault(ag, {})[f"laya/{sname}"] = ladder.evaluate(pts, raw, ident, 0.5)
            res["per_agent"][ag][f"laya_adapted/{sname}"] = {
                **ladder.evaluate(pts, raw, profiles[ag].adjust, profiles[ag].tau), "tau": profiles[ag].tau}
            res["truncation"].setdefault(sname, truncation(sname, [p["id"] for p in pts]))
    out = ROOT / "eval" / "results" / f"laya_{a.phase}.json"
    out.write_text(json.dumps(res, indent=1))
    for scope, d in [("fixed", res["fixed"])] + [(f"agent {k}", v) for k, v in res["per_agent"].items()]:
        print(f"== {scope}")
        for name, r in d.items():
            m = r["all"]
            print(f"  {name:28s} n={m['n']:4d} F1={m['macro_f1']:.3f} ECE={m['ece']:.3f} "
                  f"falseskip={m['false_skip']['k']}/{m['false_skip']['n']} "
                  f"supp={m['correct_suppression']['k']}/{m['correct_suppression']['n']}"
                  + (f" tau={r['tau']}" if "tau" in r else ""))
    print("truncation", json.dumps(res["truncation"]))
    print("wrote", out)


if __name__ == "__main__":
    main()
