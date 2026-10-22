"""Fixed-history routing benchmark and the offline comparison ladder.

uv run python -m eval.ladder --phase dev      (validation / probe / E2-train disagreement only)
uv run python -m eval.ladder --phase final    (held-out E1/E2/E3 test; run once after freezing)

Raw router outputs are cached in eval/results/routing/<router>/<set>.jsonl; metrics go to
eval/results/ladder_<phase>.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval import metrics, points
from whetherman.calibrate.adjust import FeatureHead, VectorScaling, adjust_from_dict, decide, select_tau, softmax
from whetherman.labels import IDX
from whetherman.policy.gate import Profile

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "eval" / "results" / "routing"
AGENTS = ["L", "S"]
SHARED = "shared-v3"  # checkpoint behind the "shared" router; set by --shared-checkpoint


# ---------------------------------------------------------------- raw router outputs (cached)

def get_router(name: str, device: str):
    if name == "untuned":
        from whetherman.adapters.functiongemma import FunctionGemmaRouter

        r = FunctionGemmaRouter(device=device)

        class U:
            placement = r.placement
            format = staticmethod(r.format)

            @staticmethod
            def predict_batch(fs):
                return [r.predict_untuned(f) for f in fs]

        return U()
    if name.startswith("shared") or (ROOT / "checkpoints" / name).is_dir():
        from whetherman.adapters.functiongemma import FunctionGemmaRouter

        ck = f"checkpoints/{SHARED}" if name == "shared" else f"checkpoints/{name}"
        return FunctionGemmaRouter(checkpoint=ROOT / ck, device=device)
    if name == "rules":
        from whetherman.adapters.baselines import RulesRouter

        r = RulesRouter()
        r.predict_batch = lambda fs: [r.predict(f) for f in fs]  # type: ignore[attr-defined]
        return r
    if name == "qwen_prompted":
        from whetherman.adapters.baselines import PromptedRouter
        from whetherman.providers.openai_compat import ChatClient

        r = PromptedRouter(ChatClient("qwen3.5:4b", "ollama"))
        r.predict_batch = lambda fs: [r.predict(f) for f in fs]  # type: ignore[attr-defined]
        return r
    raise ValueError(name)


class RawOut(dict):
    """id -> backbone logits; .hidden holds id -> last hidden state for FunctionGemma routers."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.hidden: dict[str, list] = {}


def raw_outputs(router_name: str, set_name: str, pts: list[dict], device: str, cache: dict) -> RawOut:
    path = RAW / (SHARED if router_name == "shared" else router_name) / f"{set_name}.jsonl"
    need_hidden = router_name.startswith("shared")
    have = RawOut()
    if path.exists():
        for r in map(json.loads, path.open()):  # later rows win (a row may be re-run to add its hidden state)
            have[r["id"]] = r["logits"]
            if "hidden" in r:
                have.hidden[r["id"]] = r["hidden"]
    todo = [p for p in pts if p["id"] not in have or (need_hidden and p["id"] not in have.hidden)]
    if todo:
        if router_name not in cache:
            cache[router_name] = get_router(router_name, device)
        r = cache[router_name]
        path.parent.mkdir(parents=True, exist_ok=True)
        fs = [r.format({"messages": p["messages"], "tools": p["tools"]}) for p in todo]
        order = sorted(range(len(todo)), key=lambda i: fs[i].get("n_tokens", 0))
        with path.open("a") as f:
            for s in range(0, len(order), 16):
                idx = order[s: s + 16]
                for i, pr in zip(idx, r.predict_batch([fs[i] for i in idx]), strict=True):
                    rec = {"id": todo[i]["id"], "logits": pr["logits"], "latency_ms": pr["latency_ms"],
                           "truncated": fs[i].get("truncated", False)}
                    if pr.get("hidden") is not None:
                        rec["hidden"] = np.round(np.asarray(pr["hidden"], dtype=float), 5).tolist()
                        have.hidden[rec["id"]] = rec["hidden"]
                    f.write(json.dumps(rec) + "\n")
                    have[rec["id"]] = rec["logits"]
            print(f"  {router_name}/{set_name}: +{len(todo)}", flush=True)
    return have


# ---------------------------------------------------------------- policies

def policy_decisions(logits: np.ndarray, adjust, tau: float, hidden: np.ndarray | None = None):
    probs = adjust.probs(logits, hidden)
    return [decide(p, tau) for p in probs], probs


def threshold_only_tau(agent: str, shared_logits: dict, prof: Profile) -> float:
    """Identity adjustment; tau selected on the same threshold split the adapted profile used."""
    ids = prof.meta["fit"]["threshold_ids"]
    labs = {p["id"]: p["label"] for p in points.probe(agent)}
    ids = [i for i in ids if i in shared_logits and i in labs]
    tau, _ = select_tau(softmax(np.array([shared_logits[i] for i in ids])), np.array([IDX[labs[i]] for i in ids]))
    return tau


def evaluate(pts: list[dict], raw: RawOut, adjust, tau: float) -> dict:
    pts = [p for p in pts if p["id"] in raw]
    lg = np.array([raw[p["id"]] for p in pts]) if pts else np.zeros((0, 4))
    hid = None
    if isinstance(adjust, FeatureHead):
        pts = [p for p in pts if p["id"] in raw.hidden]
        lg = np.array([raw[p["id"]] for p in pts]) if pts else np.zeros((0, 4))
        hid = np.array([raw.hidden[p["id"]] for p in pts]) if pts else np.zeros((0, len(adjust.w)))
    dec, probs = policy_decisions(lg, adjust, tau, hid)
    y = [p["label"] for p in pts]
    out = {"all": metrics.summarize(y, dec, probs)}
    groups = {"single_step": lambda p: not p.get("multi_step"), "multi_step": lambda p: p.get("multi_step"),
              "seen_schema": lambda p: not p.get("unseen_schema"), "unseen_schema": lambda p: p.get("unseen_schema"),
              "disagreement": lambda p: p.get("disagree")}
    for g, fn in groups.items():
        idx = [i for i, p in enumerate(pts) if fn(p)]
        if idx and len(idx) < len(pts):
            out[g] = metrics.summarize([y[i] for i in idx], [dec[i] for i in idx], probs[idx])
    for k in out.values():  # keep the JSON readable
        k.pop("reliability", None)
    out["decisions"] = {p["id"]: d for p, d in zip(pts, dec, strict=True)}
    return out


def disagreement_ids(split: str) -> set[str]:
    labs = {a: {p["id"]: p["label"] for p in points.e2(split, a)} for a in AGENTS}
    common = set(labs["L"]) & set(labs["S"])
    return {i for i in common if labs["L"][i] != labs["S"][i]}


def trained_e2_ids(checkpoint: str) -> set[str]:
    """E2-train items the shared backbone trained on; excluded from dev evaluation."""
    meta = ROOT / checkpoint / "train_meta.json"
    if not meta.exists():
        return set()
    m = json.loads(meta.read_text())
    if m.get("agents_for_agreement") and "e2_train_ids_trained" not in m:
        raise SystemExit(f"{meta} predates train-id recording; cannot exclude trained E2 items")
    return set(m.get("e2_train_ids_trained", []))


def main() -> None:
    global SHARED
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "final"], default="dev")
    ap.add_argument("--routers", nargs="+", default=["untuned", "shared", "rules", "qwen_prompted"])
    ap.add_argument("--agents", nargs="+", default=AGENTS)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--shared-checkpoint", default=SHARED, help="checkpoint name under checkpoints/")
    ap.add_argument("--profiles", default="profiles")
    ap.add_argument("--identity-checkpoint", default=None,
                    help="P1 row: identity-conditioned checkpoint, agent ID in the input, default policy")
    a = ap.parse_args()
    SHARED = a.shared_checkpoint
    cache: dict = {}
    profiles = {ag: Profile.load(ROOT / a.profiles / f"{ag}.json") for ag in a.agents
                if (ROOT / a.profiles / f"{ag}.json").exists()}
    for ag, pr in profiles.items():
        if Path(pr.backbone).name != SHARED:
            raise SystemExit(f"profile {ag} was fit on {pr.backbone}, not {SHARED}; recalibrate first")
    if a.phase == "dev":
        sets_fixed = {"e1_val": points.e1("val")}
        dis = disagreement_ids("e2_train") if all((ROOT / "data/fixtures/labels" / x / "e2_train.jsonl").exists()
                                                  for x in AGENTS) else set()
        trained = trained_e2_ids(f"checkpoints/{SHARED}")
        sets_agent = {ag: {"e2_train_val": [dict(p, disagree=p["id"] in dis) for p in points.e2("e2_train", ag)
                                            if p["id"] not in trained]}
                      for ag in a.agents}
    else:
        sets_fixed = {"e1_test": points.e1("test")}
        dis = disagreement_ids("e2_test")
        sets_agent = {ag: {"e2_test": [dict(p, disagree=p["id"] in dis) for p in points.e2("e2_test", ag)],
                           "multihop_test": points.e2("multihop_test", ag), "e3": points.e3(ag)}
                      for ag in a.agents}
    results: dict = {"phase": a.phase, "fixed": {}, "per_agent": {}}
    ident = VectorScaling()
    # agent-agnostic sets: routers with default policy only
    idc = a.identity_checkpoint
    if idc:
        from whetherman.adapters.functiongemma import with_identity

        def as_agent(pts: list[dict], agent: str) -> list[dict]:
            return [dict(p, messages=with_identity(p["messages"], agent)) for p in pts]
    for sname, pts in sets_fixed.items():
        for rn in a.routers:
            raw = raw_outputs(rn, sname, pts, a.device, cache)
            results["fixed"][f"{rn}/{sname}"] = evaluate(pts, raw, ident, 0.5)
        if idc:
            raw = raw_outputs(idc, f"{sname}_as_any", as_agent(pts, "any"), a.device, cache)
            results["fixed"][f"identity/{sname}"] = evaluate(pts, raw, ident, 0.5)
    for ag in a.agents:
        prof = profiles.get(ag)
        for sname, pts in sets_agent[ag].items():
            pts_all = pts
            for rn in a.routers:
                # cache per point id across agents: same input -> same raw output
                raw = raw_outputs(rn, sname, pts_all, a.device, cache)
                results["per_agent"].setdefault(ag, {})[f"{rn}/{sname}"] = evaluate(pts_all, raw, ident, 0.5)
            if idc:
                raw = raw_outputs(idc, f"{sname}_as_{ag}", as_agent(pts_all, ag), a.device, cache)
                results["per_agent"][ag][f"identity/{sname}"] = evaluate(pts_all, raw, ident, 0.5)
            if prof and "shared" in a.routers:
                raw = raw_outputs("shared", sname, pts_all, a.device, cache)
                probe_raw = raw_outputs("shared", f"probe_{ag}", points.probe(ag), a.device, cache)
                t_only = threshold_only_tau(ag, probe_raw, prof)
                results["per_agent"][ag][f"threshold_only/{sname}"] = {**evaluate(pts_all, raw, ident, t_only),
                                                                        "tau": t_only}
                results["per_agent"][ag][f"adapted/{sname}"] = {**evaluate(pts_all, raw, prof.adjust, prof.tau),
                                                                 "tau": prof.tau}
                abl = prof.meta.get("fit", {}).get("ablation_vector_scaling")
                if abl and isinstance(prof.adjust, FeatureHead):  # ablation: vector scaling on the same splits
                    results["per_agent"][ag][f"adapted_vecscale/{sname}"] = {
                        **evaluate(pts_all, raw, adjust_from_dict(abl["adjust"]), abl["tau"]), "tau": abl["tau"]}
                other = [o for o in profiles if o != ag]
                for o in other:  # P1 profile swap: this agent's points, the other agent's profile
                    po = profiles[o]
                    results["per_agent"][ag][f"swap_profile_{o}/{sname}"] = {
                        **evaluate(pts_all, raw, po.adjust, po.tau), "tau": po.tau}
    results["shared_checkpoint"] = SHARED
    results["identity_checkpoint"] = idc
    out = ROOT / "eval" / "results" / f"ladder_{a.phase}_{SHARED}.json"
    out.write_text(json.dumps(results, indent=1))
    # compact console table
    for scope, d in [("fixed", results["fixed"])] + [(f"agent {k}", v) for k, v in results["per_agent"].items()]:
        print(f"== {scope}")
        for name, r in d.items():
            m = r["all"]
            fs = m["false_skip"]
            cs = m["correct_suppression"]
            tau = f" tau={r['tau']}" if "tau" in r else ""
            print(f"  {name:38s} n={m['n']:4d} acc={m['accuracy']:.3f} F1={m['macro_f1']:.3f} "
                  f"falseskip={fs['k']}/{fs['n']} supp={cs['k']}/{cs['n']}{tau}")
    print("wrote", out)


if __name__ == "__main__":
    main()
