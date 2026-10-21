"""`whetherman calibrate`: measure an agent on the probe pack, fit its adjustment layer and tau.

1. load + validate the probe pack (fails before any agent call)
2. estimate runs / cost / time; confirm unless --yes
3. label_necessity over the pack (same function + config as E2), resumable
4. usable labels -> 60/40 adaptation/threshold split (label-stratified, deterministic)
5. frozen shared backbone logits -> fit vector scaling on adaptation split -> select tau on threshold split
6. write profiles/<agent>.json
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from whetherman.calibrate.adjust import VectorScaling, fit_vector_scaling, select_tau, threshold_stats
from whetherman.calibrate.probe_pack import ProbePack, load_probe_pack
from whetherman.calibrate.splits import adapt_threshold_split
from whetherman.labeling import label_necessity, load_config
from whetherman.labels import IDX
from whetherman.policy.gate import Profile
from whetherman.providers.base import Usage
from whetherman.providers.openai_compat import ChatClient, provider_for_url
from whetherman.tools.registry import REGISTRY

ROOT = Path(__file__).resolve().parents[2]
EST_CALLS_PER_EPISODE = 1.6
EST_PROMPT_TOKENS = 700
EST_COMPLETION_TOKENS = 80
MAX_FALSE_SKIP = 0.05
L2 = 2.0


def slug(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", model).strip("-").lower()


def estimate(pack: ProbePack, client: ChatClient, cfg: dict[str, Any], parallel: int) -> dict[str, Any]:
    episodes = len(pack.tasks) * 2 * cfg["screening_trials"]
    worst = len(pack.tasks) * 2 * (cfg["screening_trials"] + cfg["extra_trials"])
    calls = episodes * EST_CALLS_PER_EPISODE
    per_call = client.pricing.cost(Usage(EST_PROMPT_TOKENS, EST_COMPLETION_TOKENS, 0))
    t0 = time.perf_counter()
    try:
        client.chat([{"role": "user", "content": "Reply with OK."}], max_tokens=4)
        lat = (time.perf_counter() - t0)
    except Exception as e:  # noqa: BLE001 - surface the endpoint problem before spending
        raise SystemExit(f"agent endpoint check failed: {e}") from e
    return {"tasks": len(pack.tasks), "episodes_min": episodes, "episodes_max": worst,
            "est_calls": int(calls), "est_usd": round(calls * per_call, 4),
            "est_usd_max": round(worst * EST_CALLS_PER_EPISODE * per_call, 4),
            "est_minutes": round(calls * max(lat, 0.3) / parallel / 60, 1), "probe_latency_s": round(lat, 2)}


def collect_labels(pack: ProbePack, client_factory, cfg: dict[str, Any], out: Path, parallel: int = 1,
                   log=print) -> list[dict]:
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["task_id"]: json.loads(x) for x in out.open()} if out.exists() else {}
    todo = [t for t in pack.tasks if t.id not in done]

    def one(t):
        c = client_factory()
        try:
            return label_necessity(t, c, cfg, keep_episodes=True).to_dict()
        finally:
            c.close()

    with ThreadPoolExecutor(parallel) as ex, out.open("a") as f:
        for i, lab in enumerate(ex.map(one, todo), 1):
            f.write(json.dumps(lab) + "\n")
            f.flush()
            done[lab["task_id"]] = lab
            if i % 10 == 0 or i == len(todo):
                log(f"  labeled {i}/{len(todo)}")
    return [done[t.id] for t in pack.tasks if t.id in done]


def decision_points(pack: ProbePack, labels: list[dict]) -> list[dict]:
    by = {t.id: t for t in pack.tasks}
    pts = []
    for lab in labels:
        if lab["label"] in ("ACT", "ANSWER"):
            t = by[lab["task_id"]]
            pts.append({"id": t.id, "messages": t.messages, "tools": [REGISTRY[n].schema() for n in t.tools],
                        "label": lab["label"]})
    return pts


def router_logits(router, points: list[dict], batch: int = 16) -> np.ndarray:
    fs = [router.format({"messages": p["messages"], "tools": p["tools"]}) for p in points]
    order = sorted(range(len(fs)), key=lambda i: fs[i]["n_tokens"])
    out = np.zeros((len(fs), 4))
    for s in range(0, len(order), batch):
        idx = order[s: s + batch]
        for i, pr in zip(idx, router.predict_batch([fs[i] for i in idx]), strict=True):
            out[i] = pr["logits"]
    return out


def fit_profile(agent: str, points: list[dict], logits: np.ndarray, backbone: str,
                max_false_skip: float = MAX_FALSE_SKIP, l2: float = L2) -> tuple[Profile, dict]:
    ids = [p["id"] for p in points]
    adapt, thresh = adapt_threshold_split([{"id": i, "label": p["label"]} for i, p in zip(ids, points, strict=True)])
    pos = {i: k for k, i in enumerate(ids)}
    a_idx = [pos[x["id"]] for x in adapt]
    t_idx = [pos[x["id"]] for x in thresh]
    y = np.array([IDX[p["label"]] for p in points])
    counts = np.bincount(y[a_idx], minlength=4).astype(float)
    cw = np.where(counts > 0, counts.sum() / (2 * np.maximum(counts, 1)), 1.0)
    vs = fit_vector_scaling(logits[a_idx], y[a_idx], l2=l2, class_weight=cw) if a_idx else VectorScaling()
    probs_t = vs.probs(logits[t_idx]) if t_idx else np.zeros((0, 4))
    tau, curve = select_tau(probs_t, y[t_idx], max_false_skip)
    at = threshold_stats(probs_t, y[t_idx], tau) if t_idx else {}
    report = {"n_usable": len(points), "n_adapt": len(a_idx), "n_threshold": len(t_idx),
              "adapt_ids": [ids[i] for i in a_idx], "threshold_ids": [ids[i] for i in t_idx],
              "tau_rule": f"smallest tau in 0.50..0.99 with threshold-split false-skip rate <= {max_false_skip}; "
                          "1.0 (never intervene) if none",
              "threshold_stats_at_tau": at, "threshold_curve": curve, "l2": l2}
    prof = Profile(agent=agent, tau=tau, adjust=vs, backbone=backbone, meta={})
    return prof, report


def calibrate(endpoint: str, model: str, probe_pack: str | None = None, name: str | None = None,
              backbone: str = "checkpoints/shared-v1", yes: bool = False, parallel: int = 1,
              labels_dir: str | None = None, out_dir: str = "profiles", device: str = "auto",
              api_key: str | None = None, client_factory=None, router=None) -> Path:
    t_start = time.time()
    pack = load_probe_pack(probe_pack)
    cfg = load_config()
    agent = name or slug(model)
    spec = provider_for_url(endpoint)

    def factory() -> ChatClient:
        if client_factory is not None:
            return client_factory()
        return ChatClient(model, provider=spec, base_url=endpoint, api_key=api_key)

    est = estimate(pack, factory(), cfg, parallel)
    print(f"probe pack {pack.manifest['name']} v{pack.manifest['version']} ({pack.sha256[:12]}): "
          f"{est['tasks']} tasks")
    print(f"estimated {est['episodes_min']}-{est['episodes_max']} episodes, ~{est['est_calls']} agent calls, "
          f"~${est['est_usd']} (worst ${est['est_usd_max']}), ~{est['est_minutes']} min")
    if not yes and input("proceed? [y/N] ").strip().lower() not in ("y", "yes"):
        raise SystemExit("aborted")
    lab_path = Path(labels_dir or ROOT / "profiles" / "labels") / f"{agent}.probe.jsonl"
    labels = collect_labels(pack, factory, cfg, lab_path, parallel)
    spend = float(sum(x["cost_usd"] for x in labels))
    pts = decision_points(pack, labels)

    if router is None:
        from whetherman.adapters.functiongemma import FunctionGemmaRouter

        ck = ROOT / backbone if not Path(backbone).is_absolute() else Path(backbone)
        router = FunctionGemmaRouter(checkpoint=ck, device=device)
    logits = router_logits(router, pts)
    prof, report = fit_profile(agent, pts, logits, backbone)
    n_amb = sum(x["label"] == "ambiguous" for x in labels)
    prof.meta = {
        "endpoint_provider": spec.name, "model": model, "probe_pack": pack.manifest["name"],
        "probe_pack_version": pack.manifest["version"], "probe_pack_sha256": pack.sha256,
        "label_counts": {k: sum(x["label"] == k for x in labels) for k in ("ACT", "ANSWER", "ambiguous")},
        "usable_labels": len(pts), "ambiguous": n_amb, "calibration_usd": round(spend, 6),
        "calibration_seconds": round(time.time() - t_start, 1),
        "agent_episode_seconds": round(sum(x["wall_ms"] for x in labels) / 1000, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "labeling_config": cfg,
        "fit": {k: v for k, v in report.items() if k not in ("threshold_curve",)},
        "threshold_curve": report["threshold_curve"],
    }
    out = Path(out_dir) / f"{agent}.json"
    prof.save(out)
    print(f"usable labels {len(pts)} (ambiguous {n_amb}); adapt {report['n_adapt']} / threshold "
          f"{report['n_threshold']}; tau={prof.tau}; spent ${spend:.4f} in {prof.meta['calibration_seconds']}s")
    print(f"wrote {out}")
    return out
