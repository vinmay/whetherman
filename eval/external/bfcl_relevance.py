"""BFCL relevance / irrelevance subsets as an out-of-distribution "should any function be called?" check.
Self-run on frozen routers, not a leaderboard result. Held-out: run once.

uv run python -m eval.external.bfcl_relevance --routers shared untuned qwen_prompted

Mapping: relevance items pass if the router decides ACT; irrelevance items pass if it decides anything else.
Decisions use the default policy (ACT unless a non-ACT class is the argmax with probability > 0.5), the
same as the agent-agnostic ladder rows. When2Call was built partly from BFCL Live, so items whose question
text also appears in Whetherman's training data are counted and reported both ways.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from eval import ladder
from whetherman.calibrate.adjust import decide, softmax
from whetherman.labeling import wilson

ROOT = Path(__file__).resolve().parents[2]
REPO = "https://github.com/ShishirPatil/gorilla.git"
COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
DATA_DIR = "berkeley-function-call-leaderboard/bfcl_eval/data"
SRC = ROOT / "data" / "raw" / "external" / "gorilla"
OUT = ROOT / "eval" / "external" / "bfcl"
SUBSETS = {"irrelevance": False, "live_irrelevance": False, "live_relevance": True}  # name -> should ACT


def fetch() -> None:
    if not (SRC / ".git").exists():
        SRC.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--sparse", REPO, str(SRC)], check=True)
        subprocess.run(["git", "-C", str(SRC), "sparse-checkout", "set", DATA_DIR], check=True)
    subprocess.run(["git", "-C", str(SRC), "checkout", "-q", COMMIT], check=True)


def norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def training_texts() -> set[str]:
    """User turns of every training-side item (E1 train/val and the full When2Call training files)."""
    out = set()
    fix = ROOT / "data" / "fixtures" / "e1"
    for f in ("train.jsonl", "val.jsonl"):
        for x in (fix / f).open():
            out |= {norm(m["content"]) for m in json.loads(x)["messages"] if m["role"] == "user"}
    raw = ROOT / "data" / "raw" / "when2call" / "train"
    for f in raw.glob("*.jsonl"):
        for x in f.open():
            out |= {norm(m["content"]) for m in json.loads(x)["messages"]
                    if m["role"] == "user" and isinstance(m.get("content"), str)}
    return out


def load_subset(name: str) -> list[dict]:
    sys.path.insert(0, str(ROOT / "data"))
    from prepare import convert_schema

    path = SRC / DATA_DIR / f"BFCL_v4_{name}.json"
    pts = []
    for x in path.open():
        r = json.loads(x)
        msgs = [m for m in r["question"][0] if m["role"] in ("system", "user", "assistant")]
        pts.append({"id": f"bfcl-{r['id']}", "messages": msgs, "tools": [convert_schema(f) for f in r["function"]]})
    return pts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routers", nargs="+", default=["shared", "untuned", "qwen_prompted"])
    ap.add_argument("--shared-checkpoint", default="shared-v3")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    fetch()
    ladder.SHARED = a.shared_checkpoint
    train = training_texts()
    data = {name: load_subset(name) for name in SUBSETS}
    overlap = {name: {p["id"] for p in pts if any(norm(m["content"]) in train for m in p["messages"]
                                                  if m["role"] == "user")} for name, pts in data.items()}
    summary: dict = {"repo": REPO, "commit": COMMIT, "bfcl_version": "v4 data files", "policy": "default (tau 0.5)",
                     "shared_checkpoint": a.shared_checkpoint,
                     "n": {k: len(v) for k, v in data.items()},
                     "overlap_with_training": {k: len(v) for k, v in overlap.items()}, "routers": {}}
    OUT.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    for rn in a.routers:
        res = {}
        for name, pts in data.items():
            raw = ladder.raw_outputs(rn, f"bfcl_{name}", pts, a.device, cache)
            want_act = SUBSETS[name]
            rows = [{"id": p["id"], "decision": decide(softmax(np.array(raw[p["id"]])), 0.5),
                     "overlap": p["id"] in overlap[name]} for p in pts]
            for r in rows:
                r["pass"] = (r["decision"] == "ACT") == want_act
            (OUT / f"{rn}_{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            entry = {}
            for scope, rs in (("all", rows), ("no_overlap", [r for r in rows if not r["overlap"]])):
                k, n = sum(r["pass"] for r in rs), len(rs)
                entry[scope] = {"pass": k, "n": n, "rate": k / n if n else None, "ci": list(wilson(k, n))}
            res[name] = entry
            m = entry["all"]
            print(f"{rn:14s} {name:17s} {m['pass']}/{m['n']} ({m['rate']:.3f}) "
                  f"no-overlap {entry['no_overlap']['pass']}/{entry['no_overlap']['n']}")
        summary["routers"][rn] = res
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
