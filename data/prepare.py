"""Label extraction and conversion for E1 (When2Call) and the When2Tool task pools.

Usage:  uv run python data/prepare.py [--download]

Inputs (gitignored, data/raw/):
  when2call/  nvidia/When2Call (CC-BY-4.0)
  when2tool/  cesun/When2Tool (MIT) parquet files
Outputs (committed, data/fixtures/):
  e1/{train,val,test}.jsonl     decision points with labels ACT/ASK/DECLINE
  w2t/{e2_train,probe,e2_test}.jsonl   executable tasks (labels come from label_necessity)
  manifest.json                 counts, seeds and sha256 of every output
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
FIX = ROOT / "fixtures"
SEED = 20261021

W2C_FILES = ["test/when2call_test_mcq.jsonl", "train/when2call_train_sft.jsonl",
             "train/when2call_train_pref.jsonl"]
W2T_FILES = [f"{c}/{s}-00000-of-00001.parquet" for c in ("single_hop", "multi_hop")
             for s in ("train", "test")]

# ---------------------------------------------------------------- schema conversion

_TYPE = {"str": "string", "string": "string", "int": "integer", "integer": "integer",
         "float": "number", "number": "number", "bool": "boolean", "boolean": "boolean",
         "list": "array", "array": "array", "dict": "object", "object": "object",
         "tuple": "array", "any": "string"}


def _conv_type(t: Any) -> str:
    base = str(t).split(",")[0].strip().lower()
    base = re.sub(r"\[.*", "", base)  # List[str] -> list
    return _TYPE.get(base, "string")


def convert_schema(raw: str | dict) -> dict[str, Any]:
    """When2Call/BFCL-style tool JSON -> OpenAI Chat Completions tool schema."""
    d = json.loads(raw) if isinstance(raw, str) else raw
    params = d.get("parameters") or {}
    props = {}
    for k, v in (params.get("properties") or {}).items():
        p = {"type": _conv_type(v.get("type", "string"))}
        if v.get("description"):
            p["description"] = v["description"]
        if p["type"] == "array":
            p["items"] = {}
        props[k] = p
    req = params.get("required", d.get("required", []))
    return {"type": "function", "function": {
        "name": d["name"], "description": d.get("description", ""),
        "parameters": {"type": "object", "properties": props,
                       "required": [r for r in req if r in props]}}}


# ---------------------------------------------------------------- When2Call (E1)

_DECLINE = re.compile(r"\b(apolog|sorry|unable|can't|cannot|not able|don't have (access|the ability)|"
                      r"beyond my|outside (of )?my)", re.I)


def classify_w2c_response(content: str) -> str:
    """Label a When2Call reference response. Tool calls are explicit; ASK vs DECLINE is a
    documented heuristic (the SFT/pref files carry no explicit label)."""
    c = content or ""
    if "<TOOLCALL>" in c:
        return "ACT"
    if _DECLINE.search(c[:160]):
        return "DECLINE"
    return "ASK"


def w2c_point(i: str, tools: list, messages: list, label: str, source: str, **meta) -> dict:
    return {"id": i, "source": source, "messages": messages,
            "tools": [convert_schema(t) for t in tools], "label": label, "meta": meta}


def build_e1(rng: random.Random, n_train: int, n_val: int, n_test: int) -> dict[str, list[dict]]:
    base = RAW / "when2call"
    pool: list[dict] = []
    for j, line in enumerate(open(base / "train/when2call_train_pref.jsonl")):
        r = json.loads(line)
        lab = classify_w2c_response(r["chosen_response"]["content"])
        pool.append(w2c_point(f"w2c-pref-{j}", r["tools"], r["messages"], lab, "when2call_pref"))
    for j, line in enumerate(open(base / "train/when2call_train_sft.jsonl")):
        r = json.loads(line)
        lab = classify_w2c_response(r["messages"][-1]["content"])
        pool.append(w2c_point(f"w2c-sft-{j}", r["tools"], r["messages"][:-1], lab, "when2call_sft"))
    pool = [p for p in pool if p["tools"]]  # decision points require available tools
    by = defaultdict(list)
    for p in pool:
        by[p["label"]].append(p)
    for v in by.values():
        rng.shuffle(v)
    per = {"train": n_train // 3, "val": n_val // 3}
    out = {"train": [], "val": []}
    for lab in ("ACT", "ASK", "DECLINE"):
        items = by[lab]
        out["val"] += items[: per["val"]]
        out["train"] += items[per["val"]: per["val"] + per["train"]]
    # held-out E1 test: official MCQ test, stratified subsample (full file kept for official eval)
    mcq = [json.loads(line) for line in open(base / "test/when2call_test_mcq.jsonl")]
    lab_map = {"tool_call": "ACT", "request_for_info": "ASK", "cannot_answer": "DECLINE"}
    tby = defaultdict(list)
    for r in mcq:
        tby[r["correct_answer"]].append(r)
    test = []
    for k, items in sorted(tby.items()):
        rng.shuffle(items)
        for r in items[: n_test // 3]:
            test.append(w2c_point(f"w2c-test-{r['uuid']}", r["tools"],
                                  [{"role": "user", "content": r["question"]}],
                                  lab_map[k], "when2call_mcq_test", orig_source=r.get("source")))
    out["test"] = test
    for v in out.values():
        rng.shuffle(v)
    return out


# ---------------------------------------------------------------- When2Tool task pools

SKIP_ENVS = {"ListManipulationEnv"}  # axis semantics not reimplemented


def w2t_task(r: dict, split: str, hop: str) -> dict | None:
    from whetherman.tools.w2t import ENV_TOOLS

    if r["env_name"] in SKIP_ENVS or r["env_name"] not in ENV_TOOLS:
        return None
    if re.search(r"custom", r["instruction"], re.I):
        return None  # custom algorithms exist only in the benchmark's code
    params = json.loads(r["parameters"]) if r["parameters"] else {}
    tools = ENV_TOOLS[r["env_name"]]
    return {
        "id": f"w2t-{hop}-{split}-{r['id']}",
        "source": f"when2tool_{hop}_{split}",
        "messages": [{"role": "user", "content": r["instruction"]}],
        "tools": tools,
        "appropriate_tool": tools[0],
        "context": {"corpus": params["corpus"]} if "corpus" in params else {},
        "oracle": {"type": "answer", "gold": str(r["answer"])},
        "meta": {"env": r["env_name"], "difficulty": r["difficulty"],
                 "multi_step": bool(r["multi_step"]),
                 "steps": json.loads(r["steps"]) if r["steps"] else []},
    }


def _strat(tasks: list[dict], per_cell: int, rng: random.Random) -> tuple[list[dict], list[dict]]:
    cells = defaultdict(list)
    for t in tasks:
        cells[(t["meta"]["env"], t["meta"]["difficulty"])].append(t)
    take, rest = [], []
    for k in sorted(cells):
        v = cells[k][:]
        rng.shuffle(v)
        take += v[:per_cell]
        rest += v[per_cell:]
    return take, rest


def build_w2t(rng: random.Random) -> dict[str, list[dict]]:
    import pandas as pd

    def load(hop: str, split: str) -> list[dict]:
        df = pd.read_parquet(RAW / "when2tool" / f"{hop}/{split}-00000-of-00001.parquet")
        return [t for r in df.to_dict("records") if (t := w2t_task(r, split, hop))]

    train = load("single_hop", "train")
    test = load("single_hop", "test")
    e2_train, rest = _strat(train, 5, rng)        # ~14 envs x 3 difficulties x 5
    probe, _ = _strat(rest, 3, rng)                # disjoint from e2_train
    e2_test, _ = _strat(test, 4, rng)
    mh_train = load("multi_hop", "train")
    mh_test = load("multi_hop", "test")
    mh_train_s, _ = _strat(mh_train, 6, rng)
    mh_test_s, _ = _strat(mh_test, 4, rng)
    return {"e2_train": e2_train, "probe": probe, "e2_test": e2_test,
            "multihop_train": mh_train_s, "multihop_test": mh_test_s}


# ---------------------------------------------------------------- io


def write_jsonl(rows: list[dict], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    path.write_text(data)
    return hashlib.sha256(data.encode()).hexdigest()


def download() -> None:
    for f in W2C_FILES:
        subprocess.run(["curl", "-sSL", "--create-dirs", "-o", str(RAW / "when2call" / f),
                        f"https://huggingface.co/datasets/nvidia/When2Call/resolve/main/{f}"], check=True)
    for f in W2T_FILES:
        subprocess.run(["curl", "-sSL", "--create-dirs", "-o", str(RAW / "when2tool" / f),
                        f"https://huggingface.co/datasets/cesun/When2Tool/resolve/main/{f}"], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--e1-train", type=int, default=4500)
    ap.add_argument("--e1-val", type=int, default=600)
    ap.add_argument("--e1-test", type=int, default=900)
    a = ap.parse_args()
    if a.download:
        download()
    rng = random.Random(SEED)
    manifest: dict[str, Any] = {"seed": SEED, "files": {}}
    for name, rows in build_e1(rng, a.e1_train, a.e1_val, a.e1_test).items():
        manifest["files"][f"e1/{name}.jsonl"] = {
            "n": len(rows), "labels": dict(Counter(r["label"] for r in rows)),
            "sha256": write_jsonl(rows, FIX / "e1" / f"{name}.jsonl")}
    for name, rows in build_w2t(rng).items():
        manifest["files"][f"w2t/{name}.jsonl"] = {
            "n": len(rows), "envs": dict(Counter(r["meta"]["env"] for r in rows)),
            "sha256": write_jsonl(rows, FIX / "w2t" / f"{name}.jsonl")}
    (FIX / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    for k, v in manifest["files"].items():
        print(k, v["n"], v.get("labels", ""))


if __name__ == "__main__":
    main()
