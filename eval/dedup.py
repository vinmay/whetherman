"""Cross-split deduplication: normalized-text hash + embedding cosine > 0.95.

uv run python -m eval.dedup  -> eval/results/dedup.json
Embeddings: embeddinggemma:300m via the local Ollama /api/embed endpoint. Pairs are only reported
across *different* splits; within-split near-duplicates are legitimate (templated variants).
Training-side items found to collide with an evaluation item are listed for removal.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "data" / "fixtures"
TRAIN_SIDE = {"e1_train", "multistep", "e2_train", "probe", "probe_dev"}


def text_of(messages: list[dict]) -> str:
    return " ".join(m.get("content") or "" for m in messages if m.get("role") == "user")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def load() -> dict[str, list[tuple[str, str]]]:
    def rows(p):
        return [json.loads(x) for x in open(p)]

    sets = {
        "e1_train": FIX / "e1/train.jsonl", "e1_val": FIX / "e1/val.jsonl", "e1_test": FIX / "e1/test.jsonl",
        "multistep": FIX / "train/multistep.jsonl", "e2_train": FIX / "w2t/e2_train.jsonl",
        "probe": FIX / "w2t/probe.jsonl", "e2_test": FIX / "w2t/e2_test.jsonl",
        "multihop_train": FIX / "w2t/multihop_train.jsonl", "multihop_test": FIX / "w2t/multihop_test.jsonl",
        "probe_dev": FIX / "probe_pack/probe_dev.jsonl", "e3": FIX / "e3/e3.jsonl",
        "ab_dev": FIX / "ab/ab_dev.jsonl", "ab_final": FIX / "ab/ab_final.jsonl",
    }
    out = {}
    for k, p in sets.items():
        seen, items = set(), []
        for r in rows(p):
            t = text_of(r["messages"])
            if k in ("e3", "multistep"):  # many decision points share one conversation's user text
                if t in seen:
                    continue
                seen.add(t)
            items.append((r["id"], t))
        out[k] = items
    return out


def embed(texts: list[str], url: str = "http://localhost:11434/api/embed") -> np.ndarray:
    vecs = []
    with httpx.Client(timeout=300) as c:
        for i in range(0, len(texts), 64):
            r = c.post(url, json={"model": "embeddinggemma:300m", "input": [t[:2000] for t in texts[i:i + 64]]})
            r.raise_for_status()
            vecs += r.json()["embeddings"]
    v = np.array(vecs, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def main() -> None:
    sets = load()
    flat = [(s, i, t) for s, items in sets.items() for i, t in items]
    hashes: dict[str, list] = {}
    for s, i, t in flat:
        hashes.setdefault(hashlib.sha256(norm(t).encode()).hexdigest(), []).append((s, i))
    exact = [v for v in hashes.values() if len({s for s, _ in v}) > 1]
    emb = embed([t for _, _, t in flat])
    split = np.array([s for s, _, _ in flat])
    near = []
    for a in range(len(flat)):
        sims = emb[a + 1:] @ emb[a]
        for j in np.nonzero(sims > 0.95)[0]:
            b = a + 1 + int(j)
            if split[a] != split[b]:
                near.append({"a": flat[a][:2], "b": flat[b][:2], "cos": round(float(sims[j]), 4),
                             "text_a": flat[a][2][:120], "text_b": flat[b][2][:120]})
    to_remove = set()
    for pair in [{"a": v[0], "b": w} for v in exact for w in v[1:]] + near:
        (sa, ia), (sb, ib) = pair["a"], pair["b"]
        if sa in TRAIN_SIDE and sb not in TRAIN_SIDE:
            to_remove.add((sa, ia))
        elif sb in TRAIN_SIDE and sa not in TRAIN_SIDE:
            to_remove.add((sb, ib))
    res = {"counts": {k: len(v) for k, v in sets.items()}, "exact_cross_split_groups": exact,
           "near_cross_split_pairs": near, "train_side_removals": sorted(map(list, to_remove)),
           "threshold": 0.95, "embedding_model": "embeddinggemma:300m (Ollama)"}
    (ROOT / "eval/results/dedup.json").write_text(json.dumps(res, indent=1))
    print(f"exact cross-split groups: {len(exact)}; near pairs: {len(near)}; "
          f"train-side removals: {len(to_remove)}")


if __name__ == "__main__":
    main()
