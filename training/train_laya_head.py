"""Laya comparison (P2), option (a): frozen Laya encoder + 4-class head on the same split as the backbone.

uv run python -m training.train_laya_head --out checkpoints/laya-head

Train/val are the shipped backbone's split (same seed, same recorded E2 ids, no agent identity); sizes are
checked against its train_meta.json before anything is trained.
Embeddings are computed once (the encoder is frozen); the head uses the same class-weighted soft CE.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np

from training.train_backbone import FIX, ROOT, e2_agreement_points, macro_f1, read, split_multistep
from whetherman.labels import IDX, LABELS


def same_split(seed: int, agents: list[str], ref: Path) -> tuple[list[dict], list[dict]]:
    """The backbone's split: the multi-step split is re-derived with the same seed (it is drawn first), and
    the E2 items are exactly the ids the reference checkpoint recorded as trained / validated (labels
    added after it was trained could otherwise shift the shuffle)."""
    rng = random.Random(seed)
    ms_train, ms_val = split_multistep(read(FIX / "train" / "multistep.jsonl"), 0.15, rng)
    meta = json.loads(ref.read_text())
    e2 = {x["id"].removeprefix("e2a-"): x for x in e2_agreement_points(agents, soft=True)}
    missing = [i for i in meta["e2_train_ids_trained"] + meta["e2_train_ids_val"] if i not in e2]
    if missing:
        raise SystemExit(f"{len(missing)} recorded E2 ids have no current agreement item; refusing to train")
    removed = {i for _, i in json.loads((ROOT / "eval/results/dedup.json").read_text())["train_side_removals"]}
    train = read(FIX / "e1" / "train.jsonl") + ms_train + [e2[i] for i in meta["e2_train_ids_trained"]]
    train = [x for x in train if x["id"] not in removed and x["id"].removeprefix("e2a-") not in removed]
    val = read(FIX / "e1" / "val.jsonl") + ms_val + [e2[i] for i in meta["e2_train_ids_val"]]
    if len(train) != meta["n_train"] or len(val) != meta["n_val"]:
        raise SystemExit(f"split sizes {len(train)}/{len(val)} != reference {meta['n_train']}/{meta['n_val']}")
    rng.shuffle(train)
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoints/laya-head")
    ap.add_argument("--ref", default="checkpoints/shared-v3/train_meta.json")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--embed-batch", type=int, default=16)
    a = ap.parse_args()
    import torch

    from whetherman.adapters.laya import LAYA_REVISION, LayaRouter

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    train, val = same_split(a.seed, ["L", "S"], ROOT / a.ref)
    print(f"train {len(train)} {Counter(x['label'] for x in train)}; val {len(val)}", flush=True)
    torch.manual_seed(a.seed)
    router = LayaRouter(device="cuda")
    torch.cuda.reset_peak_memory_stats()

    def embed(items: list[dict], name: str) -> tuple[np.ndarray, dict]:
        cache = out / f"emb_{name}.npy"
        fs = [router.format({"messages": x["messages"], "tools": x["tools"]}) for x in items]
        trunc = {"n": len(fs), "over_limit": sum(f["truncated"] for f in fs),
                 "dropped_history": sum(f["dropped_messages"] > 0 for f in fs),
                 "cut_schema": sum(f["cut_schema"] for f in fs)}
        if cache.exists():
            return np.load(cache), trunc
        order = sorted(range(len(fs)), key=lambda i: fs[i]["n_tokens"])
        emb = np.zeros((len(fs), router.encoder.config.hidden_size), dtype=np.float32)
        t0 = time.time()
        with torch.inference_mode():
            for s in range(0, len(order), a.embed_batch):
                idx = order[s: s + a.embed_batch]
                emb[idx] = router.embed([fs[i]["input_ids"] for i in idx]).float().cpu().numpy()
        print(f"  embedded {name} {len(fs)} in {time.time() - t0:.0f}s; truncation {trunc}", flush=True)
        np.save(cache, emb)
        return emb, trunc

    xtr, ttr = embed(train, "train")
    xva, tva = embed(val, "val")
    peak = torch.cuda.max_memory_allocated() / 2**20
    ytr = np.array([x.get("target") or [float(i == IDX[x["label"]]) for i in range(4)] for x in train], np.float32)
    yva = np.array([IDX[x["label"]] for x in val])
    counts = ytr.sum(0)
    w = torch.tensor(np.where(counts > 0, counts.sum() / (4 * np.maximum(counts, 1)), 0.0), dtype=torch.float32)

    head = torch.nn.Linear(xtr.shape[1], 4)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.01)
    Xtr, Ytr, Xva = torch.tensor(xtr), torch.tensor(ytr), torch.tensor(xva)
    rng = np.random.default_rng(a.seed)
    best, history = -1.0, []
    for ep in range(a.epochs):
        head.train()
        perm = torch.tensor(rng.permutation(len(Xtr)))
        for s in range(0, len(Xtr), a.batch):
            bi = perm[s: s + a.batch]
            wt = Ytr[bi] * w
            loss = -(wt * torch.log_softmax(head(Xtr[bi]), -1)).sum() / wt.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            pred = head(Xva).argmax(1).numpy()
        f1 = macro_f1(yva, pred, sorted(set(yva.tolist())))
        history.append({"epoch": ep, "val_macro_f1": f1, "loss": loss.item()})
        if f1 > best:
            best = f1
            torch.save(head.state_dict(), out / "head.pt")
    meta = {"laya_repo": "convaiinnovations/laya", "laya_revision": LAYA_REVISION, "option": "a (frozen encoder)",
            "args": vars(a), "n_train": len(train), "n_val": len(val), "class_weight": w.tolist(),
            "truncation": {"train": ttr, "val": tva}, "best_val_macro_f1": best, "history": history,
            "peak_vram_mib_embedding": peak, "precision": "fp32",
            "train_labels": {LABELS[i]: float(c) for i, c in enumerate(counts)}}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("best_val_macro_f1", "peak_vram_mib_embedding", "truncation")}))


if __name__ == "__main__":
    main()
