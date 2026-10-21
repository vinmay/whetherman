"""Shared FunctionGemma backbone: LoRA + 4-class head, no agent identity in the input.

uv run python -m training.train_backbone --out checkpoints/shared-v1 [--epochs 2] [--max-train N]

Train: E1 train (ACT/ASK/DECLINE), templated multi-step points, E2-train items where all
labeled agents agree. Validation: E1 val, held-out multi-step families, E2 agreement val.
FP32 only (Turing GPU, no BF16).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np

from whetherman.labels import IDX, LABELS

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "data" / "fixtures"


def read(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.open()] if p.exists() else []


def e2_agreement_points(agents: list[str]) -> list[dict]:
    """E2-train tasks whose usable labels agree across every listed agent."""
    from whetherman.tools.registry import REGISTRY, load_all

    load_all()
    tasks = {t["id"]: t for t in read(FIX / "w2t" / "e2_train.jsonl")}
    labs = [{r["task_id"]: r["label"] for r in read(FIX / "labels" / a / "e2_train.jsonl")}
            for a in agents]
    out = []
    for tid, t in tasks.items():
        ls = {lab.get(tid) for lab in labs}
        if len(ls) == 1 and (lab := ls.pop()) in ("ACT", "ANSWER"):
            out.append({"id": f"e2a-{tid}", "source": "e2_train_agree", "messages": t["messages"],
                        "tools": [REGISTRY[n].schema() for n in t["tools"]], "label": lab,
                        "meta": {"env": t["meta"]["env"], "agents": agents}})
    return out


def split_multistep(pts: list[dict], val_frac: float, rng: random.Random):
    convs = sorted({p["id"].rsplit("-", 1)[0] for p in pts})
    rng.shuffle(convs)
    val = set(convs[: int(len(convs) * val_frac)])
    return ([p for p in pts if p["id"].rsplit("-", 1)[0] not in val],
            [p for p in pts if p["id"].rsplit("-", 1)[0] in val])


def macro_f1(y: np.ndarray, p: np.ndarray, classes: list[int]) -> float:
    f1s = []
    for c in classes:
        tp = ((p == c) & (y == c)).sum()
        fp = ((p == c) & (y != c)).sum()
        fn = ((p != c) & (y == c)).sum()
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(f1s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoints/shared-v1")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--agents", nargs="*", default=["L", "S"])
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    ap.add_argument("--eval-every", type=int, default=0, help="optimizer steps; 0 = per epoch")
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model

    from whetherman.adapters.functiongemma import FunctionGemmaRouter

    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    ms_train, ms_val = split_multistep(read(FIX / "train" / "multistep.jsonl"), 0.15, rng)
    agents = [x for x in a.agents if (FIX / "labels" / x / "e2_train.jsonl").exists()]
    e2a = e2_agreement_points(agents) if agents else []
    rng.shuffle(e2a)
    n_e2v = len(e2a) // 5
    train = read(FIX / "e1" / "train.jsonl") + ms_train + e2a[n_e2v:]
    val = read(FIX / "e1" / "val.jsonl") + ms_val + e2a[:n_e2v]
    rng.shuffle(train)
    if a.max_train:
        train = train[: a.max_train]
    print(f"agents for E2 agreement: {agents}; train {len(train)} {Counter(x['label'] for x in train)}")
    print(f"val {len(val)} {Counter(x['label'] for x in val)}")

    router = FunctionGemmaRouter(device="cuda", max_input_tokens=a.max_len)
    model = router.model
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    router.model = model
    head = router.head.train()
    model.print_trainable_parameters()

    def enc(items: list[dict]) -> list[tuple[list[int], int, bool]]:
        out = []
        for x in items:
            f = router.format({"messages": x["messages"], "tools": x["tools"]})
            out.append((f["input_ids"], IDX[x["label"]], f["truncated"]))
        return out

    t0 = time.time()
    tr, va = enc(train), enc(val)
    print(f"encoded in {time.time() - t0:.0f}s; truncated train {sum(t for *_, t in tr)}; "
          f"mean len {np.mean([len(i) for i, *_ in tr]):.0f}")
    counts = np.bincount([y for _, y, _ in tr], minlength=4).astype(float)
    if a.class_weight == "balanced":
        w = np.where(counts > 0, counts.sum() / (4 * np.maximum(counts, 1)), 0.0)
    else:
        w = np.ones(4)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device="cuda"))
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": a.lr},
        {"params": head.parameters(), "lr": a.head_lr}], weight_decay=0.01)
    steps_total = a.epochs * (len(tr) // (a.batch * a.accum) + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.head_lr], total_steps=steps_total,
                                                pct_start=0.06, anneal_strategy="cos")
    torch.cuda.reset_peak_memory_stats()

    def evaluate() -> dict:
        model.eval()
        head.eval()
        preds, ys, logits = [], [], []
        order = sorted(range(len(va)), key=lambda i: len(va[i][0]))
        with torch.inference_mode():
            for s in range(0, len(order), 16):
                idx = order[s: s + 16]
                h = router._hidden_last([va[i][0] for i in idx])
                lg = head(h).float().cpu().numpy()
                logits += list(lg)
                ys += [va[i][1] for i in idx]
        model.train()
        head.train()
        lg = np.array(logits)
        preds = lg.argmax(1)
        ys = np.array(ys)
        present = sorted(set(ys.tolist()))
        return {"acc": float((preds == ys).mean()), "macro_f1": macro_f1(ys, preds, present),
                "per_class_recall": {LABELS[c]: float((preds[ys == c] == c).mean()) for c in present}}

    out = ROOT / a.out
    best, history, step = -1.0, [], 0
    model.train()
    for ep in range(a.epochs):
        rng.shuffle(tr)
        # bucket by length to limit padding
        chunks = [tr[i: i + a.batch * 32] for i in range(0, len(tr), a.batch * 32)]
        batches = []
        for c in chunks:
            c = sorted(c, key=lambda x: len(x[0]))
            batches += [c[i: i + a.batch] for i in range(0, len(c), a.batch)]
        rng.shuffle(batches)
        run_loss, t_ep = 0.0, time.time()
        for bi, b in enumerate(batches):
            h = router._hidden_last([x[0] for x in b])
            loss = loss_fn(head(h).float(), torch.tensor([x[1] for x in b], device="cuda")) / a.accum
            loss.backward()
            run_loss += loss.item() * a.accum
            if (bi + 1) % a.accum == 0 or bi == len(batches) - 1:
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if a.eval_every and step % a.eval_every == 0:
                    m = evaluate()
                    history.append({"epoch": ep, "step": step, **m})
                    print("  eval", step, m, flush=True)
            if (bi + 1) % 100 == 0:
                print(f"ep {ep} batch {bi + 1}/{len(batches)} loss {run_loss / 100:.4f} "
                      f"{time.time() - t_ep:.0f}s peak {torch.cuda.max_memory_allocated() / 2**20:.0f}MiB",
                      flush=True)
                run_loss = 0.0
        m = evaluate()
        history.append({"epoch": ep, "step": step, **m})
        print(f"epoch {ep} val {m}", flush=True)
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out)
            torch.save(head.state_dict(), out / "head.pt")
    meta = {"args": vars(a), "agents_for_agreement": agents, "n_train": len(tr), "n_val": len(va),
            "train_labels": dict(Counter(LABELS[y] for _, y, _ in tr)), "class_weight": w.tolist(),
            "history": history, "best_val_macro_f1": best,
            "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20, "precision": "fp32",
            "train_seconds": time.time() - t0}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("best_val_macro_f1", "peak_vram_mib", "train_seconds")}))


if __name__ == "__main__":
    main()
