"""Shared FunctionGemma backbone: LoRA + 4-class head, no agent identity in the input.
(`--identity-from CKPT` trains the P1 identity-conditioned comparison model instead.)

uv run python -m training.train_backbone --out checkpoints/shared-v3 --epochs 3 --soft-disagreement [--max-train N]

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

from whetherman.labeling import infra_error
from whetherman.labels import IDX, LABELS

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "data" / "fixtures"


def read(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.open()] if p.exists() else []


def e2_agreement_points(agents: list[str], soft: bool = False) -> list[dict]:
    """E2-train tasks whose usable labels agree across every listed agent. With soft=True, tasks where
    the labels are mixed or ambiguous are added with soft target P(ACT) = mean over agents of the no-tool
    failure rate (no agent identity in the input)."""
    from whetherman.tools.registry import REGISTRY, load_all

    load_all()
    tasks = {t["id"]: t for t in read(FIX / "w2t" / "e2_train.jsonl")}
    labs = [{r["task_id"]: r for r in read(FIX / "labels" / a / "e2_train.jsonl") if not infra_error(r)}
            for a in agents]
    out = []
    for tid, t in tasks.items():
        rs = [lab.get(tid) for lab in labs]
        if any(r is None for r in rs):
            continue
        ls = [r["label"] for r in rs]
        if all(x in ("ACT", "ANSWER") for x in ls) and len(set(ls)) == 1:
            p_act = float(ls[0] == "ACT")
        elif soft:  # P(answering directly fails), averaged over agents; defined for ambiguous labels too
            p_act = sum(1 - r["notool_success"] / r["notool_trials"] for r in rs) / len(rs)
        else:
            continue
        pt = {"id": f"e2a-{tid}", "source": "e2_train_agree" if p_act in (0.0, 1.0) else "e2_train_soft",
              "messages": t["messages"], "tools": [REGISTRY[n].schema() for n in t["tools"]],
              "label": "ACT" if p_act >= 0.5 else "ANSWER",
              "meta": {"env": t["meta"]["env"], "agents": agents}}
        if p_act not in (0.0, 1.0):
            pt["target"] = [p_act, 1 - p_act, 0.0, 0.0]
        out.append(pt)
    return out


def identity_points(agents: list[str], e2_ids: set[str]) -> list[dict]:
    """Identity-conditioned comparison: each agent's own E2-train labels (restricted to `e2_ids`, the tasks
    the shared backbone trained on) and probe labels, with that agent's ID in the input."""
    from whetherman.adapters.functiongemma import with_identity
    from whetherman.calibrate.probe_pack import load_probe_pack
    from whetherman.calibrate.run import decision_points
    from whetherman.tools.registry import REGISTRY, load_all

    load_all()
    tasks = {t["id"]: t for t in read(FIX / "w2t" / "e2_train.jsonl")}
    pack = load_probe_pack(FIX / "probe_pack_large")
    pack_ids = {t.id for t in pack.tasks}
    out = []
    for ag in agents:
        for r in read(FIX / "labels" / ag / "e2_train.jsonl"):
            if r["task_id"] in e2_ids and r["label"] in ("ACT", "ANSWER") and not infra_error(r):
                t = tasks[r["task_id"]]
                out.append({"id": f"e2i-{ag}-{t['id']}", "source": "e2_train_identity", "label": r["label"],
                            "messages": with_identity(t["messages"], ag),
                            "tools": [REGISTRY[n].schema() for n in t["tools"]], "meta": {"agent": ag}})
        labs = [r for f in ("probe.jsonl", "probe_dev.jsonl", "probe_extra.jsonl")
                for r in read(FIX / "labels" / ag / f) if r["task_id"] in pack_ids and not infra_error(r)]
        for p in decision_points(pack, labs):
            out.append({**p, "id": f"probe-{ag}-{p['id']}", "source": "probe_identity",
                        "messages": with_identity(p["messages"], ag), "meta": {"agent": ag}})
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
    ap.add_argument("--out", default="checkpoints/shared-v3")
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
    ap.add_argument("--soft-disagreement", action="store_true",
                    help="train on E2 items the agents disagree on, with soft P(ACT) targets")
    ap.add_argument("--identity-from", default=None, metavar="CKPT",
                    help="identity-conditioned comparison: agent ID in the input, each agent's own E2 labels on "
                         "the E2 tasks CKPT trained on, plus probe labels; agent-agnostic data tagged 'any'")
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model

    from whetherman.adapters.functiongemma import FunctionGemmaRouter

    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    ms_train, ms_val = split_multistep(read(FIX / "train" / "multistep.jsonl"), 0.15, rng)
    missing = [x for x in a.agents if not (FIX / "labels" / x / "e2_train.jsonl").exists()]
    agents = [] if missing or len(a.agents) < 2 else a.agents
    if missing:
        print(f"WARNING: no E2 labels for {missing}; training WITHOUT E2 agreement items "
              "(agreement needs every agent, otherwise one agent's labels would leak into the shared backbone)")
    e2all = e2_agreement_points(agents, soft=a.soft_disagreement) if agents else []
    rng.shuffle(e2all)
    e2a = [x for x in e2all if "target" not in x]
    e2s = [x for x in e2all if "target" in x]
    n_e2v, n_e2s = len(e2a) // 5, len(e2s) // 5  # held out: 20% of agreement (val) and of disagreement (ladder)
    train = read(FIX / "e1" / "train.jsonl") + ms_train + e2a[n_e2v:] + e2s[n_e2s:]
    if a.identity_from:
        from whetherman.adapters.functiongemma import with_identity

        ref = json.loads((ROOT / a.identity_from / "train_meta.json").read_text())
        e2a, e2s, n_e2v, n_e2s = [], [], 0, 0
        train = [{**x, "messages": with_identity(x["messages"], "any")}
                 for x in read(FIX / "e1" / "train.jsonl") + ms_train]
        train += identity_points(agents, set(ref["e2_train_ids_trained"]))
        ms_val = [{**x, "messages": with_identity(x["messages"], "any")} for x in ms_val]
    dd = ROOT / "eval" / "results" / "dedup.json"
    removed = {i for _, i in json.loads(dd.read_text())["train_side_removals"]} if dd.exists() else set()
    n0 = len(train)
    train = [x for x in train if x["id"] not in removed and x["id"].removeprefix("e2a-") not in removed]
    print(f"dedup: removed {n0 - len(train)} training items colliding with eval items")
    val = read(FIX / "e1" / "val.jsonl") + ms_val + e2a[:n_e2v]
    if a.identity_from:
        val = [{**x, "messages": with_identity(x["messages"], "any")} for x in read(FIX / "e1" / "val.jsonl")] + ms_val
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
            tgt = x.get("target") or [float(i == IDX[x["label"]]) for i in range(4)]
            out.append((f["input_ids"], IDX[x["label"]], f["truncated"], tgt))
        return out

    t0 = time.time()
    tr, va = enc(train), enc(val)
    print(f"encoded in {time.time() - t0:.0f}s; truncated train {sum(x[2] for x in tr)}; "
          f"mean len {np.mean([len(i) for i, *_ in tr]):.0f}")
    counts = np.array([x[3] for x in tr]).sum(0)
    if a.class_weight == "balanced":
        w = np.where(counts > 0, counts.sum() / (4 * np.maximum(counts, 1)), 0.0)
    else:
        w = np.ones(4)
    wt = torch.tensor(w, dtype=torch.float32, device="cuda")

    def loss_fn(logits, target):
        """Class-weighted soft cross-entropy; equals weighted CE (mean reduction) for one-hot targets."""
        wtgt = target * wt
        return -(wtgt * torch.log_softmax(logits, -1)).sum() / wtgt.sum()
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
            loss = loss_fn(head(h).float(), torch.tensor([x[3] for x in b], device="cuda")) / a.accum
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
            "e2_train_ids_trained": sorted({x["id"].removeprefix("e2a-") if x["id"].startswith("e2a-") else
                                            x["id"].split("-", 2)[2] for x in train
                                            if x["id"].startswith(("e2a-", "e2i-"))}),
            "e2_train_ids_val": sorted(x["id"].removeprefix("e2a-") for x in e2a[:n_e2v]),
            "train_labels": dict(Counter(LABELS[x[1]] for x in tr)), "class_weight": w.tolist(),
            "n_soft_train": sum("target" in x for x in train), "identity_from": a.identity_from,
            "n_identity_train": dict(Counter(x["source"] for x in train if "identity" in x["source"])),
            "e2_disagreement_ids_heldout":
                sorted(x["id"].removeprefix("e2a-") for x in e2s[:n_e2s]),
            "history": history, "best_val_macro_f1": best,
            "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20, "precision": "fp32",
            "train_seconds": time.time() - t0}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("best_val_macro_f1", "peak_vram_mib", "train_seconds")}))


if __name__ == "__main__":
    main()
