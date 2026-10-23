"""Laya comparison (P2), option (b): short LoRA fine-tune of the Laya encoder + 4-class head.

uv run python -m training.train_laya_lora --max-steps 30            # VRAM probe first
uv run python -m training.train_laya_lora --out checkpoints/laya-lora

Same split, labels, seed and class-weighted soft CE as the backbone and option (a). FP32, batch size 2 with
gradient checkpointing. Stops at 5.5 GB peak VRAM or on any out-of-memory error (option (a) is then kept).
"""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np

from training.train_backbone import ROOT, macro_f1
from training.train_laya_head import same_split
from whetherman.labels import IDX

VRAM_STOP_MIB = 5.5 * 1024


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoints/laya-lora")
    ap.add_argument("--ref", default="checkpoints/shared-v3/train_meta.json")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N micro-batches (VRAM probe)")
    a = ap.parse_args()
    import torch
    from peft import LoraConfig, get_peft_model

    from whetherman.adapters.laya import LAYA_REVISION, LayaRouter

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    train, val = same_split(a.seed, ["L", "S"], ROOT / a.ref)
    router = LayaRouter(device="cuda")
    enc = router.encoder
    enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    enc = get_peft_model(enc, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                                         target_modules=["Wqkv", "Wo", "Wi"]))
    router.encoder = enc
    head = router.head.train()
    enc.print_trainable_parameters()

    def enc_items(items):
        return [(router.format({"messages": x["messages"], "tools": x["tools"]})["input_ids"], IDX[x["label"]],
                 x.get("target") or [float(i == IDX[x["label"]]) for i in range(4)]) for x in items]

    tr, va = enc_items(train), enc_items(val)
    counts = np.array([t[2] for t in tr]).sum(0)
    wt = torch.tensor(np.where(counts > 0, counts.sum() / (4 * np.maximum(counts, 1)), 0.0),
                      dtype=torch.float32, device="cuda")
    opt = torch.optim.AdamW([{"params": [p for p in enc.parameters() if p.requires_grad], "lr": a.lr},
                             {"params": head.parameters(), "lr": a.head_lr}], weight_decay=0.01)
    torch.cuda.reset_peak_memory_stats()

    def evaluate() -> float:
        enc.eval()
        head.eval()
        order = sorted(range(len(va)), key=lambda i: len(va[i][0]))
        out = np.zeros(len(va), dtype=int)
        with torch.inference_mode():
            for s in range(0, len(order), 16):
                idx = order[s: s + 16]
                out[idx] = head(router.embed([va[i][0] for i in idx])).argmax(1).cpu().numpy()
        enc.train()
        head.train()
        y = np.array([v[1] for v in va])
        return macro_f1(y, out, sorted(set(y.tolist())))

    out_dir = ROOT / a.out
    rng = random.Random(a.seed)
    best, history, step, stopped = -1.0, [], 0, None
    t0 = time.time()
    enc.train()
    try:
        for ep in range(a.epochs):
            rng.shuffle(tr)
            chunks = [sorted(tr[i: i + a.batch * 32], key=lambda x: len(x[0]))
                      for i in range(0, len(tr), a.batch * 32)]
            batches = [c[i: i + a.batch] for c in chunks for i in range(0, len(c), a.batch)]
            rng.shuffle(batches)
            for bi, b in enumerate(batches):
                tgt = torch.tensor([x[2] for x in b], device="cuda") * wt
                logp = torch.log_softmax(head(router.embed([x[0] for x in b])), -1)
                loss = -(tgt * logp).sum() / tgt.sum() / a.accum
                loss.backward()
                if (bi + 1) % a.accum == 0 or bi == len(batches) - 1:
                    torch.nn.utils.clip_grad_norm_([p for p in enc.parameters() if p.requires_grad]
                                                   + list(head.parameters()), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                peak = torch.cuda.max_memory_allocated() / 2**20
                if peak > VRAM_STOP_MIB:
                    stopped = f"peak VRAM {peak:.0f} MiB > {VRAM_STOP_MIB:.0f}"
                    break
                step += 1
                if step % 200 == 0:
                    print(f"ep {ep} batch {bi + 1}/{len(batches)} loss {loss.item() * a.accum:.4f} "
                          f"{time.time() - t0:.0f}s peak {peak:.0f}MiB", flush=True)
                if a.max_steps and step >= a.max_steps:
                    stopped = f"VRAM probe: {step} micro-batches"
                    break
            if stopped:
                break
            f1 = evaluate()
            history.append({"epoch": ep, "val_macro_f1": f1})
            print(f"epoch {ep} val macro-F1 {f1:.4f}", flush=True)
            if f1 > best:
                best = f1
                out_dir.mkdir(parents=True, exist_ok=True)
                enc.save_pretrained(out_dir)
                torch.save(head.state_dict(), out_dir / "head.pt")
    except torch.OutOfMemoryError:
        stopped = "out of memory"
    peak = torch.cuda.max_memory_allocated() / 2**20
    res = {"option": "b (LoRA fine-tune)", "laya_revision": LAYA_REVISION, "args": vars(a), "stopped": stopped,
           "peak_vram_mib": peak, "best_val_macro_f1": best, "history": history, "precision": "fp32",
           "train_seconds": time.time() - t0, "steps": step}
    print(json.dumps({k: res[k] for k in ("stopped", "peak_vram_mib", "best_val_macro_f1", "steps", "train_seconds")}))
    if not a.max_steps:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "train_meta.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
