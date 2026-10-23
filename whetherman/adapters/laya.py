"""Laya adapter (P2 comparison router, never the shipped default).

Laya (convaiinnovations/laya, Apache-2.0) is a ModernBERT-large decision encoder. Here its encoder is used
frozen: the decision input (tool schemas, current step, conversation) is serialized to text, mean-pooled,
and a 4-class head trained on Whetherman's training split maps the embedding to ACT/ANSWER/ASK/DECLINE.

router = LayaRouter(head="checkpoints/laya-head")
formatted = router.format(decision_input)   # token ids within Laya's 512-token limit, truncation logged
router.predict_batch([formatted])            # label, 4 logits, probabilities, latency, pooled embedding
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from whetherman.adapters.functiongemma import MAX_TOOL_RESULT_CHARS, _pred

LAYA_REPO = "convaiinnovations/laya"
LAYA_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"  # laya.PINNED_REVISIONS (reviewed SHA)
MAX_LEN = 512  # the checkpoint's trained sequence length (rl_agent_config.json max_len)
MAX_DESC_CHARS = 300


def _schema_lines(tools: list[dict[str, Any]]) -> list[str]:
    out = []
    for t in tools:
        f = t.get("function", t)
        params = (f.get("parameters") or {}).get("properties") or {}
        ps = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in params.items())
        out.append(f"- {f.get('name')}({ps}): {(f.get('description') or '')[:MAX_DESC_CHARS]}")
    return out


def _message_lines(messages: list[dict[str, Any]]) -> list[str]:
    out = []
    for m in messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            calls = "; ".join(f"{tc['function']['name']}({tc['function'].get('arguments', '')})"
                              for tc in m["tool_calls"])
            out.append(f"[assistant calls] {calls}")
        elif role == "tool":
            c = m.get("content")
            c = c if isinstance(c, str) else json.dumps(c)
            out.append(f"[tool result] {c[:MAX_TOOL_RESULT_CHARS]}")
        elif isinstance(m.get("content"), str):
            out.append(f"[{role}] {m['content']}")
    return out


class LayaRouter:
    def __init__(self, head: str | Path | None = None, device: str = "auto", max_len: int = MAX_LEN,
                 adapter: str | Path | None = None) -> None:
        import laya
        import torch

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = self.placement = device
        agent = laya.load(LAYA_REPO, device=device, revision=LAYA_REVISION)
        self.tok = agent.tok
        self.encoder = agent.model.encoder.float().eval()
        if adapter is not None:  # option (b): LoRA-fine-tuned encoder
            from peft import PeftModel

            self.encoder = PeftModel.from_pretrained(self.encoder, str(adapter)).merge_and_unload().eval()
        self.max_len = max_len
        self.head = torch.nn.Linear(self.encoder.config.hidden_size, 4).to(device)
        if head is not None:
            self.head.load_state_dict(torch.load(Path(head) / "head.pt", map_location="cpu"))
        self.head.eval()

    def _ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def format(self, d: dict[str, Any]) -> dict[str, Any]:
        """Schemas, step and conversation as text. Over the limit: drop the oldest messages after the first
        user turn (keeping the latest tool results), then cut the schema block from its end."""
        msgs = [m for m in d["messages"] if m.get("role") != "system"]
        step = sum(1 for m in msgs if m.get("role") == "assistant" and m.get("tool_calls"))
        schema = "Tools:\n" + "\n".join(_schema_lines(d.get("tools") or []))
        lines = _message_lines(msgs)
        budget = self.max_len - 2  # [CLS] ... [SEP]

        def text(ls: list[str], sch: str) -> str:
            return f"{sch}\nCurrent step: {step}\nConversation:\n" + "\n".join(ls)

        full = self._ids(text(lines, schema))
        ids, dropped = full, 0
        while len(ids) > budget and len(lines) - dropped > 2:
            dropped += 1
            ids = self._ids(text(lines[:1] + lines[1 + dropped:], schema))
        cut_schema = False
        if len(ids) > budget:  # schema block too long for what is left: keep its first tokens
            convo = self._ids(text(lines[:1] + lines[1 + dropped:], ""))
            room = max(0, budget - len(convo))
            sch_ids = self._ids(schema)[:room]
            ids = (sch_ids + convo)[-budget:]
            cut_schema = True
        ids = [self.tok.cls_token_id] + ids + [self.tok.sep_token_id]
        return {"input_ids": ids, "n_tokens": len(ids), "n_tokens_full": len(full) + 2,
                "truncated": len(full) > budget, "dropped_messages": dropped, "cut_schema": cut_schema}

    def embed(self, batch_ids: list[list[int]]):
        torch = self.torch
        n = max(len(x) for x in batch_ids)
        pad = self.tok.pad_token_id
        ids = torch.tensor([x + [pad] * (n - len(x)) for x in batch_ids], device=self.device)
        mask = torch.tensor([[1] * len(x) + [0] * (n - len(x)) for x in batch_ids], device=self.device)
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1)

    def predict(self, formatted: dict[str, Any]) -> dict[str, Any]:
        return self.predict_batch([formatted])[0]

    def predict_batch(self, formatted: list[dict[str, Any]]) -> list[dict[str, Any]]:
        torch = self.torch
        t0 = time.perf_counter()
        with torch.inference_mode():
            h = self.embed([f["input_ids"] for f in formatted])
            logits = self.head(h).float().cpu().numpy()
            hidden = h.float().cpu().numpy()
        if self.device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000 / len(formatted)
        return [{**_pred(lg, ms), "hidden": hv} for lg, hv in zip(logits, hidden, strict=True)]


__all__ = ["LayaRouter", "LAYA_REPO", "LAYA_REVISION", "MAX_LEN"]
