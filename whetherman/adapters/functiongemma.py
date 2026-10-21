"""FunctionGemma 270M adapter: native serialization, 4-class head, and an untuned baseline.

decision_input = {"messages": history, "tools": tool_schemas, "current_step": i}
formatted = adapter.format(decision_input)   # FunctionGemma chat-template text
pred = adapter.predict(formatted)            # {label, logits[4], probs[4], latency_ms}
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from whetherman.labels import LABELS

BASE_MODEL = "google/functiongemma-270m-it"
ACTIVATION = "You are a model that can do function calling with the following functions"
MAX_TOOL_RESULT_CHARS = 600
MAX_INPUT_TOKENS = 1536


def _load_env_token() -> None:
    env = Path(__file__).resolve().parents[2] / ".env"
    if "HF_TOKEN" not in os.environ and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip("'\"")


def _parse_args(a: Any) -> dict[str, Any]:
    if isinstance(a, dict):
        return a
    try:
        v = json.loads(a or "{}")
        return v if isinstance(v, dict) else {"value": v}
    except (json.JSONDecodeError, TypeError):
        return {"value": str(a)}


def to_native_messages(messages: list[dict[str, Any]],
                       max_result_chars: int = MAX_TOOL_RESULT_CHARS) -> list[dict[str, Any]]:
    """OpenAI-format history -> messages FunctionGemma's template accepts."""
    system = [m["content"] for m in messages if m["role"] == "system" and isinstance(m.get("content"), str)]
    dev = ACTIVATION + ("\n" + "\n".join(s.strip() for s in system) if system else "")
    out: list[dict[str, Any]] = [{"role": "developer", "content": dev}]
    names: dict[str, str] = {}
    for m in messages:
        role = m["role"]
        if role == "system":
            continue
        if role == "assistant":
            nm: dict[str, Any] = {"role": "assistant", "content": m.get("content") or None}
            if m.get("tool_calls"):
                calls = []
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    names[tc.get("id", "")] = fn.get("name", "tool")
                    calls.append({"function": {"name": fn.get("name", "tool"),
                                               "arguments": _parse_args(fn.get("arguments"))}})
                nm["tool_calls"] = calls
            out.append(nm)
        elif role == "tool":
            content = m.get("content")
            if not isinstance(content, str):
                content = json.dumps(content)
            if len(content) > max_result_chars:
                content = content[:max_result_chars] + " ...[truncated]"
            out.append({"role": "tool", "name": m.get("name") or names.get(m.get("tool_call_id", ""), "tool"),
                        "content": content})
        else:
            c = m.get("content")
            if isinstance(c, list):
                c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            out.append({"role": "user", "content": c or ""})
    return out


class FunctionGemmaRouter:
    """Shared backbone (+ optional LoRA) with a 4-class head on the last-token hidden state."""

    def __init__(self, checkpoint: str | Path | None = None, device: str = "auto",
                 dtype: str = "float32", max_input_tokens: int = MAX_INPUT_TOKENS) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _load_env_token()
        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_input_tokens = max_input_tokens
        self.tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        td = {"float32": torch.float32, "float16": torch.float16}[dtype]
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=td)
        self.head = torch.nn.Linear(model.config.hidden_size, 4)
        self.tuned = checkpoint is not None
        if checkpoint is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(checkpoint))
            model = model.merge_and_unload()
            self.head.load_state_dict(torch.load(Path(checkpoint) / "head.pt", map_location="cpu"))
        self.model = model.to(device).eval()
        self.head = self.head.to(device=device, dtype=td).eval()
        self.call_token = self.tok.convert_tokens_to_ids("<start_function_call>")
        self.placement = device

    # ------------------------------------------------------------ formatting

    def format(self, decision_input: dict[str, Any]) -> dict[str, Any]:
        msgs = to_native_messages(decision_input["messages"])
        tools = decision_input.get("tools") or []
        def encode(k: int) -> list[int]:
            # keep the developer turn and the first user turn; drop the k oldest messages after them
            m = msgs[:2] + msgs[2 + k:] if k else msgs
            text = self.tok.apply_chat_template(m, tools=tools, tokenize=False, add_generation_prompt=True)
            return self.tok(text, add_special_tokens=False)["input_ids"]

        ids = encode(0)
        truncated = False
        if len(ids) > self.max_input_tokens and len(msgs) > 3:
            lo, hi = 1, len(msgs) - 3  # smallest k that fits (binary search)
            while lo < hi:
                mid = (lo + hi) // 2
                if len(encode(mid)) <= self.max_input_tokens:
                    hi = mid
                else:
                    lo = mid + 1
            ids = encode(lo)
            truncated = True
        if len(ids) > self.max_input_tokens:  # left-truncate as a last resort (keeps the decision point)
            ids = ids[-self.max_input_tokens:]
            truncated = True
        return {"input_ids": ids, "truncated": truncated, "n_tokens": len(ids)}

    # ------------------------------------------------------------ prediction

    def _hidden_last(self, batch_ids: list[list[int]]):
        torch = self.torch
        n = max(len(x) for x in batch_ids)
        pad = self.tok.pad_token_id or 0
        ids = torch.full((len(batch_ids), n), pad, dtype=torch.long)
        mask = torch.zeros((len(batch_ids), n), dtype=torch.long)
        for i, x in enumerate(batch_ids):
            ids[i, : len(x)] = torch.tensor(x)
            mask[i, : len(x)] = 1
        ids, mask = ids.to(self.device), mask.to(self.device)
        base = self.model.get_decoder() if hasattr(self.model, "get_decoder") else self.model.model
        out = base(input_ids=ids, attention_mask=mask)
        last = mask.sum(1) - 1
        return out.last_hidden_state[torch.arange(len(batch_ids), device=self.device), last]

    def predict(self, formatted: dict[str, Any]) -> dict[str, Any]:
        return self.predict_batch([formatted])[0]

    def predict_batch(self, formatted: list[dict[str, Any]]) -> list[dict[str, Any]]:
        torch = self.torch
        t0 = time.perf_counter()
        with torch.inference_mode():
            h = self._hidden_last([f["input_ids"] for f in formatted])
            logits = self.head(h).float().cpu().numpy()
            hidden = h.float().cpu().numpy()
        if self.device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000 / len(formatted)
        return [{**_pred(lg, ms), "hidden": hv} for lg, hv in zip(logits, hidden, strict=True)]

    # ------------------------------------------------------------ untuned baseline

    def predict_untuned(self, formatted: dict[str, Any], max_new_tokens: int = 48) -> dict[str, Any]:
        """Zero-shot: P(first token = <start_function_call>) for ACT; the greedy text decides
        among the non-ACT classes (question -> ASK, refusal -> DECLINE, else ANSWER)."""
        torch = self.torch
        t0 = time.perf_counter()
        ids = torch.tensor([formatted["input_ids"]], device=self.device)
        with torch.inference_mode():
            # one prefill: first-step raw logits come back with the greedy continuation
            gen = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id, return_dict_in_generate=True,
                                      output_logits=True)
            p = torch.softmax(gen.logits[0][0].float(), -1)
            p_act = float(p[self.call_token])
        text = self.tok.decode(gen.sequences[0, ids.shape[1]:], skip_special_tokens=False)
        if "<start_function_call>" in text[:40]:
            p_act = max(p_act, 0.5)
        rest = 1 - p_act
        low = text.lower()
        k = 2 if "?" in text else 3 if any(w in low for w in ("sorry", "unable", "cannot", "can't")) else 1
        probs = np.full(4, 1e-6)
        probs[0], probs[k] = p_act, max(rest, 1e-6)
        probs = probs / probs.sum()
        return {**_pred(np.log(probs), (time.perf_counter() - t0) * 1000), "text": text[:200]}


def _pred(logits: np.ndarray, ms: float) -> dict[str, Any]:
    z = np.asarray(logits, dtype=float)
    p = np.exp(z - z.max())
    p /= p.sum()
    return {"label": LABELS[int(p.argmax())], "logits": z.tolist(), "probs": p.tolist(),
            "latency_ms": ms}
