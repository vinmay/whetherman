"""OpenAI Chat Completions-compatible gating proxy (non-streaming subset).

whetherman serve --profile profiles/S.json --upstream http://localhost:11434/v1
Logs JSONL decisions without any prompt, message or tool-result content.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response

from whetherman.policy.gate import Gate
from whetherman.providers.base import ChatResult
from whetherman.providers.openai_compat import ChatClient, normalize_message, parse_usage
from whetherman.proxy.rewrite import decision_input, rewrite, route_reason

HOP_HEADERS = {"host", "content-length", "connection", "accept-encoding", "transfer-encoding"}


class DecisionLog:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, rec: dict[str, Any]) -> None:
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(rec) + "\n")


def gate_request(gate: Gate | None, body: Any, profile_name: str = "") -> tuple[Any, dict[str, Any]]:
    """Decide and rewrite. Never raises; any failure forwards the original request (ACT)."""
    rid = uuid.uuid4().hex[:12]
    info: dict[str, Any] = {"request_id": rid, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "profile": profile_name}
    reason = route_reason(body)
    if reason is not None or gate is None:
        info.update({"routed": False, "reason": reason or "no_gate", "decision": "ACT"})
        return body, info
    try:
        d = gate.decide(decision_input(body))
        new_body = rewrite(body, d["decision"])
    except Exception as e:  # fail open
        d, new_body = {"decision": "ACT", "error": f"rewrite:{type(e).__name__}"}, body
    info.update({"routed": True, **{k: d.get(k) for k in ("decision", "raw_label", "confidence", "act_prob",
                                                          "tau", "latency_ms", "error", "placement")}})
    return new_body, info


def create_app(gate: Gate | None, upstream: str, api_key: str | None = None, profile_name: str = "",
               log_path: str | None = None, transport: httpx.AsyncBaseTransport | None = None):
    app = FastAPI(title="whetherman")
    log = DecisionLog(log_path)
    client = httpx.AsyncClient(base_url=upstream.rstrip("/"), timeout=300, transport=transport)

    def fwd_headers(req: Request) -> dict[str, str]:
        h = {k: v for k, v in req.headers.items() if k.lower() not in HOP_HEADERS}
        if api_key and not any(k.lower() == "authorization" for k in h):
            h["Authorization"] = f"Bearer {api_key}"
        return h

    @app.post("/v1/chat/completions")
    async def chat(req: Request) -> Response:
        raw = await req.body()
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = None
        if body is None:
            new_body, info = None, {"request_id": uuid.uuid4().hex[:12], "routed": False,
                                    "reason": "malformed", "decision": "ACT"}
        else:
            new_body, info = gate_request(gate, body, profile_name)
        content = raw if new_body is None or new_body is body else json.dumps(new_body).encode()
        h = fwd_headers(req)
        h["content-type"] = "application/json"
        t0 = time.perf_counter()
        r = await client.post("/chat/completions", content=content, headers=h)
        info["upstream_ms"] = (time.perf_counter() - t0) * 1000
        info["upstream_status"] = r.status_code
        log.write(info)
        out_h = {k: v for k, v in r.headers.items() if k.lower() not in HOP_HEADERS | {"content-encoding"}}
        out_h["x-whetherman-decision"] = str(info.get("decision"))
        return Response(content=r.content, status_code=r.status_code, headers=out_h)

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "DELETE"])
    async def passthrough(path: str, req: Request) -> Response:
        r = await client.request(req.method, f"/{path}", content=await req.body(), headers=fwd_headers(req))
        return Response(content=r.content, status_code=r.status_code,
                        headers={k: v for k, v in r.headers.items()
                                 if k.lower() not in HOP_HEADERS | {"content-encoding"}})

    @app.get("/healthz")
    async def health() -> dict[str, Any]:
        return {"ok": True, "profile": profile_name, "gate": gate is not None}

    return app


class GatedClient:
    """In-process equivalent of the proxy for the A/B harness: same gate_request + rewrite path."""

    def __init__(self, client: ChatClient, gate: Gate | None, profile_name: str = "") -> None:
        self.client = client
        self.gate = gate
        self.profile_name = profile_name
        self.model = client.model

    def chat(self, messages, tools=None, tool_choice=None, temperature=0.0, seed=0, **params) -> ChatResult:
        body = self.client.build_body(messages, tools, tool_choice, temperature=temperature, seed=seed,
                                      **params)
        new_body, info = gate_request(self.gate, body, self.profile_name)
        raw, latency = self.client.post_raw(new_body)
        choice = (raw.get("choices") or [{}])[0]
        usage = parse_usage(raw)
        res = ChatResult(message=normalize_message(choice.get("message") or {}), usage=usage,
                         latency_ms=latency + (info.get("latency_ms") or 0.0),
                         cost_usd=self.client.pricing.cost(usage), finish_reason=choice.get("finish_reason"),
                         raw=raw)
        info["agent_ms"] = latency
        info["tools_removed"] = "tools" in body and "tools" not in new_body
        res.gate = info  # type: ignore[attr-defined]
        return res


def serve(profile_path: str, upstream: str, host: str = "127.0.0.1", port: int = 8088,
          checkpoint: str | None = None, device: str = "auto", log_path: str | None = None,
          timeout_ms: float = 3000) -> None:
    import uvicorn

    from whetherman.adapters.functiongemma import FunctionGemmaRouter
    from whetherman.policy.gate import Profile

    prof = Profile.load(profile_path)
    router = FunctionGemmaRouter(checkpoint=checkpoint or prof.backbone, device=device)
    gate = Gate(router, prof, timeout_ms=timeout_ms)
    key = os.environ.get("WHETHERMAN_UPSTREAM_API_KEY")
    app = create_app(gate, upstream, api_key=key, profile_name=prof.agent,
                     log_path=log_path or f"logs/whetherman-{prof.agent}.jsonl")
    uvicorn.run(app, host=host, port=port, log_level="warning")
