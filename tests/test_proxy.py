import copy
import json
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from whetherman.calibrate.adjust import VectorScaling
from whetherman.policy.gate import Gate, Profile
from whetherman.proxy.rewrite import ANSWER_INSTRUCTION, ASK_INSTRUCTION, DECLINE_INSTRUCTION, rewrite, route_reason
from whetherman.proxy.server import create_app

REQ = json.loads((Path(__file__).parent / "fixtures" / "multicall_request.json").read_text())
LOGITS = {"ACT": [5, 0, 0, 0], "ANSWER": [0, 5, 0, 0], "ASK": [0, 0, 5, 0], "DECLINE": [0, 0, 0, 5]}


class FakeRouter:
    placement = "cpu"

    def __init__(self, label="ANSWER", mode="ok"):
        self.label, self.mode = label, mode

    def format(self, d):
        if self.mode == "format_error":
            raise RuntimeError("boom")
        return {"input_ids": [1]}

    def predict(self, f):
        if self.mode == "crash":
            raise RuntimeError("classifier died")
        if self.mode == "slow":
            time.sleep(0.5)
        if self.mode == "invalid":
            return {"logits": [float("nan"), 1, 2]}
        return {"logits": LOGITS[self.label]}


def gate(label="ANSWER", mode="ok", tau=0.5):
    return Gate(FakeRouter(label, mode), Profile(agent="t", tau=tau, adjust=VectorScaling()), timeout_ms=100)


def roles(msgs):
    return [m["role"] for m in msgs]


# ---------------------------------------------------------------- rewriting consistency

@pytest.mark.parametrize("decision", ["ANSWER", "ASK", "DECLINE"])
def test_rewrite_removes_tools_and_preserves_history(decision):
    body = copy.deepcopy(REQ)
    out = rewrite(body, decision)
    assert body == REQ, "input must not be mutated"
    for k in ("tools", "tool_choice", "parallel_tool_calls"):
        assert k not in out
    for k in ("model", "temperature", "seed", "max_tokens", "user"):
        assert out[k] == REQ[k]
    assert roles(out["messages"]) == roles(REQ["messages"])
    # every tool message still answers an assistant tool_call id, in order
    ids = [tc["id"] for m in out["messages"] if m["role"] == "assistant" for tc in m.get("tool_calls") or []]
    assert ids == ["call_a", "call_b", "call_c"]
    assert [m["tool_call_id"] for m in out["messages"] if m["role"] == "tool"] == ids
    assert out["messages"][1:] == REQ["messages"][1:]
    sys0 = out["messages"][0]["content"]
    assert sys0.startswith("Be concise.")
    if decision == "ASK":
        assert ASK_INSTRUCTION in sys0
    elif decision == "DECLINE":
        assert DECLINE_INSTRUCTION in sys0
    else:
        assert sys0 == "Be concise.\n\n" + ANSWER_INSTRUCTION


def test_rewrite_inserts_system_when_missing():
    body = {"model": "m", "tools": REQ["tools"], "messages": [{"role": "user", "content": "x"}]}
    out = rewrite(body, "ASK")
    assert roles(out["messages"]) == ["system", "user"]


def test_act_is_identity_and_unknown_fails_open():
    assert rewrite(REQ, "ACT") is REQ
    assert rewrite(REQ, "MAYBE") is REQ


@pytest.mark.parametrize("tc,expected", [
    ("required", "forced_tool_choice"), ("none", "forced_tool_choice"),
    ({"type": "function", "function": {"name": "calculator"}}, "forced_tool_choice"),
    ("auto", None), (None, None)])
def test_tool_choice_routing(tc, expected):
    body = {**REQ, "tool_choice": tc}
    if tc is None:
        body.pop("tool_choice")
    assert route_reason(body) == expected


def test_unsupported_shapes_forwarded():
    assert route_reason({**REQ, "stream": True}) == "stream"
    assert route_reason({**REQ, "tools": []}) == "no_tools"
    assert route_reason({"messages": "nope"}) == "malformed"
    assert route_reason([1, 2]) == "malformed"
    img = {**REQ, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]}
    assert route_reason(img) == "non_text_content"


# ---------------------------------------------------------------- server end to end

class Upstream:
    def __init__(self):
        self.seen = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen.append((request.url.path, request.content))
        return httpx.Response(200, json={"id": "x", "choices": [{"message": {"role": "assistant",
                              "content": "ok"}, "finish_reason": "stop"}], "usage": {}})


def app_with(g, tmp_path=None):
    up = Upstream()
    log = str(tmp_path / "log.jsonl") if tmp_path else None
    app = create_app(g, "http://upstream/v1", profile_name="t", log_path=log,
                     transport=httpx.MockTransport(up))
    return TestClient(app), up


def test_answer_decision_strips_tools_upstream(tmp_path):
    c, up = app_with(gate("ANSWER"), tmp_path)
    r = c.post("/v1/chat/completions", json=REQ)
    assert r.status_code == 200 and r.headers["x-whetherman-decision"] == "ANSWER"
    sent = json.loads(up.seen[-1][1])
    assert "tools" not in sent and sent["messages"][1:] == REQ["messages"][1:]
    # logs contain no prompt content
    rec = (tmp_path / "log.jsonl").read_text()
    assert "Oslo" not in rec and "Be concise" not in rec and "temp_c" not in rec
    assert json.loads(rec.splitlines()[0])["decision"] == "ANSWER"


@pytest.mark.parametrize("mode", ["crash", "slow", "invalid", "format_error"])
def test_classifier_failures_fail_open(mode):
    c, up = app_with(gate("ANSWER", mode))
    r = c.post("/v1/chat/completions", json=REQ)
    assert r.status_code == 200 and r.headers["x-whetherman-decision"] == "ACT"
    assert json.loads(up.seen[-1][1]) == REQ


def test_malformed_body_forwarded_verbatim():
    c, up = app_with(gate("ANSWER"))
    raw = b'{"model": "m", "messages": [ broken'
    r = c.post("/v1/chat/completions", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 200 and up.seen[-1][1] == raw


@pytest.mark.parametrize("tc", ["required", "none", {"type": "function", "function": {"name": "calculator"}}])
def test_forced_tool_choice_never_overridden(tc):
    c, up = app_with(gate("ANSWER"))
    body = {**REQ, "tool_choice": tc}
    c.post("/v1/chat/completions", json=body)
    assert json.loads(up.seen[-1][1]) == body


def test_threshold_keeps_act_when_unconfident():
    c, up = app_with(gate("ANSWER", tau=0.999))
    r = c.post("/v1/chat/completions", json=REQ)
    assert r.headers["x-whetherman-decision"] == "ACT"
    assert json.loads(up.seen[-1][1]) == REQ


def test_no_gate_is_transparent():
    c, up = app_with(None)
    c.post("/v1/chat/completions", json=REQ)
    assert json.loads(up.seen[-1][1]) == REQ


def test_gate_fails_open_when_feature_head_gets_no_hidden_state():
    from whetherman.calibrate.adjust import FeatureHead

    prof = Profile(agent="t", tau=0.5, adjust=FeatureHead(mean=[0.0], std=[1.0], w=[-5.0], b=-5.0))
    g = Gate(FakeRouter("ANSWER"), prof, timeout_ms=100)
    out = g.decide({"messages": [{"role": "user", "content": "x"}], "tools": []})
    assert out["decision"] == "ACT" and out["error"].startswith("error:")


class ToolCallingUpstream(Upstream):
    """A model that emits a tool call even when the request offers no tools."""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen.append((request.url.path, request.content))
        msg = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]}
        return httpx.Response(200, json={"id": "x", "choices": [{"message": msg, "finish_reason": "tool_calls"}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 2}})


def test_noncompliant_tool_call_fails_open_to_original_request(tmp_path):
    up = ToolCallingUpstream()
    app = create_app(gate("ANSWER"), "http://upstream/v1", profile_name="t", log_path=str(tmp_path / "l.jsonl"),
                     transport=httpx.MockTransport(up))
    r = TestClient(app).post("/v1/chat/completions", json=REQ)
    assert r.headers["x-whetherman-decision"] == "ACT" and len(up.seen) == 2
    assert "tools" not in json.loads(up.seen[0][1]) and json.loads(up.seen[1][1]) == REQ
    assert json.loads((tmp_path / "l.jsonl").read_text().splitlines()[0])["noncompliant"] is True


def test_gated_client_noncompliance_counts_both_calls():
    from whetherman.providers.openai_compat import ChatClient
    from whetherman.proxy.server import GatedClient

    up = ToolCallingUpstream()
    base = ChatClient("m", "digitalocean", api_key="k", transport=httpx.MockTransport(up))
    res = GatedClient(base, gate("ANSWER")).chat(REQ["messages"], tools=REQ["tools"])
    assert res.gate["noncompliant"] and res.gate["decision"] == "ACT" and not res.gate["tools_removed"]
    assert res.usage.prompt_tokens == 20 and len(up.seen) == 2
