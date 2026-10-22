import json
import os

import httpx
import pytest

from whetherman.episode import run_episode
from whetherman.labeling import infra_error
from whetherman.providers.base import Pricing, Usage
from whetherman.providers.openai_compat import PROVIDERS, ChatClient, normalize_message, parse_usage
from whetherman.tasks import Task


def test_parse_usage_variants():
    assert parse_usage({"usage": {"prompt_tokens": 10, "completion_tokens": 2,
                                  "prompt_tokens_details": {"cached_tokens": 8}}}) == Usage(10, 2, 8)
    assert parse_usage({"usage": {"prompt_tokens": 5, "completion_tokens": 1, "cached_tokens": 3}}) == Usage(5, 1, 3)
    assert parse_usage({}) == Usage()


def test_normalize_message_dict_args_and_missing_ids():
    m = normalize_message({"content": "", "reasoning": "x", "tool_calls": [
        {"function": {"name": "f", "arguments": {"a": 1}}}]})
    assert m["tool_calls"][0]["id"] == "call_0"
    assert json.loads(m["tool_calls"][0]["function"]["arguments"]) == {"a": 1}
    assert "reasoning" not in m


def test_pricing_cached():
    p = Pricing(input_per_m=1.0, output_per_m=2.0, cached_input_per_m=0.5)
    assert p.cost(Usage(1_000_000, 1_000_000, 500_000)) == pytest.approx(0.5 + 0.25 + 2.0)


def test_client_sends_auth_and_body():
    seen = {}

    def handler(req):
        seen["auth"] = req.headers.get("authorization")
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "hi"},
                                                      "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 1000, "completion_tokens": 10}})

    c = ChatClient("openai-gpt-oss-120b", "digitalocean", api_key="k", transport=httpx.MockTransport(handler))
    r = c.chat([{"role": "user", "content": "x"}], tools=[{"type": "function", "function": {"name": "f"}}])
    assert seen["auth"] == "Bearer k" and seen["body"]["temperature"] == 0.0 and "tools" in seen["body"]
    assert r.cost_usd == pytest.approx(1000 * 0.10 / 1e6 + 10 * 0.70 / 1e6)


def test_client_retries_rate_limits(monkeypatch):
    monkeypatch.setattr("whetherman.providers.openai_compat.time.sleep", lambda s: None)
    codes = iter([429, 503, 200])

    def handler(req):
        code = next(codes)
        if code != 200:
            return httpx.Response(code, headers={"retry-after": "1"})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    c = ChatClient("m", "digitalocean", api_key="k", transport=httpx.MockTransport(handler))
    assert c.chat([{"role": "user", "content": "x"}]).message["content"] == "ok"
    c = ChatClient("m", "digitalocean", api_key="k", max_retries=1,
                   transport=httpx.MockTransport(lambda req: httpx.Response(429)))
    with pytest.raises(httpx.HTTPStatusError):
        c.chat([{"role": "user", "content": "x"}])


def test_generation_abort_is_agent_failure_not_infra():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(500, json={"error": {"message": "prediction aborted, token repeat limit reached"}})

    c = ChatClient("m", "ollama", transport=httpx.MockTransport(handler))
    task = Task(id="t", messages=[{"role": "user", "content": "x"}], tools=[], oracle={"type": "answer", "gold": "1"})
    ep = run_episode(c, task, with_tools=False)
    assert len(calls) == 1  # not retried
    assert not ep.success and ep.error.startswith("GenerationAborted")
    assert not infra_error({"episodes": [ep.summary()]})


SEQ = Task(id="compat-seq", messages=[{"role": "user", "content":
           "Use get_weather for Oslo, then use the calculator to convert the Celsius temperature to Fahrenheit "
           "(C*9/5+32). Reply with the Fahrenheit number only."}],
           tools=["get_weather", "calculator"], oracle={"type": "answer", "gold": "40.1"})


def _ollama_up() -> bool:
    try:
        return httpx.get("http://localhost:11434/api/tags", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def _check_sequential(client):
    ep = run_episode(client, SEQ, with_tools=True, max_steps=6)
    names = [c["name"] for c in ep.tool_calls]
    assert ep.error is None, ep.error
    assert names[:2] == ["get_weather", "calculator"], names
    # every tool message answers an assistant tool-call id
    ids = [tc["id"] for m in ep.messages if m["role"] == "assistant" for tc in m.get("tool_calls") or []]
    assert [m["tool_call_id"] for m in ep.messages if m["role"] == "tool"] == ids
    assert ep.success, ep.final_text


@pytest.mark.skipif(not _ollama_up(), reason="Ollama not running")
def test_compat_ollama_qwen_sequential():
    _check_sequential(ChatClient("qwen3.5:4b", "ollama"))


@pytest.mark.skipif(not os.environ.get(PROVIDERS["digitalocean"].api_key_env or ""), reason="no DigitalOcean key")
def test_compat_digitalocean_sequential():
    _check_sequential(ChatClient("openai-gpt-oss-120b", "digitalocean"))
