import json

import numpy as np
import pytest

from whetherman.calibrate.probe_pack import ProbePackError, load_probe_pack
from whetherman.calibrate.run import calibrate
from whetherman.policy.gate import Profile
from whetherman.providers.base import ChatResult, Usage


def make_pack(tmp_path, tasks):
    (tmp_path / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    (tmp_path / "pack.json").write_text(json.dumps({"name": "tiny", "version": "0", "task_files": ["tasks.jsonl"],
                                                   "fixtures": []}))
    return tmp_path


def calc_task(i, a, b):
    return {"id": f"t{i}", "messages": [{"role": "user", "content": f"{a}*{b}=?"}], "tools": ["calculator"],
            "oracle": {"type": "answer", "gold": str(a * b)}}


TASKS = [calc_task(i, 3 + i, 4) for i in range(10)] + [calc_task(10 + i, 98765 + i, 4321) for i in range(10)]


class MockAgent:
    """Answers small products from memory; needs the calculator for big ones."""

    calls = 0
    pricing = type("P", (), {"cost": staticmethod(lambda u: 0.0)})()

    def chat(self, messages, tools=None, **kw):
        MockAgent.calls += 1
        last = messages[-1]
        if last["role"] == "tool":
            return self._say(str(json.loads(last["content"])["result"]))
        if "*" not in last["content"]:
            return self._say("OK")
        q = last["content"].split("=")[0]
        a, b = (int(x) for x in q.split("*"))
        if a < 1000:
            return self._say(str(a * b))
        if tools:
            return ChatResult(message={"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "calculator",
                                                              "arguments": json.dumps({"expression": q})}}]},
                usage=Usage(), latency_ms=1, cost_usd=0)
        return self._say("42")

    def _say(self, text):
        return ChatResult(message={"role": "assistant", "content": text}, usage=Usage(), latency_ms=1, cost_usd=0)

    def close(self):
        pass


class FakeRouter:
    placement = "cpu"

    def format(self, d):
        return {"n_tokens": 1, "q": d["messages"][-1]["content"]}

    def predict_batch(self, fs):
        # backbone is biased toward ANSWER for everything
        return [{"logits": [0.0, 1.0, -3.0, -3.0]} for _ in fs]


def test_pack_validation_fails_before_agent_calls(tmp_path):
    bad = [dict(TASKS[0], oracle=None)]
    with pytest.raises(ProbePackError, match="oracle"):
        load_probe_pack(make_pack(tmp_path, bad))
    bad = [dict(TASKS[0], tools=["teleport"])]
    with pytest.raises(ProbePackError, match="implementation"):
        load_probe_pack(make_pack(tmp_path, bad))
    MockAgent.calls = 0
    with pytest.raises(ProbePackError):
        calibrate("http://x/v1", "mock", str(tmp_path), yes=True, client_factory=MockAgent, router=FakeRouter())
    assert MockAgent.calls == 0


def test_default_pack_is_valid():
    p = load_probe_pack()
    assert len(p.tasks) >= 100 and len(p.sha256) == 64


def test_calibrate_end_to_end_on_mock_agent(tmp_path):
    (tmp_path / "pack").mkdir()
    pack = make_pack(tmp_path / "pack", TASKS)
    out = calibrate("http://x/v1", "mock-agent", str(pack), name="mock", yes=True, client_factory=MockAgent,
                    router=FakeRouter(), labels_dir=str(tmp_path / "labels"), out_dir=str(tmp_path / "profiles"))
    prof = Profile.load(out)
    assert prof.meta["label_counts"] == {"ACT": 10, "ANSWER": 10, "ambiguous": 0}
    assert prof.meta["usable_labels"] == 20 and prof.meta["fit"]["n_adapt"] == 12
    # a constant backbone cannot separate the classes: the chosen tau must not skip needed calls
    at = prof.meta["fit"]["threshold_stats_at_tau"]
    assert at["false_skips"] == 0 and at["act_n"] == 4
    # half the probe items need the tool, so the fitted layer moves mass toward ACT
    assert prof.adjust.probs(np.array([0.0, 1.0, -3.0, -3.0]))[0] > 0.4
