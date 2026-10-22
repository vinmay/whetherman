"""Templated multi-step training decision points (continuation vs stop-now), with real tool
execution so tool results in the history are genuine. Disjoint from E3 by construction:
different entities, cities, files and phrasing (E3 is hand-written and frozen separately).

uv run python -m training.multistep_data   -> data/fixtures/train/multistep.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from whetherman.tools.registry import ToolEnv, load_all

OUT = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "train" / "multistep.jsonl"
SEED = 7
load_all()


class Conv:
    """Builds an OpenAI-format history while executing tool calls for real."""

    def __init__(self, tools: list[str], user: str, context: dict | None = None):
        self.env = ToolEnv(tools, context=context or {})
        self.msgs: list[dict] = [{"role": "user", "content": user}]
        self.n = 0

    def call(self, name: str, **args) -> str:
        self.n += 1
        cid = f"call_{self.n}"
        self.msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]})
        out = self.env.execute(name, args)
        self.msgs.append({"role": "tool", "tool_call_id": cid, "content": out})
        return out

    def point(self, label: str, family: str, pid: str) -> dict:
        return {"id": pid, "source": "template_multistep", "messages": json.loads(json.dumps(self.msgs)),
                "tools": self.env.schemas(), "label": label,
                "meta": {"family": family, "step": self.n, "multi_step": True}}


def calc_chain(rng: random.Random, i: int) -> list[dict]:
    a, b, c, d = (rng.randint(2, 999) for _ in range(4))
    ops = [f"x = {a} * {b}", f"y = x + {c}", f"z = y - {d}"][: rng.choice([2, 3])]
    conv = Conv(["evaluate_expression"], f"Using the calculator step by step: first compute {ops[0]}"
                + "".join(f", then {o}" for o in ops[1:]) + ". Report the final value only.")
    pts, val = [], None
    exprs = [f"{a}*{b}", None, None]
    for k in range(len(ops)):
        pts.append(conv.point("ACT", "calc_chain", f"ms-calc-{i}-{k}"))
        if k == 0:
            expr = exprs[0]
        elif k == 1:
            expr = f"{val}+{c}"
        else:
            expr = f"{val}-{d}"
        val = json.loads(conv.call("evaluate_expression", expression=expr))["result"]
    pts.append(conv.point("ANSWER", "calc_chain", f"ms-calc-{i}-stop"))
    return pts


CITIES = ["lisbon", "toronto", "nairobi", "osaka", "denver", "cape town", "reykjavik", "mumbai", "santiago"]


def weather_convert(rng: random.Random, i: int) -> list[dict]:
    city = rng.choice(CITIES)
    want_f = rng.random() < 0.5
    q = (f"What's the temperature in {city.title()} right now in Fahrenheit? Use the calculator for the "
         "conversion." if want_f else f"Is it colder than 10°C in {city.title()} at the moment?")
    conv = Conv(["get_weather", "calculator", "web_search"], q)
    pts = [conv.point("ACT", "weather", f"ms-wx-{i}-0")]
    t = json.loads(conv.call("get_weather", city=city))["temp_c"]
    if want_f:
        pts.append(conv.point("ACT", "weather_convert", f"ms-wx-{i}-1"))
        conv.call("calculator", expression=f"{t}*9/5+32")
    pts.append(conv.point("ANSWER", "weather_stop", f"ms-wx-{i}-stop"))
    return pts


def repo_flow(rng: random.Random, i: int) -> list[dict]:
    tools = ["read_file", "search_repo", "run_tests", "create_issue"]
    kind = rng.choice(["const", "failing", "issue", "two_files"])
    if kind == "const":
        name, path = rng.choice([("SHIPPING_FEE", "shopcart/config.py"), ("LOYALTY_DISCOUNT",
                                 "shopcart/pricing.py"), ("__version__", "shopcart/__init__.py")])
        conv = Conv(tools, f"Where is {name} defined in our repo and what is its value?")
        pts = [conv.point("ACT", "repo_search", f"ms-repo-{i}-0")]
        conv.call("search_repo", query=name)
        pts.append(conv.point("ANSWER", "repo_stop_after_search", f"ms-repo-{i}-stop"))
        return pts
    if kind == "failing":
        conv = Conv(tools, "Run the test suite and tell me why the failing test fails.")
        pts = [conv.point("ACT", "tests_run", f"ms-repo-{i}-0")]
        conv.call("run_tests")
        pts.append(conv.point("ACT", "tests_then_read", f"ms-repo-{i}-1"))
        conv.call("read_file", path="shopcart/pricing.py")
        pts.append(conv.point("ANSWER", "tests_stop", f"ms-repo-{i}-stop"))
        return pts
    if kind == "issue":
        conv = Conv(tools, "Check whether the test suite passes; if anything fails, open an issue "
                    "describing the failing test.")
        pts = [conv.point("ACT", "tests_run", f"ms-repo-{i}-0")]
        conv.call("run_tests")
        pts.append(conv.point("ACT", "issue_after_fail", f"ms-repo-{i}-1"))
        conv.call("create_issue", title="test_loyal_discount fails",
                  body="LOYAL discount subtracts the rate instead of a percentage.")
        pts.append(conv.point("ANSWER", "issue_stop", f"ms-repo-{i}-stop"))
        return pts
    conv = Conv(tools, "Compare FREE_SHIPPING_THRESHOLD in the config with how shipping() uses it.")
    pts = [conv.point("ACT", "two_files", f"ms-repo-{i}-0")]
    conv.call("read_file", path="shopcart/config.py")
    pts.append(conv.point("ACT", "two_files_second", f"ms-repo-{i}-1"))
    conv.call("read_file", path="shopcart/pricing.py")
    pts.append(conv.point("ANSWER", "two_files_stop", f"ms-repo-{i}-stop"))
    return pts


def retrieve_flow(rng: random.Random, i: int, corpus_tasks: list[dict]) -> list[dict]:
    t = rng.choice(corpus_tasks)
    conv = Conv(["search_corpus", "read_doc"], t["messages"][0]["content"], t["context"])
    pts = [conv.point("ACT", "retrieve_search", f"ms-ret-{i}-0")]
    hits = json.loads(conv.call("search_corpus", query=t["messages"][0]["content"], top_k=3))["results"]
    if rng.random() < 0.5 and hits:
        pts.append(conv.point("ACT", "retrieve_read", f"ms-ret-{i}-1"))
        conv.call("read_doc", doc_id=hits[0]["doc_id"])
    pts.append(conv.point("ANSWER", "retrieve_stop", f"ms-ret-{i}-stop"))
    return pts


def email_flow(rng: random.Random, i: int) -> list[dict]:
    city = rng.choice(CITIES)
    to = rng.choice(["sam@team.example", "ops@team.example", "lee@team.example"])
    conv = Conv(["get_weather", "send_email", "calculator"],
                f"Get the current weather in {city.title()} and email a one-line summary to {to}.")
    pts = [conv.point("ACT", "email_lookup", f"ms-mail-{i}-0")]
    w = json.loads(conv.call("get_weather", city=city))
    pts.append(conv.point("ACT", "email_send", f"ms-mail-{i}-1"))
    conv.call("send_email", to=to, subject=f"Weather in {city.title()}",
              body=f"{w['condition']}, {w['temp_c']}°C")
    pts.append(conv.point("ANSWER", "email_stop", f"ms-mail-{i}-stop"))
    return pts


def followup_answerable(rng: random.Random, i: int) -> list[dict]:
    """After one tool result, the user asks something already answered by it -> ANSWER."""
    city = rng.choice(CITIES)
    conv = Conv(["get_weather", "calculator"], f"What's the weather in {city.title()}?")
    conv.call("get_weather", city=city)
    conv.msgs.append({"role": "assistant", "content": "Here is the current weather."})
    conv.msgs.append({"role": "user", "content": rng.choice(
        ["And the humidity?", "Should I bring an umbrella?", "Was that in Celsius?"])})
    return [conv.point("ANSWER", "followup_from_history", f"ms-fu-{i}")]


def error_recovery(rng: random.Random, i: int) -> list[dict]:
    """A tool call fails or comes back empty -> the agent should try again (ACT), then stop."""
    city = rng.choice(CITIES)
    kind = rng.choice(["bad_arg", "empty_search", "missing_file"])
    if kind == "bad_arg":
        conv = Conv(["get_weather", "calculator"], f"How humid is it in {city.title()} at the moment?")
        conv.call("get_weather", location=city)  # wrong argument name -> error result
        pts = [conv.point("ACT", "retry_after_error", f"ms-err-{i}-0")]
        conv.call("get_weather", city=city)
    elif kind == "empty_search":
        conv = Conv(["web_search", "get_weather"], f"Look up the current conditions in {city.title()} for me.")
        conv.call("web_search", query=f"{city} conditions today")  # fixture has no hits -> empty
        pts = [conv.point("ACT", "retry_after_empty", f"ms-err-{i}-0")]
        conv.call("get_weather", city=city)
    else:
        conv = Conv(["read_file", "search_repo"], "What does our pricing module charge for shipping?")
        conv.call("read_file", path="pricing.py")  # wrong path -> FileNotFoundError
        pts = [conv.point("ACT", "retry_after_error", f"ms-err-{i}-0")]
        conv.call("read_file", path="shopcart/pricing.py")
    pts.append(conv.point("ANSWER", "stop_after_recovery", f"ms-err-{i}-stop"))
    return pts


def multi_entity(rng: random.Random, i: int) -> list[dict]:
    """Several entities to look up: continue (ACT) until every lookup is done, then stop."""
    cities = rng.sample(CITIES, rng.choice([2, 3]))
    names = [c.title() for c in cities]
    q = (f"Tell me the difference in humidity between {names[0]} and {names[1]}." if len(cities) == 2 else
         f"Fetch the current temperature for {', '.join(names[:-1])} and {names[-1]}, then give me the average.")
    conv = Conv(["get_weather", "calculator"], q)
    pts = []
    for k, c in enumerate(cities):
        pts.append(conv.point("ACT", "multi_entity_next", f"ms-multi-{i}-{k}"))
        conv.call("get_weather", city=c)
    pts.append(conv.point("ANSWER", "multi_entity_stop", f"ms-multi-{i}-stop"))
    return pts


def main() -> None:
    rng = random.Random(SEED)
    corpus = [json.loads(x) for x in open(OUT.parents[1] / "w2t" / "e2_train.jsonl")]
    corpus = [t for t in corpus if t["meta"]["env"] == "RetrieverEnv"]
    pts: list[dict] = []
    for i in range(60):
        pts += calc_chain(rng, i)
    for i in range(50):
        pts += weather_convert(rng, i)
    for i in range(50):
        pts += repo_flow(rng, i)
    for i in range(40):
        pts += retrieve_flow(rng, i, corpus)
    for i in range(30):
        pts += email_flow(rng, i)
    for i in range(30):
        pts += followup_answerable(rng, i)
    for i in range(40):
        pts += error_recovery(rng, i)
    for i in range(40):
        pts += multi_entity(rng, i)
    rng.shuffle(pts)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for p in pts:
            f.write(json.dumps(p, sort_keys=True) + "\n")
    from collections import Counter

    print(len(pts), Counter(p["label"] for p in pts))


if __name__ == "__main__":
    main()
