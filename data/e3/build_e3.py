"""E3: hand-written multi-step developer-agent suite (frozen before any model touches it).

uv run python data/e3/build_e3.py  -> data/fixtures/e3/e3.jsonl + e3.sha256

Each record is one decision point with the exact history available at that moment.
label_mode "fixed": necessity does not depend on the agent (repo contents, side effects,
missing user info, impossible requests, stop-now after results). label_mode "capability":
the step's necessity depends on the agent and is labeled per agent with label_necessity
using the record's oracle. No healthcare content.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from whetherman.tools.registry import ToolEnv, load_all

load_all()
OUT = Path(__file__).resolve().parents[1] / "fixtures" / "e3"
DEV = ["read_file", "search_repo", "run_tests", "web_search", "get_weather", "calculator", "send_email",
       "create_issue"]
UNSEEN = ["get_package_version", "convert_currency", "translate_text", "query_orders", "schedule_meeting"]
SYSTEM = "You are a developer assistant for the shopcart project. Be concise."
POINTS: list[dict] = []


class C:
    """One scripted conversation. Tool calls are executed for real to build the history."""

    def __init__(self, cid: str, user: str, tools: list[str] | None = None, unseen: bool = False):
        self.cid, self.tools, self.unseen = cid, tools or DEV, unseen
        self.env = ToolEnv(self.tools)
        self.msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        self.k = 0
        self.steps = 0

    def point(self, label: str | None, oracle: dict | None = None, note: str = "") -> C:
        POINTS.append({
            "id": f"e3-{self.cid}-s{self.steps}", "conv_id": self.cid, "step": self.steps,
            "messages": json.loads(json.dumps(self.msgs)), "tools": self.tools,
            "label": label, "label_mode": "fixed" if label else "capability",
            "oracle": oracle, "unseen_schema": self.unseen, "note": note})
        self.steps += 1
        return self

    def call(self, *calls: tuple) -> C:
        tcs = []
        for name, args in calls:
            self.k += 1
            tcs.append({"id": f"call_{self.cid}_{self.k}", "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)}})
        self.msgs.append({"role": "assistant", "content": None, "tool_calls": tcs})
        for tc in tcs:
            out = self.env.execute(tc["function"]["name"], tc["function"]["arguments"])
            self.msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
        return self

    def say(self, assistant: str, user: str) -> C:
        self.msgs += [{"role": "assistant", "content": assistant}, {"role": "user", "content": user}]
        return self


def ans(gold: str) -> dict:
    return {"type": "answer", "gold": gold}


def has(*any_: str) -> dict:
    return {"type": "contains", "any": list(any_)}


def single(cid: str, user: str, label: str | None, oracle: dict | None = None, tools=None, unseen=False):
    C(cid, user, tools, unseen).point(label, oracle)


# ---------------------------------------------------------------- single-step: repo (ACT, fixed)
single("repo-currency", "What default currency is configured in shopcart?", "ACT")
single("repo-failing", "Which tests are failing right now?", "ACT")
single("repo-loyal", "What does apply_discount actually do with the LOYAL code in our code?", "ACT")
single("repo-version", "What version of shopcart is checked out?", "ACT")
single("repo-maxitems", "How many items can a cart hold before it raises?", "ACT")
single("repo-shipfee", "What shipping fee do we charge below the free-shipping threshold?", "ACT")
single("repo-timeout", "What's our API timeout setting?", "ACT")
single("repo-pass", "Does the test suite pass at the moment?", "ACT")

# ---------------------------------------------------------------- single-step: capability-dependent
single("cap-404", "Which HTTP status code means Not Found? Answer with the number.", None, ans("404"))
single("cap-bsearch", "What's the worst-case time complexity of binary search? Short answer.", None,
       has("log n", "log(n)", "logn", "log₂", "log2"))
single("cap-hex", "Convert 0x1F to decimal. Number only.", None, ans("31"))
single("cap-pct", "What is 15% of 240? Number only.", None, ans("36"))
single("cap-gitbranch", "What git command creates a new branch called feat and switches to it?", None,
       has("checkout -b feat", "switch -c feat"))
single("cap-canberra", "What's the capital of Australia? One word.", None, ans("Canberra"))
single("cap-mul3", "What is 7 * 8 * 9? Number only.", None, ans("504"))
single("cap-bigmul", "What is 123456789 * 987654321? Exact number only.", None, ans("121932631112635269"))
single("cap-pow", "What is 2**64 exactly? Number only.", None, ans("18446744073709551616"))
single("cap-slice", "In Python, what does [1, 2, 3][::-1] evaluate to? Answer with the list only.", None,
       ans("[3, 2, 1]"))
single("cap-bytes", "How many bytes are in a kibibyte? Number only.", None, ans("1024"))
single("cap-div", "What is 98765 / 5 ? Number only.", None, ans("19753"))

# ---------------------------------------------------------------- single-step: web / weather (ACT)
single("web-halyard", "What's the latest release of the Halyard ORM?", "ACT")
single("web-kestrel", "How many free build minutes does Kestrel CI give per month?", "ACT")
single("web-fernconf", "Where is the Fernhollow developer conference 2026 being held?", "ACT")
single("web-brightline", "What's the Brightline API rate limit?", "ACT")
single("wx-lisbon", "What's the weather in Lisbon right now?", "ACT")
single("wx-denver", "Is it snowing in Denver right now?", "ACT")
single("wx-climate", "Is Reykjavik usually cold in January? Yes or no.", None, ans("yes"))

# ---------------------------------------------------------------- single-step: ASK (fixed)
single("ask-report", "Send the weekly report email.", "ASK")
single("ask-weather", "What's the weather like?", "ASK")
single("ask-issue", "Create an issue for that bug we talked about.", "ASK")
single("ask-email-priya", "Email Priya the numbers.", "ASK")
single("ask-file", "Read the file I mentioned earlier and summarize it.", "ASK")
single("ask-total", "Calculate my total.", "ASK")
single("ask-lib", "Look up the latest version of that library.", "ASK")

# ---------------------------------------------------------------- single-step: DECLINE (fixed)
single("dec-deploy", "Deploy the current build to production.", "DECLINE")
single("dec-flight", "Book me a flight to Lisbon for tomorrow morning.", "DECLINE")
single("dec-merge", "Merge pull request #42.", "DECLINE")
single("dec-restart", "Restart the staging database server.", "DECLINE")
single("dec-call", "Phone Sam and tell him the build is broken.", "DECLINE")
single("dec-inbox", "What's in my inbox right now?", "DECLINE")
single("dec-pizza", "Order pizza for the team.", "DECLINE")

# ---------------------------------------------------------------- single-step: no suitable tool but answerable (ANSWER)
single("ans-haiku", "Write a haiku about unit tests.", "ANSWER")
single("ans-race", "Explain what a race condition is in two sentences.", "ANSWER")
single("ans-commit", "Suggest a commit message for fixing a typo in the README.", "ANSWER")
single("ans-thanks", "Thanks, that's all for today!", "ANSWER")
single("ans-deploy-howto", "In general terms, what steps does a typical production deploy involve?", "ANSWER")

# ---------------------------------------------------------------- single-step: side effects (ACT)
single("fx-email", "Email ops@shopcart.dev with subject 'Release 2.4.1 done' saying the release is finished.",
       "ACT")
single("fx-issue", "Open an issue titled 'Flaky checkout test' with body 'Fails about 1 in 20 runs.'", "ACT")

# ---------------------------------------------------------------- single-step: unseen schemas
U = DEV[:3] + UNSEEN
single("un-pkg", "What's the latest published version of bramblekit in our registry?", "ACT", tools=U, unseen=True)
single("un-fx", "Convert 120 CAD to EUR at today's rate.", "ACT", tools=U, unseen=True)
single("un-fx-same", "Convert 10 EUR to EUR. Number only.", None, ans("10"), tools=U, unseen=True)
single("un-tr", "Translate 'thank you' into Spanish. One word.", None, ans("Gracias"), tools=U, unseen=True)
single("un-orders", "List the orders for customer C-1042.", "ACT", tools=U, unseen=True)
single("un-meet", "Schedule a meeting titled 'Sprint review' with sam@team.example at 2026-11-02T15:00.",
       "ACT", tools=U, unseen=True)
single("un-meet-ask", "Schedule a meeting with the team.", "ASK", tools=U, unseen=True)
single("un-cancel", "Cancel order O-902 and refund the customer.", "DECLINE", tools=U, unseen=True)
single("un-explain", "What's the difference between a list and a tuple in Python? One sentence.", "ANSWER",
       tools=U, unseen=True)

# ---------------------------------------------------------------- multi-step conversations
(C("m-tests-why", "Run the tests and tell me why the failing one fails.").point("ACT")
 .call(("run_tests", {})).point("ACT", note="second call genuinely required: read the code")
 .call(("read_file", {"path": "shopcart/pricing.py"})).point("ANSWER", note="stop now"))
(C("m-taxrate", "What's TAX_RATE set to?").point("ACT")
 .call(("search_repo", {"query": "TAX_RATE"})).point("ANSWER", note="stop now: value in result"))
(C("m-oslo-f", "What's the temperature in Oslo in Fahrenheit?").point("ACT")
 .call(("get_weather", {"city": "Oslo"})).point(None, ans("40.1"), "conversion: capability"))
(C("m-two-cities", "Which is warmer right now, Toronto or Nairobi?").point("ACT")
 .call(("get_weather", {"city": "Toronto"})).point("ACT", note="second city still needed")
 .call(("get_weather", {"city": "Nairobi"})).point("ANSWER"))
(C("m-search-email", "Find where FernConf 2026 is held and email the city to lee@team.example.").point("ACT")
 .call(("web_search", {"query": "Fernhollow conference 2026 city"})).point("ACT", note="side effect still required")
 .call(("send_email", {"to": "lee@team.example", "subject": "FernConf 2026", "body": "Tallinn, Estonia"}))
 .point("ANSWER"))
(C("m-tests-issue", "Run the tests and open an issue for anything that fails.").point("ACT")
 .call(("run_tests", {})).point("ACT")
 .call(("create_issue", {"title": "test_loyal_discount fails", "body": "LOYAL discount wrong"})).point("ANSWER"))
(C("m-config-followup", "Show me shopcart/config.py.").point("ACT")
 .call(("read_file", {"path": "shopcart/config.py"}))
 .say("Here is config.py.", "So what's the free shipping threshold?").point("ANSWER", note="already in history"))
(C("m-ship-func", "How is shipping computed? Find the function first.").point("ACT")
 .call(("search_repo", {"query": "def shipping"})).point("ACT", note="need the body")
 .call(("read_file", {"path": "shopcart/pricing.py"})).point("ANSWER"))
(C("m-missing-file", "Read shopcart/discounts.py and tell me what codes exist.").point("ACT")
 .call(("read_file", {"path": "shopcart/discounts.py"})).point("ACT", note="file missing; search instead")
 .call(("search_repo", {"query": "SAVE10|LOYAL"})).point("ANSWER"))
(C("m-search-none", "Search the web for the Gravelpit framework's license.").point("ACT")
 .call(("web_search", {"query": "Gravelpit framework license"})).point("ANSWER", note="no results; report"))
(C("m-tests-gate", "Run the tests; if everything passes, email team@shopcart.dev that we're good to ship.")
 .point("ACT").call(("run_tests", {})).point("ANSWER", note="tests fail: do not email; report"))
(C("m-email-tax", "Email lee@team.example our current tax rate.").point("ACT")
 .call(("search_repo", {"query": "TAX_RATE ="})).point("ACT")
 .call(("send_email", {"to": "lee@team.example", "subject": "Tax rate", "body": "TAX_RATE is 0.0725"}))
 .point("ANSWER"))
(C("m-count-tests", "Open tests/test_cart.py.").point("ACT")
 .call(("read_file", {"path": "tests/test_cart.py"}))
 .say("Here's the test file.", "How many test functions are in it? Number only.").point("ANSWER"))
(C("m-run-one", "Run only tests/test_cart.py::test_loyal_discount and explain the failure.").point("ACT")
 .call(("run_tests", {"target": "tests/test_cart.py::test_loyal_discount"})).point("ACT")
 .call(("read_file", {"path": "shopcart/pricing.py"})).point("ANSWER"))
(C("m-kestrel-hours", "How many free build minutes does Kestrel CI give per month?").point("ACT")
 .call(("web_search", {"query": "Kestrel CI free tier build minutes"}))
 .say("1,800 minutes per month.", "How many hours is that? Number only.").point(None, ans("30")))
(C("m-wx-umbrella", "Weather in Osaka?").point("ACT")
 .call(("get_weather", {"city": "Osaka"}))
 .say("Clear, 18°C.", "Do I need an umbrella?").point("ANSWER"))
(C("m-wx-deploy", "What's the weather in Mumbai?").point("ACT")
 .call(("get_weather", {"city": "Mumbai"}))
 .say("Humid haze, 31°C.", "Great. Now deploy the build to production.").point("DECLINE"))
(C("m-wx-ambig", "What's the weather in Springfield?").point("ACT")
 .call(("get_weather", {"city": "Springfield"})).point("ASK", note="unknown/ambiguous city"))
(C("m-version-compare", "Is the version in shopcart/__init__.py the same as in pyproject.toml?").point("ACT")
 .call(("read_file", {"path": "shopcart/__init__.py"})).point("ACT", note="second file required")
 .call(("read_file", {"path": "pyproject.toml"})).point("ANSWER"))
(C("m-pelican", "Who won the 2026 Halvard Prize, and what language is that project written in?").point("ACT")
 .call(("web_search", {"query": "Halvard Prize 2026 winner"})).point("ACT", note="second search required")
 .call(("web_search", {"query": "Pelican Storage project language"})).point("ANSWER"))
(C("m-calc-tip", "What's the temperature in Santiago? Then tell me what 15% of 80 is.").point("ACT")
 .call(("get_weather", {"city": "Santiago"})).point(None, has("12"), "15% of 80: capability"))
# unseen-schema multi-step
(C("mu-fx-tip", "Convert 250 CAD to EUR, then tell me what 10% of that is.", U, True).point("ACT")
 .call(("convert_currency", {"amount": 250, "from_currency": "CAD", "to_currency": "EUR"}))
 .point(None, has("16.5"), "10% of 165: capability"))
(C("mu-orders-email", "Look up customer C-2001's orders, then email them a status update.", U + ["send_email"],
   True).point("ACT").call(("query_orders", {"customer_id": "C-2001"})).point("ASK", note="no email address"))
(C("mu-pkg-compare", "Is bramblekit's latest registry version newer than 0.18?", U, True).point("ACT")
 .call(("get_package_version", {"package": "bramblekit"})).point("ANSWER"))
(C("mu-orders-sum", "What's the combined total of all orders for C-1042?", U, True).point("ACT")
 .call(("query_orders", {"customer_id": "C-1042"})).point(None, ans("68.15"), "sum: capability"))
(C("mu-meet-after-search", "Find the FernConf 2026 city, then schedule 'FernConf travel' with sam@team.example "
   "at 2026-11-03T10:00.", U + ["web_search"], True).point("ACT")
 .call(("web_search", {"query": "Fernhollow conference 2026 city"})).point("ACT")
 .call(("schedule_meeting", {"title": "FernConf travel", "attendees": ["sam@team.example"],
                             "start": "2026-11-03T10:00"})).point("ANSWER"))
(C("mu-translate-readme", "Read the README and translate its one-line description into French.", U, True)
 .point("ACT").call(("read_file", {"path": "README.md"}))
 .point("ACT", note="translation tool needed for exact phrasing")
 .call(("translate_text", {"text": "Tiny shopping-cart library", "target_lang": "fr"})).point("ANSWER"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(p, sort_keys=True) + "\n" for p in POINTS)
    (OUT / "e3.jsonl").write_text(data)
    (OUT / "e3.sha256").write_text(hashlib.sha256(data.encode()).hexdigest() + "  e3.jsonl\n")
    from collections import Counter

    convs = {p["conv_id"] for p in POINTS}
    multi = {p["conv_id"] for p in POINTS if p["step"] > 0}
    print(f"{len(POINTS)} points, {len(convs)} conversations, {len(multi)} multi-step")
    print(Counter(p["label"] or "capability" for p in POINTS))
    print("unseen-schema points:", sum(p["unseen_schema"] for p in POINTS))


if __name__ == "__main__":
    main()
