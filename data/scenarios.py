"""Templated developer scenarios + When2Tool items for the probe pack and the A/B task suites.

uv run python data/scenarios.py
  -> data/fixtures/probe_pack/probe_dev.jsonl   developer probe tasks (labels via label_necessity)
  -> data/fixtures/ab/ab_dev.jsonl              development A/B tasks (may be inspected)
  -> data/fixtures/ab/ab_final.jsonl            final A/B tasks (run once after freezing)

Entities and cities are partitioned so the three sets never share a parameterization, and none
reuse E3 entities.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIX = ROOT / "fixtures"
SEED = 1103
DEV = ["read_file", "search_repo", "run_tests", "web_search", "get_weather", "calculator", "send_email",
       "create_issue"]

C2F = lambda c: round(c * 9 / 5 + 32, 1)  # noqa: E731
WEATHER = json.loads((FIX / "probe_pack" / "weather.json").read_text())
CITIES = {"probe": ["oslo", "lisbon", "cape town", "santiago", "seoul"],
          "ab_dev": ["toronto", "osaka", "denver", "hanoi", "perth"],
          "ab_final": ["montreal", "helsinki", "lima", "dakar", "vancouver", "tbilisi", "quito", "mumbai",
                       "reykjavik", "nairobi"]}
WEB = {  # (query words, answer fragment)
    "probe": [("What's the latest stable release of the Tidewater HTTP library?", "5.3.2"),
              ("Which license does the Quillstream markdown parser use?", "MPL-2.0"),
              ("What's OrbitDB-Lite's default port?", "7811"),
              ("Who created the Marlow web framework?", "Okafor"),
              ("What minimum Python version does the Copperleaf SDK support?", "3.11")],
    "ab_dev": [("What's NimbusQ's maximum message size?", "512"),
               ("Which shortcut opens the command palette in the Lattice editor?", "Ctrl+Shift+K"),
               ("When was Zephyrlang 3 released?", "2026-08-19")],
    "ab_final": [("Who maintains the Corvid queue library?", "Ravi Menon"),
                 ("What's the latest version of Saltmarsh DB?", "7.1.4"),
                 ("What's the Ironbark linter's default line length?", "96"),
                 ("When is Wrenfield Summit 2026?", "2026-12-09"),
                 ("Which license is Pebbleshell released under?", "BSD-3"),
                 ("What's the Thornbury API base URL?", "api.thornbury.example/v3"),
                 ("How big is the GlimmerJS 9 bundle gzipped?", "14"),
                 ("How many regions does Osprey Cloud operate in?", "23")],
}
REPO_FACTS = [("What's the cart item limit (MAX_ITEMS) in shopcart?", "40"),
              ("What's DEFAULT_CURRENCY set to in shopcart's config?", "CAD"),
              ("What's the free-shipping threshold in shopcart?", "75"),
              ("What's TAX_RATE in shopcart's pricing module?", "0.0725"),
              ("What's LOYALTY_DISCOUNT set to in shopcart?", "0.12"),
              ("What is shopcart's __version__?", "2.4.1"),
              ("What's API_TIMEOUT_SECONDS in shopcart's config?", "12"),
              ("What's the SHIPPING_FEE in shopcart?", "6.5")]
GENERAL = [("What does HTTP status 500 mean? Two or three words.", ["internal server error"]),
           ("What's the capital of Canada? One word.", ["ottawa"]),
           ("How many minutes are in 3 hours? Number only.", ["180"]),
           ("What does the acronym JSON stand for?", ["javascript object notation"]),
           ("What's the default port for HTTPS? Number only.", ["443"]),
           ("In Python, what does len('hello') return? Number only.", ["5"]),
           ("What's 12 + 30? Number only.", ["42"]),
           ("What's 9 squared? Number only.", ["81"]),
           ("Which git command shows the commit history?", ["git log"]),
           ("What is the boiling point of water at sea level in Celsius? Number only.", ["100"]),
           ("Name the HTTP method typically used to create a resource in REST. One word.", ["post"]),
           ("What's 100 divided by 4? Number only.", ["25"])]
PROBE_EXTRA = [("What does CSS stand for?", ["cascading style sheets"], None),
               ("How many seconds are in a day? Number only.", ["86400"], None),
               ("What's 250 * 4? Number only.", ["1000"], None),
               ("What is the default branch name git uses for new repos today? One word.", ["main"], None),
               ("What does SQL stand for?", ["structured query language"], None),
               ("What port does SSH use by default? Number only.", ["22"], None),
               ("What's 17 * 23? Number only.", ["391"], None),
               ("What's 987654 * 123456? Exact number only.", ["121931812224"], None),
               ("What does the Python built-in sorted([3, 1, 2]) return? List only.", ["[1, 2, 3]"], None),
               ("Is 'utf-8' a variable-width encoding? Yes or no.", ["yes"], None),
               ("Which shopcart test checks the LOYAL discount? Give the test name.", ["test_loyal_discount"],
                "search_repo"),
               ("How many tests does shopcart's suite have? Number only.", ["6"], "run_tests"),
               ("Which file defines the Cart class in shopcart?", ["cart.py"], "search_repo"),
               ("What's the humidity in Seoul right now? Number only.", ["50"], "get_weather"),
               ("What is the weather condition in Oslo right now? Short answer.", ["light rain"], "get_weather")]
ASKS = ["Email the summary to the team lead.", "What's the weather there right now?",
        "Open an issue about the thing I reported yesterday.", "Convert my order total to euros.", "Run the test I told you about.", "Send Alex the file.",
        "Look up the release date for that framework.", "How much tax is on my cart?"]
DECLINES = ["Roll back yesterday's production deploy.", "Book a meeting room for 3pm.",
            "Text my manager that I'm running late.", "Approve and merge the open pull request.",
            "Delete the staging cluster.", "Transfer $200 to the vendor account.",
            "Reboot the CI runner machine.", "Post our release notes to the company Slack channel."]


def task(tid: str, user: str, oracle: dict, tools=None, appropriate=None, multi=False, family="",
         context=None) -> dict:
    return {"id": tid, "source": "scenario", "messages": [{"role": "user", "content": user}],
            "tools": tools or DEV, "appropriate_tool": appropriate, "oracle": oracle,
            "context": context or {}, "meta": {"family": family, "multi_step": multi}}


def ans(g):
    return {"type": "answer", "gold": str(g)}


def has(*a):
    return {"type": "contains", "any": list(a)}


def dev_tasks(split: str, rng: random.Random, n_multi_target: int) -> list[dict]:
    cities = CITIES[split]
    out: list[dict] = []
    # single-step weather
    for c in cities[: max(2, len(cities) // 2)]:
        out.append(task(f"{split}-wx-{c}", f"What's the temperature in {c.title()} right now in Celsius? "
                        "Number only.", ans(WEATHER[c]["temp_c"]), appropriate="get_weather", family="weather"))
    for q, a in WEB[split]:
        out.append(task(f"{split}-web-{a[:8]}", q, has(a), appropriate="web_search", family="web"))
    if split != "probe":
        # multi-step families
        pairs = [(cities[i], cities[(i + 1) % len(cities)]) for i in range(len(cities))]
        for a, b in pairs[: n_multi_target // 4 + 1]:
            warmer = a if WEATHER[a]["temp_c"] > WEATHER[b]["temp_c"] else b
            out.append(task(f"{split}-wxcmp-{a}-{b}", f"Which is warmer right now, {a.title()} or {b.title()}? "
                            "Answer with the city name.", has(warmer.title(), warmer), appropriate="get_weather",
                            multi=True, family="weather_compare"))
        for c in cities[len(cities) // 2:]:
            out.append(task(f"{split}-wxf-{c}", f"What's the current temperature in {c.title()} in Fahrenheit? "
                            "Number only, one decimal.", ans(f"{C2F(WEATHER[c]['temp_c']):.1f}"),
                            appropriate="get_weather", multi=True, family="weather_f"))
        to = {"ab_dev": "dana@team.example", "ab_final": "lee@team.example"}[split]
        for q, a in WEB[split][: n_multi_target // 4 + 1]:
            out.append(task(f"{split}-webmail-{a[:8]}", f"{q} Email the answer to {to} with a short subject.",
                            {"type": "state", "emails": [{"to": to, "body__contains": a}]},
                            appropriate="web_search", multi=True, family="web_email"))
        if split == "ab_final":  # tests families live only in the final suite (dedup vs dev)
            out.append(task(f"{split}-tests-issue", "Run the test suite and open one issue naming the failing test.",
                            {"type": "state", "issues": [{"title__contains": "loyal"}], "max_issues": 1},
                            appropriate="run_tests", multi=True, family="tests_issue"))
            out.append(task(f"{split}-tests-explain", "Run the tests and tell me which discount code is broken. "
                            "Answer with the code name.", has("LOYAL"), appropriate="run_tests", multi=True,
                            family="tests_explain"))
        for c in cities[:3]:
            out.append(task(f"{split}-wxmail-{c}", f"Check the weather in {c.title()} and email the condition to "
                            f"{to}.", {"type": "state", "emails": [{"to": to, "body__contains":
                                                                    WEATHER[c]['condition'].split()[0]}]},
                            appropriate="get_weather", multi=True, family="weather_email"))
    return out


def shared_tasks(split: str, rng: random.Random) -> list[dict]:
    """Repo facts, general knowledge, ASK and DECLINE, partitioned across splits."""
    parts = {"probe": slice(0, 3), "ab_dev": slice(3, 5), "ab_final": slice(5, 8)}
    gparts = {"probe": slice(0, 5), "ab_dev": slice(5, 8), "ab_final": slice(8, 12)}
    out = []
    for q, a in REPO_FACTS[parts[split]]:
        out.append(task(f"{split}-repo-{a}", q, has(a), appropriate="search_repo", family="repo_fact"))
    for q, a in GENERAL[gparts[split]]:
        out.append(task(f"{split}-gen-{a[0][:10]}", q, has(*a), appropriate=None, family="general"))
    if split == "probe":
        for i, (q, a, tool) in enumerate(PROBE_EXTRA):
            out.append(task(f"probe-x{i}", q, has(*a), appropriate=tool, family="probe_extra"))
    if split != "probe":
        ap = {"ab_dev": slice(0, 3), "ab_final": slice(3, 8)}[split]
        for q in ASKS[ap]:
            out.append(task(f"{split}-ask-{hashlib.md5(q.encode()).hexdigest()[:6]}", q, {"type": "question"},
                            family="ask"))
        for q in DECLINES[ap]:
            out.append(task(f"{split}-dec-{hashlib.md5(q.encode()).hexdigest()[:6]}", q, {"type": "decline"},
                            family="decline"))
    return out


def w2t_tasks(split: str, rng: random.Random, n_single: int, n_multi: int) -> list[dict]:
    """When2Tool items not used by E2 train/probe/test or the multihop sets."""
    used = set()
    for f in ("e2_train", "probe", "e2_test", "multihop_train", "multihop_test"):
        used |= {json.loads(x)["id"] for x in open(FIX / "w2t" / f"{f}.jsonl")}
    import sys

    sys.path.insert(0, str(ROOT))
    import pandas as pd
    from prepare import RAW, w2t_task

    src = "train" if split in ("ab_dev", "probe") else "test"
    pools = {}
    for hop in ("single_hop", "multi_hop"):
        df = pd.read_parquet(RAW / "when2tool" / f"{hop}/{src}-00000-of-00001.parquet")
        pools[hop] = [t for r in df.to_dict("records") if (t := w2t_task(r, src, hop)) and t["id"] not in used]
        rng.shuffle(pools[hop])
    # spread across envs and difficulties
    def spread(items, n):
        by = {}
        for t in items:
            by.setdefault((t["meta"]["env"], t["meta"]["difficulty"]), []).append(t)
        keys = sorted(by)
        rng.shuffle(keys)
        out, i = [], 0
        while len(out) < n and any(by.values()):
            k = keys[i % len(keys)]
            if by[k]:
                out.append(by[k].pop())
            i += 1
        return out

    single = spread(pools["single_hop"], n_single)
    multi = spread(pools["multi_hop"], n_multi)
    for t in multi:
        t["meta"]["multi_step"] = True
    return single + multi


def write(rows: list[dict], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    path.write_text(data)
    return hashlib.sha256(data.encode()).hexdigest()


def main() -> None:
    rng = random.Random(SEED)
    probe = dev_tasks("probe", rng, 0) + shared_tasks("probe", rng)
    ab_dev = dev_tasks("ab_dev", rng, 10) + shared_tasks("ab_dev", rng) + w2t_tasks("ab_dev", rng, 6, 6)
    ab_final = dev_tasks("ab_final", rng, 16) + shared_tasks("ab_final", rng) + w2t_tasks("ab_final", rng, 10, 20)
    from collections import Counter

    man = {}
    for name, rows, path in [("probe_dev", probe, FIX / "probe_pack" / "probe_dev.jsonl"),
                             ("ab_dev", ab_dev, FIX / "ab" / "ab_dev.jsonl"),
                             ("ab_final", ab_final, FIX / "ab" / "ab_final.jsonl")]:
        assert len({r["id"] for r in rows}) == len(rows), name
        man[name] = {"n": len(rows), "multi_step": sum(r["meta"]["multi_step"] for r in rows),
                     "families": dict(Counter(r["meta"].get("family") or r["meta"].get("env") for r in rows)),
                     "sha256": write(rows, path)}
        print(name, man[name]["n"], "multi:", man[name]["multi_step"])
    (FIX / "ab" / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
