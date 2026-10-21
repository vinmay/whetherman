"""Developer-agent tools for E3, the probe pack and the A/B suite.

read_file / search_repo / run_tests act on a bundled fixture repo; web_search and get_weather
replay recorded fixture responses; send_email and create_issue only change simulated state.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from functools import cache
from pathlib import Path

from whetherman.tools.registry import tool
from whetherman.tools.w2t import safe_eval

DEFAULT_PACK = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "probe_pack"
S = {"type": "string"}


def _root(env) -> Path:
    return Path(env.context.get("pack_root") or DEFAULT_PACK)


def _repo(env) -> Path:
    return _root(env) / "fixture_repo"


@cache
def _fixture(path: str) -> dict:
    return json.loads(Path(path).read_text())


@tool("read_file", "Read a file from the project repository (path relative to repo root).",
      {"path": S})
def read_file(env, path: str):
    repo = _repo(env).resolve()
    p = (repo / path.lstrip("/")).resolve()
    if repo not in p.parents and p != repo:
        raise ValueError("path escapes the repository")
    if not p.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    return {"path": path, "content": p.read_text()[:6000]}


@tool("search_repo", "Search the project repository for a regex or literal; returns matching "
      "lines as path:line: text.", {"query": S})
def search_repo(env, query: str):
    repo = _repo(env)
    try:
        rx = re.compile(query, re.IGNORECASE)
    except re.error:
        rx = re.compile(re.escape(query), re.IGNORECASE)
    hits = []
    for f in sorted(repo.rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts and f.suffix in {".py", ".md", ".toml",
                                                                          ".txt", ".cfg"}:
            for i, line in enumerate(f.read_text().splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{f.relative_to(repo)}:{i}: {line.strip()}")
    return {"matches": hits[:30], "total": len(hits)}


@tool("run_tests", "Run the project's test suite (optionally one test file or node id) and "
      "return the pytest summary.", {"target": S}, [])
def run_tests(env, target: str = ""):
    with tempfile.TemporaryDirectory() as d:
        work = Path(d) / "repo"
        shutil.copytree(_repo(env), work)
        args = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-header"]
        if target:
            if ".." in target or target.startswith("/"):
                raise ValueError("invalid target")
            args.append(target)
        p = subprocess.run(args, cwd=work, capture_output=True, text=True, timeout=60)
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    return {"exit_code": p.returncode, "output": "\n".join(lines[-25:])}


@tool("web_search", "Search the web; returns titles and snippets.", {"query": S})
def web_search(env, query: str):
    fx = _fixture(str(_root(env) / "web_search.json"))
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    best, score = None, 0
    for entry in fx["entries"]:
        s = len(q & set(entry["keywords"]))
        if s > score:
            best, score = entry, s
    if best is None or score < best.get("min_overlap", 2):
        return {"results": []}
    return {"results": best["results"]}


@tool("get_weather", "Current weather for a city.", {"city": S})
def get_weather(env, city: str):
    fx = _fixture(str(_root(env) / "weather.json"))
    key = city.strip().lower().split(",")[0]
    if key not in fx:
        raise ValueError(f"no weather data for {city!r}")
    return {"city": city, **fx[key]}


@tool("calculator", "Evaluate an arithmetic expression.", {"expression": S})
def calculator(env, expression: str):
    return {"result": safe_eval(expression)}


@tool("send_email", "Send an email.", {"to": S, "subject": S, "body": S})
def send_email(env, to: str, subject: str, body: str):
    env.state["emails"].append({"to": to, "subject": subject, "body": body})
    return {"status": "sent", "message_id": f"msg_{len(env.state['emails'])}"}


@tool("create_issue", "Create an issue in the project tracker.",
      {"title": S, "body": S, "labels": {"type": "array", "items": S}}, ["title", "body"])
def create_issue(env, title: str, body: str, labels: list[str] | None = None):
    env.state["issues"].append({"title": title, "body": body, "labels": labels or []})
    return {"status": "created", "number": 100 + len(env.state["issues"])}


DEV_TOOLS = ["read_file", "search_repo", "run_tests", "web_search", "get_weather", "calculator",
             "send_email", "create_issue"]
