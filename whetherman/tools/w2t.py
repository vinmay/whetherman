"""Tools compatible with When2Tool task instructions (our own implementations and schemas).

Only standard algorithms are implemented. When2Tool's custom hash/cipher tasks are excluded
from our task pool because their algorithms exist only in the benchmark's own code.
"""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import math
import operator
import re
import resource
import statistics
import subprocess
import sys
import tempfile
from fractions import Fraction
from typing import Any

from whetherman.tools.registry import tool

S = {"type": "string"}
I = {"type": "integer"}  # noqa: E741
N = {"type": "number"}
A = {"type": "array", "items": {}}

# ---------------------------------------------------------------- safe arithmetic

_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil,
    "log": math.log, "log10": math.log10, "exp": math.exp, "sin": math.sin, "cos": math.cos,
    "tan": math.tan, "min": min, "max": max, "factorial": math.factorial, "pow": pow,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def safe_eval(expr: str) -> int | float:
    expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/").replace(",", "")

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10_000:
                raise ValueError("exponent too large")
            return _BIN[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
            return _UN[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        raise ValueError(f"unsupported expression element: {ast.dump(node)[:60]}")

    val = ev(ast.parse(expr.strip(), mode="eval"))
    if isinstance(val, float) and val.is_integer() and abs(val) < 1e15:
        return int(val)
    return val


@tool("evaluate_expression", "Evaluate an arithmetic expression exactly (big integers supported).",
      {"expression": S})
def evaluate_expression(env, expression: str):
    val = safe_eval(expression)
    env.state["last_result"] = val
    return {"result": val}


@tool("get_last_result", "Return the result of the previous evaluate_expression call.", {}, [])
def get_last_result(env):
    return {"result": env.state.get("last_result")}


@tool("clear_last_result", "Clear the stored previous result.", {}, [])
def clear_last_result(env):
    env.state.pop("last_result", None)
    return {"cleared": True}


# ---------------------------------------------------------------- code execution


def _limits() -> None:  # pragma: no cover - runs in the child process
    resource.setrlimit(resource.RLIMIT_AS, (512 * 2**20, 512 * 2**20))
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))


@tool("run_code", "Execute a short Python 3 program and return its stdout.", {"code": S})
def run_code(env, code: str):
    with tempfile.TemporaryDirectory() as d:
        p = subprocess.run(
            [sys.executable, "-I", "-c", code], cwd=d, capture_output=True, text=True,
            timeout=10, preexec_fn=_limits,
        )
    out = {"stdout": p.stdout[-4000:]}
    if p.returncode:
        out["stderr"] = p.stderr[-1000:]
    return out


# ---------------------------------------------------------------- counting


@tool("combination", "Number of ways to choose k items from n (n choose k).", {"n": I, "k": I})
def combination(env, n: int, k: int):
    return {"result": math.comb(int(n), int(k))}


@tool("permutation", "Number of ordered arrangements of k items from n.", {"n": I, "k": I})
def permutation(env, n: int, k: int):
    return {"result": math.perm(int(n), int(k))}


@tool("factorial", "n factorial.", {"n": I})
def factorial(env, n: int):
    if int(n) > 5000:
        raise ValueError("n too large")
    return {"result": math.factorial(int(n))}


# ---------------------------------------------------------------- dates


def _date(s: str) -> dt.date:
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%d %B %Y", "%m/%d/%Y"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"unrecognized date {s!r}; use YYYY-MM-DD")


@tool("date_diff", "Absolute number of days between two dates (YYYY-MM-DD).",
      {"date1": S, "date2": S})
def date_diff(env, date1: str, date2: str):
    return {"days": abs((_date(date2) - _date(date1)).days)}


@tool("date_add", "Add a number of days to a date (YYYY-MM-DD).", {"date": S, "days": I})
def date_add(env, date: str, days: int):
    return {"result": (_date(date) + dt.timedelta(days=int(days))).isoformat()}


@tool("day_of_week", "Day of the week for a date (YYYY-MM-DD).", {"date": S})
def day_of_week(env, date: str):
    return {"day": _date(date).strftime("%A")}


# ---------------------------------------------------------------- encodings

MORSE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.", "G": "--.",
    "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..", "M": "--", "N": "-.",
    "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.", "S": "...", "T": "-", "U": "..-",
    "V": "...-", "W": ".--", "X": "-..-", "Y": "-.--", "Z": "--..", "0": "-----",
    "1": ".----", "2": "..---", "3": "...--", "4": "....-", "5": ".....", "6": "-....",
    "7": "--...", "8": "---..", "9": "----.",
}
UNMORSE = {v: k for k, v in MORSE.items()}


def _caesar(msg: str, shift: int) -> str:
    out = []
    for c in msg:
        if c.isalpha():
            base = ord("A") if c.isupper() else ord("a")
            out.append(chr((ord(c) - base + shift) % 26 + base))
        else:
            out.append(c)
    return "".join(out)


@tool("encode", "Encode a message. encoding_type: 'morse' or 'caesar' (key = shift).",
      {"message": S, "encoding_type": S, "key": S}, ["message", "encoding_type"])
def encode(env, message: str, encoding_type: str, key: str | None = None):
    t = encoding_type.strip().lower()
    if t == "morse":
        return {"encoded": " / ".join(" ".join(MORSE.get(c, "?") for c in w)
                                      for w in message.upper().split())}
    if t == "caesar":
        return {"encoded": _caesar(message, int(key or 0))}
    raise ValueError(f"unsupported encoding_type {encoding_type!r}")


@tool("decode", "Decode a message. encoding_type: 'morse' or 'caesar' (key = shift).",
      {"encoded_message": S, "encoding_type": S, "key": S}, ["encoded_message", "encoding_type"])
def decode(env, encoded_message: str, encoding_type: str, key: str | None = None):
    t = encoding_type.strip().lower()
    if t == "morse":
        words = encoded_message.strip().split("/")
        return {"decoded": " ".join("".join(UNMORSE.get(s, "?") for s in w.split()) for w in words)}
    if t == "caesar":
        return {"decoded": _caesar(encoded_message, -int(key or 0))}
    raise ValueError(f"unsupported encoding_type {encoding_type!r}")


@tool("compute_hash", "Hex digest of a string. algorithm: md5, sha1 or sha256.",
      {"input_string": S, "algorithm": S})
def compute_hash(env, input_string: str, algorithm: str):
    algo = algorithm.strip().lower().replace("-", "")
    if algo not in ("md5", "sha1", "sha256"):
        raise ValueError(f"unsupported algorithm {algorithm!r}")
    return {"hash": hashlib.new(algo, input_string.encode()).hexdigest()}


# ---------------------------------------------------------------- corpus lookups


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower())) - {"the", "a", "of", "in", "is", "was", "what"}


def _best(query: str, items: list[tuple[str, str]], k: int = 1) -> list[tuple[str, str]]:
    q = _words(query)
    scored = sorted(items, key=lambda kv: -len(q & _words(kv[0] + " " + kv[1][:200])))
    return scored[:k]


@tool("lookup_rule", "Look up a rule in the game-rules reference.", {"query": S})
def lookup_rule(env, query: str):
    corpus = env.context.get("corpus") or {}
    hits = _best(query, list(corpus.items()))
    return {"result": hits[0][1] if hits else "no matching rule"}


@tool("lookup_year", "Look up the year of a historical event in the reference.", {"event": S})
def lookup_year(env, event: str):
    corpus = env.context.get("corpus") or {}
    hits = _best(event, list(corpus.items()))
    return {"result": hits[0][1] if hits else "no matching event"}


@tool("search_corpus", "Search the local document corpus; returns ids, titles and snippets.",
      {"query": S, "top_k": I}, ["query"])
def search_corpus(env, query: str, top_k: int = 3):
    docs = env.context.get("corpus") or []
    items = [(d["id"], d["title"] + " " + d["text"]) for d in docs]
    hits = _best(query, items, int(top_k or 3))
    by_id = {d["id"]: d for d in docs}
    return {"results": [{"doc_id": i, "title": by_id[i]["title"], "snippet": by_id[i]["text"][:200]}
                        for i, _ in hits]}


@tool("read_doc", "Read the full text of a corpus document by id.", {"doc_id": S})
def read_doc(env, doc_id: str):
    for d in env.context.get("corpus") or []:
        if d["id"] == doc_id:
            return {"doc_id": doc_id, "title": d["title"], "text": d["text"]}
    raise ValueError(f"no document {doc_id!r}")


# ---------------------------------------------------------------- matrices


def _det(m: list[list[Fraction]]) -> Fraction:
    m = [row[:] for row in m]
    n, det = len(m), Fraction(1)
    for c in range(n):
        p = next((r for r in range(c, n) if m[r][c] != 0), None)
        if p is None:
            return Fraction(0)
        if p != c:
            m[c], m[p] = m[p], m[c]
            det = -det
        det *= m[c][c]
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            m[r] = [a - f * b for a, b in zip(m[r], m[c], strict=True)]
    return det


def _num(x: Fraction) -> int | float:
    return int(x) if x.denominator == 1 else float(x)


@tool("matrix_determinant", "Exact determinant of a square matrix.", {"matrix": A})
def matrix_determinant(env, matrix: list[list[float]]):
    return {"result": _num(_det([[Fraction(str(v)) for v in row] for row in matrix]))}


@tool("matrix_trace", "Trace of a square matrix.", {"matrix": A})
def matrix_trace(env, matrix: list[list[float]]):
    return {"result": sum(matrix[i][i] for i in range(len(matrix)))}


@tool("matrix_multiply", "Product of two matrices.", {"matrix_a": A, "matrix_b": A})
def matrix_multiply(env, matrix_a, matrix_b):
    return {"result": [[sum(a * b for a, b in zip(row, col, strict=True)) for col in zip(*matrix_b,
                                                                                      strict=True)]
                       for row in matrix_a]}


# ---------------------------------------------------------------- primes


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


@tool("is_prime", "Whether n is prime.", {"n": I})
def is_prime(env, n: int):
    return {"result": _is_prime(int(n))}


@tool("nth_prime", "The n-th prime number (1-indexed).", {"n": I})
def nth_prime(env, n: int):
    n = int(n)
    if not 1 <= n <= 200_000:
        raise ValueError("n out of range")
    limit = max(15, int(n * (math.log(n) + math.log(math.log(n + 2)) + 2)))
    sieve = bytearray([1]) * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))
    primes = [i for i, v in enumerate(sieve) if v]
    return {"result": primes[n - 1]}


@tool("factorize", "Prime factorization of n, as 'p1^e1 * p2^e2' (exponent 1 omitted).", {"n": I})
def factorize(env, n: int):
    n, f, p = int(n), {}, 2
    if n < 2:
        raise ValueError("n must be >= 2")
    while p * p <= n:
        while n % p == 0:
            f[p] = f.get(p, 0) + 1
            n //= p
        p += 1 if p == 2 else 2
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return {"result": " * ".join(f"{q}^{e}" if e > 1 else str(q) for q, e in sorted(f.items()))}


# ---------------------------------------------------------------- regex


@tool("regex_match", "Run a Python regex. operation: findall, search, match or fullmatch.",
      {"pattern": S, "text": S, "operation": S}, ["pattern", "text"])
def regex_match(env, pattern: str, text: str, operation: str = "findall"):
    op = (operation or "findall").lower()
    if op == "findall":
        return {"result": repr(re.findall(pattern, text))}
    m = getattr(re, op)(pattern, text)
    return {"result": m.group(0) if m else None}


# ---------------------------------------------------------------- schedule


def _mins(t: str) -> int:
    h, m = t.strip().split(":")
    return int(h) * 60 + int(m)


def _hhmm(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


@tool("find_free_slots",
      "Free gaps between busy intervals ([[start,end],...] in HH:MM) within the day window, "
      "keeping gaps of at least min_duration minutes.",
      {"busy": A, "day_start": S, "day_end": S, "min_duration": I})
def find_free_slots(env, busy, day_start: str, day_end: str, min_duration: int):
    iv = sorted((_mins(a), _mins(b)) for a, b in busy)
    cur, end, out = _mins(day_start), _mins(day_end), []
    for a, b in iv:
        if a > cur and min(a, end) - cur >= int(min_duration):
            out.append([_hhmm(cur), _hhmm(min(a, end))])
        cur = max(cur, b)
    if end - cur >= int(min_duration):
        out.append([_hhmm(cur), _hhmm(end)])
    return {"free_slots": out, "count": len(out)}


@tool("check_conflict", "Whether any of the given [start,end] HH:MM intervals overlap.",
      {"intervals": A})
def check_conflict(env, intervals):
    iv = sorted((_mins(a), _mins(b)) for a, b in intervals)
    return {"conflict": any(iv[i + 1][0] < iv[i][1] for i in range(len(iv) - 1))}


# ---------------------------------------------------------------- statistics

_STATS = {
    "mean": statistics.fmean, "median": statistics.median, "mode": statistics.mode,
    "sum": sum, "min": min, "max": max, "range": lambda d: max(d) - min(d),
    "variance": statistics.pvariance, "population_variance": statistics.pvariance,
    "sample_variance": statistics.variance, "std": statistics.pstdev,
    "population_std": statistics.pstdev, "stdev": statistics.pstdev,
    "sample_std": statistics.stdev,
}


@tool("compute_stat",
      "Compute a statistic. stat_type: mean, median, mode, sum, min, max, range, variance "
      "(population), sample_variance, std (population), sample_std, percentile "
      "(with percentile_value), correlation (Pearson, with data2).",
      {"data": A, "stat_type": S, "data2": A, "percentile_value": N}, ["data", "stat_type"])
def compute_stat(env, data, stat_type: str, data2=None, percentile_value=None):
    t = stat_type.strip().lower().replace(" ", "_")
    data = [float(x) for x in data]
    if t in ("correlation", "pearson", "pearson_correlation"):
        return {"result": statistics.correlation(data, [float(x) for x in data2 or []])}
    if t == "percentile":
        q = statistics.quantiles(data, n=100, method="inclusive")
        p = int(percentile_value or 50)
        return {"result": statistics.median(data) if p == 50 else q[max(min(p, 99), 1) - 1]}
    if t not in _STATS:
        raise ValueError(f"unsupported stat_type {stat_type!r}")
    return {"result": _STATS[t](data)}


@tool("describe", "Summary statistics (count, mean, median, population std, min, max).",
      {"data": A})
def describe(env, data):
    d = [float(x) for x in data]
    return {"count": len(d), "mean": statistics.fmean(d), "median": statistics.median(d),
            "std": statistics.pstdev(d), "min": min(d), "max": max(d)}


ENV_TOOLS: dict[str, list[str]] = {
    "CalculatorEnv": ["evaluate_expression", "get_last_result", "clear_last_result"],
    "CodeExecutorEnv": ["run_code"],
    "CountingEnv": ["combination", "permutation", "factorial"],
    "DateTimeEnv": ["date_diff", "date_add", "day_of_week"],
    "DecodingEnv": ["encode", "decode"],
    "GameRuleEnv": ["lookup_rule"],
    "HashEnv": ["compute_hash"],
    "HistoricalYearEnv": ["lookup_year"],
    "MatrixEnv": ["matrix_determinant", "matrix_multiply", "matrix_trace"],
    "PrimeEnv": ["is_prime", "nth_prime", "factorize"],
    "RegexMatchEnv": ["regex_match"],
    "RetrieverEnv": ["search_corpus", "read_doc"],
    "ScheduleEnv": ["find_free_slots", "check_conflict"],
    "StatisticsEnv": ["compute_stat", "describe"],
}
