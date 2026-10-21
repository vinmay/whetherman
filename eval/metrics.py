"""Routing metrics. Every rate carries its denominator and a Wilson 95% interval."""

from __future__ import annotations

from typing import Any

import numpy as np

from whetherman.labeling import wilson
from whetherman.labels import IDX, LABELS


def rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(int(k), int(n))
    return {"k": int(k), "n": int(n), "rate": (k / n) if n else None, "ci95": [round(lo, 4), round(hi, 4)]}


def confusion(y: list[str], p: list[str]) -> list[list[int]]:
    m = np.zeros((4, 4), dtype=int)
    for a, b in zip(y, p, strict=True):
        m[IDX[a], IDX[b]] += 1
    return m.tolist()


def macro_f1(y: list[str], p: list[str]) -> float:
    present = sorted(set(y))
    f1 = []
    for c in present:
        tp = sum(1 for a, b in zip(y, p, strict=True) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, p, strict=True) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, p, strict=True) if a == c and b != c)
        f1.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(f1)) if f1 else 0.0


def safety(y: list[str], p: list[str]) -> dict[str, Any]:
    """False skips: truly-needed calls (y=ACT) where the gate returned non-ACT."""
    act = [b for a, b in zip(y, p, strict=True) if a == "ACT"]
    non = [(a, b) for a, b in zip(y, p, strict=True) if a != "ACT"]
    out = {"false_skip": rate(sum(b != "ACT" for b in act), len(act)),
           "false_skip_by_class": {c: rate(sum(b == c for b in act), len(act)) for c in LABELS[1:]},
           # non-ACT items the gate kept away from tools (any non-ACT decision)
           "correct_suppression": rate(sum(b != "ACT" for _, b in non), len(non)),
           "exact_non_act": rate(sum(a == b for a, b in non), len(non)),
           "intervention": rate(sum(b != "ACT" for b in p), len(p))}
    return out


def ece(probs: np.ndarray, y_idx: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    conf = probs.max(1)
    pred = probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total, diag = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            acc, c = float((pred[m] == y_idx[m]).mean()), float(conf[m].mean())
            total += m.mean() * abs(acc - c)
            diag.append({"bin": [float(lo), float(hi)], "n": int(m.sum()), "acc": acc, "conf": c})
    return float(total), diag


def risk_coverage(probs: np.ndarray, y_idx: np.ndarray) -> list[dict]:
    conf = probs.max(1)
    order = np.argsort(-conf)
    correct = (probs.argmax(1) == y_idx)[order]
    out = []
    for frac in np.linspace(0.1, 1.0, 10):
        k = max(1, int(round(frac * len(order))))
        out.append({"coverage": round(float(frac), 2), "risk": float(1 - correct[:k].mean())})
    return out


def summarize(y: list[str], p: list[str], probs: np.ndarray | None = None) -> dict[str, Any]:
    out = {"n": len(y), "accuracy": float(np.mean([a == b for a, b in zip(y, p, strict=True)])) if y else None,
           "macro_f1": macro_f1(y, p), "confusion": confusion(y, p), "labels": LABELS, **safety(y, p)}
    if probs is not None and len(y):
        yi = np.array([IDX[a] for a in y])
        out["ece"], out["reliability"] = ece(np.asarray(probs), yi)
        out["risk_coverage"] = risk_coverage(np.asarray(probs), yi)
    return out
