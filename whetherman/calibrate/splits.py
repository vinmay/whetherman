"""Deterministic, label-stratified 60/40 adaptation/threshold split of usable probe labels."""

from __future__ import annotations

import hashlib


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def adapt_threshold_split(items: list[dict], frac: float = 0.6, key: str = "id",
                          label: str = "label") -> tuple[list[dict], list[dict]]:
    adapt, thresh = [], []
    for lab in sorted({x[label] for x in items}):
        group = sorted((x for x in items if x[label] == lab), key=lambda x: _h(x[key]))
        k = int(round(frac * len(group)))
        adapt += group[:k]
        thresh += group[k:]
    return adapt, thresh
