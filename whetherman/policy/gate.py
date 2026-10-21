"""Default-to-ACT, fail-open gate: router -> per-agent adjustment -> threshold -> decision."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from whetherman.calibrate.adjust import VectorScaling, decide, softmax
from whetherman.labels import IDX, LABELS


class Router(Protocol):
    placement: str

    def format(self, decision_input: dict[str, Any]) -> dict[str, Any]: ...
    def predict(self, formatted: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class Profile:
    agent: str
    tau: float = 1.0  # 1.0 = never intervene
    adjust: VectorScaling = field(default_factory=VectorScaling)
    backbone: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> Profile:
        d = json.loads(Path(path).read_text())
        return cls(agent=d["agent"], tau=float(d["tau"]), adjust=VectorScaling(**d["adjust"]),
                   backbone=d.get("backbone", ""), meta=d.get("meta", {}))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"agent": self.agent, "tau": self.tau,
                                          "adjust": self.adjust.to_dict(), "backbone": self.backbone,
                                          "meta": self.meta}, indent=1))


def fail_open(reason: str, t0: float, placement: str = "") -> dict[str, Any]:
    return {"decision": "ACT", "raw_label": None, "confidence": None, "probs": None,
            "latency_ms": (time.perf_counter() - t0) * 1000, "error": reason, "placement": placement}


class Gate:
    def __init__(self, router: Router, profile: Profile, timeout_ms: float = 3000) -> None:
        self.router = router
        self.profile = profile
        self.timeout_s = timeout_ms / 1000
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whetherman-router")

    def _run(self, decision_input: dict[str, Any]) -> dict[str, Any]:
        return self.router.predict(self.router.format(decision_input))

    def decide(self, decision_input: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        placement = getattr(self.router, "placement", "")
        try:
            pred = self._pool.submit(self._run, decision_input).result(timeout=self.timeout_s)
            logits = np.asarray(pred["logits"], dtype=float)
            if logits.shape != (4,) or not np.all(np.isfinite(logits)):
                return fail_open("invalid_prediction", t0, placement)
            probs = softmax(self.profile.adjust.apply(logits))
            label = decide(probs, self.profile.tau)
            k = int(probs.argmax())
        except FutTimeout:
            return fail_open("timeout", t0, placement)
        except Exception as e:  # any classifier failure -> ACT
            return fail_open(f"error:{type(e).__name__}", t0, placement)
        return {"decision": label, "raw_label": LABELS[k], "confidence": float(probs[k]),
                "act_prob": float(probs[IDX["ACT"]]), "probs": probs.round(5).tolist(),
                "tau": self.profile.tau, "latency_ms": (time.perf_counter() - t0) * 1000,
                "error": None, "placement": placement}
