"""Per-agent adjustment layers and threshold selection.

FeatureHead (default): an L2-regularized logistic head on the frozen backbone's last hidden state giving
P(ACT); the remaining mass is split by the backbone's own ANSWER/ASK/DECLINE distribution.
VectorScaling (ablation): per-class scale + bias on the backbone's four logits, regularized toward identity.
Both expose probs(logits, hidden)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from whetherman.labeling import wilson
from whetherman.labels import IDX, LABELS


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class VectorScaling:
    scale: list[float] = field(default_factory=lambda: [1.0] * 4)
    bias: list[float] = field(default_factory=lambda: [0.0] * 4)

    def apply(self, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits) * np.asarray(self.scale) + np.asarray(self.bias)

    def probs(self, logits: np.ndarray, hidden: np.ndarray | None = None) -> np.ndarray:
        return softmax(self.apply(logits))

    def to_dict(self) -> dict:
        return {"kind": "vector_scaling", **asdict(self)}


@dataclass
class FeatureHead:
    mean: list[float]
    std: list[float]
    w: list[float]
    b: float
    c: float = 0.01

    def p_act(self, hidden: np.ndarray) -> np.ndarray:
        x = (np.asarray(hidden, dtype=float) - np.asarray(self.mean)) / np.asarray(self.std)
        return 1 / (1 + np.exp(-(x @ np.asarray(self.w) + self.b)))

    def probs(self, logits: np.ndarray, hidden: np.ndarray | None = None) -> np.ndarray:
        if hidden is None:
            raise ValueError("feature_head needs the backbone hidden state")
        pa = self.p_act(hidden)
        q = softmax(np.asarray(logits, dtype=float))[..., 1:]
        q = q / q.sum(-1, keepdims=True)
        return np.concatenate([np.asarray(pa)[..., None], (1 - np.asarray(pa))[..., None] * q], -1)

    def to_dict(self) -> dict:
        return {"kind": "feature_head", **asdict(self)}


def adjust_from_dict(d: dict) -> VectorScaling | FeatureHead:
    d = dict(d)
    kind = d.pop("kind", "vector_scaling")
    return FeatureHead(**d) if kind == "feature_head" else VectorScaling(**d)


def fit_feature_head(hidden: np.ndarray, is_act: np.ndarray, c: float = 0.01, iters: int = 50) -> FeatureHead:
    """Minimize c * sum_i s_i * logloss_i + ||w||^2 / 2 (intercept unpenalized; s_i balanced class weights)
    by Newton's method on standardized features."""
    h = np.asarray(hidden, dtype=float)
    y = np.asarray(is_act, dtype=float)
    mean, std = h.mean(0), h.std(0) + 1e-6
    x = np.hstack([(h - mean) / std, np.ones((len(y), 1))])
    n_pos, n = y.sum(), len(y)
    sw = np.where(y == 1, n / (2 * max(n_pos, 1)), n / (2 * max(n - n_pos, 1)))
    reg = np.ones(x.shape[1])
    reg[-1] = 0.0
    beta = np.zeros(x.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(x @ beta)))
        g = c * x.T @ (sw * (p - y)) + reg * beta
        hess = c * (x.T * (sw * p * (1 - p))) @ x + np.diag(reg + 1e-9)
        step = np.linalg.solve(hess, g)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    return FeatureHead(mean=mean.round(6).tolist(), std=std.round(6).tolist(), w=beta[:-1].round(8).tolist(),
                       b=float(beta[-1]), c=c)


def fit_vector_scaling(logits: np.ndarray, y: np.ndarray, l2: float = 1.0, lr: float = 0.05,
                       steps: int = 3000, class_weight: np.ndarray | None = None) -> VectorScaling:
    """Minimize weighted NLL + l2 * (||s-1||^2 + ||b||^2) / n by gradient descent."""
    z = np.asarray(logits, dtype=float)
    y = np.asarray(y, dtype=int)
    n = len(y)
    if n == 0:
        return VectorScaling()
    w = np.ones(n) if class_weight is None else np.asarray(class_weight)[y]
    w = w / w.sum()
    s, b = np.ones(4), np.zeros(4)
    onehot = np.eye(4)[y]
    m_s, m_b, v_s, v_b = (np.zeros(4) for _ in range(4))
    for t in range(1, steps + 1):  # Adam
        p = softmax(z * s + b)
        g = (p - onehot) * w[:, None]
        gs = (g * z).sum(0) + 2 * l2 * (s - 1) / n
        gb = g.sum(0) + 2 * l2 * b / n
        for m, v, gr, par in ((m_s, v_s, gs, s), (m_b, v_b, gb, b)):
            m[:] = 0.9 * m + 0.1 * gr
            v[:] = 0.999 * v + 0.001 * gr * gr
            par -= lr * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
    return VectorScaling(scale=s.round(6).tolist(), bias=b.round(6).tolist())


def decide(probs: np.ndarray, tau: float) -> str:
    """Default to ACT; override only if the top non-ACT class is the argmax and exceeds tau."""
    p = np.asarray(probs)
    k = int(p.argmax())
    if k != IDX["ACT"] and p[k] > tau:
        return LABELS[k]
    return "ACT"


TAU_GRID = [round(x, 2) for x in np.arange(0.50, 0.995, 0.01)]


def threshold_stats(probs: np.ndarray, y: np.ndarray, tau: float) -> dict:
    preds = np.array([IDX[decide(p, tau)] for p in probs])
    y = np.asarray(y)
    act = y == IDX["ACT"]
    non = ~act
    fs = int(((preds != IDX["ACT"]) & act).sum())
    correct_supp = int(((preds != IDX["ACT"]) & non).sum())
    return {
        "tau": tau,
        "false_skips": fs, "act_n": int(act.sum()),
        "false_skip_rate": fs / act.sum() if act.sum() else 0.0,
        "false_skip_ci": wilson(fs, int(act.sum())),
        "correct_suppressions": correct_supp, "non_act_n": int(non.sum()),
        "suppression_rate": correct_supp / non.sum() if non.sum() else 0.0,
        "intervention_rate": float((preds != IDX["ACT"]).mean()) if len(y) else 0.0,
    }


def select_tau(probs: np.ndarray, y: np.ndarray, max_false_skip: float = 0.05) -> tuple[float, list]:
    """Smallest tau on the grid whose empirical false-skip rate is <= max_false_skip.
    Returns 1.0 (never intervene) if no tau qualifies."""
    curve = [threshold_stats(probs, y, t) for t in TAU_GRID]
    for c in curve:
        if c["false_skip_rate"] <= max_false_skip:
            return c["tau"], curve
    return 1.0, curve
