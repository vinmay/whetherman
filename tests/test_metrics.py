import numpy as np

from eval.metrics import ece, macro_f1, rate, safety, summarize


def test_rate_has_denominator_and_ci():
    r = rate(0, 8)
    assert r["n"] == 8 and r["rate"] == 0 and r["ci95"][1] > 0.3


def test_safety_counts():
    y = ["ACT", "ACT", "ANSWER", "ASK", "ANSWER"]
    p = ["ACT", "ANSWER", "ANSWER", "ACT", "DECLINE"]
    s = safety(y, p)
    assert s["false_skip"]["k"] == 1 and s["false_skip"]["n"] == 2
    assert s["false_skip_by_class"]["ANSWER"]["k"] == 1
    assert s["correct_suppression"]["k"] == 2 and s["correct_suppression"]["n"] == 3
    assert s["exact_non_act"]["k"] == 1
    assert s["intervention"]["k"] == 3


def test_macro_f1_and_ece():
    assert macro_f1(["ACT", "ANSWER"], ["ACT", "ANSWER"]) == 1.0
    probs = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    assert ece(probs, np.array([0, 1]))[0] == 0.0
    s = summarize(["ACT", "ANSWER"], ["ACT", "ACT"], probs)
    assert s["accuracy"] == 0.5 and "risk_coverage" in s
