import numpy as np

from whetherman.calibrate.adjust import VectorScaling, decide, fit_vector_scaling, select_tau, softmax


def test_decide_defaults_to_act():
    assert decide([0.1, 0.8, 0.05, 0.05], 0.9) == "ACT"
    assert decide([0.05, 0.92, 0.02, 0.01], 0.9) == "ANSWER"
    assert decide([0.6, 0.4, 0, 0], 0.5) == "ACT"


def test_vector_scaling_learns_bias_shift():
    rng = np.random.default_rng(0)
    # backbone says ANSWER-ish for everything; this agent actually needs tools on most items
    z = rng.normal(0, 0.3, (200, 4)) + np.array([0.0, 1.0, -2, -2])
    y = np.where(rng.random(200) < 0.8, 0, 1)
    vs = fit_vector_scaling(z, y, l2=0.1)
    p = vs.probs(z)
    assert p[:, 0].mean() > softmax(z)[:, 0].mean() + 0.2


def test_regularization_keeps_identity_with_no_signal():
    vs = fit_vector_scaling(np.zeros((0, 4)), np.zeros(0, dtype=int))
    assert vs.scale == [1.0] * 4 and vs.bias == [0.0] * 4
    z = np.array([[1.0, 0, 0, 0]] * 3)
    vs = fit_vector_scaling(z, np.array([0, 0, 0]), l2=1e6)
    assert np.allclose(vs.scale, 1, atol=1e-2) and np.allclose(vs.bias, 0, atol=1e-2)


def test_select_tau_respects_false_skip_target():
    probs = np.array([[0.3, 0.7, 0, 0]] * 10 + [[0.05, 0.95, 0, 0]] * 10)
    y = np.array([0] * 10 + [1] * 10)
    tau, curve = select_tau(probs, y, 0.05)
    assert tau >= 0.7
    assert next(c for c in curve if c["tau"] == tau)["false_skips"] == 0
    tau, _ = select_tau(np.array([[0.001, 0.999, 0, 0]] * 4), np.array([0] * 4))
    assert tau == 1.0
    assert VectorScaling().apply(np.ones(4)).tolist() == [1, 1, 1, 1]
