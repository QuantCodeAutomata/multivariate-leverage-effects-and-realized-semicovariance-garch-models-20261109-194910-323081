"""Unit tests for out-of-sample loss functions and the Model Confidence Set
(Section 4.3; Hansen, Lunde and Nason, 2011)."""
from __future__ import annotations

import numpy as np

from src.forecast_eval import frobenius_loss, mcs_included, model_confidence_set, qlike_loss


def test_frobenius_loss_is_zero_for_perfect_forecast():
    H = np.stack([np.eye(2) for _ in range(5)])
    loss = frobenius_loss(H, H)
    np.testing.assert_allclose(loss, 0.0, atol=1e-10)


def test_frobenius_loss_is_nonnegative():
    rng = np.random.default_rng(0)
    H = np.stack([np.eye(2) + 0.01 * rng.standard_normal((2, 2)) for _ in range(10)])
    RCOV = np.stack([np.eye(2) + 0.05 * rng.standard_normal((2, 2)) for _ in range(10)])
    loss = frobenius_loss(H, RCOV)
    assert (loss >= 0).all()


def test_qlike_loss_is_zero_when_forecast_equals_realized():
    H = np.stack([np.array([[1.0, 0.2], [0.2, 0.8]]) for _ in range(4)])
    loss = qlike_loss(H, H)
    np.testing.assert_allclose(loss, np.log(np.linalg.det(H[0])) + 2.0, atol=1e-8)


def test_mcs_recovers_best_model_as_sole_survivor():
    """A model with a materially and persistently lower loss must end up
    alone in the MCS, with every worse model's p-value below the MCS level."""
    rng = np.random.default_rng(42)
    T = 400
    losses = {
        "good": 1.0 + 0.1 * rng.standard_normal(T),
        "bad": 3.0 + 0.1 * rng.standard_normal(T),
        "worse": 5.0 + 0.1 * rng.standard_normal(T),
    }
    p_values = model_confidence_set(losses, alpha=0.20, n_boot=200, block_length=5, seed=1)
    included = mcs_included(p_values, level=0.80)
    assert included == ["good"]
    assert p_values["worse"] < p_values["good"]


def test_mcs_includes_all_models_when_losses_are_indistinguishable():
    rng = np.random.default_rng(7)
    T = 300
    losses = {name: 1.0 + 0.5 * rng.standard_normal(T) for name in ("a", "b", "c")}
    p_values = model_confidence_set(losses, alpha=0.20, n_boot=200, block_length=5, seed=2)
    included = mcs_included(p_values, level=0.80)
    assert set(included) == {"a", "b", "c"}
