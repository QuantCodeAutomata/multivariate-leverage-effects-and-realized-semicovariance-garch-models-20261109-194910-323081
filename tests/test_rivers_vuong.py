"""Unit tests for the Rivers and Vuong (2002) non-nested comparison test."""
from __future__ import annotations

import numpy as np

from src.rivers_vuong import pairwise_rivers_vuong, rivers_vuong_test


def test_statistic_is_zero_for_identical_models():
    ll = np.array([1.0, 2.0, 1.5, 0.5, 1.2])
    out = rivers_vuong_test(ll, ll)
    assert out["statistic"] == 0.0
    assert out["p_value"] == 1.0


def test_favors_model_with_higher_mean_loglik():
    rng = np.random.default_rng(3)
    T = 1000
    ll_a = 1.0 + 0.2 * rng.standard_normal(T)  # mean 1.0
    ll_b = 0.0 + 0.2 * rng.standard_normal(T)  # mean 0.0, clearly worse
    out = rivers_vuong_test(ll_a, ll_b)
    assert out["favors_a"] is True
    assert out["statistic"] > 0
    assert out["p_value"] < 0.01


def test_pairwise_rivers_vuong_runs_for_multiple_challengers():
    rng = np.random.default_rng(5)
    T = 500
    per_day_ll = {
        "crBG-S": 2.0 + 0.3 * rng.standard_normal(T),
        "BG": 1.0 + 0.3 * rng.standard_normal(T),
        "tBG": 1.5 + 0.3 * rng.standard_normal(T),
    }
    out = pairwise_rivers_vuong(per_day_ll, reference="crBG-S", challengers=["BG", "tBG"])
    assert set(out.keys()) == {"BG", "tBG"}
    assert out["BG"]["favors_a"] is True
