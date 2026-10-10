"""WOE's sign convention is load-bearing: flip it and every coefficient,
every point allocation and the direction of the whole card flip with it."""

import numpy as np
import pandas as pd
import pytest

from credit_scorecard.woe import information_value, iv_strength, woe_table


def test_good_heavy_bin_has_positive_woe():
    """WOE = ln(good/bad), so a bin that is mostly good must score positive."""
    bins = pd.Series(["safe"] * 100 + ["risky"] * 100)
    y = np.r_[np.zeros(95, int), np.ones(5, int), np.zeros(50, int), np.ones(50, int)]
    t = woe_table(bins, y).table.set_index("bin")
    assert t.loc["safe", "woe"] > 0
    assert t.loc["risky", "woe"] < 0


def test_iv_is_non_negative(X, y):
    from credit_scorecard.binning import fit_numeric

    for col in ["income", "utilisation", "age"]:
        b = fit_numeric(X[col], y, feature=col)
        assert information_value(b.transform(X[col]), y) >= 0


def test_uninformative_characteristic_has_near_zero_iv():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.2, 4000)
    bins = pd.Series(rng.choice(["a", "b", "c"], 4000))  # independent of y
    assert information_value(bins, y) < 0.01


def test_iv_contributions_sum_to_iv():
    rng = np.random.default_rng(2)
    bins = pd.Series(rng.choice(list("abcd"), 2000))
    y = rng.binomial(1, 0.3, 2000)
    t = woe_table(bins, y)
    assert t.iv == pytest.approx(t.table["iv_contribution"].sum())


def test_pure_bin_does_not_produce_infinite_woe():
    """Laplace smoothing keeps a bin with no bads finite rather than inf."""
    bins = pd.Series(["pure"] * 50 + ["mixed"] * 50)
    y = np.r_[np.zeros(50, int), np.ones(25, int), np.zeros(25, int)]
    t = woe_table(bins, y).table
    assert np.isfinite(t["woe"]).all()


def test_rejects_non_binary_target():
    with pytest.raises(ValueError, match="binary"):
        woe_table(pd.Series(["a", "b"]), np.array([0, 2]))


def test_rejects_single_class():
    with pytest.raises(ValueError, match="goods and bads"):
        woe_table(pd.Series(["a", "b"]), np.array([0, 0]))


@pytest.mark.parametrize("iv,label", [
    (0.01, "unpredictive"), (0.05, "weak"), (0.2, "medium"),
    (0.4, "strong"), (0.9, "suspicious"),
])
def test_iv_strength_labels(iv, label):
    assert iv_strength(iv) == label
