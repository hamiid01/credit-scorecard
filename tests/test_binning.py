"""Monotonicity is a hard constraint, not an aspiration: a card whose points
go up then down for the same characteristic cannot be explained or signed off."""

import numpy as np
import pandas as pd
import pytest

from credit_scorecard.binning import MISSING, fit_categorical, fit_numeric
from credit_scorecard.woe import woe_table


def _woes(binning, x, y):
    t = woe_table(binning.transform(x), y, binning.feature).table
    return t[t["bin"] != MISSING]["woe"].to_numpy()


@pytest.mark.parametrize("col", ["income", "utilisation", "age", "arrears"])
def test_woe_is_monotonic(col, X, y):
    b = fit_numeric(X[col], y, feature=col)
    # bins come back in value order, so WOE must be sorted one way or the other
    t = woe_table(b.transform(X[col]), y, col).table
    t = t[t["bin"] != MISSING]
    w = t["woe"].to_numpy()
    assert np.all(np.diff(w) >= -1e-12) or np.all(np.diff(w) <= 1e-12), \
        f"{col} WOE is not monotonic: {w}"


def test_monotonic_false_can_produce_more_bins(X, y):
    free = fit_numeric(X["income"], y, monotonic=False, max_bins=10)
    forced = fit_numeric(X["income"], y, monotonic=True, max_bins=10)
    assert free.n_bins >= forced.n_bins


def test_respects_max_bins(X, y):
    for max_bins in (3, 4, 5):
        b = fit_numeric(X["utilisation"], y, max_bins=max_bins)
        assert len(b.breaks) + 1 <= max_bins


def test_respects_min_bin_rate(X, y):
    b = fit_numeric(X["utilisation"], y, min_bin_rate=0.15, max_bins=10)
    counts = b.transform(X["utilisation"]).value_counts()
    non_missing = counts[counts.index != MISSING]
    assert (non_missing / non_missing.sum()).min() >= 0.10


def test_missing_values_get_their_own_bin(X, y):
    """Blank is often predictive in credit data; imputing it destroys that."""
    b = fit_numeric(X["income"], y, feature="income")
    assert b.has_missing
    out = b.transform(X["income"])
    assert (out == MISSING).sum() == X["income"].isna().sum()


def test_transform_handles_unseen_extremes(X, y):
    """Bins are closed with -inf/inf, so production values outside the
    development range still land somewhere."""
    b = fit_numeric(X["income"], y)
    out = b.transform(pd.Series([-1e9, 1e12]))
    assert out.notna().all()
    assert (out != MISSING).all()


def test_constant_column_yields_a_single_bin(y):
    b = fit_numeric(pd.Series(np.ones(len(y))), y)
    assert b.breaks == []
    assert b.n_bins == 1


def test_every_bin_holds_both_outcomes(X, y):
    """A bin with no bads has undefined WOE; the binner must merge it away."""
    b = fit_numeric(X["utilisation"], y, max_bins=10)
    t = woe_table(b.transform(X["utilisation"]), y).table
    t = t[t["bin"] != MISSING]
    assert (t["goods"] > 0).all() and (t["bads"] > 0).all()


def test_categorical_grouping_is_ordered_by_risk(X, y):
    b = fit_categorical(X["region"], y, feature="region")
    assert b.is_categorical
    assert set(b.groups) == set(X["region"].dropna().unique())
    out = b.transform(X["region"])
    assert out.notna().all()


def test_categorical_respects_max_bins():
    rng = np.random.default_rng(0)
    cats = pd.Series(rng.choice(list("abcdefghij"), 3000))
    y = pd.Series(rng.binomial(1, 0.25, 3000))
    b = fit_categorical(cats, y, max_bins=3)
    assert len(set(b.groups.values())) <= 3
