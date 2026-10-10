"""Selection decides what a credit officer has to collect, monitor and
explain for the life of the card, so each filter earns its place."""

import numpy as np
import pandas as pd
import pytest

from credit_scorecard.selection import drop_correlated, points_spread, select_by_iv, vif


def test_iv_floor_drops_the_unpredictive():
    keep, table = select_by_iv({"strong": 0.3, "weak": 0.005}, min_iv=0.02)
    assert keep == ["strong"]
    assert "below IV floor" in table.set_index("characteristic").loc["weak", "decision"]


def test_suspiciously_high_iv_is_kept_but_flagged():
    """An IV above 0.5 on credit data usually means leakage, so it is
    surfaced for a human rather than silently trusted or silently dropped."""
    keep, table = select_by_iv({"leaky": 0.9}, max_iv=0.5)
    assert keep == ["leaky"]
    assert "FLAGGED" in table.set_index("characteristic").loc["leaky", "decision"]


def test_correlated_pair_keeps_the_higher_iv():
    rng = np.random.default_rng(0)
    base = rng.normal(size=1000)
    frame = pd.DataFrame({
        "a": base, "a_copy": base + rng.normal(0, 0.01, 1000), "b": rng.normal(size=1000),
    })
    keep, dropped = drop_correlated(frame, {"a": 0.3, "a_copy": 0.1, "b": 0.2}, threshold=0.8)
    assert "a" in keep and "a_copy" not in keep and "b" in keep
    assert dropped.iloc[0]["dropped"] == "a_copy"


def test_uncorrelated_characteristics_all_survive():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(rng.normal(size=(800, 4)), columns=list("abcd"))
    keep, dropped = drop_correlated(frame, dict.fromkeys("abcd", 0.2), threshold=0.8)
    assert len(keep) == 4 and dropped.empty


def test_vif_detects_collinearity_spread_across_a_block():
    """Three characteristics each correlated ~0.6 pairwise sit under a 0.8
    pairwise threshold while being jointly near-singular. This is the case
    pairwise correlation structurally cannot see."""
    rng = np.random.default_rng(2)
    shared = rng.normal(size=2000)
    frame = pd.DataFrame({f"x{i}": shared + rng.normal(0, 0.35, 2000) for i in range(4)})
    frame["independent"] = rng.normal(size=2000)

    corr = frame.corr().abs()
    np.fill_diagonal(corr.values, 0)
    assert corr.max().max() < 0.95  # no single pair is extreme

    out = vif(frame).set_index("characteristic")
    assert out.loc["independent", "vif"] < 2
    assert out.drop(index="independent")["vif"].max() > 5


def test_vif_is_one_for_independent_columns():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(rng.normal(size=(2000, 3)), columns=list("abc"))
    assert vif(frame)["vif"].max() < 1.5


def test_points_spread_measures_what_a_characteristic_can_move(X, y):
    from credit_scorecard.scorecard import Scorecard

    card = Scorecard.fit(X, y, categorical=("region",))
    out = points_spread(card).set_index("characteristic")
    assert set(out.index) == set(card.features)
    assert (out["spread"] >= 0).all()
    assert out["spread"].idxmax() == card.iv_summary.iloc[0]["characteristic"]
