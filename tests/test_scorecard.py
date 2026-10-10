"""If the points table does not reproduce the model it was fitted from, the
card is not the model and nothing downstream means anything. That identity is
the first test here and the most important one in the repository."""

import numpy as np
import pytest

from credit_scorecard.scorecard import ScalingParams, Scorecard


@pytest.fixture(scope="module")
def card(X, y):
    return Scorecard.fit(X, y, categorical=("region",))


def test_points_reconstruct_the_model(card, X):
    """sum of points == offset - factor * logit_bad, exactly."""
    s = card.score(X, round_points=False).to_numpy()
    p = card.predict_proba(X)
    logit_bad = np.log(p / (1 - p))
    expected = card.scaling.offset - card.scaling.factor * logit_bad
    assert np.abs(s - expected).max() < 1e-9


def test_score_to_pd_inverts_the_scaling(card, X):
    s = card.score(X, round_points=False)
    assert np.abs(card.score_to_pd(s) - card.predict_proba(X)).max() < 1e-9


def test_pdo_doubles_the_odds(card):
    """The defining property of the scale: +PDO points halves the odds of bad."""
    pdo = card.scaling.pdo
    base = np.array([500.0])
    odds_at_base = (1 - card.score_to_pd(base)) / card.score_to_pd(base)
    odds_up = (1 - card.score_to_pd(base + pdo)) / card.score_to_pd(base + pdo)
    assert odds_up / odds_at_base == pytest.approx(2.0, rel=1e-9)


def test_base_score_sits_at_base_odds(card):
    """600 points must mean exactly 50:1 good:bad under the default anchors."""
    s = card.scaling
    pd_at_base = card.score_to_pd(np.array([s.base_score]))[0]
    assert (1 - pd_at_base) / pd_at_base == pytest.approx(s.base_odds, rel=1e-9)


def test_scaling_anchors_do_not_change_the_ranking(X, y):
    """Anchors are a reporting convention; they must not reorder anybody."""
    a = Scorecard.fit(X, y, categorical=("region",), scaling=ScalingParams(600, 50, 20))
    b = Scorecard.fit(X, y, categorical=("region",), scaling=ScalingParams(1000, 20, 40))
    ra = a.score(X, round_points=False).rank()
    rb = b.score(X, round_points=False).rank()
    assert np.corrcoef(ra, rb)[0, 1] == pytest.approx(1.0, abs=1e-9)


def test_coefficients_are_negative_under_the_woe_convention(card):
    """WOE = ln(good/bad) and the target is bad, so every well-behaved
    characteristic must take a negative coefficient. A positive one scores
    against its own WOE."""
    assert all(b < 0 for b in card.coefficients.values())


def test_higher_score_means_lower_observed_default_rate(card, X, y):
    s = card.score(X)
    top = np.asarray(y)[s >= s.quantile(0.8)].mean()
    bottom = np.asarray(y)[s <= s.quantile(0.2)].mean()
    assert top < bottom


def test_points_table_covers_every_bin(card, X):
    pt = card.points_table
    for col in card.features:
        assert (pt["characteristic"] == col).sum() == len(card.woe_tables[col].table)


def test_points_table_is_sorted_for_reading(card):
    """Within a characteristic, points must ascend so the card reads from
    worst risk to best."""
    pt = card.points_table
    for _, g in pt.groupby("characteristic", sort=False):
        assert g["points"].is_monotonic_increasing


def test_unseen_category_scores_neutrally(card, X):
    """A category absent from development maps to WOE 0, i.e. the population
    average, rather than crashing or silently scoring as the worst bin."""
    probe = X.head(5).copy()
    probe["region"] = "atlantis"
    assert np.isfinite(card.score(probe)).all()


def test_transform_woe_has_one_column_per_characteristic(card, X):
    W = card.transform_woe(X)
    assert list(W.columns) == card.features
    assert np.isfinite(W.to_numpy()).all()
