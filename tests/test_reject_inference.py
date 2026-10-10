"""Reject inference extrapolates into a region where no outcome was observed.
These tests pin the mechanics; they cannot and do not validate the inference,
because on real data the counterfactual is unobservable by construction."""

import numpy as np
import pytest

from credit_scorecard.reject_inference import (
    evaluate_inference, fuzzy_augmentation, parcelling, simulate_accept_reject,
)
from credit_scorecard.scorecard import Scorecard


@pytest.fixture(scope="module")
def split(X, y):
    policy = -X["utilisation"].to_numpy() + np.log(X["income"].fillna(X["income"].median()))
    return simulate_accept_reject(X, y, policy, accept_rate=0.7, seed=0)


def test_split_respects_the_accept_rate(split, X):
    assert len(split.X_accept) + len(split.X_reject) == len(X)
    assert split.accept_rate == pytest.approx(0.7, abs=0.02)


def test_rejects_are_riskier_than_accepts(split):
    """A cut-off policy that works must decline a worse population, or the
    whole exercise is meaningless."""
    assert split.y_reject_true.mean() > split.y_accept.mean()


def test_parcelling_returns_the_full_population(split):
    kgb = Scorecard.fit(split.X_accept, split.y_accept, categorical=("region",))
    X_all, y_all = parcelling(kgb, split.X_accept, split.y_accept, split.X_reject)
    assert len(X_all) == len(split.X_accept) + len(split.X_reject)
    assert set(np.unique(y_all)) <= {0, 1}


def test_parcelling_infers_a_worse_rate_than_the_accepts(split):
    """The uplift exists because rejects were declined for a reason."""
    kgb = Scorecard.fit(split.X_accept, split.y_accept, categorical=("region",))
    _, y_all = parcelling(kgb, split.X_accept, split.y_accept, split.X_reject, uplift=2.5)
    inferred = np.asarray(y_all)[len(split.X_accept):]
    assert inferred.mean() > np.asarray(split.y_accept).mean()


def test_parcelling_is_reproducible(split):
    kgb = Scorecard.fit(split.X_accept, split.y_accept, categorical=("region",))
    args = (kgb, split.X_accept, split.y_accept, split.X_reject)
    a = parcelling(*args, seed=7)[1]
    b = parcelling(*args, seed=7)[1]
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_fuzzy_weights_keep_each_reject_counting_once(split):
    """Each reject enters twice, as good and as bad; the pair of weights must
    sum to 1 so the sample size is not silently doubled."""
    kgb = Scorecard.fit(split.X_accept, split.y_accept, categorical=("region",))
    X_all, y_all, w = fuzzy_augmentation(kgb, split.X_accept, split.y_accept, split.X_reject)

    n_acc, n_rej = len(split.X_accept), len(split.X_reject)
    assert len(X_all) == n_acc + 2 * n_rej
    assert np.allclose(w[:n_acc], 1.0)
    assert np.allclose(w[n_acc:n_acc + n_rej] + w[n_acc + n_rej:], 1.0)
    assert w.sum() == pytest.approx(n_acc + n_rej)


def test_fuzzy_sample_can_be_refitted(split):
    kgb = Scorecard.fit(split.X_accept, split.y_accept, categorical=("region",))
    X_all, y_all, w = fuzzy_augmentation(kgb, split.X_accept, split.y_accept, split.X_reject)
    card = Scorecard.fit(X_all, y_all, categorical=("region",), sample_weight=w)
    assert np.isfinite(card.score(X_all)).all()


def test_evaluation_scores_both_cards_on_the_hidden_outcomes(split):
    kgb = Scorecard.fit(split.X_accept, split.y_accept, categorical=("region",))
    X_all, y_all = parcelling(kgb, split.X_accept, split.y_accept, split.X_reject)
    agb = Scorecard.fit(X_all, y_all, categorical=("region",))
    out = evaluate_inference(split, kgb, agb)
    assert len(out) == 2
    assert out["gini_full_population"].between(-1, 1).all()
