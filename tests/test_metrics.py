"""Discrimination and calibration are different questions. A model can rank
perfectly and still be badly calibrated, which is why both are tested."""

import numpy as np
import pytest

from credit_scorecard.metrics import (
    brier_decomposition, brier_score, calibration_table, gini,
    hosmer_lemeshow, ks_statistic, score_band_report, summary,
)


def test_gini_is_positive_when_a_high_score_means_low_risk(perfect_model):
    """Scorecard points run opposite to PD. Getting the sign wrong here
    produces a confidently reported negative Gini."""
    y, score = perfect_model
    assert gini(y, score) > 0.95


def test_gini_is_zero_for_a_random_score():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.2, 5000)
    assert abs(gini(y, rng.normal(size=5000))) < 0.06


def test_gini_flips_sign_when_the_score_is_reversed(perfect_model):
    y, score = perfect_model
    assert gini(y, score) == pytest.approx(-gini(y, -score))


def test_ks_is_a_proportion(perfect_model):
    y, score = perfect_model
    assert 0.0 <= ks_statistic(y, score) <= 1.0
    assert ks_statistic(y, score) > 0.9


def test_brier_rewards_confident_correct_predictions():
    y = np.array([0, 0, 1, 1])
    assert brier_score(y, np.array([0.01, 0.01, 0.99, 0.99])) < \
           brier_score(y, np.array([0.4, 0.4, 0.6, 0.6]))


def test_brier_decomposition_identity(calibrated):
    """Brier = reliability - resolution + uncertainty."""
    y, p = calibrated
    d = brier_decomposition(y, p)
    assert d["brier"] == pytest.approx(
        d["reliability"] - d["resolution"] + d["uncertainty"], abs=1e-3
    )


def test_calibrated_predictions_have_low_reliability_error(calibrated):
    y, p = calibrated
    assert brier_decomposition(y, p)["reliability"] < 0.001


def test_hosmer_lemeshow_accepts_a_calibrated_model(calibrated):
    y, p = calibrated
    assert hosmer_lemeshow(y, p)["calibrated_at_5pct"]


def test_hosmer_lemeshow_rejects_a_miscalibrated_model(calibrated):
    """Same ranking, shifted probabilities: discrimination is untouched but
    calibration must fail."""
    y, p = calibrated
    assert not hosmer_lemeshow(y, np.clip(p * 3, 0, 0.999))["calibrated_at_5pct"]


def test_calibration_table_tracks_observed_rates(calibrated):
    y, p = calibrated
    t = calibration_table(y, p)
    assert (t["difference"].abs() < 0.05).all()
    assert t["predicted"].is_monotonic_increasing


def test_score_bands_order_risk(perfect_model):
    y, score = perfect_model
    t = score_band_report(y, score, n_bands=5)
    assert t["bad_rate"].iloc[0] > t["bad_rate"].iloc[-1]
    assert t["count"].sum() == len(y)


def test_summary_reports_both_families(perfect_model, calibrated):
    y, score = perfect_model
    out = summary(y, score, 1 / (1 + np.exp((score - 550) / 30)))
    for key in ("gini", "ks", "auc", "brier", "reliability", "hl_p_value"):
        assert key in out
