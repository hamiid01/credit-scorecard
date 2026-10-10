"""PSI is how a card's decay is noticed before the default rate reveals it."""

import numpy as np
import pandas as pd
import pytest

from credit_scorecard.stability import csi, interpret, psi, psi_table


@pytest.fixture
def reference():
    return np.random.default_rng(0).normal(550, 50, 5000)


def test_identical_populations_have_zero_psi(reference):
    assert psi(reference, reference) < 1e-6


def test_same_distribution_resampled_stays_stable(reference):
    other = np.random.default_rng(1).normal(550, 50, 5000)
    assert psi(reference, other) < 0.1


def test_a_large_shift_is_flagged(reference):
    shifted = np.random.default_rng(2).normal(480, 50, 5000)
    value = psi(reference, shifted)
    assert value > 0.25
    assert interpret(value) == "significant shift"


def test_psi_grows_with_the_size_of_the_shift(reference):
    values = [
        psi(reference, np.random.default_rng(3).normal(550 + d, 50, 5000))
        for d in (0, 10, 30, 60)
    ]
    assert values == sorted(values)


def test_psi_is_never_negative(reference):
    for d in (-40, 0, 25):
        assert psi(reference, reference + d) >= 0


def test_psi_table_contributions_sum_to_psi(reference):
    shifted = np.random.default_rng(4).normal(520, 60, 5000)
    t = psi_table(reference, shifted)
    assert t["contribution"].sum() == pytest.approx(psi(reference, shifted))
    assert t["expected_share"].sum() == pytest.approx(1.0, abs=1e-6)


def test_csi_detects_a_mix_shift():
    a = pd.Series(["low"] * 500 + ["high"] * 500)
    b = pd.Series(["low"] * 100 + ["high"] * 900)
    assert csi(a, a) < 1e-6
    assert csi(a, b) > 0.25


def test_csi_handles_a_bin_absent_from_one_period():
    a = pd.Series(["x"] * 400 + ["y"] * 100)
    b = pd.Series(["x"] * 500)
    assert np.isfinite(csi(a, b))


@pytest.mark.parametrize("value,verdict", [
    (0.05, "stable"), (0.15, "moderate shift"), (0.4, "significant shift"),
])
def test_interpretation_thresholds(value, verdict):
    assert interpret(value) == verdict
