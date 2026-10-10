import numpy as np
import pandas as pd
import pytest

from credit_scorecard.data import make_synthetic


@pytest.fixture(scope="session")
def credit():
    """Synthetic credit frame with known monotone relationships. No network."""
    return make_synthetic(n=4000, bad_rate=0.15, seed=0)


@pytest.fixture(scope="session")
def X(credit):
    return credit[0]


@pytest.fixture(scope="session")
def y(credit):
    return credit[1]


@pytest.fixture
def perfect_model():
    """A score that separates the classes completely."""
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(500, int), np.ones(150, int)]
    score = np.r_[rng.normal(700, 20, 500), rng.normal(400, 20, 150)]
    return y, score


@pytest.fixture
def calibrated():
    """Predictions that are correct by construction: y drawn from p."""
    rng = np.random.default_rng(1)
    p = rng.beta(2, 8, 6000)
    y = rng.binomial(1, p)
    return y, p
