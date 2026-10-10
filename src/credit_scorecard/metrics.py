"""Discrimination and calibration.

These answer two different questions, and a scorecard can pass one while
failing the other:

**Discrimination** -- does the score rank bads above goods? Gini, KS, AUC.
Invariant to any monotone transformation of the score, so a model can rank
perfectly and still assign nonsense probabilities.

**Calibration** -- when the model says 4%, do 4% default? Brier, its
decomposition, and Hosmer-Lemeshow. This is what pricing, provisioning and
IFRS 9 depend on, and it is the half that portfolio write-ups usually omit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def gini(y: np.ndarray, score: np.ndarray) -> float:
    """2*AUC - 1, computed so that a higher score means *lower* risk.

    Scorecard points run the opposite way to a probability of default, so the
    score is negated before being handed to the AUC. Getting this backwards
    yields a confidently reported negative Gini.
    """
    return 2 * roc_auc_score(np.asarray(y), -np.asarray(score, dtype=float)) - 1


def auc_pd(y: np.ndarray, pd_hat: np.ndarray) -> float:
    """AUC for a probability of default (higher = riskier)."""
    return float(roc_auc_score(np.asarray(y), np.asarray(pd_hat, dtype=float)))


def ks_statistic(y: np.ndarray, score: np.ndarray) -> float:
    """Maximum separation between the cumulative good and bad distributions."""
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    order = np.argsort(score)
    y_sorted = y[order]
    cum_bad = np.cumsum(y_sorted) / max(1, y_sorted.sum())
    cum_good = np.cumsum(1 - y_sorted) / max(1, (1 - y_sorted).sum())
    return float(np.max(np.abs(cum_bad - cum_good)))


def brier_score(y: np.ndarray, pd_hat: np.ndarray) -> float:
    """Mean squared error of the predicted probabilities."""
    return float(np.mean((np.asarray(pd_hat, dtype=float) - np.asarray(y)) ** 2))


def brier_decomposition(y: np.ndarray, pd_hat: np.ndarray, n_bins: int = 10) -> dict:
    """Murphy's decomposition: Brier = reliability - resolution + uncertainty.

    ``reliability`` is calibration error (lower is better); ``resolution`` is
    how far the bin rates move away from the base rate (higher is better);
    ``uncertainty`` is the irreducible variance of the outcome.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(pd_hat, dtype=float)
    base = y.mean()

    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.searchsorted(np.unique(edges)[1:-1], p, side="right")

    reliability = resolution = 0.0
    for b in np.unique(idx):
        m = idx == b
        n_b, p_b, o_b = m.sum(), p[m].mean(), y[m].mean()
        reliability += n_b * (p_b - o_b) ** 2
        resolution += n_b * (o_b - base) ** 2
    n = len(y)
    return {
        "brier": brier_score(y, p),
        "reliability": reliability / n,
        "resolution": resolution / n,
        "uncertainty": base * (1 - base),
    }


def hosmer_lemeshow(y: np.ndarray, pd_hat: np.ndarray, n_groups: int = 10) -> dict:
    """Hosmer-Lemeshow goodness-of-fit test.

    H0 is that the model is calibrated, so a **small p-value rejects
    calibration**. Note the test is sensitive to sample size: on 30k accounts
    it will reject for deviations too small to matter commercially, which is
    why the calibration table is reported next to it rather than the p-value
    alone.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(pd_hat, dtype=float)

    edges = np.quantile(p, np.linspace(0, 1, n_groups + 1))
    inner = np.unique(edges)[1:-1]
    idx = np.searchsorted(inner, p, side="right")

    chi2 = 0.0
    groups = 0
    for b in np.unique(idx):
        m = idx == b
        n_b = int(m.sum())
        if n_b == 0:
            continue
        observed = y[m].sum()
        expected = p[m].sum()
        denom = expected * (1 - expected / n_b)
        if denom > 0:
            chi2 += (observed - expected) ** 2 / denom
            groups += 1

    df = max(1, groups - 2)
    return {
        "chi2": float(chi2),
        "df": df,
        "p_value": float(stats.chi2.sf(chi2, df)),
        "calibrated_at_5pct": bool(stats.chi2.sf(chi2, df) > 0.05),
    }


def calibration_table(y: np.ndarray, pd_hat: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed default rate by predicted-risk band."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(pd_hat, dtype=float)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    inner = np.unique(edges)[1:-1]
    idx = np.searchsorted(inner, p, side="right")

    rows = []
    for b in np.unique(idx):
        m = idx == b
        rows.append({
            "band": int(b),
            "count": int(m.sum()),
            "predicted": float(p[m].mean()),
            "observed": float(y[m].mean()),
            "difference": float(p[m].mean() - y[m].mean()),
        })
    return pd.DataFrame(rows)


def score_band_report(
    y: np.ndarray, score: np.ndarray, n_bands: int = 10
) -> pd.DataFrame:
    """Default rate by score band -- the table a credit committee actually reads.

    A monotone ``bad_rate`` column going down as the score goes up is the
    practical test that the card works.
    """
    y = np.asarray(y)
    s = np.asarray(score, dtype=float)
    edges = np.quantile(s, np.linspace(0, 1, n_bands + 1))
    inner = np.unique(edges)[1:-1]
    idx = np.searchsorted(inner, s, side="right")

    rows = []
    for b in np.unique(idx):
        m = idx == b
        rows.append({
            "band": int(b),
            "score_min": float(s[m].min()),
            "score_max": float(s[m].max()),
            "count": int(m.sum()),
            "bads": int(y[m].sum()),
            "bad_rate": float(y[m].mean()),
        })
    return pd.DataFrame(rows)


def summary(y: np.ndarray, score: np.ndarray, pd_hat: np.ndarray) -> dict:
    """Headline discrimination and calibration figures."""
    out = {
        "gini": gini(y, score),
        "auc": auc_pd(y, pd_hat),
        "ks": ks_statistic(y, score),
        **{k: v for k, v in brier_decomposition(y, pd_hat).items()},
    }
    out.update({f"hl_{k}": v for k, v in hosmer_lemeshow(y, pd_hat).items()})
    return out
