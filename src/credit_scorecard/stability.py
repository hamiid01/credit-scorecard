"""Population and characteristic stability.

A scorecard degrades because the population moves, not because the
coefficients change. PSI and CSI are how that movement is detected before the
default rate reveals it a year later.

    PSI = sum_i ( a_i - e_i ) * ln( a_i / e_i )

over score bands, where ``e`` is the development (expected) share and ``a`` the
current (actual) share. The same formula over a characteristic's bins is the
CSI, which localises *which* characteristic moved once the PSI alerts.

Conventional thresholds:

    < 0.10   stable
    0.10-0.25 moderate shift -- investigate
    > 0.25   significant shift -- the card likely needs redevelopment

Those cut-offs are industry convention, not a statistical test: PSI has no
null distribution attached, and on a large enough sample a commercially
irrelevant shift still clears 0.25. Treat them as a trigger to look, not a
verdict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Added to each share so an empty band contributes a large finite term
#: rather than infinity.
EPS = 1e-6


def _shares(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges, values, side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    return counts / max(1, counts.sum())


def psi(expected: np.ndarray, actual: np.ndarray, n_bands: int = 10) -> float:
    """Population stability index between two score distributions.

    Bands are cut on the *expected* (development) distribution, so the
    comparison is against the population the card was built on.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bands + 1)))[1:-1]

    e = _shares(expected, edges) + EPS
    a = _shares(actual, edges) + EPS
    return float(np.sum((a - e) * np.log(a / e)))


def psi_table(expected: np.ndarray, actual: np.ndarray, n_bands: int = 10) -> pd.DataFrame:
    """Per-band PSI contributions, which say *where* the population moved."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bands + 1)))[1:-1]

    e = _shares(expected, edges) + EPS
    a = _shares(actual, edges) + EPS
    bounds = [-np.inf, *edges, np.inf]
    return pd.DataFrame({
        "band": [f"({bounds[i]:.4g}, {bounds[i + 1]:.4g}]" for i in range(len(e))],
        "expected_share": e - EPS,
        "actual_share": a - EPS,
        "contribution": (a - e) * np.log(a / e),
    })


def csi(
    expected_bins: pd.Series, actual_bins: pd.Series
) -> float:
    """Characteristic stability index over one characteristic's bins."""
    e_counts = expected_bins.value_counts(normalize=True)
    a_counts = actual_bins.value_counts(normalize=True)
    bins = e_counts.index.union(a_counts.index)
    e = e_counts.reindex(bins).fillna(0).to_numpy() + EPS
    a = a_counts.reindex(bins).fillna(0).to_numpy() + EPS
    return float(np.sum((a - e) * np.log(a / e)))


def csi_report(card, X_expected: pd.DataFrame, X_actual: pd.DataFrame) -> pd.DataFrame:
    """CSI for every characteristic in a fitted card, worst first."""
    rows = []
    for col in card.features:
        b = card.binnings[col]
        rows.append({
            "characteristic": col,
            "csi": csi(b.transform(X_expected[col]), b.transform(X_actual[col])),
        })
    out = pd.DataFrame(rows).sort_values("csi", ascending=False).reset_index(drop=True)
    out["verdict"] = out["csi"].map(interpret)
    return out


def interpret(value: float) -> str:
    """The conventional verbal reading of a PSI/CSI value."""
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "moderate shift"
    return "significant shift"
