"""Characteristic selection.

Two filters, in order:

1. **Information value.** Drop characteristics below a floor (conventionally
   0.02) as unpredictive, and flag those above a ceiling (0.5) as suspicious
   rather than excellent -- an IV that high on credit data usually means the
   characteristic encodes the outcome.

2. **Correlation.** WOE-transformed characteristics are frequently collinear
   (six months of bill amounts measure one thing). Where a pair exceeds the
   threshold, keep the higher IV. This matters more for a scorecard than for a
   black box: collinearity destabilises coefficient *signs*, and a card with a
   characteristic scoring against its own WOE direction cannot be signed off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_by_iv(
    iv: dict[str, float], min_iv: float = 0.02, max_iv: float = 0.5
) -> tuple[list[str], pd.DataFrame]:
    """Keep characteristics whose IV clears the floor; flag suspicious ones."""
    rows = []
    keep = []
    for feature, value in sorted(iv.items(), key=lambda kv: -kv[1]):
        if value < min_iv:
            decision = "dropped: below IV floor"
        elif value > max_iv:
            decision = "kept: FLAGGED, check for leakage"
            keep.append(feature)
        else:
            decision = "kept"
            keep.append(feature)
        rows.append({"characteristic": feature, "iv": value, "decision": decision})
    return keep, pd.DataFrame(rows)


def drop_correlated(
    woe_frame: pd.DataFrame, iv: dict[str, float], threshold: float = 0.8
) -> tuple[list[str], pd.DataFrame]:
    """Greedily drop the lower-IV member of each over-correlated pair."""
    corr = woe_frame.corr().abs()
    ordered = sorted(woe_frame.columns, key=lambda c: -iv.get(c, 0.0))

    keep: list[str] = []
    dropped = []
    for feature in ordered:
        clash = next(
            (k for k in keep if corr.loc[feature, k] >= threshold), None
        )
        if clash is None:
            keep.append(feature)
        else:
            dropped.append({
                "dropped": feature,
                "kept": clash,
                "correlation": float(corr.loc[feature, clash]),
                "iv_dropped": iv.get(feature, np.nan),
                "iv_kept": iv.get(clash, np.nan),
            })
    return keep, pd.DataFrame(dropped)


def vif(woe_frame: pd.DataFrame) -> pd.DataFrame:
    """Variance inflation factor per characteristic.

    Pairwise correlation only sees collinearity concentrated in a *pair*. When
    six characteristics each measure a slice of the same thing -- six months of
    repayment status, say -- every pairwise correlation can sit comfortably
    below a 0.8 threshold while the block is jointly near-singular. VIF catches
    that, because it regresses each characteristic on *all* the others.

    Conventional reading: > 5 is worth investigating, > 10 is serious.

    Reported even when it comes back clean, because a low VIF is itself a
    finding: it rules collinearity *out* as the explanation when a
    characteristic with a strong IV turns out to contribute nothing.
    """
    from sklearn.linear_model import LinearRegression

    cols = list(woe_frame.columns)
    rows = []
    for col in cols:
        others = [c for c in cols if c != col]
        if not others:
            rows.append({"characteristic": col, "vif": 1.0})
            continue
        r2 = LinearRegression().fit(woe_frame[others], woe_frame[col]).score(
            woe_frame[others], woe_frame[col]
        )
        rows.append({
            "characteristic": col,
            "vif": float("inf") if r2 >= 1 else 1.0 / (1.0 - r2),
        })
    out = pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)
    out["verdict"] = pd.cut(
        out["vif"], [-np.inf, 5, 10, np.inf], labels=["ok", "investigate", "serious"]
    )
    return out


def points_spread(card) -> pd.DataFrame:
    """How many points each characteristic can actually move a score by.

    This is the question a scorecard owner asks that a coefficient table does
    not answer: a characteristic with a wide WOE range but a tiny coefficient
    contributes nothing, yet still has to be collected, monitored, explained
    and signed off. Spread is ``max(points) - min(points)`` over its bins.
    """
    pt = card.points_table.groupby("characteristic")["points"].agg(["min", "max"])
    pt["spread"] = pt["max"] - pt["min"]
    pt["coefficient"] = pd.Series(card.coefficients)
    pt["wrong_sign"] = pt["coefficient"] > 0
    return pt.sort_values("spread").reset_index()
