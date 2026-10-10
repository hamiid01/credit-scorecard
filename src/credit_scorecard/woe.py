"""Weight of evidence and information value.

Sign convention
---------------
    WOE_i = ln( P(x in bin i | good) / P(x in bin i | bad) )

so **higher WOE means better credit quality**. The opposite convention,
ln(bad/good), is equally common in the literature and flips the sign of every
coefficient and every point allocation downstream. Mixing the two is the single
most common way to produce a scorecard that ranks backwards, so the convention
is fixed here and asserted in the tests.

Information value aggregates WOE across a characteristic's bins:

    IV = sum_i ( P(bin i | good) - P(bin i | bad) ) * WOE_i

Conventional reading (Siddiqi):

    < 0.02   unpredictive
    0.02-0.1 weak
    0.1-0.3  medium
    0.3-0.5  strong
    > 0.5    suspiciously strong -- check for leakage before celebrating
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Added to each cell before taking logs, so a bin that happens to contain no
#: bads yields a large finite WOE rather than infinity. Bins this pure should
#: normally be merged away by the binner instead of relying on this.
LAPLACE = 0.5


@dataclass(frozen=True)
class WoeTable:
    """WOE and IV for one characteristic, one row per bin."""

    feature: str
    table: pd.DataFrame  # bin, count, goods, bads, bad_rate, woe, iv_contribution

    @property
    def iv(self) -> float:
        return float(self.table["iv_contribution"].sum())

    @property
    def mapping(self) -> dict:
        """bin label -> WOE, for transforming new data."""
        return dict(zip(self.table["bin"], self.table["woe"]))

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"WoeTable({self.feature!r}, bins={len(self.table)}, iv={self.iv:.4f})"


def woe_table(bins: pd.Series, y: np.ndarray, feature: str = "x") -> WoeTable:
    """Compute the WOE table for an already-binned characteristic.

    ``y`` is 1 for bad (the event being modelled) and 0 for good.
    """
    y = np.asarray(y)
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must be binary 0/1 with 1 = bad")

    df = pd.DataFrame({"bin": pd.Series(bins).astype(object), "y": y})
    grouped = df.groupby("bin", dropna=False, observed=True)["y"].agg(["count", "sum"])
    grouped.columns = ["count", "bads"]
    grouped["goods"] = grouped["count"] - grouped["bads"]

    total_good = grouped["goods"].sum()
    total_bad = grouped["bads"].sum()
    if total_good == 0 or total_bad == 0:
        raise ValueError("need both goods and bads to compute WOE")

    dist_good = (grouped["goods"] + LAPLACE) / (total_good + LAPLACE * len(grouped))
    dist_bad = (grouped["bads"] + LAPLACE) / (total_bad + LAPLACE * len(grouped))

    grouped["bad_rate"] = grouped["bads"] / grouped["count"]
    grouped["woe"] = np.log(dist_good / dist_bad)
    grouped["iv_contribution"] = (dist_good - dist_bad) * grouped["woe"]

    out = grouped.reset_index()[
        ["bin", "count", "goods", "bads", "bad_rate", "woe", "iv_contribution"]
    ]
    return WoeTable(feature=feature, table=out)


def information_value(bins: pd.Series, y: np.ndarray) -> float:
    """IV of one binned characteristic."""
    return woe_table(bins, y).iv


def iv_strength(iv: float) -> str:
    """The conventional verbal label for an IV."""
    if iv < 0.02:
        return "unpredictive"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "suspicious"
