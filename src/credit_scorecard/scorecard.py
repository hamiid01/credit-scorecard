"""Logistic regression on WOE, scaled to a points table.

The scaling
-----------
A scorecard is a linear model reported in points instead of log-odds, so that
it can be applied by hand and audited line by line. The standard
transformation (Siddiqi, *Credit Risk Scorecards*) fixes two anchors:

    factor = PDO / ln(2)
    offset = base_score - factor * ln(base_odds)
    score  = offset + factor * ln(odds_good)

``PDO`` is "points to double the odds": with PDO = 20, every 20 points doubles
the good:bad odds. The anchors are reporting conventions and carry no
statistical content -- they change what the numbers look like, never who is
ranked above whom.

Distributing the score over characteristics
-------------------------------------------
With y = 1 meaning bad, the fitted model gives

    logit_bad = b0 + sum_i b_i * woe_i        so  ln(odds_good) = -logit_bad

and the score splits across the n characteristics as

    points_ij = offset/n - factor * ( b_i * woe_ij + b0/n )

Summing over i returns ``offset - factor * logit_bad`` exactly, which is the
identity ``test_points_reconstruct_the_model`` asserts. If that test fails the
card does not implement the model it was fitted from, and nothing downstream
is trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .binning import Binning, fit_categorical, fit_numeric
from .woe import WoeTable, woe_table


@dataclass
class ScalingParams:
    """Reporting anchors for the points scale.

    Defaults are the common industry choice: 600 points at 50:1 good:bad odds,
    20 points to double the odds.
    """

    base_score: float = 600.0
    base_odds: float = 50.0
    pdo: float = 20.0

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * np.log(self.base_odds)


@dataclass
class Scorecard:
    """A fitted scorecard: bins, WOE tables, coefficients and a points table."""

    features: list[str] = field(default_factory=list)
    binnings: dict[str, Binning] = field(default_factory=dict)
    woe_tables: dict[str, WoeTable] = field(default_factory=dict)
    coefficients: dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    scaling: ScalingParams = field(default_factory=ScalingParams)

    # ------------------------------------------------------------------ fit

    @classmethod
    def fit(
        cls,
        X: pd.DataFrame,
        y: np.ndarray,
        *,
        categorical: tuple[str, ...] = (),
        scaling: ScalingParams | None = None,
        max_bins: int = 6,
        min_bin_rate: float = 0.05,
        C: float = 1.0,
        monotonic: bool = True,
        sample_weight: np.ndarray | None = None,
    ) -> "Scorecard":
        y = np.asarray(y).astype(int)
        card = cls(features=list(X.columns), scaling=scaling or ScalingParams())

        for col in X.columns:
            is_cat = col in categorical or not pd.api.types.is_numeric_dtype(X[col])
            fit_fn = fit_categorical if is_cat else fit_numeric
            kwargs = dict(feature=col, max_bins=max_bins, min_bin_rate=min_bin_rate)
            if not is_cat:
                kwargs["monotonic"] = monotonic
            card.binnings[col] = fit_fn(X[col], y, **kwargs)
            card.woe_tables[col] = woe_table(card.binnings[col].transform(X[col]), y, col)

        W = card.transform_woe(X)
        # L2 with a weak penalty: WOE features are strongly collinear by
        # construction, and an unpenalised fit produces unstable signs that
        # make the card impossible to sign off.
        model = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
        model.fit(W, y, sample_weight=sample_weight)

        card.coefficients = dict(zip(card.features, model.coef_[0]))
        card.intercept = float(model.intercept_[0])
        return card

    # ------------------------------------------------------------- transform

    def transform_woe(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace each raw value by the WOE of the bin it falls in."""
        out = {}
        for col in self.features:
            bins = self.binnings[col].transform(X[col])
            mapping = self.woe_tables[col].mapping
            # an unseen bin maps to 0.0: neutral, i.e. the population average
            out[col] = bins.map(mapping).astype(float).fillna(0.0).to_numpy()
        return pd.DataFrame(out, index=X.index)

    # ----------------------------------------------------------------- score

    @property
    def points_table(self) -> pd.DataFrame:
        """The card itself: points earned by each bin of each characteristic."""
        n = len(self.features)
        f, o = self.scaling.factor, self.scaling.offset
        rows = []
        for col in self.features:
            beta = self.coefficients[col]
            t = self.woe_tables[col].table
            for _, r in t.iterrows():
                rows.append({
                    "characteristic": col,
                    "bin": r["bin"],
                    "count": int(r["count"]),
                    "bad_rate": r["bad_rate"],
                    "woe": r["woe"],
                    "coefficient": beta,
                    "points": o / n - f * (beta * r["woe"] + self.intercept / n),
                })
        out = pd.DataFrame(rows)
        out["points"] = out["points"].round(0).astype(int)
        # Order characteristics by IV and bins by points ascending, so the card
        # reads top-to-bottom from strongest characteristic to weakest and
        # within each from worst risk to best. A card nobody can read down the
        # page is a card nobody will sign off.
        order = {c: i for i, c in enumerate(
            sorted(self.features, key=lambda c: -self.woe_tables[c].iv)
        )}
        out["_o"] = out["characteristic"].map(order)
        return (out.sort_values(["_o", "points"])
                   .drop(columns="_o").reset_index(drop=True))

    def score(self, X: pd.DataFrame, *, round_points: bool = True) -> pd.Series:
        """Total points per applicant. Higher is better credit quality."""
        W = self.transform_woe(X)
        n = len(self.features)
        f, o = self.scaling.factor, self.scaling.offset
        total = np.zeros(len(X), dtype=float)
        for col in self.features:
            pts = o / n - f * (self.coefficients[col] * W[col].to_numpy() + self.intercept / n)
            total += np.round(pts) if round_points else pts
        return pd.Series(total, index=X.index, name="score")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of default (the bad outcome)."""
        W = self.transform_woe(X)
        logit = self.intercept + sum(
            self.coefficients[c] * W[c].to_numpy() for c in self.features
        )
        return 1.0 / (1.0 + np.exp(-logit))

    def score_to_pd(self, score: np.ndarray | pd.Series) -> np.ndarray:
        """Invert the scaling: points back to a probability of default."""
        score = np.asarray(score, dtype=float)
        log_odds_good = (score - self.scaling.offset) / self.scaling.factor
        return 1.0 / (1.0 + np.exp(log_odds_good))

    # ------------------------------------------------------------------ misc

    @property
    def iv_summary(self) -> pd.DataFrame:
        from .woe import iv_strength

        rows = [
            {"characteristic": c, "iv": self.woe_tables[c].iv,
             "strength": iv_strength(self.woe_tables[c].iv),
             "bins": self.binnings[c].n_bins,
             "coefficient": self.coefficients.get(c, np.nan)}
            for c in self.features
        ]
        return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
