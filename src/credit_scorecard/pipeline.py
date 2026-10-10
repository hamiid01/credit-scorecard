"""End-to-end development, in the order a scorecard is actually built."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .binning import fit_categorical, fit_numeric
from .metrics import calibration_table, score_band_report, summary
from .scorecard import ScalingParams, Scorecard
from .selection import drop_correlated, points_spread, select_by_iv, vif
from .stability import csi_report, psi, interpret
from .woe import woe_table

log = logging.getLogger(__name__)


@dataclass
class DevelopmentResult:
    card: Scorecard
    iv_decisions: pd.DataFrame
    correlation_drops: pd.DataFrame
    contribution_drops: pd.DataFrame
    vif: pd.DataFrame
    train_metrics: dict
    test_metrics: dict
    score_bands: pd.DataFrame
    calibration: pd.DataFrame
    psi: float
    csi: pd.DataFrame
    features_in: list[str] = field(default_factory=list)
    features_out: list[str] = field(default_factory=list)


def develop(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    categorical: tuple[str, ...] = (),
    test_size: float = 0.3,
    min_iv: float = 0.02,
    max_iv: float = 0.5,
    corr_threshold: float = 0.8,
    min_points_spread: int = 3,
    max_bins: int = 6,
    min_bin_rate: float = 0.05,
    scaling: ScalingParams | None = None,
    seed: int = 42,
) -> DevelopmentResult:
    """Bin, select, fit, scale and validate.

    Binning and selection are fitted on train only. Doing either on the full
    sample leaks the test outcomes into the cut points, which inflates the
    holdout Gini by a few points and is a common way to ship a card that
    underperforms in production.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    X_tr, X_te = X_tr.reset_index(drop=True), X_te.reset_index(drop=True)
    y_tr, y_te = y_tr.reset_index(drop=True), y_te.reset_index(drop=True)

    # --- 1. bin and measure IV, on train only -------------------------------
    ivs, woe_cols = {}, {}
    for col in X.columns:
        is_cat = col in categorical or not pd.api.types.is_numeric_dtype(X[col])
        kwargs = dict(feature=col, max_bins=max_bins, min_bin_rate=min_bin_rate)
        binning = (fit_categorical if is_cat else fit_numeric)(X_tr[col], y_tr, **kwargs)
        table = woe_table(binning.transform(X_tr[col]), y_tr, col)
        ivs[col] = table.iv
        woe_cols[col] = binning.transform(X_tr[col]).map(table.mapping).astype(float)

    kept_iv, iv_decisions = select_by_iv(ivs, min_iv, max_iv)
    log.info("IV filter kept %d of %d characteristics", len(kept_iv), len(ivs))

    # --- 2. prune collinear characteristics ---------------------------------
    woe_frame = pd.DataFrame({c: woe_cols[c] for c in kept_iv})
    kept, corr_drops = drop_correlated(woe_frame, ivs, corr_threshold)
    log.info("correlation filter kept %d of %d", len(kept), len(kept_iv))
    if not kept:
        raise ValueError("no characteristics survived selection; relax min_iv")

    vif_table = vif(woe_frame[kept])

    # --- 3. fit, then prune and refit until every characteristic earns its place
    #
    # Two things get a characteristic thrown out here, and neither is visible
    # before the model is fitted:
    #
    #   * a positive coefficient, which means it scores against its own WOE --
    #     the card would award points for getting riskier, and no credit
    #     committee will sign that off whatever the Gini says;
    #   * a points spread below `min_points_spread`, meaning it cannot move a
    #     decision. It still has to be collected, monitored, documented and
    #     explained, so it is cost without benefit.
    #
    # Neither can be predicted from the IV table, because IV is univariate and
    # a card is not. On the Taiwan panel PAY_2 has IV 0.55 -- "strong" by the
    # conventional reading -- and still moves a score by zero points once
    # PAY_0 is in the model, because PAY_0 already carries that signal. This
    # is not near-singularity: the VIF table tops out below 3. The information
    # is simply reallocated to the strongest characteristic, which no
    # collinearity diagnostic will flag, so contribution has to be measured
    # after the fit rather than inferred before it.
    drops = []
    for _ in range(len(kept)):
        card = Scorecard.fit(
            X_tr[kept], y_tr,
            categorical=tuple(c for c in categorical if c in kept),
            scaling=scaling, max_bins=max_bins, min_bin_rate=min_bin_rate,
        )
        if len(kept) <= 2:
            break
        spread = points_spread(card)

        wrong = spread[spread["wrong_sign"]]
        if not wrong.empty:
            victim = wrong.sort_values("coefficient", ascending=False).iloc[0]
            reason = "wrong sign: scores against its own WOE"
        else:
            weak = spread[spread["spread"] < min_points_spread]
            if weak.empty:
                break
            victim = weak.iloc[0]
            reason = f"points spread {int(victim['spread'])} < {min_points_spread}"

        drops.append({
            "characteristic": victim["characteristic"],
            "points_spread": int(victim["spread"]),
            "coefficient": float(victim["coefficient"]),
            "reason": reason,
        })
        log.info("dropping %s (%s)", victim["characteristic"], reason)
        kept = [c for c in kept if c != victim["characteristic"]]
    contribution_drops = pd.DataFrame(drops)

    s_tr, s_te = card.score(X_tr[kept]), card.score(X_te[kept])
    p_tr, p_te = card.predict_proba(X_tr[kept]), card.predict_proba(X_te[kept])

    return DevelopmentResult(
        card=card,
        iv_decisions=iv_decisions,
        correlation_drops=corr_drops,
        contribution_drops=contribution_drops,
        vif=vif_table,
        train_metrics=summary(y_tr, s_tr, p_tr),
        test_metrics=summary(y_te, s_te, p_te),
        score_bands=score_band_report(y_te, s_te),
        calibration=calibration_table(y_te, p_te),
        psi=psi(s_tr, s_te),
        csi=csi_report(card, X_tr[kept], X_te[kept]),
        features_in=list(X.columns),
        features_out=kept,
    )


def report(result: DevelopmentResult) -> str:
    """Human-readable development summary."""
    c = result.card
    lines = [
        "=" * 72,
        "SCORECARD DEVELOPMENT REPORT",
        "=" * 72,
        "",
        f"Characteristics: {len(result.features_in)} in -> {len(result.features_out)} in the card",
        f"Scaling: {c.scaling.base_score} points at {c.scaling.base_odds:.0f}:1 odds, "
        f"PDO {c.scaling.pdo} (factor {c.scaling.factor:.3f}, offset {c.scaling.offset:.2f})",
        "",
        "-- Selection " + "-" * 59,
        result.iv_decisions.to_string(index=False),
    ]
    if not result.correlation_drops.empty:
        lines += ["", "Dropped for collinearity:", result.correlation_drops.to_string(index=False)]
    if not result.contribution_drops.empty:
        lines += ["", "Dropped after fitting (contribute nothing to the card):",
                  result.contribution_drops.to_string(index=False)]
    serious = result.vif[result.vif["vif"] > 5]
    if serious.empty:
        lines += ["", f"Max VIF {result.vif['vif'].max():.2f} -- no collinearity problem, "
                      "so weak contributors are redundancy, not instability."]
    else:
        lines += ["", "VIF above 5:", serious.to_string(index=False)]

    lines += [
        "",
        "-- Performance " + "-" * 57,
        f"{'':22s}{'train':>12s}{'test':>12s}",
    ]
    for key in ("gini", "ks", "auc", "brier", "reliability", "resolution"):
        lines.append(f"{key:22s}{result.train_metrics[key]:>12.4f}{result.test_metrics[key]:>12.4f}")
    hl = result.test_metrics
    lines += [
        f"{'Hosmer-Lemeshow p':22s}{'':>12s}{hl['hl_p_value']:>12.4f}"
        f"   ({'calibrated' if hl['hl_calibrated_at_5pct'] else 'rejects calibration'} at 5%)",
        "",
        f"PSI train vs test: {result.psi:.4f} ({interpret(result.psi)})",
        "",
        "-- Score bands (test) " + "-" * 50,
        result.score_bands.to_string(index=False),
        "",
        "-- Calibration (test) " + "-" * 50,
        result.calibration.to_string(index=False),
        "",
        "-- The card " + "-" * 60,
        c.points_table.to_string(index=False),
        "",
    ]
    return "\n".join(lines)
