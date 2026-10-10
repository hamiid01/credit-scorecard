"""Credit scorecards, built the way a bank builds them.

Monotonic WOE binning, information-value and collinearity selection, logistic
regression scaled to a points table, discrimination *and* calibration, PSI/CSI
stability monitoring, and reject inference.

>>> from credit_scorecard import develop, report, load_taiwan, CATEGORICAL
>>> X, y = load_taiwan()
>>> result = develop(X, y, categorical=CATEGORICAL)
>>> print(report(result))
"""

from .binning import Binning, fit_categorical, fit_numeric
from .data import CATEGORICAL, PROTECTED, RESTRICTED, load_taiwan, make_synthetic
from .metrics import (
    brier_decomposition,
    brier_score,
    calibration_table,
    gini,
    hosmer_lemeshow,
    ks_statistic,
    score_band_report,
    summary,
)
from .pipeline import DevelopmentResult, develop, report
from .reject_inference import (
    AcceptRejectSplit,
    evaluate_inference,
    fuzzy_augmentation,
    parcelling,
    simulate_accept_reject,
)
from .scorecard import ScalingParams, Scorecard
from .selection import drop_correlated, select_by_iv
from .stability import csi, csi_report, psi, psi_table
from .woe import WoeTable, information_value, iv_strength, woe_table

__all__ = [
    "AcceptRejectSplit", "Binning", "CATEGORICAL", "DevelopmentResult", "PROTECTED",
    "RESTRICTED", "ScalingParams", "Scorecard", "WoeTable", "brier_decomposition",
    "brier_score", "calibration_table", "csi", "csi_report", "develop",
    "drop_correlated", "evaluate_inference", "fit_categorical", "fit_numeric",
    "fuzzy_augmentation", "gini", "hosmer_lemeshow", "information_value",
    "iv_strength", "ks_statistic", "load_taiwan", "make_synthetic", "parcelling",
    "psi", "psi_table", "report", "score_band_report", "select_by_iv",
    "simulate_accept_reject", "summary", "woe_table",
]

__version__ = "1.0.0"
