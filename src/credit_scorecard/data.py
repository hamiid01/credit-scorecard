"""Dataset loading.

The worked example is the UCI *Default of Credit Card Clients* panel (Taiwan,
2005): 30,000 accounts, 23 characteristics, 22.1% default on the next month's
payment. It is used because the target is a genuine credit event rather than a
proxy, and because the sample is large enough that binning, PSI and reject
inference all have something to bite on.

OpenML serves it with the columns anonymised to ``x1``..``x23``; the published
schema is restored here, because a points table reading ``PAY_0`` can be
discussed with a credit officer and one reading ``x6`` cannot.

Protected characteristics
-------------------------
``SEX`` and ``MARRIAGE`` are **excluded by default**. Using sex in credit
pricing is unlawful in the EU (Directive 2004/113/EC, following *Test-Achats*,
C-236/09), and marital status is a prohibited basis under ECOA in the US. They
remain in the raw frame so the exclusion is visible and auditable rather than
silent, and ``protected=True`` will include them for fairness analysis -- but
a card that scores them is not deployable.

``AGE`` is kept. It is a prohibited basis under ECOA except where used in an
empirically derived, demonstrably sound system, which is what a scorecard is;
in the EU it is permitted but attracts scrutiny. It is flagged in
``RESTRICTED`` so the choice is deliberate.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

TAIWAN_OPENML_ID = 42477
TARGET = "default"

#: OpenML's anonymised names -> the published UCI schema
TAIWAN_COLUMNS = {
    "x1": "LIMIT_BAL", "x2": "SEX", "x3": "EDUCATION", "x4": "MARRIAGE", "x5": "AGE",
    "x6": "PAY_0", "x7": "PAY_2", "x8": "PAY_3", "x9": "PAY_4", "x10": "PAY_5", "x11": "PAY_6",
    "x12": "BILL_AMT1", "x13": "BILL_AMT2", "x14": "BILL_AMT3",
    "x15": "BILL_AMT4", "x16": "BILL_AMT5", "x17": "BILL_AMT6",
    "x18": "PAY_AMT1", "x19": "PAY_AMT2", "x20": "PAY_AMT3",
    "x21": "PAY_AMT4", "x22": "PAY_AMT5", "x23": "PAY_AMT6",
}

#: Unlawful to score on. Excluded unless explicitly requested.
PROTECTED = ("SEX", "MARRIAGE")

#: Lawful but restricted; kept, but the choice should be a conscious one.
RESTRICTED = ("AGE",)

CATEGORICAL = ("EDUCATION", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")


def load_taiwan(
    cache_dir: str | Path = "data",
    protected: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the Taiwan default panel, caching the download as CSV.

    Returns ``(X, y)`` with ``y == 1`` meaning default (bad).
    """
    cache = Path(cache_dir) / "taiwan_default.csv"
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        from sklearn.datasets import fetch_openml

        log.info("downloading dataset %d from OpenML", TAIWAN_OPENML_ID)
        bunch = fetch_openml(data_id=TAIWAN_OPENML_ID, as_frame=True, parser="auto")
        df = bunch.data.rename(columns=TAIWAN_COLUMNS)
        df[TARGET] = bunch.target.astype(int).to_numpy()
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        log.info("cached to %s", cache)

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    if not protected:
        X = X.drop(columns=[c for c in PROTECTED if c in X.columns])
    return X, y


def make_synthetic(
    n: int = 4000, bad_rate: float = 0.15, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    """Small synthetic credit-like frame, for tests that must not touch the network.

    Built so that each characteristic has a known, monotone relationship with
    the log-odds of default, which is what lets the binning tests assert a
    direction rather than merely assert that the code ran.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    income = rng.lognormal(10.5, 0.6, n)
    utilisation = np.clip(rng.beta(2, 5, n), 0, 1)
    age = rng.integers(20, 75, n).astype(float)
    arrears = rng.poisson(0.4, n).astype(float)
    region = rng.choice(["north", "south", "east", "west"], n, p=[0.4, 0.3, 0.2, 0.1])

    logit = (
        -1.0
        - 1.1 * (np.log(income) - 10.5)      # more income -> fewer defaults
        + 2.4 * (utilisation - 0.3)          # more utilisation -> more defaults
        - 0.025 * (age - 45)                 # older -> fewer defaults
        + 0.55 * arrears                     # arrears -> more defaults
        + pd.Series(region).map({"north": -0.2, "south": 0.0, "east": 0.25, "west": 0.4}).to_numpy()
    )
    p = 1 / (1 + np.exp(-logit))
    p = p * (bad_rate / p.mean())
    y = rng.binomial(1, np.clip(p, 0, 0.99))

    X = pd.DataFrame(
        {"income": income, "utilisation": utilisation, "age": age,
         "arrears": arrears, "region": region}
    )
    X.loc[rng.choice(n, size=n // 25, replace=False), "income"] = None  # missing bin
    return X, pd.Series(y, name=TARGET)
