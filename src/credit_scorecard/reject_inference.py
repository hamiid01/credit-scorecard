"""Reject inference.

The problem
-----------
A scorecard is fitted on accepted applicants, because they are the only ones
whose repayment is observed. But it will be *applied* to everyone who applies.
The development sample is therefore selected on the very thing being
predicted, and the sample-selection bias is structural, not a data-quality
accident. Measured performance on accepts is optimistic about the through-the-
door population, and the bias is worst exactly where it matters: near the
cut-off.

What these methods can and cannot do
------------------------------------
Reject inference extrapolates the accept relationship into a region where no
outcome was ever observed. It cannot create information that is not in the
data, and **it cannot be validated**, because the counterfactual -- how the
rejects would have performed -- is never observed by construction. Under
MNAR, where the decline decision used information the model never sees
(a declined reference, something from a phone call), every method here is
biased and no diagnostic will reveal it.

So the honest claim is narrow: inference usually improves the card's ranking
on the full applicant population relative to ignoring rejects, and the
assumptions it rests on should be written down next to the result. It is
included here because knowing *why* it is weak is the part that distinguishes
someone who has built a scorecard from someone who has read about one.

Methods
-------
**Parcelling** assigns a hard good/bad label to each reject, at a bad rate
taken from accepts in the same score band and multiplied by an uplift factor
(conventionally 2-4x, since rejects were declined for a reason).

**Fuzzy augmentation** avoids the hard assignment: each reject enters the
refit twice, once as good and once as bad, weighted by its inferred
probability. This keeps the uncertainty in the sample instead of discarding
it with a coin flip, and is usually described as the better of the two.

That is not what the experiment in ``experiments/`` finds. Once the decline
decision depends on information the model cannot see, fuzzy augmentation
makes the card **worse than ignoring rejects altogether** (Gini -0.035 to
-0.039 on the full population), while parcelling stays marginally positive.
The weights are computed from the KGB model, so under MNAR fuzzy augmentation
confidently propagates the KGB model's own blind spot across every reject;
parcelling's coin flip injects noise that happens to dilute the same bias.
Neither is a reason to trust either method -- it is a reason to measure on
your own selection mechanism rather than inherit the textbook ranking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AcceptRejectSplit:
    """A simulated accept/reject population.

    Real reject data is rarely shareable, so the worked example manufactures
    the selection by applying a cut-off policy to a fully-observed dataset.
    That makes this the one setting where the counterfactual *is* known, so
    the inference can actually be scored -- which is the point of simulating
    it rather than just describing the method.
    """

    X_accept: pd.DataFrame
    y_accept: pd.Series
    X_reject: pd.DataFrame
    y_reject_true: pd.Series  # never visible to the inference; for scoring only
    accept_rate: float


def simulate_accept_reject(
    X: pd.DataFrame,
    y: pd.Series,
    policy_score: np.ndarray,
    accept_rate: float = 0.7,
    noise: float = 0.3,
    seed: int = 0,
) -> AcceptRejectSplit:
    """Split a fully-observed dataset into accepts and rejects by a cut-off.

    ``noise`` perturbs the policy score before cutting, standing in for the
    part of a real decline decision the data never captures. With noise = 0
    the selection is MAR given the score and inference is easy; raising it
    moves the problem towards MNAR, where the methods start to fail.
    """
    rng = np.random.default_rng(seed)
    s = np.asarray(policy_score, dtype=float)
    s = (s - s.mean()) / (s.std() or 1.0)
    perturbed = s + rng.normal(0, noise, len(s))

    cutoff = np.quantile(perturbed, 1 - accept_rate)
    accepted = perturbed >= cutoff

    return AcceptRejectSplit(
        X_accept=X[accepted].reset_index(drop=True),
        y_accept=y[accepted].reset_index(drop=True),
        X_reject=X[~accepted].reset_index(drop=True),
        y_reject_true=y[~accepted].reset_index(drop=True),
        accept_rate=float(accepted.mean()),
    )


def parcelling(
    kgb_card,
    X_accept: pd.DataFrame,
    y_accept: pd.Series,
    X_reject: pd.DataFrame,
    *,
    uplift: float = 2.5,
    n_bands: int = 10,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Assign hard inferred labels to rejects, band by band.

    Returns the combined accept + inferred-reject sample.
    """
    rng = np.random.default_rng(seed)
    s_acc = kgb_card.score(X_accept).to_numpy()
    s_rej = kgb_card.score(X_reject).to_numpy()

    edges = np.unique(np.quantile(s_acc, np.linspace(0, 1, n_bands + 1)))[1:-1]
    band_acc = np.searchsorted(edges, s_acc, side="right")
    band_rej = np.searchsorted(edges, s_rej, side="right")

    y_acc = np.asarray(y_accept)
    inferred = np.zeros(len(X_reject), dtype=int)

    for b in np.unique(band_rej):
        in_acc = band_acc == b
        # a band with no accepts falls back to the overall accept bad rate
        rate = y_acc[in_acc].mean() if in_acc.any() else y_acc.mean()
        rate = min(0.99, rate * uplift)
        m = band_rej == b
        inferred[m] = rng.binomial(1, rate, int(m.sum()))

    X = pd.concat([X_accept, X_reject], ignore_index=True)
    y = pd.Series(np.r_[y_acc, inferred], name=getattr(y_accept, "name", "y"))
    return X, y


def fuzzy_augmentation(
    kgb_card,
    X_accept: pd.DataFrame,
    y_accept: pd.Series,
    X_reject: pd.DataFrame,
    *,
    uplift: float = 2.5,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Duplicate each reject as both outcomes, weighted by its inferred PD.

    Returns ``(X, y, sample_weight)`` ready for a weighted refit. Accepts
    carry weight 1; each reject contributes ``p`` as a bad and ``1 - p`` as a
    good, so the pair still counts as one observation.
    """
    p_rej = np.clip(kgb_card.predict_proba(X_reject) * uplift, 0.0, 0.99)

    X = pd.concat([X_accept, X_reject, X_reject], ignore_index=True)
    y = pd.Series(
        np.r_[np.asarray(y_accept), np.ones(len(X_reject), int), np.zeros(len(X_reject), int)],
        name=getattr(y_accept, "name", "y"),
    )
    weight = np.r_[np.ones(len(X_accept)), p_rej, 1.0 - p_rej]
    return X, y, weight


def evaluate_inference(
    split: AcceptRejectSplit, kgb_card, inferred_card
) -> pd.DataFrame:
    """Score both cards on the full population, using the hidden reject outcomes.

    Only possible because the split was simulated. On real data this table
    cannot be produced -- which is the whole difficulty with reject inference,
    and the reason it is argued about rather than settled.
    """
    from .metrics import gini, ks_statistic

    X_all = pd.concat([split.X_accept, split.X_reject], ignore_index=True)
    y_all = np.r_[np.asarray(split.y_accept), np.asarray(split.y_reject_true)]

    rows = []
    for name, card in (("KGB (accepts only)", kgb_card), ("after inference", inferred_card)):
        s_all = card.score(X_all).to_numpy()
        s_acc = card.score(split.X_accept).to_numpy()
        rows.append({
            "model": name,
            "gini_accepts": gini(np.asarray(split.y_accept), s_acc),
            "gini_full_population": gini(y_all, s_all),
            "ks_full_population": ks_statistic(y_all, s_all),
        })
    return pd.DataFrame(rows)
