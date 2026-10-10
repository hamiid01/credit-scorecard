"""Monotonic WOE binning.

Why monotonic
-------------
A scorecard is read, argued about and signed off by people who are not the
modeller. If "higher balance" earns more points in one band and fewer in the
next, the card cannot be explained to a credit committee and cannot be
defended to a regulator. Monotonicity is therefore a *constraint on the fit*,
not a property one hopes the data happens to have -- and it also acts as
regularisation, since a non-monotonic WOE pattern on a continuous
characteristic is usually sampling noise.

The problem
-----------
Choose cut points c_1 < ... < c_k over a characteristic to

    maximise   IV(c)
    subject to WOE being monotonic across bins,
               every bin holding at least a minimum share of observations,
               every bin holding at least one good and one bad,
               k <= max_bins.

Exhaustive search over cut points is exponential, so this uses the standard
greedy agglomeration: start from a fine quantile grid and repeatedly merge the
adjacent pair whose merge destroys the least information value, first to
repair monotonicity violations, then to satisfy the bin-count ceiling. Greedy
merging is not guaranteed optimal, but it is monotone by construction on exit
and is what production scorecard tooling does.

Missing values get their own bin rather than being imputed: in credit data
"this field is blank" is frequently predictive in its own right, and imputing
it silently destroys that signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .woe import LAPLACE

MISSING = "Missing"


def _woe_iv_from_counts(goods: np.ndarray, bads: np.ndarray) -> tuple[np.ndarray, float]:
    """WOE per bin and total IV, from raw good/bad counts."""
    n = len(goods)
    dist_good = (goods + LAPLACE) / (goods.sum() + LAPLACE * n)
    dist_bad = (bads + LAPLACE) / (bads.sum() + LAPLACE * n)
    woe = np.log(dist_good / dist_bad)
    return woe, float(((dist_good - dist_bad) * woe).sum())


def _merge(goods: np.ndarray, bads: np.ndarray, i: int) -> tuple[np.ndarray, np.ndarray]:
    """Merge bin ``i`` into bin ``i+1``."""
    g, b = goods.copy(), bads.copy()
    g[i + 1] += g[i]
    b[i + 1] += b[i]
    return np.delete(g, i), np.delete(b, i)


def _violations(woe: np.ndarray, increasing: bool) -> list[int]:
    """Indices i where the pair (i, i+1) breaks the required direction."""
    d = np.diff(woe)
    return list(np.where(d < 0)[0] if increasing else np.where(d > 0)[0])


def _best_direction(woe: np.ndarray) -> bool:
    """True if WOE should increase with the characteristic.

    Chosen by whichever direction needs fewer repairs, with Spearman sign as
    the tie-break.
    """
    up, down = len(_violations(woe, True)), len(_violations(woe, False))
    if up != down:
        return up < down
    idx = np.arange(len(woe))
    return bool(np.corrcoef(idx, woe)[0, 1] >= 0)


@dataclass
class Binning:
    """Fitted bins for one characteristic."""

    feature: str
    #: interior cut points; bins are (-inf, b0], (b0, b1], ..., (bk, inf)
    breaks: list[float] = field(default_factory=list)
    #: category -> group label, for categorical characteristics
    groups: dict | None = None
    has_missing: bool = False

    @property
    def is_categorical(self) -> bool:
        return self.groups is not None

    @property
    def n_bins(self) -> int:
        if self.is_categorical:
            return len(set(self.groups.values())) + int(self.has_missing)
        return len(self.breaks) + 1 + int(self.has_missing)

    def transform(self, x: pd.Series) -> pd.Series:
        """Map raw values to bin labels."""
        x = pd.Series(x).reset_index(drop=True)
        if self.is_categorical:
            out = x.map(self.groups).astype(object)
            return out.where(~x.isna(), MISSING).fillna(MISSING)
        edges = [-np.inf, *self.breaks, np.inf]
        labels = [f"({_fmt(edges[i])}, {_fmt(edges[i + 1])}]" for i in range(len(edges) - 1)]
        binned = pd.cut(x, bins=edges, labels=labels, include_lowest=True)
        return binned.astype(object).where(~x.isna(), MISSING)


def _fmt(v: float) -> str:
    if v == -np.inf:
        return "-inf"
    if v == np.inf:
        return "inf"
    return f"{v:.4g}"


def fit_numeric(
    x: pd.Series,
    y: np.ndarray,
    *,
    feature: str = "x",
    max_bins: int = 6,
    min_bin_rate: float = 0.05,
    n_prebins: int = 20,
    monotonic: bool = True,
) -> Binning:
    """Greedy monotonic binning of a numeric characteristic."""
    x = pd.Series(x).reset_index(drop=True)
    y = np.asarray(y)
    mask = x.notna().to_numpy()
    xv, yv = x[mask].to_numpy(dtype=float), y[mask]

    if len(np.unique(xv)) <= 1:
        return Binning(feature=feature, breaks=[], has_missing=bool((~mask).any()))

    # --- fine grid on quantiles, deduplicated for ties -----------------------
    qs = np.linspace(0, 1, n_prebins + 1)[1:-1]
    edges = np.unique(np.quantile(xv, qs))
    edges = [e for e in edges if np.isfinite(e)]
    idx = np.searchsorted(edges, xv, side="left")

    n_fine = len(edges) + 1
    goods = np.array([((idx == i) & (yv == 0)).sum() for i in range(n_fine)], dtype=float)
    bads = np.array([((idx == i) & (yv == 1)).sum() for i in range(n_fine)], dtype=float)
    cuts = list(edges)

    min_count = max(1.0, min_bin_rate * len(xv))

    def drop(i: int):
        nonlocal goods, bads, cuts
        goods, bads = _merge(goods, bads, i)
        del cuts[i]

    # --- 1. bins that are too small, empty, or pure --------------------------
    changed = True
    while changed and len(goods) > 2:
        changed = False
        totals = goods + bads
        for i in range(len(goods) - 1):
            if totals[i] < min_count or goods[i] == 0 or bads[i] == 0:
                drop(i)
                changed = True
                break
        else:
            # the last bin cannot merge forward, so merge it backward
            if totals[-1] < min_count or goods[-1] == 0 or bads[-1] == 0:
                drop(len(goods) - 2)
                changed = True

    # --- 2. repair monotonicity, cheapest merge first ------------------------
    if monotonic and len(goods) > 2:
        woe, _ = _woe_iv_from_counts(goods, bads)
        increasing = _best_direction(woe)
        while len(goods) > 2:
            woe, _ = _woe_iv_from_counts(goods, bads)
            bad_pairs = _violations(woe, increasing)
            if not bad_pairs:
                break
            # among the violating pairs, merge the one costing least IV
            best_i, best_iv = None, -np.inf
            for i in bad_pairs:
                _, iv = _woe_iv_from_counts(*_merge(goods, bads, i))
                if iv > best_iv:
                    best_i, best_iv = i, iv
            drop(best_i)

    # --- 3. bin-count ceiling, cheapest merge first --------------------------
    while len(goods) > max_bins:
        best_i, best_iv = None, -np.inf
        for i in range(len(goods) - 1):
            _, iv = _woe_iv_from_counts(*_merge(goods, bads, i))
            if iv > best_iv:
                best_i, best_iv = i, iv
        drop(best_i)

    return Binning(feature=feature, breaks=[float(c) for c in cuts], has_missing=bool((~mask).any()))


def fit_categorical(
    x: pd.Series,
    y: np.ndarray,
    *,
    feature: str = "x",
    max_bins: int = 6,
    min_bin_rate: float = 0.05,
) -> Binning:
    """Bin a categorical characteristic by ordering categories on bad rate.

    Ordering by bad rate before merging makes the grouping monotonic in risk,
    which is what lets a categorical characteristic sit in the same card as a
    numeric one without the card contradicting itself.
    """
    x = pd.Series(x).reset_index(drop=True).astype(object)
    y = np.asarray(y)
    mask = x.notna().to_numpy()

    stats = (
        pd.DataFrame({"c": x[mask], "y": y[mask]})
        .groupby("c", observed=True)["y"]
        .agg(["count", "sum"])
        .rename(columns={"sum": "bads"})
    )
    stats["goods"] = stats["count"] - stats["bads"]
    stats = stats.sort_values("bads" if stats["count"].sum() == 0 else "bads", kind="stable")
    stats["bad_rate"] = stats["bads"] / stats["count"]
    stats = stats.sort_values("bad_rate", kind="stable")

    cats = list(stats.index)
    goods = stats["goods"].to_numpy(dtype=float)
    bads = stats["bads"].to_numpy(dtype=float)
    members = [[c] for c in cats]
    min_count = max(1.0, min_bin_rate * mask.sum())

    def drop(i: int):
        nonlocal goods, bads, members
        goods, bads = _merge(goods, bads, i)
        members[i + 1] = members[i] + members[i + 1]
        del members[i]

    changed = True
    while changed and len(goods) > 2:
        changed = False
        totals = goods + bads
        for i in range(len(goods) - 1):
            if totals[i] < min_count or goods[i] == 0 or bads[i] == 0:
                drop(i)
                changed = True
                break
        else:
            if totals[-1] < min_count or goods[-1] == 0 or bads[-1] == 0:
                drop(len(goods) - 2)
                changed = True

    while len(goods) > max_bins:
        best_i, best_iv = None, -np.inf
        for i in range(len(goods) - 1):
            _, iv = _woe_iv_from_counts(*_merge(goods, bads, i))
            if iv > best_iv:
                best_i, best_iv = i, iv
        drop(best_i)

    groups = {}
    for group in members:
        label = " | ".join(str(c) for c in group)
        for c in group:
            groups[c] = label
    return Binning(feature=feature, groups=groups, has_missing=bool((~mask).any()))
