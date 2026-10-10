"""Does reject inference actually help?

A real portfolio cannot answer this: the rejects' outcomes are never observed,
so there is nothing to score against. Here the selection is *simulated* from a
fully-observed dataset, which makes the counterfactual available and the
question answerable -- for this selection mechanism, at this accept rate.

The experiment sweeps the share of the decline decision that the model cannot
see. At ``noise = 0`` the selection is MAR given the policy score and
inference should work; as noise rises the mechanism moves towards MNAR, which
is where the method is supposed to break down. Watching *where* it breaks is
the point.
"""

from __future__ import annotations

import pandas as pd

from credit_scorecard import (
    CATEGORICAL, Scorecard, evaluate_inference, fuzzy_augmentation,
    load_taiwan, parcelling, simulate_accept_reject,
)

ACCEPT_RATE = 0.7
UPLIFT = 2.5


def main() -> None:
    X, y = load_taiwan()

    # A plausible legacy policy: decline on recent repayment status and limit.
    policy = -(X["PAY_0"].astype(float)) + 0.000002 * X["LIMIT_BAL"].astype(float)

    rows = []
    for noise in (0.0, 0.3, 0.6, 1.0):
        split = simulate_accept_reject(X, y, policy, ACCEPT_RATE, noise=noise, seed=0)
        kgb = Scorecard.fit(split.X_accept, split.y_accept, categorical=CATEGORICAL)

        Xp, yp = parcelling(kgb, split.X_accept, split.y_accept, split.X_reject, uplift=UPLIFT)
        parcelled = Scorecard.fit(Xp, yp, categorical=CATEGORICAL)

        Xf, yf, w = fuzzy_augmentation(
            kgb, split.X_accept, split.y_accept, split.X_reject, uplift=UPLIFT
        )
        fuzzy = Scorecard.fit(Xf, yf, categorical=CATEGORICAL, sample_weight=w)

        for name, card in (("parcelling", parcelled), ("fuzzy", fuzzy)):
            table = evaluate_inference(split, kgb, card)
            base = table.iloc[0]["gini_full_population"]
            after = table.iloc[1]["gini_full_population"]
            rows.append({
                "unobserved_noise": noise,
                "method": name,
                "accept_bad_rate": round(float(split.y_accept.mean()), 4),
                "reject_bad_rate": round(float(split.y_reject_true.mean()), 4),
                "gini_kgb": round(base, 4),
                "gini_after": round(after, 4),
                "delta": round(after - base, 4),
            })

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv("reject_inference_results.csv", index=False)
    print("\nwrote reject_inference_results.csv")


if __name__ == "__main__":
    main()
