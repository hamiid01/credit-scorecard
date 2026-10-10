# credit-scorecard — a PD scorecard built the way a bank builds one

Monotonic WOE binning, information-value and collinearity selection, logistic
regression scaled to a **points table**, discrimination *and* calibration,
PSI/CSI stability monitoring, and reject inference.

Not "train a classifier on loan data": the deliverable is a card a credit
committee can read, challenge and sign off.

```bash
make demo        # develop a card on the Taiwan default panel
make reject      # does reject inference actually help?
make test        # 85 tests
```

## Result

UCI *Default of Credit Card Clients* (Taiwan, 30,000 accounts, 22.1% default),
30% holdout:

| | train | test |
|---|---|---|
| **Gini** | 0.5468 | **0.5179** |
| **KS** | 0.4159 | 0.3985 |
| AUC | 0.7741 | 0.7600 |
| Brier | 0.1357 | 0.1391 |
| reliability (calibration error) | 0.0002 | 0.0004 |

21 characteristics in, **11 in the card**. A train–test Gini gap of 0.029 and
a PSI of 0.0003 across the split.

For context, published gradient-boosting benchmarks on this dataset land
around AUC 0.77–0.78. A monotonic, 11-characteristic card that a human can
apply by hand gets 0.760. That gap is the price of explainability, and on this
problem it is about two points of AUC.

An extract of the card itself:

| characteristic | bin | count | bad rate | WOE | points |
|---|---|---|---|---|---|
| PAY_0 | 5 \| 8 \| 6 \| 4 \| 2 \| 7 \| 3 | 2,195 | 69.8% | −2.095 | **1** |
| PAY_0 | 1 | 2,616 | 34.2% | −0.605 | 34 |
| PAY_0 | −1 | 3,971 | 17.0% | +0.329 | 55 |
| PAY_0 | 0 | 10,285 | 12.6% | +0.675 | **62** |
| LIMIT_BAL | (−inf, 30k] | 2,864 | 36.9% | −0.722 | 29 |
| LIMIT_BAL | (360k, inf) | 1,722 | 11.6% | +0.775 | **46** |

`PAY_0` alone swings 61 points — at PDO 20, three doublings of the good:bad
odds.

---

## What the pipeline does, and why each step is there

### 1. Monotonic WOE binning

```
WOE_i = ln( P(bin i | good) / P(bin i | bad) )
IV    = sum_i ( P(bin i | good) - P(bin i | bad) ) * WOE_i
```

Higher WOE means better credit quality. The opposite convention, `ln(bad/good)`,
is equally common and flips every coefficient and every point allocation;
mixing the two is the standard way to ship a card that ranks backwards, so the
convention is fixed and asserted in the tests.

Binning is posed as a constrained optimisation:

> maximise IV, subject to WOE monotonic across bins, every bin holding a
> minimum share of observations and at least one good and one bad, and at most
> `max_bins` bins.

Exhaustive search over cut points is exponential, so this uses greedy
agglomeration from a fine quantile grid, merging the adjacent pair that
destroys the least IV — first to repair monotonicity violations, then to meet
the bin ceiling. Not provably optimal; monotone by construction on exit.

**Monotonicity is a constraint, not a hope.** If "higher balance" earns more
points in one band and fewer in the next, the card cannot be explained to a
credit committee or defended to a regulator. It also regularises: a
non-monotonic WOE pattern on a continuous characteristic is usually noise.

**Missing values get their own bin.** In credit data "this field is blank" is
frequently predictive, and imputing it silently destroys that signal.

### 2. Selection — and why IV is not enough

Three filters, in order: IV floor (0.02), pairwise correlation (0.8), then a
post-fit contribution check.

That third one exists because of something the Taiwan data shows clearly:

> **`PAY_2` has IV 0.55 — "strong" on the conventional scale — and moves a
> score by zero points.**

IV is univariate. A card is not. Once `PAY_0` is in the model it already
carries that signal, so `PAY_2`'s marginal contribution collapses. This is not
collinearity in the usual sense: **max VIF across the card is 2.94**, nowhere
near the conventional threshold of 5, and every pairwise correlation sits
below 0.67. Nothing you can compute before fitting would flag it.

So the pipeline fits, measures each characteristic's **points spread**
(`max − min` over its bins), drops anything below 3 points or carrying a
wrong-signed coefficient, and refits — until every characteristic left can
actually move a decision.

On this data that removes `EDUCATION`, `PAY_2` and `PAY_AMT5` and costs
**0.004 Gini**. Three fewer characteristics to collect, monitor, document and
explain for the life of the card, for four thousandths of a Gini point.

A wrong-signed coefficient gets dropped on sight: it means the characteristic
scores *against* its own WOE, so the card would award points for getting
riskier. No committee signs that off whatever the Gini says.

### 3. Scaling to points

```
factor = PDO / ln(2)
offset = base_score - factor * ln(base_odds)
score  = offset + factor * ln(odds_good)
```

Defaults: 600 points at 50:1 good:bad, PDO 20 — every 20 points doubles the
odds. With `y = 1` meaning bad, the score distributes across characteristics as

```
points_ij = offset/n - factor * ( b_i * woe_ij + b0/n )
```

and summing over characteristics returns `offset − factor · logit_bad`
exactly. `test_points_reconstruct_the_model` asserts that identity to 1e-9. If
it fails, the card is not the model it was fitted from and nothing downstream
is trustworthy.

The anchors are a reporting convention and carry no statistical content —
`test_scaling_anchors_do_not_change_the_ranking` pins that.

### 4. Discrimination *and* calibration

Portfolio write-ups usually stop at Gini. Pricing, provisioning and IFRS 9 all
depend on the other half.

Gini, KS and AUC are invariant to any monotone transformation of the score, so
a model can rank perfectly and still assign nonsense probabilities. Brier with
Murphy's decomposition, a calibration table, and Hosmer–Lemeshow test whether
a predicted 4% actually defaults at 4%.

**Hosmer–Lemeshow rejects calibration here (p = 0.0029).** Reliability is
0.0004, nine of the ten calibration bands sit within 2.0 percentage points of
observed, and the tenth is out by 3.9pp in the highest-risk decile — where
predicted is 71.2% against 67.3% observed, and where every applicant is
declined anyway. On 9,000 holdout accounts the test has the power to reject
deviations this small, which is why the calibration table is reported next to
the p-value rather than the p-value alone. The honest reading is that the card
is slightly conservative at the top of the risk range, not that it is
miscalibrated in any way that changes a decision.

### 5. Stability

PSI over score bands, CSI per characteristic, same formula. Conventional
thresholds (0.1 / 0.25) are industry convention, not a statistical test: PSI
has no null distribution, and on a large enough sample a commercially
irrelevant shift clears 0.25. A trigger to look, not a verdict.

### 6. Reject inference

A card is fitted on accepts, because they are the only ones whose repayment is
observed — then applied to everyone who applies. The development sample is
selected on the very thing being predicted.

**This cannot be validated on real data**: the counterfactual is unobservable
by construction. So `experiments/` simulates the selection from a
fully-observed dataset, which makes the hidden outcomes available, and sweeps
how much of the decline decision the model cannot see — from MAR at the left
to increasingly MNAR at the right.

| unobserved noise | KGB Gini | parcelling | Δ | fuzzy | Δ |
|---|---|---|---|---|---|
| 0.0 (MAR) | 0.4380 | 0.5086 | **+0.071** | 0.4971 | +0.059 |
| 0.3 | 0.4480 | 0.5114 | +0.063 | 0.4940 | +0.046 |
| 0.6 | 0.5124 | 0.5254 | +0.013 | 0.4772 | **−0.035** |
| 1.0 (MNAR) | 0.5177 | 0.5347 | +0.017 | 0.4786 | **−0.039** |

Two things worth stating plainly:

**Fuzzy augmentation goes negative.** The literature generally prefers it to
parcelling, because weighting each reject by its inferred probability keeps
the uncertainty in the sample instead of discarding it with a coin flip. Here,
once the decline decision depends on information the model cannot see, it
makes the card **worse than ignoring rejects altogether**. The weights come
from the KGB model, so under MNAR fuzzy augmentation confidently propagates
the KGB model's own blind spot across every reject; parcelling's coin flip
injects noise that happens to dilute the same bias. Measure on your own
selection mechanism rather than inheriting the textbook ranking.

**The KGB baseline improves as noise rises** (0.438 → 0.518). More noise means
a less cleanly selected accept population, so it is more representative of
through-the-door and the card generalises better. The benefit of inference
shrinks precisely because the problem it solves is shrinking.

---

## Protected characteristics

`SEX` and `MARRIAGE` are **excluded by default**. Using sex in credit pricing
is unlawful in the EU (Directive 2004/113/EC, following *Test-Achats*,
C-236/09); marital status is a prohibited basis under ECOA in the US. They
stay in the raw frame so the exclusion is visible and auditable rather than
silent, and `--protected` will include them for fairness analysis — but a card
that scores them is not deployable.

`AGE` is kept and flagged in `RESTRICTED`. It is a prohibited basis under ECOA
except in an empirically derived, demonstrably sound system — which is what a
scorecard is — and is permitted but scrutinised in the EU. It is dropped here
anyway, by the IV floor.

## Install and use

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

```python
from credit_scorecard import develop, report, load_taiwan, CATEGORICAL

X, y = load_taiwan()
result = develop(X, y, categorical=CATEGORICAL)
print(report(result))
result.card.points_table.to_csv("scorecard.csv", index=False)
```

```bash
python -m credit_scorecard                      # the Taiwan panel
python -m credit_scorecard mydata.csv --target default --pdo 25
```

## Layout

```
src/credit_scorecard/
  woe.py                WOE / IV, with the sign convention fixed
  binning.py            monotonic binning as constrained optimisation
  selection.py          IV filter, correlation pruning, VIF, points spread
  scorecard.py          logistic regression -> points table
  metrics.py            Gini, KS, Brier decomposition, Hosmer-Lemeshow
  stability.py          PSI, CSI
  reject_inference.py   parcelling, fuzzy augmentation, simulated evaluation
  pipeline.py           bin -> select -> fit -> prune -> refit -> validate
  data.py               Taiwan loader, synthetic generator
experiments/            the demo and the reject-inference sweep
tests/                  85 tests
```

## Design notes

**Binning and selection are fitted on train only.** Doing either on the full
sample leaks holdout outcomes into the cut points and inflates the reported
Gini. `test_binning_is_fitted_on_train_only` checks that changing the split
seed changes the cut points.

**L2 with a weak penalty.** WOE features are collinear by construction; an
unpenalised fit produces unstable coefficient signs, and a card with a
wrong-signed characteristic cannot be signed off.

**Coefficients should come out near −1 in magnitude.** WOE is already on the
log-odds scale, so a well-specified characteristic needs little rescaling.
`PAY_0` at −0.77 and `LIMIT_BAL` at −0.40 are reasonable; anything near zero
is the contribution problem above, and anything positive is a defect.

**Laplace smoothing in the WOE.** A bin with no bads would otherwise give
infinite WOE. Bins that pure should be merged away by the binner rather than
rely on this, and `test_every_bin_holds_both_outcomes` checks they are.

**An unseen category scores at WOE 0** — the population average — rather than
crashing or silently scoring as the worst bin.

## Licence

MIT
