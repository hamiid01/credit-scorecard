"""End to end. The thing most likely to be wrong in a scorecard pipeline is
not the maths but the order of operations: binning on the full sample leaks
the holdout into the cut points and quietly inflates every reported number."""

import numpy as np
import pandas as pd
import pytest

from credit_scorecard.pipeline import develop, report


@pytest.fixture(scope="module")
def result(X, y):
    return develop(X, y, categorical=("region",), seed=42)


def test_card_is_built_and_discriminates(result, X, y):
    assert result.test_metrics["gini"] > 0.2
    assert 0 < result.test_metrics["ks"] < 1


def test_weak_characteristics_are_dropped(result):
    """region has almost no signal here, so it must not reach the card."""
    assert len(result.features_out) < len(result.features_in)
    assert "region" not in result.features_out


def test_every_surviving_characteristic_earns_its_place(result):
    """No wrong signs and nothing that cannot move a decision."""
    from credit_scorecard.selection import points_spread

    spread = points_spread(result.card)
    assert not spread["wrong_sign"].any()
    assert (spread["spread"] >= 3).all()


def test_train_and_test_performance_are_close(result):
    """A large gap means the selection leaked or the card is overfitted."""
    gap = result.train_metrics["gini"] - result.test_metrics["gini"]
    assert gap < 0.1, f"train-test Gini gap of {gap:.3f} suggests leakage"


def test_binning_is_fitted_on_train_only(X, y):
    """Changing the split seed must change the cut points. If it does not,
    the binner is seeing the whole sample."""
    a = develop(X, y, categorical=("region",), seed=1)
    b = develop(X, y, categorical=("region",), seed=2)
    shared = set(a.features_out) & set(b.features_out)
    assert shared
    assert any(a.card.binnings[c].breaks != b.card.binnings[c].breaks for c in shared)


def test_random_split_is_stable_by_psi(result):
    """A random train/test split cannot be a population shift."""
    assert result.psi < 0.1
    assert (result.csi["csi"] < 0.1).all()


def test_score_bands_rank_risk(result):
    bands = result.score_bands
    assert bands["bad_rate"].iloc[0] > bands["bad_rate"].iloc[-1]
    assert bands["count"].sum() > 0


def test_report_renders_every_section(result):
    text = report(result)
    for section in ("SCORECARD DEVELOPMENT REPORT", "Selection", "Performance",
                    "Score bands", "Calibration", "The card"):
        assert section in text


def test_refuses_when_nothing_survives_selection(X, y):
    with pytest.raises(ValueError, match="no characteristics survived"):
        develop(X, y, categorical=("region",), min_iv=99.0)


def test_cli_runs_on_a_csv(tmp_path, X, y):
    from credit_scorecard.__main__ import main

    df = X.copy()
    df["default"] = np.asarray(y)
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    out = tmp_path / "card.csv"

    assert main([str(path), "--categorical", "region", "-o", str(out)]) == 0
    card = pd.read_csv(out)
    assert {"characteristic", "bin", "woe", "points"} <= set(card.columns)


def test_cli_rejects_a_missing_target(tmp_path, X, capsys):
    from credit_scorecard.__main__ import main

    path = tmp_path / "d.csv"
    X.to_csv(path, index=False)
    assert main([str(path), "--target", "nope"]) == 2
    assert "not in" in capsys.readouterr().err
