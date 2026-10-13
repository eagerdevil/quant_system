"""Regression tests for leak-resistant optimizer validation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimizer import (
    oos_gate_passed, purged_temporal_split, walk_forward_cv,
    evaluate_params, random_params, perturb_params,
)
from quant_engine import FACTOR_NAMES, DEFAULT_WEIGHTS


def test_purged_split_is_date_based_and_keeps_label_embargo():
    rows = [
        {"date": f"2026{day:04d}", "code": code}
        for day in range(1, 101)
        for code in ("A", "B", "C")
    ]
    train, test, test_start_row, meta = purged_temporal_split(
        rows, train_fraction=0.70, embargo_days=15
    )
    train_dates = {row["date"] for row in train}
    test_dates = {row["date"] for row in test}
    assert len(train_dates) == 55
    assert len(test_dates) == 30
    assert max(train_dates) < "2026071"  # train ends before the 15-session purge
    assert min(test_dates) == meta["split_date"]
    assert rows[test_start_row]["date"] == meta["split_date"]
    assert meta["embargo_dates"] == 15


def test_oos_gate_fails_closed_when_validation_is_missing_or_weak():
    assert not oos_gate_passed({})
    assert not oos_gate_passed({"n": 5, "t": 3.0, "positive_rate": 0.8, "overfit_windows": 2})
    assert oos_gate_passed({"n": 5, "t": 2.1, "positive_rate": 0.8, "overfit_windows": 1})


def test_walk_forward_windows_measure_trading_days_not_panel_rows():
    rows = []
    for day in range(300):
        date = f"2025{day:03d}"
        for asset in range(12):
            factors = {name: float(asset) for name in FACTOR_NAMES}
            rows.append({
                "date": date,
                "code": f"ETF{asset}",
                "factors": factors,
                "forward_return": float(asset) / 100.0,
            })
    params = {
        "factor_weights": dict(DEFAULT_WEIGHTS),
        "premium_steepness": 0.07,
        "premium_threshold": 2.5,
    }
    result = walk_forward_cv(
        rows, params, params, params,
        n_windows=4, train_days=40, test_days=10,
        min_start=240 * 12, embargo_days=5,
    )
    assert result["n_windows"] == 4
    for window in result["windows"]:
        assert window["n_test"] == 10 * 12
        assert window["n_train"] == 40 * 12


def test_ic_evaluation_is_cross_sectional_and_premium_requires_history():
    rows = []
    for day in range(12):
        for asset in range(10):
            rows.append({
                "date": f"2026{day:03d}",
                "code": "513100",
                "factors": {name: float(asset) for name in FACTOR_NAMES},
                "forward_return": float(asset) / 100.0,
            })
    params = {"factor_weights": dict(DEFAULT_WEIGHTS), "premium_steepness": 0.07,
              "premium_threshold": 2.5}
    assert evaluate_params(params, rows) > 0.99

    base = {"factor_weights": dict(DEFAULT_WEIGHTS), "premium_steepness": 0.083,
            "premium_threshold": 3.1, "grade_thresholds": {}}
    assert random_params(base)["premium_steepness"] == 0.083
    changed = perturb_params(base)
    assert changed["premium_steepness"] == 0.083
    assert changed["premium_threshold"] == 3.1
