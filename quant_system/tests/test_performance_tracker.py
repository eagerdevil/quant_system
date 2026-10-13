import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from performance_tracker import compute_score_accuracy


def _report(date, score, price):
    return {"_date": date, "scores": [{"code": "ETF1", "name": "ETF1",
                                         "score": score, "grade": "C", "price": price}]}


def test_prediction_accuracy_skips_long_report_gaps():
    reports = [
        _report("20261008", 50, 10.0),
        _report("20261009", 55, 11.0),
        _report("20261020", 40, 9.0),
        _report("20261021", 45, 10.0),
    ]

    result = compute_score_accuracy(reports)["ETF1"]

    # Neither adjacent pair around the multi-session outage is a valid T+1 sample.
    assert result["pred_signals"] == 0
    # The same-day comparison still counts the two near-consecutive pairs.
    assert result["same_day_accuracy"] == 100.0


def test_prediction_accuracy_accepts_normal_weekend_gap():
    reports = [
        _report("20261008", 50, 10.0),
        _report("20261009", 55, 11.0),
        _report("20261012", 60, 12.0),
        _report("20261013", 65, 13.0),
    ]

    result = compute_score_accuracy(reports)["ETF1"]

    assert result["pred_signals"] == 2
    assert result["pred_accuracy"] == 100.0
