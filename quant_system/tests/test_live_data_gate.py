"""Regression tests for stale-data order gating."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from daily_runner import gate_stale_buy_orders, _stale_signal_codes


def test_stale_quotes_block_new_buys_but_keep_sell_orders():
    plan = {
        "buy_list": [{"code": "510300"}, {"code": "159915"}],
        "sell_list": [{"code": "510300"}],
    }
    result = gate_stale_buy_orders(plan, {"510300"})
    assert result["buy_list"] == [{"code": "159915"}]
    assert result["sell_list"] == [{"code": "510300"}]
    assert "510300" in result["data_warnings"][0]


def test_missing_or_old_bar_marks_scored_and_held_symbol_stale():
    scores = [{"code": "ETF_A"}, {"code": "ETF_B"}]
    portfolio = {"ETF_C": {"shares": 100}, "_available_cash": 1000}
    etf_data = {
        "ETF_A": {"kline": [{"date": "2026-10-13", "close": 1.0}]},
        "ETF_B": {"kline": [{"date": "2026-10-12", "close": 1.0}]},
    }
    assert _stale_signal_codes(scores, portfolio, etf_data, {}, "20261013") == {"ETF_B", "ETF_C"}
