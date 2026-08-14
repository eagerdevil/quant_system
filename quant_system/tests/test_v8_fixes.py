"""
8/14 第二批/第三批修复回归测试 — OOS门禁/权重时间切片/等权基准/权重扰动VaR/模拟盘胜率

运行: pytest tests/test_v8_fixes.py -v
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from backtest_engine import _weight_slice_for, _build_equal_weight_benchmark
from risk_engine import compute_var_weight_uncertainty
from data_engine import compute_market_sentiment


# ================================================================
# 权重时间切片 (todo#8: 消除回测权重前视)
# ================================================================
class TestWeightSlice:
    def test_slice_returns_latest_version_on_or_before_date(self):
        hist = [
            {"date": "20260623", "factor_weights": {"F1": 1.0}},
            {"date": "20260801", "factor_weights": {"F1": 2.0}},
        ]
        assert _weight_slice_for("20260623", hist) == {"F1": 1.0}    # 生效日当天
        assert _weight_slice_for("20260715", hist) == {"F1": 1.0}    # 版本间
        assert _weight_slice_for("20260810", hist) == {"F1": 2.0}    # 最新版本
        assert _weight_slice_for("20260814", hist) == {"F1": 2.0}

    def test_slice_none_when_no_version_in_force(self):
        hist = [{"date": "20260801", "factor_weights": {"F1": 2.0}}]
        assert _weight_slice_for("20260701", hist) is None  # 早于首版权重 → 调用方需警告前视

    def test_slice_dash_date_normalized(self):
        hist = [{"date": "2026-06-23", "factor_weights": {"F1": 1.0}}]
        assert _weight_slice_for("20260625", hist) == {"F1": 1.0}  # 横杠日期也能匹配


# ================================================================
# 池内等权基准 (todo#9: 替代沪深300)
# ================================================================
class TestEqualWeightBenchmark:
    # 5只ETF (实现要求当日>=5只才计收益, 防稀疏日)
    def _etfs(self):
        # 01-02: A +10%, B 0%, C/D/E +5% → 平均 +5%
        # 01-03: A 0%, B +10%, C/D/E +5% → 平均 +5%
        base = {
            "510300": {"name": "A", "klines": [
                {"date": "2026-01-01", "close": 1.0},
                {"date": "2026-01-02", "close": 1.1},   # +10%
                {"date": "2026-01-03", "close": 1.1},   # +0%
            ]},
            "513100": {"name": "B", "klines": [
                {"date": "2026-01-01", "close": 2.0},
                {"date": "2026-01-02", "close": 2.0},   # +0%
                {"date": "2026-01-03", "close": 2.2},   # +10%
            ]},
        }
        for i, code in enumerate(["159915", "512100", "562500"]):
            base[code] = {"name": code, "klines": [
                {"date": "2026-01-01", "close": 1.0},
                {"date": "2026-01-02", "close": 1.05},  # +5%
                {"date": "2026-01-03", "close": 1.1025},# +5%
            ]}
        return base

    def test_daily_average_return_compounded(self):
        days = ["2026-01-01", "2026-01-02", "2026-01-03"]
        bench = _build_equal_weight_benchmark(self._etfs(), days)
        by_date = {b["date"]: b["close"] for b in bench}
        assert abs(by_date["2026-01-02"] - 1.05) < 1e-6    # +5% → nav 1.05
        assert abs(by_date["2026-01-03"] - 1.1025) < 1e-6  # 复利 +5% → 1.1025

    def test_less_than_5_etfs_day_skipped(self):
        etfs = self._etfs()
        del etfs["510300"]["klines"][1]  # 01-02 缺A → 只剩4只 → <5 当日不计收益
        days = ["2026-01-01", "2026-01-02", "2026-01-03"]
        bench = _build_equal_weight_benchmark(etfs, days)
        by_date = {b["date"]: b["close"] for b in bench}
        assert by_date["2026-01-02"] == 1.0                # 01-02 nav不变
        assert abs(by_date["2026-01-03"] - 1.07) < 1e-6    # 01-03: (10+10+5+5+5)/5=7%


# ================================================================
# 权重扰动VaR带 (todo#11: 权重估计不确定性)
# ================================================================
class TestVarWeightUncertainty:
    def _etf_data(self):
        # 两只低相关序列
        n = 60
        rng = np.random.default_rng(7)
        rets_a = rng.normal(0.001, 0.01, n)
        rets_b = -rets_a + rng.normal(0, 0.005, n)
        closes_a = [100.0]
        closes_b = [100.0]
        for ra, rb in zip(rets_a, rets_b):
            closes_a.append(closes_a[-1] * (1 + ra))
            closes_b.append(closes_b[-1] * (1 + rb))
        return {
            "510300": {"kline": [{"close": c} for c in closes_a]},
            "513100": {"kline": [{"close": c} for c in closes_b]},
        }

    def test_returns_band(self):
        holdings = {
            "510300": {"shares": 1000, "current_price": 3.0},
            "513100": {"shares": 1000, "current_price": 3.0},
        }
        res = compute_var_weight_uncertainty(self._etf_data(), holdings, 6000)
        assert res is not None
        assert 0 <= res["var_pct_p5"] <= res["var_pct_p50"] <= res["var_pct_p95"]
        assert res["n_sims"] == 200

    def test_single_holding_returns_none(self):
        holdings = {"510300": {"shares": 1000, "current_price": 3.0}}
        assert compute_var_weight_uncertainty(self._etf_data(), holdings, 3000) is None


# ================================================================
# 情绪评分: total_volume缺失(None)不崩溃 (todo#6: 成交额口径)
# ================================================================
class TestSentimentNoneVolume:
    def test_none_volume_neutral(self):
        breadth = {"up_count": 3000, "down_count": 2000, "limit_up": 50, "limit_down": 30}
        res = compute_market_sentiment(breadth, None, 1800)
        assert 0 <= res["score"] <= 100
        assert "signals" in res
        assert res["signals"]["成交额(亿)"] == 0


# ================================================================
# 模拟盘报告: 历史平仓胜率 (todo#11: 胜率口径)
# ================================================================
class TestPaperWinRate:
    def test_report_contains_win_rate_line(self):
        from paper_trading import format_paper_report, INITIAL_CAPITAL
        trades_history = [
            {"action": "SELL", "pnl_pct": 5.0, "code": "510300"},
            {"action": "SELL", "pnl_pct": -3.0, "code": "513100"},
            {"action": "SELL", "pnl_pct": 8.0, "code": "159915"},
            {"action": "BUY", "pnl_pct": None, "code": "512100"},  # 买入不计入
        ]
        report = format_paper_report(
            [], {"total_assets": 500000.0, "available_cash": 100000.0, "cash_ratio": 20.0,
                 "total_daily_pnl": 0.0, "holdings": []},
            {"regime": "TREND_UP", "regime_name": "趋势上行", "bull_signals": 2,
             "total_signals": 5, "base_position": 0.9},
            {}, {"message": "test"}, [], {"total_return_pct": 1.0},
            trades_history=trades_history,
        )
        assert "历史平仓" in report
        assert "胜率 67%" in report  # 2/3 盈利
