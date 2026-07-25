"""
回测引擎测试 — 手续费 / 止损 / 绩效指标

运行: pytest tests/test_backtest_engine.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from backtest_engine import (
    trade_cost, COMMISSION_RATE, MIN_COMMISSION, SLIPPAGE_BPS,
    _calc_metrics,
)


# ================================================================
# 手续费模型测试
# ================================================================
class TestTradeCost:
    def test_large_trade_rate_based(self):
        """大额交易: 佣金按费率计算"""
        amount, fee = trade_cost(10000, 5.0, is_buy=True)
        expected_commission = max(MIN_COMMISSION, 10000 * 5.0 * COMMISSION_RATE)
        expected_slippage = 10000 * 5.0 * SLIPPAGE_BPS / 10000
        assert abs(fee - (expected_commission + expected_slippage)) < 0.01
        # 买入: amount = trade_amount + fee
        assert amount > 10000 * 5.0

    def test_small_trade_min_commission(self):
        """小账户交易: 最低佣金5元"""
        amount_buy, fee_buy = trade_cost(100, 2.0, is_buy=True)
        # 100股 x 2元 = 200元交易额, 佣金=max(5, 200*0.00025)=5元
        assert fee_buy >= 5.0, f"Fee={fee_buy} should be >= 5 (min commission)"
        # 实际花费: 200 + 5 + slippage
        assert amount_buy > 200.0

    def test_buy_vs_sell(self):
        """买入和卖出的费用处理方向"""
        shares, price = 500, 3.0
        amt_buy, fee_buy = trade_cost(shares, price, is_buy=True)
        amt_sell, fee_sell = trade_cost(shares, price, is_buy=False)
        # 费用相同
        assert abs(fee_buy - fee_sell) < 0.01
        # 买入: 花费 > 成交额; 卖出: 到账 < 成交额
        trade_amount = shares * price
        assert amt_buy > trade_amount
        assert amt_sell < trade_amount

    def test_min_commission_violates_old_model(self):
        """修复验证: 200元交易在新旧模型下差异巨大"""
        shares, price = 100, 2.0
        amt, fee = trade_cost(shares, price, is_buy=True)
        # 旧模型: fee = 200 * 0.0005 = 0.1元
        old_fee = 200 * 0.0005
        # 新模型: fee >= 5元
        assert fee >= 5.0
        # 差距 > 40倍 (证明旧模型严重低估)
        assert fee / old_fee > 40


# ================================================================
# 绩效指标测试
# ================================================================
class TestMetrics:
    def setup_method(self):
        """构造模拟权益曲线: 轻微上涨+波动"""
        np.random.seed(123)
        nav = 100000
        self.curve = []
        for i in range(252):
            ret = np.random.randn() * 0.01 + 0.0003  # 年均+7.5%
            nav *= (1 + ret)
            self.curve.append({
                "date": f"2024-{i//21+1:02d}-{i%28+1:02d}",
                "nav": round(nav, 2),
                "cash": round(nav * 0.2, 2),
                "holdings_value": round(nav * 0.8, 2),
                "holdings_count": 3,
                "benchmark_nav": round(100000 * (1 + 0.0002) ** i, 2),
            })

    def test_metrics_return(self):
        m = _calc_metrics(self.curve, 100000, [])
        assert "total_return_pct" in m
        assert "sharpe_ratio" in m
        assert "max_drawdown_pct" in m
        assert "win_rate" in m

    def test_max_drawdown_non_negative(self):
        m = _calc_metrics(self.curve, 100000, [])
        assert m["max_drawdown_pct"] >= 0

    def test_win_rate_range(self):
        m = _calc_metrics(self.curve, 100000, [])
        assert 0 <= m["win_rate"] <= 100

    def test_final_nav(self):
        m = _calc_metrics(self.curve, 100000, [])
        assert m["final_nav"] > 0
        assert m["initial_capital"] == 100000

    def test_with_trades(self):
        """有交易记录时的指标"""
        trades = [
            {"date": "2024-01-15", "code": "510300", "action": "BUY", "price": 4.5, "shares": 2000, "amount": 9000, "reason": "test"},
            {"date": "2024-06-15", "code": "510300", "action": "SELL", "price": 5.0, "shares": 2000, "amount": 10000, "reason": "test"},
        ]
        m = _calc_metrics(self.curve, 100000, trades)
        assert m["total_trades"] == 2
        # 盈利交易: 买入4.5→卖出5.0
        assert m["winning_trades"] == 1
        assert m["trade_win_rate"] > 0

    def test_empty_curve(self):
        m = _calc_metrics([], 100000, [])
        assert m == {}


# ================================================================
# 止损逻辑测试 (集成在回测逻辑中验证)
# ================================================================
class TestStopLossLogic:
    def test_trade_cost_consistency(self):
        """验证手续费模型的一致性"""
        test_cases = [
            (100, 2.0),    # 小额
            (500, 3.0),    # 中额
            (2000, 5.0),   # 大额
            (10000, 1.5),  # 超大额
        ]
        for shares, price in test_cases:
            amt_buy, fee_buy = trade_cost(shares, price, is_buy=True)
            amt_sell, fee_sell = trade_cost(shares, price, is_buy=False)
            trade_amt = shares * price
            # 买入花费 = 成交额 + 费用
            assert abs(amt_buy - trade_amt - fee_buy) < 0.01
            # 卖得到账 = 成交额 - 费用
            assert abs(amt_sell - trade_amt + fee_sell) < 0.01
            # 费用 >= 最低佣金
            assert fee_buy >= MIN_COMMISSION
            assert fee_sell >= MIN_COMMISSION
