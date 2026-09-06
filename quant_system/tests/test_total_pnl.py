"""
9/6 总盈亏(总资产-累计净投入)回归测试

运行: pytest tests/test_total_pnl.py -v

口径: 总盈亏 = 总资产 - 累计净投入(入金-出金)
- 提现/入金不算盈亏(与基准对比净出金修正口径一致)
- 无现金流流水时字段为None, 渲染器显示"—"
- 模拟盘流水只有初始资金deposit → 总盈亏=总资产-初始资金
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from datetime import datetime
from report_mailer import generate_html_report, generate_wechat_markdown


def _portfolio(flows=None, holdings=None, cash=0.0):
    p = {"_available_cash": cash}
    if flows is not None:
        p["_cash_flows"] = flows
    for code, h in (holdings or {}).items():
        p[code] = h
    return p


def _holding(shares, cost, price, prev_close):
    """构造持仓字段(compute_portfolio_summary需要 current_price/prev_close)"""
    return {"shares": shares, "cost": cost, "name": "测试ETF",
            "current_price": price, "prev_close": prev_close}


def _summary(portfolio):
    from daily_runner import compute_portfolio_summary
    return compute_portfolio_summary(portfolio, scores=[])


class TestTotalPnlComputation:
    def test_real_account_with_flows(self):
        """实盘: 4次入金+1次出金, 总盈亏=总资产-净投入"""
        flows = [
            {"date": "20260625", "type": "deposit", "amount": 4000},
            {"date": "20260715", "type": "withdraw", "amount": 2000},
            {"date": "20260823", "type": "deposit", "amount": 1075.73},
        ]
        # 市值 8000 + 现金 428.57 = 总资产 8428.57
        portfolio = _portfolio(flows=flows, holdings={
            "159529": _holding(1400, 1.0, 2.0, 2.05),   # 1400股@1.0 → 值2800
            "513800": _holding(500, 1.0, 10.4, 10.0),   # 500股@1.0 → 值5200
        }, cash=428.57)
        s = _summary(portfolio)
        assert s["total_assets"] == pytest.approx(8428.57, abs=0.01)
        # 净投入 = 4000+1075.73-2000 = 3075.73
        assert s["total_invested"] == pytest.approx(3075.73, abs=0.01)
        assert s["total_deposits"] == pytest.approx(5075.73, abs=0.01)
        assert s["total_withdrawals"] == pytest.approx(2000, abs=0.01)
        assert s["total_pnl_all"] == pytest.approx(8428.57 - 3075.73, abs=0.01)
        assert s["total_pnl_all_pct"] == pytest.approx(
            (8428.57 - 3075.73) / 3075.73 * 100, abs=0.01)

    def test_withdraw_not_counted_as_loss(self):
        """提现2万后总资产降2万: 总盈亏不变(提现不算亏损, 分母同步变小)"""
        flows = [
            {"date": "20260625", "type": "deposit", "amount": 4000},
            {"date": "20260715", "type": "withdraw", "amount": 2000},
        ]
        # 场景A: 提现后资产3000 → 总盈亏=3000-2000=1000
        s = _summary(_portfolio(flows=flows, holdings={
            "159529": _holding(2000, 1.5, 1.5, 1.5),
        }, cash=0))
        assert s["total_pnl_all"] == pytest.approx(1000, abs=0.01)
        assert s["total_pnl_all_pct"] == pytest.approx(50.0, abs=0.01)

    def test_no_flows_returns_none(self):
        """无现金流流水(老账户兜底) → 总盈亏字段None, 渲染示—"""
        s = _summary(_portfolio(holdings={
            "159529": _holding(1000, 1.0, 2.0, 2.0),
        }, cash=100))
        assert s["total_pnl_all"] is None
        assert s["total_invested"] is None

    def test_paper_account(self):
        """模拟盘: 流水只有初始资金deposit → 总盈亏=总资产-初始资金"""
        flows = [{"date": "20260812", "type": "deposit", "amount": 500000}]
        s = _summary(_portfolio(flows=flows, holdings={
            "513500": _holding(100000, 2.0, 1.5, 1.52),
        }, cash=300000))
        # 市值10万股@1.5=15万 + 现金30万 = 总资产45万, 比初始资金50万亏5万
        assert s["total_assets"] == pytest.approx(450000, abs=0.01)
        assert s["total_pnl_all"] == pytest.approx(-50000, abs=0.01)
        assert s["total_invested"] == pytest.approx(500000, abs=0.01)
        assert s["total_pnl_all_pct"] == pytest.approx(-10.0, abs=0.01)


def _sample_report():
    port = {
        "holdings": [], "available_cash": 428.57,
        "total_value": 8000.0, "total_cost": 6000.0, "total_assets": 8428.57,
        "total_pnl": 2000.0, "total_pnl_pct": 33.33, "total_daily_pnl": -12.5,
        "cash_ratio": 5.1,
        "total_invested": 3075.73, "total_deposits": 5075.73,
        "total_withdrawals": 2000.0, "total_pnl_all": 5352.84,
        "total_pnl_all_pct": 174.03,
    }
    timing = {
        "regime_name": "震荡整理", "bull_signals": 2, "total_signals": 6,
        "base_position": 0.4, "regime_stop_loss": -0.08,
        "regime_buy_grade_min": "B_买入", "signal_detail": {},
    }
    return {"date": "20260906", "timing": timing, "portfolio": port,
            "scores": [], "plan": {}, "benchmark": {}}


class TestTotalPnlRendering:
    def test_html_report_contains_total_pnl(self):
        html = generate_html_report(_sample_report())
        assert "总盈亏" in html
        assert "累计净投入" in html
        assert "+5352.84元" in html
        assert "3,075.73元" in html  # 累计净投入千分位

    def test_html_report_no_flows_shows_dash(self):
        r = _sample_report()
        r["portfolio"]["total_pnl_all"] = None
        r["portfolio"]["total_invested"] = None
        html = generate_html_report(r)
        assert "总盈亏" in html
        assert ">—<" in html  # 速览和账户概览各一个"—"
        assert "None" not in html

    def test_wechat_markdown_contains_total_pnl(self):
        md = generate_wechat_markdown(_sample_report())
        assert "总盈亏" in md
        assert "+5352.84元" in md
