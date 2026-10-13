"""
模拟盘测试 — T+1执行 / 隔离性 / 计划转换 / 基准前缀

运行: pytest tests/test_paper_trading.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import json
import paper_trading

from paper_trading import (
    load_paper_portfolio, save_paper_portfolio,
    execute_pending, build_pending_from_plan,
    MAX_PENDING_RETRY, INITIAL_CAPITAL, REPORT_PREFIX, gate_paper_plan_freshness,
    validate_paper_portfolio, expire_stale_pending,
)
from backtest_engine import trade_cost
from daily_runner import compute_benchmark_comparison


def test_resend_email_only_sends_existing_report_without_running_trading(tmp_path, monkeypatch):
    report_path = tmp_path / f"{REPORT_PREFIX}20261013.json"
    original = {"date": "20261013", "report": "existing"}
    report_path.write_text(json.dumps(original), encoding="utf-8")
    sent = {}

    monkeypatch.setattr(paper_trading, "TODAY", "20261013")
    monkeypatch.setattr(paper_trading, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(paper_trading, "generate_html_report", lambda report: "<html>existing</html>")
    monkeypatch.setattr(paper_trading, "send_email", lambda html, pw, **kwargs: sent.update(
        html=html, password=pw, kwargs=kwargs) or True)
    monkeypatch.setenv("QQMAIL_AUTH_CODE", "test-secret")
    monkeypatch.setattr(sys, "argv", ["paper_trading.py", "--resend-email"])

    paper_trading.main()

    assert sent["html"] == "<html>existing</html>"
    assert sent["password"] == "test-secret"
    assert "补发" in sent["kwargs"]["subject"]
    assert json.loads(report_path.read_text(encoding="utf-8")) == original


def _mk_pending(code="510300", action="BUY", shares=1000, retries=0, signal_date="20260811"):
    return {"code": code, "name": f"ETF{code}", "action": action,
            "shares": shares, "signal_date": signal_date, "retries": retries,
            "reason": "test"}


# ================================================================
# T+1 执行层
# ================================================================
class TestExecutePending:
    def test_buy_at_open_price(self):
        """买入按今日开盘价成交（非收盘价），现金扣款=成交额+费"""
        cash = 100000.0
        holdings = {}
        price = 4.0
        pending = [_mk_pending(shares=1000)]
        price_map = {"510300": price}  # 今日开盘价
        cash, holdings, trades, remaining = execute_pending(cash, holdings, pending, price_map)
        assert len(trades) == 1
        assert trades[0]["action"] == "BUY"
        assert trades[0]["price"] == price
        assert trades[0]["shares"] == 1000
        assert "510300" in holdings
        amount, _ = trade_cost(1000, price, is_buy=True)
        assert abs(cash - (100000 - amount)) < 0.01
        assert remaining == []

    def test_sell_first_then_buy(self):
        """同日先卖后买: 卖出回笼资金可用于当日买入"""
        cash = 500.0
        holdings = {"510300": {"shares": 1000, "cost": 3.0, "name": "300ETF"}}
        pending = [
            _mk_pending(code="510300", action="SELL", shares=1000),
            _mk_pending(code="159915", action="BUY", shares=10000),
        ]
        price_map = {"510300": 4.0, "159915": 1.5}
        cash, holdings, trades, remaining = execute_pending(cash, holdings, pending, price_map)
        actions = [t["action"] for t in trades]
        assert actions == ["SELL", "BUY"]  # 执行顺序保证
        assert "510300" not in holdings
        assert "159915" in holdings
        assert holdings["159915"]["shares"] % 100 == 0
        assert holdings["159915"]["shares"] > 0
        assert cash >= 0
        assert remaining == []

    def test_min_commission_applied(self):
        """小单受最低佣金5元约束"""
        cash = 100000.0
        holdings = {}
        pending = [_mk_pending(shares=100)]  # 100股 x 4元 = 400元
        price_map = {"510300": 4.0}
        cash, holdings, trades, remaining = execute_pending(cash, holdings, pending, price_map)
        assert trades[0]["fee"] >= 5.0

    def test_affordable_shrink_no_negative_cash(self):
        """预算不足缩量至100整数倍, 现金永不为负"""
        cash = 500.0
        holdings = {}
        pending = [_mk_pending(shares=10000)]  # 计划1万股, 但现金只够约100股
        price_map = {"510300": 4.0}
        cash, holdings, trades, remaining = execute_pending(cash, holdings, pending, price_map)
        assert cash >= 0
        assert remaining == []
        if trades:
            assert holdings["510300"]["shares"] % 100 == 0
            assert holdings["510300"]["shares"] <= 100

    def test_scaled_buy_real_cost_never_negative_cash(self):
        """P0回归: 缩量到100股后实际成本(含最低5元佣金)仍可超现金(实测404.9→-0.14), 必须放弃"""
        for cash in (404.9, 403.0, 401.0, 405.04, 420.0):
            holdings = {}
            pending = [_mk_pending(shares=1000)]  # 计划1000股@4.0=4000元, 现金不足
            price_map = {"510300": 4.0}
            c, h, trades, remaining = execute_pending(cash, holdings, pending, price_map)
            assert c >= 0, f"cash={cash} 修复前会变负数: 实际{c}"
            # 现金足够覆盖实际成本(如420>405.04)时允许买1手
            if cash >= 405.04:
                assert len(trades) == 1 and h["510300"]["shares"] == 100
            else:
                assert trades == [] and h == {}

    def test_insufficient_budget_drop(self):
        """现金不足100股 → 放弃买入"""
        cash = 100.0
        holdings = {}
        pending = [_mk_pending(shares=1000)]
        price_map = {"510300": 4.0}  # 100股需约405元 > 100元
        cash, holdings, trades, remaining = execute_pending(cash, holdings, pending, price_map)
        assert trades == []
        assert "510300" not in holdings
        assert cash == 100.0

    def test_same_day_signal_deferred(self):
        """T+1语义: 今日生成的信号最早明日执行(防同一天重复运行前视偏差)"""
        import paper_trading
        cash = 100000.0
        holdings = {}
        pending = [_mk_pending(signal_date=paper_trading.TODAY)]
        price_map = {"510300": 4.0}
        cash, holdings, trades, remaining = execute_pending(cash, holdings, pending, price_map)
        assert trades == []
        assert len(remaining) == 1
        assert remaining[0]["retries"] == 0  # 延期但不累计重试
        assert "510300" not in holdings

    def test_no_open_price_deferral(self):
        """无今日开盘价 → 延期, retries+1"""
        cash = 100000.0
        holdings = {}
        pending = [_mk_pending()]
        price_map = {"510300": None}  # 今日无数据（缓存滞后/停牌）
        cash, holdings, trades, remaining = execute_pending(cash, holdings, pending, price_map)
        assert trades == []
        assert len(remaining) == 1
        assert remaining[0]["retries"] == 1

    def test_retry_limit_drop(self):
        """连续2日无开盘价 → 丢弃"""
        cash = 100000.0
        holdings = {}
        pending = [_mk_pending(retries=MAX_PENDING_RETRY - 1)]
        price_map = {"510300": None}
        cash, holdings, trades, remaining = execute_pending(cash, holdings, pending, price_map)
        assert trades == []
        assert remaining == []  # 超限丢弃

    def test_sell_no_data_deferral(self):
        """卖出标的今日无数据 → 持仓不动, pending保留"""
        cash = 0.0
        holdings = {"510300": {"shares": 1000, "cost": 3.0, "name": "300ETF"}}
        pending = [_mk_pending(code="510300", action="SELL", shares=1000)]
        price_map = {"510300": None}
        cash, holdings, trades, remaining = execute_pending(cash, holdings, pending, price_map)
        assert trades == []
        assert holdings["510300"]["shares"] == 1000
        assert len(remaining) == 1

    def test_sell_missing_position_void(self):
        """计划卖出但已无持仓 → 该笔作废(不执行不保留)"""
        cash = 0.0
        holdings = {}
        pending = [_mk_pending(code="510300", action="SELL", shares=1000)]
        price_map = {"510300": 4.0}
        cash, holdings, trades, remaining = execute_pending(cash, holdings, pending, price_map)
        assert trades == []
        assert remaining == []


# ================================================================
# 隔离性: 虚拟账户绝不触碰实盘 portfolio.json
# ================================================================
class TestIsolation:
    def test_first_run_creates_account(self, monkeypatch, tmp_path):
        """首次运行创建初始账户, 不创建/不触碰 portfolio.json"""
        monkeypatch.setattr("paper_trading.PAPER_PORTFOLIO_FILE",
                            str(tmp_path / "portfolio_paper.json"))
        portfolio = load_paper_portfolio()
        assert portfolio["_initial_capital"] == INITIAL_CAPITAL
        assert portfolio["_available_cash"] == INITIAL_CAPITAL
        assert portfolio["_pending"] == []
        assert portfolio["_cash_flows"][0]["type"] == "deposit"
        assert not (tmp_path / "portfolio.json").exists()

    def test_state_roundtrip(self, monkeypatch, tmp_path):
        """save → load 字段完整"""
        monkeypatch.setattr("paper_trading.PAPER_PORTFOLIO_FILE",
                            str(tmp_path / "portfolio_paper.json"))
        p = load_paper_portfolio()
        p["_available_cash"] = 50000.0
        p["510300"] = {"shares": 1000, "cost": 4.0, "name": "300ETF"}
        p["_pending"] = [_mk_pending()]
        save_paper_portfolio(p)
        p2 = load_paper_portfolio()
        assert p2["_available_cash"] == 50000.0
        assert p2["510300"]["shares"] == 1000
        assert p2["_pending"][0]["code"] == "510300"

    def test_state_validation_rejects_negative_cash(self):
        """坏账户状态必须在撮合前失败关闭，不能形成负现金模拟成交。"""
        with pytest.raises(ValueError, match="现金不能为负"):
            validate_paper_portfolio({"_initial_capital": INITIAL_CAPITAL,
                                      "_available_cash": -1, "_pending": []})

    def test_state_validation_rejects_non_finite_numbers(self):
        with pytest.raises(ValueError, match="有限数字"):
            validate_paper_portfolio({"_initial_capital": INITIAL_CAPITAL,
                                      "_available_cash": float("nan"), "_pending": []})

    def test_state_validation_rejects_invalid_holding(self):
        with pytest.raises(ValueError, match="必须大于0"):
            validate_paper_portfolio({"_initial_capital": INITIAL_CAPITAL,
                                      "_available_cash": 1000, "_pending": [],
                                      "510300": {"shares": 100, "cost": 0}})


# ================================================================
# 计划 → T+1 待执行订单
# ================================================================
class TestPlanToPending:
    def _plan(self, buy=None, sell=None):
        return {"buy_list": buy or [], "sell_list": sell or []}

    def test_sell_full_amount(self):
        """sell_list → SELL pending(全额)"""
        plan = self._plan(sell=[{"code": "510300", "name": "300ETF", "shares": 1000, "reason": "止损"}])
        pending = build_pending_from_plan(plan)
        assert len(pending) == 1
        assert pending[0]["action"] == "SELL"
        assert pending[0]["shares"] == 1000
        assert pending[0]["retries"] == 0
        assert pending[0]["signal_date"] != ""

    def test_buy_pending(self):
        """buy_list → BUY pending"""
        plan = self._plan(buy=[{"code": "159915", "name": "创业板ETF", "shares": 5000, "reason": "B_买入"}])
        pending = build_pending_from_plan(plan)
        assert len(pending) == 1
        assert pending[0]["action"] == "BUY"
        assert pending[0]["shares"] == 5000

    def test_empty_plan(self):
        """空计划 → 无pending"""
        pending = build_pending_from_plan(self._plan())
        assert pending == []


# ================================================================
# 基准对比前缀隔离
# ================================================================
class TestBenchmarkPrefix:
    def test_prefix_isolates_from_live_reports(self):
        """report_prefix 参数: 模拟盘基准起点=初始资金, 不读实盘 report_*.json 历史"""
        portfolio = {
            "_comment": "paper",
            "_initial_capital": INITIAL_CAPITAL,
            "_available_cash": INITIAL_CAPITAL,
            "_cash_flows": [{"date": "20260812", "type": "deposit", "amount": INITIAL_CAPITAL}],
        }
        index_data = {"000300": {"name": "沪深300",
                                 "data": [{"date": "20260810", "close": 3800.0, "open": 3800.0},
                                          {"date": "20260811", "close": 3820.0, "open": 3810.0}]}}
        bench = compute_benchmark_comparison(portfolio, index_data, report_prefix=REPORT_PREFIX)
        # 起点=初始资金（无report_paper时取Σdeposit; 有report_paper时取首日总资产=初始资金, 两种口径一致）
        assert bench["portfolio_start_value"] == INITIAL_CAPITAL

    def test_gitignore_has_negation(self):
        """.gitignore 必须放行模拟盘报告(否则Actions上状态无法持久化, 每天重置初始账户)"""
        gi = os.path.join(os.path.dirname(__file__), "..", ".gitignore")
        with open(gi, 'r', encoding='utf-8') as f:
            content = f.read()
        assert "!report_paper_*.json" in content
        # 否定规则必须位于 report_*.json 之后才生效
        assert content.index("!report_paper_*.json") > content.index("report_*.json")


class TestPaperFreshnessGate:
    def test_stale_bar_blocks_new_buy_and_keeps_risk_reducing_sell(self):
        plan = {
            "buy_list": [{"code": "510300"}, {"code": "159915"}],
            "sell_list": [{"code": "510300"}],
        }
        scores = [{"code": "510300"}, {"code": "159915"}]
        etf_data = {
            "510300": {"kline": [{"date": "2026-10-12", "close": 4.0}]},
            "159915": {"kline": [{"date": "2026-10-13", "close": 2.0}]},
        }
        filtered = gate_paper_plan_freshness(
            plan, scores, {"510300": {"shares": 100}}, etf_data, "20261013"
        )
        assert filtered["buy_list"] == [{"code": "159915"}]
        assert filtered["sell_list"] == [{"code": "510300"}]
        assert "510300" in filtered["data_warnings"][0]

    def test_missing_bar_blocks_buy(self):
        plan = {"buy_list": [{"code": "510300"}], "sell_list": []}
        filtered = gate_paper_plan_freshness(
            plan, [{"code": "510300"}], {}, {}, "20261013"
        )
        assert filtered["buy_list"] == []
        assert filtered["data_warnings"]


class TestPendingSignalAge:
    def test_keeps_only_previous_session_signal(self):
        bars = [{"date": "2026-10-12", "open": 4.0},
                {"date": "2026-10-13", "open": 4.1}]
        valid = _mk_pending(signal_date="20261012")
        stale = _mk_pending(code="159915", signal_date="20261009")
        kept, expired = expire_stale_pending(
            [valid, stale], {"510300": {"kline": bars},
                             "159915": {"kline": bars}}, "20261013")
        assert kept == [valid]
        assert expired == [stale]

    def test_missing_current_session_keeps_retry_behavior(self):
        pending = _mk_pending(signal_date="20261012")
        kept, expired = expire_stale_pending(
            [pending], {"510300": {"kline": [{"date": "2026-10-12"}]}}, "20261013")
        assert kept == [pending]
        assert expired == []

    def test_current_bar_without_previous_bar_fails_closed(self):
        pending = _mk_pending(signal_date="20261012")
        kept, expired = expire_stale_pending(
            [pending], {"510300": {"kline": [{"date": "2026-10-13"}]}}, "20261013")
        assert kept == []
        assert expired == [pending]
