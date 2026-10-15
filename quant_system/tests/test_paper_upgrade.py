"""Offline accounting, failure recovery and prospective research acceptance tests."""
import copy
import json
import sys

import pytest

import paper_trading as paper
from paper_research import (
    order_id, parse_corporate_actions, apply_corporate_actions, fill_constraint,
    execution_quote, paper_timing_quality, gate_quality_plan, strategy_identity,
    research_review, mark_paper_portfolio, render_research_card, REFERENCE_WEIGHTS, validate_market_reference,
)
from quant_engine import TradeDecider


def order(action="BUY", shares=200, code="510300"):
    return {"code": code, "action": action, "shares": shares, "name": "test",
            "signal_date": "20261014", "reason": "test"}


def bars(price=2., today="20261015"):
    return [{"date": "20261014", "open": 1., "close": 1., "high": 1.1, "low": .9, "volume": 1000000},
            {"date": today, "open": price, "close": price+.1, "high": price+.2, "low": price-.1, "volume": 1000000}]


@pytest.fixture(autouse=True)
def fixed_day(monkeypatch):
    monkeypatch.setattr(paper, "TODAY", "20261015")


def test_top_up_updates_average_cost_lots_and_net_cost():
    holdings = {"510300": {"shares": 100, "cost": 1., "buy_date": "20261001"}}
    cash, holdings, trades, _ = paper.execute_pending(10000., holdings, [order()], {"510300": 2.})
    position = holdings["510300"]
    assert position["shares"] == 300
    assert position["cost"] == pytest.approx(500/300)
    assert position["economic_cost"] == pytest.approx((100+10000-cash)/300)
    assert len(position["lots"]) == 2
    assert position["buy_date"] == "20261001"
    assert trades[0]["commission"] == 5.
    assert trades[0]["slippage"] == .04


def test_duplicate_order_and_terminal_receipt_never_fill_twice():
    pending = order()
    audit = []
    cash, holdings, trades, remaining = paper.execute_pending(
        10000., {}, [pending, copy.deepcopy(pending)], {"510300": 2.}, audit=audit)
    assert len(trades) == 1
    cash2, holdings2, trades2, _ = paper.execute_pending(
        cash, holdings, [pending], {"510300": 2.}, completed_ids=[order_id(pending)])
    assert not trades2
    assert cash2 == cash
    assert holdings2["510300"]["shares"] == 200


def test_partial_sell_cost_and_allocation():
    pos = {"510300": {"shares": 1000, "cost": 1., "economic_cost": 1.01,
                      "lots": [{"shares": 1000, "cost": 1.}]}}
    cash, holdings, trades, _ = paper.execute_pending(0., pos, [order("SELL", 200)], {"510300": 2.})
    assert holdings["510300"]["shares"] == 800
    assert holdings["510300"]["lots"][0]["shares"] == 800
    assert trades[0]["realized_pnl"] == pytest.approx(cash-202., abs=.0001)


def test_volume_limit_and_one_price_gap_are_not_infinite_liquidity():
    quote = execution_quote(bars(), "20261015")
    quote["volume"] = 40000
    assert fill_constraint(order(shares=1000), quote)[0] == 200
    quote.update(high=2., low=2., open=2., prev_close=1.)
    assert fill_constraint(order(), quote)[0] == 0
    assert fill_constraint(order(), {**quote, "basis": "adjusted"})[0] == 0


def test_capacity_fill_cancels_remainder_and_obeys_single_position_cap():
    quote = execution_quote(bars(), "20261015")
    quote["volume"] = 400000
    audit = []
    _, holdings, trades, remaining = paper.execute_pending(
        10000., {}, [order(shares=4000)], {"510300": 2.}, quotes={"510300": quote},
        audit=audit, initial_assets=20000)
    assert trades[0]["shares"] == 2000
    assert not remaining
    assert audit[0]["status"] == "partial_cancelled"
    _, holdings, trades, _ = paper.execute_pending(10000., holdings, [order(shares=2000)],
                                                  {"510300": 2.}, initial_assets=16000)
    assert not trades  # already at the 25% cap


DIVIDEND_PAGE = """<table class='cfxq'><tbody><tr><td>2026年</td><td>2026-10-14</td>
<td>2026-10-15</td><td>每10份派现金1.2000元</td><td>2026-10-20</td></tr></tbody></table>
<table class='fhxq'><tbody><tr><td colspan='4'>暂无拆分信息!</td></tr></tbody></table>"""


def test_dividend_receivable_then_payment_without_double_credit():
    events = parse_corporate_actions("510300", DIVIDEND_PAGE)
    assert events[0]["cash_per_share"] == .12
    account = {"_available_cash": 100., "510300": {"shares": 1000, "cost": 2.}}
    actions = {"510300": {"ok": True, "events": events}}
    booked, warnings = apply_corporate_actions(account, actions, "20261014", "20261015")
    assert not warnings
    assert booked[0]["amount"] == 120.
    assert account["_available_cash"] == 100.
    assert account["_receivables"][0]["amount"] == 120.
    apply_corporate_actions(account, actions, "20261014", "20261020")
    apply_corporate_actions(account, actions, "20261014", "20261020")
    assert account["_available_cash"] == 220.
    assert not account["_receivables"]


def test_dividend_record_date_reconstruction_uses_sell_ledger():
    actions = {"510300": {"ok": True, "events": parse_corporate_actions("510300", DIVIDEND_PAGE)}}
    account = {"_available_cash": 0., "_trades": [{"code": "510300", "date": "20261016", "action": "SELL", "shares": 1000}]}
    booked, _ = apply_corporate_actions(account, actions, "20261014", "20261020")
    assert booked[0]["eligible_shares"] == 1000
    assert account["_available_cash"] == 120.


def test_unrecognized_action_source_is_unknown_not_zero():
    with pytest.raises(ValueError):
        parse_corporate_actions("510300", "<html>blocked</html>")
    account = {"_available_cash": 10., "510300": {"shares": 100, "cost": 1.}}
    _, warnings = apply_corporate_actions(account, {"510300": {"ok": False}}, "20261014", "20261015")
    assert warnings


def test_split_adjusts_units_and_cost_and_cancels_pre_split_orders():
    page = DIVIDEND_PAGE.replace("<td colspan='4'>暂无拆分信息!</td>",
                                "<td>2026年</td><td>2026-10-15</td><td>份额分拆</td><td>1:2.0000</td>")
    events = [e for e in parse_corporate_actions("510300", page) if e["kind"] == "split"]
    account = {"_available_cash": 0., "510300": {"shares": 100, "cost": 2.}, "_pending": [order()]}
    apply_corporate_actions(account, {"510300": {"ok": True, "events": events}}, "20261014", "20261015")
    assert account["510300"] == {"shares": 200, "cost": 1.}
    assert not account["_pending"]


def test_missing_timing_data_does_not_authorize_mechanical_sell():
    timing = paper_timing_quality({"signal_detail": {"a": True, "b": None, "c": None}, "base_position": .05})
    plan = {"buy_list": [{"code": "510300"}], "sell_list": [
        {"code": "510300", "reason": "目标仓位收敛"}, {"code": "513800", "reason": "止损触发"}]}
    gated = gate_quality_plan(plan, timing, [])
    assert not gated["buy_list"]
    assert gated["sell_list"] == [{"code": "513800", "reason": "止损触发"}]
    assert plan["buy_list"]  # no mutation


def test_qdii_missing_premium_blocks_new_risk():
    gated = gate_quality_plan({"buy_list": [{"code": "513800"}], "sell_list": []}, {},
                             [{"code": "513800", "premium_info": {"is_qdii": True}}])
    assert not gated["buy_list"]


def test_sufficient_partial_timing_uses_valid_ratio_and_regime_cap():
    result = paper_timing_quality({"signal_detail": {str(i): v for i, v in enumerate(
        [True, True, True, False, False, None, None])}, "base_position": .65, "regime": "CHOPPY"})
    assert not result["data_insufficient"]
    assert result["base_position"] == .85  # 3/5, not 3/7


def test_missing_benchmark_dates_never_publish_outperformance():
    result = validate_market_reference({"start_date": "20260812", "benchmark_start": 100.,
                                        "beat_benchmark": True}, {}, "20261015")
    assert result["benchmark_start"] is None
    assert result["beat_benchmark"] is None


def test_corrected_dividend_amount_does_not_create_duplicate_entitlement():
    first = parse_corporate_actions("510300", DIVIDEND_PAGE)
    changed = parse_corporate_actions("510300", DIVIDEND_PAGE.replace("1.2000", "1.3000"))
    assert first[0]["event_id"] == changed[0]["event_id"]


def test_reference_is_frozen_and_rule_change_starts_new_segment():
    account = {"_available_cash": 10000.}
    raw = {c: bars() for c in REFERENCE_WEIGHTS}
    actions = {c: {"ok": True, "events": []} for c in raw}
    identity = {"fingerprint": "first", "model": "test"}
    first = research_review(account, {"total_assets": 10000.}, {}, [], identity,
                            raw, actions, "20261015", 0., [])
    reference_positions = copy.deepcopy(account["_research"]["reference"])
    for code in raw:
        raw[code] = [{**raw[code][-1], "date": "20261016", "close": 2.2}]
    second = research_review(account, {"total_assets": 10100.}, {}, [], identity,
                             raw, actions, "20261016", 0., [])
    assert second["segment_start"] == first["segment_start"]
    assert second["segment_return_pct"] == pytest.approx(1.)
    assert account["_research"]["reference"]["510300"]["shares"] == reference_positions["510300"]["shares"]
    changed = research_review(account, {"total_assets": 10100.}, {}, [], {**identity, "fingerprint": "second"},
                              raw, actions, "20261016", 0., [])
    assert changed["segment_start"] == "20261016"
    assert changed["segment_return_pct"] == 0.
    assert len(account["_research_segments"]) == 2


def test_prepared_report_recovers_after_file_failure_without_retrading(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "PAPER_PORTFOLIO_FILE", str(tmp_path / "portfolio_paper.json"))
    monkeypatch.setattr(paper, "OUTPUT_DIR", str(tmp_path))
    account = {"_initial_capital": 10000., "_available_cash": 9000., "_pending": []}
    original_write = paper.write_paper_report
    monkeypatch.setattr(paper, "write_paper_report", lambda output: (_ for _ in ()).throw(OSError("disk")))
    output = {"date": "20261015", "account": {"total_assets": 9000.}}
    with pytest.raises(OSError):
        paper.commit_paper_report(account, output, [order()], 9000.)
    persisted = paper.load_paper_portfolio()
    assert persisted["_session"]["stage"] == "report_prepared"
    assert persisted["_pending"] == [order()]
    monkeypatch.setattr(paper, "write_paper_report", original_write)
    paper.recover_prepared_report(persisted)
    assert paper.load_paper_portfolio()["_session"]["stage"] == "complete"
    assert json.loads((tmp_path / "report_paper_20261015.json").read_text()) == output


def test_raw_endpoint_uses_isolated_cache_and_normalizes_lots(monkeypatch):
    import data_engine
    seen = {}
    def fetch(code, url, days, fallback, ver):
        seen.update(url=url, ver=ver)
        return [{"date": "20261015", "volume": 123}]
    monkeypatch.setattr(data_engine, "_fetch_kline_with_cache", fetch)
    result = data_engine.fetch_etf_execution_kline("510300")
    assert "fqt=0" in seen["url"]
    assert "execution_raw" in seen["ver"]
    assert result[0]["volume"] == 12300
    assert result[0]["basis"] == "unadjusted"


def test_strategy_fingerprint_changes_with_rules_but_not_commit_only(tmp_path, monkeypatch):
    source = tmp_path / "quant_engine.py"
    source.write_text("first")
    first = strategy_identity(str(tmp_path), ["510300"])
    monkeypatch.setenv("GITHUB_SHA", "different-code-commit")
    assert strategy_identity(str(tmp_path), ["510300"])["fingerprint"] == first["fingerprint"]
    source.write_text("second")
    assert strategy_identity(str(tmp_path), ["510300"])["fingerprint"] != first["fingerprint"]


def test_cash_and_trading_price_attribution_reconciles():
    identity = {"fingerprint": "test", "model": "test"}
    raw = {c: bars() for c in REFERENCE_WEIGHTS}
    actions = {c: {"ok": True, "events": []} for c in raw}
    previous = {"date": "20261014", "account": {"total_assets": 10000.}, "research": {"identity": identity},
                "portfolio": {"holdings": [{"code": "510300", "shares": 100, "price": 1.}]}}
    account = {"_available_cash": 10000-100-405.04, "510300": {"shares": 300, "cost": 5/3},
               "_trades": [{"code": "510300", "date": "20261015", "shares": 200, "price": 2., "action": "BUY"}]}
    mark_paper_portfolio(account, raw, "20261015")
    assets = account["_available_cash"]+630
    review = research_review(account, {"total_assets": assets}, {}, [previous], identity,
                             raw, actions, "20261015", 5.04, [])
    attr = review["attribution"]
    assert attr["available"]
    assert attr["hold_price_pnl"] == pytest.approx(110.)
    assert attr["trade_timing_pnl"] == pytest.approx(20.)
    assert attr["unexplained"] == pytest.approx(0., abs=1e-8)
    report = {"research": review, "portfolio": {}, "timing": {}}
    assert "区间归因" in render_research_card(report)


def test_end_to_end_resume_after_post_fill_crash(tmp_path, monkeypatch):
    """Use a temporary account and fake data; no network, mail or real portfolio writes."""
    monkeypatch.setattr(paper, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(paper, "PAPER_PORTFOLIO_FILE", str(tmp_path / "portfolio_paper.json"))
    monkeypatch.setattr(paper, "KEY_ETFS", {"510300": "test"})
    monkeypatch.setattr(paper, "USER_WATCHLIST", [])
    monkeypatch.setattr(paper, "PAPER_MIN_ORDER_AMOUNT", 0.)
    monkeypatch.setattr(paper, "is_rest_day", lambda: (False, ""))
    monkeypatch.setattr(paper, "collect_all_data", lambda *args: {"etfs": {"510300": {"kline": bars()}}})
    monkeypatch.setattr(paper, "fetch_etf_execution_kline", lambda code: bars())
    monkeypatch.setattr(paper, "fetch_corporate_actions", lambda code: {"ok": True, "events": []})
    monkeypatch.setattr(paper, "score_all_etfs_cross_sectional", lambda *a, **k: [])
    monkeypatch.setattr(paper, "compute_industry_rotation_score", lambda *a, **k: {})
    monkeypatch.setattr(paper, "compute_market_sentiment", lambda *a, **k: {})
    monkeypatch.setattr(paper, "send_email", lambda *a, **k: pytest.fail("must not mail"))
    monkeypatch.setattr(paper, "send_failure_alert", lambda *a, **k: pytest.fail("must not alert"))
    monkeypatch.setattr(sys, "argv", ["paper_trading.py", "--no-email"])
    paper.save_paper_portfolio({"_initial_capital": 10000., "_available_cash": 9900., "_pending": [order()],
                                "510300": {"shares": 100, "cost": 1.}})
    (tmp_path / "report_paper_20261014.json").write_text(json.dumps({"date": "20261014", "account": {"total_assets": 10000.},
                      "portfolio": {"total_assets": 10000., "holdings": []}}))
    monkeypatch.setattr(paper, "fetch_sw_industry_returns", lambda **k: (_ for _ in ()).throw(RuntimeError("after fill")))
    with pytest.raises(RuntimeError, match="after fill"):
        paper.main()
    checkpoint = paper.load_paper_portfolio()
    assert checkpoint["510300"]["shares"] == 300
    assert checkpoint["_pending"] == []
    assert len(checkpoint["_trades"]) == 1
    monkeypatch.setattr(paper, "fetch_sw_industry_returns", lambda **k: {})
    paper.main()
    recovered = paper.load_paper_portfolio()
    assert recovered["510300"]["shares"] == 300
    assert recovered["_available_cash"] == checkpoint["_available_cash"]
    assert len(recovered["_trades"]) == 1
    assert recovered["_session"]["stage"] == "complete"
    first_contents = (tmp_path / "portfolio_paper.json").read_bytes()
    paper.main()
    assert (tmp_path / "portfolio_paper.json").read_bytes() == first_contents
