"""Paper-only execution evidence, corporate actions and forward research records.

No trading, network requests or file writes occur on import.
"""
import copy
import hashlib
import html
import json
import math
import os
import re
from datetime import datetime

PAPER_MODEL_VERSION = "paper-2026-10-15-v2"
REFERENCE_WEIGHTS = {"510300": .30, "513500": .25, "513800": .15,
                     "518880": .15}  # cash 15%; research reference, not personal targets


def day(value):
    return str(value or "").replace("-", "")[:8]


def order_id(order):
    payload = {k: order.get(k) for k in ("signal_date", "code", "action", "shares")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def strategy_identity(root, pool):
    """Fingerprint source/config/pool so rule changes cannot silently share a segment."""
    digest = hashlib.sha256()
    for name in ("paper_trading.py", "paper_research.py", "quant_engine.py", "data_engine.py",
                 "backtest_engine.py", "factor_weights.json", "paper_config.json"):
        path = os.path.join(root, name)
        if os.path.isfile(path):
            with open(path, "rb") as stream:
                digest.update(name.encode() + stream.read())
    digest.update(json.dumps(sorted(pool)).encode())
    return {"model": PAPER_MODEL_VERSION, "fingerprint": digest.hexdigest()[:16],
            "code_commit": os.environ.get("GITHUB_SHA", "local"), "pool": sorted(pool),
            "sizing": "volatility_risk_budget", "risk_budget": .01,
            "risk_horizon_sessions": 20, "max_single": .25}


def parse_corporate_actions(code, source):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(source, "html.parser")
    dividends = soup.select_one("table.cfxq")
    splits = soup.select_one("table.fhxq")
    if dividends is None or splits is None:
        raise ValueError("分红/拆分表不完整，不能据此确认无公司行动")
    events = []
    if not dividends.select("tbody tr") or not splits.select("tbody tr"):
        raise ValueError("公司行动表缺少状态行")
    for row in dividends.select("tbody tr"):
        cells = [c.get_text(" ", strip=True) for c in row.select("td")]
        if len(cells) != 5:
            continue
        match = re.search(r"每10份派现金([\d.]+)元", cells[3])
        if not match or not all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", cells[i]) for i in (1, 2, 4)):
            raise ValueError("分红字段无法识别")
        events.append({"code": code, "kind": "dividend", "record_date": day(cells[1]),
                       "ex_date": day(cells[2]), "pay_date": day(cells[4]),
                       "cash_per_share": float(match[1]) / 10})
    for row in splits.select("tbody tr"):
        cells = [c.get_text(" ", strip=True) for c in row.select("td")]
        if len(cells) != 4:
            continue
        ratio = re.fullmatch(r"1[:：]([\d.]+)", cells[3])
        if not ratio or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", cells[1]):
            raise ValueError("份额折算字段无法识别")
        events.append({"code": code, "kind": "split", "ex_date": day(cells[1]),
                       "ratio": float(ratio[1])})
    for event in events:
        event["source_url"] = f"https://fundf10.eastmoney.com/fhsp_{code}.html"
        natural_key = (code, event["kind"], event["ex_date"])
        event["event_id"] = hashlib.sha256(json.dumps(natural_key).encode()).hexdigest()[:24]
    return sorted(events, key=lambda e: (e["ex_date"], e["kind"]))


def fetch_corporate_actions(code):
    from data_engine import _simple_get
    url = f"https://fundf10.eastmoney.com/fhsp_{code}.html"
    response = _simple_get(url, timeout=12)
    if not response:
        return {"ok": False, "events": [], "source_url": url}
    try:
        source = response.text if hasattr(response, "text") else response.read().decode("utf-8")
        return {"ok": True, "events": parse_corporate_actions(code, source), "source_url": url}
    except (ValueError, UnicodeError):
        return {"ok": False, "events": [], "source_url": url}


def apply_corporate_actions(portfolio, actions, since, today):
    """Accrue entitlements before today's open; pay only on the payment date.

    Missing sources are explicit quality warnings. Events before migration are never
    guessed or retroactively credited. Fractional splits stop instead of rounding money away.
    """
    applied = portfolio.setdefault("_applied_actions", [])
    receivables = portfolio.setdefault("_receivables", [])
    warnings, booked = [], []
    for code, result in sorted(actions.items()):
        if not result.get("ok"):
            if code in portfolio:
                warnings.append(f"{code}分红/拆分未核验；账户收益暂定")
            continue
        for event in result.get("events", []):
            event_day = day(event.get("ex_date"))
            if not (since < event_day <= today) or event["event_id"] in applied:
                continue
            position = portfolio.get(code)
            if event["kind"] == "split":
                if position:
                    if event_day < today:
                        raise ValueError(f"{code}漏跑期间发生折算，需要核对实际持仓后恢复")
                    new_shares = position["shares"] * event["ratio"]
                    if event["ratio"] <= 0 or abs(new_shares - round(new_shares)) > 1e-6:
                        raise ValueError(f"{code}折算含零碎份额，需要核对")
                    position["shares"] = round(new_shares)
                    position["cost"] /= event["ratio"]
                    if position.get("economic_cost"):
                        position["economic_cost"] /= event["ratio"]
                    if position.get("stop_reference_cost"):
                        position["stop_reference_cost"] /= event["ratio"]
                    for lot in position.get("lots", []):
                        lot["shares"] *= event["ratio"]
                        lot["cost"] /= event["ratio"]
                    portfolio["_pending"] = [p for p in portfolio.get("_pending", []) if p["code"] != code]
            elif event["kind"] == "dividend":
                # Reconstruct record-date holding from current pre-open position and intervening trades.
                shares = position.get("shares", 0) if position else 0
                for trade in portfolio.get("_trades", []):
                    if trade["code"] == code and day(trade["date"]) > event["record_date"]:
                        shares += trade["shares"] * (-1 if trade["action"] == "BUY" else 1)
                if shares > 0:
                    receivables.append({"event_id": event["event_id"], "code": code,
                                        "amount": shares * event["cash_per_share"],
                                        "pay_date": event["pay_date"]})
                    booked.append({**event, "eligible_shares": shares,
                                   "amount": shares * event["cash_per_share"]})
                    if position and event_day == today:
                        position["stop_reference_cost"] = max(.000001, position.get("stop_reference_cost", position["cost"]) - event["cash_per_share"])
            applied.append(event["event_id"])
    unpaid = []
    for item in receivables:
        if day(item["pay_date"]) <= today:
            portfolio["_available_cash"] += item["amount"]
        else:
            unpaid.append(item)
    portfolio["_receivables"] = unpaid
    return booked, warnings


def execution_quote(bars, today):
    if not bars or day(bars[-1].get("date")) != today:
        return None
    quote = dict(bars[-1])
    if len(bars) > 1:
        quote["prev_close"] = bars[-2]["close"]
    quote["basis"] = "unadjusted"
    return quote


def mark_paper_portfolio(portfolio, raw_data, today):
    for code, pos in portfolio.items():
        if code.startswith("_"):
            continue
        bars = raw_data.get(code) or []
        if bars:
            pos["current_price"] = bars[-1]["close"]
            pos["prev_close"] = bars[-2]["close"] if len(bars) > 1 else None
            pos["_last_mark"] = {"price": bars[-1]["close"], "date": day(bars[-1]["date"])}
        else:
            pos["current_price"] = pos.get("_last_mark", {}).get("price", pos["cost"])
            pos["prev_close"] = None
        mark_date = pos.get("_last_mark", {}).get("date", "unknown")
        pos["stale_date"] = "" if mark_date == today else mark_date
    return portfolio


def fill_constraint(order, quote, participation=.005):
    """Daily volume is a conservative capacity proxy, not proof of opening liquidity."""
    if not quote or quote.get("basis") != "unadjusted":
        return 0, "缺少当日未复权开盘行情"
    required = ("open", "close", "high", "low", "volume")
    if any(not math.isfinite(float(quote.get(k) or 0)) or float(quote.get(k) or 0) <= 0 for k in required):
        return 0, "行情字段无效或无成交量"
    previous = quote.get("prev_close")
    if (previous and quote["high"] == quote["low"] and
            abs(quote["open"] / previous - 1) >= .095):
        return 0, "大幅跳空且全天一字价格，保守假设无法成交"
    # Raw endpoint volumes are normalized to shares by fetch_etf_execution_kline.
    cap = int(quote["volume"] * participation / 100) * 100
    amount = min(int(order["shares"]), cap)
    return amount, "成交量约束缩量" if amount < order["shares"] else ""


def paper_timing_quality(timing, min_coverage=.6):
    result = copy.deepcopy(timing)
    signals = result.get("signal_detail", {})
    valid = sum(v is not None for v in signals.values())
    coverage = valid / len(signals) if signals else 0
    result["data_quality"] = {"valid": valid, "total": len(signals),
                              "coverage": coverage, "sufficient": coverage >= min_coverage}
    result["data_insufficient"] = coverage < min_coverage
    if coverage < min_coverage:
        result["advice"] = "择时数据不足：暂停新增买入及普通目标降仓；有效行情下止损与熔断仍保留"
    elif valid and valid < len(signals):
        # Normalize only when enough evidence exists; missing signals are not negative votes.
        from quant_engine import REGIME_STRATEGY
        effective_bulls = sum(v is True for v in signals.values()) / valid * len(signals)
        target = next((position for threshold, position in ((5, 1.), (4, .85), (3, .65), (2, .4), (1, .2))
                       if effective_bulls >= threshold), .05)
        low, high = REGIME_STRATEGY.get(result.get("regime"), REGIME_STRATEGY["CHOPPY"])["base_position"]
        target = max(low, min(high, target))
        if result.get("force_capped"):
            target = min(target, .9)
        result["base_position"] = round(target, 2)
        result["advice"] = f"有效择时信号{valid}/{len(signals)}，按有效比例归一后目标仓位{target:.0%}；保留市场状态上限"
    return result


def gate_quality_plan(plan, timing, scores):
    result = copy.deepcopy(plan)
    reasons = []
    if timing.get("data_insufficient"):
        result["buy_list"] = []
        result["sell_list"] = [s for s in result.get("sell_list", []) if "目标仓位收敛" not in s.get("reason", "")]
        reasons.append("择时数据不足，未把缺失信号当作市场看跌")
    score_map = {s["code"]: s for s in scores}
    kept = []
    for buy in result.get("buy_list", []):
        premium = score_map.get(buy["code"], {}).get("premium_info", {})
        if premium.get("is_qdii") and premium.get("premium_pct") is None:
            reasons.append(f"{buy['code']}跨境溢价缺失，暂停买入")
        else:
            kept.append(buy)
    result["buy_list"] = kept
    result.setdefault("data_warnings", []).extend(reasons)
    return result


def validate_market_reference(benchmark, index_data, today):
    result = dict(benchmark)
    bars = (index_data.get("000300") or {}).get("data") or []
    start = day(result.get("start_date"))
    dates = {day(b.get("date")) for b in bars}
    valid = bool(start in dates and bars and day(bars[-1].get("date")) == today)
    result["available"] = valid
    result["basis_note"] = "沪深300价格指数市场参照，不含分红，不是跨资产组合的完整业绩基准"
    if not valid:
        result.update(benchmark_start=None, portfolio_return=None, beat_benchmark=None,
                      excess_return=None, message="沪深300起止日期数据不完整，暂不发布跑赢/跑输结论")
    return result


def research_review(portfolio, summary, timing, history, identity, raw_data, actions, today, costs, dividends):
    """Start prospective segments; never rewrite old performance as a new strategy."""
    state = portfolio.setdefault("_research", {})
    fingerprint = identity["fingerprint"]
    if state.get("fingerprint") != fingerprint:
        portfolio.setdefault("_research_segments", []).append({
            "fingerprint": fingerprint, "start_date": today, "start_assets": summary["total_assets"]})
        state = {"fingerprint": fingerprint, "start_date": today, "start_assets": summary["total_assets"],
                 "weights": dict(REFERENCE_WEIGHTS), "reference": None}
        portfolio["_research"] = state
    warnings = []
    fresh = all(execution_quote(raw_data.get(c), today) for c in state["weights"])
    ref = state.get("reference")
    if ref is None and fresh:
        from backtest_engine import trade_cost
        capital = summary["total_assets"]
        ref = {"_available_cash": capital, "_trades": [], "_pending": []}
        for code, weight in state["weights"].items():
            price = raw_data[code][-1]["close"]
            shares = int(capital * weight / price / 100) * 100
            if shares < 100:
                continue
            cost, _ = trade_cost(shares, price)
            ref["_available_cash"] -= cost
            ref[code] = {"shares": shares, "cost": price, "buy_date": today}
        state["reference"] = ref
        state["reference_start_date"] = today
        state["reference_initial_capital"] = capital
        state["reference_strategy_start_assets"] = summary["total_assets"]
    passive_return = None
    reference_quality = False
    if ref is not None and fresh:
        _, action_warnings = apply_corporate_actions(ref, actions, state.get("reference_as_of", today), today)
        warnings.extend(action_warnings)
        nav = ref["_available_cash"] + sum(r["amount"] for r in ref.get("_receivables", []))
        nav += sum(p["shares"] * raw_data[c][-1]["close"] for c, p in ref.items() if not c.startswith("_"))
        passive_return = nav / state["reference_initial_capital"] - 1
        reference_quality = not action_warnings
        state["reference_as_of"] = today
    else:
        warnings.append("固定配置参照缺少当日原始行情，暂停发布其收益")
    previous = history[-1] if history else None
    previous_assets = (previous or {}).get("account", {}).get("total_assets")
    delta = summary["total_assets"] - previous_assets if previous_assets else None
    accrued = sum(e.get("amount", 0) for e in dividends)
    comparable = bool(previous and previous.get("research", {}).get("identity", {}).get("fingerprint") == fingerprint)
    curve = [r["account"]["total_assets"] for r in history if
             r.get("research", {}).get("identity", {}).get("fingerprint") == fingerprint and not r.get("stale_codes")]
    curve.append(summary["total_assets"])
    peak, max_dd = curve[0], 0.
    for value in curve:
        peak = max(peak, value)
        max_dd = min(max_dd, value / peak - 1)
    attribution = {"available": False, "note": "同版本、完整原始估值的相邻归档后才归因；不跨旧价格口径推算"}
    if comparable and not previous.get("stale_codes") and not any(p.get("stale_date") for c,p in portfolio.items() if not c.startswith("_")):
        prior_positions = previous.get("portfolio", {}).get("holdings", [])
        codes = {p["code"] for p in prior_positions} | {t["code"] for t in portfolio.get("_trades", []) if day(t["date"]) == today}
        if all(execution_quote(raw_data.get(c), today) for c in codes):
            contributions = {}
            for pos in prior_positions:
                contributions[pos["code"]] = pos["shares"] * (raw_data[pos["code"]][-1]["close"] - pos["price"])
            hold_pnl = sum(contributions.values())
            trading_pnl = 0.
            for trade in portfolio.get("_trades", []):
                if day(trade["date"]) != today:
                    continue
                close = raw_data[trade["code"]][-1]["close"]
                change = trade["shares"] * (close-trade["price"]) * (1 if trade["action"] == "BUY" else -1)
                contributions[trade["code"]] = contributions.get(trade["code"], 0.) + change
                trading_pnl += change
            residual = delta - (hold_pnl + trading_pnl + accrued - costs) if delta is not None else None
            attribution = {"available": residual is not None and abs(residual) < .1,
                           "hold_price_pnl": hold_pnl, "trade_timing_pnl": trading_pnl,
                           "by_etf_price_pnl": contributions, "dividends": accrued, "costs": costs,
                           "cash_interest": 0., "unexplained": residual,
                           "note": "持仓价格变化＋当期开盘交易相对不交易的影响＋分红−成本；交易影响不是长期择时alpha。拆分或缺档对不上时不宣称精确归因"}
    return {"identity": identity, "segment_start": state["start_date"],
            "segment_return_pct": (summary["total_assets"] / state["start_assets"] - 1) * 100,
            "segment_max_drawdown_pct": max_dd * 100,
            "reference": {"start_date": state.get("reference_start_date"), "weights": state["weights"],
                          "cash_weight": 1 - sum(state["weights"].values()), "return_pct":
                          passive_return * 100 if passive_return is not None else None,
                          "verified": reference_quality,
                          "strategy_same_period_return_pct": (summary["total_assets"] / state["reference_strategy_start_assets"] - 1) * 100 if ref is not None else None,
                          "note": "独立实验参照：一次整手建仓、费用后买入持有，现金不计息；非个人目标配置"},
            "interval": {"from": (previous or {}).get("date"), "to": today,
                         "label": "相邻归档区间（不一定是单日）", "net_pnl": delta,
                         "costs": costs, "dividend_accrual": accrued,
                         "price_pnl_residual": delta + costs - accrued if delta is not None else None,
                         "same_strategy": comparable,
                         "note": "价格损益为净值变动加回成本、扣除新增应收分红的余额；不是独立选基或择时alpha"},
            "attribution": attribution, "warnings": warnings,
            "interpretation": "沪深300只作市场参照；高现金少跌不等于选基能力。新版本从本段起点重新观察，不追溯改写旧成绩。"}


def research_text(report):
    review = report.get("research") or {}
    if not review:
        return ""
    timing = report.get("timing", {})
    quality = timing.get("data_quality", {})
    interval = review["interval"]
    reference = review["reference"]
    lines = [f"实验版本：{review['identity']['model']} / {review['identity']['fingerprint']}",
             f"本段开始：{review['segment_start']} | 分段收益 {review['segment_return_pct']:+.2f}% | 最大回撤 {review['segment_max_drawdown_pct']:.2f}%",
             f"实际仓位 {100-report.get('portfolio',{}).get('cash_ratio',100):.1f}% / 目标 {timing.get('base_position',0)*100:.1f}%",
             f"择时有效数据 {quality.get('valid',0)}/{quality.get('total',0)}；数据不足与市场下跌分别处理"]
    if reference.get("return_pct") is not None:
        lines.append(f"固定配置参照（{reference['start_date']}起）：{reference['return_pct']:+.2f}%" +
                     ("（分红已核验）" if reference.get("verified") else "（公司行动未完全核验，暂定）"))
        lines.append(f"同起点策略收益：{reference['strategy_same_period_return_pct']:+.2f}%（各自扣除交易成本）")
    lines.append(f"区间 {interval.get('from') or '无前期'} → {interval['to']}：费用 {interval['costs']:.2f}元；新增应收分红 {interval['dividend_accrual']:.2f}元")
    if interval.get("net_pnl") is not None:
        lines.append(f"账户区间损益 {interval['net_pnl']:+.2f}元；价格损益余额 {interval['price_pnl_residual']:+.2f}元（不等于选基能力）")
    lines.extend(review.get("warnings", []))
    lines.extend(report.get("execution_warnings", []))
    attribution = review.get("attribution", {})
    if attribution.get("available"):
        lines.append(f"区间归因：原持仓价格 {attribution['hold_price_pnl']:+.2f}元，开盘交易影响 {attribution['trade_timing_pnl']:+.2f}元，分红 {attribution['dividends']:+.2f}元，成本 −{attribution['costs']:.2f}元")
    for event in report.get("order_events", []):
        if event.get("status") != "filled":
            lines.append(f"订单 {event.get('code','')}：{event['status']}，{event.get('reason','')}")
    lines.append(review["interpretation"])
    lines.append("成交：未复权开盘价＋费用；日成交量0.5%为容量上限，无法还原集合竞价排队；历史账户保留旧口径。")
    return "\n".join(lines)


def render_research_card(report):
    text = research_text(report)
    return ('<div class="card"><h2>模拟盘实验与运行状态</h2><pre style="white-space:pre-wrap;line-height:1.7">' +
            html.escape(text) + '</pre></div>') if text else ""
