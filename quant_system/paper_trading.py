#!/usr/bin/env python
"""
模拟盘（Paper Trading）— 纯程序验证量化策略可操作性
====================================================
设计原则:
1. 与实盘完全隔离: 只读写 portfolio_paper.json / report_paper_*.json，绝不触碰 portfolio.json
2. 复用与实盘完全相同的决策代码: collect_all_data → score_all_etfs_cross_sectional → MarketTiming → TradeDecider
3. T+1 真实执行: T日收盘生成计划 → T+1日开盘价成交（与回测引擎同一语义，用真实次日数据）
4. 零人工干预: cron-job.org 触发 GitHub Actions 每天 18:10 北京时间运行，状态 git 提交持久化

用法: python paper_trading.py [--force-weekend]
"""
import json, sys, os, io, logging, traceback
from datetime import datetime

# 先配置日志（再 import daily_runner，其 basicConfig 因 root 已有 handler 而 no-op，
# 避免每天误建一个空的 daily_*.log）
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"paper_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("paper")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from daily_runner import (
    update_portfolio_prices, compute_portfolio_summary,
    compute_benchmark_comparison, is_rest_day, send_failure_alert, OUTPUT_DIR
)
from data_engine import (
    collect_all_data, KEY_ETFS, USER_WATCHLIST,
    fetch_etf_fund_nav, compute_etf_premium, compute_etf_premium_history,
    compute_market_sentiment, fetch_sw_industry_returns
)
from quant_engine import (
    MarketTiming, TradeDecider, score_all_etfs_cross_sectional,
    _apply_premium_penalty, classify_market_regime,
    compute_industry_rotation_score, get_etf_industry_momentum, OPTIMIZED_PARAMS
)
from backtest_engine import trade_cost, COMMISSION_RATE, SLIPPAGE_BPS
from report_mailer import generate_html_report, send_email
from performance_tracker import load_history, compute_equity_curve

TODAY = datetime.now().strftime("%Y%m%d")
PAPER_PORTFOLIO_FILE = os.path.join(OUTPUT_DIR, "portfolio_paper.json")
INITIAL_CAPITAL = 500000.0   # 虚拟初始资金
MAX_PENDING_RETRY = 2        # 连续2个交易日无开盘价则放弃该笔计划
REPORT_PREFIX = "report_paper_"


# ============================================================
# 虚拟账户: 加载 / 原子保存（与实盘 portfolio.json 完全隔离）
# ============================================================
def load_paper_portfolio():
    """加载模拟盘虚拟账户。首次运行创建初始账户（初始资金10万）。
    只读写 portfolio_paper.json，绝不触碰实盘 portfolio.json"""
    if os.path.exists(PAPER_PORTFOLIO_FILE):
        with open(PAPER_PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "_comment": "模拟盘虚拟账户（与 portfolio.json 完全隔离）| 初始资金10万 | 由 paper_trading.py 每日18:10自动运行",
        "_initial_capital": INITIAL_CAPITAL,
        "_available_cash": INITIAL_CAPITAL,
        "_cash_flows": [{"date": TODAY, "type": "deposit", "amount": INITIAL_CAPITAL,
                         "note": "模拟盘初始资金"}],
        "_pending": [],
    }


def save_paper_portfolio(portfolio):
    """原子写虚拟账户（先写临时文件再替换）"""
    tmp = PAPER_PORTFOLIO_FILE + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(portfolio, f, ensure_ascii=False, indent=2)
    os.replace(tmp, PAPER_PORTFOLIO_FILE)


# ============================================================
# 执行层（T+1 开盘价成交）— 照搬回测引擎语义, 用真实次日数据
# ============================================================
def execute_pending(cash, holdings, pending, price_map):
    """按昨日信号以今日开盘价成交（先卖后买）。
    纯函数, 不访问全局状态。

    price_map: {code: 今日开盘价 | None} — None 表示该标的今日无数据
               （缓存滞后/停牌），该笔延期执行, retries+1, 超 MAX_PENDING_RETRY 丢弃。
    返回: (cash, holdings, executed_trades, remaining_pending)
    """
    executed = []
    remaining = []
    # 先执行全部卖出, 再执行买入 — 保证卖出回笼资金可用于当日买入
    # (注: 回测引擎同款排序是BUY优先即先买后卖, 与注释矛盾 — 模拟盘按真实可执行顺序先卖后买)
    for p in sorted(pending, key=lambda x: 0 if x["action"] == "SELL" else 1):
        code = p["code"]
        exec_price = price_map.get(code)
        # T+1 语义: 今日生成的信号最早明日执行
        # (防同一交易日重复运行——否则今日开盘价成交今日信号=前视偏差; 延期不累计retries)
        if p.get("signal_date", "0") >= TODAY:
            remaining.append(p)
            continue
        if not exec_price or exec_price <= 0:
            # 今日无开盘价: 延期重试
            p["retries"] = p.get("retries", 0) + 1
            if p["retries"] >= MAX_PENDING_RETRY:
                logger.warning(f"[执行] {code} {p.get('name','')} 连续{p['retries']}日无开盘价, "
                               f"放弃{p['action']}({p.get('reason','')})")
            else:
                remaining.append(p)
            continue

        if p["action"] == "SELL":
            pos = holdings.get(code)
            if pos is None:
                continue  # 计划卖出但已无持仓（如曾被其他信号卖出）→ 直接作废
            sell_shares = min(p["shares"], pos["shares"])
            sell_amount, fee = trade_cost(sell_shares, exec_price, is_buy=False)
            cash += sell_amount
            pnl_pct = (exec_price / pos["cost"] - 1) * 100 if pos.get("cost") else 0
            executed.append({
                "date": TODAY, "code": code, "name": pos.get("name", code),
                "action": "SELL", "price": round(exec_price, 4),
                "shares": sell_shares, "amount": round(sell_amount, 2),
                "fee": round(fee, 2), "pnl_pct": round(pnl_pct, 2),
                "reason": p.get("reason", "")
            })
            if sell_shares >= pos["shares"]:
                del holdings[code]
            else:
                pos["shares"] -= sell_shares
        else:
            # BUY: 防止重复持仓
            if code in holdings:
                continue
            buy_shares = p["shares"]
            cost_with_fee, _ = trade_cost(buy_shares, exec_price, is_buy=True)
            if cost_with_fee > cash:
                # 开盘价高于预算: 按可负担缩量至100整数倍, 不足100股放弃, 现金永不为负
                per_unit = exec_price * (1 + COMMISSION_RATE) + exec_price * SLIPPAGE_BPS / 10000
                affordable = int(cash / per_unit / 100) * 100
                if affordable < 100:
                    logger.info(f"[执行] {code} 预算不足100股(@{exec_price}), 放弃买入")
                    continue
                buy_shares = affordable
                cost_with_fee, _ = trade_cost(buy_shares, exec_price, is_buy=True)
                if cost_with_fee > cash:
                    # P0修复: 估算公式漏了最低5元佣金, 实际成本可能仍超现金(实测cash=404.9→-0.14)
                    logger.info(f"[执行] {code} 实际成本{cost_with_fee:.2f}>现金{cash:.2f}, 放弃买入")
                    continue
            if buy_shares < 100:
                continue
            cash -= cost_with_fee
            holdings[code] = {"shares": buy_shares, "cost": exec_price, "name": p.get("name", code),
                              "buy_date": TODAY}  # 8/14长持改造: 记录买入日期(最短持有期20交易日判断)
            executed.append({
                "date": TODAY, "code": code, "name": p.get("name", code),
                "action": "BUY", "price": round(exec_price, 4),
                "shares": buy_shares, "amount": round(cost_with_fee, 2),
                "fee": round(cost_with_fee - buy_shares * exec_price, 2),
                "reason": p.get("reason", "")
            })
    return cash, holdings, executed, remaining


def build_pending_from_plan(plan):
    """将今日生成的计划转为明日待执行订单（T+1语义）。
    sell_list 全额卖出; buy_list 按计划股数（次日按开盘价重算可负担量）"""
    pending = []
    for s in (plan.get("sell_list") or []):
        pending.append({
            "code": s["code"], "name": s.get("name", s["code"]),
            "action": "SELL", "shares": s.get("shares", 0),
            "signal_date": TODAY, "retries": 0, "reason": s.get("reason", "")
        })
    for b in (plan.get("buy_list") or []):
        pending.append({
            "code": b["code"], "name": b.get("name", b["code"]),
            "action": "BUY", "shares": b.get("shares", 0),
            "signal_date": TODAY, "retries": 0, "reason": b.get("reason", "")
        })
    return pending


# ============================================================
# 报告文本
# ============================================================
def format_paper_report(executed_trades, port_summary, timing_result, plan,
                        benchmark, final_pending, account, trades_history=None):
    """【模拟盘日报】简洁文本"""
    lines = []
    lines.append("=" * 64)
    lines.append(f"【模拟盘日报】 {TODAY}")
    lines.append(f"虚拟账户: 初始 {INITIAL_CAPITAL:,.0f} 元 | 总资产 {port_summary['total_assets']:,.2f} 元 "
                 f"| 累计收益 {account.get('total_return_pct', 0):+.2f}%")
    # 8/14: 历史平仓胜率(真实交易口径, 来自_trades流水)
    sell_pnls = [t.get("pnl_pct") for t in (trades_history or [])
                 if t.get("action") == "SELL" and t.get("pnl_pct") is not None]
    if sell_pnls:
        wins = sum(1 for p in sell_pnls if p > 0)
        lines.append(f"历史平仓: {len(sell_pnls)}笔 | 胜率 {wins/len(sell_pnls)*100:.0f}% | "
                     f"平均盈亏 {sum(sell_pnls)/len(sell_pnls):+.1f}%")
    lines.append(f"现金 {port_summary.get('available_cash', 0):,.2f} 元 | "
                 f"持仓仓位 {100 - port_summary.get('cash_ratio', 100):.1f}% | "
                 f"今日盈亏 {port_summary.get('total_daily_pnl', 0):+.2f} 元")
    lines.append("-" * 64)
    if executed_trades:
        lines.append(f"今日成交 {len(executed_trades)} 笔 (T+1 开盘价):")
        for t in executed_trades:
            lines.append(f"  {t['action']} {t['name']}({t['code']}) {t['shares']}股 @{t['price']} "
                         f"金额 {t['amount']:.2f} 元 {t.get('reason', '')}")
    else:
        lines.append("今日无成交")
    if port_summary.get("holdings"):
        lines.append("-" * 64)
        lines.append("当前持仓:")
        for h in port_summary["holdings"]:
            lines.append(f"  {h['name']}({h['code']}) {h['shares']}股 "
                         f"成本{h['cost']:.4f} 现价{h['price']:.4f} "
                         f"盈亏{h['pnl_pct']:+.2f}% 仓位{h.get('weight', 0):.1f}%")
    lines.append("-" * 64)
    lines.append(f"市场状态: {timing_result.get('regime_name', timing_result.get('regime', ''))} "
                 f"({timing_result.get('regime', '')}) | 看多信号 "
                 f"{timing_result.get('bull_signals', 0)}/{timing_result.get('total_signals', 0)} "
                 f"| 建议仓位 {timing_result.get('base_position', 0) * 100:.0f}%")
    if plan.get("buy_list"):
        lines.append("明日买入计划: " + "、".join(
            f"{b['name']}({b['code']}) {b['shares']}股({b.get('grade', '')})" for b in plan["buy_list"]))
    if plan.get("sell_list"):
        lines.append("明日卖出计划: " + "、".join(
            f"{s['name']}({s['code']})" for s in plan["sell_list"]))
    if benchmark.get("message"):
        lines.append(f"基准对比: {benchmark['message']}")
    if final_pending:
        lines.append(f"待执行订单: {len(final_pending)} 笔")
    lines.append("=" * 64)
    return "\n".join(lines)


# ============================================================
# 主流程（每日状态机）
# ============================================================
def main():
    is_rest, reason = is_rest_day()
    if is_rest:
        logger.info(f"{reason}，休市跳过（模拟盘状态零改动）")
        return

    # 8/14: 邮件幂等 — 当日模拟盘报告已生成则跳过(防cron双触发重复邮件)
    import glob as _glob
    today_iso = TODAY
    if "--force" not in sys.argv and "--rerun" not in sys.argv:
        existing = [f for f in _glob.glob(os.path.join(OUTPUT_DIR, f"report_paper_{today_iso}.json"))]
        if existing:
            logger.info(f"今日模拟盘报告已存在, 跳过运行(--force可强制重跑)")
            return

    logger.info("启动模拟盘每日运行...")

    # S1. 加载虚拟账户
    portfolio = load_paper_portfolio()
    cash = portfolio.get("_available_cash", INITIAL_CAPITAL)
    holdings = {k: v for k, v in portfolio.items() if not k.startswith("_")}
    pending = portfolio.get("_pending", [])

    # S2. 采集数据（与实盘同源同构; 模拟盘只交易ETF不采集个股）
    etf_list = list(set(list(KEY_ETFS.keys()) + USER_WATCHLIST))
    logger.info("Step 1/4: 采集数据...")
    all_data = collect_all_data(etf_list, [])

    # S3. 执行层: T+1 开盘价成交昨日计划
    logger.info("Step 2/4: 执行昨日计划(T+1开盘价)...")
    etf_data = all_data.get("etfs", {})
    price_map = {}
    for p in pending:
        edata = etf_data.get(p["code"]) or {}
        kline = edata.get("kline", [])
        # 严格语义: 今日K线存在才成交, 绝不用stale数据
        # (数据源日期带横杠如 "2026-08-13", 归一化后与 TODAY 比较; 否则永远判定无开盘价)
        if kline and str(kline[-1].get("date", "")).replace("-", "") == TODAY and kline[-1].get("open", 0) > 0:
            price_map[p["code"]] = kline[-1]["open"]
        else:
            price_map[p["code"]] = None
    cash, holdings, executed_trades, remaining = execute_pending(cash, holdings, pending, price_map)
    if executed_trades:
        logger.info("  成交 " + " | ".join(
            f"{t['action']} {t['code']} {t['shares']}股@{t['price']}" for t in executed_trades))
    else:
        logger.info("  无成交（昨日无计划或均延期）")

    # 8/14修复: 执行原子化 — 成交立即落盘, 防止S4a-S6(评分/网络/择时)崩溃导致当日成交丢失顺延T+2
    if executed_trades:
        portfolio["_available_cash"] = cash
        for k in [k for k in portfolio if not k.startswith("_")]:
            del portfolio[k]
        for code, pos in holdings.items():
            portfolio[code] = pos
        # 止损卖出→写入冷却期(5个交易日内禁止重买, 配合quant_engine买入过滤)
        cooldowns = portfolio.get("_cooldowns") if isinstance(portfolio.get("_cooldowns"), dict) else {}
        for t in executed_trades:
            if t["action"] == "SELL" and "止损" in t["reason"]:
                cooldowns[t["code"]] = TODAY
                logger.info(f"  [冷却] {t['code']} 止损卖出, 5个交易日内禁止重买")
        portfolio["_cooldowns"] = cooldowns
        # 交易流水(可审计): 保留最近200条
        trades = portfolio.get("_trades") if isinstance(portfolio.get("_trades"), list) else []
        trades.extend(executed_trades)
        del trades[:-200]
        portfolio["_trades"] = trades
        save_paper_portfolio(portfolio)
        logger.info(f"  [原子化] 当日{len(executed_trades)}笔成交已立即落盘(portfolio_paper.json)")

    # S4a. 评分（照搬 daily_runner Step2 的 ETF 部分, 含溢价惩罚+行业轮动）
    logger.info("Step 3/4: 评分+择时+生成计划...")
    pre_regime = "CHOPPY"
    indices_data = all_data.get("indices", {})
    if "000300" in indices_data:
        hs300_data = indices_data["000300"].get("data", [])
        if len(hs300_data) >= 60:
            hs_closes = [d["close"] for d in hs300_data]
            hs_highs = [d.get("high", d["close"] * 1.005) for d in hs300_data]
            hs_lows = [d.get("low", d["close"] * 0.995) for d in hs300_data]
            pre_regime = classify_market_regime(hs_closes, hs_highs, hs_lows)["regime"]
            logger.info(f"  市场状态预判: {pre_regime}")

    industry_data = fetch_sw_industry_returns(days=60)
    industry_rotation = compute_industry_rotation_score(industry_data, n_days=20)
    flow_map = {}
    etf_flow = all_data.get("etf_flow") or {}
    for s in (etf_flow.get("inflow") or []):
        flow_map[s["code"]] = 1
    for s in (etf_flow.get("outflow") or []):
        flow_map[s["code"]] = -1
    batch_scores = score_all_etfs_cross_sectional(etf_data, regime=pre_regime, etf_flow_map=flow_map)

    scores = []
    for result in batch_scores:
        code = result["code"]
        edata = etf_data.get(code, {})
        kline = edata.get("kline", [])
        closes = [k["close"] for k in kline]
        rt = edata.get("realtime") or {}
        current_price = rt.get("price") if rt.get("price") else (closes[-1] if closes else 0)
        fund_nav = fetch_etf_fund_nav(code)
        premium_info = compute_etf_premium(code, current_price, fund_nav)
        premium_pct = premium_info.get("premium_pct")
        # 8/14修复(P1): NAV失败时溢价惩罚整体丢失→用历史溢价兜底
        if premium_pct is None and premium_info.get("is_qdii"):
            fb_history = compute_etf_premium_history(code, kline)
            if fb_history.get("has_history"):
                premium_pct = fb_history.get("current")
                premium_info["premium_pct"] = premium_pct
                premium_info["premium_source"] = "history_fallback"
                premium_info["warning"] = f"NAV不可用, 历史溢价兜底({premium_pct:.1f}%)"

        blended_tech = result["blended_technical"]
        if premium_pct is not None:
            premium_history = None
            if premium_info.get("is_qdii"):
                premium_history = compute_etf_premium_history(code, kline)
                premium_info["history"] = premium_history
                if premium_history.get("has_history"):
                    premium_pct = premium_history.get("current")  # 与历史序列对齐
            adjusted_score, multiplier, warning = _apply_premium_penalty(
                blended_tech, premium_pct, premium_history)
            result["score"] = adjusted_score
            result["premium_multiplier"] = multiplier
            result["premium_warning"] = warning
        else:
            result["premium_info_raw"] = {"warning": "QDII溢价数据缺失(含历史兜底均不可用)"}

        # 重新判定等级（因溢价可能改变分数）
        grade_thresholds = OPTIMIZED_PARAMS.get("grade_thresholds",
                                                {"A_强烈买入": 78, "B_买入": 65, "C_观察": 55, "D_谨慎": 42})
        s = result["score"]
        if s >= grade_thresholds.get("A_强烈买入", 78): result["grade"] = "A_强烈买入"
        elif s >= grade_thresholds.get("B_买入", 65): result["grade"] = "B_买入"
        elif s >= grade_thresholds.get("C_观察", 55): result["grade"] = "C_观察"
        elif s >= grade_thresholds.get("D_谨慎", 42): result["grade"] = "D_谨慎"
        else: result["grade"] = "E_回避"

        industry_bonus = get_etf_industry_momentum(code, industry_rotation)
        if industry_bonus != 0:
            result["score"] = max(0, min(100, result["score"] + int(industry_bonus)))
            result["industry_bonus"] = industry_bonus

        result["is_watchlist"] = code in USER_WATCHLIST
        result["is_holding"] = code in holdings
        result["premium_info"] = premium_info
        scores.append(result)

    # S4b. 大盘择时 + 市场情绪
    timing_engine = MarketTiming(
        indices_data,
        all_data.get("north_bound", []),
        all_data.get("total_volume", 0),
        all_data.get("breadth", {}),
        all_data.get("margin", {}),
        volume_history=all_data.get("volume_history", []),
        bond_yield=all_data.get("bond_yield")
    )
    timing_result = timing_engine.position_advice()
    nf_5d = timing_result.get("north_flow_5d", 0)
    sentiment = compute_market_sentiment(
        all_data.get("breadth", {}), all_data.get("total_volume", 0), nf_5d)
    timing_result["sentiment"] = sentiment

    # S4c. 模拟盘独立回撤熔断（用模拟盘自己的历史, 与实盘互不干扰）
    try:
        hist_reports = load_history(OUTPUT_DIR, days=90, prefix=REPORT_PREFIX)
        hist_curve = compute_equity_curve(hist_reports)
        hist_assets = [e["total_assets"] for e in hist_curve if e.get("total_assets", 0) > 0]
        if len(hist_assets) >= 5:
            dd_peak = max(hist_assets)
            dd_now = hist_assets[-1]
            dd_pct = (dd_now / dd_peak - 1) * 100
            if dd_pct <= -15:
                timing_result["base_position"] = min(timing_result["base_position"], 0.10)
                timing_result["circuit_breaker"] = {"triggered": True, "level": "L2_重仓回撤",
                                                    "drawdown_pct": round(dd_pct, 2), "cap_position": 0.10}
                logger.warning(f"  [熔断L2] 模拟盘回撤{dd_pct:.1f}%≤-15%，仓位强制降至10%")
            elif dd_pct <= -10:
                timing_result["base_position"] = min(timing_result["base_position"], 0.30)
                timing_result["circuit_breaker"] = {"triggered": True, "level": "L1_中度回撤",
                                                    "drawdown_pct": round(dd_pct, 2), "cap_position": 0.30}
                logger.warning(f"  [熔断L1] 模拟盘回撤{dd_pct:.1f}%≤-10%，仓位强制降至30%")
            else:
                timing_result["circuit_breaker"] = {"triggered": False,
                                                    "drawdown_pct": round(dd_pct, 2)}
    except Exception as e:
        logger.info(f"  [熔断] 回撤计算失败(忽略): {e}")

    # S4d. 盯市 + 组合概览 + 生成明日计划
    portfolio["_available_cash"] = cash
    for k in list(portfolio.keys()):
        if not k.startswith("_"):
            del portfolio[k]
    for code, pos in holdings.items():
        portfolio[code] = pos
    portfolio = update_portfolio_prices(portfolio, etf_data)
    # 8/14长持改造: 每日评分写入持仓历史(供quant_engine"连续5日<55卖出"判断), 保留最近7天
    score_by_code = {s["code"]: s["score"] for s in scores}
    for code, pos in list(portfolio.items()):
        if code.startswith("_") or not isinstance(pos, dict) or "shares" not in pos:
            continue
        if code in score_by_code:
            hist = pos.setdefault("score_history", [])
            hist.append(round(score_by_code[code], 1))
            del hist[:-7]
    port_summary = compute_portfolio_summary(portfolio, scores)
    decider = TradeDecider(scores, timing_result, portfolio)
    plan = decider.generate_plan()

    # S5. 基准对比（模拟盘无出入金, 口径简单: 总资产/初始资金）
    benchmark = compute_benchmark_comparison(portfolio, indices_data, report_prefix=REPORT_PREFIX)
    if port_summary.get("total_assets", 0) > 0 and benchmark.get("portfolio_start_value", 0) > 0:
        portfolio_return = port_summary["total_assets"] / benchmark["portfolio_start_value"] - 1.0
        benchmark["portfolio_return"] = round(portfolio_return, 4)
        benchmark["net_cash_outflow"] = 0.0
        benchmark["excess_return"] = round(portfolio_return - benchmark.get("benchmark_return", 0), 4)
        benchmark["beat_benchmark"] = benchmark["excess_return"] > 0
        if benchmark["beat_benchmark"]:
            benchmark["message"] = (
                f"模拟组合 {portfolio_return:+.2%} vs 沪深300 {benchmark['benchmark_return']:+.2%}，"
                f"超额 {benchmark['excess_return']:+.2%} ✅ 跑赢基准"
            )
        else:
            benchmark["message"] = (
                f"模拟组合 {portfolio_return:+.2%} vs 沪深300 {benchmark['benchmark_return']:+.2%}，"
                f"落后 {benchmark['excess_return']:+.2%} ❌ 跑输基准"
            )
    logger.info(f"  [基准对比] {benchmark.get('message', '数据不足')}")

    # S6. 落盘: 报告 + 虚拟账户
    account = {
        "initial_capital": INITIAL_CAPITAL,
        "total_assets": round(port_summary.get("total_assets", 0), 2),
        "total_return_pct": round((port_summary.get("total_assets", 0) / INITIAL_CAPITAL - 1) * 100, 2),
    }
    # 防堆积: 数据滞后时旧订单未执行会不断累积 — 同代码同方向的旧订单被今日新信号覆盖
    # (执行过的订单已从remaining移除; 不同方向如SELL+BUY同日保留, 先卖后买)
    new_pending = build_pending_from_plan(plan)
    pending_keys = {(p["code"], p["action"]) for p in new_pending}
    # 8/14修复: 停牌订单的retries不因新信号覆盖而归零(原覆盖后重试上限失效, 订单无限滞留)
    retry_carry = {}
    for p in remaining:
        if (p["code"], p["action"]) in pending_keys and p.get("retries", 0) > 0:
            retry_carry[(p["code"], p["action"])] = p["retries"]
    final_pending = [p for p in remaining if (p["code"], p["action"]) not in pending_keys]
    for p in new_pending:
        key = (p["code"], p["action"])
        if key in retry_carry:
            p["retries"] = retry_carry[key]
        final_pending.append(p)

    report = format_paper_report(executed_trades, port_summary, timing_result, plan,
                                 benchmark, final_pending, account,
                                 trades_history=portfolio.get("_trades"))
    try:
        print(report)
    except UnicodeEncodeError:
        logger.info("[输出] 终端编码不支持全文打印(报告仍会保存)")

    output = {
        "date": TODAY,
        "timestamp": datetime.now().isoformat(),
        "type": "paper",
        "timing": timing_result,
        "scores": scores,
        "plan": plan,
        "executed_trades": executed_trades,
        "pending": final_pending,
        "portfolio": port_summary,
        "benchmark": benchmark,
        "account": account,
        "report": report,
    }
    output_path = os.path.join(OUTPUT_DIR, f"{REPORT_PREFIX}{TODAY}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"模拟盘报告已保存: {output_path}")

    # 保存虚拟账户: 清理盯市临时字段, 只保留状态
    portfolio["_pending"] = final_pending
    portfolio["_available_cash"] = cash
    for code in list(portfolio.keys()):
        if not code.startswith("_"):
            portfolio[code].pop("current_price", None)
            portfolio[code].pop("prev_close", None)
    save_paper_portfolio(portfolio)
    logger.info(f"虚拟账户已保存: {PAPER_PORTFOLIO_FILE}")

    # S7. 邮件（发送失败不阻塞落盘与git提交）
    email_pw = os.environ.get("QQMAIL_AUTH_CODE")
    if not email_pw:
        config_path = os.path.join(OUTPUT_DIR, "email_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    email_pw = json.load(f).get("QQMAIL_AUTH_CODE")
            except Exception:
                pass
    if email_pw:
        logger.info("生成HTML报告并发送邮件...")
        ok = False
        try:
            html = generate_html_report(output)
            ok = send_email(html, email_pw, report_date=TODAY,
                            subject=f"【模拟盘日报】 - {datetime.now().strftime('%Y.%m.%d')}")
        except Exception as e:
            logger.error(f"邮件发送失败: {e}\n{traceback.format_exc()}")
        if not ok:
            # P0修复: 邮件失败原为静默, 现在告警+exit非0让workflow失败可被GitHub通知
            send_failure_alert(f"模拟盘日报邮件发送失败({TODAY})", traceback.format_exc())
            sys.exit(1)
    else:
        logger.info("未配置QQ邮箱授权码，跳过发送")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"模拟盘运行失败: {e}\n{traceback.format_exc()}")
        try:
            send_failure_alert(f"模拟盘运行失败: {e}", traceback.format_exc())
        except Exception:
            logger.error(f"崩溃通知也失败了: {traceback.format_exc()}")
        sys.exit(1)
