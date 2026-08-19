#!/usr/bin/env python
"""
量化系统 每日自动运行主程序
============================
整合: 数据采集 → 因子计算 → 择时判断 → 决策生成 → 输出报告
用法: python daily_runner.py [--portfolio portfolio.json]
"""
import json, sys, os, io, shutil, urllib.request, logging, traceback
from datetime import datetime

# Fix Windows encoding (only when running as script)
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# === 日志系统 ===
# 同时输出到控制台（INFO级别）和文件（DEBUG级别，保留7天）
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"daily_{datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("quant")

# 清理7天前的旧日志
import glob as _glob
for old_log in _glob.glob(os.path.join(LOG_DIR, "daily_*.log")):
    try:
        log_date = os.path.basename(old_log).replace("daily_", "").replace(".log", "")
        from datetime import datetime as _dt
        if (_dt.now() - _dt.strptime(log_date, "%Y%m%d")).days > 7:
            os.remove(old_log)
    except (ValueError, KeyError, OSError):
        pass  # 日志文件名格式异常或删除失败，跳过

# 确保能找到模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_engine import (
    collect_all_data, KEY_ETFS, USER_WATCHLIST, USER_STOCKS,
    fetch_market_breadth, fetch_total_volume,
    fetch_north_bound_flow, fetch_margin_balance, fetch_etf_kline,
    fetch_etf_realtime, get_all_index_data,
    fetch_etf_fund_nav, compute_etf_premium, compute_etf_premium_history,
    compute_market_sentiment, fetch_sw_industry_returns
)
from quant_engine import (
    score_etf_comprehensive, score_all_etfs_cross_sectional,
    MarketTiming, TradeDecider,
    compute_atr_stop_loss, MARKET_REGIME,
    _apply_premium_penalty,
    compute_industry_rotation_score, get_etf_industry_momentum,
    classify_market_regime,
    compute_factor_ic_ranking, format_factor_ic_section,
    OPTIMIZED_PARAMS
)
from report_mailer import generate_html_report, send_email, MonthlyReview
from risk_engine import (portfolio_risk_report, format_risk_section,
                         stress_test_portfolio, format_stress_test_section,
                         monte_carlo_simulation, format_monte_carlo_section)
from performance_tracker import generate_performance_summary

TODAY = datetime.now().strftime("%Y%m%d")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
def load_portfolio(filepath=None):
    """加载当前持仓，包含可用资金。
    优先使用指定文件，其次 portfolio.json，最后报错。
    修改持仓只需编辑 portfolio.json，无需改代码。"""
    if filepath and os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    # 默认从 portfolio.json 加载
    default_path = os.path.join(OUTPUT_DIR, "portfolio.json")
    if os.path.exists(default_path):
        with open(default_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    raise FileNotFoundError(
        f"找不到持仓文件。请创建 {default_path} 或使用 --portfolio 指定路径。"
        f"\n格式: {{\"_available_cash\": 金额, \"代码\": {{\"shares\": 股数, \"cost\": 成本, \"name\": \"名称\"}}}}"
    )

def update_portfolio_prices(portfolio, etf_data):
    """更新持仓的当前价格和昨日收盘价"""
    for code in portfolio:
        if code.startswith("_"): continue  # 跳过元数据
        if code in etf_data:
            rt = etf_data[code].get("realtime") or {}
            kline = etf_data[code].get("kline", [])
            portfolio[code]["current_price"] = rt.get("price") if rt.get("price") else (kline[-1]["close"] if kline else portfolio[code].get("cost", 0))
            # 昨收价：优先实时数据，其次K线倒数第二根
            prev_close = rt.get("prev_close")
            if not prev_close and len(kline) >= 2:
                prev_close = kline[-2].get("close")
            portfolio[code]["prev_close"] = prev_close
            # 8/17修复: 标记K线滞后(工作日且最后bar非今日) — 报告显示⚠️防把昨日行情当今日盈亏
            if kline:
                last_bar = str(kline[-1].get("date", "")).replace("-", "")
                portfolio[code]["stale_date"] = "" if last_bar == TODAY else last_bar
    return portfolio


def compute_portfolio_summary(portfolio, scores):
    """计算账户概览：市值、盈亏、仓位等"""
    holdings = []
    total_value = 0
    total_cost = 0
    total_daily_pnl = 0

    score_map = {s["code"]: s for s in scores}

    for code, pos in portfolio.items():
        if code.startswith("_"): continue
        shares = pos["shares"]
        cost = pos.get("cost", 0)
        price = pos.get("current_price", cost)
        prev_close = pos.get("prev_close")

        value = shares * price
        cost_value = shares * cost
        pnl = value - cost_value
        pnl_pct = (price / cost - 1) * 100 if cost > 0 else 0

        # 今日盈亏
        daily_pnl = 0
        daily_pnl_pct = 0
        if prev_close and prev_close > 0:
            daily_pnl = (price - prev_close) * shares
            daily_pnl_pct = (price / prev_close - 1) * 100

        total_value += value
        total_cost += cost_value
        total_daily_pnl += daily_pnl

        holdings.append({
            "code": code,
            "name": pos.get("name", code),
            "shares": shares,
            "cost": cost,
            "price": price,
            "prev_close": prev_close,
            "value": round(value, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "daily_pnl": round(daily_pnl, 2),
            "daily_pnl_pct": round(daily_pnl_pct, 2),
            "score": score_map[code]["score"] if code in score_map else None,
            "grade": score_map[code]["grade"] if code in score_map else None,
        })

    available_cash = portfolio.get("_available_cash", 0)
    total_assets = total_value + available_cash
    total_pnl = total_value - total_cost

    # 为每个持仓计算仓位占比
    for h in holdings:
        h["weight"] = round(h["value"] / total_assets * 100, 1) if total_assets > 0 else 0

    # 8/17: 数据滞后持仓(现价非当日) — 报告显示⚠️防误导
    stale_codes = [code for code, pos in portfolio.items()
                   if not code.startswith("_") and pos.get("stale_date")]

    return {
        "holdings": holdings,
        "available_cash": available_cash,
        "total_value": round(total_value, 2),
        "total_cost": round(total_cost, 2),
        "total_assets": round(total_assets, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl / total_cost * 100, 2) if total_cost > 0 else 0,
        "total_daily_pnl": round(total_daily_pnl, 2),
        "cash_ratio": round(available_cash / total_assets * 100, 1) if total_assets > 0 else 0,
        "stale_codes": stale_codes,
    }

# ============================================================
# v7.3: 行业暴露分析
# ============================================================
# 导入 ETF→行业映射
from quant_engine import ETF_INDUSTRY_MAP as _INDUSTRY_MAP

def compute_industry_exposure(portfolio, port_summary):
    """
    将持仓映射到申万一级行业，聚合行业权重。

    返回: {
        "industries": [{name, weight_pct, bar, etfs}],  # 按权重大→小排序
        "cash_pct": float,
        "max_single_industry": {name, weight_pct} | None,
        "diversification_note": str
    }
    """
    industry_weights = {}  # {行业名: {"weight":累计%, "etfs": [etf名称]}}
    total_assets = port_summary.get("total_assets", 1)

    for h in port_summary.get("holdings", []):
        code = h["code"]
        weight = h.get("weight", 0)
        name = h.get("name", code)
        industries = _INDUSTRY_MAP.get(code, ["综合"])

        weight_per_ind = weight / len(industries) if industries else weight
        for ind in industries:
            if ind not in industry_weights:
                industry_weights[ind] = {"weight": 0, "etfs": []}
            industry_weights[ind]["weight"] += weight_per_ind
            industry_weights[ind]["etfs"].append(name)

    # 排序
    sorted_inds = sorted(industry_weights.items(), key=lambda x: x[1]["weight"], reverse=True)

    result = []
    for ind_name, data in sorted_inds:
        w = data["weight"]
        bar_len = int(w / 5)  # 每5%一个方块
        bar = "█" * bar_len
        result.append({
            "name": ind_name,
            "weight_pct": round(w, 1),
            "bar": bar,
            "etfs": list(set(data["etfs"]))  # 去重
        })

    cash_pct = port_summary.get("cash_ratio", 0)
    max_ind = result[0] if result else None

    # 集中度评估
    if max_ind and max_ind["weight_pct"] > 40:
        note = f"⚠️ {max_ind['name']}行业占比{max_ind['weight_pct']:.0f}%，过于集中"
    elif max_ind and max_ind["weight_pct"] > 25:
        note = f"⚡ {max_ind['name']}行业占比{max_ind['weight_pct']:.0f}%，注意集中度"
    elif len(result) >= 3:
        note = "✅ 行业分布较分散"
    else:
        note = "行业覆盖偏少，可通过增持不同行业ETF改善"

    return {
        "industries": result,
        "cash_pct": round(cash_pct, 1),
        "max_single_industry": max_ind,
        "diversification_note": note
    }


def format_industry_exposure_section(exposure):
    """格式化行业暴露为文本板块"""
    lines = []
    lines.append(f"\n  {'─'*60}")
    lines.append(f"  [行业暴露热力图] v7.3")
    lines.append(f"  {'─'*60}")
    lines.append(f"  {'行业':<12s} {'权重':>6s}  {'分布'}")
    lines.append(f"  {'─'*12} {'─'*6}  {'─'*40}")

    for ind in exposure["industries"]:
        etf_list = ", ".join(ind["etfs"][:3])
        if len(ind["etfs"]) > 3:
            etf_list += f" +{len(ind['etfs'])-3}"
        lines.append(f"  {ind['name']:<12s} {ind['weight_pct']:>5.1f}%  {ind['bar']} ({etf_list})")

    # 现金占比
    cash_pct = exposure["cash_pct"]
    cash_bar = "░" * int(cash_pct / 5)
    lines.append(f"  {'现金':<12s} {cash_pct:>5.1f}%  {cash_bar}")

    lines.append(f"\n  {exposure['diversification_note']}")
    return "\n".join(lines)


def analyze_watchlist_etf(s, timing, portfolio, kline_data=None):
    """对单只关注ETF生成买入/观望/回避建议

    v7.2: kline_data 可选，传入后使用ATR动态止损替代固定-5%/-8%
           kline_data = {"closes": [...], "highs": [...], "lows": [...]}
    """
    code = s["code"]
    name = s["name"]
    score = s["score"]
    grade = s["grade"]
    ind = s["indicators"]
    ret = s["returns"]
    vs = s["vs_ma"]
    factors = s.get("factors", {})

    # 已持仓的单独处理
    is_holding = code in portfolio
    holding_info = portfolio.get(code, {})

    reasons_buy = []
    reasons_avoid = []
    action = "HOLD"

    # 溢价风险检查（v2.1 新增 — 优先于所有技术信号）
    premium_info = s.get("premium_info", {})
    if premium_info.get("premium_pct") is not None and premium_info["premium_pct"] > 5:
        reasons_avoid.append(f"🚨 ETF溢价{premium_info['premium_pct']:.1f}%！溢价回归将直接亏损{premium_info['premium_pct']:.1f}%")
    elif premium_info.get("premium_pct") is not None and premium_info["premium_pct"] > 3:
        reasons_avoid.append(f"⚠️ ETF溢价{premium_info['premium_pct']:.1f}%，偏高需等待回落")
    # 历史溢价风险（v7.5，8/5新增 — 结合历史各时期溢价）
    hist = premium_info.get("history") or {}
    if hist.get("has_history") and hist.get("percentile") is not None:
        if hist["percentile"] >= 90:
            reasons_avoid.append(f"🚨 溢价处自身历史{hist['percentile']:.0f}%分位（中枢{hist['median']}%），回归风险大")
        elif hist["percentile"] >= 75 and premium_info.get("premium_pct", 0) > 0:
            reasons_avoid.append(f"⚠️ 溢价处历史{hist['percentile']:.0f}%分位（中枢{hist['median']}%），偏高需等回落")
        elif hist["percentile"] < 25 and premium_info.get("premium_pct", 0) < 0:
            reasons_buy.append(f"💚 折价{abs(premium_info['premium_pct']):.1f}%且处历史{hist['percentile']:.0f}%低位，安全垫充足")

    # 买入条件
    if ind["rsi"] <= 58 and ind["consecutive_up"] <= 2:
        reasons_buy.append(f"RSI={ind['rsi']:.0f}不热，连涨仅{ind['consecutive_up']}天")
    if ret["r5d"] <= 1 and ret["r20d"] >= -5:
        reasons_buy.append("短期未大涨，中期不差")
    if factors.get("F1_趋势强度", 0) >= 7:
        reasons_buy.append("趋势动能充沛")
    if ind.get("sortino", 0) > 0.8:
        reasons_buy.append(f"风险调整收益好(Sortino={ind['sortino']:.2f})")
    if 30 <= ind["rsi"] <= 45:
        reasons_buy.append("RSI偏冷，接近超卖区，反弹概率大")

    # 回避条件
    if ind["rsi"] > 68:
        reasons_avoid.append(f"RSI={ind['rsi']:.0f}过热，追高风险大")
    if ind["consecutive_up"] >= 5:
        reasons_avoid.append(f"连涨{ind['consecutive_up']}天，获利盘压力极大")
    if ret["r5d"] > 12:
        reasons_avoid.append(f"5日涨{ret['r5d']:.1f}%，短期透支")
    if factors.get("F1_趋势强度", 0) <= 3:
        reasons_avoid.append("趋势信号偏弱")
    if ind.get("sortino", 0) < -1.0:
        reasons_avoid.append("风险收益比为负，持有风险大")
    if ind["rsi"] < 25 and factors.get("F1_趋势强度", 0) <= 3:
        reasons_avoid.append("超跌但趋势未反转，不建议接飞刀")

    # 综合判断
    if grade in ["A_强烈买入", "B_买入"] and len(reasons_avoid) <= 1:
        action = "BUY"
    elif grade in ["D_谨慎", "E_回避"] or len(reasons_avoid) >= 3:
        action = "AVOID"
    else:
        action = "WATCH"

    # 持仓特殊建议
    holding_advice = ""
    if is_holding:
        cost = holding_info.get("cost", s["price"])
        pnl = (s["price"]/cost - 1)*100
        if pnl <= -8:
            holding_advice = f"止损触发！浮亏{pnl:.1f}%，建议减仓"
            action = "SELL"
        elif action == "AVOID" and pnl > 3:
            holding_advice = f"浮盈{pnl:.1f}%，但评分下滑，建议止盈"
            action = "REDUCE"
        elif action == "AVOID":
            holding_advice = f"浮亏{pnl:.1f}%，评分走弱，关注止损线"
        elif action == "BUY":
            holding_advice = f"浮盈{pnl:.1f}%，可以加仓"
        else:
            holding_advice = f"浮盈{pnl:.1f}%，继续持有观察"

    buy_price = s["price"]
    # v7.2: ATR动态止损 — 用kline数据计算（有数据时），否则回退固定值
    if kline_data and kline_data.get("closes") and len(kline_data["closes"]) >= 15:
        atr_stop = compute_atr_stop_loss(
            kline_data["closes"], kline_data.get("highs", kline_data["closes"]),
            kline_data.get("lows", kline_data["closes"]), buy_price,
            atr_period=14, atr_mult=2.5
        )
        stop_loss = atr_stop["stop_price"]
        stop_pct = atr_stop["stop_pct"]
        stop_method = atr_stop["method"]
        # 止盈与止损对称：ATR倍数一致但方向相反
        take_profit = round(buy_price * (1 + abs(stop_pct) / 100 * 1.5), 3)
    else:
        stop_loss = round(buy_price * 0.95, 3)
        stop_pct = -5.0
        stop_method = "固定止损(无K线)"
        take_profit = round(buy_price * 1.08, 3)

    return {
        "code": code, "name": name, "action": action,
        "score": score, "grade": grade, "price": buy_price,
        "reasons_buy": reasons_buy, "reasons_avoid": reasons_avoid,
        "is_holding": is_holding, "holding_advice": holding_advice,
        "stop_loss": stop_loss, "take_profit": take_profit,
        "stop_pct": stop_pct, "stop_method": stop_method,
        "rsi": ind["rsi"], "consecutive_up": ind["consecutive_up"],
        "r5d": ret["r5d"], "r20d": ret["r20d"]
    }

def format_institutional_flow(all_data):
    """格式化机构资金流向"""
    lines = []
    lines.append(f"\n  {'─'*60}")
    lines.append(f"  [机构资金流向]")
    lines.append(f"  {'─'*60}")

    # 行业主力资金 TOP5
    sf = all_data.get("sector_flow")
    if sf:
        lines.append(f"  行业主力净流入 TOP5:")
        for i, s in enumerate(sf[:5]):
            emoji = "[+]" if s["main_net"] > 0 else "[-]"
            lines.append(f"    {i+1}. {emoji} {s['name']} : {s['main_net']:+.1f}亿")

    # 龙虎榜机构净买
    dt = all_data.get("dragon_tiger")
    if dt:
        lines.append(f"  龙虎榜机构净买入:")
        for s in dt[:5]:
            lines.append(f"    {s['name']}({s['code']}) : {s['inst_net']:+.0f}万")

    # ETF份额申赎 (v7.6: 结构改为 {inflow, outflow})
    ef = all_data.get("etf_flow")
    if ef:
        for s in (ef.get("inflow") or [])[:5]:
            lines.append(f"  🟢 机构净申购: {s['name']}({s['code']}) 份额+{s['share_change']:+.0f}")
        for s in (ef.get("outflow") or [])[:5]:
            lines.append(f"  🔴 机构净赎回: {s['name']}({s['code']}) 份额{s['share_change']:+.0f}")

    # 北向行业偏好已删除(原接口MK0354实为可转债板块, 假数据P0, 2026/8/14审查移除)

    return "\n".join(lines)

def format_report(plan, scores, timing, portfolio, all_data=None, stock_scores=None, port_summary=None):
    """格式化输出报告 — 增强版：持仓实时+逐只分析+操作建议"""
    lines = []
    lines.append("=" * 70)
    lines.append(f"  [A股量化决策系统] 每日报告")
    lines.append(f"  [日期] {datetime.now().strftime('%Y年%m月%d日 %H:%M')}")
    lines.append("=" * 70)

    # ===== 大盘择时 =====
    lines.append(f"\n  {'─'*60}")
    lines.append(f"  [一] 大盘择时 + 市场状态")
    lines.append(f"  {'─'*60}")
    # 市场状态（v3.0 新增）
    regime = timing.get("regime", "CHOPPY")
    regime_emoji = {"TREND_UP": "🟢", "CHOPPY": "🟡", "TREND_DOWN": "🟠", "CRISIS": "🔴"}
    lines.append(f"  市场状态: {regime_emoji.get(regime, '⚪')} {timing.get('regime_name', regime)} (置信度{timing.get('regime_confidence', 0):.0%})")
    regime_signals = timing.get('regime_signals', [])
    if regime_signals:
        lines.append(f"    判定依据: {' | '.join(regime_signals)}")
    lines.append(f"    策略: {timing.get('regime_description', '')}")
    lines.append(f"  传统信号: {timing['bull_signals']}/{timing['total_signals']}看多 -> 建议仓位 {timing['base_position']*100:.0f}%")
    lines.append(f"    状态止损: {timing.get('regime_stop_loss', -0.08)*100:.0f}% | 最低买入等级: {timing.get('regime_buy_grade_min', 'B_买入')}")
    for name, value in timing['signal_detail'].items():
        icon = "[PASS]" if value else "[FAIL]"
        label = name.replace("S1_HS300_above_MA20","沪深300在20日线上").replace("S2_HS300_MA60_up","沪深300的60日线向上").replace("S3_NorthFlow_5d_positive","北向资金5日净流入").replace("S4_Volume_active","成交额>2万亿").replace("S5_LimitDown_low","跌停<20家").replace("S6_Margin_increasing","融资余额增加")
        lines.append(f"    {icon} {label}")
    if timing['force_capped']:
        lines.append(f"  [WARNING] 强制限制生效: 仓位上限30%")
    nf_5d = timing.get("north_flow_5d")
    if nf_5d is not None:
        lines.append(f"  沪深股通5日均成交: {nf_5d:.1f}亿元(北向净买额2024/8起停披露,以成交活跃度替代)")
    else:
        lines.append("  沪深股通成交数据缺失(中性处理)")
    # v5.0: 市场情绪
    sentiment = timing.get("sentiment", {})
    if sentiment:
        emoji = {"贪婪":"\U0001f525","偏乐观":"\U0001f60a","中性":"\U0001f610","偏恐慌":"\U0001f628","恐慌":"\U0001f480"}
        est_note = "(含估算)" if sentiment.get("estimated") else ""  # 8/14: 广度数据估算时标注, 防假数据
        lines.append(f"  \U0001f4ca 市场情绪: {emoji.get(sentiment.get('level',''),'')} {sentiment.get('level','?')} ({sentiment.get('score',50)}分){est_note}")

    # ===== 账户概览 =====
    ps = port_summary or {}
    lines.append(f"\n  {'─'*60}")
    lines.append(f"  [二] 账户概览")
    lines.append(f"  {'─'*60}")
    lines.append(f"  总资产: {ps.get('total_assets', 0):.2f}元 | 总市值: {ps.get('total_value', 0):.2f}元 | 可用资金: {ps.get('available_cash', 0):.2f}元")
    lines.append(f"  持仓盈亏: {ps.get('total_pnl', 0):+.2f}元 ({ps.get('total_pnl_pct', 0):+.2f}%) | 今日盈亏: {ps.get('total_daily_pnl', 0):+.2f}元")
    lines.append(f"  现金占比: {ps.get('cash_ratio', 0):.1f}%")
    # 8/17: 数据滞后告警 — 现价非当日时今日盈亏不可信
    if ps.get("stale_codes"):
        lines.append(f"  ⚠️ 数据滞后: {', '.join(ps['stale_codes'])} 现价非当日K线, 今日盈亏≠当日实际(数据源更新失败)")

    # ===== 持仓明细 =====
    lines.append(f"\n  {'─'*60}")
    lines.append(f"  [三] 持仓明细")
    lines.append(f"  {'─'*60}")
    lines.append(f"  {'代码':<8} {'名称':<16} {'持股':>5} {'成本':>8} {'现价':>8} {'市值':>10} {'持仓盈亏':>12} {'今日盈亏':>12} {'仓位':>6} {'评分'}")
    lines.append(f"  {'─'*8} {'─'*16} {'─'*5} {'─'*8} {'─'*8} {'─'*10} {'─'*12} {'─'*12} {'─'*6} {'─'*4}")

    holdings = ps.get("holdings", [])
    for h in holdings:
        lines.append(
            f"  {h['code']:<8} {h['name']:<16} {h['shares']:>5} "
            f"{h['cost']:>8.3f} {h['price']:>8.3f} {h['value']:>10.2f} "
            f"{h['pnl']:>+10.2f}({h['pnl_pct']:>+.1f}%) "
            f"{h['daily_pnl']:>+10.2f}({h['daily_pnl_pct']:>+.1f}%) "
            f"{h['weight']:>5.1f}% {h.get('grade','?'):>4s}"
        )

    # ===== 持仓操作建议 =====
    lines.append(f"\n  {'─'*60}")
    lines.append(f"  [四] 持仓操作建议")
    lines.append(f"  {'─'*60}")

    for code, pos in portfolio.items():
        if code.startswith("_"): continue
        score_data = next((s for s in scores if s["code"] == code), None)
        if not score_data:
            lines.append(f"  {pos['name']}({code}): 数据缺失，无法评分")
            continue

        # v7.2: 提取K线数据用于ATR动态止损
        kd = None
        if all_data:
            etf_info = all_data.get("etfs", {}).get(code, {})
            kline = etf_info.get("kline", [])
            if len(kline) >= 15:
                kd = {"closes": [k["close"] for k in kline],
                      "highs": [k["high"] for k in kline],
                      "lows": [k["low"] for k in kline]}

        analysis = analyze_watchlist_etf(score_data, timing, portfolio, kline_data=kd)
        action_label = {"BUY":"[加仓]","SELL":"[卖出]","REDUCE":"[减仓]","WATCH":"[持有观察]","HOLD":"[持有]","AVOID":"[回避]"}.get(analysis["action"], "[?]")

        lines.append(f"  {action_label} {pos['name']}({code}) | 评分{analysis['score']}分 | 现价{analysis['price']:.3f}")
        # 溢价信息（v2.1 新增）
        p_info = score_data.get("premium_info", {})
        tech_score = score_data.get("technical_score", analysis["score"])
        if p_info.get("premium_pct") is not None and p_info["premium_pct"] > 2:
            lines.append(f"    ⚡ 溢价{p_info['premium_pct']:.1f}% → 技术分{tech_score}扣至{analysis['score']}分")
        elif p_info.get("premium_pct") is not None and p_info["premium_pct"] < -1:
            lines.append(f"    💚 折价{abs(p_info['premium_pct']):.1f}%，低于净值买入")
        # 历史溢价（v7.5，8/5新增）
        hist_p = p_info.get("history") or {}
        if hist_p.get("has_history"):
            lines.append(f"    历史溢价: 中枢{hist_p.get('median')}% | 分位{hist_p.get('percentile')}% | 近10日均{hist_p.get('trend_10d')}%")
        stop_pct_h = analysis.get("stop_pct", -5)
        stop_method_h = analysis.get("stop_method", "固定")
        lines.append(f"    止损线:{analysis['stop_loss']:.3f}({stop_pct_h:+.1f}% {stop_method_h}) | 止盈线:{analysis['take_profit']:.3f}")
        if analysis.get("holding_advice"):
            lines.append(f"    {analysis['holding_advice']}")
        for r in analysis.get("reasons_avoid", []):
            lines.append(f"    [注意] {r}")

    # ===== 投资组合风险（v3.0 新增）=====
    etf_data_map = all_data.get("etfs", {}) if all_data else {}
    if port_summary and etf_data_map:
        risk_report = portfolio_risk_report(portfolio, etf_data_map, scores)
        lines.append(format_risk_section(risk_report))

    # ===== 行业暴露热力图（v7.3 新增）=====
    if port_summary and port_summary.get("holdings"):
        exposure = compute_industry_exposure(portfolio, port_summary)
        lines.append(format_industry_exposure_section(exposure))

        # v7.3: 压力测试（依赖行业暴露数据）
        stress = stress_test_portfolio(portfolio, port_summary)
        lines.append(format_stress_test_section(stress))

        # v8.0: 蒙特卡洛模拟
        mc = monte_carlo_simulation(etf_data_map, portfolio, n_simulations=1000, horizon_days=20)
        lines.append(format_monte_carlo_section(mc))

    # ===== 因子IC跟踪（v7.3 新增）=====
    if etf_data_map:
        ic_results = compute_factor_ic_ranking(etf_data_map)
        lines.append(format_factor_ic_section(ic_results))

    # ===== 关注ETF逐只分析 =====
    lines.append(f"\n  {'─'*60}")
    lines.append(f"  [五] 关注ETF逐只分析")
    lines.append(f"  {'─'*60}")

    watchlist = [s for s in scores if s.get("is_watchlist") and s["code"] not in portfolio]
    for s in watchlist:
        # v7.2: 提取K线数据用于ATR动态止损
        kd_w = None
        if all_data:
            etf_info = all_data.get("etfs", {}).get(s["code"], {})
            kline_w = etf_info.get("kline", [])
            if len(kline_w) >= 15:
                kd_w = {"closes": [k["close"] for k in kline_w],
                        "highs": [k["high"] for k in kline_w],
                        "lows": [k["low"] for k in kline_w]}

        analysis = analyze_watchlist_etf(s, timing, portfolio, kline_data=kd_w)
        action_label = {"BUY":"[可买入]","SELL":"[卖出]","REDUCE":"[减仓]","WATCH":"[观望]","HOLD":"[持有]","AVOID":"[回避]"}.get(analysis["action"], "[?]")

        lines.append(f"  {action_label} {s['name']}({s['code']}) | 评分{s['score']}分 | 现价{s['price']:.4f}")
        lines.append(f"    RSI:{analysis['rsi']:.0f} | 连涨:{analysis['consecutive_up']}天 | 5日:{analysis['r5d']:+.1f}% | 20日:{analysis['r20d']:+.1f}%")
        # 历史溢价（v7.5，8/5新增 — 跨境ETF结合历史各时期溢价）
        p_info_w = s.get("premium_info", {})
        hist_w = p_info_w.get("history") or {}
        if hist_w.get("has_history"):
            lines.append(f"    溢价{hist_w.get('current')}% | 历史中枢{hist_w.get('median')}% | 分位{hist_w.get('percentile')}% | 近10日均{hist_w.get('trend_10d')}%")
        for r in analysis.get("reasons_buy", []):
            lines.append(f"    [+] {r}")
        for r in analysis.get("reasons_avoid", []):
            lines.append(f"    [-] {r}")
        if analysis["action"] == "BUY":
            stop_pct = analysis.get("stop_pct", -5)
            stop_method = analysis.get("stop_method", "固定")
            lines.append(f"    建议买入价:{analysis['price']:.4f} | 止损:{analysis['stop_loss']:.4f}({stop_pct:+.1f}% {stop_method}) | 止盈:{analysis['take_profit']:.4f}")

    # ===== 个股评分 =====
    if stock_scores:
        lines.append(f"\n  {'─'*60}")
        lines.append(f"  [六] 个股评分")
        lines.append(f"  {'─'*60}")
        for s in stock_scores:
            # v7.2: 提取K线用于ATR止损（个股）
            kd_s = None
            if all_data:
                st_info = all_data.get("stocks", {}).get(s["code"], {})
                kline_s = st_info.get("kline", [])
                if len(kline_s) >= 15:
                    kd_s = {"closes": [k["close"] for k in kline_s],
                            "highs": [k["high"] for k in kline_s],
                            "lows": [k["low"] for k in kline_s]}

            analysis = analyze_watchlist_etf(s, timing, portfolio, kline_data=kd_s)
            action_label = {"BUY":"[可买入]","WATCH":"[观望]","AVOID":"[回避]"}.get(analysis["action"], "[?]")
            lines.append(f"  {action_label} {s['name']}({s['code']}) | {s['score']}分 | 现价{s['price']:.2f}")
            lines.append(f"    RSI:{s['indicators']['rsi']:.0f} | 20日:{s['returns']['r20d']:+.1f}% | 60日:{s['returns']['r60d']:+.1f}%")
            for r in analysis.get("reasons_buy", []):
                lines.append(f"    [+] {r}")
            for r in analysis.get("reasons_avoid", []):
                lines.append(f"    [-] {r}")

    # ===== 全市场TOP10 =====
    other_scores = [s for s in scores if not s.get("is_watchlist")]
    lines.append(f"\n  {'─'*60}")
    lines.append(f"  [七] 全市场ETF排名 TOP10")
    lines.append(f"  {'─'*60}")
    grade_icon = {"A_强烈买入":"[A]","B_买入":"[B]","C_观察":"[C]","D_谨慎":"[D]","E_回避":"[E]"}
    for i, s in enumerate(other_scores[:10]):
        gi = grade_icon.get(s['grade'], "[?]")
        lines.append(f"  {i+1:2d}. {gi} {s['name']}({s['code']}) | {s['score']}分 | RSI:{s['indicators']['rsi']:.0f} | 5日:{s['returns']['r5d']:+.1f}% | 20日:{s['returns']['r20d']:+.1f}%")

    # ===== 绩效追踪（v3.0 新增）=====
    perf_summary = generate_performance_summary()
    lines.append(perf_summary)

    lines.append(f"\n  {'='*70}")
    lines.append(f"  [免责声明] 量化模型仅供辅助决策，不构成投资建议。投资有风险，入市需谨慎。")
    return "\n".join(lines)

def is_rest_day():
    """判断今天是否为休息日（周末或中国法定节假日）
    使用 timor.tech 免费节假日API
    返回 (is_rest: bool, reason: str)
    """
    # 第一道：周末快速判断（--force-weekend 可绕过）
    weekday = datetime.now().weekday()
    if weekday >= 5 and "--force-weekend" not in sys.argv:
        return True, f"周末 (weekday={weekday})"

    # --force-weekend: 强制按工作日运行
    if "--force-weekend" in sys.argv:
        return False, "强制运行(忽略周末)"

    # 第二道：查询法定节假日API
    today_str = datetime.now().strftime("%Y-%m-%d")
    try:
        url = f"https://timor.tech/api/holiday/info/{today_str}"
        req = urllib.request.Request(url, headers={"User-Agent": "QuantSystem/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        if data.get("code") == 0:
            t = data.get("type", {})
            type_code = t.get("type", 0)
            # type: 0=工作日, 1=周末, 2=节假日, 3=调休工作日
            if type_code == 2:
                name = t.get("name", "节假日")
                return True, f"法定假日 ({name})"
            elif type_code == 1:
                return True, f"周末"
            # type==0 or type==3: 可以交易
            return False, "工作日"
    except Exception as e:
        logger.warning(f"节假日API查询失败: {e}，按工作日处理")
        return False, "API不可用，假定为工作日"

    return False, "工作日"


# ================================================================
# v8.0: 基准对比 — 组合 vs 沪深300全收益
# ================================================================
def compute_benchmark_comparison(portfolio, index_data, report_prefix="report_"):
    """
    P0-2: 计算组合与沪深300全收益的对比。

    思路:
    1. 找到最早报告日期作为对比起点
    2. 计算从起点到现在的组合累计收益
    3. 计算同期沪深300全收益
    4. 返回对比结果（超额收益、相对强弱）
    """
    start_date = None
    portfolio_start_value = 0

    from glob import glob as _glob
    report_dir = os.path.dirname(os.path.abspath(__file__))
    report_files = sorted(_glob(os.path.join(report_dir, f"{report_prefix}[0-9]*.json")))
    if report_files:
        # 找第一个包含有效总资产的报告作为对比起点（旧版报告无portfolio字段，跳过）
        for rf in report_files:
            try:
                with open(rf, 'r', encoding='utf-8') as f:
                    old_report = json.load(f)
                old_port = old_report.get("portfolio", {})
                if isinstance(old_port, dict):
                    val = old_port.get("total_assets", 0)
                    if val and val > 0:
                        fname = os.path.basename(rf)
                        start_date = fname.replace(report_prefix, "").replace(".json", "")
                        portfolio_start_value = val
                        break
            except (json.JSONDecodeError, KeyError, OSError):
                continue
        if start_date is None:
            # 全部报告无有效持仓数据：用最早报告的日期，起点值置0（后续不渲染对比结论）
            fname = os.path.basename(report_files[0])
            start_date = fname.replace(report_prefix, "").replace(".json", "")
    else:
        # v7.6: 无历史报告(GitHub Actions环境, report被gitignore)时,
        # 起点用累计投入本金(Σdeposit), 而非当前资产 — 否则收益恒为0且出入金无法修正
        start_date = datetime.now().strftime("%Y%m%d")
        total_deposit = sum(cf.get("amount", 0) for cf in (portfolio.get("_cash_flows", []) or [])
                            if cf.get("type") == "deposit")
        if total_deposit > 0:
            portfolio_start_value = total_deposit
        else:
            portfolio_start_value = sum(
                h.get("shares", 0) * h.get("cost", 0)
                for k, h in portfolio.items()
                if isinstance(h, dict) and "shares" in h and k != "_available_cash"
            ) + portfolio.get("_available_cash", 0)

    # 获取沪深300指数数据
    benchmark_start = None
    benchmark_end = None
    benchmark_return = 0.0

    if "000300" in index_data:
        hs300_info = index_data["000300"]
        hs300_data = hs300_info.get("data", [])
        if len(hs300_data) >= 2:
            # P0修复: K线日期带横杠(2026-05-19) vs start_date无横杠(20260812)字符串比较恒False
            # → 基准起点恒为窗口第一根K线且每天滚动, 统一为8位数字再比较
            start_date_norm = str(start_date).replace("-", "")
            start_idx = 0
            for i, d in enumerate(hs300_data):
                if str(d.get("date", "")).replace("-", "") >= start_date_norm:
                    start_idx = i
                    break
            benchmark_start = hs300_data[start_idx]["close"]
            benchmark_end = hs300_data[-1]["close"]
            if benchmark_start > 0:
                benchmark_return = (benchmark_end / benchmark_start - 1.0)

    return {
        "start_date": start_date,
        "portfolio_start_value": portfolio_start_value,
        "benchmark_start": benchmark_start,
        "benchmark_end": benchmark_end,
        "benchmark_return": benchmark_return,
        "benchmark_name": "沪深300 (000300)",
        "has_history": bool(report_files)  # v7.6: 是否有历史报告(影响出入金修正口径)
    }


# ================================================================
# v8.0: 发送失败告警
# ================================================================
def send_failure_alert(error_msg, traceback_str):
    """日报发送失败时发送简单告警邮件"""
    try:
        email_pw = os.environ.get("QQMAIL_AUTH_CODE")
        if not email_pw:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "email_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    email_pw = json.load(f).get("QQMAIL_AUTH_CODE")
        if email_pw:
            from report_mailer import send_crash_report
            send_crash_report(error_msg, traceback_str, email_pw)
            logger.info("  [告警] 已发送失败通知邮件")
    except Exception as e:
        logger.error(f"  [告警] 发送失败通知也失败了: {e}")


def main():
    # 休息日跳过
    is_rest, reason = is_rest_day()
    if is_rest:
        logger.info(f"{reason}，休市跳过")
        return

    # 8/14: 邮件幂等 — 当日实盘报告已生成则跳过(防cron双触发/手动重跑导致重复邮件)
    today_iso = datetime.now().strftime("%Y%m%d")
    if "--force" not in sys.argv and "--rerun" not in sys.argv:
        existing = [f for f in _glob.glob(os.path.join(OUTPUT_DIR, f"report_{today_iso}.json"))]
        if existing:
            logger.info(f"今日实盘报告已存在({len(existing)}个), 跳过运行(--force可强制重跑)")
            return

    logger.info("启动每日量化分析...")

    # 参数解析
    portfolio_file = None
    if "--portfolio" in sys.argv:
        idx = sys.argv.index("--portfolio")
        if idx+1 < len(sys.argv):
            portfolio_file = sys.argv[idx+1]

    # 加载持仓
    portfolio = load_portfolio(portfolio_file)

    # 确定分析标的
    etf_list = list(set(list(KEY_ETFS.keys()) + USER_WATCHLIST))
    stock_list = list(USER_STOCKS.keys())

    # 1. 采集数据
    logger.info("Step 1/4: 采集数据...")
    try:
        all_data = collect_all_data(etf_list, stock_list)
    except Exception as e:
        logger.error(f"数据采集失败: {e}\n{traceback.format_exc()}")
        raise RuntimeError(f"数据采集失败，量化分析无法继续: {e}") from e

    # 1b. 预判市场状态（v7.2: 用于自适应因子权重）
    pre_regime = "CHOPPY"
    indices_data = all_data.get("indices", {})
    if "000300" in indices_data:
        hs300_data = indices_data["000300"].get("data", [])
        if len(hs300_data) >= 60:
            hs_closes = [d["close"] for d in hs300_data]
            hs_highs = [d.get("high", d["close"] * 1.005) for d in hs300_data]
            hs_lows = [d.get("low", d["close"] * 0.995) for d in hs300_data]
            pre_regime = classify_market_regime(hs_closes, hs_highs, hs_lows)["regime"]
            logger.info(f"市场状态预判: {pre_regime} → 自适应权重生效")

    # 2. 计算因子得分 - ETFs (v5.0: 横截面比较 + v7.2: regime自适应)
    logger.info("Step 2/4: 计算因子(含截面比较)...")
    try:
        etf_data = all_data.get("etfs", {})

        # v5.0: 行业轮动
        logger.info("Step 2/4a: 行业轮动...")
        industry_data = fetch_sw_industry_returns(days=60)
        industry_rotation = compute_industry_rotation_score(industry_data, n_days=20)

        # v5.0: 批量评分 + 横截面比较 + v7.2: regime自适应权重
        # v7.6: ETF份额申赎信号 → F18因子
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

            # 获取溢价率，应用溢价惩罚
            kline = edata.get("kline", [])
            closes = [k["close"] for k in kline]
            rt = edata.get("realtime") or {}
            current_price = rt.get("price") if rt.get("price") else (closes[-1] if closes else 0)
            fund_nav = fetch_etf_fund_nav(code)
            premium_info = compute_etf_premium(code, current_price, fund_nav)
            premium_pct = premium_info.get("premium_pct")
            # 8/14修复(P1): NAV失败时溢价惩罚整体丢失→用历史溢价兜底, 10%溢价的QDII不能再零扣分买入
            if premium_pct is None and premium_info.get("is_qdii"):
                fb_history = compute_etf_premium_history(code, kline)
                if fb_history.get("has_history"):
                    premium_pct = fb_history.get("current")
                    premium_info["premium_pct"] = premium_pct
                    premium_info["premium_source"] = "history_fallback"
                    premium_info["warning"] = f"NAV不可用, 历史溢价兜底({premium_pct:.1f}%)"

            # 溢价惩罚：在截面调整后的技术分上再应用
            blended_tech = result["blended_technical"]
            if premium_pct is not None:
                # v7.5 (8/5): 跨境/QDII 结合历史各时期溢价评分 — 历史分位+趋势
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
            grade_thresholds = OPTIMIZED_PARAMS.get("grade_thresholds", {"A_强烈买入": 78, "B_买入": 65, "C_观察": 55, "D_谨慎": 42})
            s = result["score"]
            if s >= grade_thresholds.get("A_强烈买入", 78): result["grade"] = "A_强烈买入"
            elif s >= grade_thresholds.get("B_买入", 65): result["grade"] = "B_买入"
            elif s >= grade_thresholds.get("C_观察", 55): result["grade"] = "C_观察"
            elif s >= grade_thresholds.get("D_谨慎", 42): result["grade"] = "D_谨慎"
            else: result["grade"] = "E_回避"

            # v5.0: 行业轮动加成
            industry_bonus = get_etf_industry_momentum(code, industry_rotation)
            if industry_bonus != 0:
                result["score"] = max(0, min(100, result["score"] + int(industry_bonus)))
                result["industry_bonus"] = industry_bonus

            result["is_watchlist"] = code in USER_WATCHLIST
            result["is_holding"] = code in portfolio
            result["premium_info"] = premium_info
            scores.append(result)

    # 2b. 计算因子得分 - 个股
    except Exception as e:
        logger.error(f"评分计算失败: {e}\n{traceback.format_exc()}")
        raise RuntimeError(f"评分计算失败: {e}") from e

    stock_data = all_data.get("stocks", {})
    stock_scores = []
    for code, sdata in stock_data.items():
        kline = sdata.get("kline", [])
        if len(kline) < 30: continue
        closes = [k["close"] for k in kline]
        highs = [k["high"] for k in kline]
        lows = [k["low"] for k in kline]
        volumes = [k["volume"] for k in kline]
        turnovers = [k.get("turnover", 0) for k in kline]  # v7.6: 个股换手率F17
        result = score_etf_comprehensive(code, sdata["name"], closes, highs, lows, volumes, turnover=turnovers)
        result["is_stock"] = True
        stock_scores.append(result)

    # 3. 大盘择时
    logger.info("Step 3/4: 大盘择时...")
    try:
        timing_engine = MarketTiming(
            all_data.get("indices", {}),
            all_data.get("north_bound", []),
            all_data.get("total_volume", 0),
            all_data.get("breadth", {}),
            all_data.get("margin", {}),
            volume_history=all_data.get("volume_history", []),  # v7.6: S4动态分位
            bond_yield=all_data.get("bond_yield")  # v7.6: S7流动性信号
        )
        timing_result = timing_engine.position_advice()
    except Exception as e:
        logger.error(f"大盘择时失败: {e}\n{traceback.format_exc()}")
        raise RuntimeError(f"大盘择时失败: {e}") from e

    # v5.0: 市场情绪
    nf_5d = timing_result.get("north_flow_5d", 0)
    sentiment = compute_market_sentiment(
        all_data.get("breadth", {}),
        all_data.get("total_volume", 0),
        nf_5d
    )
    timing_result["sentiment"] = sentiment

    # v7.6: 组合级回撤熔断 — 从历史报告权益曲线计算近期峰值回撤, 超限强制降仓
    try:
        from performance_tracker import load_history, compute_equity_curve
        hist_reports = load_history(OUTPUT_DIR, days=90)
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
                logger.warning(f"  [熔断L2] 组合回撤{dd_pct:.1f}%≤-15%，仓位强制降至10%")
            elif dd_pct <= -10:
                timing_result["base_position"] = min(timing_result["base_position"], 0.30)
                timing_result["circuit_breaker"] = {"triggered": True, "level": "L1_中度回撤",
                                                    "drawdown_pct": round(dd_pct, 2), "cap_position": 0.30}
                logger.warning(f"  [熔断L1] 组合回撤{dd_pct:.1f}%≤-10%，仓位强制降至30%")
            else:
                timing_result["circuit_breaker"] = {"triggered": False,
                                                    "drawdown_pct": round(dd_pct, 2)}
    except Exception as e:
        logger.info(f"  [熔断] 回撤计算失败(忽略): {e}")

    # 4. 生成决策
    logger.info("Step 4/4: 生成决策...")
    portfolio = load_portfolio(portfolio_file)
    portfolio = update_portfolio_prices(portfolio, etf_data)

    # 计算持仓概览（市值、盈亏、仓位等）
    port_summary = compute_portfolio_summary(portfolio, scores)

    decider = TradeDecider(scores, timing_result, portfolio)
    plan = decider.generate_plan()

    # v8.0: 基准对比
    benchmark = compute_benchmark_comparison(portfolio, all_data.get("indices", {}))
    if port_summary.get("total_assets", 0) > 0 and benchmark.get("portfolio_start_value", 0) > 0:
        # v7.6: 修正 — 期间提现/入金不算盈亏, 加回净出金后再算收益率
        # (原口径把提现2000算成亏损, 曾误报组合-56.2%)
        cash_flows = portfolio.get("_cash_flows", []) or []
        # v7.6: 有历史报告→只调整起点后的出入金; 无历史(GitHub环境)→全部提现加回
        # (无历史时起点=累计投入本金, 所有提现都发生在起点之后)
        if benchmark.get("has_history"):
            bm_start = benchmark.get("start_date", "")
            withdraw_after = sum(cf.get("amount", 0) for cf in cash_flows
                                 if cf.get("type") == "withdraw" and str(cf.get("date", "")) >= bm_start)
            deposit_after = sum(cf.get("amount", 0) for cf in cash_flows
                                if cf.get("type") == "deposit" and str(cf.get("date", "")) > bm_start)
            net_outflow = withdraw_after - deposit_after
        else:
            net_outflow = sum(cf.get("amount", 0) for cf in cash_flows
                              if cf.get("type") == "withdraw")
        adjusted_assets = port_summary["total_assets"] + net_outflow
        portfolio_return = adjusted_assets / benchmark["portfolio_start_value"] - 1.0
        benchmark["portfolio_return"] = round(portfolio_return, 4)
        benchmark["net_cash_outflow"] = round(net_outflow, 2)
        benchmark["excess_return"] = round(portfolio_return - benchmark.get("benchmark_return", 0), 4)
        benchmark["beat_benchmark"] = benchmark["excess_return"] > 0
        # 生成对比消息
        adj_note = f" (净出金{net_outflow:+.0f}元已剔除)" if abs(net_outflow) > 1 else ""
        if benchmark["beat_benchmark"]:
            benchmark["message"] = (
                f"组合 {portfolio_return:+.2%}{adj_note} vs 沪深300 {benchmark['benchmark_return']:+.2%}，"
                f"超额 {benchmark['excess_return']:+.2%} ✅ 跑赢基准"
            )
        else:
            benchmark["message"] = (
                f"组合 {portfolio_return:+.2%}{adj_note} vs 沪深300 {benchmark['benchmark_return']:+.2%}，"
                f"落后 {benchmark['excess_return']:+.2%} ❌ 跑输基准"
            )
    logger.info(f"  [基准对比] {benchmark.get('message', '数据不足')}")

    # 输出报告
    # 机构资金流向
    inst_flow_section = format_institutional_flow(all_data)

    report = format_report(plan, scores, timing_result, portfolio, all_data, stock_scores, port_summary)
    report += inst_flow_section
    # v7.6: Windows GBK终端打印emoji会UnicodeEncodeError导致报告不保存 — 加固
    try:
        print(report)
    except UnicodeEncodeError:
        logger.info("[输出] 终端编码不支持全文打印(报告仍会保存)")

    # 保存结果
    output = {
        "date": TODAY,
        "timestamp": datetime.now().isoformat(),
        "timing": timing_result,
        "scores": scores,
        "plan": plan,
        "portfolio": port_summary,
        "report": report,
        "benchmark": benchmark  # v8.0: 基准对比
    }
    output_path = os.path.join(OUTPUT_DIR, f"report_{TODAY}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"报告已保存: {output_path}")

    # ---- 回测 + 仪表盘（可选）----
    if "--backtest" in sys.argv:
        logger.info("=== 回测模式 ===")
        try:
            from backtest_engine import prepare_backtest_data, run_backtest
            from dashboard import generate_from_backtest_result

            # 运行回测
            bt_data = prepare_backtest_data("2024-01-01", TODAY)
            bt_result = run_backtest(bt_data)

            if bt_result:
                # 生成仪表盘
                dash_path = generate_from_backtest_result(bt_result)
                logger.info(f"仪表盘: {dash_path}")

                # 复制到部署目录
                deploy_dir = os.path.join(OUTPUT_DIR, "deploy")
                os.makedirs(deploy_dir, exist_ok=True)
                deploy_path = os.path.join(deploy_dir, "index.html")
                shutil.copy(dash_path, deploy_path)
                logger.info(f"部署副本: {deploy_path}")
        except Exception as e:
            logger.error(f"回测失败: {e}\n{traceback.format_exc()}")

    # ---- 邮件发送 ----
    email_pw = os.environ.get("QQMAIL_AUTH_CODE")
    if not email_pw:
        config_path = os.path.join(OUTPUT_DIR, "email_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                email_pw = config.get("QQMAIL_AUTH_CODE")
            except:
                pass

    if email_pw:
        logger.info("生成HTML报告并发送邮件...")
        ok = False
        try:
            html = generate_html_report(output)
            # 8/19加固: stale数据时邮件标题显著标记, 避免误读
            subject = None
            if output.get("stale_codes"):
                subject = f"[数据滞后] A股量化日报 - {output.get('date')} ({', '.join(output['stale_codes'])})"
            ok = send_email(html, email_pw, report_date=output.get("date"), subject=subject)
        except Exception as e:
            logger.error(f"邮件发送失败: {e}\n{traceback.format_exc()}")
        if not ok:
            # P0修复: 邮件失败原为静默(workflow仍success), 现在发告警邮件+exit非0让workflow失败
            send_failure_alert("日报邮件发送失败(系统可能无有效信号)", traceback.format_exc())
            sys.exit(1)
    else:
        logger.info("未配置QQ邮箱授权码，跳过发送")

    # 每月第一个周六自动复盘（与optimizer月度优化对齐）
    if datetime.now().weekday() == 5 and datetime.now().day <= 7:  # 每月第一个周六
        logger.info("月度自动复盘...")
        reviewer = MonthlyReview()
        review_report = reviewer.monthly_summary()
        review_path = os.path.join(OUTPUT_DIR, f"review_{TODAY}.txt")
        with open(review_path, 'w', encoding='utf-8') as f:
            f.write(review_report)
        logger.info(f"月度复盘报告已保存: {review_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"量化系统运行失败: {e}\n{traceback.format_exc()}")
        # 即使崩溃也尝试通知
        try:
            email_pw = os.environ.get("QQMAIL_AUTH_CODE")
            if not email_pw:
                config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "email_config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        email_pw = json.load(f).get("QQMAIL_AUTH_CODE")
            if email_pw:
                from report_mailer import send_crash_report
                send_crash_report(str(e), traceback.format_exc(), email_pw)
        except:
            logger.error(f"崩溃通知也失败了: {traceback.format_exc()}")
        sys.exit(1)
