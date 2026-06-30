#!/usr/bin/env python
"""
量化系统 每日自动运行主程序
============================
整合: 数据采集 → 因子计算 → 择时判断 → 决策生成 → 输出报告
用法: python daily_runner.py [--portfolio portfolio.json]
"""
import json, sys, os, io, urllib.request, logging, traceback
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
    except:
        pass

# 确保能找到模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_engine import (
    collect_all_data, KEY_ETFS, USER_WATCHLIST, USER_STOCKS,
    fetch_market_breadth, fetch_total_volume,
    fetch_north_bound_flow, fetch_margin_balance, fetch_etf_kline,
    fetch_etf_realtime, get_all_index_data,
    fetch_etf_fund_nav, compute_etf_premium,
    compute_market_sentiment, fetch_sw_industry_returns
)
from quant_engine import (
    score_etf_comprehensive, score_all_etfs_cross_sectional,
    MarketTiming, TradeDecider,
    compute_atr_stop_loss, MARKET_REGIME,
    _apply_premium_penalty,
    compute_industry_rotation_score, get_etf_industry_momentum,
    OPTIMIZED_PARAMS
)
from report_mailer import generate_html_report, send_email, WeeklyReview
from risk_engine import portfolio_risk_report, format_risk_section
from performance_tracker import generate_performance_summary

TODAY = datetime.now().strftime("%Y%m%d")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

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
            portfolio[code]["current_price"] = rt.get("price") if rt.get("price") else portfolio[code].get("cost", 0)
            # 昨收价：优先实时数据，其次K线倒数第二根
            prev_close = rt.get("prev_close")
            if not prev_close and len(kline) >= 2:
                prev_close = kline[-2].get("close")
            portfolio[code]["prev_close"] = prev_close
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
    }

def analyze_watchlist_etf(s, timing, portfolio):
    """对单只关注ETF生成买入/观望/回避建议"""
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
    stop_loss = round(buy_price * 0.95, 3)
    take_profit = round(buy_price * 1.08, 3)

    return {
        "code": code, "name": name, "action": action,
        "score": score, "grade": grade, "price": buy_price,
        "reasons_buy": reasons_buy, "reasons_avoid": reasons_avoid,
        "is_holding": is_holding, "holding_advice": holding_advice,
        "stop_loss": stop_loss, "take_profit": take_profit,
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

    # ETF份额增长
    ef = all_data.get("etf_flow")
    if ef:
        lines.append(f"  机构增持ETF:")
        for s in ef[:5]:
            lines.append(f"    {s['name']}({s['code']}) : 份额变化{s['share_change']:+.0f}")

    # 北向行业偏好
    nb = all_data.get("north_top")
    if nb:
        lines.append(f"  北向资金偏好行业:")
        for s in nb[:5]:
            lines.append(f"    {s['name']}")

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
    lines.append(f"  北向资金5日累计: {timing.get('north_flow_5d', 'N/A'):.1f}亿元")
    # v5.0: 市场情绪
    sentiment = timing.get("sentiment", {})
    if sentiment:
        emoji = {"贪婪":"\U0001f525","偏乐观":"\U0001f60a","中性":"\U0001f610","偏恐慌":"\U0001f628","恐慌":"\U0001f480"}
        lines.append(f"  \U0001f4ca 市场情绪: {emoji.get(sentiment.get('level',''),'')} {sentiment.get('level','?')} ({sentiment.get('score',50)}分)")

    # ===== 账户概览 =====
    ps = port_summary or {}
    lines.append(f"\n  {'─'*60}")
    lines.append(f"  [二] 账户概览")
    lines.append(f"  {'─'*60}")
    lines.append(f"  总资产: {ps.get('total_assets', 0):.2f}元 | 总市值: {ps.get('total_value', 0):.2f}元 | 可用资金: {ps.get('available_cash', 0):.2f}元")
    lines.append(f"  持仓盈亏: {ps.get('total_pnl', 0):+.2f}元 ({ps.get('total_pnl_pct', 0):+.2f}%) | 今日盈亏: {ps.get('total_daily_pnl', 0):+.2f}元")
    lines.append(f"  现金占比: {ps.get('cash_ratio', 0):.1f}%")

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

        analysis = analyze_watchlist_etf(score_data, timing, portfolio)
        action_label = {"BUY":"[加仓]","SELL":"[卖出]","REDUCE":"[减仓]","WATCH":"[持有观察]","HOLD":"[持有]","AVOID":"[回避]"}.get(analysis["action"], "[?]")

        lines.append(f"  {action_label} {pos['name']}({code}) | 评分{analysis['score']}分 | 现价{analysis['price']:.3f}")
        # 溢价信息（v2.1 新增）
        p_info = score_data.get("premium_info", {})
        tech_score = score_data.get("technical_score", analysis["score"])
        if p_info.get("premium_pct") is not None and p_info["premium_pct"] > 2:
            lines.append(f"    ⚡ 溢价{p_info['premium_pct']:.1f}% → 技术分{tech_score}扣至{analysis['score']}分")
        elif p_info.get("premium_pct") is not None and p_info["premium_pct"] < -1:
            lines.append(f"    💚 折价{abs(p_info['premium_pct']):.1f}%，低于净值买入")
        lines.append(f"    止损线:{analysis['stop_loss']:.3f}(-5%) | 止盈线:{analysis['take_profit']:.3f}(+8%)")
        if analysis.get("holding_advice"):
            lines.append(f"    {analysis['holding_advice']}")
        for r in analysis.get("reasons_avoid", []):
            lines.append(f"    [注意] {r}")

    # ===== 投资组合风险（v3.0 新增）=====
    etf_data_map = all_data.get("etfs", {}) if all_data else {}
    if port_summary and etf_data_map:
        risk_report = portfolio_risk_report(portfolio, etf_data_map, scores)
        lines.append(format_risk_section(risk_report))

    # ===== 关注ETF逐只分析 =====
    lines.append(f"\n  {'─'*60}")
    lines.append(f"  [五] 关注ETF逐只分析")
    lines.append(f"  {'─'*60}")

    watchlist = [s for s in scores if s.get("is_watchlist") and s["code"] not in portfolio]
    for s in watchlist:
        analysis = analyze_watchlist_etf(s, timing, portfolio)
        action_label = {"BUY":"[可买入]","SELL":"[卖出]","REDUCE":"[减仓]","WATCH":"[观望]","HOLD":"[持有]","AVOID":"[回避]"}.get(analysis["action"], "[?]")

        lines.append(f"  {action_label} {s['name']}({s['code']}) | 评分{s['score']}分 | 现价{s['price']:.4f}")
        lines.append(f"    RSI:{analysis['rsi']:.0f} | 连涨:{analysis['consecutive_up']}天 | 5日:{analysis['r5d']:+.1f}% | 20日:{analysis['r20d']:+.1f}%")
        for r in analysis.get("reasons_buy", []):
            lines.append(f"    [+] {r}")
        for r in analysis.get("reasons_avoid", []):
            lines.append(f"    [-] {r}")
        if analysis["action"] == "BUY":
            lines.append(f"    建议买入价:{analysis['price']:.4f} | 止损:{analysis['stop_loss']:.4f}(-5%) | 止盈:{analysis['take_profit']:.4f}(+8%)")

    # ===== 个股评分 =====
    if stock_scores:
        lines.append(f"\n  {'─'*60}")
        lines.append(f"  [六] 个股评分")
        lines.append(f"  {'─'*60}")
        for s in stock_scores:
            analysis = analyze_watchlist_etf(s, timing, portfolio)
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
    # 第一道：周末快速判断
    weekday = datetime.now().weekday()
    if weekday >= 5:
        return True, f"周末 (weekday={weekday})"

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


def main():
    # 休息日跳过
    is_rest, reason = is_rest_day()
    if is_rest:
        logger.info(f"{reason}，休市跳过")
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

    # 2. 计算因子得分 - ETFs (v5.0: 横截面比较)
    logger.info("Step 2/4: 计算因子(含截面比较)...")
    try:
        etf_data = all_data.get("etfs", {})

        # v5.0: 行业轮动
        logger.info("Step 2/4a: 行业轮动...")
        industry_data = fetch_sw_industry_returns(days=60)
        industry_rotation = compute_industry_rotation_score(industry_data, n_days=20)

        # v5.0: 批量评分 + 横截面比较
        batch_scores = score_all_etfs_cross_sectional(etf_data)

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

            # 溢价惩罚：在截面调整后的技术分上再应用
            blended_tech = result["blended_technical"]
            if premium_pct is not None:
                adjusted_score, multiplier, warning = _apply_premium_penalty(blended_tech, premium_pct)
                result["score"] = adjusted_score
                result["premium_multiplier"] = multiplier
                result["premium_warning"] = warning
            else:
                result["premium_info_raw"] = {"warning": "QDII溢价数据缺失"}

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
        result = score_etf_comprehensive(code, sdata["name"], closes, highs, lows, volumes)
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
            all_data.get("margin", {})
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

    # 4. 生成决策
    logger.info("Step 4/4: 生成决策...")
    portfolio = load_portfolio(portfolio_file)
    portfolio = update_portfolio_prices(portfolio, etf_data)

    # 计算持仓概览（市值、盈亏、仓位等）
    port_summary = compute_portfolio_summary(portfolio, scores)

    decider = TradeDecider(scores, timing_result, portfolio)
    plan = decider.generate_plan()

    # 输出报告
    # 机构资金流向
    inst_flow_section = format_institutional_flow(all_data)

    report = format_report(plan, scores, timing_result, portfolio, all_data, stock_scores, port_summary)
    report += inst_flow_section
    print(report)

    # 保存结果
    output = {
        "date": TODAY,
        "timestamp": datetime.now().isoformat(),
        "timing": timing_result,
        "scores": scores,
        "plan": plan,
        "portfolio": port_summary,
        "report": report
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
                import shutil
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
        try:
            html = generate_html_report(output)
            send_email(html, email_pw)
        except Exception as e:
            logger.error(f"邮件发送失败: {e}\n{traceback.format_exc()}")
    else:
        logger.info("未配置QQ邮箱授权码，跳过发送")

    # 每月第一个周六自动复盘（与optimizer月度优化对齐）
    if datetime.now().weekday() == 5 and datetime.now().day <= 7:  # 每月第一个周六
        logger.info("月度自动复盘...")
        reviewer = WeeklyReview()
        review_report = reviewer.weekly_summary()
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
